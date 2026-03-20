# Copy from https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py
#
# coding=utf-8
# Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional, Tuple, Union
from dataclasses import dataclass
from models.t5_segmem import T5Config, T5SegMem
from transformers.models.t5.modeling_t5 import Seq2SeqLMOutput, BaseModelOutput, BaseModelOutputWithPastAndCrossAttentions, checkpoint, T5LayerNorm, T5Block
from transformers.utils import logging
import torch.nn as nn
import torch.nn.functional as F
import copy
import torch
from einops import rearrange
from tqdm import tqdm

from models.issm import InstrumentSpecificSlotMemory


logger = logging.get_logger(__name__)


@dataclass
class Seq2SeqLMOutputNumInsts(Seq2SeqLMOutput):
    loss_inst: Optional[torch.FloatTensor] = None


class T5SegMemV2(T5SegMem):
    """
    V2 appends segmem on encoder_outputs and influence via cross attention
    instead of V1 which directly prepends segmem on decoder_inputs_embeds
    """
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(
        self, 
        config: T5Config,
        segmem_num_layers: int = 1,
        segmem_length: int = 64,
    ):
        super().__init__(
            config=config,
            segmem_num_layers=segmem_num_layers,
            segmem_length=segmem_length,
        )
        d_model = getattr(config, "d_model", None) or getattr(self.config, "d_model")
        self.issm = InstrumentSpecificSlotMemory(d_model=d_model, num_slots=17)
        self.memory_state = None
    
    def get_model_outputs(
        self,
        inputs: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        reset_memory: bool = False,
    ):
        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                decoder_head_mask = head_mask
        if inputs is not None:
            inputs_embeds = self.proj(inputs)
        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=None,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(
                    encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(
                    encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]
        assert hidden_states.dim() == 3, f"Expected 3D encoder outputs (B, T, D), got {hidden_states.dim()}D"

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)
        
        assert decoder_inputs_embeds is None
        decoder_inputs_embeds = self.decoder_embed_tokens(decoder_input_ids)                # (b, l, d)

        # NOTE: Legacy T5SegMemV2 explicitly concatenated the b-1 label states.
        # This was stripped in favor of the specialized Instrument-Specific Slot Memory (ISSM).

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]  

        # --- ISSM SECONDARY ATTENTION INJECTION ---
        try:
            if reset_memory or self.memory_state is None or self.memory_state.shape[0] != hidden_states.shape[0]:
                self.memory_state = self.issm.init_memory(hidden_states.shape[0], hidden_states.device)
            
            # Explicit detached & cloned context for safety guarantees
            detached_memory = self.memory_state.detach().clone()
            updated_memory = self.issm.extract_and_update(hidden_states, detached_memory)
            
            issm_output = self.issm.memory_cross_attention(sequence_output, updated_memory)
            
            # NaN/Inf safeguard
            if torch.isnan(issm_output).any() or torch.isinf(issm_output).any():
                logger.warning("NaN/Inf detected in ISSM attention. Bypassing ISSM gate for this batch.")
            else:
                self.memory_state = updated_memory
                sequence_output = issm_output
        except Exception as e:
            logger.error(f"ISSM integration failed: {str(e)}. Bypassing memory gate.")

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)
        
        lm_logits = self.lm_head(sequence_output)
        return lm_logits, encoder_outputs, decoder_outputs

    def generate(self, inputs, max_length=1024, output_hidden_states=False, **kwargs):
        batch_size = inputs.shape[0]
        inputs_embeds = self.proj(inputs)
        encoder_outputs = self.encoder(
            inputs_embeds=inputs_embeds,
            return_dict=True
        )
        hidden_states = encoder_outputs[0]
        
        # ISSM Update for generation
        try:
            memory_state = self.issm.init_memory(hidden_states.shape[0], hidden_states.device)
            memory_state = self.issm.extract_and_update(hidden_states, memory_state)
        except Exception as e:
            logger.error(f"ISSM initialization failed during generation: {str(e)}.")
            memory_state = None

        # Fully Batched Decode
        bs = hidden_states.size(0)
        decoder_tokens = torch.zeros((bs, 1), dtype=torch.long, device=hidden_states.device)
        finished = torch.zeros(bs, dtype=torch.bool, device=hidden_states.device)

        for l in range(max_length):  
            decoder_outputs = self.decoder(
                input_ids=decoder_tokens,
                encoder_hidden_states=hidden_states,
                return_dict=True,
            )
            sequence_output = decoder_outputs[0]
            
            # --- ISSM SECONDARY ATTENTION INJECTION ---
            if memory_state is not None:
                try:
                    issm_output = self.issm.memory_cross_attention(sequence_output, memory_state)
                    if not (torch.isnan(issm_output).any() or torch.isinf(issm_output).any()):
                        sequence_output = issm_output
                except Exception as e:
                    pass

            lm_logits = self.lm_head(sequence_output)[:, -1, :]
            cur = torch.argmax(lm_logits, dim=-1)

            # Mask out finished sequences with pad_token_id
            cur = torch.where(finished, torch.tensor(self.config.pad_token_id, device=hidden_states.device), cur)

            decoder_tokens = torch.cat([decoder_tokens, cur.unsqueeze(1)], dim=1)
            
            finished |= (cur == self.config.eos_token_id)
            if finished.all():
                break

        # Uniform batched padding if early break
        if decoder_tokens.shape[1] < max_length:
            decoder_tokens = F.pad(
                decoder_tokens,
                (0, max_length - decoder_tokens.shape[1]),
                value=self.config.pad_token_id
            )
        
        return decoder_tokens