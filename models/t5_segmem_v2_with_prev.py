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
from models.t5_segmem_v2 import T5Config, T5SegMemV2
from transformers.models.t5.modeling_t5 import Seq2SeqLMOutput, BaseModelOutput, BaseModelOutputWithPastAndCrossAttentions, checkpoint, T5LayerNorm, T5Block
from transformers.utils import logging
import torch.nn as nn
import torch.nn.functional as F
import copy
import torch
from einops import rearrange
from tqdm import tqdm

from models.issm import InstrumentSpecificSlotMemory

from models.contrastive_timbre_embedding import ContrastiveTimbreEmbedding
import math

try:
    from models.layers import SaliencyHead
except ImportError:
    # Fallback if layers.py doesn't contain SaliencyHead yet
    import torch.nn as nn
    class SaliencyHead(nn.Module):
        def __init__(self, d_model: int, num_instruments: int):
            super().__init__()
            self.proj = nn.Linear(d_model, num_instruments)
        def forward(self, hidden_states):
            return torch.sigmoid(self.proj(hidden_states))


logger = logging.get_logger(__name__)


@dataclass
class Seq2SeqLMOutputNumInsts(Seq2SeqLMOutput):
    loss_inst: Optional[torch.FloatTensor] = None


class T5SegMemV2WithPrev(T5SegMemV2):
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
        cte_proj_dim = int(getattr(config, "cte_proj_dim", 64))
        cte_temperature = float(getattr(config, "cte_temperature", 0.07))
        self.cte = ContrastiveTimbreEmbedding(in_dim=d_model, proj_dim=cte_proj_dim, temperature=cte_temperature)

        self.issm = InstrumentSpecificSlotMemory(d_model=d_model, num_slots=17)
        self.memory_state = None
        
        # TAG Mechanism
        self.use_tag = getattr(config, "use_tag", False)
        if self.use_tag:
            num_instruments = getattr(config, "num_instruments", 128)  # default max midi instruments if not provided
            self.saliency_head = SaliencyHead(d_model=d_model, num_instruments=num_instruments)

    def _extract_m_gate(self, saliency_map: torch.Tensor, num_insts: torch.LongTensor) -> torch.Tensor:
        """Safely extracts the instrument-specific saliency gating vector."""
        if num_insts.dim() == 1 and num_insts.size(0) == saliency_map.size(0):
            batch_indices = torch.arange(saliency_map.size(0), device=saliency_map.device)
            return saliency_map[batch_indices, :, num_insts]
        elif num_insts.dim() == 0 or num_insts.size(0) == 1:
            idx = num_insts.item() if num_insts.dim() == 0 else num_insts[0].item()
            return saliency_map[:, :, idx]
        else:
            logger.warning(f"Shape mismatch in num_insts: {num_insts.shape}. Defaulting to uniform gate.")
            return torch.ones(saliency_map.size(0), saliency_map.size(1), device=saliency_map.device)

    
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
        targets_prev: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        reset_memory: bool = False,
        num_insts: Optional[torch.LongTensor] = None,
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
        
        # NOTE: Legacy T5SegMemV2WithPrev explicitly concatenated the b-1 label states.
        # This was stripped in favor of the specialized Instrument-Specific Slot Memory (ISSM).


        # TAG GATING COMPUTATION
        m_gate = None
        if self.use_tag and num_insts is not None:
            # P = SaliencyMap (B, T, K)
            saliency_map = self.saliency_head(hidden_states)
            m_gate = self._extract_m_gate(saliency_map, num_insts)
            
        extended_encoder_attention_mask = None
        if attention_mask is not None:
            extended_encoder_attention_mask = self.get_extended_attention_mask(
                attention_mask, hidden_states.shape[:2]
            )
        elif m_gate is not None:
            extended_encoder_attention_mask = torch.zeros(
                (hidden_states.size(0), 1, 1, hidden_states.size(1)),
                dtype=hidden_states.dtype,
                device=hidden_states.device
            )
            
        if m_gate is not None and extended_encoder_attention_mask is not None:
            eps = 1e-9
            tag_penalty = torch.log(m_gate + eps).unsqueeze(1).unsqueeze(2)
            extended_encoder_attention_mask = extended_encoder_attention_mask + tag_penalty

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=extended_encoder_attention_mask if extended_encoder_attention_mask is not None else attention_mask,
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
    
    def forward(
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
        targets_prev: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        num_insts: Optional[torch.LongTensor] = None,
        cte_family_id: Optional[torch.LongTensor] = None,
        reset_memory: bool = False,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutputNumInsts]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`
        Returns:
        Examples:
        ```python
        >>> from transformers import T5Tokenizer, T5ForConditionalGeneration
        >>> tokenizer = T5Tokenizer.from_pretrained("t5-small")
        >>> model = T5ForConditionalGeneration.from_pretrained("t5-small")
        >>> # training
        >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
        >>> # inference
        >>> input_ids = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
        >>> ).input_ids  # Batch size 1
        >>> outputs = model.generate(input_ids)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        >>> # studies have shown that owning a dog is good for you.
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        lm_logits, encoder_outputs, decoder_outputs = self.get_model_outputs(
            inputs=inputs,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            targets_prev=targets_prev,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            reset_memory=reset_memory,
            num_insts=num_insts,
        )

        loss_cte = None
        if cte_family_id is not None:
            enc_last_hidden = encoder_outputs[0]  # (B, T, D)
            try:
                # Add bounds validation for family_id if necessary, assuming valid inputs here
                _, loss_cte = self.cte(enc_last_hidden, attention_mask=attention_mask, family_id=cte_family_id, return_loss=True)
                
                # Safeguard against contrastive divergence NaN/Inf explosion
                if torch.isnan(loss_cte) or torch.isinf(loss_cte):
                    logger.warning("NaN/Inf detected in CTE loss. Zeroing out loss to prevent graph explosion.")
                    loss_cte = torch.tensor(0.0, device=loss_cte.device, requires_grad=True)
            except Exception as e:
                logger.error(f"CTE loss computation failed: {str(e)}. Zeroing out CTE loss.")
                loss_cte = torch.tensor(0.0, device=enc_last_hidden.device, requires_grad=True)
            
        if return_dict:
            return Seq2SeqLMOutputNumInsts(
                logits=lm_logits,
                loss_inst=loss_cte,
            )
        return (lm_logits, loss_cte)
        
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
                        # Update memory incrementally with the latest token representation
                        memory_state = self.issm.extract_and_update(issm_output[:, -1:, :].detach().clone(), memory_state)
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