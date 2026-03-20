import torch
import torch.nn as nn
from einops import rearrange

class InstrumentSpecificSlotMemory(nn.Module):
    def __init__(self, d_model=768, num_slots=17):
        """
        K = 17 slots (16 MIDI families + 1 Drum track)
        D = 768 (Conformer / Transformer Hidden Dimension)
        """
        super().__init__()
        self.d_model = d_model
        self.num_slots = num_slots

        # 1. Routing Head (W): Predicts frame-level instrument independent probabilities W (B, T, K)
        # Using Sigmoid allows polyphony across slots (unlike Softmax)
        self.routing_head = nn.Sequential(
            nn.Linear(d_model, num_slots),
            nn.Sigmoid()
        )

        # 2. Slot Update Mechanism: GRU handling (B*K, 1, D) for parallel batches
        self.gru = nn.GRU(input_size=d_model, hidden_size=d_model, batch_first=True)
        self.norm_memory = nn.LayerNorm(d_model)

        # 3. Decoder Integration Projections (Memory Bank Keys & Values)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm_attn = nn.LayerNorm(d_model)

        # 4. Gating Mechanism initialized to 0.1 for faster learning
        self.m_gate = nn.Parameter(torch.tensor([0.1]))

    def init_memory(self, batch_size, device):
        """Initializes empty/zero memory for start of sequence."""
        return torch.zeros(batch_size, self.num_slots, self.d_model, device=device)

    def extract_and_update(self, H, M_old):
        """
        Aggregates frame-level encoder signals and updates the memory state.
        H: (B, T, D) - Encoder Output
        M_old: (B, K, D) - Previous Segment's Memory
        Returns: M_new (B, K, D)
        """
        B, T, D = H.shape

        # A) Frame-Level Routing
        W = self.routing_head(H)     # (B, T, K)
        
        # B) Aggregation: Weighted sum of H over T 
        # W^T -> (B, K, T) multiplied by H -> (B, T, D) results in Aggregator -> (B, K, D)
        # Using batched matrix multiplication for high vectorization efficiency
        Agg_H = torch.bmm(W.transpose(1, 2), H)

        # C) GRU Mechanism Memory Slot Update
        # Reshape to fold Batch and Slot dimensions for generic GRU interaction: (B*K, D)
        H_in = rearrange(Agg_H, 'b k d -> (b k) d')
        M_in = rearrange(M_old, 'b k d -> (b k) d')

        # Using B*K as batch dimension to fully parallelize:
        # H_in: (B*K, 1, D), M_in: (1, B*K, D)
        M_new_flat, _ = self.gru(H_in.unsqueeze(1), M_in.unsqueeze(0))
        M_new_flat = M_new_flat.squeeze(1)  # (B*K, D)
        
        # Apply normalization logic
        M_new_flat = self.norm_memory(M_new_flat)

        # D) Unpack shape
        M_new = rearrange(M_new_flat, '(b k) d -> b k d', b=B, k=self.num_slots)
        return M_new

    def memory_cross_attention(self, decoder_hidden, M):
        """
        Lightweight Secondary Pass: The Decoder queries the updated memory bank.
        decoder_hidden: (B, L_q, D) - Output from standard T5 decoder cross-attention
        M: (B, K, D) - The updated Memory tensor
        """
        B, L_q, D = decoder_hidden.shape

        # Project representations
        Q = self.q_proj(decoder_hidden) # (B, L_q, D)
        K = self.k_proj(M)              # (B, K, D)
        V = self.v_proj(M)              # (B, K, D)

        # Dot-product attention scores
        scores = torch.bmm(Q, K.transpose(1, 2)) / (D ** 0.5)   # (B, L_q, K)
        attn_weights = torch.softmax(scores, dim=-1)            # (B, L_q, K)

        # Context vector
        memory_context = torch.bmm(attn_weights, V)             # (B, L_q, D)
        
        # Normalization layer stabilizing the cross attention scores
        memory_context = self.norm_attn(memory_context)

        # Add Gated Residual connection
        return decoder_hidden + self.m_gate * memory_context
