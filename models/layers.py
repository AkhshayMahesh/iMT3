import torch
import torch.nn as nn

class SaliencyHead(nn.Module):
    def __init__(self, d_model: int, num_instruments: int):
        super().__init__()
        self.proj = nn.Linear(d_model, num_instruments)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: (Batch, SeqLen, D_model)
        Returns:
        saliency_map: (Batch, SeqLen, NumInstruments)
        """
        logits = self.proj(hidden_states)
        return torch.sigmoid(logits)
