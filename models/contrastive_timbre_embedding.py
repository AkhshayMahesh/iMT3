from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import yaml  # type: ignore
except Exception as e:  # pragma: no cover
    yaml = None  # type: ignore[assignment]
    _YAML_IMPORT_ERROR = e


PathLike = Union[str, Path]


@dataclass(frozen=True)
class SlakhStemLabel:
    """
    A compact, model-ready label for supervised contrastive learning.
    """

    family_id: int  # 0..15 for GM family, 16 reserved for drums


def _require_yaml() -> None:
    if yaml is None:  # pragma: no cover
        raise ImportError(
            "PyYAML is required to parse Slakh metadata.yaml files. "
            "Install it (e.g. `pip install pyyaml`)."
        ) from _YAML_IMPORT_ERROR


_metadata_cache: dict[str, dict] = {}

def _load_slakh_metadata(metadata_yaml_path: str) -> dict:
    if metadata_yaml_path not in _metadata_cache:
        _require_yaml()
        meta_txt = Path(metadata_yaml_path).read_text()
        _metadata_cache[metadata_yaml_path] = yaml.safe_load(meta_txt) or {}
    return _metadata_cache[metadata_yaml_path]


def slakh_family_id_from_metadata(
    metadata_yaml_path: PathLike,
    stem_id: str,
) -> SlakhStemLabel:
    """
    Computes a family_id label from Slakh `metadata.yaml` for a specific stem.

    Rules:
    - **Drums**: if `is_drum=True` -> family_id = 16
    - **General MIDI family**: family_id = program_num // 8   (0..15)
    """
    meta = _load_slakh_metadata(str(metadata_yaml_path))
    stems = meta.get("stems", {}) or {}
    stem_meta = stems.get(stem_id, {}) or {}

    if bool(stem_meta.get("is_drum", False)):
        return SlakhStemLabel(family_id=16)

    program_num = stem_meta.get("program_num", None)
    if program_num is None:
        raise KeyError(
            f"Missing program_num for stem_id={stem_id!r} in metadata file {str(metadata_yaml_path)!r}."
        )
    program_num = int(program_num)
    if program_num < 0:
        raise ValueError(f"program_num must be >= 0, got {program_num} for stem_id={stem_id!r}.")
    return SlakhStemLabel(family_id=program_num // 8)


def slakh_family_ids_from_batch(
    metadata_yaml_paths: Sequence[PathLike],
    stem_ids: Sequence[str],
    *,
    device: Optional[torch.device] = None,
) -> torch.LongTensor:
    """
    Batch helper for producing `family_id` labels.
    """
    if len(metadata_yaml_paths) != len(stem_ids):
        raise ValueError(
            f"metadata_yaml_paths and stem_ids must have same length, got {len(metadata_yaml_paths)} and {len(stem_ids)}."
        )
    family = [slakh_family_id_from_metadata(p, s).family_id for p, s in zip(metadata_yaml_paths, stem_ids)]
    return torch.tensor(family, dtype=torch.long, device=device)


class ContrastiveTimbreEmbedding(nn.Module):
    """
    Contrastive Timbre Embeddings (CTE) module.

    - Input: Conformer encoder output `H` with shape [B, T, 768]
    - Temporal aggregation: Global Average Pooling (masked if attention_mask provided)
    - Projection head (MLP): 768 -> 768 -> 64
    - Output: L2-normalized embeddings with shape [B, 64]

    Also provides a supervised contrastive loss (SupCon / generalized NT-Xent).
    """

    def __init__(
        self,
        *,
        in_dim: int = 768,
        proj_dim: int = 64,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.temperature = float(temperature)
        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}.")

        self.proj = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, proj_dim),
        )

    @staticmethod
    def global_average_pool(x: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        x: [B, T, D]
        attention_mask: [B, T] with 1/True for valid tokens, 0/False for padding.
        """
        if attention_mask is None:
            return x.mean(dim=1)

        m = attention_mask.to(dtype=x.dtype)
        if m.dim() != 2:
            raise ValueError(f"attention_mask must be [B, T], got shape {tuple(m.shape)}.")
        m = m.unsqueeze(-1)  # [B, T, 1]
        denom = m.sum(dim=1).clamp_min(1.0)
        return (x * m).sum(dim=1) / denom

    def embeddings(
        self,
        conformer_out: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns L2-normalized embeddings: [B, 64]
        """
        if conformer_out.dim() != 3:
            raise ValueError(f"conformer_out must be rank-3 [B,T,D], got shape {tuple(conformer_out.shape)}.")
        pooled = self.global_average_pool(conformer_out, attention_mask)  # [B, 768]
        z = self.proj(pooled)  # [B, 64]
        return F.normalize(z, dim=-1)

    def supcon_loss(
        self,
        z: torch.Tensor,
        family_id: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Supervised Contrastive Loss (SupCon / generalized NT-Xent).

        Requirements satisfied:
        - **Temperature**: self.temperature (default 0.07)
        - **Boolean similarity mask [B,B]** based on `family_id`
        - **Multiple positives per anchor** supported
        - **LogSumExp** trick for numerical stability
        - **Diagonal masked out** (no self-matching)
        - **Averages correctly over #positives per row**
        """
        if z.dim() != 2:
            raise ValueError(f"z must be [B, D], got shape {tuple(z.shape)}.")
        if family_id.dim() != 1 or family_id.shape[0] != z.shape[0]:
            raise ValueError(
                f"family_id must be [B] matching z, got {tuple(family_id.shape)} vs B={z.shape[0]}."
            )
        b = z.shape[0]
        if b < 2:
            return z.new_zeros(())

        logits = (z @ z.transpose(0, 1)) / self.temperature  # [B, B]

        device = z.device
        diag = torch.eye(b, dtype=torch.bool, device=device)

        family_id = family_id.to(device=device)
        pos_mask = (family_id.unsqueeze(0) == family_id.unsqueeze(1)) & ~diag  # [B, B] bool
        pos_count = pos_mask.sum(dim=1)  # [B]
        valid = pos_count > 0
        if not torch.any(valid):
            return z.new_zeros(())

        logits = logits.masked_fill(diag, float("-inf"))  # exclude self from denom

        # log_den[i] = log sum_{a != i} exp(logits[i, a])
        log_den = torch.logsumexp(logits, dim=1)  # [B]
        log_prob = logits - log_den.unsqueeze(1)  # [B, B]
        log_prob = log_prob.masked_fill(diag, 0.0)

        # SupCon: mean over positives per anchor, then mean across valid anchors
        loss_per_anchor = -(log_prob * pos_mask.to(dtype=log_prob.dtype)).sum(dim=1) / pos_count.clamp_min(1).to(
            dtype=log_prob.dtype
        )
        return loss_per_anchor[valid].mean()

    def forward(
        self,
        conformer_out: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        family_id: Optional[torch.LongTensor] = None,
        metadata_yaml_paths: Optional[Sequence[PathLike]] = None,
        stem_ids: Optional[Sequence[str]] = None,
        return_loss: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        If `return_loss=True`, computes SupCon loss using either:
        - provided `family_id` tensor, OR
        - `metadata_yaml_paths` + `stem_ids` to derive labels on the fly.
        """
        z = self.embeddings(conformer_out, attention_mask=attention_mask)

        loss: Optional[torch.Tensor] = None
        if return_loss:
            if family_id is None:
                if metadata_yaml_paths is None or stem_ids is None:
                    raise ValueError("To compute loss, provide `family_id` OR (`metadata_yaml_paths` and `stem_ids`).")
                family_id = slakh_family_ids_from_batch(
                    metadata_yaml_paths,
                    stem_ids,
                    device=z.device,
                )
            loss = self.supcon_loss(z, family_id)

        return z, loss

