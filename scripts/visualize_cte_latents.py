from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from models.t5_segmem_v2_with_prev import T5SegMemV2WithPrev
from models.t5_segmem_v2 import T5Config
from models.contrastive_timbre_embedding import ContrastiveTimbreEmbedding


def _collect_embeddings(
    model: T5SegMemV2WithPrev,
    dataloader,
    *,
    max_batches: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    zs: List[np.ndarray] = []
    ys: List[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for bi, batch in enumerate(dataloader):
            if bi >= max_batches:
                break

            if len(batch) == 4:
                inputs, _, targets_prev, family_id = batch
            else:
                raise ValueError("Expected batch to include cte_family_id (4th item).")

            inputs = inputs.to(device)
            targets_prev = targets_prev.to(device)
            family_id = family_id.to(device)

            # Run encoder only (match model's internal projection)
            inputs_embeds = model.proj(inputs)
            enc = model.encoder(inputs_embeds=inputs_embeds, return_dict=True)
            enc_last = enc.last_hidden_state  # [B, T, D]

            z = model.cte.embeddings(enc_last, attention_mask=None)  # [B, 64]
            zs.append(z.detach().cpu().numpy())
            ys.append(family_id.detach().cpu().numpy())

    return np.concatenate(zs, axis=0), np.concatenate(ys, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="Path to .ckpt (Lightning) or HF-style state dict")
    ap.add_argument("--config", type=str, required=True, help="Path to hydra config.yaml used for training")
    ap.add_argument("--max_batches", type=int, default=50)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--method", type=str, choices=["tsne", "umap"], default="tsne")
    ap.add_argument("--out", type=str, default="cte_latents.png")
    args = ap.parse_args()

    # Minimal config load: hydra writes full resolved yaml at outputs/.../.hydra/config.yaml
    import yaml  # type: ignore

    cfg = yaml.safe_load(Path(args.config).read_text())
    t5cfg = T5Config.from_dict(cfg["model"] if "model" in cfg else cfg)

    model = T5SegMemV2WithPrev(config=t5cfg, segmem_num_layers=t5cfg.segmem_num_layers, segmem_length=t5cfg.segmem_length)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    # Lightning prefixes "model." in MT3 tasks
    state = {k.replace("model.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)

    device = torch.device(args.device)
    model.to(device)

    # Build dataset/dataloader the same way training does (via hydra target)
    from omegaconf import OmegaConf
    import hydra.utils as hy

    ds_cfg = OmegaConf.create(cfg["dataset"]["val"] if "dataset" in cfg and "val" in cfg["dataset"] else cfg["dataset"])
    dataset = hy.instantiate(ds_cfg)
    collate_fn = hy.get_method(cfg["dataset"]["collate_fn"]) if "dataset" in cfg and "collate_fn" in cfg["dataset"] else None
    from torch.utils.data import DataLoader

    dl = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False, collate_fn=collate_fn)

    z, y = _collect_embeddings(model, dl, max_batches=args.max_batches, device=device)

    if args.method == "tsne":
        from sklearn.manifold import TSNE

        emb2 = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=30).fit_transform(z)
    else:
        import umap  # type: ignore

        emb2 = umap.UMAP(n_components=2, metric="cosine").fit_transform(z)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    for fam in sorted(set(y.tolist())):
        mask = y == fam
        plt.scatter(emb2[mask, 0], emb2[mask, 1], s=10, alpha=0.7, label=f"fam{fam}")
    plt.legend(markerscale=2, fontsize=8, ncol=2, frameon=False)
    plt.title("CTE latent space (colored by GM family_id; drums=16)")
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)


if __name__ == "__main__":
    main()

