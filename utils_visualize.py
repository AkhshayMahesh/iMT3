import torch
import numpy as np
import matplotlib

# Use non-interactive backend (safe for servers)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def plot_latent_embeddings(
    embeddings,
    labels,
    logger=None,
    current_epoch=None,
    save_path=None,
    max_samples=5000,
):
    """
    Plot 2D t-SNE of embeddings colored by labels.
    Handles edge cases: shape issues, NaNs, small/large datasets.
    """

    if embeddings is None or len(embeddings) == 0:
        return

    # --- Convert to numpy ---
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    embeddings = np.array(embeddings)
    labels = np.array(labels).flatten()

    # --- Fix shape ---
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(-1, 1)
    elif embeddings.ndim > 2:
        embeddings = embeddings.reshape(embeddings.shape[0], -1)

    # --- Validate ---
    if embeddings.shape[0] != labels.shape[0]:
        raise ValueError("Embeddings and labels size mismatch")

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain NaN or Inf")

    # --- Subsample if too large ---
    n_samples = embeddings.shape[0]
    if n_samples > max_samples:
        idx = np.random.choice(n_samples, max_samples, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]
        n_samples = max_samples

    # --- Compute t-SNE ---
    if n_samples > 1:
        perplexity = max(1, min(30, (n_samples - 1) // 3))
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=42,
            init="pca",
            learning_rate="auto",
        )
        embeddings_2d = tsne.fit_transform(embeddings)
    else:
        embeddings_2d = np.zeros((n_samples, 2))

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=labels.astype(int),
        cmap="tab20",
        alpha=0.7,
        s=15,
    )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("CTE Family ID")

    ax.set_title("t-SNE of CTE Latent Embeddings")
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")

    # --- Save ---
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)

    # --- Log ---
    if logger is not None and current_epoch is not None:
        try:
            logger.experiment.add_figure(
                "CTE_Embeddings", fig, global_step=current_epoch
            )
        except Exception as e:
            print(f"TensorBoard logging failed: {e}")

    plt.close(fig)