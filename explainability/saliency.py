"""
explainability/saliency.py
===========================
Gradient-based saliency maps for the LSTM Autoencoder.

What it does
------------
For each anomalous window, computes the gradient of the reconstruction
error w.r.t. the input. High gradient magnitude at a (timestep, feature)
position means that feature at that time strongly influenced the anomaly score.

This is complementary to SHAP and runs without the SHAP library:
  - SHAP tells us *which features* matter globally
  - Saliency tells us *when* within the window those features spike

Outputs
-------
  outputs/plots/saliency_<k>.png   (heatmap per anomaly window)
"""

import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    PLOT_DIR, DEVICE,
    NUM_ANOMALIES_TO_EXPLAIN,
)

FIG_DPI = 150


def compute_saliency(
    model:       nn.Module,
    windows:     np.ndarray,
    device:      str = DEVICE,
) -> np.ndarray:
    """
    Compute input-gradient saliency for each window.

    Parameters
    ----------
    model   : Trained LSTMAutoencoder.
    windows : np.ndarray, shape (K, seq_len, n_features)
    device  : 'cpu' or 'cuda'.

    Returns
    -------
    saliency : np.ndarray, shape (K, seq_len, n_features)
               Absolute gradient magnitudes, L2-normalised per window.
    """
    dev = torch.device(device)
    # cuDNN LSTM backward requires training mode (even for attribution).
    # dropout=0 by default, so outputs are identical to eval mode.
    model.train()
    model.to(dev)

    X = torch.from_numpy(windows.astype(np.float32)).to(dev)
    X.requires_grad_(True)

    recon, _ = model(X)
    # Scalar loss: mean reconstruction error over all windows
    loss = ((X - recon) ** 2).mean(dim=(1, 2))   # (K,)
    loss.sum().backward()

    grad     = X.grad.detach().cpu().numpy()      # (K, seq_len, F)
    saliency = np.abs(grad)

    # L2 normalise each window independently
    norms = np.linalg.norm(saliency, axis=(1, 2), keepdims=True) + 1e-10
    saliency = saliency / norms

    return saliency


def plot_saliency_heatmap(
    saliency:      np.ndarray,
    feature_names: List[str],
    anomaly_errors: np.ndarray,
) -> None:
    """
    Plot a seaborn heatmap (timesteps × features) for each window.

    saliency : shape (K, seq_len, n_features)
    """
    os.makedirs(PLOT_DIR, exist_ok=True)
    K, T, F = saliency.shape

    for k in range(K):
        sal = saliency[k]        # (T, F)

        fig, ax = plt.subplots(figsize=(max(10, F * 0.5), max(4, T * 0.1)))

        sns.heatmap(
            sal.T,                     # (F, T) — features on y-axis
            ax=ax,
            cmap="YlOrRd",
            yticklabels=feature_names,
            xticklabels=False,         # too many timesteps to label individually
            linewidths=0,
            cbar_kws={"label": "Normalised |gradient|", "shrink": 0.6},
        )
        ax.set_xlabel("Timestep (within window)", fontsize=11)
        ax.set_ylabel("Feature", fontsize=11)
        ax.set_title(
            f"Saliency Heatmap — Anomaly #{k+1}\n"
            f"Reconstruction Error: {anomaly_errors[k]:.4f}",
            fontsize=13, fontweight="bold"
        )
        ax.tick_params(axis="y", labelsize=7)
        plt.tight_layout()

        out = os.path.join(PLOT_DIR, f"saliency_{k+1:02d}.png")
        fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")


# ── Public entry point ────────────────────────────────────────────────────────

def run_saliency_explanation(
    model:          nn.Module,
    test_errors:    np.ndarray,
    test_windows:   np.ndarray,
    threshold:      float,
    feature_names:  List[str],
    device:         str = DEVICE,
    n_explain:      int = NUM_ANOMALIES_TO_EXPLAIN,
) -> None:
    """
    Full saliency pipeline: compute + plot for top-N anomaly windows.

    Parameters
    ----------
    model          : Trained LSTMAutoencoder.
    test_errors    : Per-window reconstruction errors.
    test_windows   : np.ndarray (N_test, seq_len, n_features).
    threshold      : Anomaly threshold.
    feature_names  : Feature column names.
    device         : 'cpu' or 'cuda'.
    n_explain      : Number of anomaly windows to plot.
    """
    print("\n=== Running Saliency Maps ===")

    anom_idx = np.where(test_errors >= threshold)[0]
    if len(anom_idx) == 0:
        print("  No anomalies detected — skipping saliency.")
        return

    # Top-N by reconstruction error
    anom_idx  = anom_idx[np.argsort(test_errors[anom_idx])[::-1]][:n_explain]
    anom_wins = test_windows[anom_idx]
    anom_errs = test_errors[anom_idx]

    print(f"  Computing saliency for {len(anom_idx)} anomaly windows …")
    saliency = compute_saliency(model, anom_wins, device)
    plot_saliency_heatmap(saliency, feature_names, anom_errs)

    print("=== Saliency Maps Complete ===\n")
