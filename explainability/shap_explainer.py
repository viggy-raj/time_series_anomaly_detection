"""
explainability/shap_explainer.py
=================================
Feature attribution for the LSTM Autoencoder using Integrated Gradients (IG).

Why Integrated Gradients instead of SHAP GradientExplainer?
------------------------------------------------------------
SHAP's GradientExplainer has known tensor-dimension issues with LSTM models
that return a scalar output (batch,). Integrated Gradients (Sundararajan et al.,
NeurIPS 2017) is equally well-cited in the XAI literature, is fully compatible
with PyTorch LSTM models, and satisfies the same completeness & sensitivity axioms
as SHAP. It is the recommended attribution method for deep models in the
Captum library (Facebook) and Google's TCAV framework.

Method
------
For each anomalous window x, IG computes:

    IG(x) = (x - x0) * INTEGRAL_0^1 [grad_F(x0 + alpha*(x-x0)) d_alpha]

where x0 is the baseline (zero vector = "no signal") and F is the scalar
reconstruction error. We approximate the integral with N_STEPS Riemann steps.

Outputs
-------
  outputs/plots/ig_summary.png           -- Global feature importance bar chart
  outputs/plots/ig_anomaly_<k>.png       -- Per-anomaly top-10 feature bars
  outputs/metrics/shap_feature_importance.json  -- Feature ranking (JSON)
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    PLOT_DIR, METRIC_DIR,
    NUM_ANOMALIES_TO_EXPLAIN,
    DEVICE,
)

FIG_DPI     = 150
IG_STEPS    = 50   # Riemann integration steps (50 is standard in literature)


# ── Model wrapper ─────────────────────────────────────────────────────────────

class ReconErrorWrapper(nn.Module):
    """Returns per-window scalar reconstruction error (MSE)."""

    def __init__(self, autoencoder: nn.Module):
        super().__init__()
        self.ae = autoencoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        recon, _ = self.ae(x)
        return ((x - recon) ** 2).mean(dim=(1, 2))   # shape: (batch,)


# ── Integrated Gradients ──────────────────────────────────────────────────────

def integrated_gradients(
    model:  nn.Module,
    inputs: torch.Tensor,
    steps:  int = IG_STEPS,
) -> np.ndarray:
    """
    Compute Integrated Gradients attributions for a batch of windows.

    Parameters
    ----------
    model  : ReconErrorWrapper (or any differentiable model returning (batch,))
    inputs : Tensor, shape (K, seq_len, n_features) — anomaly windows
    steps  : Number of Riemann integration steps

    Returns
    -------
    attrs : np.ndarray, shape (K, seq_len, n_features)
            Signed attributions (positive = increased anomaly score)
    """
    baseline = torch.zeros_like(inputs)   # zero-signal baseline

    # Build interpolated path: alpha in [0, 1] with `steps` points
    # Shape: (steps, K, seq_len, n_features)
    alphas      = torch.linspace(0.0, 1.0, steps, device=inputs.device)
    # Expand dims for broadcasting
    interp      = baseline.unsqueeze(0) + \
                  alphas.view(steps, 1, 1, 1) * (inputs - baseline).unsqueeze(0)
    # Merge steps into batch: (steps*K, seq_len, n_features)
    K, T, F     = inputs.shape
    interp_flat = interp.view(steps * K, T, F).detach().requires_grad_(True)

    # Forward pass on interpolated inputs
    scores = model(interp_flat)    # (steps*K,)
    scores.sum().backward()

    # Gradients: (steps*K, seq_len, n_features) → (steps, K, T, F)
    grads     = interp_flat.grad.detach().view(steps, K, T, F)
    avg_grads = grads.mean(dim=0)              # (K, T, F) — average over path

    # Element-wise product with (input - baseline)
    delta     = (inputs - baseline).detach()   # (K, T, F)
    attrs     = (delta * avg_grads).cpu().numpy()
    return attrs


# ── Aggregate importance ──────────────────────────────────────────────────────

def _mean_abs_importance(attrs: np.ndarray) -> np.ndarray:
    """Mean |attribution| over (windows, timesteps) → per-feature. Shape: (F,)"""
    return np.abs(attrs).mean(axis=(0, 1))


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_ig_summary(attrs: np.ndarray, feature_names: List[str]) -> None:
    """Global IG feature importance: horizontal bar chart (all features)."""
    os.makedirs(PLOT_DIR, exist_ok=True)
    importances = _mean_abs_importance(attrs)
    order       = np.argsort(importances)       # ascending for barh
    labels      = [feature_names[i] for i in order]
    vals        = importances[order]
    colors      = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, len(vals)))

    fig, ax = plt.subplots(figsize=(10, max(5, len(feature_names) * 0.38)))
    ax.barh(labels, vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Mean |Integrated Gradient|  (impact on anomaly score)", fontsize=11)
    ax.set_title(
        "Integrated Gradients -- Global Feature Importance\n"
        "(Higher = more responsible for anomaly scores)",
        fontsize=13, fontweight="bold"
    )
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "ig_summary.png")
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_ig_per_anomaly(attrs: np.ndarray,
                        feature_names: List[str],
                        anomaly_errors: np.ndarray) -> None:
    """Per-anomaly top-10 feature attribution bar chart."""
    os.makedirs(PLOT_DIR, exist_ok=True)
    K = attrs.shape[0]

    for k in range(K):
        # Mean |IG| over timesteps for this window: (F,)
        imp   = np.abs(attrs[k]).mean(axis=0)
        order = np.argsort(imp)[::-1][:10]    # top-10
        vals  = imp[order]
        lbl   = [feature_names[i] for i in order]
        clrs  = ["#e05252" if v > vals.mean() else "#5b8dee" for v in vals]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(lbl[::-1], vals[::-1], color=clrs[::-1],
                edgecolor="white", linewidth=0.5)
        ax.set_xlabel("Mean |Integrated Gradient| over window timesteps", fontsize=10)
        ax.set_title(
            f"Anomaly #{k+1} -- Top Feature Attributions\n"
            f"Reconstruction Error: {anomaly_errors[k]:.5f}",
            fontsize=12, fontweight="bold"
        )
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        out = os.path.join(PLOT_DIR, f"shap_anomaly_{k+1:02d}.png")
        fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")


def save_feature_importance(attrs: np.ndarray,
                             feature_names: List[str]) -> Dict[str, float]:
    """Save mean |IG| per feature to JSON (same key as SHAP for compatibility)."""
    os.makedirs(METRIC_DIR, exist_ok=True)
    importances = _mean_abs_importance(attrs)
    feat_imp    = {
        name: round(float(v), 6)
        for name, v in zip(feature_names, importances)
    }
    feat_imp_sorted = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True))

    path = os.path.join(METRIC_DIR, "shap_feature_importance.json")
    with open(path, "w") as f:
        json.dump(feat_imp_sorted, f, indent=2)
    print(f"  Saved: {path}")
    return feat_imp_sorted


# ── Global IG summary heatmap ─────────────────────────────────────────────────

def plot_ig_summary_heatmap(attrs: np.ndarray,
                             feature_names: List[str]) -> None:
    """
    Heatmap of mean signed IG over time (averaged across all anomaly windows).
    Rows = features, Cols = timesteps within window.
    Shows WHEN each feature drives the anomaly score.
    """
    try:
        import seaborn as sns
    except ImportError:
        return   # optional dependency

    # Mean signed attribution over anomaly windows: (T, F)
    mean_attrs = attrs.mean(axis=0)   # (T, F)

    fig, ax = plt.subplots(figsize=(14, max(5, len(feature_names) * 0.35)))
    sns.heatmap(
        mean_attrs.T,             # (F, T)
        ax=ax,
        cmap="coolwarm", center=0,
        yticklabels=feature_names,
        xticklabels=False,
        linewidths=0,
        cbar_kws={"label": "Mean Signed IG Attribution", "shrink": 0.6},
    )
    ax.set_xlabel("Timestep within window", fontsize=11)
    ax.set_ylabel("Feature", fontsize=11)
    ax.set_title(
        "Integrated Gradients -- Mean Attribution Heatmap\n"
        "(Averaged across all explained anomaly windows)",
        fontsize=13, fontweight="bold"
    )
    ax.tick_params(axis="y", labelsize=7)
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "ig_heatmap.png")
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Public entry point ────────────────────────────────────────────────────────

def run_shap_explanation(
    model:          nn.Module,
    test_errors:    np.ndarray,
    test_windows:   np.ndarray,
    train_windows:  np.ndarray,   # kept for API compatibility, not used by IG
    threshold:      float,
    feature_names:  List[str],
    device:         str = DEVICE,
    n_explain:      int = NUM_ANOMALIES_TO_EXPLAIN,
) -> Dict[str, float]:
    """
    Full Integrated Gradients explanation pipeline.

    API is identical to the original SHAP version so main.py is unchanged.
    """
    print("\n=== Running Integrated Gradients Explainability ===")

    anom_idx = np.where(test_errors >= threshold)[0]
    if len(anom_idx) == 0:
        print("  No anomalies detected -- skipping IG.")
        return {}

    # Top-N highest-error anomaly windows
    anom_idx  = anom_idx[np.argsort(test_errors[anom_idx])[::-1]][:n_explain]
    anom_wins = test_windows[anom_idx]    # (K, T, F)
    anom_errs = test_errors[anom_idx]
    print(f"  Selected {len(anom_idx)} anomaly windows (highest reconstruction error).")

    # Wrap model
    dev     = torch.device(device)
    wrapper = ReconErrorWrapper(model).to(dev)
    # NOTE: cuDNN LSTM backward requires training mode, even for attribution.
    # We set train() here; since LSTM dropout defaults to 0, outputs are identical.
    wrapper.train()

    # Tensor inputs
    X = torch.from_numpy(anom_wins.astype(np.float32)).to(dev)

    print(f"  Computing Integrated Gradients ({IG_STEPS} steps) ...")
    with torch.enable_grad():
        attrs = integrated_gradients(wrapper, X, steps=IG_STEPS)   # (K, T, F)

    # Plots
    plot_ig_summary(attrs, feature_names)
    plot_ig_per_anomaly(attrs, feature_names, anom_errs)
    plot_ig_summary_heatmap(attrs, feature_names)

    # Save JSON
    feat_importance = save_feature_importance(attrs, feature_names)

    print("=== Integrated Gradients Complete ===\n")
    return feat_importance
