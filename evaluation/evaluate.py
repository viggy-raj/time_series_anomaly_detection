"""
evaluation/evaluate.py
======================
Evaluation pipeline for the trained LSTM Autoencoder.

Steps
-----
1. Compute per-window reconstruction error on the test set
2. Compute per-window reconstruction error on the training set
   → Threshold = THRESHOLD_PERCENTILE-th percentile of train errors
3. Predict anomaly labels (error > threshold → anomaly)
4. Compute: Precision, Recall, F1, AUROC, Average Precision
5. Save metrics to outputs/metrics/metrics.json and metrics.csv
6. Save plots:
   - ROC curve
   - Precision-Recall curve
   - Anomaly overlay (score + threshold + ground truth)
   - Error distribution histogram
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    THRESHOLD_PERCENTILE, PLOT_DIR, METRIC_DIR, DEVICE,
)

plt.style.use("seaborn-v0_8-darkgrid")
FIG_DPI = 150


# ── Scoring ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_reconstruction_errors(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: str = DEVICE,
    has_labels: bool = False,
) -> Tuple[np.ndarray, np.ndarray | None]:
    """
    Run model on all windows in `loader` and return per-window MSE scores.

    Returns
    -------
    errors : np.ndarray, shape (N_windows,)
    labels : np.ndarray of int32 or None
    """
    dev = torch.device(device)
    model.eval()
    model.to(dev)

    all_errors = []
    all_labels = [] if has_labels else None

    for batch in loader:
        if has_labels:
            X, y = batch
            all_labels.append(y.numpy())
        else:
            X = batch[0]
        X = X.to(dev)
        recon, _ = model(X)
        # Per-window MSE: mean over (seq_len, n_features)
        errors = ((X - recon) ** 2).mean(dim=(1, 2)).cpu().numpy()
        all_errors.append(errors)

    errors = np.concatenate(all_errors, axis=0)
    labels = np.concatenate(all_labels, axis=0) if has_labels else None
    return errors, labels


def compute_threshold(train_errors: np.ndarray,
                      percentile: float = THRESHOLD_PERCENTILE) -> float:
    """Threshold = percentile of training reconstruction errors."""
    return float(np.percentile(train_errors, percentile))


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(
    test_errors: np.ndarray,
    test_labels: np.ndarray,
    threshold:   float,
) -> Dict[str, float]:
    """
    Compute binary classification metrics at the given threshold.

    Returns
    -------
    metrics : dict with Precision, Recall, F1, AUROC, AveragePrecision
    """
    preds = (test_errors >= threshold).astype(int)

    prec   = precision_score(test_labels, preds, zero_division=0)
    rec    = recall_score(test_labels, preds, zero_division=0)
    f1     = f1_score(test_labels, preds, zero_division=0)
    auroc  = roc_auc_score(test_labels, test_errors)
    avg_p  = average_precision_score(test_labels, test_errors)

    metrics = {
        "threshold":        round(threshold, 6),
        "precision":        round(prec,  4),
        "recall":           round(rec,   4),
        "f1_score":         round(f1,    4),
        "auroc":            round(auroc, 4),
        "average_precision": round(avg_p, 4),
        "n_anomalies_true": int(test_labels.sum()),
        "n_anomalies_pred": int(preds.sum()),
        "n_windows_total":  int(len(test_labels)),
    }

    print("\n=== Evaluation Metrics ===")
    for k, v in metrics.items():
        print(f"  {k:<25} {v}")
    print("=" * 30 + "\n")
    return metrics


def save_metrics(metrics: Dict[str, float]) -> None:
    """Save metrics to JSON and CSV in outputs/metrics/."""
    import csv
    os.makedirs(METRIC_DIR, exist_ok=True)

    json_path = os.path.join(METRIC_DIR, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved: {json_path}")

    csv_path = os.path.join(METRIC_DIR, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        writer.writeheader()
        writer.writerow(metrics)
    print(f"  Saved: {csv_path}")


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_roc_curve(test_errors: np.ndarray,
                   test_labels: np.ndarray,
                   auroc:       float) -> None:
    """ROC curve with AUC annotation."""
    fpr, tpr, _ = roc_curve(test_labels, test_errors)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#5b8dee", linewidth=2.5,
            label=f"LSTM-AE  (AUC = {auroc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random classifier")
    ax.fill_between(fpr, tpr, alpha=0.12, color="#5b8dee")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curve — Anomaly Detection", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.4)
    out = os.path.join(PLOT_DIR, "roc_curve.png")
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out}")


def plot_pr_curve(test_errors: np.ndarray,
                  test_labels: np.ndarray,
                  avg_prec:    float) -> None:
    """Precision-Recall curve."""
    prec, rec, _ = precision_recall_curve(test_labels, test_errors)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(rec, prec, color="#e07c5b", linewidth=2.5,
            label=f"LSTM-AE  (AP = {avg_prec:.4f})")
    ax.fill_between(rec, prec, alpha=0.12, color="#e07c5b")
    baseline = test_labels.mean()
    ax.axhline(baseline, linestyle="--", color="gray", linewidth=1.5,
               label=f"Baseline (prevalence={baseline:.3f})")
    ax.set_xlabel("Recall",    fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve — Anomaly Detection",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.4)
    out = os.path.join(PLOT_DIR, "pr_curve.png")
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out}")


def plot_anomaly_overlay(test_errors: np.ndarray,
                         test_labels: np.ndarray,
                         threshold:   float,
                         max_points:  int = 3000) -> None:
    """
    Reconstruction error time-series with ground-truth anomaly shading
    and the threshold line.
    """
    # Subsample for readability if test set is very large
    n = min(len(test_errors), max_points)
    idx    = np.linspace(0, len(test_errors) - 1, n, dtype=int)
    errors = test_errors[idx]
    labels = test_labels[idx]

    fig, ax = plt.subplots(figsize=(15, 5))

    # Ground-truth anomaly shading
    in_anom = False
    start   = 0
    for i, lbl in enumerate(labels):
        if lbl == 1 and not in_anom:
            start = i; in_anom = True
        elif lbl == 0 and in_anom:
            ax.axvspan(start, i, color="#e05252", alpha=0.25, linewidth=0)
            in_anom = False
    if in_anom:
        ax.axvspan(start, len(labels), color="#e05252", alpha=0.25, linewidth=0)

    # Reconstruction error trace
    ax.plot(errors, color="#5b8dee", linewidth=0.8, alpha=0.9,
            label="Reconstruction Error")
    # Threshold line
    ax.axhline(threshold, color="#f5a623", linestyle="--",
               linewidth=1.8, label=f"Threshold ({threshold:.4f})")

    # Legend patches
    anom_patch  = mpatches.Patch(color="#e05252", alpha=0.3, label="True Anomaly")
    ax.legend(handles=[ax.lines[0], ax.lines[1], anom_patch],
              fontsize=11, loc="upper left")

    ax.set_xlabel("Window Index (subsampled)", fontsize=12)
    ax.set_ylabel("MSE Reconstruction Error",  fontsize=12)
    ax.set_title("Anomaly Detection — Reconstruction Error vs Ground Truth",
                 fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "anomaly_overlay.png")
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out}")


def plot_error_distribution(train_errors: np.ndarray,
                             test_errors:  np.ndarray,
                             threshold:    float) -> None:
    """Histogram of reconstruction errors (train vs test)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(train_errors, bins=80, color="#5b8dee", alpha=0.65,
            label="Train errors", density=True)
    ax.hist(test_errors,  bins=80, color="#e07c5b", alpha=0.55,
            label="Test errors",  density=True)
    ax.axvline(threshold, color="#f5a623", linestyle="--",
               linewidth=2, label=f"Threshold ({threshold:.4f})")
    ax.set_xlabel("Reconstruction Error (MSE)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Error Distribution — Train vs Test",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.4)
    out = os.path.join(PLOT_DIR, "error_distribution.png")
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out}")


# ── Public entry point ────────────────────────────────────────────────────────

def run_evaluation(
    model,
    loaders: Dict,
    device:  str = DEVICE,
    threshold_pct: float = THRESHOLD_PERCENTILE,
) -> Tuple[Dict[str, float], np.ndarray, float]:
    """
    Full evaluation pipeline.

    Returns
    -------
    metrics      : dict of performance metrics
    test_errors  : per-window reconstruction errors (test set)
    threshold    : anomaly threshold value
    """
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(METRIC_DIR, exist_ok=True)

    print("\n=== Running Evaluation ===")

    # Compute reconstruction errors
    print("  Computing train reconstruction errors …")
    train_errors, _ = compute_reconstruction_errors(
        model, loaders["train"], device, has_labels=False)

    print("  Computing test reconstruction errors  …")
    test_errors, test_labels = compute_reconstruction_errors(
        model, loaders["test"], device, has_labels=True)

    # Threshold
    threshold = compute_threshold(train_errors, threshold_pct)
    print(f"  Threshold ({threshold_pct}th percentile): {threshold:.6f}")

    # Metrics
    metrics = compute_metrics(test_errors, test_labels, threshold)
    save_metrics(metrics)

    # Plots
    print("  Generating evaluation plots …")
    plot_roc_curve(test_errors, test_labels, metrics["auroc"])
    plot_pr_curve(test_errors, test_labels, metrics["average_precision"])
    plot_anomaly_overlay(test_errors, test_labels, threshold)
    plot_error_distribution(train_errors, test_errors, threshold)

    print("=== Evaluation Complete ===\n")
    return metrics, test_errors, threshold
