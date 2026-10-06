"""
eda/eda.py
==========
Exploratory Data Analysis for the PSM dataset.

Generates and saves:
  - Basic statistics (shape, dtypes, null counts, summary stats)
  - Time-series line plots for a sample of features
  - Anomaly label distribution bar chart
  - Feature correlation heatmap

All plots are saved to outputs/plots/eda/.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    PSM_TRAIN_FILE, PSM_TEST_FILE, PSM_LABEL_FILE,
    EDA_DIR,
)

# ── Styling ──────────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-darkgrid")
PALETTE   = "viridis"
FIG_DPI   = 150
SAMPLE_N  = 4          # Number of features to plot as time-series


def _ensure_dir():
    os.makedirs(EDA_DIR, exist_ok=True)


def load_data():
    """Load PSM CSVs and return train_df, test_df, label_series."""
    train_df = pd.read_csv(PSM_TRAIN_FILE)
    test_df  = pd.read_csv(PSM_TEST_FILE)
    label_df = pd.read_csv(PSM_LABEL_FILE)

    # Drop any non-numeric 'timestamp' column if present
    for df in [train_df, test_df]:
        if "timestamp_(min)" in df.columns:
            df.drop(columns=["timestamp_(min)"], inplace=True)

    # Robust label extraction: prefer 'label' column, fall back to last column
    if "label" in label_df.columns:
        label_series = label_df["label"].values
    else:
        label_series = label_df.iloc[:, -1].values
    return train_df, test_df, label_series


def print_basic_stats(train_df, test_df, label_series):
    """Print dataset statistics to console."""
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS — PSM Dataset")
    print("=" * 60)
    print(f"\nTrain set : {train_df.shape[0]:,} timesteps × {train_df.shape[1]} features")
    print(f"Test set  : {test_df.shape[0]:,}  timesteps × {test_df.shape[1]} features")
    print(f"Anomaly labels : {label_series.shape[0]:,} entries")
    n_anom = int(label_series.sum())
    pct    = 100 * n_anom / len(label_series)
    print(f"Anomalies in test: {n_anom:,} / {len(label_series):,} ({pct:.1f}%)")
    print(f"\nNull values in train: {train_df.isnull().sum().sum()}")
    print(f"Null values in test : {test_df.isnull().sum().sum()}")
    print("\nTrain feature statistics (first 5 columns):")
    print(train_df.iloc[:, :5].describe().round(4).to_string())
    print("=" * 60 + "\n")


def plot_sample_timeseries(train_df, test_df, label_series):
    """Plot SAMPLE_N features as time-series with anomaly shading on test set."""
    features = train_df.columns[:SAMPLE_N].tolist()

    fig, axes = plt.subplots(SAMPLE_N, 2, figsize=(18, SAMPLE_N * 3),
                             gridspec_kw={"width_ratios": [1, 1]})
    fig.suptitle("PSM — Sample Feature Time-Series\n(Train | Test + Anomaly Regions)",
                 fontsize=14, fontweight="bold", y=1.01)

    anom_indices = np.where(label_series == 1)[0]
    # Find contiguous anomaly regions
    regions = []
    if len(anom_indices) > 0:
        start = anom_indices[0]
        prev  = anom_indices[0]
        for idx in anom_indices[1:]:
            if idx == prev + 1:
                prev = idx
            else:
                regions.append((start, prev))
                start = prev = idx
        regions.append((start, prev))

    colors = plt.cm.get_cmap("tab10").colors

    for i, feat in enumerate(features):
        # --- Train panel ---
        ax_tr = axes[i][0]
        ax_tr.plot(train_df[feat].values, color=colors[i], linewidth=0.8, alpha=0.9)
        ax_tr.set_title(f"[Train] {feat}", fontsize=9)
        ax_tr.set_xlabel("Timestep")
        ax_tr.set_ylabel("Value")

        # --- Test panel with anomaly shading ---
        ax_te = axes[i][1]
        ax_te.plot(test_df[feat].values, color=colors[i], linewidth=0.8, alpha=0.9)
        for (s, e) in regions:
            ax_te.axvspan(s, e, color="red", alpha=0.2)
        ax_te.set_title(f"[Test] {feat} (red = anomaly)", fontsize=9)
        ax_te.set_xlabel("Timestep")
        ax_te.set_ylabel("Value")

    plt.tight_layout()
    out_path = os.path.join(EDA_DIR, "sample_timeseries.png")
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_label_distribution(label_series):
    """Bar chart of normal vs anomaly labels in the test set."""
    counts   = {"Normal": int((label_series == 0).sum()),
                "Anomaly": int((label_series == 1).sum())}
    pcts     = {k: 100 * v / len(label_series) for k, v in counts.items()}

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(list(counts.keys()), list(counts.values()),
                  color=["#4caf97", "#e05252"], edgecolor="white", linewidth=1.5)
    for bar, (k, pct) in zip(bars, pcts.items()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 50,
                f"{pct:.1f}%",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title("Test Set — Label Distribution", fontsize=13, fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_xlabel("Class")
    ax.set_ylim(0, max(counts.values()) * 1.15)
    plt.tight_layout()
    out_path = os.path.join(EDA_DIR, "label_distribution.png")
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_correlation_heatmap(train_df):
    """Correlation heatmap of all features on the training set."""
    corr = train_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))   # upper triangle

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0,
                linewidths=0.3, linecolor="white",
                annot=False, square=True, ax=ax,
                cbar_kws={"shrink": 0.7, "label": "Pearson r"})
    ax.set_title("PSM Training Set — Feature Correlation Heatmap",
                 fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    out_path = os.path.join(EDA_DIR, "correlation_heatmap.png")
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_feature_histograms(train_df):
    """Distribution histograms for all features (train set)."""
    n_feat = train_df.shape[1]
    n_cols = 5
    n_rows = (n_feat + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 4, n_rows * 3))
    axes = axes.flatten()

    for i, col in enumerate(train_df.columns):
        axes[i].hist(train_df[col].dropna().values, bins=50,
                     color="#5b8dee", edgecolor="white", linewidth=0.3)
        axes[i].set_title(col, fontsize=8)
        axes[i].set_xlabel("Value", fontsize=7)
        axes[i].tick_params(labelsize=7)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("PSM Training Set — Feature Distributions",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    out_path = os.path.join(EDA_DIR, "feature_histograms.png")
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def run_eda():
    """Run the full EDA pipeline and save all plots."""
    _ensure_dir()
    print("\n=== Running EDA ===")
    train_df, test_df, label_series = load_data()
    print_basic_stats(train_df, test_df, label_series)
    print("  Generating plots …")
    plot_sample_timeseries(train_df, test_df, label_series)
    plot_label_distribution(label_series)
    plot_correlation_heatmap(train_df)
    plot_feature_histograms(train_df)
    print("=== EDA Complete ===\n")
    return train_df, test_df, label_series


if __name__ == "__main__":
    run_eda()
