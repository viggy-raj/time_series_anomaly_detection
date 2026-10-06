"""
main.py
=======
Full pipeline for:
  "Explainable Deep Learning for Anomaly Detection on Public Time-Series Data"

Steps executed in order:
  1. Download PSM dataset
  2. Exploratory Data Analysis (EDA)
  3. Preprocess → sliding windows → DataLoaders
  4. Build & train LSTM Autoencoder
  5. Evaluate (Precision, Recall, F1, AUROC, AP)
  6. Explain anomalies (SHAP + Saliency maps)
  7. Generate summary report

Usage:
  python main.py                    # full pipeline
  python main.py --skip-download    # skip download (if data already present)
  python main.py --skip-train       # skip training (load saved checkpoint)
  python main.py --epochs 30        # override epochs
  python main.py --device cuda      # use GPU

Outputs:
  outputs/
    model.pth
    plots/  (all figures)
    metrics/ (metrics.json, metrics.csv, shap_feature_importance.json)
    reports/ summary_report.txt
"""

import argparse
import os
import sys
import io
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# Force UTF-8 output on Windows to avoid cp1252 encoding errors
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


# ── Reproducibility ───────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── Argument parsing ──────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Explainable Anomaly Detection — Full Pipeline"
    )
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip dataset download (data already present)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training and load existing model checkpoint")
    parser.add_argument("--skip-eda", action="store_true",
                        help="Skip EDA plots")
    parser.add_argument("--skip-shap", action="store_true",
                        help="Skip SHAP (faster but no global explanations)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of training epochs")
    parser.add_argument("--device", type=str, default=None,
                        choices=["cpu", "cuda"],
                        help="Override compute device")
    return parser.parse_args()


# ── Report generation ─────────────────────────────────────────────────────────
def generate_report(
    metrics:         dict,
    feat_importance: dict,
    training_epochs: int,
    elapsed_secs:    float,
    args,
) -> str:
    """Create and save a human-readable summary report."""
    from config import REPORT_DIR, DATASET_NAME, WINDOW_SIZE, HIDDEN_SIZE, LATENT_SIZE

    os.makedirs(REPORT_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    top_features = list(feat_importance.items())[:5] if feat_importance else []

    lines = [
        "=" * 70,
        "SUMMARY REPORT",
        "Explainable Deep Learning for Anomaly Detection on Time-Series Data",
        "=" * 70,
        f"",
        f"Generated    : {timestamp}",
        f"Dataset      : {DATASET_NAME} (Pooled Server Metrics)",
        f"Model        : LSTM Autoencoder",
        f"",
        "--- Architecture ---",
        f"  Window size  : {WINDOW_SIZE} timesteps",
        f"  Hidden size  : {HIDDEN_SIZE}",
        f"  Latent size  : {LATENT_SIZE}",
        f"  Trained for  : {'loaded from checkpoint' if training_epochs == 0 else f'{training_epochs} epochs'}",
        f"  Train time   : {'N/A (checkpoint loaded)' if training_epochs == 0 else f'{elapsed_secs:.1f} seconds'}",
        f"",
        "--- Evaluation Metrics (Test Set) ---",
        f"  Threshold (99th pct) : {metrics.get('threshold', 'N/A')}",
        f"  Precision            : {metrics.get('precision', 'N/A')}",
        f"  Recall               : {metrics.get('recall', 'N/A')}",
        f"  F1 Score             : {metrics.get('f1_score', 'N/A')}",
        f"  AUROC                : {metrics.get('auroc', 'N/A')}",
        f"  Average Precision    : {metrics.get('average_precision', 'N/A')}",
        f"  True anomalies       : {metrics.get('n_anomalies_true', 'N/A')}",
        f"  Predicted anomalies  : {metrics.get('n_anomalies_pred', 'N/A')}",
        f"  Total test windows   : {metrics.get('n_windows_total', 'N/A')}",
        f"",
        "--- Top Features (Integrated Gradients attribution) ---",
    ]

    if top_features:
        for rank, (feat, score) in enumerate(top_features, 1):
            lines.append(f"  #{rank}: {feat:<30}  |IG| = {score:.6f}")
    else:
        lines.append("  Explainability was skipped (--skip-shap flag used).")

    lines += [
        f"",
        "--- Interpretation ---",
        f"  The LSTM Autoencoder was trained exclusively on normal (non-anomalous)",
        f"  server metric windows. It learns to reconstruct normal behaviour.",
        f"  At inference time, windows with high reconstruction error are flagged",
        f"  as anomalous. The threshold was set at the 99th percentile of training",
        f"  reconstruction errors, providing a principled, data-driven cutoff.",
        f"",
        f"  Integrated Gradients (IG) attributed anomaly scores back to input",
        f"  features for the top-5 highest-error anomaly windows. The top features",
        f"  listed above contributed most to the reconstruction error spikes.",
        f"",
        "--- Output Files ---",
        f"  outputs/model.pth                       -- Trained model checkpoint",
        f"  outputs/metrics/metrics.json            -- All evaluation metrics",
        f"  outputs/metrics/metrics.csv             -- Same in tabular form",
        f"  outputs/metrics/shap_feature_importance.json -- SHAP feature ranking",
        f"  outputs/plots/train_loss.png            -- Training curve",
        f"  outputs/plots/roc_curve.png             -- ROC curve",
        f"  outputs/plots/pr_curve.png              -- Precision-Recall curve",
        f"  outputs/plots/anomaly_overlay.png       -- Error vs ground truth",
        f"  outputs/plots/error_distribution.png   -- Error histogram",
        f"  outputs/plots/ig_summary.png            -- Global IG bar chart",
        f"  outputs/plots/ig_heatmap.png            -- IG attribution heatmap",
        f"  outputs/plots/shap_anomaly_*.png        -- Per-anomaly IG bars",
        f"  outputs/plots/saliency_*.png            -- Saliency heatmaps",
        f"  outputs/plots/eda/                      -- EDA figures",
        f"",
        "=" * 70,
    ]

    report_text = "\n".join(lines)
    report_path = os.path.join(REPORT_DIR, "summary_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\n  Saved: {report_path}")
    return report_text


# ── Main pipeline ─────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # Late imports so config is the single source of truth
    from config import (
        RANDOM_SEED, DEVICE, EPOCHS,
        HIDDEN_SIZE, LATENT_SIZE, NUM_LAYERS,
        WINDOW_SIZE, BATCH_SIZE, MODEL_PATH,
        THRESHOLD_PERCENTILE,
        NUM_ANOMALIES_TO_EXPLAIN,
    )

    device = args.device or DEVICE
    epochs = args.epochs or EPOCHS

    print("\n" + "=" * 70)
    print("  EXPLAINABLE ANOMALY DETECTION -- FULL PIPELINE")
    print("=" * 70)
    print(f"  Device : {device}")
    print(f"  Epochs : {epochs}")
    print(f"  Seed   : {RANDOM_SEED}")
    print("=" * 70 + "\n")

    set_seed(RANDOM_SEED)

    # ── Step 1: Download ──────────────────────────────────────────────────────
    if not args.skip_download:
        from data.download import download_psm
        download_psm()
    else:
        print("[skip] Dataset download skipped.\n")

    # ── Step 2: EDA ───────────────────────────────────────────────────────────
    if not args.skip_eda:
        from eda.eda import run_eda
        run_eda()
    else:
        print("[skip] EDA skipped.\n")

    # ── Step 3: Preprocess ────────────────────────────────────────────────────
    print("=== Preprocessing ===")
    from data.preprocess import get_dataloaders
    loaders, scaler, meta = get_dataloaders()
    n_features = meta["n_features"]
    print(f"=== Preprocessing Complete ({n_features} features) ===\n")

    # ── Step 4: Build model ───────────────────────────────────────────────────
    from models.lstm_autoencoder import LSTMAutoencoder
    model = LSTMAutoencoder(
        n_features  = n_features,
        hidden_size = HIDDEN_SIZE,
        latent_size = LATENT_SIZE,
        seq_len     = WINDOW_SIZE,
        num_layers  = NUM_LAYERS,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}\n")

    # ── Step 5: Train (or load) ───────────────────────────────────────────────
    training_epochs = 0
    t0 = time.time()

    if args.skip_train and os.path.exists(MODEL_PATH):
        print(f"[skip] Loading model from {MODEL_PATH}\n")
        model.load_state_dict(
            torch.load(MODEL_PATH, map_location=torch.device(device))
        )
        model.eval()
    else:
        from models.trainer import train_model
        model, history = train_model(
            model     = model,
            loaders   = loaders,
            device    = device,
            epochs    = epochs,
        )
        training_epochs = len(history["train_loss"])

    elapsed = time.time() - t0

    # ── Step 6: Evaluate ──────────────────────────────────────────────────────
    from evaluation.evaluate import run_evaluation
    metrics, test_errors, threshold = run_evaluation(
        model          = model,
        loaders        = loaders,
        device         = device,
        threshold_pct  = THRESHOLD_PERCENTILE,
    )

    # ── Step 7: Explain ───────────────────────────────────────────────────────
    feat_importance = {}

    # 7a: SHAP
    if not args.skip_shap:
        try:
            from explainability.shap_explainer import run_shap_explanation
            feat_importance = run_shap_explanation(
                model         = model,
                test_errors   = test_errors,
                test_windows  = meta["test_windows"],
                train_windows = meta["train_windows"],
                threshold     = threshold,
                feature_names = meta["feature_names"],
                device        = device,
                n_explain     = NUM_ANOMALIES_TO_EXPLAIN,
            )
        except Exception as exc:
            print(f"  [Warning] SHAP failed: {exc}")
            print("  Falling back to saliency maps only.\n")
    else:
        print("[skip] SHAP skipped.\n")

    # 7b: Saliency maps (always run)
    try:
        from explainability.saliency import run_saliency_explanation
        run_saliency_explanation(
            model         = model,
            test_errors   = test_errors,
            test_windows  = meta["test_windows"],
            threshold     = threshold,
            feature_names = meta["feature_names"],
            device        = device,
            n_explain     = NUM_ANOMALIES_TO_EXPLAIN,
        )
    except Exception as exc:
        print(f"  [Warning] Saliency failed: {exc}\n")

    # ── Step 8: Summary report ────────────────────────────────────────────────
    print("=== Generating Summary Report ===")
    report = generate_report(
        metrics         = metrics,
        feat_importance = feat_importance,
        training_epochs = training_epochs,
        elapsed_secs    = elapsed,
        args            = args,
    )

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(report)


if __name__ == "__main__":
    main()
