# Explainable Deep Learning for Anomaly Detection on Public Time-Series Data

> **Research Project** · Python · PyTorch · SHAP · PSM Dataset

---

## Overview

This project implements a complete, reproducible pipeline for **unsupervised time-series anomaly detection** using an LSTM Autoencoder, with explainability via SHAP attributions and gradient saliency maps.

The project is structured as supporting material for a college-level research paper and generates all paper-ready figures, metrics, and a human-readable summary report automatically.

**Dataset**: PSM (Pooled Server Metrics) — 25-dimensional server KPI time-series, publicly hosted on GitHub.  
**Model**: LSTM Autoencoder trained only on normal windows (unsupervised setting).  
**Explainability**: Integrated Gradients (Sundararajan et al. 2017) — global feature importance + per-anomaly attribution + temporal heatmaps.

---

## Folder Structure

```
time_series/
├── README.md
├── requirements.txt          ← pip install
├── environment.yml           ← conda install
├── config.py                 ← ALL hyperparameters & paths
├── main.py                   ← Single entry point for the full pipeline
│
├── data/
│   ├── download.py           ← Auto-downloads PSM from GitHub
│   └── preprocess.py         ← Normalisation, windowing, DataLoaders
│
├── eda/
│   └── eda.py                ← Statistics + 4 analysis plots
│
├── models/
│   ├── lstm_autoencoder.py   ← LSTM AE architecture (PyTorch)
│   └── trainer.py            ← Training loop, early stopping, checkpointing
│
├── evaluation/
│   └── evaluate.py           ← Metrics + 4 evaluation plots
│
├── explainability/
│   ├── shap_explainer.py     ← SHAP GradientExplainer
│   └── saliency.py           ← Gradient saliency heatmaps
│
└── outputs/                  ← Auto-created at runtime
    ├── model.pth
    ├── plots/
    │   ├── eda/
    │   ├── train_loss.png
    │   ├── roc_curve.png
    │   ├── pr_curve.png
    │   ├── anomaly_overlay.png
    │   ├── error_distribution.png
    │   ├── shap_summary.png
    │   ├── shap_anomaly_01.png … shap_anomaly_05.png
    │   └── saliency_01.png … saliency_05.png
    ├── metrics/
    │   ├── metrics.json
    │   ├── metrics.csv
    │   └── shap_feature_importance.json
    └── reports/
        └── summary_report.txt
```

---

## Quick Start

### Option A — pip (local Python)

```bash
# 1. Clone / navigate to the project folder
cd time_series

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full pipeline
python main.py
```

### Option B — Conda

```bash
conda env create -f environment.yml
conda activate anomaly-xai
python main.py
```

### Option C — Google Colab

Open a new Colab notebook and run:

```python
# Mount Drive or clone the repo
!git clone <your-repo-url>
%cd time_series

# Install
!pip install -r requirements.txt

# Run
!python main.py
```

> **GPU**: Colab provides free GPU. Change `DEVICE = "cuda"` in `config.py` or pass `--device cuda` for ~5× speedup.

---

## Command-Line Options

```
python main.py [options]

  --skip-download   Skip dataset download (data already in data/raw/)
  --skip-train      Skip training, load existing model.pth checkpoint
  --skip-eda        Skip EDA plots
  --skip-shap       Skip SHAP (saliency maps still run; useful if SHAP is slow)
  --epochs N        Override number of training epochs (default: 50)
  --device {cpu,cuda}  Override compute device (default: cpu)
```

### Examples

```bash
# First run — full pipeline
python main.py

# Subsequent runs — skip re-download and re-train
python main.py --skip-download --skip-train

# Quick smoke test (few epochs, no SHAP)
python main.py --epochs 5 --skip-shap

# GPU training
python main.py --device cuda
```

---

## Dataset: PSM (Pooled Server Metrics)

| Property       | Value                          |
|----------------|--------------------------------|
| Source         | eBay/RANSynCoders (GitHub) — auto-downloaded |
| Type           | Multivariate time-series       |
| Features       | 25 server KPIs (CPU, memory, I/O, …) |
| Train samples  | ~132,000 timesteps (normal)   |
| Test samples   | ~87,000 timesteps (mixed)     |
| Anomaly rate   | ~27% in test set               |
| Download       | **Automatic** via `data/download.py` |

If automatic download fails (no internet), manually place these files in `data/raw/`:
- `train.csv` — [URL in config.py](config.py)
- `test.csv`
- `test_label.csv`

---

## Model Architecture

```
Input  (batch, 60, 25)
  │
  ▼ LSTMEncoder
  LSTM(25→64, 2 layers)
  LSTM(64→32, 1 layer)   ← bottleneck / latent vector
  │
  ▼ LSTMDecoder
  repeat latent × 60 timesteps
  LSTM(32→64, 1 layer)
  LSTM(64→64, 2 layers)
  Linear(64→25)
  │
  ▼
Output (batch, 60, 25)

Anomaly Score = MSE(input, reconstruction)
Threshold     = 99th percentile of training scores
```

All hyperparameters are in [`config.py`](config.py).

---

## Outputs for the Paper

| File | Use in paper |
|------|-------------|
| `outputs/reports/summary_report.txt` | Key results section |
| `outputs/metrics/metrics.json` | Table: precision / recall / F1 / AUROC |
| `outputs/plots/roc_curve.png` | Figure: ROC curve |
| `outputs/plots/pr_curve.png` | Figure: PR curve |
| `outputs/plots/anomaly_overlay.png` | Figure: qualitative detection result |
| `outputs/plots/shap_summary.png` | Figure: global feature importance |
| `outputs/plots/shap_anomaly_*.png` | Figure: case-study explanations |
| `outputs/plots/saliency_*.png` | Figure: temporal attribution heatmaps |
| `outputs/plots/eda/*.png` | Appendix: dataset analysis |

---

## Reproducing Results

All random seeds are fixed in `config.py` (`RANDOM_SEED = 42`).  
The full pipeline is deterministic on CPU. GPU training may introduce minor floating-point non-determinism.

To reproduce exactly:
```bash
python main.py --device cpu
```

---

## Customisation

Edit [`config.py`](config.py) to change:

| Variable | Default | Description |
|----------|---------|-------------|
| `WINDOW_SIZE` | 60 | Sliding window length |
| `HIDDEN_SIZE` | 64 | LSTM hidden units |
| `LATENT_SIZE` | 32 | Bottleneck dimension |
| `EPOCHS` | 50 | Max training epochs |
| `THRESHOLD_PERCENTILE` | 99 | Anomaly threshold percentile |
| `NUM_ANOMALIES_TO_EXPLAIN` | 5 | Anomaly case studies for SHAP/saliency |
| `DEVICE` | `"cpu"` | `"cuda"` for GPU |

---

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- SHAP ≥ 0.44
- scikit-learn ≥ 1.3
- matplotlib, seaborn, pandas, numpy, tqdm

See [`requirements.txt`](requirements.txt) for pinned versions.

---

## References

1. Hundman et al. (2018). *Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding.* KDD.
2. Su et al. (2019). *Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network.* KDD (OmniAnomaly).
3. Lundberg & Lee (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS (SHAP).
4. Doshi-Velez & Kim (2017). *Towards a Rigorous Science of Interpretable Machine Learning.* arXiv.

---

*Project generated for academic research purposes. All data is publicly available.*
