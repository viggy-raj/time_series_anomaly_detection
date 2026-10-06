"""
config.py
=========
Central configuration for the Explainable Anomaly Detection project.
All hyperparameters and paths are defined here so every module stays in sync.
Edit this file to reproduce experiments with different settings.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data", "raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

PLOT_DIR    = os.path.join(OUTPUT_DIR, "plots")
EDA_DIR     = os.path.join(PLOT_DIR,  "eda")
METRIC_DIR  = os.path.join(OUTPUT_DIR, "metrics")
REPORT_DIR  = os.path.join(OUTPUT_DIR, "reports")
MODEL_PATH  = os.path.join(OUTPUT_DIR, "model.pth")

# ---------------------------------------------------------------------------
# Dataset — PSM (Pooled Server Metrics)
# ---------------------------------------------------------------------------
DATASET_NAME = "PSM"

# PSM dataset from the official eBay/RANSynCoders repository
# Source paper: "Practical Approach to Asynchronous Multivariate Time Series Anomaly Detection"
# These files are served via GitHub's blob download API (bypasses LFS size limit)
PSM_TRAIN_URL = "https://github.com/eBay/RANSynCoders/raw/main/data/train.csv"
PSM_TEST_URL  = "https://github.com/eBay/RANSynCoders/raw/main/data/test.csv"
PSM_LABEL_URL = "https://github.com/eBay/RANSynCoders/raw/main/data/test_label.csv"

PSM_TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
PSM_TEST_FILE  = os.path.join(DATA_DIR, "test.csv")
PSM_LABEL_FILE = os.path.join(DATA_DIR, "test_label.csv")

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
WINDOW_SIZE     = 60    # Sliding window length (timesteps)
STRIDE          = 1     # Sliding window stride
VAL_SPLIT       = 0.1   # Fraction of training windows reserved for validation

# ---------------------------------------------------------------------------
# Model — LSTM Autoencoder
# ---------------------------------------------------------------------------
HIDDEN_SIZE  = 64   # LSTM hidden dimension
LATENT_SIZE  = 32   # Bottleneck (inner LSTM) dimension
NUM_LAYERS   = 2    # Number of stacked LSTM layers in encoder/decoder

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE    = 64
EPOCHS        = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-5
PATIENCE      = 7     # Early stopping patience (epochs without val-loss improvement)
RANDOM_SEED   = 42

# ---------------------------------------------------------------------------
# Anomaly Scoring & Thresholding
# ---------------------------------------------------------------------------
# Threshold = percentile of reconstruction errors on the *training* set.
# A test window whose error exceeds this threshold is flagged as anomalous.
THRESHOLD_PERCENTILE = 99

# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------
NUM_BACKGROUND_SAMPLES = 100   # Background samples for SHAP GradientExplainer
NUM_ANOMALIES_TO_EXPLAIN = 5   # Number of detected anomaly windows to visualise

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
DEVICE = "cpu"   # Change to "cuda" if a GPU is available
