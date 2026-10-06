"""
data/preprocess.py
==================
Preprocessing pipeline for the PSM dataset:

  1. Load raw CSVs
  2. Drop any non-feature columns (e.g. timestamp)
  3. Handle NaN values (forward-fill, then zero-fill)
  4. MinMax-normalise (fit on train set only — no data leakage)
  5. Create sliding-window samples (window_size × n_features)
  6. Split training windows into train / validation
  7. Return PyTorch DataLoaders

Usage:
    from data.preprocess import get_dataloaders
    loaders, scaler, meta = get_dataloaders()
"""

import sys
import os
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    PSM_TRAIN_FILE, PSM_TEST_FILE, PSM_LABEL_FILE,
    WINDOW_SIZE, STRIDE, VAL_SPLIT,
    BATCH_SIZE, RANDOM_SEED,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_csv(path: str) -> pd.DataFrame:
    """Load a CSV and drop the timestamp column if present."""
    df = pd.read_csv(path)
    ts_cols = [c for c in df.columns if "timestamp" in c.lower()]
    df.drop(columns=ts_cols, inplace=True, errors="ignore")
    return df


def _clean(df: pd.DataFrame) -> np.ndarray:
    """Forward-fill then zero-fill NaN, return float32 numpy array."""
    df = df.ffill().fillna(0.0)
    return df.values.astype(np.float32)


def _sliding_windows(data: np.ndarray,
                     window: int,
                     stride: int) -> np.ndarray:
    """
    Create non-overlapping (stride>1) or overlapping (stride=1) windows.

    Returns
    -------
    np.ndarray of shape (N_windows, window, n_features)
    """
    n_steps, n_feat = data.shape
    # Number of complete windows
    n_windows = (n_steps - window) // stride + 1
    windows = np.stack(
        [data[i * stride: i * stride + window]
         for i in range(n_windows)],
        axis=0
    )
    return windows   # shape: (N, window, features)


# ── Public API ────────────────────────────────────────────────────────────────

def get_dataloaders(
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
    val_split: float = VAL_SPLIT,
    batch_size: int = BATCH_SIZE,
    seed: int = RANDOM_SEED,
) -> Tuple[Dict[str, DataLoader], MinMaxScaler, Dict[str, Any]]:
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    window_size : int
        Length of each sliding window in timesteps.
    stride : int
        Step size between consecutive windows.
    val_split : float
        Fraction of training windows held out for validation.
    batch_size : int
        Mini-batch size for all DataLoaders.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    loaders : dict with keys 'train', 'val', 'test'
    scaler  : fitted MinMaxScaler (needed to invert transforms later)
    meta    : dict with useful metadata (feature names, label array, shapes)
    """
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # 1. Load raw data
    # ------------------------------------------------------------------
    print("  Loading raw CSV files …")
    train_df = _load_csv(PSM_TRAIN_FILE)
    test_df  = _load_csv(PSM_TEST_FILE)
    label_df = pd.read_csv(PSM_LABEL_FILE)

    feature_names = train_df.columns.tolist()
    n_features    = len(feature_names)

    train_arr  = _clean(train_df)            # (N_train, F)
    test_arr   = _clean(test_df)             # (N_test,  F)

    # Robust label extraction: prefer 'label' column, fall back to last column
    if "label" in label_df.columns:
        test_labels = label_df["label"].values.astype(np.int32)
    else:
        # Some versions only have a single unlabelled column
        test_labels = label_df.iloc[:, -1].values.astype(np.int32)

    # ------------------------------------------------------------------
    # 2. Normalise — fit ONLY on train
    # ------------------------------------------------------------------
    print("  Normalising (MinMax, fit on train) …")
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_arr)   # (N_train, F)
    test_scaled  = scaler.transform(test_arr)        # (N_test,  F)

    # ------------------------------------------------------------------
    # 3. Sliding windows
    # ------------------------------------------------------------------
    print(f"  Building sliding windows (size={window_size}, stride={stride}) …")
    train_windows = _sliding_windows(train_scaled, window_size, stride)
    test_windows  = _sliding_windows(test_scaled,  window_size, stride)

    # Align test labels: label for each window = label at the LAST timestep
    last_timestep_idx = np.arange(window_size - 1,
                                   window_size - 1 + len(test_windows) * stride,
                                   stride)
    # Clip to valid range
    last_timestep_idx = np.clip(last_timestep_idx, 0, len(test_labels) - 1)
    window_labels = test_labels[last_timestep_idx]   # (N_test_windows,)

    # ------------------------------------------------------------------
    # 4. Train / Validation split
    # ------------------------------------------------------------------
    n_train = len(train_windows)
    n_val   = max(1, int(n_train * val_split))
    indices = rng.permutation(n_train)
    val_idx   = indices[:n_val]
    train_idx = indices[n_val:]

    X_train = torch.from_numpy(train_windows[train_idx])   # (N_tr, W, F)
    X_val   = torch.from_numpy(train_windows[val_idx])     # (N_val, W, F)
    X_test  = torch.from_numpy(test_windows)               # (N_te, W, F)
    y_test  = torch.from_numpy(window_labels)              # (N_te,)

    print(f"  Train windows : {X_train.shape}")
    print(f"  Val   windows : {X_val.shape}")
    print(f"  Test  windows : {X_test.shape}")
    print(f"  Test labels   : anomalies={int(y_test.sum())} / {len(y_test)} "
          f"({100*float(y_test.float().mean()):.1f}%)")

    # ------------------------------------------------------------------
    # 5. DataLoaders
    # ------------------------------------------------------------------
    def make_loader(X, y=None, shuffle=False):
        if y is not None:
            ds = TensorDataset(X, y)
        else:
            ds = TensorDataset(X)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=0, pin_memory=False)

    loaders = {
        "train": make_loader(X_train, shuffle=True),
        "val":   make_loader(X_val,   shuffle=False),
        "test":  make_loader(X_test, y_test, shuffle=False),
    }

    meta = {
        "feature_names":  feature_names,
        "n_features":     n_features,
        "window_size":    window_size,
        "train_windows":  train_windows,     # raw numpy (for SHAP background)
        "test_windows":   test_windows,      # raw numpy
        "window_labels":  window_labels,     # numpy int32
        "train_arr_raw":  train_arr,         # un-normalised
        "test_arr_raw":   test_arr,
        "test_labels_ts": test_labels,       # per-timestep labels
    }

    return loaders, scaler, meta


if __name__ == "__main__":
    loaders, scaler, meta = get_dataloaders()
    print("Preprocessing complete.")
    print(f"Features: {meta['feature_names'][:5]} … ({meta['n_features']} total)")
