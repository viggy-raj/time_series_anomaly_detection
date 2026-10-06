"""
models/trainer.py
=================
Training loop for the LSTM Autoencoder.

Features
--------
- MSE reconstruction loss (train on normal windows only)
- Validation loss tracked every epoch
- Early stopping (configurable patience)
- Best model checkpointing (saved to outputs/model.pth)
- Training / validation loss curve saved as PNG
- Returns trained model and full loss history
"""

import os
import sys
import io
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

# Force UTF-8 on Windows to handle any Unicode in print strings
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    PATIENCE, MODEL_PATH, PLOT_DIR, DEVICE,
)


# ── Loss ─────────────────────────────────────────────────────────────────────

def reconstruction_loss(x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    """Mean-squared reconstruction error across the full batch."""
    return nn.functional.mse_loss(recon, x, reduction="mean")


# ── Training loop ─────────────────────────────────────────────────────────────

def train_model(
    model:      nn.Module,
    loaders:    Dict,
    device:     str  = DEVICE,
    epochs:     int  = EPOCHS,
    lr:         float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    patience:   int  = PATIENCE,
    model_path: str  = MODEL_PATH,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Train the LSTM Autoencoder.

    Parameters
    ----------
    model      : Un-trained LSTMAutoencoder.
    loaders    : Dict with 'train' and 'val' DataLoaders.
    device     : 'cpu' or 'cuda'.
    epochs     : Maximum training epochs.
    lr         : Adam learning rate.
    weight_decay : L2 regularisation coefficient.
    patience   : Early-stopping patience (epochs without improvement).
    model_path : Where to save the best checkpoint.

    Returns
    -------
    model      : Best model loaded from checkpoint.
    history    : Dict with 'train_loss' and 'val_loss' lists.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    device = torch.device(device)
    model  = model.to(device)

    optimizer = optim.Adam(model.parameters(),
                           lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    epochs_no_improve = 0

    print(f"\n=== Training LSTM Autoencoder on {device} ===")
    print(f"    Epochs={epochs}, LR={lr}, Patience={patience}")
    print(f"    Train batches: {len(loaders['train'])} | "
          f"Val batches: {len(loaders['val'])}\n")

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # ── Train phase ──
        model.train()
        train_loss = 0.0
        pbar = tqdm(loaders["train"],
                    desc=f"Epoch {epoch:03d}/{epochs} [Train]",
                    leave=False, ncols=80)
        for batch in pbar:
            # DataLoader for train set returns (X,) — no labels
            X = batch[0].to(device)
            optimizer.zero_grad()
            recon, _ = model(X)
            loss = reconstruction_loss(X, recon)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * X.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        train_loss /= len(loaders["train"].dataset)

        # ── Validation phase ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in loaders["val"]:
                X = batch[0].to(device)
                recon, _ = model(X)
                loss = reconstruction_loss(X, recon)
                val_loss += loss.item() * X.size(0)
        val_loss /= len(loaders["val"].dataset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        scheduler.step(val_loss)

        elapsed = time.time() - start_time
        print(f"Epoch {epoch:03d}/{epochs} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"Time: {elapsed:.1f}s")

        # ── Checkpoint ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_path)
            print(f"  [BEST] New best val loss {best_val_loss:.6f} -- model saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n  Early stopping triggered after {epoch} epochs.")
                break

    # Load best weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"\n=== Training Complete. Best Val Loss: {best_val_loss:.6f} ===\n")

    _save_loss_curve(history)
    return model, history


# ── Loss curve ────────────────────────────────────────────────────────────────

def _save_loss_curve(history: Dict[str, List[float]]) -> None:
    """Save train/val loss curve to outputs/plots/train_loss.png."""
    os.makedirs(PLOT_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    epochs_range = range(1, len(history["train_loss"]) + 1)

    ax.plot(epochs_range, history["train_loss"],
            label="Train Loss", color="#5b8dee", linewidth=2)
    ax.plot(epochs_range, history["val_loss"],
            label="Val Loss",   color="#e05252", linewidth=2, linestyle="--")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MSE Reconstruction Loss", fontsize=12)
    ax.set_title("LSTM Autoencoder -- Training Curve", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.4)

    out_path = os.path.join(PLOT_DIR, "train_loss.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
