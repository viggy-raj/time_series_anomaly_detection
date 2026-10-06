"""
models/lstm_autoencoder.py
==========================
LSTM Autoencoder for unsupervised time-series anomaly detection.

Architecture
------------
Encoder:
  LSTM(n_features → hidden_size, num_layers)
  └─ takes the last hidden state as the bottleneck representation

Decoder:
  Repeats the bottleneck vector `window_size` times
  LSTM(hidden_size → hidden_size, num_layers)
  Linear(hidden_size → n_features)

Anomaly Score
-------------
The reconstruction error (MSE) per window is used as the anomaly score.
A window with score above a learned threshold is flagged as anomalous.
"""

import torch
import torch.nn as nn
from typing import Tuple


class LSTMEncoder(nn.Module):
    """Encodes a window of shape (batch, seq_len, n_features) into a
    latent vector of shape (batch, hidden_size)."""

    def __init__(self, n_features: int, hidden_size: int,
                 latent_size: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers  = num_layers

        # First LSTM: input → hidden
        self.lstm1 = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Second LSTM: hidden → latent (bottleneck)
        self.lstm2 = nn.LSTM(
            input_size=hidden_size,
            hidden_size=latent_size,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (batch, seq_len, n_features)

        Returns
        -------
        latent : Tensor, shape (batch, latent_size)
        """
        out1, _ = self.lstm1(x)               # (batch, seq, hidden_size)
        out2, _ = self.lstm2(out1)             # (batch, seq, latent_size)
        latent   = out2[:, -1, :]             # take last timestep
        return latent                          # (batch, latent_size)


class LSTMDecoder(nn.Module):
    """Decodes a latent vector back to a window of shape
    (batch, seq_len, n_features)."""

    def __init__(self, n_features: int, hidden_size: int,
                 latent_size: int, seq_len: int,
                 num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.seq_len = seq_len

        # Expand latent → hidden
        self.lstm1 = nn.LSTM(
            input_size=latent_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        # Hidden → hidden (mirrors encoder depth)
        self.lstm2 = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Project back to feature space
        self.output_layer = nn.Linear(hidden_size, n_features)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        latent : Tensor, shape (batch, latent_size)

        Returns
        -------
        recon : Tensor, shape (batch, seq_len, n_features)
        """
        # Repeat latent vector across the time axis
        x    = latent.unsqueeze(1).repeat(1, self.seq_len, 1)  # (batch, seq, latent)
        out1, _ = self.lstm1(x)                                  # (batch, seq, hidden)
        out2, _ = self.lstm2(out1)                               # (batch, seq, hidden)
        recon   = self.output_layer(out2)                        # (batch, seq, n_feat)
        return recon


class LSTMAutoencoder(nn.Module):
    """
    Full LSTM Autoencoder: Encoder → bottleneck → Decoder.

    Parameters
    ----------
    n_features  : Number of input features (channels).
    hidden_size : LSTM hidden dimension.
    latent_size : Bottleneck dimension.
    seq_len     : Window / sequence length.
    num_layers  : Number of stacked LSTM layers.
    dropout     : Dropout rate (applied between LSTM layers).
    """

    def __init__(self,
                 n_features:  int,
                 hidden_size: int,
                 latent_size: int,
                 seq_len:     int,
                 num_layers:  int = 2,
                 dropout:     float = 0.1):
        super().__init__()
        self.n_features  = n_features
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.seq_len     = seq_len

        self.encoder = LSTMEncoder(n_features, hidden_size, latent_size,
                                   num_layers, dropout)
        self.decoder = LSTMDecoder(n_features, hidden_size, latent_size,
                                   seq_len, num_layers, dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : Tensor, shape (batch, seq_len, n_features)

        Returns
        -------
        recon  : Tensor, shape (batch, seq_len, n_features) — reconstruction
        latent : Tensor, shape (batch, latent_size)         — bottleneck
        """
        latent = self.encoder(x)
        recon  = self.decoder(latent)
        return recon, latent

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute per-window scalar reconstruction error (MSE).

        Parameters
        ----------
        x : Tensor, shape (batch, seq_len, n_features)

        Returns
        -------
        errors : Tensor, shape (batch,)
        """
        recon, _ = self.forward(x)
        # Mean over seq_len and n_features → scalar per window
        errors = ((x - recon) ** 2).mean(dim=(1, 2))
        return errors
