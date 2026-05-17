"""
L3 Behavioral Autoencoder.

Architecture: 52 → 32 → 16 → 32 → 52  (encoder-bottleneck-decoder)
Input: 45 behavioral features + 7 graph features merged on account_id = 52 dims.
Loss: MSE reconstruction.

The 7 graph features merged per sender account:
  pagerank, betweenness, fan_in_score, fan_out_score,
  community_density, benford_chi2_community, volume_asymmetry

Input/output dim is defined by INPUT_DIM = 52.
"""

from __future__ import annotations

import torch
import torch.nn as nn

INPUT_DIM = 52   # 45 behavioral + 7 graph
BOTTLENECK_DIM = 16


class BehavioralAutoencoder(nn.Module):
    """
    Reconstruction-based anomaly detector.
    High reconstruction error at inference -> anomalous account behaviour.
    """

    def __init__(self, input_dim: int = INPUT_DIM) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, BOTTLENECK_DIM),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(BOTTLENECK_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE reconstruction error. Shape: (batch,)"""
        x_hat = self.forward(x)
        return ((x - x_hat) ** 2).mean(dim=1)

    @torch.no_grad()
    def score(self, x: torch.Tensor) -> torch.Tensor:
        """Return reconstruction errors with no gradient tracking."""
        self.eval()
        return self.reconstruction_error(x)


def build_autoencoder(input_dim: int = INPUT_DIM) -> BehavioralAutoencoder:
    return BehavioralAutoencoder(input_dim=input_dim)
