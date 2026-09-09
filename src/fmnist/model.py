"""Neural-network definitions for Fashion-MNIST."""
from __future__ import annotations

import torch
from torch import nn


class FashionMNISTANN(nn.Module):
    """Regularized fully connected classifier for 28x28 grayscale images."""

    def __init__(self, dropout: float = 0.2) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
