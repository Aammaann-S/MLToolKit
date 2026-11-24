"""A tiny PyTorch CNN for a synthetic single-channel image batch."""

from __future__ import annotations

import numpy as np


def train_cnn(images, labels, epochs: int = 5, learning_rate: float = 0.01, random_state: int = 42):
    """Train on images shaped (n, 1, height, width)."""
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise ImportError("Install PyTorch with: pip install torch") from exc

    torch.manual_seed(random_state)
    features = torch.tensor(np.asarray(images), dtype=torch.float32)
    target = torch.tensor(np.asarray(labels), dtype=torch.long)

    class SmallCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 4, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(4, int(target.max().item()) + 1),
            )

        def forward(self, batch):
            return self.net(batch)

    model = SmallCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(features), target)
        loss.backward()
        optimizer.step()
    return {"model": model, "loss": float(loss.detach())}
