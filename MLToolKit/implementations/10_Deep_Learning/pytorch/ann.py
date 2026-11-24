"""A small PyTorch feed-forward classifier with dropout.

PyTorch is optional. Import happens inside the training function.
"""

from __future__ import annotations

import numpy as np


def train_ann(X, y, epochs: int = 30, hidden: int = 16, dropout: float = 0.2, learning_rate: float = 0.01, random_state: int = 42):
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise ImportError("Install PyTorch with: pip install torch") from exc

    torch.manual_seed(random_state)
    features = torch.tensor(np.asarray(X), dtype=torch.float32)
    target = torch.tensor(np.asarray(y), dtype=torch.long)
    model = nn.Sequential(
        nn.Linear(features.shape[1], hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, int(target.max().item()) + 1),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(features), target)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        predictions = model(features).argmax(dim=1).numpy()
    return {"model": model, "predictions": predictions, "loss": float(loss.detach())}
