"""A tiny PyTorch recurrent classifier for sequence batches."""

from __future__ import annotations

import numpy as np


def train_rnn(sequences, labels, epochs: int = 15, hidden: int = 8, learning_rate: float = 0.01, random_state: int = 42):
    """Train on sequences shaped (n, time, features)."""
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise ImportError("Install PyTorch with: pip install torch") from exc

    torch.manual_seed(random_state)
    features = torch.tensor(np.asarray(sequences), dtype=torch.float32)
    target = torch.tensor(np.asarray(labels), dtype=torch.long)

    class SmallRNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.rnn = nn.RNN(features.shape[-1], hidden, batch_first=True)
            self.output = nn.Linear(hidden, int(target.max().item()) + 1)

        def forward(self, batch):
            output, _ = self.rnn(batch)
            return self.output(output[:, -1, :])

    model = SmallRNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(features), target)
        loss.backward()
        optimizer.step()
    return {"model": model, "loss": float(loss.detach())}
