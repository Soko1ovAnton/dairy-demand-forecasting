"""
Обучение и оценка модели.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import LEARNING_RATE, EPOCHS, DEVICE, SEED


def set_seed(seed: int = SEED) -> None:
    """
    Фиксация случайных сидов для воспроизводимости.
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = DEVICE,
) -> float:
    """
    Оценивает модель на данных из data_loader с помощью метрики R^2.
    """
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for xb, yb in data_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            y_pred = model(xb)

            preds.append(y_pred.cpu().numpy())
            targets.append(yb.cpu().numpy())

    y_true = np.vstack(targets)
    y_pred = np.vstack(preds)

    r2 = r2_score(y_true, y_pred)
    return float(r2)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
    device: str = DEVICE,
) -> Tuple[List[float], List[float]]:
    """
    Обучает модель и возвращает:
    - список значений loss на train по эпохам
    - список значений R^2 на val по эпохам
    """

    set_seed()

    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses: List[float] = []
    val_r2_scores: List[float] = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        # Оценка на валидации / тесте
        val_r2 = evaluate(model, val_loader, device=device)
        val_r2_scores.append(val_r2)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train_loss={epoch_loss:.4f} | val_R2={val_r2:.4f}"
        )

    return train_losses, val_r2_scores