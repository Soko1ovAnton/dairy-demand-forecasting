"""
Определение нейросетевой модели и вспомогательные функции для DataLoader'ов.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class DairyDemandNet(nn.Module):
    """
    Простая полносвязная нейросеть для регрессии.
    Структура:
    input -> Linear(128) -> ReLU -> Linear(64) -> ReLU -> Linear(1)
    """

    def __init__(self, input_size: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def create_dataloaders(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    batch_size: int = 64,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Создаёт PyTorch DataLoader'ы для обучающей и тестовой выборок.
    """

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32).view(-1, 1)
    y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.float32).view(-1, 1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, test_loader