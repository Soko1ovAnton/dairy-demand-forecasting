#!/usr/bin/env python
"""
CLI-скрипт для обучения модели прогноза спроса на молочную продукцию.

Пример запуска:
    python -m scripts.train_model \
        --data-path data/dairy_data.csv \
        --epochs 50 \
        --batch-size 64 \
        --lr 1e-3 \
        --output-dir artifacts
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import torch
from joblib import dump

from dairy_demand.config import (
    EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    DEVICE,
)
from dairy_demand.data import load_and_preprocess
from dairy_demand.model import DairyDemandNet, create_dataloaders
from dairy_demand.train import train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train dairy demand forecasting model"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to input CSV file with dairy data",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Number of training epochs (default: {EPOCHS})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=LEARNING_RATE,
        help=f"Learning rate (default: {LEARNING_RATE})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts",
        help="Directory to save model and plots (default: artifacts)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Device to train on: 'cpu' or 'cuda' (default: auto)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = args.device or DEVICE
    print(f"Using device: {device}")

    # 1. Загрузка и препроцессинг данных
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess(
        args.data_path
    )

    # 2. Dataloader'ы
    print("Creating data loaders...")
    train_loader, test_loader = create_dataloaders(
        X_train,
        X_test,
        y_train,
        y_test,
        batch_size=args.batch_size,
    )

    # 3. Модель
    input_size = X_train.shape[1]
    model = DairyDemandNet(input_size)
