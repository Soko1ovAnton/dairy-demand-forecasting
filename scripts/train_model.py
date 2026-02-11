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

    # 4. Обучение
    print(f"Starting training for {args.epochs} epochs...")
    train_losses, val_r2_scores = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
    )

    best_r2 = max(val_r2_scores) if val_r2_scores else 0.0
    print(f"Training finished. Best R2 on validation: {best_r2:.4f}")

    # 5. Сохранение модели
    model_path = os.path.join(args.output_dir, "dairy_demand_model.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_size": input_size,
        },
        model_path,
    )
    print(f"Model saved to: {model_path}")

    # 6. Сохранение препроцессора
    preprocessor_path = os.path.join(args.output_dir, "preprocessor.joblib")
    dump(preprocessor, preprocessor_path)
    print(f"Preprocessor saved to: {preprocessor_path}")

    # 7. Сохранение метрик в JSON
    metrics = {
        "train_loss": train_losses,
        "val_r2": val_r2_scores,
        "best_val_r2": best_r2,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "device": device,
    }
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")

    # 8. Графики обучения
    print("Saving training curves plot...")
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_r2_scores, label="Validation R2", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("R2 Score")
    plt.title("Validation R2")
    plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, "training_curves.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Training curves saved to: {plot_path}")


if __name__ == "__main__":
    main()