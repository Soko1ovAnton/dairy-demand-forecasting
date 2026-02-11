"""
Конфигурация проекта: имена колонок, гиперпараметры и пр.
"""

from __future__ import annotations

import torch

# --- Колонки данных ---

# Название столбца с датой
DATE_COLUMN = "Date"

# Целевая переменная
TARGET_COLUMN = "Quantity Sold (liters/kg)"

# Признаки (features), используемые в модели
FEATURES = [
    "Total Land Area (acres)",
    "Number of Cows",
    "Farm Size",
    "Month",
    "Season",
    "Price per Unit",
    "Total Value",
    "Shelf Life (days)",
    "Price per Unit (sold)",
    "Approx. Total Revenue(INR)",
    "Quantity in Stock (liters/kg)",
    "Minimum Stock Threshold (liters/kg)",
    "Reorder Quantity (liters/kg)",
    "Location",
    "Brand",
    "Sales Channel",
]

# Категориальные признаки
CATEGORICAL = [
    "Farm Size",
    "Season",
    "Location",
    "Brand",
    "Sales Channel",
]

# --- Параметры разбиения данных ---

TEST_SIZE = 0.2
RANDOM_STATE = 42

# --- Гиперпараметры обучения ---

EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

# --- Устройство для обучения (CPU / GPU) ---

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Seed для воспроизводимости ---

SEED = 42