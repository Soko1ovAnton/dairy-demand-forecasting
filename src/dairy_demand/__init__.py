"""
dairy_demand
============

Пакет для прогнозирования объёма продаж молочной продукции.
Содержит утилиты для загрузки данных, препроцессинга и обучения модели.
"""

from .config import FEATURES, CATEGORICAL, TARGET_COLUMN
from .data import load_and_preprocess, add_date_features
from .model import DairyDemandNet
from .train import train_model

__all__ = [
    "FEATURES",
    "CATEGORICAL",
    "TARGET_COLUMN",
    "add_date_features",
    "load_and_preprocess",
    "DairyDemandNet",
    "train_model",
]