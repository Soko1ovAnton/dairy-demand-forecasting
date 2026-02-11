"""
Загрузка и препроцессинг данных:
- чтение CSV
- генерация признаков по дате
- стандартизация и one-hot encoding
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from .config import (
    DATE_COLUMN,
    TARGET_COLUMN,
    FEATURES,
    CATEGORICAL,
    TEST_SIZE,
    RANDOM_STATE,
)


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет признаки "Month" и "Season" на основе столбца даты.
    Ожидается, что дата хранится в колонке DATE_COLUMN.
    """
    df = df.copy()
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    df["Month"] = df[DATE_COLUMN].dt.month

    def _season_from_month(m: int) -> str:
        if m in (12, 1, 2):
            return "Winter"
        if m in (3, 4, 5):
            return "Spring"
        if m in (6, 7, 8):
            return "Summer"
        return "Autumn"

    df["Season"] = df["Month"].apply(_season_from_month)
    return df


def get_preprocessor() -> ColumnTransformer:
    """
    Создаёт ColumnTransformer для числовых и категориальных признаков.
    Числовые признаки стандартизируются, категориальные — one-hot.
    """
    numerical = [f for f in FEATURES if f not in CATEGORICAL]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )

    return preprocessor


def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Загружает датасет из CSV и добавляет признаки по дате.
    """
    df = pd.read_csv(csv_path)
    df = add_date_features(df)
    return df


def prepare_features_and_target(df: pd.DataFrame):
    """
    Из датафрейма формирует матрицу признаков X и целевой вектор y.
    """
    X = df[FEATURES]
    y = df[TARGET_COLUMN]
    return X, y


def load_and_preprocess(
    csv_path: str,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, ColumnTransformer]:
    """
    Полный пайплайн:
    - загрузка CSV
    - добавление фич по дате
    - разбиение на train/test
    - обучение препроцессора на train и трансформация train/test

    Возвращает:
    X_train_preprocessed, X_test_preprocessed, y_train, y_test, preprocessor
    """
    df = load_dataset(csv_path)
    X, y = prepare_features_and_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    preprocessor = get_preprocessor()
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    return X_train_preprocessed, X_test_preprocessed, y_train, y_test, preprocessor