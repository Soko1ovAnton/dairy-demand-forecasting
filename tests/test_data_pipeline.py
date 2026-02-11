"""
Простые тесты для проверки пайплайна данных и модели.

Запуск:
    pytest tests/test_data_pipeline.py -v
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

# Чтобы pytest видел пакет dairy_demand
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from dairy_demand.config import (
    FEATURES,
    CATEGORICAL,
    TARGET_COLUMN,
    DATE_COLUMN,
)
from dairy_demand.data import (
    add_date_features,
    get_preprocessor,
    prepare_features_and_target,
)
from dairy_demand.model import DairyDemandNet, create_dataloaders


# ──────────────────────────────────────────────
# Фикстура: маленький фейковый датасет
# ──────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """
    Создаёт маленький датафрейм, имитирующий реальные данные.
    Используется вместо настоящего CSV, чтобы тесты не зависели от файла.
    """
    np.random.seed(42)
    n = 100

    df = pd.DataFrame({
        DATE_COLUMN: pd.date_range("2023-01-01", periods=n, freq="D"),
        "Total Land Area (acres)": np.random.uniform(1, 100, n),
        "Number of Cows": np.random.randint(5, 200, n),
        "Farm Size": np.random.choice(["Small", "Medium", "Large"], n),
        "Price per Unit": np.random.uniform(10, 100, n),
        "Total Value": np.random.uniform(100, 10000, n),
        "Shelf Life (days)": np.random.randint(1, 30, n),
        "Price per Unit (sold)": np.random.uniform(10, 100, n),
        "Approx. Total Revenue(INR)": np.random.uniform(1000, 50000, n),
        "Quantity in Stock (liters/kg)": np.random.uniform(10, 500, n),
        "Minimum Stock Threshold (liters/kg)": np.random.uniform(5, 50, n),
        "Reorder Quantity (liters/kg)": np.random.uniform(10, 100, n),
        "Location": np.random.choice(["Delhi", "Mumbai", "Bangalore"], n),
        "Brand": np.random.choice(["Amul", "Mother Dairy", "Nestle"], n),
        "Sales Channel": np.random.choice(["Retail", "Wholesale", "Online"], n),
        TARGET_COLUMN: np.random.uniform(50, 500, n),
    })

    return df


# ──────────────────────────────────────────────
# Тесты для add_date_features
# ──────────────────────────────────────────────

class TestAddDateFeatures:
    """Тесты генерации признаков из даты."""

    def test_month_column_created(self, sample_df):
        """После add_date_features должна появиться колонка Month."""
        df = add_date_features(sample_df)
        assert "Month" in df.columns

    def test_season_column_created(self, sample_df):
        """После add_date_features должна появиться колонка Season."""
        df = add_date_features(sample_df)
        assert "Season" in df.columns

    def test_month_values_range(self, sample_df):
        """Месяц должен быть от 1 до 12."""
        df = add_date_features(sample_df)
        assert df["Month"].min() >= 1
        assert df["Month"].max() <= 12

    def test_season_values(self, sample_df):
        """Сезон должен быть одним из четырёх."""
        df = add_date_features(sample_df)
        valid_seasons = {"Winter", "Spring", "Summer", "Autumn"}
        assert set(df["Season"].unique()).issubset(valid_seasons)

    def test_original_df_not_modified(self, sample_df):
        """Исходный датафрейм не должен меняться."""
        original_cols = set(sample_df.columns)
        _ = add_date_features(sample_df)
        assert set(sample_df.columns) == original_cols


# ──────────────────────────────────────────────
# Тесты для prepare_features_and_target
# ──────────────────────────────────────────────

class TestPrepareData:
    """Тесты подготовки признаков и целевой переменной."""

    def test_output_shapes(self, sample_df):
        """X и y должны иметь правильные размеры."""
        df = add_date_features(sample_df)
        X, y = prepare_features_and_target(df)

        assert X.shape[0] == len(df)
        assert X.shape[1] == len(FEATURES)
        assert len(y) == len(df)

    def test_target_no_nan(self, sample_df):
        """В целевой переменной не должно быть NaN."""
        df = add_date_features(sample_df)
        _, y = prepare_features_and_target(df)

        assert y.isna().sum() == 0

    def test_features_columns(self, sample_df):
        """X должен содержать ровно те колонки, что в FEATURES."""
        df = add_date_features(sample_df)
        X, _ = prepare_features_and_target(df)

        assert list(X.columns) == FEATURES


# ──────────────────────────────────────────────
# Тесты для preprocessor
# ──────────────────────────────────────────────

class TestPreprocessor:
    """Тесты ColumnTransformer (стандартизация + one-hot)."""

    def test_output_is_numeric(self, sample_df):
        """После препроцессинга все значения должны быть числовыми."""
        df = add_date_features(sample_df)
        X, _ = prepare_features_and_target(df)

        preprocessor = get_preprocessor()
        X_transformed = preprocessor.fit_transform(X)

        assert np.issubdtype(X_transformed.dtype, np.floating)

    def test_no_nan_after_transform(self, sample_df):
        """После препроцессинга не должно быть NaN."""
        df = add_date_features(sample_df)
        X, _ = prepare_features_and_target(df)

        preprocessor = get_preprocessor()
        X_transformed = preprocessor.fit_transform(X)

        assert not np.isnan(X_transformed).any()

    def test_output_rows_count(self, sample_df):
        """Количество строк не должно меняться после препроцессинга."""
        df = add_date_features(sample_df)
        X, _ = prepare_features_and_target(df)

        preprocessor = get_preprocessor()
        X_transformed = preprocessor.fit_transform(X)

        assert X_transformed.shape[0] == X.shape[0]

    def test_output_columns_increased(self, sample_df):
        """
        После one-hot encoding колонок должно стать больше,
        чем исходных FEATURES.
        """
        df = add_date_features(sample_df)
        X, _ = prepare_features_and_target(df)

        preprocessor = get_preprocessor()
        X_transformed = preprocessor.fit_transform(X)

        assert X_transformed.shape[1] > len(FEATURES) - len(CATEGORICAL)


# ──────────────────────────────────────────────
# Тесты для модели DairyDemandNet
# ──────────────────────────────────────────────

class TestModel:
    """Тесты архитектуры нейросети."""

    def test_output_shape(self):
        """Выход модели должен иметь форму (batch_size, 1)."""
        input_size = 20
        batch_size = 8

        model = DairyDemandNet(input_size)
        x = torch.randn(batch_size, input_size)
        out = model(x)

        assert out.shape == (batch_size, 1)

    def test_output_is_finite(self):
        """Выход модели не должен содержать NaN или Inf."""
        input_size = 15
        model = DairyDemandNet(input_size)
        x = torch.randn(4, input_size)
        out = model(x)

        assert torch.isfinite(out).all()

    def test_single_sample(self):
        """Модель должна работать даже с одним примером."""
        input_size = 10
        model = DairyDemandNet(input_size)
        x = torch.randn(1, input_size)
        out = model(x)

        assert out.shape == (1, 1)

    def test_model_has_parameters(self):
        """У модели должны быть обучаемые параметры."""
        model = DairyDemandNet(input_size=10)
        params = list(model.parameters())

        assert len(params) > 0
        assert all(p.requires_grad for p in params)


# ──────────────────────────────────────────────
# Тесты для create_dataloaders
# ──────────────────────────────────────────────

class TestDataLoaders:
    """Тесты создания DataLoader'ов."""

    def test_loaders_not_empty(self, sample_df):
        """DataLoader'ы не должны быть пустыми."""
        df = add_date_features(sample_df)
        X, y = prepare_features_and_target(df)

        preprocessor = get_preprocessor()
        X_proc = preprocessor.fit_transform(X)

        n = X_proc.shape[0]
        split = int(n * 0.8)

        train_loader, test_loader = create_dataloaders(
            X_train=X_proc[:split],
            X_test=X_proc[split:],
            y_train=y.iloc[:split],
            y_test=y.iloc[split:],
            batch_size=16,
        )

        assert len(train_loader) > 0
        assert len(test_loader) > 0

    def test_batch_shapes(self, sample_df):
        """Батч из DataLoader должен содержать тензоры правильной формы."""
        df = add_date_features(sample_df)
        X, y = prepare_features_and_target(df)

        preprocessor = get_preprocessor()
        X_proc = preprocessor.fit_transform(X)

        n = X_proc.shape[0]
        split = int(n * 0.8)

        train_loader, _ = create_dataloaders(
            X_train=X_proc[:split],
            X_test=X_proc[split:],
            y_train=y.iloc[:split],
            y_test=y.iloc[split:],
            batch_size=16,
        )

        xb, yb = next(iter(train_loader))

        assert xb.ndim == 2  # (batch, features)
        assert yb.ndim == 2  # (batch, 1)
        assert xb.shape[0] == yb.shape[0]
        assert yb.shape[1] == 1

    def test_tensors_are_float(self, sample_df):
        """Тензоры в батче должны быть float32."""
        df = add_date_features(sample_df)
        X, y = prepare_features_and_target(df)

        preprocessor = get_preprocessor()
        X_proc = preprocessor.fit_transform(X)

        n = X_proc.shape[0]
        split = int(n * 0.8)

        train_loader, _ = create_dataloaders(
            X_train=X_proc[:split],
            X_test=X_proc[split:],
            y_train=y.iloc[:split],
            y_test=y.iloc[split:],
            batch_size=16,
        )

        xb, yb = next(iter(train_loader))

        assert xb.dtype == torch.float32
        assert yb.dtype == torch.float32