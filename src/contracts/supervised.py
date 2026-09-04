"""Importable output schemas for supervised learning Contract v0.2."""

from __future__ import annotations

from typing import Any, TypedDict

from typing_extensions import NotRequired

from pandas import DataFrame


CLASSIFICATION_PREDICTION_COLUMNS = ("trade_date", "y_true", "y_pred")
REGRESSION_PREDICTION_COLUMNS = ("trade_date", "y_true", "y_pred")


class ClassificationMetrics(TypedDict):
    """Required P0 classification metrics."""

    accuracy: float
    confusion_matrix: list[list[int]]


class RegressionMetrics(TypedDict):
    """Required P0 regression metrics."""

    mae: float
    r2: float


class ClassificationResult(TypedDict):
    """Stable top-level result returned by classification use cases.

    Core fields (always present):
        model, feature_names, metrics, predictions

    Extended fields (added for dynamic single-stock input, optional):
        n_raw_trading_days, n_effective_samples,
        n_train_samples, n_test_samples, feature_importance
    """

    model: Any
    feature_names: list[str]
    metrics: ClassificationMetrics
    predictions: DataFrame
    # --- sample metadata (added for dynamic single-stock input) ---
    n_raw_trading_days: NotRequired[int]
    n_effective_samples: NotRequired[int]
    n_train_samples: NotRequired[int]
    n_test_samples: NotRequired[int]
    feature_importance: NotRequired[list[float]]


class RegressionResult(TypedDict):
    """Stable top-level result returned by regression use cases."""

    model: Any
    feature_names: list[str]
    metrics: RegressionMetrics
    predictions: DataFrame
