"""Importable output schemas for supervised learning Contract v0.2."""

from typing import Any, TypedDict

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
    """Stable top-level result returned by classification use cases."""

    model: Any
    feature_names: list[str]
    metrics: ClassificationMetrics
    predictions: DataFrame


class RegressionResult(TypedDict):
    """Stable top-level result returned by regression use cases."""

    model: Any
    feature_names: list[str]
    metrics: RegressionMetrics
    predictions: DataFrame
