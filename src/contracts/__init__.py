"""Shared cross-role Contract v0.2 schemas."""

from .clustering import (
    CLUSTER_CENTER_COLUMNS,
    CLUSTERING_RESULT_KEYS,
    PROFILE_COLUMNS,
    PROFILE_FEATURES,
    ClusteringResult,
)
from .market_data import (
    BASE_MARKET_COLUMNS,
    COMMON_FEATURE_COLUMNS,
    MODEL_PRIVATE_FIELDS,
    MarketDataRow,
    MarketFeatureRow,
)
from .supervised import (
    CLASSIFICATION_PREDICTION_COLUMNS,
    REGRESSION_PREDICTION_COLUMNS,
    ClassificationMetrics,
    ClassificationResult,
    RegressionMetrics,
    RegressionResult,
)

__all__ = [
    "BASE_MARKET_COLUMNS",
    "CLASSIFICATION_PREDICTION_COLUMNS",
    "CLUSTER_CENTER_COLUMNS",
    "CLUSTERING_RESULT_KEYS",
    "COMMON_FEATURE_COLUMNS",
    "MODEL_PRIVATE_FIELDS",
    "PROFILE_COLUMNS",
    "PROFILE_FEATURES",
    "REGRESSION_PREDICTION_COLUMNS",
    "ClassificationMetrics",
    "ClassificationResult",
    "ClusteringResult",
    "MarketDataRow",
    "MarketFeatureRow",
    "RegressionMetrics",
    "RegressionResult",
]
