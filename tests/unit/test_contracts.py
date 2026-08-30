"""Contract shape tests that do not implement domain algorithms."""

from src.contracts.clustering import (
    CLUSTER_CENTER_COLUMNS,
    CLUSTERING_RESULT_KEYS,
    PROFILE_COLUMNS,
    PROFILE_FEATURES,
    ClusteringResult,
)
from src.contracts.market_data import (
    BASE_MARKET_COLUMNS,
    COMMON_FEATURE_COLUMNS,
    MODEL_PRIVATE_FIELDS,
    MarketDataRow,
    MarketFeatureRow,
)
from src.contracts.supervised import (
    CLASSIFICATION_PREDICTION_COLUMNS,
    REGRESSION_PREDICTION_COLUMNS,
    ClassificationMetrics,
    ClassificationResult,
    RegressionMetrics,
    RegressionResult,
)


def test_market_data_fields_are_stable_and_private_fields_are_separate() -> None:
    assert BASE_MARKET_COLUMNS == (
        "symbol",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
    )
    assert COMMON_FEATURE_COLUMNS == (
        "return",
        "cumulative_return",
        "ma5",
        "ma20",
        "volatility_20d",
        "volume_change",
        "drawdown",
    )
    assert MODEL_PRIVATE_FIELDS.isdisjoint(
        set(BASE_MARKET_COLUMNS) | set(COMMON_FEATURE_COLUMNS)
    )
    assert MODEL_PRIVATE_FIELDS == frozenset(
        {"label", "target", "next_return", "prediction", "cluster"}
    )
    assert MarketDataRow.__required_keys__ == frozenset(BASE_MARKET_COLUMNS)
    assert MarketFeatureRow.__required_keys__ == frozenset(
        BASE_MARKET_COLUMNS + COMMON_FEATURE_COLUMNS
    )


def test_supervised_result_keys_and_metrics_are_stable() -> None:
    expected_result_keys = frozenset(
        {"model", "feature_names", "metrics", "predictions"}
    )

    assert ClassificationResult.__required_keys__ == expected_result_keys
    assert RegressionResult.__required_keys__ == expected_result_keys
    assert ClassificationMetrics.__required_keys__ == frozenset(
        {"accuracy", "confusion_matrix"}
    )
    assert RegressionMetrics.__required_keys__ == frozenset({"mae", "r2"})
    assert CLASSIFICATION_PREDICTION_COLUMNS == (
        "trade_date",
        "y_true",
        "y_pred",
    )
    assert REGRESSION_PREDICTION_COLUMNS == (
        "trade_date",
        "y_true",
        "y_pred",
    )


def test_clustering_result_and_table_columns_are_stable() -> None:
    assert PROFILE_FEATURES == ("mean_return", "volatility", "max_drawdown")
    assert PROFILE_COLUMNS == (
        "symbol",
        "mean_return",
        "volatility",
        "max_drawdown",
        "cluster",
    )
    assert CLUSTER_CENTER_COLUMNS == (
        "cluster",
        "mean_return",
        "volatility",
        "max_drawdown",
    )
    assert ClusteringResult.__required_keys__ == frozenset(CLUSTERING_RESULT_KEYS)
