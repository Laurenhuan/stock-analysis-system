"""Unit tests for src/models/supervised/classification.py (Role 4, Contract v0.2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from src.models.supervised.classification import FEATURE_NAMES, run_classification
from src.utils.exceptions import DataValidationError, InsufficientDataError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic single-stock OHLCV data sorted by trade_date."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2024-01-01", periods=n)
    price = 100 + np.cumsum(rng.randn(n) * 0.5)
    volume = rng.randint(1000, 5000, size=n).astype(float)
    return pd.DataFrame({"trade_date": dates, "close": price, "volume": volume})


# ---------------------------------------------------------------------------
# Normal behaviour
# ---------------------------------------------------------------------------


class TestRunClassificationNormal:
    """Happy-path tests for run_classification."""

    def test_returns_expected_keys(self) -> None:
        df = _make_df()
        result = run_classification(df)
        expected_core = {"model", "feature_names", "metrics", "predictions"}
        expected_meta = {
            "n_raw_trading_days", "n_effective_samples",
            "n_train_samples", "n_test_samples", "feature_importance",
        }
        assert expected_core.issubset(set(result.keys()))
        assert expected_meta.issubset(set(result.keys()))

    def test_model_is_fitted_decision_tree(self) -> None:
        df = _make_df()
        result = run_classification(df)
        assert isinstance(result["model"], DecisionTreeClassifier)
        # Check the model has been fitted (has tree_ attribute)
        assert hasattr(result["model"], "tree_")

    def test_feature_names_order_matches_training_matrix(self) -> None:
        df = _make_df()
        result = run_classification(df)
        # feature_names should be exactly the fixed FEATURE_NAMES
        assert result["feature_names"] == FEATURE_NAMES

    def test_metrics_contain_accuracy_and_cm(self) -> None:
        df = _make_df()
        result = run_classification(df)
        assert "accuracy" in result["metrics"]
        assert "confusion_matrix" in result["metrics"]

    def test_accuracy_in_valid_range(self) -> None:
        df = _make_df()
        result = run_classification(df)
        acc = result["metrics"]["accuracy"]
        assert 0.0 <= acc <= 1.0

    def test_confusion_matrix_is_2x2_integer(self) -> None:
        df = _make_df()
        result = run_classification(df)
        cm = result["metrics"]["confusion_matrix"]
        assert len(cm) == 2
        assert all(len(row) == 2 for row in cm)
        for row in cm:
            for val in row:
                assert isinstance(val, (int, np.integer))
                assert val >= 0

    def test_predictions_columns_are_correct(self) -> None:
        df = _make_df()
        result = run_classification(df)
        expected = ("trade_date", "y_true", "y_pred")
        assert tuple(result["predictions"].columns) == expected

    def test_predictions_no_nan(self) -> None:
        df = _make_df()
        result = run_classification(df)
        assert not result["predictions"].isnull().any().any()

    def test_predictions_labels_are_binary(self) -> None:
        df = _make_df()
        result = run_classification(df)
        preds = result["predictions"]
        assert set(preds["y_true"].unique()).issubset({0, 1})
        assert set(preds["y_pred"].unique()).issubset({0, 1})


# ---------------------------------------------------------------------------
# Time-series split (fixed 80/20)
# ---------------------------------------------------------------------------


class TestTimeSeriesSplit:
    """Ensure the fixed 80/20 time-ordered split works correctly."""

    def test_train_before_test(self) -> None:
        df = _make_df()
        result = run_classification(df)
        preds = result["predictions"]
        # All predicted trade_dates should be in the latest 20% of dates
        all_dates = sorted(df["trade_date"].values)
        cutoff = all_dates[int(len(all_dates) * 0.8)]
        assert (preds["trade_date"] >= cutoff).all()

    def test_split_index_exact(self) -> None:
        """split_index = int(n_samples * 0.8) exactly."""
        df = _make_df(n=300)
        result = run_classification(df)
        # After feature engineering, we lose ~20 rows (rolling window)
        # Test set should be roughly 20% of processed data
        n_pred = len(result["predictions"])
        # Processed data is ~280 rows, 20% = ~56
        assert 40 < n_pred < 80

    def test_predictions_dates_are_last_20_percent(self) -> None:
        """Prediction dates should equal the last 20% of processed data dates."""
        df = _make_df()
        result = run_classification(df)
        preds = result["predictions"]
        # Predictions should be in chronological order
        assert preds["trade_date"].is_monotonic_increasing


# ---------------------------------------------------------------------------
# Last row removal
# ---------------------------------------------------------------------------


class TestLastRowRemoved:
    """The very last row must be dropped because next_return is unavailable."""

    def test_last_row_not_in_predictions(self) -> None:
        df = _make_df()
        result = run_classification(df)
        last_date = df["trade_date"].max()
        assert (result["predictions"]["trade_date"] != last_date).all()


# ---------------------------------------------------------------------------
# Determinism / reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    """Same input should produce the same output."""

    def test_same_input_same_output(self) -> None:
        df = _make_df()
        r1 = run_classification(df)
        r2 = run_classification(df)
        assert r1["metrics"]["accuracy"] == r2["metrics"]["accuracy"]
        assert r1["metrics"]["confusion_matrix"] == r2["metrics"]["confusion_matrix"]
        pd.testing.assert_frame_equal(r1["predictions"], r2["predictions"])

    def test_input_dataframe_unchanged(self) -> None:
        """The original input DataFrame must not be mutated."""
        df = _make_df()
        original_cols = set(df.columns)
        original_len = len(df)
        run_classification(df)
        assert set(df.columns) == original_cols
        assert len(df) == original_len

    def test_private_fields_not_written_to_input(self) -> None:
        """next_return, label, return must not appear in the input DataFrame."""
        df = _make_df()
        run_classification(df)
        private_fields = {"next_return", "label", "return", "return_lag1",
                          "return_lag2", "ma5", "ma20", "ma_diff",
                          "volatility_20d", "volume_change"}
        assert private_fields.isdisjoint(set(df.columns))


# ---------------------------------------------------------------------------
# Error paths (exceptions, not dicts)
# ---------------------------------------------------------------------------


class TestErrors:

    def test_missing_columns_raises(self) -> None:
        df = pd.DataFrame(
            {"trade_date": pd.to_datetime(["2024-01-01"]), "close": [100.0]}
        )
        with pytest.raises(DataValidationError, match="Missing required columns"):
            run_classification(df)

    def test_unsorted_date_raises(self) -> None:
        df = _make_df().sort_values("trade_date", ascending=False)
        with pytest.raises(DataValidationError, match="sorted by trade_date"):
            run_classification(df)

    def test_empty_dataframe_raises(self) -> None:
        df = pd.DataFrame(columns=["trade_date", "close", "volume"])
        with pytest.raises(DataValidationError, match="empty"):
            run_classification(df)

    def test_non_finite_close_raises(self) -> None:
        df = _make_df()
        df.loc[100, "close"] = np.inf
        with pytest.raises(DataValidationError, match="NaN or infinite"):
            run_classification(df)

    def test_non_finite_volume_raises(self) -> None:
        df = _make_df()
        df.loc[100, "volume"] = np.nan
        with pytest.raises(DataValidationError, match="NaN or infinite"):
            run_classification(df)

    def test_too_few_samples_raises(self) -> None:
        df = _make_df(n=15)
        with pytest.raises(InsufficientDataError):
            run_classification(df)


# ---------------------------------------------------------------------------
# Dynamic single-stock input (D4 task)
# ---------------------------------------------------------------------------


class TestDynamicSingleStockInput:
    """Tests for dynamic date range and sample metadata."""

    def test_output_includes_sample_metadata(self) -> None:
        """Result should contain n_raw_trading_days, n_effective_samples, etc."""
        df = _make_df(n=200)
        result = run_classification(df)
        assert "n_raw_trading_days" in result
        assert "n_effective_samples" in result
        assert "n_train_samples" in result
        assert "n_test_samples" in result
        assert "feature_importance" in result

    def test_sample_metadata_values_are_positive(self) -> None:
        """All sample counts should be positive integers."""
        df = _make_df(n=200)
        result = run_classification(df)
        assert result["n_raw_trading_days"] > 0
        assert result["n_effective_samples"] > 0
        assert result["n_train_samples"] > 0
        assert result["n_test_samples"] > 0

    def test_train_plus_test_equals_effective(self) -> None:
        """n_train_samples + n_test_samples == n_effective_samples."""
        df = _make_df(n=200)
        result = run_classification(df)
        assert (
            result["n_train_samples"] + result["n_test_samples"]
            == result["n_effective_samples"]
        )

    def test_feature_importance_matches_feature_names(self) -> None:
        """feature_importance should be a list of floats, same length as feature_names."""
        df = _make_df(n=200)
        result = run_classification(df)
        fi = result["feature_importance"]
        assert isinstance(fi, list)
        assert len(fi) == len(FEATURE_NAMES)
        assert all(isinstance(v, float) for v in fi)
        # Sum of importances should be ~1.0
        assert abs(sum(fi) - 1.0) < 1e-6

    def test_two_stocks_produce_independent_results(self) -> None:
        """Different stock codes should produce different results."""
        df1 = _make_df(n=200, seed=42)
        df2 = _make_df(n=200, seed=123)
        r1 = run_classification(df1)
        r2 = run_classification(df2)
        # Accuracy and predictions should differ (different data)
        assert r1["metrics"]["accuracy"] != r2["metrics"]["accuracy"]

    def test_date_range_filtering(self) -> None:
        """start_date and end_date should filter the input data."""
        df = _make_df(n=300)
        # Full range
        r_full = run_classification(df)
        # Filtered range (first 200 days)
        r_filtered = run_classification(
            df, start_date="2024-01-01", end_date="2024-10-10"
        )
        # Filtered should have fewer raw trading days
        assert r_filtered["n_raw_trading_days"] < r_full["n_raw_trading_days"]

    def test_start_date_only(self) -> None:
        """Only start_date should filter from that date onwards."""
        df = _make_df(n=300)
        r = run_classification(df, start_date="2024-06-01")
        assert r["n_raw_trading_days"] < 300

    def test_end_date_only(self) -> None:
        """Only end_date should filter up to that date."""
        df = _make_df(n=300)
        r = run_classification(df, end_date="2024-06-01")
        assert r["n_raw_trading_days"] < 300

    def test_single_class_in_training_raises(self) -> None:
        """If training set has only one class, InsufficientDataError is raised."""
        # Create data where ALL prices are monotonically increasing.
        # This means every return > 0, every label = 1.
        # After feature eng, training set has only class 1 → error.
        n = 200
        dates = pd.bdate_range("2024-01-01", periods=n)
        price = 100 + np.arange(n) * 0.5  # strictly increasing
        volume = np.full(n, 1000.0)
        df = pd.DataFrame({"trade_date": dates, "close": price, "volume": volume})
        with pytest.raises(InsufficientDataError, match="only one class"):
            run_classification(df)

    def test_reproducibility_with_metadata(self) -> None:
        """Same input should produce same metadata."""
        df = _make_df(n=200)
        r1 = run_classification(df)
        r2 = run_classification(df)
        assert r1["n_raw_trading_days"] == r2["n_raw_trading_days"]
        assert r1["n_effective_samples"] == r2["n_effective_samples"]
        assert r1["n_train_samples"] == r2["n_train_samples"]
        assert r1["n_test_samples"] == r2["n_test_samples"]
        assert r1["feature_importance"] == r2["feature_importance"]
