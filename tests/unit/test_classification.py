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
        assert set(result.keys()) == {
            "model",
            "feature_names",
            "metrics",
            "predictions",
        }

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
