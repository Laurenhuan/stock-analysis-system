"""Unit tests for src/models/supervised/classification.py (Role 4, Contract v0.3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.supervised.classification import (
    FEATURE_NAMES,
    ErrorCode,
    run_classification,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic single-stock OHLCV data sorted by trade_date."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2024-01-01", periods=n)
    price = 100 + np.cumsum(rng.randn(n) * 0.5)
    volume = rng.randint(1000, 5000, size=n).astype(float)
    return pd.DataFrame(
        {
            "trade_date": dates,
            "close": price,
            "volume": volume,
        }
    )


# ---------------------------------------------------------------------------
# Normal behaviour
# ---------------------------------------------------------------------------

class TestRunClassificationNormal:
    """Happy-path tests for run_classification."""

    def test_returns_expected_keys(self) -> None:
        df = _make_df()
        result = run_classification(df)
        assert set(result.keys()) == {"model", "feature_names", "metrics", "predictions"}

    def test_status_is_not_error(self) -> None:
        df = _make_df()
        result = run_classification(df)
        assert "status" not in result  # success result has no status key

    def test_metrics_contain_accuracy_and_cm(self) -> None:
        df = _make_df()
        result = run_classification(df)
        assert "accuracy" in result["metrics"]
        assert "confusion_matrix" in result["metrics"]
        assert 0.0 <= result["metrics"]["accuracy"] <= 1.0

    def test_confusion_matrix_is_2x2(self) -> None:
        df = _make_df()
        result = run_classification(df)
        cm = result["metrics"]["confusion_matrix"]
        assert len(cm) == 2
        assert all(len(row) == 2 for row in cm)

    def test_confusion_matrix_values_are_non_negative_integers(self) -> None:
        df = _make_df()
        result = run_classification(df)
        for row in result["metrics"]["confusion_matrix"]:
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

    def test_predictions_labels_are_0_or_1(self) -> None:
        df = _make_df()
        result = run_classification(df)
        preds = result["predictions"]
        assert set(preds["y_true"].unique()).issubset({0, 1})
        assert set(preds["y_pred"].unique()).issubset({0, 1})

    def test_feature_names_are_fixed(self) -> None:
        df = _make_df()
        result = run_classification(df)
        assert result["feature_names"] == FEATURE_NAMES

    def test_feature_names_match_contract(self) -> None:
        expected = ["return_lag1", "return_lag2", "ma_diff", "volatility_20d", "volume_change"]
        assert FEATURE_NAMES == expected


# ---------------------------------------------------------------------------
# Time-series split
# ---------------------------------------------------------------------------

class TestTimeSeriesSplit:
    """Ensure the 80/20 time-ordered split works correctly."""

    def test_train_before_test(self) -> None:
        df = _make_df()
        result = run_classification(df, train_ratio=0.8)
        preds = result["predictions"]
        all_dates = sorted(df["trade_date"].values)
        cutoff = all_dates[int(len(all_dates) * 0.8)]
        assert (preds["trade_date"] >= cutoff).all()

    def test_split_ratio_respected(self) -> None:
        df = _make_df(n=300)
        result = run_classification(df, train_ratio=0.7)
        n_pred = len(result["predictions"])
        assert 50 < n_pred < 120  # rough sanity band


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
# Error paths (returned as dicts, not raised)
# ---------------------------------------------------------------------------

class TestErrors:

    def test_missing_columns_returns_error(self) -> None:
        df = pd.DataFrame({"trade_date": [], "close": []})  # missing volume
        result = run_classification(df)
        assert result["status"] == "error"
        assert result["code"] == ErrorCode.MISSING_COLUMNS
        assert result["data"] is None

    def test_unsorted_date_returns_error(self) -> None:
        df = _make_df().sort_values("trade_date", ascending=False)
        result = run_classification(df)
        assert result["status"] == "error"
        assert result["code"] == ErrorCode.UNSORTED_DATE

    def test_too_few_samples_returns_error(self) -> None:
        df = _make_df(n=15)  # well below _MIN_SAMPLES
        result = run_classification(df)
        assert result["status"] == "error"
        assert result["code"] == ErrorCode.INSUFFICIENT_DATA
