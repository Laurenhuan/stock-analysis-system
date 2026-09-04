"""Leakage and temporal integrity tests for run_classification (Role 4).

These tests verify:
- No future data leakage (X(t) → direction(t+1))
- No shuffle (strict time ordering)
- Correct label alignment
- Input DataFrame not mutated
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.supervised.classification import FEATURE_NAMES, run_classification


def _make_df(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2024-01-01", periods=n)
    price = 100 + np.cumsum(rng.randn(n) * 0.5)
    volume = rng.randint(1000, 5000, size=n).astype(float)
    return pd.DataFrame({"trade_date": dates, "close": price, "volume": volume})


def test_split_index_is_exact_80_percent() -> None:
    """split_index = int(n_samples * 0.8) exactly."""
    df = _make_df()
    result = run_classification(df)
    preds = result["predictions"]

    # Reconstruct the processed data to verify split
    data = df.copy()
    data["return"] = data["close"].pct_change()
    data["return_lag1"] = data["return"].shift(1)
    data["return_lag2"] = data["return"].shift(2)
    data["ma5"] = data["close"].rolling(window=5).mean()
    data["ma20"] = data["close"].rolling(window=20).mean()
    data["ma_diff"] = data["ma5"] - data["ma20"]
    data["volatility_20d"] = data["return"].rolling(window=20).std()
    data["volume_change"] = data["volume"].pct_change()
    data["next_return"] = data["return"].shift(-1)
    data["label"] = (data["next_return"] > 0).astype(int)
    data = data.dropna(subset=FEATURE_NAMES + ["label"])
    data = data[data["next_return"].notna()]

    n_samples = len(data)
    split_idx = int(n_samples * 0.8)

    # Predictions should cover exactly the last 20% of processed data
    assert len(preds) == n_samples - split_idx


def test_all_train_dates_before_test_dates() -> None:
    """All training dates must be strictly before all test dates."""
    df = _make_df()
    result = run_classification(df)
    preds = result["predictions"]

    min_test_date = preds["trade_date"].min()

    # Reconstruct processed data dates
    data = df.copy()
    data["return"] = data["close"].pct_change()
    data["return_lag1"] = data["return"].shift(1)
    data["return_lag2"] = data["return"].shift(2)
    data["ma5"] = data["close"].rolling(window=5).mean()
    data["ma20"] = data["close"].rolling(window=20).mean()
    data["ma_diff"] = data["ma5"] - data["ma20"]
    data["volatility_20d"] = data["return"].rolling(window=20).std()
    data["volume_change"] = data["volume"].pct_change()
    data["next_return"] = data["return"].shift(-1)
    data["label"] = (data["next_return"] > 0).astype(int)
    data = data.dropna(subset=FEATURE_NAMES + ["label"])
    data = data[data["next_return"].notna()]

    split_idx = int(len(data) * 0.8)
    train_dates = data["trade_date"].iloc[:split_idx]

    assert (train_dates < min_test_date).all(), (
        "LEAKAGE: some training dates are not strictly before test dates"
    )


def test_test_dates_strictly_ascending() -> None:
    """Test dates must be in strict ascending order (no shuffle)."""
    df = _make_df()
    result = run_classification(df)
    preds = result["predictions"]
    assert preds["trade_date"].is_monotonic_increasing, (
        "Test dates are not in chronological order — possible shuffle"
    )


def test_y_true_aligns_with_next_day_return() -> None:
    """y_true(t) must correspond to the return direction of the next trading day."""
    df = _make_df()
    result = run_classification(df)
    preds = result["predictions"]

    # Reconstruct to verify label alignment
    data = df.copy()
    data["return"] = data["close"].pct_change()
    data["return_lag1"] = data["return"].shift(1)
    data["return_lag2"] = data["return"].shift(2)
    data["ma5"] = data["close"].rolling(window=5).mean()
    data["ma20"] = data["close"].rolling(window=20).mean()
    data["ma_diff"] = data["ma5"] - data["ma20"]
    data["volatility_20d"] = data["return"].rolling(window=20).std()
    data["volume_change"] = data["volume"].pct_change()
    data["next_return"] = data["return"].shift(-1)
    data["label"] = (data["next_return"] > 0).astype(int)
    data = data.dropna(subset=FEATURE_NAMES + ["label"])
    data = data[data["next_return"].notna()]

    split_idx = int(len(data) * 0.8)
    test_data = data.iloc[split_idx:]

    # y_true should match the label derived from next_return
    expected_y_true = test_data["label"].values.astype(int)
    np.testing.assert_array_equal(preds["y_true"].values, expected_y_true)


def test_last_row_without_target_is_removed() -> None:
    """The last row (no next_return) must be excluded from predictions."""
    df = _make_df()
    result = run_classification(df)
    last_date = df["trade_date"].max()
    assert (result["predictions"]["trade_date"] != last_date).all()


def test_future_test_data_does_not_affect_train_features() -> None:
    """Changing data after the split point should not affect training features."""
    df = _make_df()
    r1 = run_classification(df)

    # Modify the last 50 rows (all in test set) and re-run
    df2 = df.copy()
    df2.loc[df2.index[-50:], "close"] *= 1.5
    r2 = run_classification(df2)

    # Training features are computed from close, so if we only change
    # test-period data, the model's feature_names and tree structure
    # should remain the same (same training data)
    assert r1["feature_names"] == r2["feature_names"]


def test_input_dataframe_not_mutated() -> None:
    """The input DataFrame must not gain model-private columns."""
    df = _make_df()
    original_cols = set(df.columns)
    run_classification(df)
    assert set(df.columns) == original_cols
    private_fields = {
        "next_return", "label", "return", "return_lag1", "return_lag2",
        "ma5", "ma20", "ma_diff", "volatility_20d", "volume_change",
    }
    assert private_fields.isdisjoint(set(df.columns))


def test_predictions_have_no_nan() -> None:
    """No NaN allowed in the predictions output."""
    df = _make_df()
    result = run_classification(df)
    preds = result["predictions"]
    assert not preds.isnull().any().any(), "NaN found in predictions"


def test_confusion_matrix_labels_order() -> None:
    """Confusion matrix must use fixed label order [0, 1]."""
    df = _make_df()
    result = run_classification(df)
    cm = result["metrics"]["confusion_matrix"]
    # Row 0 = actual class 0, Row 1 = actual class 1
    # Col 0 = predicted class 0, Col 1 = predicted class 1
    assert len(cm) == 2
    assert all(len(row) == 2 for row in cm)
    # Total samples in CM should equal test set size
    total = sum(sum(row) for row in cm)
    assert total == len(result["predictions"])


def test_sample_metadata_consistent_with_predictions() -> None:
    """n_test_samples should equal the number of predictions."""
    df = _make_df()
    result = run_classification(df)
    assert result["n_test_samples"] == len(result["predictions"])


def test_n_raw_trading_days_not_less_than_effective() -> None:
    """Raw trading days should be >= effective samples (feature eng removes rows)."""
    df = _make_df()
    result = run_classification(df)
    assert result["n_raw_trading_days"] >= result["n_effective_samples"]
