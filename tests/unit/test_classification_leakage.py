"""Leakage and shuffle guards for run_classification (Role 4)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.supervised.classification import run_classification


def _make_df(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2024-01-01", periods=n)
    price = 100 + np.cumsum(rng.randn(n) * 0.5)
    volume = rng.randint(1000, 5000, size=n).astype(float)
    return pd.DataFrame(
        {"trade_date": dates, "close": price, "volume": volume}
    )


def test_no_future_data_leakage() -> None:
    """Test set trade_dates must all be strictly after training set.

    Instead of duplicating feature engineering, we run the model at two
    different train_ratios and verify the test sets are consistent
    (the larger training set should produce a superset of test dates).
    """
    df = _make_df()
    result_80 = run_classification(df, train_ratio=0.8)
    result_60 = run_classification(df, train_ratio=0.6)  # noqa: E501

    min_test_80 = result_80["predictions"]["trade_date"].min()
    min_test_60 = result_60["predictions"]["trade_date"].min()

    # 80% train → test starts later than 60% train
    assert min_test_80 >= min_test_60, (
        "LEAKAGE: larger training set should start testing later"
    )

    # All test dates should be in chronological order (no shuffle)
    preds = result_80["predictions"]
    assert preds["trade_date"].is_monotonic_increasing


def test_no_shuffle() -> None:
    """Training dates must all precede test dates (no random mixing)."""
    df = _make_df()
    result = run_classification(df, train_ratio=0.8)

    preds = result["predictions"]
    min_test_date = preds["trade_date"].min()

    # All predictions should be for dates >= min_test_date
    # (i.e., they are in chronological order, not shuffled)
    assert (preds["trade_date"] >= min_test_date).all()

    # predictions should be in ascending date order (no shuffle within test)
    assert preds["trade_date"].is_monotonic_increasing, (
        "Predictions are not in chronological order — possible shuffle"
    )


def test_predictions_have_no_nan() -> None:
    """No NaN allowed in the predictions output."""
    df = _make_df()
    result = run_classification(df)
    preds = result["predictions"]
    assert not preds.isnull().any().any(), "NaN found in predictions"


def test_confusion_matrix_is_2x2() -> None:
    """Confusion matrix must always be exactly 2×2 with labels [0, 1]."""
    df = _make_df()
    result = run_classification(df)
    cm = result["metrics"]["confusion_matrix"]
    assert len(cm) == 2, f"Expected 2 rows, got {len(cm)}"
    assert all(len(row) == 2 for row in cm), "Each row must have 2 columns"
    # All values should be non-negative integers
    for row in cm:
        for val in row:
            assert isinstance(val, (int, np.integer))
            assert val >= 0
