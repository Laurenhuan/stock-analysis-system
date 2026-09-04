"""Decision-tree classification for next-day direction prediction (Role 4).

Contract v0.2 — features are fixed, errors raised as exceptions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

from src.contracts.supervised import (
    CLASSIFICATION_PREDICTION_COLUMNS,
    ClassificationMetrics,
    ClassificationResult,
)
from src.utils.exceptions import DataValidationError, InsufficientDataError

# ---------------------------------------------------------------------------
# Fixed feature set (Contract v0.2 — not configurable)
# ---------------------------------------------------------------------------

FEATURE_NAMES: list[str] = [
    "return_lag1",
    "return_lag2",
    "ma_diff",
    "volatility_20d",
    "volume_change",
]

_REQUIRED_INPUT_COLS = ("trade_date", "close", "volume")

# Minimum samples after feature engineering.
# Rationale: with max_depth=3 and min_samples_leaf=20, the tree needs at
# least ~60 training samples to form a meaningful structure (3 levels × 20
# leaves).  30 samples as a hard floor catches obviously insufficient data
# before the split, while the train/test split (80/20) ensures the training
# set has ~24+ samples for a basic fit.  This is a safety net, not a
# substitute for checking train/test emptiness after the split.
_MIN_SAMPLES = 30


# ---------------------------------------------------------------------------
# Validation helpers (private)
# ---------------------------------------------------------------------------


def _validate_input(df: pd.DataFrame) -> None:
    """Validate raw input DataFrame before feature engineering.

    Raises DataValidationError for structural issues.
    """
    if df.empty:
        raise DataValidationError("Input DataFrame is empty")

    missing = set(_REQUIRED_INPUT_COLS) - set(df.columns)
    if missing:
        raise DataValidationError(
            f"Missing required columns: {sorted(missing)}"
        )

    if not df["trade_date"].is_monotonic_increasing:
        raise DataValidationError(
            "DataFrame must be sorted by trade_date ascending"
        )

    # Check for non-finite values in numeric columns
    for col in ("close", "volume"):
        if not np.isfinite(df[col]).all():
            raise DataValidationError(
                f"Column '{col}' contains NaN or infinite values"
            )


# ---------------------------------------------------------------------------
# Feature engineering (private)
# ---------------------------------------------------------------------------


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construct features and the binary label from raw OHLCV data.

    Returns the cleaned DataFrame (NaN dropped, last row removed).
    All intermediate columns (return, next_return, label) are model-private
    and never written back to the caller's DataFrame.
    """
    data = df.copy()

    # --- basic returns ---
    data["return"] = data["close"].pct_change()
    # Replace inf/-inf from pct_change (when close=0) with NaN, then drop
    data["return"] = data["return"].replace([np.inf, -np.inf], np.nan)

    data["return_lag1"] = data["return"].shift(1)
    data["return_lag2"] = data["return"].shift(2)

    # --- moving averages ---
    data["ma5"] = data["close"].rolling(window=5).mean()
    data["ma20"] = data["close"].rolling(window=20).mean()
    data["ma_diff"] = data["ma5"] - data["ma20"]

    # --- volatility & volume ---
    data["volatility_20d"] = data["return"].rolling(window=20).std()
    data["volume_change"] = data["volume"].pct_change()
    data["volume_change"] = data["volume_change"].replace(
        [np.inf, -np.inf], np.nan
    )

    # --- label: next-day direction (model-private) ---
    data["next_return"] = data["return"].shift(-1)
    data["label"] = (data["next_return"] > 0).astype(int)

    # Drop rows with NaN (from lag/rolling/inf) and the last row (no next_return)
    data = data.dropna(subset=FEATURE_NAMES + ["label"])
    data = data[data["next_return"].notna()]

    return data


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_classification(
    df: pd.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
) -> ClassificationResult:
    """Train a DecisionTreeClassifier and evaluate on a time-ordered test set.

    Parameters
    ----------
    df : pd.DataFrame
        Raw market data for a **single** stock.  Must contain at least
        ``trade_date``, ``close``, ``volume``.  Rows must be sorted by
        ``trade_date`` ascending.
    start_date : str, optional
        Inclusive start date for filtering (format: 'YYYY-MM-DD').
    end_date : str, optional
        Inclusive end date for filtering (format: 'YYYY-MM-DD').

    Returns
    -------
    ClassificationResult
        Dict with keys model, feature_names, metrics, predictions,
        and sample metadata (n_raw_trading_days, n_effective_samples,
        n_train_samples, n_test_samples, feature_importance).

    Raises
    ------
    DataValidationError
        If required columns are missing, data is not sorted, or
        non-finite values are found in close/volume.
    InsufficientDataError
        If not enough samples remain after feature engineering, or
        if train/test set is empty after the split, or if only one
        class exists in the training set.
    """
    # --- validate input ---
    _validate_input(df)

    # --- date range filtering ---
    data = df.copy()
    if start_date is not None:
        data = data[data["trade_date"] >= start_date]
    if end_date is not None:
        data = data[data["trade_date"] <= end_date]
    data = data.reset_index(drop=True)

    # Record raw trading days (after date filtering, before feature eng)
    n_raw_trading_days = len(data)

    # --- feature engineering ---
    data = _build_features(data)

    n_effective_samples = len(data)

    if n_effective_samples < _MIN_SAMPLES:
        raise InsufficientDataError(
            f"Need at least {_MIN_SAMPLES} samples after feature engineering, "
            f"got {n_effective_samples}"
        )

    # --- time-ordered 80/20 split (Contract v0.2: fixed ratio) ---
    X = data[FEATURE_NAMES]
    y = data["label"]

    split_idx = int(len(X) * 0.8)

    if split_idx == 0:
        raise InsufficientDataError("Training set is empty after split")

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    if len(X_test) == 0:
        raise InsufficientDataError("Test set is empty after split")

    # --- edge case: single class in training set ---
    if y_train.nunique() < 2:
        raise InsufficientDataError(
            "Training set has only one class; need both up and down samples"
        )

    # --- train ---
    model = DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=20,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # --- predict ---
    y_pred = model.predict(X_test)

    # --- metrics ---
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist()

    metrics: ClassificationMetrics = {
        "accuracy": acc,
        "confusion_matrix": cm,
    }

    # --- predictions DataFrame ---
    predictions = pd.DataFrame(
        {
            "trade_date": data["trade_date"].iloc[split_idx:].values,
            "y_true": y_test.values.astype(int),
            "y_pred": y_pred.astype(int),
        }
    )
    predictions = predictions[list(CLASSIFICATION_PREDICTION_COLUMNS)]

    # --- sample metadata ---
    n_train_samples = len(X_train)
    n_test_samples = len(X_test)
    feature_importance = model.feature_importances_.tolist()

    return {
        "model": model,
        "feature_names": FEATURE_NAMES,
        "metrics": metrics,
        "predictions": predictions,
        "n_raw_trading_days": n_raw_trading_days,
        "n_effective_samples": n_effective_samples,
        "n_train_samples": n_train_samples,
        "n_test_samples": n_test_samples,
        "feature_importance": feature_importance,
    }
