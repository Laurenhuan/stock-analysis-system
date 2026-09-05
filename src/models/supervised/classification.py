"""Decision-tree classification for next-day direction prediction (Role 4).

Contract v0.2 — features are fixed, errors raised as exceptions.
"""

from __future__ import annotations

from typing import TypedDict

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

_SPLIT_RATIO = 0.8
_MIN_SAMPLES_LEAF = 20
# A split needs at least two leaves. With an 80% training partition this
# requires at least 50 effective samples so the training set contains 40 rows.
_MIN_TRAIN_SAMPLES = 2 * _MIN_SAMPLES_LEAF
_MIN_SAMPLES = 50


class ClassificationSampleInfo(TypedDict):
    """Public, presentation-safe sample diagnostics for one classification run."""

    input_rows: int
    effective_rows: int
    dropped_rows: int
    train_rows: int
    test_rows: int
    train_date_range: str
    test_date_range: str
    split_ratio: float

class NextDirectionForecast(TypedDict):
    """Latest next-trading-day signal, separate from evaluation Contract v0.2."""

    as_of_date: str
    predicted_class: int
    direction_label: str
    training_rows: int


# ---------------------------------------------------------------------------
# Validation helpers (private)
# ---------------------------------------------------------------------------


def _validate_input(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize one stock without mutating the caller's frame."""
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError(
            f"分类输入必须是 pandas.DataFrame，收到 {type(df).__name__}"
        )
    if df.empty:
        raise DataValidationError("分类输入 DataFrame 为空")

    missing = set(_REQUIRED_INPUT_COLS) - set(df.columns)
    if missing:
        raise DataValidationError(f"分类输入缺少必要字段：{sorted(missing)}")

    data = df.copy()
    dates = pd.to_datetime(data["trade_date"], errors="coerce")
    if dates.isna().any():
        raise DataValidationError("trade_date 存在无法解析的日期")
    if not dates.is_monotonic_increasing or dates.duplicated().any():
        raise DataValidationError("trade_date 必须严格升序且不能重复")
    data["trade_date"] = dates

    for col in ("close", "volume"):
        original = data[col]
        converted = pd.to_numeric(original, errors="coerce")
        if (original.notna() & converted.isna()).any():
            raise DataValidationError(f"{col} 存在无法转换为数值的内容")
        values = converted.to_numpy(dtype=float, na_value=np.nan)
        if not np.isfinite(values).all():
            raise DataValidationError(f"{col} 含 NaN 或无穷值")
        data[col] = converted

    if "symbol" in data.columns:
        symbols = data["symbol"]
        if symbols.isna().any() or symbols.astype(str).str.strip().eq("").any():
            raise DataValidationError("symbol 不能为空")
        if symbols.astype(str).nunique() != 1:
            raise DataValidationError("单次分类调用只处理一只股票")

    return data


# ---------------------------------------------------------------------------
# Feature engineering (private)
# ---------------------------------------------------------------------------


def _build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Construct predictors available at each date without creating a target."""
    data = df.copy()
    data["return"] = data["close"].pct_change().replace(
        [np.inf, -np.inf], np.nan
    )
    data["return_lag1"] = data["return"].shift(1)
    data["return_lag2"] = data["return"].shift(2)
    data["ma5"] = data["close"].rolling(window=5).mean()
    data["ma20"] = data["close"].rolling(window=20).mean()
    data["ma_diff"] = data["ma5"] - data["ma20"]
    data["volatility_20d"] = data["return"].rolling(window=20).std()
    data["volume_change"] = data["volume"].pct_change().replace(
        [np.inf, -np.inf], np.nan
    )
    return data


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construct historical predictors and the private next-day label."""
    data = _build_feature_frame(df)
    data["next_return"] = data["return"].shift(-1)
    data["label"] = (data["next_return"] > 0).astype(int)
    data = data.dropna(subset=FEATURE_NAMES + ["next_return"])
    return data


def _new_classifier() -> DecisionTreeClassifier:
    """Return the single Contract-approved deterministic estimator."""
    return DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=_MIN_SAMPLES_LEAF,
        random_state=42,
    )

def _prepare_model_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Return validated input, effective model rows and the split index."""
    prepared = _validate_input(df)
    data = _build_features(prepared)
    if len(data) < _MIN_SAMPLES:
        raise InsufficientDataError(
            f"滚动窗口与次日标签处理后至少需要 {_MIN_SAMPLES} 个有效样本，"
            f"当前仅 {len(data)} 个"
        )
    split_idx = int(len(data) * _SPLIT_RATIO)
    if split_idx < _MIN_TRAIN_SAMPLES:
        raise InsufficientDataError(
            f"训练集至少需要 {_MIN_TRAIN_SAMPLES} 个样本才能满足 "
            f"min_samples_leaf={_MIN_SAMPLES_LEAF} 的基本分裂条件"
        )
    if len(data) - split_idx == 0:
        raise InsufficientDataError("测试集为空")
    return prepared, data, split_idx


def _format_date_range(dates: pd.Series) -> str:
    return (
        f"{dates.iloc[0].date().isoformat()} 至 "
        f"{dates.iloc[-1].date().isoformat()}"
    )


def get_classification_sample_info(
    df: pd.DataFrame,
) -> ClassificationSampleInfo:
    """Describe effective rows and the exact time-ordered 80/20 split."""
    prepared, data, split_idx = _prepare_model_data(df)
    return ClassificationSampleInfo(
        input_rows=int(len(prepared)),
        effective_rows=int(len(data)),
        dropped_rows=int(len(prepared) - len(data)),
        train_rows=int(split_idx),
        test_rows=int(len(data) - split_idx),
        train_date_range=_format_date_range(data["trade_date"].iloc[:split_idx]),
        test_date_range=_format_date_range(data["trade_date"].iloc[split_idx:]),
        split_ratio=_SPLIT_RATIO,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_classification(df: pd.DataFrame) -> ClassificationResult:
    """Train a DecisionTreeClassifier and evaluate on a time-ordered test set.

    Parameters
    ----------
    df : pd.DataFrame
        Raw market data for a **single** stock.  Must contain at least
        ``trade_date``, ``close``, ``volume``.  Rows must be sorted by
        ``trade_date`` ascending.  Date filtering is handled by the caller
        (Role 1 Service); this function processes whatever rows it receives.

    Returns
    -------
    ClassificationResult
        Dict with keys model, feature_names, metrics, predictions.

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
    # Validation and feature engineering are shared with the public diagnostics.
    _, data, split_idx = _prepare_model_data(df)

    # --- time-ordered 80/20 split (Contract v0.2: fixed ratio) ---
    X = data[FEATURE_NAMES]
    y = data["label"]

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
    model = _new_classifier()

    model.fit(X_train, y_train)
    if model.tree_.node_count == 1:
        raise InsufficientDataError(
            "有效特征未能形成决策树分裂，当前模型会退化为单一类别预测"
        )

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

    return {
        "model": model,
        "feature_names": list(FEATURE_NAMES),
        "metrics": metrics,
        "predictions": predictions,
    }

def forecast_next_direction(df: pd.DataFrame) -> NextDirectionForecast:
    """Fit the same tree on all realised labels and signal the next trading day.

    Historical test metrics remain produced by ``run_classification``.  This
    function deliberately returns no claimed confidence: a tree leaf share is
    not a calibrated market probability.
    """
    prepared, labelled, _ = _prepare_model_data(df)
    latest_features = _build_feature_frame(prepared).dropna(
        subset=FEATURE_NAMES
    )
    if latest_features.empty:
        raise InsufficientDataError("最新交易日尚未形成完整分类特征")
    if labelled["label"].nunique() < 2:
        raise InsufficientDataError("完整历史样本只有一个方向类别，无法形成分类模型")

    model = _new_classifier()
    model.fit(labelled[FEATURE_NAMES], labelled["label"])
    if model.tree_.node_count == 1:
        raise InsufficientDataError("完整历史特征未能形成有效决策树分裂")
    latest = latest_features.iloc[[-1]]
    predicted = int(model.predict(latest[FEATURE_NAMES])[0])
    return NextDirectionForecast(
        as_of_date=latest["trade_date"].iloc[0].date().isoformat(),
        predicted_class=predicted,
        direction_label="上涨倾向" if predicted == 1 else "非上涨倾向",
        training_rows=int(len(labelled)),
    )
