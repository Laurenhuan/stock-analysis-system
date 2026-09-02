"""Decision-tree classification for next-day direction prediction (Role 4).

Contract v0.3 — features are fixed, errors returned as dicts.
"""

from __future__ import annotations

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from src.contracts.supervised import (
    CLASSIFICATION_PREDICTION_COLUMNS,
    ClassificationMetrics,
    ClassificationResult,
)

# ---------------------------------------------------------------------------
# Fixed feature set (Contract v0.3 — not configurable)
# ---------------------------------------------------------------------------

FEATURE_NAMES: list[str] = [
    "return_lag1",
    "return_lag2",
    "ma_diff",
    "volatility_20d",
    "volume_change",
]

_REQUIRED_INPUT_COLS = ("trade_date", "close", "volume")

_MIN_SAMPLES = 30  # minimum rows after feature engineering


# ---------------------------------------------------------------------------
# Error code constants
# ---------------------------------------------------------------------------

class ErrorCode:
    MISSING_COLUMNS = "MISSING_COLUMNS"
    UNSORTED_DATE = "UNSORTED_DATE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


def _error(status: str, code: str, message: str) -> dict:
    """Build a standard error response dict."""
    return {"status": status, "data": None, "code": code, "message": message}


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
    data["return_lag1"] = data["return"].shift(1)
    data["return_lag2"] = data["return"].shift(2)

    # --- moving averages ---
    data["ma5"] = data["close"].rolling(window=5).mean()
    data["ma20"] = data["close"].rolling(window=20).mean()
    data["ma_diff"] = data["ma5"] - data["ma20"]

    # --- volatility & volume ---
    data["volatility_20d"] = data["return"].rolling(window=20).std()
    data["volume_change"] = data["volume"].pct_change()

    # --- label: next-day direction (model-private) ---
    data["next_return"] = data["return"].shift(-1)
    data["label"] = (data["next_return"] > 0).astype(int)

    # Drop rows with NaN (from lag/rolling) and the last row (no next_return)
    data = data.dropna(subset=FEATURE_NAMES + ["label"])
    data = data[data["next_return"].notna()]

    return data


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_classification(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
) -> ClassificationResult | dict:
    """Train a DecisionTreeClassifier and evaluate on a time-ordered test set.

    Parameters
    ----------
    df : pd.DataFrame
        Raw market data for a **single** stock.  Must contain at least
        ``trade_date``, ``close``, ``volume``.  Rows must be sorted by
        ``trade_date`` ascending.
    train_ratio : float
        Fraction of data used for training (time-ordered split, default 0.8).

    Returns
    -------
    ClassificationResult | dict
        On success: dict with keys model, feature_names, metrics, predictions.
        On error: dict with keys status="error", data=None, code, message.
    """
    # --- validate input ---
    missing = set(_REQUIRED_INPUT_COLS) - set(df.columns)
    if missing:
        return _error(
            "error",
            ErrorCode.MISSING_COLUMNS,
            f"Missing required columns: {sorted(missing)}",
        )

    if not df["trade_date"].is_monotonic_increasing:
        return _error(
            "error",
            ErrorCode.UNSORTED_DATE,
            "DataFrame must be sorted by trade_date ascending",
        )

    # --- feature engineering ---
    data = _build_features(df)

    if len(data) < _MIN_SAMPLES:
        return _error(
            "error",
            ErrorCode.INSUFFICIENT_DATA,
            f"Need at least {_MIN_SAMPLES} samples, got {len(data)}",
        )

    # --- time-ordered split ---
    X = data[FEATURE_NAMES]
    y = data["label"]

    split_idx = int(len(X) * train_ratio)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

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
    from sklearn.metrics import accuracy_score, confusion_matrix

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
        "feature_names": FEATURE_NAMES,
        "metrics": metrics,
        "predictions": predictions,
    }


# ---------------------------------------------------------------------------
# Demo: run directly to see the decision tree
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sklearn.tree import plot_tree
    import matplotlib.pyplot as plt

    # ---- 读取假数据 ----
    csv_path = r"C:\Users\zhouc\WorkBuddy\2026-09-02-08-57-05\fake_data\fake_stock_000001_seed1.csv"
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    # 适配列名：date → trade_date
    df.rename(columns={"date": "trade_date"}, inplace=True)

    # CSV 缺 volume 列，用 1.0 占位（run_classification 内部会重新算 volume_change）
    if "volume" not in df.columns:
        df["volume"] = 1.0

    # 删除最后一行（next_direction 为空）
    df.dropna(subset=["next_direction"], inplace=True)
    df["next_direction"] = df["next_direction"].astype(int)

    print(f"数据量: {len(df)} 行")
    print(f"列: {list(df.columns)}")

    # ---- 直接用预计算特征训练（不重复算） ----
    import numpy as np
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix

    feature_cols = ["return_lag1", "return_lag2", "ma_diff", "volatility_20d", "volume_change"]
    X = df[feature_cols]
    y = df["next_direction"]

    # 时间顺序 80/20 切分
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"训练集: {len(X_train)} 样本, 测试集: {len(X_test)} 样本")
    print(f"训练集时间: {df['trade_date'].iloc[0]} ~ {df['trade_date'].iloc[split_idx-1]}")
    print(f"测试集时间: {df['trade_date'].iloc[split_idx]} ~ {df['trade_date'].iloc[-1]}")

    # 训练
    model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 评价
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    print(f"\n===== 分类结果 =====")
    print(f"Accuracy: {acc:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    # ---- 画决策树 ----
    plt.figure(figsize=(16, 8), dpi=120)
    plot_tree(
        model,
        feature_names=feature_cols,
        class_names=["Down (0)", "Up (1)"],
        filled=True,
        rounded=True,
        fontsize=10,
    )
    plt.title("Decision Tree — fake_stock_000001", fontsize=14)
    plt.tight_layout()
    plt.savefig("decision_tree.png", dpi=150, bbox_inches="tight")
    print("\n决策树已保存到 decision_tree.png")
