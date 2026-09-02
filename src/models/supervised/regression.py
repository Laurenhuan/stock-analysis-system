"""Linear regression module — Role 6 核心实现 (P0)。

对照 Supervised Learning Contract v0.2 与 `docs/role_boundaries.md`：
- 研究框架：X(t) → return(t+1)，只用第 t 个交易日及之前可获得的信息。
- 单次调用处理**一只**股票；批量协调交给 Service Layer（Role 1）。
- 私有目标：next_return = return.shift(-1)。
- 切分：按 trade_date 升序，`split_index = int(n * 0.8)`，不随机打乱。
- P0 指标：mae、r2；测试集样本不足以定义 R² 时抛出 InsufficientDataError。
- 输出：`RegressionResult`（model、feature_names、metrics、date-aligned predictions）。

私有字段（next_return）只存在于模型内部，不写回公共 Market DataFrame。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

from src.contracts.supervised import (
    REGRESSION_PREDICTION_COLUMNS,
    RegressionMetrics,
    RegressionResult,
)
from src.utils.exceptions import DataValidationError, InsufficientDataError

# P0 使用的公共特征（t 日可获得；不含单调的 cumulative_return，避免伪相关/共线主导）。
FEATURE_COLS: list[str] = [
    "return",
    "ma5",
    "ma20",
    "volatility_20d",
    "volume_change",
    "drawdown",
]

SPLIT_RATIO: float = 0.8


def _prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    """校验并规范化输入（单只股票、升序、数值类型），仅做只读校验。"""
    required = ["symbol", "trade_date", "close", "return"] + FEATURE_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError(f"回归输入缺少必要字段：{missing}")
    if df is None or df.empty:
        raise DataValidationError("回归输入 DataFrame 为空")

    out = df.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    for c in FEATURE_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def fit_regression(df: pd.DataFrame) -> RegressionResult:
    """对一只股票的公共特征做次日收益率线性回归，返回 Contract 结果。

    Raises:
        DataValidationError: 字段缺失、日期顺序异常或非有限值。
        InsufficientDataError: 处理 NaN 与次日目标后训练/测试样本不足，或
            测试集样本不足以计算 R²。
    """
    prepared = _prepare_frame(df)
    if prepared["symbol"].nunique() != 1:
        raise DataValidationError("单次回归调用只处理一只股票")

    frame = prepared.sort_values("trade_date").reset_index(drop=True)
    if not frame["trade_date"].is_monotonic_increasing:
        raise DataValidationError("trade_date 必须严格升序")

    # 私有目标：下一交易日收益率。首行 return 为 NaN，末行 next_return 为 NaN。
    frame["next_return"] = frame["return"].shift(-1)

    cols = FEATURE_COLS + ["next_return"]
    data = frame.dropna(subset=cols).reset_index(drop=True)
    if data.empty:
        raise InsufficientDataError("处理 NaN 与次日目标后没有有效样本")

    split_index = int(len(data) * SPLIT_RATIO)
    X = data[FEATURE_COLS]
    y = data["next_return"].astype(float)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    if len(y_train) == 0 or len(y_test) == 0:
        raise InsufficientDataError("训练集或测试集为空")
    if len(y_test) < 2:
        raise InsufficientDataError("测试集样本不足以计算 R²")

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test).astype(float)
    y_true = y_test.values.astype(float)

    predictions = pd.DataFrame(
        {
            "trade_date": data["trade_date"].iloc[split_index:].values,
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )[list(REGRESSION_PREDICTION_COLUMNS)]

    return RegressionResult(
        model=model,
        feature_names=list(FEATURE_COLS),
        metrics=RegressionMetrics(
            mae=float(mean_absolute_error(y_true, y_pred)),
            r2=float(r2_score(y_true, y_pred)),
        ),
        predictions=predictions,
    )
