"""Common financial features built on the base Market Data Contract (Role 2).

Computes the 7 P0 derived fields consumed by Roles 3–5. Every window is
computed within a single symbol in ascending ``trade_date`` order, never
across symbols.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.contracts.market_data import BASE_MARKET_COLUMNS, COMMON_FEATURE_COLUMNS
from src.utils.exceptions import DataValidationError, InsufficientDataError


def build_common_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append the 7 P0 common derived fields to a base market DataFrame.

    Fields (all within one symbol, ascending by trade_date):
    - ``return``            : close_t / close_(t-1) - 1
    - ``cumulative_return`` : (1 + return).cumprod() - 1
    - ``ma5`` / ``ma20``    : 5 / 20 day rolling mean of close
    - ``volatility_20d``    : 20 day rolling sample std of return, ddof=1,
                              not annualized
    - ``volume_change``     : volume_t / volume_(t-1) - 1; NaN when the
                              previous volume is 0 (never inf)
    - ``drawdown``          : close_t / close.cummax() - 1 (<= 0)

    Rolling windows only emit values once full, so ``ma5`` has 4 leading NaN,
    ``ma20`` 19, and ``volatility_20d`` 20 (return itself carries one leading
    NaN). These leading NaN are legitimate and are preserved.

    Raises:
        InsufficientDataError: Input is empty.
        DataValidationError: Base columns are missing.
    """
    if df is None or df.empty:
        raise InsufficientDataError("没有可用于特征计算的数据")

    missing = [c for c in BASE_MARKET_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"缺少基础字段：{missing}")

    result = df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    grouped = result.groupby("symbol", sort=False)

    result["return"] = grouped["close"].transform(lambda s: s / s.shift(1) - 1)
    result["cumulative_return"] = grouped["return"].transform(
        lambda s: (1 + s).cumprod() - 1
    )
    result["ma5"] = grouped["close"].transform(lambda s: s.rolling(5).mean())
    result["ma20"] = grouped["close"].transform(lambda s: s.rolling(20).mean())
    result["volatility_20d"] = grouped["return"].transform(
        lambda s: s.rolling(20).std(ddof=1)
    )
    result["volume_change"] = grouped["volume"].transform(_volume_change)
    result["drawdown"] = grouped["close"].transform(lambda s: s / s.cummax() - 1)

    return result[list(BASE_MARKET_COLUMNS) + list(COMMON_FEATURE_COLUMNS)]


def _volume_change(volume: pd.Series) -> pd.Series:
    prev = volume.shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        change = volume / prev - 1
    # Previous-day volume of 0 must yield NaN, not +/-inf.
    return change.where(prev != 0, np.nan)
