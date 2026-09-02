"""Exploratory data analysis owned by Role 3 (金融数据分析与可视化工程师).

Pure functions over Contract-compliant market DataFrames. Each function
accepts a ``pandas.DataFrame`` carrying the shared Market Data columns and
returns a reusable analysis table. This module performs no data acquisition,
cleaning, model training, Streamlit rendering or CSV access.
"""

from __future__ import annotations

import pandas as pd
from pandas import DataFrame

from src.utils.exceptions import DataValidationError, InsufficientDataError, NoDataError

# Default numeric fields for describe_statistics. Columns absent from the input
# are skipped silently.
_DESCRIBE_COLUMNS = ("open", "high", "low", "close", "volume", "amount", "return")


def _require_columns(
    df: DataFrame, required: tuple[str, ...], *, label: str
) -> None:
    """Raise NoDataError on empty input, DataValidationError on missing columns."""
    if not isinstance(df, DataFrame):
        raise DataValidationError(
            f"{label} 需要 pandas.DataFrame，收到 {type(df).__name__}"
        )
    if df.empty:
        raise NoDataError(f"{label} 输入数据为空")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError(f"{label} 缺少必需字段：{missing}")


def describe_statistics(
    df: DataFrame, *, columns: tuple[str, ...] = _DESCRIBE_COLUMNS
) -> DataFrame:
    """Per-symbol descriptive statistics for the given numeric columns.

    Returns a DataFrame indexed by ``symbol`` whose columns are a
    ``(field, statistic)`` MultiIndex over ``count / mean / std / min / max``.
    Columns named in ``columns`` but absent from ``df`` are skipped.
    """
    _require_columns(df, ("symbol",), label="describe_statistics")
    available = [c for c in columns if c in df.columns]
    if not available:
        raise DataValidationError("describe_statistics 没有可统计的数值列")
    return df.groupby("symbol")[available].agg(
        ["count", "mean", "std", "min", "max"]
    )


def date_range_summary(df: DataFrame) -> DataFrame:
    """Per-symbol start date, end date and valid row count."""
    _require_columns(df, ("symbol", "trade_date"), label="date_range_summary")
    return (
        df.groupby("symbol")["trade_date"]
        .agg(start_date="min", end_date="max", n_rows="count")
        .reset_index()
    )


def risk_return_summary(df: DataFrame) -> DataFrame:
    """Per-symbol return / volatility / max-drawdown summary.

    Returns columns:

    - ``mean_return``  : arithmetic mean of daily ``return``;
    - ``volatility``   : mean of the rolling ``volatility_20d`` (interval-average
      volatility, not annualized);
    - ``max_drawdown`` : minimum of ``drawdown`` (most negative, i.e. deepest).
    """
    _require_columns(
        df, ("symbol", "return", "volatility_20d", "drawdown"),
        label="risk_return_summary",
    )
    return (
        df.groupby("symbol")
        .agg(
            mean_return=("return", "mean"),
            volatility=("volatility_20d", "mean"),
            max_drawdown=("drawdown", "min"),
        )
        .reset_index()
    )


def _cumulative_return(returns: pd.Series) -> float:
    return float((1 + returns.dropna()).prod() - 1)


def _win_rate(returns: pd.Series) -> float:
    return float((returns > 0).mean())


def _std_return(returns: pd.Series) -> float:
    return float(returns.std(ddof=1))


def returns_comparison(df: DataFrame) -> DataFrame:
    """Per-symbol return comparison table.

    Returns columns: ``mean_return``, ``cumulative_return``, ``win_rate``
    (fraction of positive daily returns) and ``std_return`` (sample std,
    ddof=1).
    """
    _require_columns(df, ("symbol", "return"), label="returns_comparison")
    return (
        df.groupby("symbol")["return"]
        .agg(
            mean_return="mean",
            cumulative_return=_cumulative_return,
            win_rate=_win_rate,
            std_return=_std_return,
        )
        .reset_index()
    )


def correlation_matrix(df: DataFrame, *, method: str = "spearman") -> DataFrame:
    """Cross-symbol correlation matrix of daily returns.

    Daily returns are pivoted into a ``trade_date`` x ``symbol`` table and the
    pairwise correlation is computed with ``method``. Spearman (the default) is
    robust to the non-normal, fat-tailed distribution of daily returns.
    """
    _require_columns(
        df, ("symbol", "trade_date", "return"), label="correlation_matrix"
    )
    pivot = df.pivot(index="trade_date", columns="symbol", values="return")
    if pivot.shape[1] < 2:
        raise InsufficientDataError("至少需要 2 只股票才能计算相关系数矩阵")
    return pivot.corr(method=method)


def missing_values_summary(df: DataFrame) -> DataFrame:
    """Per-column missing-value count and ratio for any DataFrame."""
    if not isinstance(df, DataFrame):
        raise DataValidationError(
            f"missing_values_summary 需要 pandas.DataFrame，收到 {type(df).__name__}"
        )
    if df.empty:
        raise NoDataError("missing_values_summary 输入数据为空")
    return DataFrame(
        {"missing_count": df.isna().sum(), "missing_ratio": df.isna().mean()}
    )
