"""Presentation-ready summaries and input handling owned by Role 1."""

from __future__ import annotations

from collections.abc import Iterable
import re
from typing import TypedDict

from pandas import DataFrame, to_datetime

from src.utils.exceptions import DataValidationError, NoDataError


class MarketSummary(TypedDict):
    """Small factual summary for a loaded historical-data query."""

    row_count: int
    symbol_count: int
    first_date: str
    last_date: str


class ModelSampleSummary(TypedDict):
    """Counts that can be proven from the current public model Contract."""

    input_rows: int
    test_rows: int
    test_date_range: str


def prepare_symbol_selection(
    values: str | Iterable[str],
    *,
    min_count: int,
    max_count: int,
) -> list[str]:
    """Trim, uppercase and de-duplicate user-entered symbols.

    Exchange inference and security validation remain in Role 2's data layer.
    This Service helper only turns UI values into a stable ordered list.
    """
    if min_count < 1 or max_count < min_count:
        raise ValueError("股票数量限制配置无效")

    raw_values = [values] if isinstance(values, str) else list(values)
    tokens: list[str] = []
    for value in raw_values:
        if not isinstance(value, str):
            raise DataValidationError("股票代码必须是字符串")
        tokens.extend(re.split(r"[\s,，;；]+", value.strip()))

    symbols: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        symbol = token.strip().upper()
        if symbol and symbol not in seen:
            seen.add(symbol)
            symbols.append(symbol)

    if len(symbols) < min_count:
        raise DataValidationError(f"至少需要选择 {min_count} 只股票")
    if len(symbols) > max_count:
        raise DataValidationError(f"最多只能选择 {max_count} 只股票")
    return symbols


def get_market_summary(data: DataFrame) -> MarketSummary:
    """Summarize only values present in a loaded market DataFrame."""
    if data.empty:
        raise NoDataError("行情数据为空，无法生成概览")
    dates = to_datetime(data["trade_date"], errors="coerce").dropna()
    if dates.empty:
        raise DataValidationError("行情数据没有有效 trade_date")
    return MarketSummary(
        row_count=int(len(data)),
        symbol_count=int(data["symbol"].nunique()),
        first_date=dates.min().date().isoformat(),
        last_date=dates.max().date().isoformat(),
    )


def get_model_sample_summary(
    input_data: DataFrame,
    predictions: DataFrame,
) -> ModelSampleSummary:
    """Report raw and public test-set counts without recreating model internals."""
    if predictions.empty:
        raise NoDataError("模型未返回测试集预测")
    dates = to_datetime(predictions["trade_date"], errors="coerce").dropna()
    if dates.empty:
        raise DataValidationError("模型预测没有有效 trade_date")
    return ModelSampleSummary(
        input_rows=int(len(input_data)),
        test_rows=int(len(predictions)),
        test_date_range=(
            f"{dates.min().date().isoformat()} 至 "
            f"{dates.max().date().isoformat()}"
        ),
    )
