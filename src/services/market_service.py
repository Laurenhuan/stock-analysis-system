"""Application-facing market overview service for the Day 1 prototype."""

from datetime import date

from pandas import DataFrame, Timestamp

from src.services.demo_market_data import load_demo_market_data
from src.utils.exceptions import DataValidationError, InvalidSymbolError, NoDataError


def get_demo_symbols() -> list[str]:
    """Return the symbols that actually exist in the temporary sample CSV."""
    data = load_demo_market_data()
    return sorted(data["symbol"].unique().tolist())


def get_demo_date_bounds(symbol: str) -> tuple[date, date]:
    """Return the first and last available demo dates for one symbol."""
    data = load_demo_market_data()
    symbol_data = data[data["symbol"] == symbol]
    if symbol_data.empty:
        raise InvalidSymbolError(f"Sample Data 中不存在股票：{symbol}")

    return (
        symbol_data["trade_date"].min().date(),
        symbol_data["trade_date"].max().date(),
    )


def get_market_overview(
    symbol: str,
    start_date: date,
    end_date: date,
) -> DataFrame:
    """Filter Day 1 sample data for the UI.

    This is integration-only logic. It does not fetch, clean, or calculate
    formal financial features owned by Role 2.
    """
    if start_date > end_date:
        raise DataValidationError("开始日期不能晚于结束日期")

    data = load_demo_market_data()
    if symbol not in data["symbol"].unique():
        raise InvalidSymbolError(f"Sample Data 中不存在股票：{symbol}")

    start = Timestamp(start_date)
    end = Timestamp(end_date)
    result = data[
        (data["symbol"] == symbol)
        & (data["trade_date"] >= start)
        & (data["trade_date"] <= end)
    ].copy()

    if result.empty:
        raise NoDataError("所选日期范围内没有 Sample Data")

    return result.reset_index(drop=True)
