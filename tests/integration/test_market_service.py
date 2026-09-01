"""Tests for the temporary Day 1 market overview integration flow."""

from datetime import date

import pytest

from src.services.market_service import get_market_overview
from src.utils.exceptions import DataValidationError, InvalidSymbolError, NoDataError


def test_market_overview_returns_filtered_demo_rows() -> None:
    result = get_market_overview(
        symbol="600519.SH",
        start_date=date(2026, 7, 6),
        end_date=date(2026, 7, 10),
    )

    assert not result.empty
    assert set(result["symbol"]) == {"600519.SH"}
    assert result["trade_date"].min().date() == date(2026, 7, 6)
    assert result["trade_date"].max().date() == date(2026, 7, 10)


def test_market_overview_rejects_reversed_dates() -> None:
    with pytest.raises(DataValidationError, match="开始日期不能晚于结束日期"):
        get_market_overview(
            symbol="600519.SH",
            start_date=date(2026, 7, 10),
            end_date=date(2026, 7, 6),
        )


def test_market_overview_rejects_unknown_symbol() -> None:
    with pytest.raises(InvalidSymbolError, match="不存在股票"):
        get_market_overview(
            symbol="999999.SH",
            start_date=date(2026, 7, 1),
            end_date=date(2026, 7, 21),
        )


def test_market_overview_reports_empty_date_range() -> None:
    with pytest.raises(NoDataError, match="没有 Sample Data"):
        get_market_overview(
            symbol="000001.SZ",
            start_date=date(2026, 8, 1),
            end_date=date(2026, 8, 5),
        )
