"""Integration tests for the formal Role 2 market-data Service flow."""

from datetime import date

import pandas as pd
import pytest

from src.contracts.market_data import BASE_MARKET_COLUMNS, COMMON_FEATURE_COLUMNS
from src.services.market_service import (
    get_market_metadata,
    get_market_overview,
    get_sample_date_bounds,
    get_sample_symbols,
    load_market_data,
    load_realtime_quotes,
    search_stocks,
)
from src.utils.exceptions import DataValidationError, InvalidSymbolError, NoDataError


def test_market_overview_returns_contract_data_and_features() -> None:
    result = get_market_overview(
        symbol="600519.SH",
        start_date=date(2024, 1, 2),
        end_date=date(2024, 1, 10),
    )

    assert not result.empty
    assert set(result["symbol"]) == {"600519.SH"}
    assert tuple(result.columns) == BASE_MARKET_COLUMNS + COMMON_FEATURE_COLUMNS
    assert result["trade_date"].min().date() == date(2024, 1, 2)
    assert result["trade_date"].max().date() == date(2024, 1, 10)


def test_market_service_preserves_sample_provenance() -> None:
    result = load_market_data("000001.SZ", source="sample")

    assert get_market_metadata(result) == {
        "data_source": "sample",
        "provider": "sample",
        "fetched_at": None,
        "is_sample": True,
        "fallback_reason": None,
    }


def test_realtime_service_preserves_independent_schema_and_provenance(
    monkeypatch,
) -> None:
    expected_columns = [
        "symbol", "name", "price", "change", "pct_change", "prev_close",
        "open", "high", "low", "volume", "amount", "timestamp",
    ]
    quotes = pd.DataFrame(
        [[
            "600519.SH", "贵州茅台", 1500.0, 10.0, 0.67, 1490.0,
            1495.0, 1510.0, 1480.0, 1000.0, 1500000.0,
            pd.Timestamp("2026-09-03 10:00:00"),
        ]],
        columns=expected_columns,
    )
    quotes.attrs.update(
        data_source="akshare_sina",
        provider="sina",
        fetched_at="2026-09-03T10:00:01+08:00",
        is_sample=False,
    )

    monkeypatch.setattr(
        "src.services.market_service.fetch_realtime_quotes",
        lambda symbols: quotes,
    )

    result = load_realtime_quotes("600519.SH")

    assert list(result.columns) == expected_columns
    assert get_market_metadata(result) == {
        "data_source": "akshare_sina",
        "provider": "sina",
        "fetched_at": "2026-09-03T10:00:01+08:00",
        "is_sample": False,
        "fallback_reason": None,
    }


def test_sample_universe_and_bounds_match_documented_snapshot() -> None:
    symbols = get_sample_symbols()
    first_date, last_date = get_sample_date_bounds()

    assert len(symbols) == 10
    assert len(set(symbols)) == 10
    assert first_date == date(2024, 1, 2)
    assert last_date == date(2024, 12, 31)


def test_stock_search_service_delegates_to_role2_public_api(monkeypatch) -> None:
    expected = pd.DataFrame(
        [{"symbol": "600519.SH", "name": "贵州茅台", "market": "SH"}]
    )
    monkeypatch.setattr(
        "src.services.market_service.search_stock_symbols",
        lambda query, *, limit: expected,
    )

    result = search_stocks("茅台", limit=8)

    pd.testing.assert_frame_equal(result, expected)


def test_market_overview_rejects_reversed_dates() -> None:
    with pytest.raises(DataValidationError, match="开始日期不能晚于结束日期"):
        get_market_overview(
            symbol="600519.SH",
            start_date=date(2024, 1, 10),
            end_date=date(2024, 1, 2),
        )


def test_market_service_rejects_invalid_single_date_bound() -> None:
    with pytest.raises(DataValidationError, match="日期格式无效"):
        load_market_data(
            "600519.SH",
            start_date="not-a-date",
            source="sample",
        )


def test_market_overview_rejects_malformed_symbol() -> None:
    # D4 起裸代码 600519 已合法，改用真正非法的未知交易所后缀。
    with pytest.raises(InvalidSymbolError, match="无效证券代码"):
        get_market_overview(
            symbol="600519.XY",
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 10),
        )


def test_market_overview_reports_unknown_sample_symbol() -> None:
    # 600000.SH 格式合法（沪 6 开头）但不在 Sample Data 中。
    with pytest.raises(NoDataError, match="Sample Data 中没有"):
        get_market_overview(
            symbol="600000.SH",
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 10),
        )


def test_market_overview_reports_empty_date_range() -> None:
    with pytest.raises(NoDataError, match="Sample Data 中没有"):
        get_market_overview(
            symbol="000001.SZ",
            start_date=date(2025, 1, 2),
            end_date=date(2025, 1, 10),
        )
