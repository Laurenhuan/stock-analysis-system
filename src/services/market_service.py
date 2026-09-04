"""Application-facing orchestration for Contract-compliant market data."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from typing import TypedDict

from pandas import DataFrame, Timestamp

from src.data.clean import clean_market_data
from src.data.features import build_common_features
from src.data.fetch import (
    fetch_market_data,
    fetch_realtime_quotes,
    search_stock_symbols,
)
from src.utils.exceptions import DataValidationError


# Application demo universe. The symbols match Role 2's documented offline
# sample snapshot; data acquisition and parsing remain inside ``src.data``.
SAMPLE_SYMBOLS: tuple[str, ...] = (
    "000001.SZ",
    "000333.SZ",
    "000725.SZ",
    "000858.SZ",
    "002594.SZ",
    "300750.SZ",
    "600276.SH",
    "600519.SH",
    "601318.SH",
    "601899.SH",
)


class MarketDataMetadata(TypedDict):
    """Source information surfaced to Streamlit without exposing adapters."""

    data_source: str
    provider: str | None
    fetched_at: str | None
    is_sample: bool
    fallback_reason: str | None


def get_sample_symbols() -> list[str]:
    """Return the documented Role 2 offline-demo stock universe."""
    return list(SAMPLE_SYMBOLS)


def search_stocks(query: str, *, limit: int = 20) -> DataFrame:
    """Search the online A-share directory through Role 2's public API.

    The Streamlit layer calls this Service wrapper so provider access remains
    behind the application boundary. An empty query or no match returns the
    same empty, display-ready table as the data API.
    """
    return search_stock_symbols(query, limit=limit)


def get_sample_date_bounds(symbol: str | None = None) -> tuple[date, date]:
    """Return available dates from the formal Role 2 sample pipeline."""
    selected = symbol or SAMPLE_SYMBOLS[0]
    data = load_market_data(selected, source="sample")
    return data["trade_date"].min().date(), data["trade_date"].max().date()


def load_market_data(
    symbols: str | Iterable[str],
    start_date: date | str | None = None,
    end_date: date | str | None = None,
    *,
    source: str = "sample",
    fallback: bool = True,
) -> DataFrame:
    """Fetch, clean and enrich market data through Role 2 public functions.

    Provenance from the provider is preserved in ``DataFrame.attrs`` so pages
    can clearly distinguish offline Sample Data from provider data.
    """
    _validate_date_range(start_date, end_date)

    raw = fetch_market_data(
        symbols,
        start_date=start_date,
        end_date=end_date,
        source=source,
        fallback=fallback,
    )
    provenance = {
        "data_source": str(raw.attrs.get("data_source", source)),
        "provider": raw.attrs.get("provider"),
        "fetched_at": raw.attrs.get("fetched_at"),
        "is_sample": bool(raw.attrs.get("is_sample", source == "sample")),
        "fallback_reason": raw.attrs.get("fallback_reason"),
    }

    cleaned = clean_market_data(raw)
    featured = build_common_features(cleaned)
    featured.attrs.update(provenance)
    return featured


def get_market_metadata(data: DataFrame) -> MarketDataMetadata:
    """Return stable, display-ready provenance for a Service result."""
    source = str(data.attrs.get("data_source", "unknown"))
    return MarketDataMetadata(
        data_source=source,
        provider=data.attrs.get("provider"),
        fetched_at=data.attrs.get("fetched_at"),
        is_sample=bool(data.attrs.get("is_sample", source == "sample")),
        fallback_reason=data.attrs.get("fallback_reason"),
    )


def load_realtime_quotes(symbols: str | Iterable[str]) -> DataFrame:
    """Load current quote snapshots through Role 2's public data API.

    Realtime quotes intentionally bypass cleaning and feature engineering: they
    follow their own 12-column display schema rather than the daily-market
    Contract. Provider provenance remains available through DataFrame.attrs.
    """
    return fetch_realtime_quotes(symbols)


def get_market_overview(
    symbol: str,
    start_date: date,
    end_date: date,
    *,
    source: str = "sample",
) -> DataFrame:
    """Compatibility entry point for the single-stock overview page."""
    return load_market_data(
        symbol,
        start_date=start_date,
        end_date=end_date,
        source=source,
    )


# Compatibility aliases for callers created during the D1 prototype. They now
# use the formal Role 2 pipeline and no longer read the temporary D1 CSV.
get_demo_symbols = get_sample_symbols
get_demo_date_bounds = get_sample_date_bounds


def _validate_date_range(
    start_date: date | str | None,
    end_date: date | str | None,
) -> None:
    try:
        start = Timestamp(start_date) if start_date is not None else None
        end = Timestamp(end_date) if end_date is not None else None
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"日期格式无效：{exc}") from exc
    if start is not None and end is not None and start > end:
        raise DataValidationError("开始日期不能晚于结束日期")
