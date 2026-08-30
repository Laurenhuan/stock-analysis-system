"""Importable field definitions for Market Data Contract v0.2."""

from typing import TypedDict

from pandas import Timestamp


BASE_MARKET_COLUMNS = (
    "symbol",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
)

COMMON_FEATURE_COLUMNS = (
    "return",
    "cumulative_return",
    "ma5",
    "ma20",
    "volatility_20d",
    "volume_change",
    "drawdown",
)

MODEL_PRIVATE_FIELDS = frozenset(
    {"label", "target", "next_return", "prediction", "cluster"}
)


class MarketDataRow(TypedDict):
    """Required base fields for one normalized daily market observation."""

    symbol: str
    trade_date: Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float


MarketFeatureRow = TypedDict(
    "MarketFeatureRow",
    {
        **MarketDataRow.__annotations__,
        "return": float,
        "cumulative_return": float,
        "ma5": float,
        "ma20": float,
        "volatility_20d": float,
        "volume_change": float,
        "drawdown": float,
    },
)
