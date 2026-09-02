"""Normalize raw provider data into the shared Market Data Contract (Role 2).

Turns raw Tushare ``pro_bar`` output into the 8 base columns, converting units
(手→股, 千元→元), coercing types, sorting, de-duplicating, and rejecting
conflicting duplicate ``(symbol, trade_date)`` records.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.contracts.market_data import BASE_MARKET_COLUMNS
from src.utils.exceptions import DataValidationError, NoDataError

# Tushare daily `pro_bar` units: vol = 手 (lots of 100 shares), amount = 千元.
_LOT_TO_SHARES = 100
_THOUSAND_YUAN_TO_YUAN = 1000

_NUMERIC_COLUMNS = ("open", "high", "low", "close", "volume", "amount")


def clean_market_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw provider data into the base Market Data schema.

    Responsibilities:
    - Map provider columns (``ts_code`` → ``symbol``).
    - Convert units to the contract's (``vol`` 手 → ``volume`` 股,
      ``amount`` 千元 → 元).
    - Coerce ``trade_date`` to timezone-naive ``datetime64[ns]`` and prices to
      numeric; drop rows whose date or numeric fields are missing/non-finite.
    - Sort by ``(symbol, trade_date)`` ascending.
    - Drop exact duplicates; raise on conflicting duplicates.

    Raises:
        NoDataError: Input is empty, or nothing remains after cleaning.
        DataValidationError: Required columns are missing, or duplicate
            ``(symbol, trade_date)`` keys carry conflicting values.
    """
    if raw is None or raw.empty:
        raise NoDataError("原始行情数据为空")

    df = raw.copy()

    # 1) Provider column mapping.
    df = df.rename(columns={"ts_code": "symbol"})

    # 2) Unit conversion (Tushare native units → contract units).
    if "vol" in df.columns and "volume" not in df.columns:
        df["volume"] = pd.to_numeric(df["vol"], errors="coerce") * _LOT_TO_SHARES
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce") * _THOUSAND_YUAN_TO_YUAN

    # 3) Required columns must now all be present.
    missing = [c for c in BASE_MARKET_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"缺少必要字段：{missing}")

    df = df[list(BASE_MARKET_COLUMNS)].copy()

    # 4) Type coercion. The contract requires timezone-naive datetime64[ns];
    #    pandas 3.x otherwise defaults to microsecond resolution.
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").astype(
        "datetime64[ns]"
    )
    for col in _NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 5) Drop rows that cannot be made valid (unparseable date, missing or
    #    non-finite numeric). This is the contract's "delete clearly invalid
    #    records" path.
    valid = df["trade_date"].notna()
    for col in _NUMERIC_COLUMNS:
        valid &= np.isfinite(df[col])
    df = df.loc[valid].copy()

    if df.empty:
        raise NoDataError("清洗后没有有效数据")

    # 6) Sort, then drop exact duplicates.
    df = df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    df = df.drop_duplicates().reset_index(drop=True)

    # 7) Conflicting duplicates on the unique key (symbol, trade_date).
    conflicts = df.duplicated(subset=["symbol", "trade_date"], keep=False)
    if conflicts.any():
        examples = df.loc[conflicts, ["symbol", "trade_date"]].drop_duplicates()
        raise DataValidationError(
            f"同一 (symbol, trade_date) 存在数值冲突："
            f"{examples.head(5).to_dict('records')}"
        )

    return df
