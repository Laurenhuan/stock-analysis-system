"""Market data acquisition for Role 2 (金融数据工程师).

Fetch A-share daily history via Tushare Pro, with a fixed sample-data
fallback so the demo still runs without a token, points, or network.

The returned DataFrame keeps the provider's raw shape (Tushare ``pro_bar``
columns). Call :func:`clean_market_data` to normalize it into the shared
Market Data Contract, then :func:`build_common_features` for derived fields.
"""

from __future__ import annotations

import os
import re
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.utils.exceptions import InvalidSymbolError, NoDataError

# 6-digit code + exchange suffix, e.g. 600519.SH / 000001.SZ.
_SYMBOL_RE = re.compile(r"^\d{6}\.(SH|SZ)$")

# Fixed sample data committed under data/sample/ (not gitignored) so the
# fallback works on a fresh clone with no token / no network.
_SAMPLE_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "sample" / "sample_daily.csv"
)


def fetch_market_data(
    symbols: str | Iterable[str],
    start_date=None,
    end_date=None,
    token: str | None = None,
    *,
    fallback: bool = True,
) -> pd.DataFrame:
    """Fetch A-share daily history, falling back to fixed sample data.

    Args:
        symbols: A single symbol or an iterable of symbols, e.g. ``"600519.SH"``
            or ``["600519.SH", "000001.SZ"]``.
        start_date: Inclusive start date. Accepts ``date``/``datetime``/
            ``pd.Timestamp`` or ``"YYYYMMDD"``/``"YYYY-MM-DD"`` strings.
        end_date: Inclusive end date, same formats.
        token: Tushare token; defaults to ``TUSHARE_TOKEN`` from the environment.
        fallback: When True (default), return fixed sample data if Tushare is
            unavailable or returns nothing. When False, raise instead.

    Returns:
        A raw provider-shaped DataFrame (Tushare ``pro_bar`` columns).
        An empty result is never returned; ``NoDataError`` is raised instead.

    Raises:
        InvalidSymbolError: A symbol is malformed or unsupported.
        NoDataError: No data is available and ``fallback`` is disabled, or the
            sample data has no rows for the requested symbols/range.
    """
    symbols = _normalize_symbols(symbols)
    for symbol in symbols:
        _validate_symbol(symbol)

    token = token or os.getenv("TUSHARE_TOKEN")
    start = _to_yyyymmdd(start_date)
    end = _to_yyyymmdd(end_date)

    if token:
        try:
            df = _fetch_tushare(symbols, start, end, token)
            if df is not None and not df.empty:
                return df.reset_index(drop=True)
        except Exception:
            # Provider unavailable (insufficient points, bad token, network...).
            # Fall through to the sample fallback below.
            pass

    if fallback:
        return _load_sample(symbols, start, end)

    raise NoDataError(f"未获取到 {symbols} 的行情数据")


def _normalize_symbols(symbols: str | Iterable[str]) -> list[str]:
    if isinstance(symbols, str):
        symbols = [symbols]
    result = [s for s in symbols]
    if not result:
        raise InvalidSymbolError("至少需要提供一个证券代码")
    return result


def _validate_symbol(symbol: str) -> None:
    if not isinstance(symbol, str) or not _SYMBOL_RE.fullmatch(symbol):
        raise InvalidSymbolError(
            f"无效证券代码：{symbol!r}（应为 6 位数字 + .SH/.SZ，如 600519.SH）"
        )


def _to_yyyymmdd(value) -> str:
    if value is None:
        return ""
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y%m%d")
    if isinstance(value, (datetime, date)):
        return value.strftime("%Y%m%d")
    return str(value).strip().replace("-", "").replace("/", "")


def _fetch_tushare(symbols: list[str], start: str, end: str, token: str) -> pd.DataFrame:
    import tushare as ts

    pro = ts.pro_api(token)
    frames = []
    for symbol in symbols:
        df = ts.pro_bar(
            ts_code=symbol,
            api=pro,
            adj="qfq",  # 前复权，符合契约口径
            start_date=start,
            end_date=end,
        )
        if df is not None and not df.empty:
            frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _load_sample(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    if not _SAMPLE_PATH.exists():
        raise NoDataError("未获取到行情数据，且缺少 Sample Data 回退文件")

    df = pd.read_csv(_SAMPLE_PATH, dtype={"ts_code": str, "trade_date": str})
    df = df[df["ts_code"].isin(symbols)]
    if start:
        df = df[df["trade_date"] >= start]
    if end:
        df = df[df["trade_date"] <= end]

    if df.empty:
        raise NoDataError(f"Sample Data 中没有 {symbols} 在指定区间内的数据")

    return df.reset_index(drop=True)
