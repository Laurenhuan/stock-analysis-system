"""Market data acquisition for Role 2 (金融数据工程师).

Fetch A-share daily history via Tushare Pro (前复权 ``qfq``), with a fixed
sample-data fallback so the demo still runs without a token, points, or
network.

Data-source semantics (``source`` parameter):

- ``"sample"``  : read local sample only; never touch the network.
- ``"tushare"`` : require Tushare; raise on any provider failure (no fallback).
- ``"auto"``    : try Tushare when a token is present, else fall back to the
                  local sample. Falls back only on clearly-identified
                  permission / points / network errors; programming errors,
                  ``InvalidSymbolError``, date errors, and ``DataValidationError``
                  are never swallowed.

The returned DataFrame carries provenance in ``df.attrs`` so the Service layer
can tell the data origin without extra columns:

- ``df.attrs["data_source"]`` -> ``"sample"`` | ``"tushare"``
- ``df.attrs["is_sample"]``   -> ``True`` | ``False``
"""

from __future__ import annotations

import logging
import os
import re
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.utils.exceptions import DataValidationError, InvalidSymbolError, NoDataError

logger = logging.getLogger(__name__)

# 6-digit code + exchange suffix, e.g. 600519.SH / 000001.SZ.
_SYMBOL_RE = re.compile(r"^\d{6}\.(SH|SZ)$")

_SAMPLE_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "sample" / "sample_daily.csv"
)

_VALID_SOURCES = ("sample", "tushare", "auto")

# Tushare error-message fragments that indicate a quota / permission / points
# problem (safe to fall back from in "auto" mode).
_FALLBACK_HINTS = (
    "积分", "权限", "频率", "没有访问", "权限不足",
    "points", "permission", "quota",
)

_NETWORK_HINTS = ("connection", "reset", "timeout", "refused", "recv", "network")


def fetch_market_data(
    symbols: str | Iterable[str],
    start_date=None,
    end_date=None,
    token: str | None = None,
    *,
    source: str = "auto",
    fallback: bool = True,
) -> pd.DataFrame:
    """Fetch A-share daily history for one or more symbols.

    Args:
        symbols: Single symbol or iterable, e.g. ``"600519.SH"``.
        start_date: Inclusive start; ``date``/``datetime``/``pd.Timestamp`` or
            ``"YYYYMMDD"``/``"YYYY-MM-DD"``.
        end_date: Inclusive end, same formats.
        token: Tushare token; defaults to ``TUSHARE_TOKEN`` from the environment.
        source: ``"sample"`` | ``"tushare"`` | ``"auto"`` (default).
        fallback: Only used with ``source="auto"``. When False, a failed or empty
            Tushare request raises instead of falling back to sample.

    Returns:
        DataFrame carrying ``df.attrs["data_source"]`` and
        ``df.attrs["is_sample"]``.

    Raises:
        InvalidSymbolError: malformed or unsupported symbol.
        NoDataError: no data available and no fallback permitted, or the sample
            has no rows for the request.
    """
    if source not in _VALID_SOURCES:
        raise ValueError(f"source 必须是 {_VALID_SOURCES} 之一，收到 {source!r}")

    symbols = _normalize_symbols(symbols)
    for symbol in symbols:
        _validate_symbol(symbol)  # InvalidSymbolError must always propagate.

    token = token or os.getenv("TUSHARE_TOKEN")
    start = _to_yyyymmdd(start_date)
    end = _to_yyyymmdd(end_date)

    if source == "sample":
        return _load_sample(symbols, start, end)

    if source == "tushare":
        return _fetch_tushare_strict(symbols, start, end, token)

    # source == "auto"
    if token:
        try:
            df = _fetch_tushare(symbols, start, end, token)
        except Exception as exc:  # noqa: BLE001 - categorize provider errors
            if not _is_fallback_allowed(exc):
                raise  # programming / structure error: surface it
            logger.warning("Tushare 不可用，回退 Sample：%s", exc)
        else:
            if df is not None and not df.empty:
                return _mark_source(df, "tushare")
            # Empty provider result also falls through to the sample below.

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


def _fetch_tushare(symbols: list[str], start: str, end: str, token: str) -> pd.DataFrame | None:
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


def _fetch_tushare_strict(symbols: list[str], start: str, end: str, token: str | None) -> pd.DataFrame:
    """``source="tushare"``: never fall back; raise a clear provider error."""
    if not token:
        raise NoDataError("source='tushare' 需要 Tushare Token（TUSHARE_TOKEN 未设置）")
    try:
        df = _fetch_tushare(symbols, start, end, token)
    except Exception as exc:  # noqa: BLE001
        raise NoDataError(f"Tushare 获取失败：{exc}") from exc
    if df is None or df.empty:
        raise NoDataError(f"Tushare 未返回 {symbols} 在指定区间内的数据")
    return _mark_source(df, "tushare")


def _load_sample(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    if not _SAMPLE_PATH.exists():
        raise NoDataError("未获取到行情数据，且缺少 Sample Data 回退文件")

    df = pd.read_csv(_SAMPLE_PATH, dtype={"symbol": str, "trade_date": str})
    df = df[df["symbol"].isin(symbols)]
    if start:
        df = df[df["trade_date"] >= start]
    if end:
        df = df[df["trade_date"] <= end]

    if df.empty:
        raise NoDataError(f"Sample Data 中没有 {symbols} 在指定区间内的数据")

    return _mark_source(df, "sample")


def _mark_source(df: pd.DataFrame, source: str) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    df.attrs["data_source"] = source
    df.attrs["is_sample"] = source == "sample"
    return df


def _is_fallback_allowed(exc: BaseException) -> bool:
    """Whether an exception from Tushare may trigger a sample fallback.

    Only provider-side failures (quota / permission / points / network) are
    allowed. Local programming errors, ``InvalidSymbolError``,
    ``DataValidationError`` and date errors must propagate.
    """
    if isinstance(exc, (InvalidSymbolError, NoDataError, DataValidationError)):
        return False
    msg = str(exc)
    if any(hint in msg for hint in _FALLBACK_HINTS):
        return True
    # tushare surfaces provider/network failures as IOError("ERROR.") and
    # requests network failures as OSError subclasses.
    if isinstance(exc, OSError):
        return True
    low = msg.lower()
    return any(k in low for k in _NETWORK_HINTS)
