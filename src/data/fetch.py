"""Market data acquisition for Role 2 (金融数据工程师).

Fetch A-share daily history and realtime quotes online on demand. Daily history
goes through AkShare with a multi-source fallback so eastmoney being
intermittently unavailable does not break queries:

- primary  : AkShare ``stock_zh_a_hist`` (eastmoney, 前复权 ``qfq``, 10s timeout,
             2 retries with backoff)
- fallback : Tencent's ``newfqkline`` qfq endpoint (called directly, see note
             below), same daily ``qfq`` and ``sh600519`` / ``sz000001`` codes

Realtime quotes are exposed separately via ``fetch_realtime_quotes``
(eastmoney ``stock_zh_a_spot_em`` → sina ``stock_zh_a_spot``). Online results are
never written to a local CSV; only the fixed Sample fallback reads the committed
``data/sample`` file.

Data-source semantics (``source`` parameter):

- ``"sample"``  : read local sample only; never touch the network.
- ``"tushare"`` : require Tushare; raise on any provider failure (no fallback).
- ``"akshare"`` : require AkShare; eastmoney first, then Tencent; raise on total
                  failure (no silent sample fallback).
- ``"auto"``    : try Tushare when a token is present, else fall back to the
                  local sample. Falls back only on clearly-identified
                  permission / points / network errors; programming errors,
                  ``InvalidSymbolError``, date errors, and ``DataValidationError``
                  are never swallowed.

Provenance (``df.attrs``) so the Service layer can tell the origin without extra
columns:

- ``data_source``     -> ``"sample"`` | ``"tushare"`` | ``"akshare_eastmoney"``
                         | ``"akshare_tencent"`` (daily), or
                         ``"akshare_eastmoney"`` | ``"akshare_sina"`` (realtime)
- ``provider``        -> ``"sample"`` | ``"tushare"`` | ``"eastmoney"`` |
                         ``"tencent"`` | ``"sina"``
- ``is_sample``       -> ``True`` | ``False``
- ``fetched_at``      -> ISO-8601 UTC timestamp of the fetch (not set for Sample)
- ``fallback_reason`` -> only set when ``source="auto"`` fell back to sample.

.. note::
    AkShare's ``stock_zh_a_hist_tx`` mislabels the 6th column of Tencent's qfq
    kline (volume in 手) as ``"amount"`` and, via ``iloc[:, :6]``, drops the real
    amount column. Tencent's raw ``newfqkline`` rows are
    ``[date, open, close, high, low, volume(手), {}, turnover, amount(万元), ""]``.
    We therefore call that endpoint directly and convert volume 手→股 (×100) and
    amount 万元→元 (×10000) so the result matches the Market Data Contract.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from src.contracts.market_data import BASE_MARKET_COLUMNS
from src.utils.exceptions import DataValidationError, InvalidSymbolError, NoDataError

logger = logging.getLogger(__name__)

# 6-digit code + exchange suffix, e.g. 600519.SH / 000001.SZ.
_SYMBOL_RE = re.compile(r"^\d{6}\.(SH|SZ)$")

_SAMPLE_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "sample" / "sample_daily.csv"
)

_VALID_SOURCES = ("sample", "tushare", "akshare", "auto")

# Tushare error-message fragments that indicate a quota / permission / points
# problem (safe to fall back from in "auto" mode).
_FALLBACK_HINTS = (
    "积分", "权限", "频率", "没有访问", "权限不足",
    "points", "permission", "quota",
)

_NETWORK_HINTS = ("connection", "reset", "timeout", "refused", "recv", "network")

# AkShare ``stock_zh_a_hist`` 输出为中文列名；成交量单位为“手”（1 手 = 100 股），
# 成交额单位已经是“元”。列名映射、.SH/.SZ 补全与成交量换算在此完成，返回标准 Schema。
_AKSHARE_COLUMN_MAP = {
    "日期": "trade_date",
    "股票代码": "symbol",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
}
_AKSHARE_LOT_TO_SHARES = 100

# Tencent ``newfqkline`` qfq endpoint and its unit factors.
_TENCENT_URL = "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get"
_TENCENT_VOLUME_TO_SHARES = 100   # 手 → 股
_TENCENT_AMOUNT_TO_YUAN = 10000   # 万元 → 元
_TENCENT_TIMEOUT = 10.0
_TENCENT_MIN_YEAR = 1990          # 未指定起始日期时的最早回看年份

_EASTMONEY_TIMEOUT = 10.0
_EASTMONEY_MAX_RETRIES = 2

# Realtime snapshot output schema (separate from the daily Market Data Contract).
_REALTIME_COLUMNS = (
    "symbol", "name", "price", "change", "pct_change",
    "prev_close", "open", "high", "low", "volume", "amount", "timestamp",
)
_REALTIME_NUMERIC = (
    "price", "change", "pct_change", "prev_close",
    "open", "high", "low", "volume", "amount",
)

_REALTIME_EM_COLUMN_MAP = {
    "代码": "symbol",
    "名称": "name",
    "最新价": "price",
    "涨跌额": "change",
    "涨跌幅": "pct_change",
    "昨收": "prev_close",
    "今开": "open",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
}

_REALTIME_SINA_COLUMN_MAP = {
    "代码": "symbol",
    "名称": "name",
    "最新价": "price",
    "涨跌额": "change",
    "涨跌幅": "pct_change",
    "昨收": "prev_close",
    "今开": "open",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "时间戳": "timestamp",
}


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
        source: ``"sample"`` | ``"tushare"`` | ``"akshare"`` | ``"auto"`` (default).
        fallback: Only used with ``source="auto"``. When False, a failed or empty
            Tushare request raises instead of falling back to sample.

    Returns:
        DataFrame carrying ``df.attrs["data_source"]``, ``df.attrs["provider"]``,
        ``df.attrs["is_sample"]`` and (for online sources) ``df.attrs["fetched_at"]``.

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

    if source == "akshare":
        return _fetch_akshare_strict(symbols, start, end)

    # source == "auto"
    fallback_reason: str | None = None
    if token:
        try:
            df = _fetch_tushare(symbols, start, end, token)
        except Exception as exc:  # noqa: BLE001 - categorize provider errors
            if not _is_fallback_allowed(exc):
                raise  # programming / structure error: surface it
            fallback_reason = f"Tushare 失败：{exc}"
            logger.warning("Tushare 不可用，回退 Sample：%s", exc)
        else:
            if df is not None and not df.empty:
                return _mark_source(df, "tushare", provider="tushare")
            fallback_reason = "Tushare 返回空数据"
    else:
        fallback_reason = "未配置 TUSHARE_TOKEN"

    if fallback:
        df = _load_sample(symbols, start, end)
        df.attrs["fallback_reason"] = fallback_reason
        return df
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


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _retry(func: Callable, *, max_retries: int = 2, backoff: float = 1.0):
    """Call ``func`` up to ``1 + max_retries`` times with exponential backoff."""
    last: BaseException | None = None
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as exc:  # noqa: BLE001 - provider/network errors
            last = exc
            if attempt < max_retries:
                time.sleep(backoff * (2 ** attempt))
    assert last is not None
    raise last


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
    return _mark_source(df, "tushare", provider="tushare")


def _fetch_akshare_strict(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    """``source="akshare"``: eastmoney→tencent fallback, never falls back to Sample."""
    try:
        return _fetch_akshare(symbols, start, end)
    except NoDataError:
        raise
    except Exception as exc:  # noqa: BLE001 - surface unexpected errors clearly
        raise NoDataError(f"AkShare 获取失败：{exc}") from exc


def _fetch_akshare(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    """Fetch daily history via AkShare with eastmoney→tencent fallback.

    Returns contract-shaped data (``symbol`` 带 .SH/.SZ 后缀、``volume`` 股、
    ``amount`` 元、``trade_date`` 字符串)。任一源成功即返回并记录 provider；
    全部失败时抛出 ``NoDataError``，消息中列出已尝试的数据源。
    """
    attempts = [
        ("akshare_eastmoney", "eastmoney", _fetch_eastmoney),
        ("akshare_tencent", "tencent", _fetch_tencent),
    ]
    errors: list[str] = []
    for source, provider, fetcher in attempts:
        try:
            df = fetcher(symbols, start, end)
        except Exception as exc:  # noqa: BLE001 - provider/network errors
            logger.warning("AkShare 数据源 %s 失败：%s", provider, exc)
            errors.append(f"{provider}：{exc}")
            continue
        if df is not None and not df.empty:
            return _mark_source(df, source, provider=provider)
        errors.append(f"{provider}：返回空数据")

    raise NoDataError(f"AkShare 未能获取 {symbols} 的行情数据（已尝试 {'；'.join(errors)}）")


def _fetch_eastmoney(symbols: list[str], start: str, end: str) -> pd.DataFrame | None:
    """Daily history via AkShare ``stock_zh_a_hist`` (eastmoney, 前复权 ``qfq``).

    10s timeout，最多重试 2 次（指数退避）。返回标准 Schema 或空时返回 ``None``。
    """
    import akshare as ak

    frames = []
    for symbol in symbols:
        code = symbol[:6]  # AkShare API 只接受 6 位代码，不带交易所后缀
        kwargs = {
            "symbol": code,
            "period": "daily",
            "adjust": "qfq",
            "timeout": _EASTMONEY_TIMEOUT,
        }
        if start:
            kwargs["start_date"] = start
        if end:
            kwargs["end_date"] = end
        raw = _retry(
            lambda: ak.stock_zh_a_hist(**kwargs),
            max_retries=_EASTMONEY_MAX_RETRIES,
        )
        if raw is None or raw.empty:
            continue
        frames.append(_convert_akshare(raw))
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _fetch_tencent(symbols: list[str], start: str, end: str) -> pd.DataFrame | None:
    """Daily history from Tencent's qfq kline endpoint (``newfqkline``).

    See the module note: we call the endpoint directly because AkShare's
    ``stock_zh_a_hist_tx`` mislabels volume (手) as "amount" and drops the real
    amount. We loop per year (the endpoint returns the trailing ~640 bars ending
    at each year's end) and filter to ``[start, end]``.
    """
    import requests

    start_year = int(start[:4]) if start else _TENCENT_MIN_YEAR
    end_year = int(end[:4]) if end else datetime.now().year

    frames = []
    for symbol in symbols:
        tx_symbol = _code_to_tx_symbol(symbol[:6])
        per_symbol = []
        for year in range(start_year, end_year + 1):
            params = {
                "param": f"{tx_symbol},day,{year}-01-01,{year + 1}-12-31,640,qfq",
            }
            resp = requests.get(_TENCENT_URL, params=params, timeout=_TENCENT_TIMEOUT)
            resp.raise_for_status()
            payload = json.loads(resp.text)
            rows = (((payload.get("data") or {}).get(tx_symbol)) or {}).get("qfqday")
            if not rows:
                continue
            per_symbol.append(_convert_tencent(rows))
        if per_symbol:
            symbol_df = pd.concat(per_symbol, ignore_index=True)
            symbol_df["symbol"] = symbol
            frames.append(symbol_df)

    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True)
    # 各年请求区间存在重叠，按 (symbol, trade_date) 去重后再过滤/排序。
    df = df.drop_duplicates(subset=["symbol", "trade_date"])
    dates = pd.to_datetime(df["trade_date"], errors="coerce")
    mask = pd.Series(True, index=df.index)
    if start:
        mask &= dates >= pd.Timestamp(start)
    if end:
        mask &= dates <= pd.Timestamp(end)
    df = df[mask].sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    if df.empty:
        return None
    return df[list(BASE_MARKET_COLUMNS)]


def _convert_akshare(raw: pd.DataFrame) -> pd.DataFrame:
    """AkShare 输出 → 标准 Schema：中文列名→英文、补 .SH/.SZ、成交量 手→股 ×100。"""
    df = raw.rename(columns=_AKSHARE_COLUMN_MAP)
    df = df[list(BASE_MARKET_COLUMNS)].copy()
    df["symbol"] = df["symbol"].map(_symbol_with_exchange)
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce") * _AKSHARE_LOT_TO_SHARES
    return df


def _convert_tencent(rows: list[list]) -> pd.DataFrame:
    """Tencent ``newfqkline`` qfqday rows → 标准 Schema（不含 symbol 列）。

    Row layout: ``[date, open, close, high, low, volume(手), {}, turnover,
    amount(万元), ""]``。OHLC 为前复权价；volume 手→股 ×100、amount 万元→元 ×10000。
    """
    df = pd.DataFrame(
        {
            "trade_date": [r[0] for r in rows],
            "open": [r[1] for r in rows],
            "close": [r[2] for r in rows],
            "high": [r[3] for r in rows],
            "low": [r[4] for r in rows],
            "volume": [r[5] for r in rows],
            "amount": [r[8] if len(r) > 8 else None for r in rows],
        }
    )
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce") * _TENCENT_VOLUME_TO_SHARES
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce") * _TENCENT_AMOUNT_TO_YUAN
    return df


def _symbol_with_exchange(code: str) -> str:
    """裸 6 位代码 → 契约后缀形式：6 开头 ``.SH``，0/3 开头 ``.SZ``。"""
    code = str(code).strip()
    if not re.fullmatch(r"\d{6}", code):
        raise InvalidSymbolError(f"无效股票代码：{code!r}")
    if code[0] == "6":
        return f"{code}.SH"
    if code[0] in ("0", "3"):
        return f"{code}.SZ"
    raise InvalidSymbolError(f"不支持的交易所代码：{code!r}（仅支持沪 6 开头 / 深 0、3 开头）")


def _code_to_tx_symbol(code: str) -> str:
    """裸 6 位代码 → 腾讯代码格式：6 开头 ``sh600519``，0/3 开头 ``sz000001``。"""
    code = str(code).strip()
    if not re.fullmatch(r"\d{6}", code):
        raise InvalidSymbolError(f"无效股票代码：{code!r}")
    if code[0] == "6":
        return f"sh{code}"
    if code[0] in ("0", "3"):
        return f"sz{code}"
    raise InvalidSymbolError(f"不支持的交易所代码：{code!r}")


def _symbol_with_exchange_safe(code: str) -> str | None:
    """Like ``_symbol_with_exchange`` but returns ``None`` for unsupported codes.

    Used when scanning a full-market snapshot that may contain 北交所 or other
    non-SH/SZ securities we do not support.
    """
    code = str(code).strip()
    if not re.fullmatch(r"\d{6}", code):
        return None
    if code[0] == "6":
        return f"{code}.SH"
    if code[0] in ("0", "3"):
        return f"{code}.SZ"
    return None


def _tx_symbol_with_exchange_safe(code: str) -> str | None:
    """新浪/腾讯代码 ``sh600519`` → 契约后缀 ``600519.SH``（不支持则返回 None）。"""
    code = str(code).strip().lower()
    m = re.fullmatch(r"(sh|sz)(\d{6})", code)
    if not m:
        return None
    return f"{m.group(2)}.{m.group(1).upper()}"


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

    return _mark_source(df, "sample", provider="sample", fetched=False)


def _mark_source(
    df: pd.DataFrame,
    source: str,
    *,
    provider: str | None = None,
    fetched: bool = True,
) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    df.attrs["data_source"] = source
    df.attrs["is_sample"] = source == "sample"
    if provider is not None:
        df.attrs["provider"] = provider
    if fetched:
        df.attrs["fetched_at"] = _now_iso()
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


def fetch_realtime_quotes(symbols: str | Iterable[str]) -> pd.DataFrame:
    """Fetch latest realtime snapshot quotes for one or more A-share symbols.

    Tries eastmoney ``stock_zh_a_spot_em`` (full-market snapshot) first, then
    falls back to sina ``stock_zh_a_spot``. Filters to the requested symbols and
    returns one row per symbol with: ``price`` (最新价), ``change`` (涨跌额),
    ``pct_change`` (涨跌幅), ``prev_close`` (昨收), ``open`` (今开), ``high``,
    ``low``, ``volume`` (股), ``amount`` (元) and ``timestamp`` (供应商时间戳；
    仅新浪提供，东方财富快照为空).

    Realtime quotes may be delayed and are not guaranteed trade-grade. Never
    writes a local file. Provenance is recorded in ``df.attrs``
    (``data_source`` / ``provider`` / ``fetched_at``).

    Raises:
        InvalidSymbolError: malformed or unsupported symbol.
        NoDataError: every provider failed or could not find the symbols.
    """
    symbols = _normalize_symbols(symbols)
    for symbol in symbols:
        _validate_symbol(symbol)

    attempts = [
        ("akshare_eastmoney", "eastmoney", _fetch_realtime_eastmoney),
        ("akshare_sina", "sina", _fetch_realtime_sina),
    ]
    errors: list[str] = []
    for source, provider, fetcher in attempts:
        try:
            df = fetcher(symbols)
        except Exception as exc:  # noqa: BLE001 - provider/network errors
            logger.warning("实时行情源 %s 失败：%s", provider, exc)
            errors.append(f"{provider}：{exc}")
            continue
        if df is not None and not df.empty:
            return _mark_source(df, source, provider=provider)
        errors.append(f"{provider}：未找到 {symbols}")

    raise NoDataError(f"未能获取 {symbols} 的实时行情（已尝试 {'；'.join(errors)}）")


def _fetch_realtime_eastmoney(symbols: list[str]) -> pd.DataFrame | None:
    import akshare as ak

    raw = ak.stock_zh_a_spot_em()
    if raw is None or raw.empty:
        return None
    return _convert_realtime_em(raw, symbols)


def _fetch_realtime_sina(symbols: list[str]) -> pd.DataFrame | None:
    import akshare as ak

    raw = ak.stock_zh_a_spot()
    if raw is None or raw.empty:
        return None
    return _convert_realtime_sina(raw, symbols)


def _convert_realtime_em(raw: pd.DataFrame, symbols: list[str]) -> pd.DataFrame | None:
    df = raw.rename(columns=_REALTIME_EM_COLUMN_MAP)
    df["symbol"] = df["symbol"].map(_symbol_with_exchange_safe)
    df = df[df["symbol"].isin(symbols)].copy()
    if df.empty:
        return None
    # 东方财富成交量单位为“手”，换算为股；成交额已是元。
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce") * _AKSHARE_LOT_TO_SHARES
    for col in _REALTIME_NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["timestamp"] = None  # 东方财富快照不提供时间戳
    return df[list(_REALTIME_COLUMNS)].reset_index(drop=True)


def _convert_realtime_sina(raw: pd.DataFrame, symbols: list[str]) -> pd.DataFrame | None:
    df = raw.rename(columns=_REALTIME_SINA_COLUMN_MAP)
    df["symbol"] = df["symbol"].map(_tx_symbol_with_exchange_safe)
    df = df[df["symbol"].isin(symbols)].copy()
    if df.empty:
        return None
    # 新浪成交量已是股、成交额已是元，不换算。
    for col in _REALTIME_NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["timestamp"] = df["timestamp"].astype(str)
    return df[list(_REALTIME_COLUMNS)].reset_index(drop=True)
