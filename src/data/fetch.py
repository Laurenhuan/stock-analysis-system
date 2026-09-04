"""Market data acquisition for Role 2 (金融数据工程师).

Fetch A-share daily history and realtime quotes online on demand. Daily history
goes through AkShare with a multi-source fallback so eastmoney being
intermittently unavailable does not break queries:

- primary  : AkShare ``stock_zh_a_hist`` (eastmoney, 前复权 ``qfq``, 10s timeout,
             2 retries with backoff)
- fallback : AkShare ``stock_zh_a_hist_tx`` (Tencent, 前复权 ``qfq``). We use the
             official AkShare wrapper instead of maintaining Tencent's private
             ``newfqkline`` protocol directly.

Realtime quotes are exposed separately via ``fetch_realtime_quotes``
(eastmoney ``stock_zh_a_spot_em`` → sina ``stock_zh_a_spot``). Online results are
never written to a local CSV; only the fixed Sample fallback reads the committed
``data/sample`` file.

When several symbols are requested, each provider may return only a subset. The
fetchers therefore compare the actually-returned symbol set against the request,
call the next provider only for the missing symbols, merge the results, and raise
``NoDataError`` listing any still-missing codes — a partial result is never
reported as complete success.

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
                         | ``"akshare_tencent"`` | ``"akshare_mixed"`` (daily), or
                         ``"akshare_eastmoney"`` | ``"akshare_sina"`` |
                         ``"akshare_mixed"`` (realtime)
- ``provider``        -> ``"sample"`` | ``"tushare"`` | ``"eastmoney"`` |
                         ``"tencent"`` | ``"sina"``, or ``"eastmoney+tencent"`` /
                         ``"eastmoney+sina"`` when a request was merged across
                         providers.
- ``is_sample``       -> ``True`` | ``False``
- ``fetched_at``      -> ISO-8601 UTC timestamp of the fetch (not set for Sample)
- ``fallback_reason`` -> only set when ``source="auto"`` fell back to sample.

.. note::
    AkShare ``stock_zh_a_hist_tx`` returns ``volume`` (股) and ``amount`` (元)
    directly. One known upstream quirk is corrected here: in akshare 1.18.94 the
    function skips the 手→股 ×100 for ``sz000``-prefixed codes (it misclassifies
    them as indices), so 深市主板 ``000xxx`` volume is left in 手. Reproducible:
    ``stock_zh_a_hist_tx("sz000001", "2024-01-01", "2024-01-05", "qfq")`` returns
    volume ``1158366`` (手) while the authoritative value is ``115836645`` (股).
    ``_convert_tencent`` therefore multiplies 000xxx volume by 100.
"""

from __future__ import annotations

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

# 网络 / 连接 / 超时 / HTTP 类错误。只有这些错误允许重试或跨 Provider 降级；字段
# 映射、JSON 结构、程序错误（KeyError / TypeError / DataValidationError 等）不得
# 被静默吞掉，必须向上抛出。
# 说明：requests 的所有异常（RequestException/HTTPError/Timeout/ConnectionError）
# 均为 OSError 的子类，故 OSError 已覆盖连接、超时与 HTTP 失败。
_RETRYABLE_EXCEPTIONS = (ConnectionError, TimeoutError, OSError)

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

# 官方 stock_zh_a_hist_tx 的列名映射（volume 已是股、amount 已是元，turnover 丢弃）。
_TX_COLUMN_MAP = {
    "date": "trade_date",
    "open": "open",
    "close": "close",
    "high": "high",
    "low": "low",
    "volume": "volume",
    "amount": "amount",
}

_EASTMONEY_TIMEOUT = 10.0
_EASTMONEY_MAX_RETRIES = 2

_TENCENT_TIMEOUT = 10.0
_TENCENT_MAX_RETRIES = 1
# 未传起始日期时，官方 stock_zh_a_hist_tx 按年逐段请求；回看窗口过大会导致请求量
# 膨胀。这里限制默认回看年数，作为请求数量保护。
_TENCENT_DEFAULT_LOOKBACK_YEARS = 2

# provider 名 → daily/realtime ``data_source`` 取值。
_AKSHARE_SOURCE_BY_PROVIDER = {
    "eastmoney": "akshare_eastmoney",
    "tencent": "akshare_tencent",
    "sina": "akshare_sina",
}

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

    symbols = _normalize_symbols(symbols)  # InvalidSymbolError propagates here

    token = token or os.getenv("TUSHARE_TOKEN")
    start = _to_yyyymmdd(start_date)
    end = _to_yyyymmdd(end_date)

    if start and end and start > end:
        raise DataValidationError(f"开始日期 {start} 不能晚于结束日期 {end}")

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
    """把单个/多个证券代码统一成规范形式 ``600519.SH`` / ``000001.SZ``。"""
    if isinstance(symbols, str):
        symbols = [symbols]
    result = [_normalize_symbol(s) for s in symbols]
    if not result:
        raise InvalidSymbolError("至少需要提供一个证券代码")
    return result


def _normalize_symbol(symbol: str) -> str:
    """标准化单个证券代码 → 规范形式 ``600519.SH`` / ``000001.SZ``。

    接受裸 6 位代码（``600519``）、带交易所后缀（``600519.SH``，大小写不敏感）、
    腾讯/新浪前缀（``sh600519``）。市场由 6 位代码首数字判定（6→沪、0/3→深），
    不支持的市场（北交所 4/8/92 等）抛 ``InvalidSymbolError``，绝不猜测。
    """
    if not isinstance(symbol, str):
        raise InvalidSymbolError(f"证券代码必须是字符串，收到 {symbol!r}")
    s = symbol.strip()

    # 腾讯/新浪前缀：sh600519 / sz000001（大小写不敏感）
    m = re.fullmatch(r"(?i)(sh|sz)(\d{6})", s)
    if m:
        code, affix = m.group(2), m.group(1).lower()
    else:
        # 带后缀：600519.SH / 000001.sz
        m = re.fullmatch(r"(?i)(\d{6})\.(sh|sz)", s)
        if m:
            code, affix = m.group(1), m.group(2).lower()
        elif re.fullmatch(r"\d{6}", s):
            return _symbol_with_exchange(s)  # 裸 6 位代码
        else:
            raise InvalidSymbolError(
                f"无效证券代码：{symbol!r}（应为 6 位数字，可带 .SH/.SZ 后缀或 sh/sz 前缀，"
                f"如 600519 或 600519.SH）"
            )

    canonical = _symbol_with_exchange(code)  # 按代码首数字判市场，不支持抛错
    expected = "sh" if canonical.endswith(".SH") else "sz"
    if affix != expected:
        raise InvalidSymbolError(
            f"证券代码与交易所不匹配：{symbol!r}（{code} 属于 {expected.upper()}）"
        )
    return canonical


def _to_yyyymmdd(value) -> str:
    if value is None:
        return ""
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y%m%d")
    if isinstance(value, (datetime, date)):
        return value.strftime("%Y%m%d")
    return str(value).strip().replace("-", "").replace("/", "")


def _to_dashed(value: str) -> str:
    """``"YYYYMMDD"`` → ``"YYYY-MM-DD"``（腾讯官方函数要求带连字符的日期格式）。"""
    if not value:
        return ""
    if len(value) == 8 and value.isdigit():
        return f"{value[:4]}-{value[4:6]}-{value[6:8]}"
    return value


def _default_start(end: str = "") -> str:
    """无起始日期时的默认回看起点（限制腾讯按年请求的次数）。

    默认起点按 ``end`` 的年份回看：传了历史 ``end_date`` 时按该年份回看，避免出现
    ``start > end`` 的越界查询；未传 ``end`` 时才按当前年份回看。
    """
    end_year = datetime.now().year
    if end and len(end) >= 4 and end[:4].isdigit():
        end_year = int(end[:4])
    return f"{end_year - _TENCENT_DEFAULT_LOOKBACK_YEARS}-01-01"


def _default_end() -> str:
    return f"{datetime.now().year}-12-31"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _retry(func: Callable, *, max_retries: int = 2, backoff: float = 1.0):
    """Call ``func`` up to ``1 + max_retries`` times with exponential backoff.

    Only network / connection / timeout / HTTP errors (``_RETRYABLE_EXCEPTIONS``)
    are retried. Field-mapping, JSON-structure and programming errors propagate
    immediately rather than being swallowed.
    """
    last: BaseException | None = None
    for attempt in range(max_retries + 1):
        try:
            return func()
        except _RETRYABLE_EXCEPTIONS as exc:
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
    except _RETRYABLE_EXCEPTIONS as exc:
        raise NoDataError(f"AkShare 获取失败：{exc}") from exc


def _fetch_akshare(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    """Fetch daily history via AkShare with eastmoney→tencent fallback + merge.

    Each provider is called for the symbols still missing after the previous one;
    results are merged. Returns contract-shaped data (``symbol`` 带 .SH/.SZ 后缀、
    ``volume`` 股、``amount`` 元、``trade_date`` 字符串)。若最终仍有缺失代码，抛出
    ``NoDataError`` 并明确列出缺失代码——部分结果绝不当作全部成功。
    """
    remaining = list(symbols)
    frames: list[pd.DataFrame] = []
    used_providers: list[str] = []
    tried_providers: list[str] = []
    attempts = [
        ("eastmoney", _fetch_eastmoney),
        ("tencent", _fetch_tencent),
    ]

    for provider, fetcher in attempts:
        if not remaining:
            break
        tried_providers.append(provider)
        try:
            raw = fetcher(remaining, start, end)
        except _RETRYABLE_EXCEPTIONS as exc:
            logger.warning("AkShare 数据源 %s 失败：%s", provider, exc)
            continue
        if raw is None or raw.empty:
            continue
        # 先按请求股票 + 日期区间过滤并去重，再据此判断该 Provider 真正拿到了哪些
        # 股票。若 Provider 返回了代码但所有记录都在区间外，此处会得到空表并跳过，
        # 从而继续调用下一 Provider，而不是把 0 行结果误判为成功。
        df = _finalize_daily(raw, start, end, symbols=remaining)
        if df.empty:
            continue
        frames.append(df)
        used_providers.append(provider)
        got = {s for s in df["symbol"].unique()}
        remaining = [s for s in remaining if s not in got]

    if remaining:
        raise NoDataError(
            f"AkShare 未能获取 {'、'.join(sorted(remaining))} 的行情数据"
            f"（已尝试 {'、'.join(tried_providers)}）"
        )

    merged = pd.concat(frames, ignore_index=True)
    merged = _finalize_daily(merged, start, end, symbols=symbols)
    # 最终合并后再次核对全部请求股票，防止部分结果被当作全部成功。
    got = {s for s in merged["symbol"].unique()}
    missing = [s for s in symbols if s not in got]
    if missing:
        raise NoDataError(
            f"AkShare 未能获取 {'、'.join(sorted(missing))} 的行情数据"
            f"（已尝试 {'、'.join(tried_providers)}）"
        )
    return _mark_merged_source(merged, used_providers)


def _fetch_eastmoney(symbols: list[str], start: str, end: str) -> pd.DataFrame | None:
    """Daily history via AkShare ``stock_zh_a_hist`` (eastmoney, 前复权 ``qfq``).

    10s timeout，最多重试 2 次（指数退避）。逐股票抓取，单只股票网络失败只跳过该股
    并记录日志，不影响其余股票；全部失败时返回 ``None``。
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
        try:
            raw = _retry(
                lambda: ak.stock_zh_a_hist(**kwargs),
                max_retries=_EASTMONEY_MAX_RETRIES,
            )
        except _RETRYABLE_EXCEPTIONS as exc:
            logger.warning("eastmoney 获取 %s 失败：%s", symbol, exc)
            continue
        if raw is None or raw.empty:
            continue
        frames.append(_convert_akshare(raw))
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _fetch_tencent(symbols: list[str], start: str, end: str) -> pd.DataFrame | None:
    """Daily history via official AkShare ``stock_zh_a_hist_tx`` (Tencent, 前复权).

    使用官方封装，不直接维护腾讯 ``newfqkline`` 私有协议。官方函数返回
    ``volume`` 已统一为股、``amount`` 已统一为元，并已按年份去重/排序/过滤。逐
    股票抓取，单只失败只跳过该股；全部失败返回 ``None``。
    """
    import akshare as ak

    start_arg = _to_dashed(start) if start else _default_start(end)
    end_arg = _to_dashed(end) if end else _default_end()

    frames = []
    for symbol in symbols:
        tx_symbol = _code_to_tx_symbol(symbol[:6])
        try:
            raw = _retry(
                lambda: ak.stock_zh_a_hist_tx(
                    symbol=tx_symbol,
                    start_date=start_arg,
                    end_date=end_arg,
                    adjust="qfq",
                    timeout=_TENCENT_TIMEOUT,
                ),
                max_retries=_TENCENT_MAX_RETRIES,
            )
        except _RETRYABLE_EXCEPTIONS as exc:
            logger.warning("tencent 获取 %s 失败：%s", symbol, exc)
            continue
        if raw is None or raw.empty:
            continue
        frames.append(_convert_tencent(raw, symbol))
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _finalize_daily(
    df: pd.DataFrame, start: str, end: str, symbols: list[str] | None = None
) -> pd.DataFrame:
    """按请求股票过滤 + (symbol, trade_date) 去重 + 过滤到 [start, end] + 排序。"""
    if symbols is not None:
        df = df[df["symbol"].isin(symbols)].copy()
    df = df.drop_duplicates(subset=["symbol", "trade_date"]).copy()
    dates = pd.to_datetime(df["trade_date"], errors="coerce")
    mask = pd.Series(True, index=df.index)
    if start:
        mask &= dates >= pd.Timestamp(start)
    if end:
        mask &= dates <= pd.Timestamp(end)
    df = df[mask].sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    return df[list(BASE_MARKET_COLUMNS)]


def _convert_akshare(raw: pd.DataFrame) -> pd.DataFrame:
    """AkShare 输出 → 标准 Schema：中文列名→英文、补 .SH/.SZ、成交量 手→股 ×100。"""
    df = raw.rename(columns=_AKSHARE_COLUMN_MAP)
    df = df[list(BASE_MARKET_COLUMNS)].copy()
    df["symbol"] = df["symbol"].map(_symbol_with_exchange)
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce") * _AKSHARE_LOT_TO_SHARES
    return df


def _convert_tencent(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """官方 ``stock_zh_a_hist_tx`` 输出 → 标准 Schema。

    官方已把 ``volume`` 统一为股、``amount`` 统一为元（``turnover`` 丢弃）。这里只
    做列名映射 + 补 ``symbol`` 列；并修正 akshare 1.18.94 对深市主板 ``sz000`` 前
    缀未 ×100 的 bug（000xxx 成交量仍是手，见模块 note）。
    """
    df = raw.rename(columns=_TX_COLUMN_MAP)
    df["symbol"] = symbol
    df = df[list(BASE_MARKET_COLUMNS)].copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.strftime(
        "%Y-%m-%d"
    )
    for col in ("open", "high", "low", "close", "volume", "amount"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if symbol.startswith("000"):
        # 深市主板（sz000xxx）：官方函数未 ×100，此处按 Contract（volume=股）补正。
        df["volume"] = df["volume"] * _AKSHARE_LOT_TO_SHARES
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


# 全市场沪深 A 股代码/名称表（进程内内存缓存，避免搜索框每次敲键都重拉全市场）。
_stock_universe_cache: pd.DataFrame | None = None


def _get_stock_universe() -> pd.DataFrame:
    """在线拉取全市场沪深 A 股代码/名称表，列 ``code, symbol, name, market``。

    数据源 AkShare ``stock_info_a_code_name``（东财）。只保留沪（6）/深（0、3），
    北交所（4/8/92）等不支持的市场被剔除。网络失败抛 ``NoDataError``；不写本地文件。
    """
    global _stock_universe_cache
    if _stock_universe_cache is not None:
        return _stock_universe_cache

    import akshare as ak

    try:
        raw = _retry(ak.stock_info_a_code_name, max_retries=1)
    except _RETRYABLE_EXCEPTIONS as exc:
        raise NoDataError(f"股票代码表获取失败：{exc}") from exc
    if raw is None or raw.empty or not {"code", "name"}.issubset(raw.columns):
        raise NoDataError("股票代码表为空或缺少 code/name 列")

    df = pd.DataFrame({
        "code": raw["code"].astype(str).str.strip(),
        "name": raw["name"].astype(str).str.strip(),
    })
    df["symbol"] = df["code"].map(_symbol_with_exchange_safe)  # 北交所等→None
    df = df.dropna(subset=["symbol"])
    df["market"] = df["symbol"].str[-2:]
    _stock_universe_cache = df[["code", "symbol", "name", "market"]].reset_index(drop=True)
    return _stock_universe_cache


def search_stock_symbols(query: str, *, limit: int = 20) -> pd.DataFrame:
    """按代码或名称模糊搜索沪深 A 股，返回 ``symbol, name, market``。

    Args:
        query: 搜索关键词；支持股票代码（``600519`` / ``600519.SH`` / ``sh600519``，
            提取其中数字做前缀模糊）或中文名称（``茅台``，包含匹配）。
        limit: 返回条数上限，自动收敛到 ``[1, 200]``。

    Returns:
        DataFrame，列 ``symbol``（规范形式 ``600519.SH``）、``name``、
        ``market``（"SH"|"SZ"）。结果去重、按 symbol 稳定升序排序；无匹配返回空表
        （不抛异常，便于前端把空结果直接渲染成“未找到”）。

    Raises:
        NoDataError: 在线股票代码表获取失败。
    """
    limit = max(1, min(int(limit), 200))
    universe = _get_stock_universe()

    q = str(query).strip()
    if not q:
        return pd.DataFrame(columns=["symbol", "name", "market"])

    q_digits = "".join(ch for ch in q if ch.isdigit())
    mask = pd.Series(False, index=universe.index)
    if q_digits:
        mask |= universe["code"].str.startswith(q_digits)
    mask |= universe["name"].str.contains(q, regex=False, na=False)

    result = (
        universe[mask]
        .drop_duplicates(subset=["symbol"])
        .sort_values("symbol", kind="stable")
        .head(limit)
    )
    return result[["symbol", "name", "market"]].reset_index(drop=True)


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


def _mark_merged_source(df: pd.DataFrame, used_providers: list[str]) -> pd.DataFrame:
    """Set provenance for a (possibly cross-provider) AkShare result.

    Single provider → its ``data_source``/``provider``; merged across providers →
    ``"akshare_mixed"`` and a ``"+"``-joined provider label.
    """
    if len(used_providers) == 1:
        provider = used_providers[0]
        return _mark_source(df, _AKSHARE_SOURCE_BY_PROVIDER[provider], provider=provider)
    return _mark_source(df, "akshare_mixed", provider="+".join(used_providers))


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
    falls back to sina ``stock_zh_a_spot`` for the symbols eastmoney did not
    return. Returns one row per symbol with: ``price`` (最新价), ``change``
    (涨跌额), ``pct_change`` (涨跌幅, 单位 %，即东财“涨跌幅”原始值), ``prev_close``
    (昨收), ``open`` (今开), ``high``, ``low``, ``volume`` (股), ``amount`` (元)
    and ``timestamp`` (供应商时间戳；仅新浪提供，东方财富快照为空).

    Realtime quotes may be delayed and are not guaranteed trade-grade. Never
    writes a local file. Provenance is recorded in ``df.attrs``
    (``data_source`` / ``provider`` / ``fetched_at``). 若最终仍有缺失代码，抛出
    ``NoDataError`` 并明确列出缺失代码。

    Raises:
        InvalidSymbolError: malformed or unsupported symbol.
        NoDataError: every provider failed or could not find the symbols.
    """
    symbols = _normalize_symbols(symbols)  # InvalidSymbolError propagates here

    remaining = list(symbols)
    frames: list[pd.DataFrame] = []
    used_providers: list[str] = []
    tried_providers: list[str] = []
    attempts = [
        ("eastmoney", _fetch_realtime_eastmoney),
        ("sina", _fetch_realtime_sina),
    ]

    for provider, fetcher in attempts:
        if not remaining:
            break
        tried_providers.append(provider)
        try:
            df = fetcher(remaining)
        except _RETRYABLE_EXCEPTIONS as exc:
            logger.warning("实时行情源 %s 失败：%s", provider, exc)
            continue
        if df is None or df.empty:
            continue
        frames.append(df)
        used_providers.append(provider)
        got = {s for s in df["symbol"].unique()}
        remaining = [s for s in remaining if s not in got]

    if remaining:
        raise NoDataError(
            f"未能获取 {'、'.join(sorted(remaining))} 的实时行情"
            f"（已尝试 {'、'.join(tried_providers)}）"
        )

    merged = pd.concat(frames, ignore_index=True)
    return _mark_merged_source(merged, used_providers)


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
