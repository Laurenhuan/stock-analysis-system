"""Tests for the Role 2 AkShare provider (``fetch_market_data(source="akshare")``).

Covers the eastmoney → tencent multi-source fallback, per-provider partial
success with cross-provider merge, unit conversion (including the 科创板/深市
主板 edge cases), and provenance. All network access is mocked — no test here
hits a real endpoint, so the suite runs offline and deterministically.

The Tencent path uses the official AkShare ``stock_zh_a_hist_tx`` wrapper rather
than Tencent's private ``newfqkline`` protocol. That function internally maps
Tencent's ``day`` / ``hfqday`` / ``qfqday`` responses to one uniform column set
(``date/open/close/high/low/volume/turnover/amount``) with ``volume`` already in
股 and ``amount`` already in 元, so ``_convert_tencent`` only needs to map columns
(plus a targeted correction for the 深市主板 ``sz000`` unit bug, see module note).
"""

import json
from datetime import datetime

import pandas as pd
import pytest

from src.contracts.market_data import BASE_MARKET_COLUMNS
from src.data.clean import _detect_units, clean_market_data
from src.data.fetch import (
    _code_to_tx_symbol,
    _convert_akshare,
    _convert_tencent,
    _fetch_tencent,
    _finalize_daily,
    _normalize_symbol,
    _symbol_with_exchange,
    fetch_market_data,
)
from src.utils.exceptions import DataValidationError, InvalidSymbolError, NoDataError


# ---------------------------------------------------------------------------
# 后缀补全
# ---------------------------------------------------------------------------

def test_symbol_with_exchange_sh_and_sz():
    assert _symbol_with_exchange("600519") == "600519.SH"
    assert _symbol_with_exchange("601318") == "601318.SH"
    assert _symbol_with_exchange("688981") == "688981.SH"  # 科创板，6 开头
    assert _symbol_with_exchange("000001") == "000001.SZ"
    assert _symbol_with_exchange("300750") == "300750.SZ"  # 创业板，3 开头
    assert _symbol_with_exchange("002594") == "002594.SZ"  # 中小板，0 开头


def test_symbol_with_exchange_rejects_invalid():
    with pytest.raises(InvalidSymbolError):
        _symbol_with_exchange("830000")  # 北交所，8 开头不支持
    with pytest.raises(InvalidSymbolError):
        _symbol_with_exchange("60051")  # 位数不足
    with pytest.raises(InvalidSymbolError):
        _symbol_with_exchange("abcdef")


def test_code_to_tx_symbol_sh_and_sz():
    assert _code_to_tx_symbol("600519") == "sh600519"
    assert _code_to_tx_symbol("601318") == "sh601318"
    assert _code_to_tx_symbol("688981") == "sh688981"
    assert _code_to_tx_symbol("000001") == "sz000001"
    assert _code_to_tx_symbol("300750") == "sz300750"


def test_code_to_tx_symbol_rejects_invalid():
    with pytest.raises(InvalidSymbolError):
        _code_to_tx_symbol("830000")  # 北交所
    with pytest.raises(InvalidSymbolError):
        _code_to_tx_symbol("60051")


# ---------------------------------------------------------------------------
# 代码标准化（裸代码 / 带后缀 / 腾讯·新浪前缀）
# ---------------------------------------------------------------------------

def test_normalize_symbol_accepts_bare_suffixed_prefixed():
    assert _normalize_symbol("600519") == "600519.SH"
    assert _normalize_symbol("600519.SH") == "600519.SH"
    assert _normalize_symbol("600519.sh") == "600519.SH"
    assert _normalize_symbol("sh600519") == "600519.SH"
    assert _normalize_symbol("SH600519") == "600519.SH"
    assert _normalize_symbol("000001") == "000001.SZ"
    assert _normalize_symbol("000001.SZ") == "000001.SZ"
    assert _normalize_symbol("000001.sz") == "000001.SZ"
    assert _normalize_symbol("sz000001") == "000001.SZ"
    assert _normalize_symbol("300750") == "300750.SZ"   # 创业板
    assert _normalize_symbol("688981") == "688981.SH"   # 科创板


def test_normalize_symbol_rejects_invalid():
    for bad in (
        "", "bad", "60051", "12345", "6005196",      # 位数/字符非法
        "600519.SS", "600519.XSHG",                   # 未知后缀
        "800001", "830000", "920001",                 # 北交所等不支持市场
        "000001.SH", "sz600519",                      # 前后缀与市场不匹配
    ):
        with pytest.raises(InvalidSymbolError):
            _normalize_symbol(bad)


# ---------------------------------------------------------------------------
# 列名映射 + 单位转换（eastmoney）
# ---------------------------------------------------------------------------

def _akshare_raw_frame(code: str = "600519") -> pd.DataFrame:
    """AkShare ``stock_zh_a_hist`` 的真实输出形状（中文列名）。"""
    return pd.DataFrame({
        "日期": ["2024-01-02", "2024-01-03"],
        "股票代码": [code, code],
        "开盘": [1580.66, 1546.77],
        "收盘": [1550.67, 1559.66],
        "最高": [1583.85, 1560.88],
        "最低": [1543.76, 1541.99],
        "成交量": [32156, 20229],          # 单位：手
        "成交额": [5.44e9, 3.41e9],        # 单位：元
        "振幅": [2.52, 1.22],
        "涨跌幅": [-2.58, 0.58],
        "涨跌额": [-40.99, 8.99],
        "换手率": [0.26, 0.16],
    })


def test_convert_akshare_maps_columns_and_units():
    out = _convert_akshare(_akshare_raw_frame())
    assert list(out.columns) == list(BASE_MARKET_COLUMNS)
    assert (out["symbol"] == "600519.SH").all()
    # 成交量 手 -> 股 ×100
    assert out["volume"].tolist() == [3215600, 2022900]
    # 成交额已经是元，保持不变（不得像 Tushare 那样再 ×1000）
    assert out["amount"].tolist() == pytest.approx([5.44e9, 3.41e9])
    # 正确映射 OHLC：AkShare 顺序是 开盘,收盘,最高,最低
    assert out["close"].tolist() == pytest.approx([1550.67, 1559.66])
    assert out["high"].tolist() == pytest.approx([1583.85, 1560.88])
    assert out["low"].tolist() == pytest.approx([1543.76, 1541.99])


def test_convert_akshare_amount_not_scaled_by_1000():
    """AkShare 与 Tushare 的关键差异：成交额单位是元，不得再乘 1000。"""
    out = _convert_akshare(_akshare_raw_frame())
    assert out["amount"].iloc[0] == pytest.approx(5.44e9)


# ---------------------------------------------------------------------------
# 腾讯：官方 stock_zh_a_hist_tx 输出 → 列名 + 单位映射
# ---------------------------------------------------------------------------

def _tx_raw(tx_symbol: str = "sh600519") -> pd.DataFrame:
    """官方 ``stock_zh_a_hist_tx`` 输出形状。

    ``volume`` 已被官方统一为股、``amount`` 已是元；唯一例外是官方对 ``sz000``
    前缀的 bug（volume 仍是手），这里按真实情况模拟，供补正测试使用。
    """
    if tx_symbol == "sz000001":
        volume = [1158366.0, 733610.0]        # 手（官方 sz000 bug）
    elif tx_symbol.startswith("sh688"):
        volume = [13459147.0, 12847542.0]     # 股（科创板，官方不 ×100）
    else:
        volume = [3215600.0, 2022900.0]       # 股（官方已 ×100）
    return pd.DataFrame({
        "date": ["2024-01-02", "2024-01-03"],
        "open": [1580.66, 1546.77],
        "close": [1550.67, 1559.66],
        "high": [1583.85, 1560.88],
        "low": [1543.76, 1541.99],
        "volume": volume,
        "turnover": [0.26, 0.16],
        "amount": [5440082500.0, 3411400700.0],  # 元
    })


def test_convert_tencent_maps_columns_and_units():
    """主板上官方已把 volume 统一为股、amount 统一为元，直接映射不再二次换算。"""
    out = _convert_tencent(_tx_raw("sh600519"), "600519.SH")
    assert list(out.columns) == list(BASE_MARKET_COLUMNS)
    assert (out["symbol"] == "600519.SH").all()
    assert out["trade_date"].tolist() == ["2024-01-02", "2024-01-03"]
    assert out["open"].tolist() == pytest.approx([1580.66, 1546.77])
    assert out["close"].tolist() == pytest.approx([1550.67, 1559.66])
    assert out["high"].tolist() == pytest.approx([1583.85, 1560.88])
    assert out["low"].tolist() == pytest.approx([1543.76, 1541.99])
    # 官方已 ×100，这里不得再 ×100
    assert out["volume"].tolist() == pytest.approx([3215600.0, 2022900.0])
    # 官方已 ×10000（万元→元），这里不得再换算
    assert out["amount"].tolist() == pytest.approx([5440082500.0, 3411400700.0])


def test_convert_tencent_star_board_volume_is_shares():
    """科创板 688981：官方不 ×100（volume 本就是股），这里也不得 ×100。

    对应 Tencent 对科创板返回 ``day``（未复权）而非 ``qfqday`` 的情形；官方函数
    内部把 ``day``/``hfqday``/``qfqday`` 统一成同一列集，故结果形状与 qfq 一致。
    """
    out = _convert_tencent(_tx_raw("sh688981"), "688981.SH")
    assert (out["symbol"] == "688981.SH").all()
    assert out["volume"].tolist() == pytest.approx([13459147.0, 12847542.0])


def test_convert_tencent_sz000_fixes_lot_bug():
    """深市主板 000001：官方 sz000 前缀 bug 未 ×100，这里补正 手→股 ×100。"""
    out = _convert_tencent(_tx_raw("sz000001"), "000001.SZ")
    assert (out["symbol"] == "000001.SZ").all()
    assert out["volume"].tolist() == pytest.approx([115836600.0, 73361000.0])


def _no_sleep(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *a, **k: None)


# ---------------------------------------------------------------------------
# 抓取（mock 网络）：eastmoney 成功 / 回退腾讯 / 部分成功合并 / 双双失败
# ---------------------------------------------------------------------------

def test_fetch_akshare_calls_stock_zh_a_hist_with_qfq_and_timeout(monkeypatch):
    import akshare

    captured = {}

    def fake_stock_zh_a_hist(**kwargs):
        captured.update(kwargs)
        return _akshare_raw_frame()

    monkeypatch.setattr(akshare, "stock_zh_a_hist", fake_stock_zh_a_hist)

    df = fetch_market_data("600519.SH", source="akshare")
    # 传给 AkShare 的参数必须是裸 6 位代码 + daily + qfq + timeout
    assert captured["symbol"] == "600519"
    assert captured["period"] == "daily"
    assert captured["adjust"] == "qfq"
    assert captured["timeout"] == 10.0
    # 返回标准 Schema + 来源标记
    assert list(df.columns) == list(BASE_MARKET_COLUMNS)
    assert (df["symbol"] == "600519.SH").all()
    assert df["volume"].tolist() == [3215600, 2022900]
    assert df.attrs["data_source"] == "akshare_eastmoney"
    assert df.attrs["provider"] == "eastmoney"
    assert df.attrs["is_sample"] is False
    assert "fetched_at" in df.attrs


def test_fetch_akshare_passes_dates_and_multiple_symbols(monkeypatch):
    import akshare

    calls = []

    def fake_stock_zh_a_hist(**kwargs):
        calls.append(kwargs)
        return _akshare_raw_frame(kwargs["symbol"])

    monkeypatch.setattr(akshare, "stock_zh_a_hist", fake_stock_zh_a_hist)

    df = fetch_market_data(
        ["600519.SH", "000001.SZ"],
        start_date="20240102",
        end_date="20241231",
        source="akshare",
    )
    assert len(calls) == 2
    assert calls[0]["start_date"] == "20240102"
    assert calls[0]["end_date"] == "20241231"
    assert calls[1]["symbol"] == "000001"
    # 后缀各自正确
    assert set(df["symbol"].unique()) == {"600519.SH", "000001.SZ"}
    assert df.attrs["data_source"] == "akshare_eastmoney"


def test_fetch_akshare_falls_back_to_tencent(monkeypatch):
    import akshare

    def em_down(**kwargs):
        raise OSError("Connection reset by peer")

    monkeypatch.setattr(akshare, "stock_zh_a_hist", em_down)
    _no_sleep(monkeypatch)

    captured = {}

    def fake_tx(**kwargs):
        captured.update(kwargs)
        return _tx_raw(kwargs["symbol"])

    monkeypatch.setattr(akshare, "stock_zh_a_hist_tx", fake_tx)

    df = fetch_market_data(
        "600519.SH", start_date="20240102", end_date="20240105", source="akshare"
    )
    # 官方函数收到腾讯代码格式 sh600519 + qfq + 连字符日期 + timeout
    assert captured["symbol"] == "sh600519"
    assert captured["adjust"] == "qfq"
    assert captured["start_date"] == "2024-01-02"
    assert captured["end_date"] == "2024-01-05"
    assert captured["timeout"] == 10.0
    assert df.attrs["data_source"] == "akshare_tencent"
    assert df.attrs["provider"] == "tencent"
    assert df.attrs["is_sample"] is False
    assert (df["symbol"] == "600519.SH").all()
    assert df["volume"].tolist() == pytest.approx([3215600.0, 2022900.0])
    assert "fetched_at" in df.attrs


def test_fetch_akshare_partial_success_merges_across_providers(monkeypatch):
    """多股票部分成功：eastmoney 返回部分，tencent 只补缺失，最终合并。

    eastmoney 只返回 600519.SH；tencent 补齐 000001.SZ 与 300750.SZ。
    """
    import akshare

    def em_partial(**kwargs):
        code = kwargs["symbol"]
        if code == "600519":
            return _akshare_raw_frame(code)
        return pd.DataFrame()

    def tx_partial(**kwargs):
        if kwargs["symbol"] in ("sz000001", "sz300750"):
            return _tx_raw(kwargs["symbol"])
        return pd.DataFrame()

    monkeypatch.setattr(akshare, "stock_zh_a_hist", em_partial)
    monkeypatch.setattr(akshare, "stock_zh_a_hist_tx", tx_partial)
    _no_sleep(monkeypatch)

    df = fetch_market_data(
        ["600519.SH", "000001.SZ", "300750.SZ"],
        start_date="20240102",
        end_date="20240105",
        source="akshare",
    )
    assert set(df["symbol"].unique()) == {"600519.SH", "000001.SZ", "300750.SZ"}
    assert df.attrs["data_source"] == "akshare_mixed"
    assert df.attrs["provider"] == "eastmoney+tencent"


def test_fetch_akshare_reports_missing_symbols(monkeypatch):
    """两个 Provider 都缺某只股票时，必须明确报告缺失代码而非当作全部成功。"""
    import akshare

    def em_partial(**kwargs):
        code = kwargs["symbol"]
        return _akshare_raw_frame(code) if code == "600519" else pd.DataFrame()

    def tx_partial(**kwargs):
        return _tx_raw(kwargs["symbol"]) if kwargs["symbol"] == "sz000001" else pd.DataFrame()

    monkeypatch.setattr(akshare, "stock_zh_a_hist", em_partial)
    monkeypatch.setattr(akshare, "stock_zh_a_hist_tx", tx_partial)
    _no_sleep(monkeypatch)

    with pytest.raises(NoDataError) as excinfo:
        fetch_market_data(
            ["600519.SH", "000001.SZ", "300750.SZ"],
            start_date="20240102",
            end_date="20240105",
            source="akshare",
        )
    msg = str(excinfo.value)
    # 明确列出缺失代码 300750.SZ，而不是把部分结果当全部成功
    assert "300750.SZ" in msg


def test_fetch_akshare_eastmoney_out_of_range_still_calls_tencent(monkeypatch):
    """eastmoney 返回了代码但记录都在请求区间外时，必须继续调用腾讯补齐。"""
    import akshare

    tx_called = []

    def em_out_of_range(**kwargs):
        raw = _akshare_raw_frame(kwargs["symbol"])
        raw["日期"] = ["2023-01-02", "2023-01-03"]  # 全部在请求区间外
        return raw

    def fake_tx(**kwargs):
        tx_called.append(kwargs)
        return _tx_raw(kwargs["symbol"])

    monkeypatch.setattr(akshare, "stock_zh_a_hist", em_out_of_range)
    monkeypatch.setattr(akshare, "stock_zh_a_hist_tx", fake_tx)
    _no_sleep(monkeypatch)

    df = fetch_market_data(
        "600519.SH", start_date="20240102", end_date="20240105", source="akshare"
    )

    # 关键：eastmoney 区间外结果不能当作已成功，必须继续调用腾讯
    assert len(tx_called) == 1
    assert tx_called[0]["symbol"] == "sh600519"
    assert df.attrs["data_source"] == "akshare_tencent"
    assert (df["symbol"] == "600519.SH").all()


def test_fetch_akshare_both_out_of_range_raises_no_data(monkeypatch):
    """两个 Provider 都只有区间外数据时，抛 NoDataError 并列出缺失代码，不能返回空表。"""
    import akshare

    def em_out_of_range(**kwargs):
        raw = _akshare_raw_frame(kwargs["symbol"])
        raw["日期"] = ["2023-01-02", "2023-01-03"]
        return raw

    def tx_out_of_range(**kwargs):
        raw = _tx_raw(kwargs["symbol"])
        raw["date"] = ["2023-01-02", "2023-01-03"]
        return raw

    monkeypatch.setattr(akshare, "stock_zh_a_hist", em_out_of_range)
    monkeypatch.setattr(akshare, "stock_zh_a_hist_tx", tx_out_of_range)
    _no_sleep(monkeypatch)

    with pytest.raises(NoDataError) as excinfo:
        fetch_market_data(
            "600519.SH", start_date="20240102", end_date="20240105", source="akshare"
        )
    # 必须列出缺失代码，而不是返回空 DataFrame
    assert "600519.SH" in str(excinfo.value)


def test_fetch_akshare_both_providers_fail_raises(monkeypatch):
    import akshare

    def em_down(**kwargs):
        raise OSError("em down")

    def tx_down(**kwargs):
        raise OSError("tx down")

    monkeypatch.setattr(akshare, "stock_zh_a_hist", em_down)
    monkeypatch.setattr(akshare, "stock_zh_a_hist_tx", tx_down)
    _no_sleep(monkeypatch)

    with pytest.raises(NoDataError) as excinfo:
        fetch_market_data("600519.SH", source="akshare")
    msg = str(excinfo.value)
    # 错误消息必须列出已尝试的两个数据源
    assert "eastmoney" in msg
    assert "tencent" in msg


def test_fetch_akshare_explicit_never_marks_sample(monkeypatch):
    """显式 akshare 源失败必须抛异常，绝不静默回退 Sample。"""
    import akshare

    monkeypatch.setattr(akshare, "stock_zh_a_hist", lambda **k: (_ for _ in ()).throw(OSError("em down")))
    monkeypatch.setattr(akshare, "stock_zh_a_hist_tx", lambda **k: (_ for _ in ()).throw(OSError("tx down")))
    _no_sleep(monkeypatch)

    with pytest.raises(NoDataError):
        fetch_market_data("600519.SH", source="akshare")


def test_fetch_akshare_strict_raises_on_empty(monkeypatch):
    import akshare

    monkeypatch.setattr(akshare, "stock_zh_a_hist", lambda **k: pd.DataFrame())
    monkeypatch.setattr(akshare, "stock_zh_a_hist_tx", lambda **k: pd.DataFrame())
    _no_sleep(monkeypatch)

    with pytest.raises(NoDataError):
        fetch_market_data("600519.SH", source="akshare")


def test_fetch_akshare_does_not_swallow_programming_error(monkeypatch):
    """字段映射 / 程序错误（非网络错误）不得被吞掉，必须向上抛出。"""
    import akshare

    def em_bad_structure(**kwargs):
        # 返回缺列的坏结构，触发映射/结构错误（KeyError / DataValidationError 之类）
        return pd.DataFrame({"wrong_column": [1, 2, 3]})

    monkeypatch.setattr(akshare, "stock_zh_a_hist", em_bad_structure)
    _no_sleep(monkeypatch)

    with pytest.raises(KeyError):
        fetch_market_data("600519.SH", source="akshare")


def test_fetch_akshare_invalid_symbol_not_swallowed(monkeypatch):
    """无效代码在抓取前就被拒绝，即使 provider 会失败也不吞掉。"""
    import akshare

    def boom(**kwargs):
        raise OSError("network down")

    monkeypatch.setattr(akshare, "stock_zh_a_hist", boom)
    with pytest.raises(InvalidSymbolError):
        fetch_market_data("800001", source="akshare")  # 北交所，不支持的市场


def test_fetch_market_data_rejects_start_after_end(monkeypatch):
    """公共入口校验 start_date <= end_date，越界时抛 DataValidationError。"""
    import akshare

    monkeypatch.setattr(akshare, "stock_zh_a_hist", lambda **k: _akshare_raw_frame())

    with pytest.raises(DataValidationError):
        fetch_market_data(
            "600519.SH", start_date="20250101", end_date="20240101", source="akshare"
        )


def test_online_fetch_never_writes_local_csv(monkeypatch):
    """在线抓取（eastmoney / tencent / realtime）不得把结果写成本地 CSV。"""
    import akshare

    def no_write(*a, **k):
        raise AssertionError("在线抓取不应写本地 CSV")

    monkeypatch.setattr("pandas.DataFrame.to_csv", no_write)
    monkeypatch.setattr(akshare, "stock_zh_a_hist", lambda **k: _akshare_raw_frame())

    df = fetch_market_data("600519.SH", source="akshare")
    assert df.attrs["data_source"] == "akshare_eastmoney"


# ---------------------------------------------------------------------------
# 腾讯：请求数量保护 / 跨年去重与日期过滤
# ---------------------------------------------------------------------------

def test_fetch_tencent_no_start_date_uses_bounded_default(monkeypatch):
    """未传开始日期时，官方函数会按年逐段请求；这里用有界默认起点保护请求数量。"""
    import akshare

    captured = {}

    def em_down(**kwargs):
        raise OSError("em down")

    def fake_tx(**kwargs):
        captured.update(kwargs)
        return _tx_raw(kwargs["symbol"])

    monkeypatch.setattr(akshare, "stock_zh_a_hist", em_down)
    monkeypatch.setattr(akshare, "stock_zh_a_hist_tx", fake_tx)
    _no_sleep(monkeypatch)

    fetch_market_data("600519.SH", source="akshare")

    # 默认回看起点为 2 年前，而不是 1900（后者会触发数十次按年请求）
    year = datetime.now().year
    assert captured["start_date"] == f"{year - 2}-01-01"
    assert captured["end_date"] == f"{year}-12-31"


def test_fetch_tencent_historical_end_date_uses_end_year_lookback(monkeypatch):
    """只传历史 end_date 时，腾讯默认起点按 end 年份回看，避免 start > end 越界。"""
    import akshare

    captured = {}

    def em_down(**kwargs):
        raise OSError("em down")

    def fake_tx(**kwargs):
        captured.update(kwargs)
        # 返回落在 2020 区间内的数据，保证最终成功并命中断言
        return pd.DataFrame({
            "date": ["2020-01-02", "2020-01-03"],
            "open": [10.0, 11.0],
            "close": [10.5, 11.5],
            "high": [11.0, 12.0],
            "low": [9.0, 10.0],
            "volume": [100.0, 120.0],
            "turnover": [0.1, 0.1],
            "amount": [1000.0, 1200.0],
        })

    monkeypatch.setattr(akshare, "stock_zh_a_hist", em_down)
    monkeypatch.setattr(akshare, "stock_zh_a_hist_tx", fake_tx)
    _no_sleep(monkeypatch)

    df = fetch_market_data("600519.SH", end_date="20200105", source="akshare")

    assert captured["symbol"] == "sh600519"
    # 起点按 end 年份 2020 回看 2 年 = 2018-01-01，而不是当前年份（会 > end 触发越界）
    assert captured["start_date"] == "2018-01-01"
    assert captured["end_date"] == "2020-01-05"
    assert df.attrs["data_source"] == "akshare_tencent"
    assert (df["symbol"] == "600519.SH").all()


def test_fetch_tencent_builds_tx_symbol_and_dashed_dates(monkeypatch):
    """多股票时逐只以腾讯代码格式调用，并把 YYYYMMDD 转成连字符日期。"""
    import akshare

    calls = []

    def em_down(**kwargs):
        raise OSError("em down")

    def fake_tx(**kwargs):
        calls.append(kwargs)
        return _tx_raw(kwargs["symbol"])

    monkeypatch.setattr(akshare, "stock_zh_a_hist", em_down)
    monkeypatch.setattr(akshare, "stock_zh_a_hist_tx", fake_tx)
    _no_sleep(monkeypatch)

    df = fetch_market_data(
        ["600519.SH", "000001.SZ"], start_date="20240102", end_date="20241231",
        source="akshare",
    )
    assert len(calls) == 2
    assert calls[0]["symbol"] == "sh600519"
    assert calls[0]["start_date"] == "2024-01-02"
    assert calls[0]["end_date"] == "2024-12-31"
    assert calls[1]["symbol"] == "sz000001"
    assert set(df["symbol"].unique()) == {"600519.SH", "000001.SZ"}


def test_finalize_daily_dedups_and_filters():
    """跨年请求重叠时按 (symbol, trade_date) 去重，并过滤到 [start, end]。"""
    df = pd.DataFrame({
        "symbol": ["600519.SH", "600519.SH", "600519.SH"],
        "trade_date": ["2024-01-02", "2024-01-02", "2024-01-05"],
        "open": [10.0, 10.0, 11.0],
        "high": [11.0, 11.0, 12.0],
        "low": [9.0, 9.0, 10.0],
        "close": [10.5, 10.5, 11.5],
        "volume": [100.0, 100.0, 120.0],
        "amount": [1000.0, 1000.0, 1200.0],
    })
    out = _finalize_daily(df, "20240102", "20240103")
    # 2024-01-02 去重后只剩一行；2024-01-05 超出 end 被过滤
    assert out["trade_date"].tolist() == ["2024-01-02"]
    assert len(out) == 1
    assert list(out.columns) == list(BASE_MARKET_COLUMNS)


def test_fetch_tencent_direct_returns_contract_shape(monkeypatch):
    """_fetch_tencent 单独调用也返回标准 Schema。"""
    import akshare

    monkeypatch.setattr(akshare, "stock_zh_a_hist_tx", lambda **k: _tx_raw(k["symbol"]))
    _no_sleep(monkeypatch)

    df = _fetch_tencent(["600519.SH", "000001.SZ"], "20240102", "20240105")
    assert df is not None
    assert list(df.columns) == list(BASE_MARKET_COLUMNS)
    assert set(df["symbol"].unique()) == {"600519.SH", "000001.SZ"}
    # 000001 深市主板 volume 已补正为股
    assert df[df["symbol"] == "000001.SZ"]["volume"].tolist() == pytest.approx(
        [115836600.0, 73361000.0]
    )


# ---------------------------------------------------------------------------
# 与 clean 的衔接：不得二次换算
# ---------------------------------------------------------------------------

def test_akshare_through_clean_no_double_conversion(monkeypatch):
    """端到端：AkShare Provider 输出 → clean_market_data 不得二次换算单位。

    AkShare 成交量在 Provider 内已完成 手→股 ×100，成交额本就是元；clean 的
    ``units="auto"`` 必须识别为 standard（有 ``volume`` 列而非 Tushare 的 ``vol``），
    不能像 Tushare 那样再把 amount ×1000。
    """
    import akshare

    monkeypatch.setattr(akshare, "stock_zh_a_hist", lambda **k: _akshare_raw_frame())

    raw = fetch_market_data("600519.SH", source="akshare")
    # 关键：clean 的 auto 检测必须判为 standard，否则会按 Tushare 口径把 amount ×1000
    assert _detect_units(raw) == "standard"
    clean = clean_market_data(raw)  # units="auto"
    assert clean["volume"].tolist() == [3215600, 2022900]  # 仍是股，未被再 ×100
    assert clean["amount"].iloc[0] == pytest.approx(5.44e9)  # 仍是元，未被 ×1000


def test_tencent_through_clean_no_double_conversion(monkeypatch):
    """腾讯官方结果（volume 已股、amount 已元）经 clean_market_data 不二次换算。"""
    import akshare

    def em_down(**kwargs):
        raise OSError("em down")

    monkeypatch.setattr(akshare, "stock_zh_a_hist", em_down)
    monkeypatch.setattr(akshare, "stock_zh_a_hist_tx", lambda **k: _tx_raw(k["symbol"]))
    _no_sleep(monkeypatch)

    raw = fetch_market_data(
        "600519.SH", start_date="20240102", end_date="20240105", source="akshare"
    )
    assert raw.attrs["data_source"] == "akshare_tencent"
    assert _detect_units(raw) == "standard"
    clean = clean_market_data(raw)
    assert clean["volume"].tolist() == pytest.approx([3215600.0, 2022900.0])
    assert clean["amount"].tolist() == pytest.approx([5440082500.0, 3411400700.0])
