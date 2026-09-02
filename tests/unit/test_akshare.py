"""Tests for the Role 2 AkShare provider (``fetch_market_data(source="akshare")``).

All network access is mocked — no test here hits the real eastmoney endpoint, so
the suite runs offline and deterministically.
"""

import pandas as pd
import pytest

from src.contracts.market_data import BASE_MARKET_COLUMNS
from src.data.clean import _detect_units, clean_market_data
from src.data.fetch import (
    _convert_akshare,
    _symbol_with_exchange,
    fetch_market_data,
)
from src.utils.exceptions import InvalidSymbolError, NoDataError


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


# ---------------------------------------------------------------------------
# 列名映射 + 单位转换
# ---------------------------------------------------------------------------

def _akshare_raw_frame() -> pd.DataFrame:
    """AkShare ``stock_zh_a_hist`` 的真实输出形状（中文列名）。"""
    return pd.DataFrame({
        "日期": ["2024-01-02", "2024-01-03"],
        "股票代码": ["600519", "600519"],
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
# 抓取（mock 网络）
# ---------------------------------------------------------------------------

def test_fetch_akshare_calls_stock_zh_a_hist_with_qfq(monkeypatch):
    import akshare

    captured = {}

    def fake_stock_zh_a_hist(**kwargs):
        captured.update(kwargs)
        return _akshare_raw_frame()

    monkeypatch.setattr(akshare, "stock_zh_a_hist", fake_stock_zh_a_hist)

    df = fetch_market_data("600519.SH", source="akshare")
    # 传给 AkShare 的参数必须是裸 6 位代码 + daily + qfq
    assert captured["symbol"] == "600519"
    assert captured["period"] == "daily"
    assert captured["adjust"] == "qfq"
    # 返回标准 Schema + 来源标记
    assert list(df.columns) == list(BASE_MARKET_COLUMNS)
    assert (df["symbol"] == "600519.SH").all()
    assert df["volume"].tolist() == [3215600, 2022900]
    assert df.attrs["data_source"] == "akshare"
    assert df.attrs["is_sample"] is False


def test_fetch_akshare_passes_dates_and_multiple_symbols(monkeypatch):
    import akshare

    calls = []

    def fake_stock_zh_a_hist(**kwargs):
        calls.append(kwargs)
        raw = _akshare_raw_frame().copy()
        raw["股票代码"] = kwargs["symbol"]  # 让每只股票返回自己的代码
        return raw

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


def test_fetch_akshare_strict_raises_on_provider_error(monkeypatch):
    import akshare

    def boom(**kwargs):
        raise OSError("network down")

    monkeypatch.setattr(akshare, "stock_zh_a_hist", boom)
    with pytest.raises(NoDataError):
        fetch_market_data("600519.SH", source="akshare")


def test_fetch_akshare_strict_raises_on_empty(monkeypatch):
    import akshare

    monkeypatch.setattr(akshare, "stock_zh_a_hist", lambda **k: pd.DataFrame())
    with pytest.raises(NoDataError):
        fetch_market_data("600519.SH", source="akshare")


def test_fetch_akshare_invalid_symbol_not_swallowed(monkeypatch):
    """无效代码在抓取前就被拒绝，即使 provider 会失败也不吞掉。"""
    import akshare

    def boom(**kwargs):
        raise OSError("network down")

    monkeypatch.setattr(akshare, "stock_zh_a_hist", boom)
    with pytest.raises(InvalidSymbolError):
        fetch_market_data("600519", source="akshare")  # 缺少 .SH/.SZ 后缀


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
