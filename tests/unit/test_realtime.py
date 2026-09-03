"""Tests for ``fetch_realtime_quotes`` (Role 2 realtime snapshot interface).

All network access is mocked — no test here hits a real endpoint. Covers the
eastmoney → sina fallback, symbol filtering, unit handling and provenance.
"""

import pandas as pd
import pytest

from src.data.fetch import _REALTIME_COLUMNS, fetch_realtime_quotes
from src.utils.exceptions import InvalidSymbolError, NoDataError


def _em_spot_frame() -> pd.DataFrame:
    """AkShare ``stock_zh_a_spot_em`` 输出形状（中文列名，成交量单位=手）。"""
    return pd.DataFrame({
        "序号": [1, 2],
        "代码": ["600519", "000001"],
        "名称": ["贵州茅台", "平安银行"],
        "最新价": [1300.09, 11.5],
        "涨跌幅": [0.20, -1.2],
        "涨跌额": [2.59, -0.14],
        "成交量": [616175, 500000],          # 手
        "成交额": [801172182.0, 57500000.0],  # 元
        "振幅": [1.0, 2.0],
        "最高": [1305.0, 11.8],
        "最低": [1293.02, 11.2],
        "今开": [1297.5, 11.6],
        "昨收": [1297.5, 11.64],
    })


def _sina_spot_frame() -> pd.DataFrame:
    """AkShare ``stock_zh_a_spot`` 输出形状（代码 sh/sz 前缀，成交量单位=股）。"""
    return pd.DataFrame({
        "代码": ["sh600519", "sz000001"],
        "名称": ["贵州茅台", "平安银行"],
        "最新价": [1300.09, 11.5],
        "涨跌额": [2.59, -0.14],
        "涨跌幅": [0.2, -1.2],
        "买入": [1300.0, 11.49],
        "卖出": [1300.1, 11.51],
        "昨收": [1297.5, 11.64],
        "今开": [1297.5, 11.6],
        "最高": [1305.0, 11.8],
        "最低": [1293.02, 11.2],
        "成交量": [616175.0, 500000.0],       # 股
        "成交额": [801172182.0, 57500000.0],  # 元
        "时间戳": ["10:16:22", "10:16:22"],
    })


def test_realtime_eastmoney_success(monkeypatch):
    import akshare

    monkeypatch.setattr(akshare, "stock_zh_a_spot_em", lambda: _em_spot_frame())

    df = fetch_realtime_quotes("600519.SH")
    assert list(df.columns) == list(_REALTIME_COLUMNS)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["symbol"] == "600519.SH"
    assert row["name"] == "贵州茅台"
    assert row["price"] == pytest.approx(1300.09)
    assert row["change"] == pytest.approx(2.59)
    assert row["pct_change"] == pytest.approx(0.20)
    assert row["prev_close"] == pytest.approx(1297.5)
    assert row["open"] == pytest.approx(1297.5)
    assert row["high"] == pytest.approx(1305.0)
    assert row["low"] == pytest.approx(1293.02)
    # 成交量 手 -> 股 ×100；成交额已是元
    assert row["volume"] == pytest.approx(616175 * 100)
    assert row["amount"] == pytest.approx(801172182.0)
    # 东方财富快照不提供时间戳
    assert row["timestamp"] is None
    # provenance
    assert df.attrs["data_source"] == "akshare_eastmoney"
    assert df.attrs["provider"] == "eastmoney"
    assert df.attrs["is_sample"] is False
    assert "fetched_at" in df.attrs


def test_realtime_falls_back_to_sina(monkeypatch):
    import akshare

    def em_down():
        raise ConnectionError("em spot down")

    monkeypatch.setattr(akshare, "stock_zh_a_spot_em", em_down)
    monkeypatch.setattr(akshare, "stock_zh_a_spot", lambda: _sina_spot_frame())

    df = fetch_realtime_quotes("600519.SH")
    assert df.attrs["data_source"] == "akshare_sina"
    assert df.attrs["provider"] == "sina"
    assert df.attrs["is_sample"] is False
    row = df.iloc[0]
    assert row["symbol"] == "600519.SH"
    # 新浪成交量已是股，不换算
    assert row["volume"] == pytest.approx(616175.0)
    assert row["amount"] == pytest.approx(801172182.0)
    # 新浪提供时间戳
    assert row["timestamp"] == "10:16:22"
    assert "fetched_at" in df.attrs


def test_realtime_filters_requested_symbols(monkeypatch):
    import akshare

    monkeypatch.setattr(akshare, "stock_zh_a_spot_em", lambda: _em_spot_frame())

    df = fetch_realtime_quotes(["000001.SZ"])
    assert len(df) == 1
    assert df.iloc[0]["symbol"] == "000001.SZ"
    assert df.iloc[0]["name"] == "平安银行"


def test_realtime_multiple_symbols(monkeypatch):
    import akshare

    monkeypatch.setattr(akshare, "stock_zh_a_spot", lambda: _sina_spot_frame())
    monkeypatch.setattr(akshare, "stock_zh_a_spot_em", lambda: (_ for _ in ()).throw(ConnectionError("em down")))

    df = fetch_realtime_quotes(["600519.SH", "000001.SZ"])
    assert set(df["symbol"]) == {"600519.SH", "000001.SZ"}
    assert df.attrs["provider"] == "sina"


def test_realtime_both_providers_fail_raises(monkeypatch):
    import akshare

    monkeypatch.setattr(akshare, "stock_zh_a_spot_em", lambda: (_ for _ in ()).throw(ConnectionError("em down")))
    monkeypatch.setattr(akshare, "stock_zh_a_spot", lambda: (_ for _ in ()).throw(ConnectionError("sina down")))

    with pytest.raises(NoDataError) as excinfo:
        fetch_realtime_quotes("600519.SH")
    msg = str(excinfo.value)
    assert "eastmoney" in msg
    assert "sina" in msg


def test_realtime_symbol_not_found_in_either_provider(monkeypatch):
    import akshare

    monkeypatch.setattr(akshare, "stock_zh_a_spot_em", lambda: _em_spot_frame())  # 无 999999
    monkeypatch.setattr(akshare, "stock_zh_a_spot", lambda: _sina_spot_frame())   # 无 999999

    with pytest.raises(NoDataError):
        fetch_realtime_quotes("999999.SH")


def test_realtime_invalid_symbol_rejected(monkeypatch):
    with pytest.raises(InvalidSymbolError):
        fetch_realtime_quotes("600519")  # 缺少后缀
    with pytest.raises(InvalidSymbolError):
        fetch_realtime_quotes("sh600519")


def test_realtime_never_writes_local_csv(monkeypatch):
    import akshare

    def no_write(*a, **k):
        raise AssertionError("实时行情不应写本地 CSV")

    monkeypatch.setattr("pandas.DataFrame.to_csv", no_write)
    monkeypatch.setattr(akshare, "stock_zh_a_spot_em", lambda: _em_spot_frame())

    df = fetch_realtime_quotes("600519.SH")
    assert df.attrs["data_source"] == "akshare_eastmoney"


def test_realtime_partial_success_merges_providers(monkeypatch):
    """东方财富只返回部分股票时，新浪只补缺失股票，最终合并。"""
    import akshare

    em_only = _em_spot_frame().iloc[[0]]  # 只含 600519.SH
    monkeypatch.setattr(akshare, "stock_zh_a_spot_em", lambda: em_only)
    monkeypatch.setattr(akshare, "stock_zh_a_spot", lambda: _sina_spot_frame())

    df = fetch_realtime_quotes(["600519.SH", "000001.SZ"])
    assert set(df["symbol"]) == {"600519.SH", "000001.SZ"}
    assert df.attrs["data_source"] == "akshare_mixed"
    assert df.attrs["provider"] == "eastmoney+sina"


def test_realtime_reports_missing_symbols(monkeypatch):
    """两个 Provider 都缺某只股票时，必须明确报告缺失代码。"""
    import akshare

    em_only = _em_spot_frame().iloc[[0]]       # 只有 600519.SH
    sina_only = _sina_spot_frame().iloc[[0]]   # 只有 600519.SH
    monkeypatch.setattr(akshare, "stock_zh_a_spot_em", lambda: em_only)
    monkeypatch.setattr(akshare, "stock_zh_a_spot", lambda: sina_only)

    with pytest.raises(NoDataError) as excinfo:
        fetch_realtime_quotes(["600519.SH", "000001.SZ"])
    assert "000001.SZ" in str(excinfo.value)


def test_realtime_pct_change_is_percent_not_decimal(monkeypatch):
    """涨跌幅（pct_change）单位为百分比：0.20 表示 +0.20%，绝不换算成小数 0.002。"""
    import akshare

    monkeypatch.setattr(akshare, "stock_zh_a_spot_em", lambda: _em_spot_frame())

    df = fetch_realtime_quotes("600519.SH")
    row = df.iloc[0]
    assert row["pct_change"] == pytest.approx(0.20)
    # 涨跌额为绝对值（元），与涨跌幅相互独立
    assert row["change"] == pytest.approx(2.59)


def test_realtime_sina_pct_change_also_percent(monkeypatch):
    """新浪源的涨跌幅同样是百分比，不换算。"""
    import akshare

    monkeypatch.setattr(akshare, "stock_zh_a_spot_em", lambda: (_ for _ in ()).throw(ConnectionError("em down")))
    monkeypatch.setattr(akshare, "stock_zh_a_spot", lambda: _sina_spot_frame())

    df = fetch_realtime_quotes("000001.SZ")
    row = df.iloc[0]
    assert row["pct_change"] == pytest.approx(-1.2)  # 百分比，非 -0.012
