"""Tests for the Role 2 data layer (fetch / clean / features).

These tests never touch the network or a real Tushare token: the provider is
mocked, and the sample path reads the committed ``data/sample`` CSV.
"""

import numpy as np
import pandas as pd
import pytest

from src.contracts.market_data import BASE_MARKET_COLUMNS, COMMON_FEATURE_COLUMNS
from src.data.clean import clean_market_data
from src.data.features import build_common_features
from src.data.fetch import fetch_market_data
from src.utils.exceptions import (
    DataValidationError,
    InsufficientDataError,
    InvalidSymbolError,
    NoDataError,
)


# --------------------------------------------------------------------------
# fetch
# --------------------------------------------------------------------------

@pytest.fixture
def no_token(monkeypatch):
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)


def test_fetch_rejects_invalid_symbol(no_token):
    with pytest.raises(InvalidSymbolError):
        fetch_market_data("bad")  # 非 6 位数字
    with pytest.raises(InvalidSymbolError):
        fetch_market_data("600519.SS")  # 未知交易所后缀
    with pytest.raises(InvalidSymbolError):
        fetch_market_data("12345")  # 位数不足
    with pytest.raises(InvalidSymbolError):
        fetch_market_data("800001")  # 北交所，不支持的市场
    with pytest.raises(InvalidSymbolError):
        fetch_market_data(["600519.SH", "bad"])  # 任一非法即整体拒绝


def test_fetch_accepts_bare_and_prefixed_symbols(no_token):
    # 裸 6 位、带后缀、腾讯/新浪前缀都会被标准化后按样例读取。
    for sym in ("600519", "600519.SH", "600519.sh", "sh600519"):
        df = fetch_market_data(sym, source="sample")
        assert (df["symbol"] == "600519.SH").all()
    df = fetch_market_data("000001", source="sample")
    assert (df["symbol"] == "000001.SZ").all()


def test_fetch_invalid_symbol_not_swallowed_in_auto(no_token):
    # Even with a token and a provider that would fail, InvalidSymbolError must
    # surface before any fallback.
    with pytest.raises(InvalidSymbolError):
        fetch_market_data("bad", source="auto", token="fake-token")


def test_fetch_sample_reads_local_standard_format(no_token):
    df = fetch_market_data(
        "600519.SH", start_date="20240102", end_date="20240110", source="sample"
    )
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert list(df.columns) == list(BASE_MARKET_COLUMNS)
    assert (df["symbol"] == "600519.SH").all()
    assert df.attrs["data_source"] == "sample"
    assert df.attrs["is_sample"] is True


def test_fetch_sample_contains_five_stocks(no_token):
    symbols = ["600519.SH", "000001.SZ", "300750.SZ", "601318.SH", "000858.SZ"]
    df = fetch_market_data(symbols, source="sample")
    assert df["symbol"].nunique() == 5


def test_fetch_sample_missing_symbol_raises(no_token):
    with pytest.raises(NoDataError):
        fetch_market_data("600000.SH", source="sample")  # 合法代码但样例中没有


def test_fetch_tushare_requires_token(no_token):
    with pytest.raises(NoDataError):
        fetch_market_data("600519.SH", source="tushare")


def test_fetch_tushare_does_not_fall_back(monkeypatch, no_token):
    def boom(*a, **k):
        raise OSError("Connection reset")

    monkeypatch.setattr("src.data.fetch._fetch_tushare", boom)
    with pytest.raises(NoDataError):
        fetch_market_data("600519.SH", source="tushare", token="fake-token")


def test_fetch_auto_falls_back_on_quota_error(monkeypatch, no_token):
    def boom(*a, **k):
        raise Exception("抱歉，您没有访问该接口的权限，积分不足")

    monkeypatch.setattr("src.data.fetch._fetch_tushare", boom)
    df = fetch_market_data("600519.SH", source="auto", token="fake-token")
    assert df.attrs["data_source"] == "sample"
    assert df.attrs["is_sample"] is True


def test_fetch_auto_falls_back_on_network_error(monkeypatch, no_token):
    def boom(*a, **k):
        raise OSError("Recv failure: Connection was reset")

    monkeypatch.setattr("src.data.fetch._fetch_tushare", boom)
    df = fetch_market_data("600519.SH", source="auto", token="fake-token")
    assert df.attrs["is_sample"] is True


def test_fetch_auto_records_fallback_reason(monkeypatch, no_token):
    # 1) 权限/积分错误 → attrs 记录 Tushare 失败原因
    def boom(*a, **k):
        raise Exception("抱歉，您没有访问该接口的权限，积分不足")

    monkeypatch.setattr("src.data.fetch._fetch_tushare", boom)
    df = fetch_market_data("600519.SH", source="auto", token="fake-token")
    assert df.attrs["data_source"] == "sample"
    assert "积分" in df.attrs["fallback_reason"]

    # 2) 无 token → 记录未配置 token
    df2 = fetch_market_data("600519.SH", source="auto")
    assert df2.attrs["data_source"] == "sample"
    assert df2.attrs["fallback_reason"] == "未配置 TUSHARE_TOKEN"

    # 3) Tushare 返回空 → 记录空数据
    monkeypatch.setattr("src.data.fetch._fetch_tushare", lambda *a, **k: None)
    df3 = fetch_market_data("600519.SH", source="auto", token="fake-token")
    assert df3.attrs["data_source"] == "sample"
    assert df3.attrs["fallback_reason"] == "Tushare 返回空数据"


def test_fetch_auto_does_not_swallow_programming_error(monkeypatch, no_token):
    def boom(*a, **k):
        raise TypeError("unexpected internal bug")

    monkeypatch.setattr("src.data.fetch._fetch_tushare", boom)
    with pytest.raises(TypeError):
        fetch_market_data("600519.SH", source="auto", token="fake-token")


def test_fetch_auto_no_token_no_fallback_raises(no_token):
    with pytest.raises(NoDataError):
        fetch_market_data("600519.SH", fallback=False)


def test_fetch_tushare_uses_qfq_and_marks_source(monkeypatch):
    """Mock the provider and assert ``adj="qfq"`` is passed (no real network)."""
    import tushare as ts

    captured = {}

    def fake_pro_api(token):
        return object()

    def fake_pro_bar(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame({
            "ts_code": ["600519.SH"],
            "trade_date": ["20240102"],
            "open": [100.0], "high": [101.0], "low": [99.0], "close": [100.5],
            "pre_close": [100.0], "change": [0.5], "pct_chg": [0.5],
            "vol": [1000.0], "amount": [100000.0],
        })

    monkeypatch.setattr(ts, "pro_api", fake_pro_api)
    monkeypatch.setattr(ts, "pro_bar", fake_pro_bar)

    df = fetch_market_data("600519.SH", source="tushare", token="fake-token")
    assert captured["adj"] == "qfq"
    assert captured["ts_code"] == "600519.SH"
    assert df.attrs["data_source"] == "tushare"
    assert df.attrs["is_sample"] is False


# --------------------------------------------------------------------------
# clean
# --------------------------------------------------------------------------

def _raw_frame() -> pd.DataFrame:
    """Raw Tushare ``pro_bar`` shape: ``ts_code``, ``vol`` (手), ``amount`` (千元)."""
    return pd.DataFrame({
        "ts_code": ["000001.SZ", "000001.SZ", "600519.SH"],
        "trade_date": ["20240103", "20240102", "20240102"],
        "open": [10.2, 10.0, 1600.0],
        "high": [10.6, 10.5, 1620.0],
        "low": [10.0, 9.8, 1590.0],
        "close": [10.4, 10.3, 1610.0],
        "vol": [1100.0, 1000.0, 500.0],       # 手
        "amount": [1144.0, 1030.0, 8050.0],   # 千元
    })


def _standard_frame() -> pd.DataFrame:
    """Standard shape: ``symbol``, ``volume`` (股), ``amount`` (元)."""
    return pd.DataFrame({
        "symbol": ["000001.SZ", "600519.SH"],
        "trade_date": ["20240102", "20240102"],
        "open": [10.0, 1600.0],
        "high": [10.5, 1620.0],
        "low": [9.8, 1590.0],
        "close": [10.3, 1610.0],
        "volume": [100000.0, 50000.0],        # shares
        "amount": [1030000.0, 80500000.0],    # RMB
    })


def test_clean_raw_converts_units_once():
    out = clean_market_data(_raw_frame(), units="raw")
    assert list(out.columns) == list(BASE_MARKET_COLUMNS)
    # sorted by (symbol, trade_date): 000001.SZ 02 -> 03 -> 600519.SH 02
    assert out["volume"].tolist() == [100000.0, 110000.0, 50000.0]  # 手 -> 股
    assert out["amount"].tolist() == [1030000.0, 1144000.0, 8050000.0]  # 千元 -> 元


def test_clean_standard_units_not_converted():
    out = clean_market_data(_standard_frame(), units="standard")
    assert out["volume"].tolist() == [100000.0, 50000.0]  # unchanged shares
    assert out["amount"].tolist() == [1030000.0, 80500000.0]  # unchanged RMB


def test_clean_auto_detects_raw_and_standard():
    raw_out = clean_market_data(_raw_frame())
    assert raw_out["volume"].tolist() == [100000.0, 110000.0, 50000.0]
    std_out = clean_market_data(_standard_frame())
    assert std_out["volume"].tolist() == [100000.0, 50000.0]


def test_clean_idempotent_no_double_scale():
    once = clean_market_data(_raw_frame(), units="raw")
    twice = clean_market_data(once)
    assert (twice["volume"] == once["volume"]).all()
    assert (twice["amount"] == once["amount"]).all()
    # attrs marker present
    assert once.attrs["_cleaned"] is True


def test_clean_sorts_ascending_and_datetime_ns():
    out = clean_market_data(_raw_frame(), units="raw")
    assert out["symbol"].tolist() == ["000001.SZ", "000001.SZ", "600519.SH"]
    assert out["trade_date"].dt.strftime("%Y%m%d").tolist() == [
        "20240102", "20240103", "20240102",
    ]
    assert str(out["trade_date"].dtype) == "datetime64[ns]"


def test_clean_dedups_exact_duplicates():
    raw = _raw_frame()
    raw = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    out = clean_market_data(raw, units="raw")
    assert len(out) == len(_raw_frame())


def test_clean_rejects_conflicting_duplicates():
    raw = _raw_frame()
    conflict = raw.iloc[[0]].copy()
    conflict["close"] = 999.0
    raw = pd.concat([raw, conflict], ignore_index=True)
    with pytest.raises(DataValidationError):
        clean_market_data(raw, units="raw")


def test_clean_empty_raises():
    with pytest.raises(NoDataError):
        clean_market_data(pd.DataFrame())


def test_clean_missing_column_raises():
    raw = _raw_frame().drop(columns=["vol"])
    with pytest.raises(DataValidationError):
        clean_market_data(raw, units="raw")


def test_clean_drops_nonfinite_rows():
    raw = _raw_frame()
    raw.loc[0, "close"] = np.nan
    out = clean_market_data(raw, units="raw")
    assert len(out) == len(raw) - 1


def test_clean_unknown_units_raises():
    with pytest.raises(DataValidationError):
        clean_market_data(_raw_frame(), units="bogus")


# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------

def _base_frame() -> pd.DataFrame:
    dates = ["20240102", "20240103", "20240104", "20240105", "20240108"]
    return pd.DataFrame({
        "symbol": ["600519.SH"] * 5 + ["000001.SZ"] * 5,
        "trade_date": pd.to_datetime(dates * 2),
        "open": [10, 11, 10.5, 12, 11, 20, 22, 21, 23, 24],
        "high": [11, 11.5, 11, 12.5, 11.5, 21, 22.5, 21.5, 23.5, 24.5],
        "low": [9.5, 10.5, 10, 11.5, 10.5, 19.5, 21.5, 20.5, 22.5, 23.5],
        "close": [10, 11, 10.5, 12, 11, 20, 22, 21, 23, 24],
        "volume": [100, 110, 0, 130, 140, 200, 220, 240, 260, 280],
        "amount": [1000, 1100, 0, 1300, 1400, 2000, 2200, 2400, 2600, 2800],
    })


def _by_symbol(out: pd.DataFrame, symbol: str) -> pd.DataFrame:
    return out[out["symbol"] == symbol].reset_index(drop=True)


def test_features_return_and_cumulative():
    out = build_common_features(_base_frame())
    a = _by_symbol(out, "600519.SH")
    assert np.isnan(a.loc[0, "return"])
    assert a.loc[1, "return"] == pytest.approx(0.1)
    assert a.loc[2, "return"] == pytest.approx(10.5 / 11 - 1)
    assert np.isnan(a.loc[0, "cumulative_return"])
    assert a.loc[4, "cumulative_return"] == pytest.approx(11 / 10 - 1)


def test_features_ma5_leading_nan_and_mean():
    out = build_common_features(_base_frame())
    a = _by_symbol(out, "600519.SH")
    assert a["ma5"].iloc[:4].isna().all()
    assert a["ma5"].iloc[4] == pytest.approx(np.mean([10, 11, 10.5, 12, 11]))
    assert a["ma20"].isna().all()  # only 5 rows, window never fills


def test_features_volume_change_nan_when_prev_zero():
    out = build_common_features(_base_frame())
    a = _by_symbol(out, "600519.SH")
    assert a.loc[2, "volume_change"] == pytest.approx(0 / 110 - 1)
    assert np.isnan(a.loc[3, "volume_change"])  # previous volume == 0
    assert not np.isinf(a["volume_change"]).any()


def test_features_no_infinite_values():
    out = build_common_features(_base_frame())
    numeric = out[list(COMMON_FEATURE_COLUMNS)]
    assert not np.isinf(numeric.to_numpy()).any()


def test_features_drawdown_nonpositive():
    out = build_common_features(_base_frame())
    a = _by_symbol(out, "600519.SH")
    assert a.loc[2, "drawdown"] == pytest.approx(10.5 / 11 - 1)
    assert (out["drawdown"].dropna() <= 0).all()


def test_features_group_isolation():
    out = build_common_features(_base_frame())
    b = _by_symbol(out, "000001.SZ")
    assert np.isnan(b.loc[0, "return"])
    assert np.isnan(b.loc[0, "ma5"])
    assert b.loc[4, "ma5"] == pytest.approx(np.mean([20, 22, 21, 23, 24]))


def test_features_volatility_ddof1_not_annualized():
    rng = np.random.default_rng(0)
    close = np.round(np.cumprod(1 + rng.normal(0, 0.01, 25)) * 100, 4)
    df = pd.DataFrame({
        "symbol": ["600519.SH"] * 25,
        "trade_date": pd.bdate_range("2024-01-02", periods=25),
        "open": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": np.full(25, 100.0),
        "amount": close * 1000,
    })
    out = build_common_features(df)
    vol = out["volatility_20d"]
    assert vol.iloc[:20].isna().all()
    ret = df["close"] / df["close"].shift(1) - 1
    expected = ret.iloc[1:21].std(ddof=1)
    assert vol.iloc[20] == pytest.approx(expected)


def test_features_does_not_mutate_input():
    df = _base_frame()
    before = df.copy(deep=True)
    build_common_features(df)
    pd.testing.assert_frame_equal(df, before)


def test_features_sorted_by_symbol_and_date():
    df = _base_frame().sample(frac=1, random_state=0)  # shuffle input
    out = build_common_features(df)
    assert out["symbol"].is_monotonic_increasing
    for _, g in out.groupby("symbol"):
        assert g["trade_date"].is_monotonic_increasing


def test_features_empty_raises():
    with pytest.raises(InsufficientDataError):
        build_common_features(pd.DataFrame())


def test_features_missing_columns_raises():
    with pytest.raises(DataValidationError):
        build_common_features(pd.DataFrame({"close": [1.0, 2.0]}))


# --------------------------------------------------------------------------
# end-to-end (sample fallback path)
# --------------------------------------------------------------------------

def test_end_to_end_sample_pipeline(no_token):
    raw = fetch_market_data("600519.SH", source="sample")
    clean = clean_market_data(raw)
    feats = build_common_features(clean)
    assert list(feats.columns) == list(BASE_MARKET_COLUMNS) + list(COMMON_FEATURE_COLUMNS)
    assert (feats["volume"] > 0).all()
    assert (feats["amount"] > 0).all()
    assert (feats["drawdown"] <= 0).all()
    assert (feats["symbol"] == "600519.SH").all()
    # provenance survives the whole pipeline
    assert feats.attrs["is_sample"] is True
    assert feats.attrs["data_source"] == "sample"
