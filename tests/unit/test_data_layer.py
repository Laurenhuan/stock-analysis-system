"""Tests for the Role 2 data layer (fetch / clean / features)."""

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
        fetch_market_data("600519")  # missing exchange suffix
    with pytest.raises(InvalidSymbolError):
        fetch_market_data("sh600519")  # wrong format
    with pytest.raises(InvalidSymbolError):
        fetch_market_data(["600519.SH", "bad"])


def test_fetch_fallback_returns_sample_data(no_token):
    df = fetch_market_data("600519.SH", start_date="20240102", end_date="20240110")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "ts_code" in df.columns
    assert (df["ts_code"] == "600519.SH").all()


def test_fetch_fallback_missing_symbol_raises(no_token):
    with pytest.raises(NoDataError):
        fetch_market_data("999999.SH")  # not present in sample data


def test_fetch_no_fallback_no_token_raises(no_token):
    with pytest.raises(NoDataError):
        fetch_market_data("600519.SH", fallback=False)


# --------------------------------------------------------------------------
# clean
# --------------------------------------------------------------------------

def _raw_frame() -> pd.DataFrame:
    # Deliberately out of order to exercise sorting.
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


def test_clean_converts_units_and_sorts():
    out = clean_market_data(_raw_frame())
    assert list(out.columns) == list(BASE_MARKET_COLUMNS)
    assert out["symbol"].tolist() == ["000001.SZ", "000001.SZ", "600519.SH"]
    assert out["trade_date"].dt.strftime("%Y%m%d").tolist() == [
        "20240102", "20240103", "20240102",
    ]
    assert str(out["trade_date"].dtype) == "datetime64[ns]"

    first = out.iloc[0]
    assert first["volume"] == pytest.approx(1000.0 * 100)   # 手 -> 股
    assert first["amount"] == pytest.approx(1030.0 * 1000)  # 千元 -> 元


def test_clean_dedups_exact_duplicates():
    raw = _raw_frame()
    raw = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)  # duplicate row
    out = clean_market_data(raw)
    assert len(out) == len(_raw_frame())


def test_clean_rejects_conflicting_duplicates():
    raw = _raw_frame()
    conflict = raw.iloc[[0]].copy()
    conflict["close"] = 999.0
    raw = pd.concat([raw, conflict], ignore_index=True)
    with pytest.raises(DataValidationError):
        clean_market_data(raw)


def test_clean_empty_raises():
    with pytest.raises(NoDataError):
        clean_market_data(pd.DataFrame())


def test_clean_missing_column_raises():
    raw = _raw_frame().drop(columns=["vol"])  # volume can no longer be derived
    with pytest.raises(DataValidationError):
        clean_market_data(raw)


def test_clean_drops_nonfinite_rows():
    raw = _raw_frame()
    raw.loc[0, "close"] = np.nan
    out = clean_market_data(raw)
    assert len(out) == len(raw) - 1


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
    # cumulative return equals close / first_close - 1
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
    assert not np.isinf(a["volume_change"]).any()  # never inf


def test_features_drawdown_nonpositive():
    out = build_common_features(_base_frame())
    a = _by_symbol(out, "600519.SH")
    assert a.loc[2, "drawdown"] == pytest.approx(10.5 / 11 - 1)
    assert (out["drawdown"].dropna() <= 0).all()


def test_features_group_isolation():
    out = build_common_features(_base_frame())
    b = _by_symbol(out, "000001.SZ")
    # first row of the second symbol must not inherit the first symbol's history
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
    assert vol.iloc[:20].isna().all()  # return itself carries one leading NaN
    ret = df["close"] / df["close"].shift(1) - 1
    expected = ret.iloc[1:21].std(ddof=1)
    assert vol.iloc[20] == pytest.approx(expected)


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
    raw = fetch_market_data("600519.SH", start_date="20240102", end_date="20240329")
    clean = clean_market_data(raw)
    feats = build_common_features(clean)
    assert list(feats.columns) == list(BASE_MARKET_COLUMNS) + list(COMMON_FEATURE_COLUMNS)
    assert (feats["volume"] > 0).all()
    assert (feats["amount"] > 0).all()
    assert (feats["drawdown"] <= 0).all()
    assert (feats["symbol"] == "600519.SH").all()
