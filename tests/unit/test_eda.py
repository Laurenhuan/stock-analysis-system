"""Unit tests for src.analysis.eda (Role 3).

Coverage: missing columns, empty data, single-symbol input, multi-symbol input
and hand-calculated numeric expectations.
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis.eda import (
    correlation_matrix,
    date_range_summary,
    describe_statistics,
    missing_values_summary,
    returns_comparison,
    risk_return_summary,
)
from src.utils.exceptions import (
    DataValidationError,
    InsufficientDataError,
    NoDataError,
)


def _make_df(closes_by_symbol: dict) -> pd.DataFrame:
    """Build a Contract-shaped DataFrame with hand-calculable returns/drawdown."""
    rows = []
    for sym, closes in closes_by_symbol.items():
        for i, c in enumerate(closes):
            prev = closes[i - 1] if i > 0 else None
            ret = c / prev - 1 if prev else np.nan
            drawdown = c / max(closes[: i + 1]) - 1
            rows.append(
                {
                    "symbol": sym,
                    "trade_date": pd.Timestamp("2024-01-01")
                    + pd.Timedelta(days=i),
                    "open": c * 0.99,
                    "high": c * 1.01,
                    "low": c * 0.98,
                    "close": c,
                    "volume": 1000.0 + i,
                    "amount": c * 1000.0,
                    "return": ret,
                    "cumulative_return": np.nan,
                    "ma5": c,
                    "ma20": np.nan,
                    "volatility_20d": np.nan if i == 0 else 0.08,
                    "volume_change": np.nan,
                    "drawdown": drawdown,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def multi_df() -> pd.DataFrame:
    return _make_df(
        {
            "600519.SH": [100.0, 110.0, 105.0, 120.0, 115.0],
            "000001.SZ": [10.0, 9.0, 9.5, 10.0, 10.5],
        }
    )


@pytest.fixture
def single_df(multi_df) -> pd.DataFrame:
    return multi_df[multi_df["symbol"] == "600519.SH"].reset_index(drop=True)


# --- describe_statistics -----------------------------------------------------

def test_describe_statistics_multi_symbol(multi_df):
    result = describe_statistics(multi_df)
    assert isinstance(result, pd.DataFrame)
    assert set(result.index) == {"600519.SH", "000001.SZ"}
    assert result.columns.nlevels == 2  # (field, statistic)
    assert "close" in result.columns.get_level_values(0)


def test_describe_statistics_single_symbol(single_df):
    result = describe_statistics(single_df)
    assert result.shape[0] == 1
    assert result.index[0] == "600519.SH"


def test_describe_statistics_skips_missing_column(multi_df):
    result = describe_statistics(multi_df, columns=("close", "not_a_column"))
    assert "close" in result.columns.get_level_values(0)
    assert "not_a_column" not in result.columns.get_level_values(0)


def test_describe_statistics_missing_symbol(multi_df):
    with pytest.raises(DataValidationError):
        describe_statistics(multi_df.drop(columns=["symbol"]))


def test_describe_statistics_empty():
    with pytest.raises(NoDataError):
        describe_statistics(pd.DataFrame())


# --- date_range_summary ------------------------------------------------------

def test_date_range_summary(multi_df):
    result = date_range_summary(multi_df)
    assert list(result.columns) == ["symbol", "start_date", "end_date", "n_rows"]
    row = result.set_index("symbol").loc["600519.SH"]
    assert row["start_date"] == pd.Timestamp("2024-01-01")
    assert row["end_date"] == pd.Timestamp("2024-01-05")
    assert row["n_rows"] == 5


def test_date_range_summary_missing_column(multi_df):
    with pytest.raises(DataValidationError):
        date_range_summary(multi_df.drop(columns=["trade_date"]))


# --- risk_return_summary -----------------------------------------------------

def test_risk_return_summary_max_drawdown(multi_df):
    result = risk_return_summary(multi_df)
    assert list(result.columns) == [
        "symbol", "mean_return", "volatility", "max_drawdown",
    ]
    a = result.set_index("symbol").loc["600519.SH"]
    b = result.set_index("symbol").loc["000001.SZ"]
    # max_drawdown is the deepest (most negative) drawdown.
    assert a["max_drawdown"] == pytest.approx(105 / 110 - 1)
    assert b["max_drawdown"] == pytest.approx(-0.1)
    # volatility is the sample std (ddof=1) of daily returns.
    returns_a = np.array([110 / 100 - 1, 105 / 110 - 1, 120 / 105 - 1, 115 / 120 - 1])
    assert a["volatility"] == pytest.approx(returns_a.std(ddof=1))


def test_risk_return_summary_missing_column(multi_df):
    with pytest.raises(DataValidationError):
        risk_return_summary(multi_df.drop(columns=["drawdown"]))


# --- returns_comparison ------------------------------------------------------

def test_returns_comparison_cumulative(multi_df):
    result = returns_comparison(multi_df)
    assert list(result.columns) == [
        "symbol", "mean_return", "cumulative_return", "win_rate", "std_return",
    ]
    a = result.set_index("symbol").loc["600519.SH"]
    # (110/100)*(105/110)*(120/105)*(115/120) - 1 == 115/100 - 1 == 0.15
    assert a["cumulative_return"] == pytest.approx(0.15)


def test_returns_comparison_win_rate_ignores_nan(multi_df):
    result = returns_comparison(multi_df)
    a = result.set_index("symbol").loc["600519.SH"]
    # 首日 NaN 不应计为下跌日：4 个有效交易日中 2 天上涨 → 2/4 = 0.5
    assert a["win_rate"] == pytest.approx(2 / 4)


def test_returns_comparison_all_nan_returns(multi_df):
    df = multi_df.copy()
    df["return"] = np.nan
    result = returns_comparison(df)
    row = result.set_index("symbol").loc["600519.SH"]
    assert np.isnan(row["cumulative_return"])
    assert np.isnan(row["win_rate"])
    assert np.isnan(row["std_return"])


def test_returns_comparison_missing_column(multi_df):
    with pytest.raises(DataValidationError):
        returns_comparison(multi_df.drop(columns=["return"]))


# --- correlation_matrix ------------------------------------------------------

def test_correlation_matrix_shape_and_diagonal(multi_df):
    corr = correlation_matrix(multi_df)
    assert corr.shape == (2, 2)
    assert corr.loc["600519.SH", "600519.SH"] == pytest.approx(1.0)
    assert corr.loc["000001.SZ", "000001.SZ"] == pytest.approx(1.0)


def test_correlation_matrix_spearman_perfect_rank():
    # 手工可算秩关系：A、B 收益率逐日严格递增 → +1；C 严格递减 → 与 A 为 -1
    closes = {
        "A": [100.0, 101.0, 103.0, 108.0],
        "B": [10.0, 10.5, 11.5, 15.0],
        "C": [20.0, 19.5, 19.0, 18.0],
    }
    rows = []
    for i in range(4):
        for sym, cs in closes.items():
            prev = cs[i - 1] if i > 0 else None
            rows.append(
                {
                    "symbol": sym,
                    "trade_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                    "return": cs[i] / prev - 1 if prev else np.nan,
                }
            )
    corr = correlation_matrix(pd.DataFrame(rows), method="spearman")
    assert corr.loc["A", "B"] == pytest.approx(1.0)
    assert corr.loc["A", "C"] == pytest.approx(-1.0)
    assert corr.loc["B", "C"] == pytest.approx(-1.0)


def test_correlation_matrix_invalid_method(multi_df):
    with pytest.raises(DataValidationError):
        correlation_matrix(multi_df, method="bogus")


def test_correlation_matrix_insufficient_overlap():
    # 两只股票只有 1 个共同有效交易日 → 明确报错而不是返回全 NaN
    rows = [
        {"symbol": "A", "trade_date": pd.Timestamp("2024-01-02"), "return": 0.01},
        {"symbol": "A", "trade_date": pd.Timestamp("2024-01-03"), "return": -0.01},
        {"symbol": "B", "trade_date": pd.Timestamp("2024-01-02"), "return": 0.02},
        {"symbol": "B", "trade_date": pd.Timestamp("2024-06-01"), "return": 0.03},
    ]
    with pytest.raises(InsufficientDataError):
        correlation_matrix(pd.DataFrame(rows))


def test_correlation_matrix_insufficient_symbols(single_df):
    with pytest.raises(InsufficientDataError):
        correlation_matrix(single_df)


def test_correlation_matrix_missing_column(multi_df):
    with pytest.raises(DataValidationError):
        correlation_matrix(multi_df.drop(columns=["return"]))


# --- missing_values_summary --------------------------------------------------

def test_missing_values_summary_counts(multi_df):
    result = missing_values_summary(multi_df)
    assert list(result.columns) == ["missing_count", "missing_ratio"]
    # ma20 is entirely NaN across all 10 rows.
    assert result.loc["ma20", "missing_count"] == 10
    # each symbol's first return is NaN.
    assert result.loc["return", "missing_count"] == 2


def test_missing_values_summary_empty():
    with pytest.raises(NoDataError):
        missing_values_summary(pd.DataFrame())


# --- 10-symbol scale ---------------------------------------------------------

def test_describe_statistics_ten_symbols():
    syms = {f"S{i:03d}": [100.0 + i, 110.0 + i, 105.0 + i] for i in range(10)}
    result = describe_statistics(_make_df(syms))
    assert result.shape[0] == 10


def test_correlation_matrix_ten_by_ten():
    syms = {
        f"S{i:03d}": [100.0 + i, 110.0 + i, 105.0 + i, 120.0 + i, 115.0 + i]
        for i in range(10)
    }
    corr = correlation_matrix(_make_df(syms))
    assert corr.shape == (10, 10)
    assert np.allclose(np.diag(corr), 1.0)
    # index 与 columns 的股票标签及顺序一致
    assert list(corr.index) == list(syms)
    assert list(corr.columns) == list(syms)


def test_eda_functions_do_not_mutate_input(multi_df):
    before = multi_df.copy(deep=True)
    describe_statistics(multi_df)
    date_range_summary(multi_df)
    risk_return_summary(multi_df)
    returns_comparison(multi_df)
    correlation_matrix(multi_df)
    missing_values_summary(multi_df)
    pd.testing.assert_frame_equal(multi_df, before)
