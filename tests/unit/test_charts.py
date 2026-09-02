"""Unit tests for src.visualization.charts (Role 3).

Coverage: Figure return type, axis/trace data, missing columns, empty data,
single-symbol and multi-symbol inputs.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from src.utils.exceptions import DataValidationError, NoDataError
from src.visualization.charts import (
    plot_actual_vs_predicted,
    plot_confusion_matrix,
    plot_correlation_matrix,
    plot_price,
    plot_returns_comparison,
    plot_risk_comparison,
)


@pytest.fixture
def single_price_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "close": [100.0, 110.0, 105.0],
            "ma5": [np.nan, np.nan, 105.0],
            "ma20": [np.nan, np.nan, np.nan],
        }
    )


@pytest.fixture
def multi_price_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-03"] * 2),
            "symbol": ["600519.SH", "600519.SH", "000001.SZ", "000001.SZ"],
            "close": [100.0, 110.0, 10.0, 11.0],
        }
    )


@pytest.fixture
def returns_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-03"] * 2),
            "symbol": ["600519.SH", "600519.SH", "000001.SZ", "000001.SZ"],
            "cumulative_return": [0.0, 0.1, 0.0, -0.05],
        }
    )


@pytest.fixture
def risk_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["600519.SH", "000001.SZ"],
            "volatility": [0.08, 0.12],
            "max_drawdown": [-0.05, -0.10],
        }
    )


# --- plot_price --------------------------------------------------------------

def test_plot_price_returns_figure(single_price_df):
    fig = plot_price(single_price_df)
    assert isinstance(fig, go.Figure)


def test_plot_price_single_trace_data(single_price_df):
    fig = plot_price(single_price_df)
    assert len(fig.data) == 1
    assert list(fig.data[0].y) == [100.0, 110.0, 105.0]


def test_plot_price_multi_symbol_traces(multi_price_df):
    fig = plot_price(multi_price_df)
    assert len(fig.data) == 2  # one line per symbol


def test_plot_price_show_ma(single_price_df):
    fig = plot_price(single_price_df, show_ma=True)
    # close + ma5 + ma20
    assert len(fig.data) == 3


def test_plot_price_missing_column(single_price_df):
    with pytest.raises(DataValidationError):
        plot_price(single_price_df.drop(columns=["close"]))


def test_plot_price_empty():
    with pytest.raises(NoDataError):
        plot_price(pd.DataFrame())


# --- plot_returns_comparison -------------------------------------------------

def test_plot_returns_comparison_figure_and_traces(returns_df):
    fig = plot_returns_comparison(returns_df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2


def test_plot_returns_comparison_missing_column(returns_df):
    with pytest.raises(DataValidationError):
        plot_returns_comparison(returns_df.drop(columns=["cumulative_return"]))


# --- plot_risk_comparison ----------------------------------------------------

def test_plot_risk_comparison_figure_and_data(risk_df):
    fig = plot_risk_comparison(risk_df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2  # volatility + max drawdown bars
    assert list(fig.data[0].y) == [0.08, 0.12]


def test_plot_risk_comparison_missing_column(risk_df):
    with pytest.raises(DataValidationError):
        plot_risk_comparison(risk_df.drop(columns=["volatility"]))


# --- plot_confusion_matrix ---------------------------------------------------

def test_plot_confusion_matrix_figure_and_values():
    fig = plot_confusion_matrix([[8, 2], [1, 9]])
    assert isinstance(fig, go.Figure)
    assert np.asarray(fig.data[0].z).tolist() == [[8, 2], [1, 9]]


def test_plot_confusion_matrix_invalid_shape():
    with pytest.raises(DataValidationError):
        plot_confusion_matrix([[1, 2, 3]])


# --- plot_actual_vs_predicted ------------------------------------------------

def test_plot_actual_vs_predicted_figure_and_traces():
    df = pd.DataFrame({"y_true": [1.0, 2.0, 3.0], "y_pred": [1.1, 1.9, 3.2]})
    fig = plot_actual_vs_predicted(df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2  # samples + y=x line
    assert list(fig.data[0].x) == [1.0, 2.0, 3.0]
    assert list(fig.data[0].y) == [1.1, 1.9, 3.2]


def test_plot_actual_vs_predicted_missing_column():
    df = pd.DataFrame({"y_true": [1.0, 2.0]})
    with pytest.raises(DataValidationError):
        plot_actual_vs_predicted(df)


# --- selection via ``symbols`` ----------------------------------------------

def test_plot_price_symbols_selection(multi_price_df):
    fig = plot_price(multi_price_df, symbols=["600519.SH"])
    assert len(fig.data) == 1


def test_plot_price_symbols_not_found(multi_price_df):
    with pytest.raises(NoDataError):
        plot_price(multi_price_df, symbols=["999999.SZ"])


def test_plot_returns_comparison_symbols_selection(returns_df):
    fig = plot_returns_comparison(returns_df, symbols=["000001.SZ"])
    assert len(fig.data) == 1


def test_plot_risk_comparison_symbols_selection(risk_df):
    fig = plot_risk_comparison(risk_df, symbols=["600519.SH"])
    assert len(fig.data) == 2  # volatility + max drawdown bars


# --- plot_correlation_matrix -------------------------------------------------

def _corr10() -> pd.DataFrame:
    symbols = [f"{i:06d}.SH" for i in range(1, 11)]
    return pd.DataFrame(np.eye(10), index=symbols, columns=symbols)


def test_plot_correlation_matrix_10x10():
    fig = plot_correlation_matrix(_corr10())
    assert isinstance(fig, go.Figure)
    z = np.asarray(fig.data[0].z)
    assert z.shape == (10, 10)
    assert np.allclose(np.diag(z), 1.0)


def test_plot_correlation_matrix_invalid_shape():
    with pytest.raises(DataValidationError):
        plot_correlation_matrix(np.eye(3)[:, :2])
