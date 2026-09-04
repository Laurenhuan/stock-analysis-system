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
    _COLOR_SEQUENCE,
    plot_actual_vs_predicted,
    plot_candlestick,
    plot_cluster_centers,
    plot_cluster_parallel_coordinates,
    plot_cluster_scatter,
    plot_confusion_matrix,
    plot_correlation_matrix,
    plot_price,
    plot_return_distribution,
    plot_returns_comparison,
    plot_risk_comparison,
    plot_rolling_volatility,
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


@pytest.fixture
def ohlc_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "open": [100.0, 110.0, 105.0],
            "high": [112.0, 115.0, 108.0],
            "low": [98.0, 108.0, 103.0],
            "close": [110.0, 105.0, 107.0],
        }
    )


@pytest.fixture
def multi_ohlc_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-03"] * 2),
            "symbol": ["600519.SH", "600519.SH", "000001.SZ", "000001.SZ"],
            "open": [100.0, 110.0, 10.0, 11.0],
            "high": [112.0, 115.0, 12.0, 13.0],
            "low": [98.0, 108.0, 9.0, 10.0],
            "close": [110.0, 105.0, 11.0, 12.0],
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


def test_plot_price_symbols_requires_symbol_column(single_price_df):
    with pytest.raises(DataValidationError):
        plot_price(single_price_df, symbols=["600519.SH"])


def test_plot_price_ten_symbols_traces():
    symbols = [f"S{i:03d}" for i in range(10)]
    rows = [
        {
            "trade_date": pd.Timestamp("2024-01-02") + pd.Timedelta(days=d),
            "symbol": s,
            "close": 100.0 + i + d,
        }
        for i, s in enumerate(symbols)
        for d in range(2)
    ]
    fig = plot_price(pd.DataFrame(rows))
    assert len(fig.data) == 10
    assert [t.name for t in fig.data] == symbols


# --- plot_candlestick --------------------------------------------------------

def test_plot_candlestick_returns_figure(ohlc_df):
    fig = plot_candlestick(ohlc_df)
    assert isinstance(fig, go.Figure)


def test_plot_candlestick_trace_type(ohlc_df):
    fig = plot_candlestick(ohlc_df)
    assert len(fig.data) == 1
    assert isinstance(fig.data[0], go.Candlestick)


def test_plot_candlestick_single_trace_data(ohlc_df):
    fig = plot_candlestick(ohlc_df)
    assert list(fig.data[0].open) == [100.0, 110.0, 105.0]
    assert list(fig.data[0].high) == [112.0, 115.0, 108.0]
    assert list(fig.data[0].low) == [98.0, 108.0, 103.0]
    assert list(fig.data[0].close) == [110.0, 105.0, 107.0]


def test_plot_candlestick_rejects_multi_symbol(multi_ohlc_df):
    # 蜡烛图只允许单只股票：多只股票时直接报错，而不是叠加多条蜡烛。
    with pytest.raises(DataValidationError):
        plot_candlestick(multi_ohlc_df)


def test_plot_candlestick_rejects_multiple_selected_symbols(multi_ohlc_df):
    with pytest.raises(DataValidationError):
        plot_candlestick(multi_ohlc_df, symbols=["600519.SH", "000001.SZ"])


def test_plot_candlestick_missing_column(ohlc_df):
    with pytest.raises(DataValidationError):
        plot_candlestick(ohlc_df.drop(columns=["open"]))


def test_plot_candlestick_empty():
    with pytest.raises(NoDataError):
        plot_candlestick(pd.DataFrame())


def test_plot_candlestick_symbols_requires_symbol_column(ohlc_df):
    with pytest.raises(DataValidationError):
        plot_candlestick(ohlc_df, symbols=["600519.SH"])


def test_plot_candlestick_symbols_selection(multi_ohlc_df):
    fig = plot_candlestick(multi_ohlc_df, symbols=["600519.SH"])
    assert len(fig.data) == 1
    assert fig.data[0].name == "600519.SH"


def test_plot_candlestick_symbols_not_found(multi_ohlc_df):
    with pytest.raises(NoDataError):
        plot_candlestick(multi_ohlc_df, symbols=["999999.SZ"])


def test_plot_candlestick_rangeslider_hidden(ohlc_df):
    fig = plot_candlestick(ohlc_df)
    assert fig.layout.xaxis.rangeslider.visible is False


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


def test_plot_confusion_matrix_rejects_3x3():
    with pytest.raises(DataValidationError):
        plot_confusion_matrix(np.eye(3))


def test_plot_confusion_matrix_requires_two_labels():
    with pytest.raises(DataValidationError):
        plot_confusion_matrix([[8, 2], [1, 9]], labels=("a", "b", "c"))


def test_plot_confusion_matrix_axis_labels():
    fig = plot_confusion_matrix([[8, 2], [1, 9]], labels=("负", "正"))
    assert list(fig.data[0].x) == ["负", "正"]
    assert list(fig.data[0].y) == ["负", "正"]


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


def test_plot_actual_vs_predicted_rejects_nan():
    df = pd.DataFrame({"y_true": [1.0, np.nan], "y_pred": [1.1, 1.9]})
    with pytest.raises(DataValidationError):
        plot_actual_vs_predicted(df)


def test_plot_actual_vs_predicted_rejects_inf():
    df = pd.DataFrame({"y_true": [1.0, np.inf], "y_pred": [1.1, 1.9]})
    with pytest.raises(DataValidationError):
        plot_actual_vs_predicted(df)


# --- selection via ``symbols`` ----------------------------------------------

def test_plot_price_symbols_selection(multi_price_df):
    fig = plot_price(multi_price_df, symbols=["600519.SH"])
    assert len(fig.data) == 1


def test_plot_price_symbols_selection_content(multi_price_df):
    fig = plot_price(multi_price_df, symbols=["000001.SZ"])
    assert len(fig.data) == 1
    assert fig.data[0].name == "000001.SZ"


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


def test_plot_correlation_matrix_labels_match_dataframe():
    corr = _corr10()
    fig = plot_correlation_matrix(corr)
    assert list(fig.data[0].x) == list(corr.columns)
    assert list(fig.data[0].y) == list(corr.index)


def test_plot_correlation_matrix_zmid_zero():
    fig = plot_correlation_matrix(_corr10())
    assert fig.data[0].zmid == 0.0


def test_plot_correlation_matrix_mismatched_labels():
    bad = pd.DataFrame(np.eye(2), index=["A", "B"], columns=["A", "C"])
    with pytest.raises(DataValidationError):
        plot_correlation_matrix(bad)


def test_plot_correlation_matrix_out_of_range():
    bad = _corr10()
    bad.iloc[0, 1] = 1.5
    with pytest.raises(DataValidationError):
        plot_correlation_matrix(bad)


def test_plot_correlation_matrix_non_finite():
    bad = _corr10()
    bad.iloc[0, 1] = np.nan
    with pytest.raises(DataValidationError):
        plot_correlation_matrix(bad)


def test_plot_correlation_matrix_scalar_input():
    with pytest.raises(DataValidationError):
        plot_correlation_matrix(0.5)


def test_plot_correlation_matrix_1d_input():
    with pytest.raises(DataValidationError):
        plot_correlation_matrix([1.0, 0.5, 0.5])


# --- theme / immutability ----------------------------------------------------

def test_color_sequence_twenty_distinct_colors():
    assert len(_COLOR_SEQUENCE) == 20
    assert len(set(_COLOR_SEQUENCE)) == len(_COLOR_SEQUENCE)


def test_chart_functions_do_not_mutate_input(
    multi_price_df, returns_df, risk_df, multi_ohlc_df
):
    price_before = multi_price_df.copy(deep=True)
    returns_before = returns_df.copy(deep=True)
    risk_before = risk_df.copy(deep=True)
    ohlc_before = multi_ohlc_df.copy(deep=True)
    plot_price(multi_price_df)
    plot_returns_comparison(returns_df)
    plot_risk_comparison(risk_df)
    plot_candlestick(multi_ohlc_df, symbols=["600519.SH"])
    pd.testing.assert_frame_equal(multi_price_df, price_before)
    pd.testing.assert_frame_equal(returns_df, returns_before)
    pd.testing.assert_frame_equal(risk_df, risk_before)
    pd.testing.assert_frame_equal(multi_ohlc_df, ohlc_before)


# --- 15 / 20 stock comparison scale ------------------------------------------

def test_plot_price_twenty_symbols_traces():
    symbols = [f"STK{i:02d}" for i in range(20)]
    rows = [
        {
            "trade_date": pd.Timestamp("2021-06-01") + pd.Timedelta(days=d),
            "symbol": s,
            "close": 100.0 + i + d,
        }
        for i, s in enumerate(symbols)
        for d in range(3)
    ]
    fig = plot_price(pd.DataFrame(rows))
    assert len(fig.data) == 20
    assert [t.name for t in fig.data] == symbols


def test_plot_returns_comparison_fifteen_symbols_traces():
    symbols = [f"STK{i:02d}" for i in range(15)]
    rows = [
        {
            "trade_date": pd.Timestamp("2020-01-06") + pd.Timedelta(days=d),
            "symbol": s,
            "cumulative_return": 0.01 * i + 0.002 * d,
        }
        for i, s in enumerate(symbols)
        for d in range(4)
    ]
    fig = plot_returns_comparison(pd.DataFrame(rows))
    assert len(fig.data) == 15
    assert [t.name for t in fig.data] == symbols


# --- plot_return_distribution ------------------------------------------------

def test_plot_return_distribution_figure_and_traces():
    df = pd.DataFrame({
        "trade_date": pd.to_datetime(["2024-01-02", "2024-01-03"] * 2),
        "symbol": ["A", "A", "B", "B"],
        "return": [0.01, -0.01, 0.02, 0.03],
    })
    fig = plot_return_distribution(df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2  # 每股一个箱线
    assert {t.name for t in fig.data} == {"A", "B"}


def test_plot_return_distribution_symbols_selection():
    df = pd.DataFrame({"symbol": ["A", "B"], "return": [0.01, 0.02]})
    fig = plot_return_distribution(df, symbols=["B"])
    assert len(fig.data) == 1
    assert fig.data[0].name == "B"


def test_plot_return_distribution_missing_column():
    with pytest.raises(DataValidationError):
        plot_return_distribution(pd.DataFrame({"symbol": ["A"], "close": [1.0]}))


# --- plot_rolling_volatility -------------------------------------------------

def test_plot_rolling_volatility_figure_and_traces():
    df = pd.DataFrame({
        "trade_date": pd.to_datetime(["2024-01-02", "2024-01-03"] * 2),
        "symbol": ["A", "A", "B", "B"],
        "volatility_20d": [np.nan, 0.08, np.nan, 0.12],
    })
    fig = plot_rolling_volatility(df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2


def test_plot_rolling_volatility_missing_column():
    with pytest.raises(DataValidationError):
        plot_rolling_volatility(pd.DataFrame({
            "symbol": ["A"], "trade_date": [pd.Timestamp("2024-01-02")],
        }))


def test_plot_rolling_volatility_symbols_selection():
    df = pd.DataFrame({
        "trade_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
        "symbol": ["A", "A"],
        "volatility_20d": [np.nan, 0.08],
    })
    fig = plot_rolling_volatility(df, symbols=["A"])
    assert len(fig.data) == 1


def test_new_chart_functions_do_not_mutate_input():
    df = pd.DataFrame({
        "trade_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
        "symbol": ["A", "A"],
        "return": [0.01, -0.01],
        "volatility_20d": [np.nan, 0.08],
    })
    before = df.copy(deep=True)
    plot_return_distribution(df)
    plot_rolling_volatility(df)
    pd.testing.assert_frame_equal(df, before)


# --- plot_cluster_scatter ----------------------------------------------------

def _profiles_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D", "E"],
            "mean_return": [0.05, 0.01, -0.02, 0.04, -0.03],
            "volatility": [0.10, 0.20, 0.15, 0.12, 0.18],
            "max_drawdown": [-0.05, -0.30, -0.15, -0.08, -0.25],
            "cluster": [0, 1, 2, 0, 2],
        }
    )


def _centers_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cluster": [0, 1, 2],
            "mean_return": [0.045, 0.01, -0.025],
            "volatility": [0.11, 0.20, 0.165],
            "max_drawdown": [-0.065, -0.30, -0.20],
        }
    )


def test_plot_cluster_scatter_figure_and_traces():
    fig = plot_cluster_scatter(_profiles_df())
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 3  # 每股一个箱线 → 每簇一条散点
    assert {t.name for t in fig.data} == {"簇 0", "簇 1", "簇 2"}


def test_plot_cluster_scatter_groups_points_by_cluster():
    fig = plot_cluster_scatter(_profiles_df())
    by_name = {t.name: t for t in fig.data}
    # cluster 0 = A、D；默认 x=volatility、y=mean_return。
    assert list(by_name["簇 0"].x) == [0.10, 0.12]
    assert list(by_name["簇 0"].y) == [0.05, 0.04]
    assert list(by_name["簇 0"].text) == ["A", "D"]


def test_plot_cluster_scatter_distinct_cluster_colors():
    fig = plot_cluster_scatter(_profiles_df())
    colors = [t.marker.color for t in fig.data]
    assert len(set(colors)) == 3


def test_plot_cluster_scatter_with_centers():
    fig = plot_cluster_scatter(_profiles_df(), cluster_centers=_centers_df())
    center = next(t for t in fig.data if t.name == "聚类中心")
    assert list(center.x) == [0.11, 0.20, 0.165]
    assert list(center.y) == [0.045, 0.01, -0.025]
    assert center.marker.symbol == "x"
    # 中心点颜色与对应簇的散点颜色一致。
    point_color = {t.name: t.marker.color for t in fig.data if t.name != "聚类中心"}
    assert list(center.marker.color) == [
        point_color["簇 0"], point_color["簇 1"], point_color["簇 2"],
    ]


def test_plot_cluster_scatter_custom_axes():
    fig = plot_cluster_scatter(_profiles_df(), x="max_drawdown", y="mean_return")
    assert fig.layout.xaxis.title.text == "最大回撤"
    assert fig.layout.yaxis.title.text == "平均日收益率"
    by_name = {t.name: t for t in fig.data}
    assert list(by_name["簇 0"].x) == [-0.05, -0.08]


def test_plot_cluster_scatter_missing_column():
    with pytest.raises(DataValidationError):
        plot_cluster_scatter(_profiles_df().drop(columns=["cluster"]))


def test_plot_cluster_scatter_centers_missing_column():
    with pytest.raises(DataValidationError):
        plot_cluster_scatter(
            _profiles_df(), cluster_centers=_centers_df().drop(columns=["volatility"])
        )


def test_plot_cluster_scatter_invalid_axis():
    with pytest.raises(DataValidationError):
        plot_cluster_scatter(_profiles_df(), x="bogus")
    with pytest.raises(DataValidationError):
        plot_cluster_scatter(_profiles_df(), x="volatility", y="volatility")


def test_plot_cluster_scatter_empty():
    with pytest.raises(NoDataError):
        plot_cluster_scatter(pd.DataFrame())


def test_plot_cluster_scatter_does_not_mutate_input():
    profiles = _profiles_df()
    centers = _centers_df()
    p_before = profiles.copy(deep=True)
    c_before = centers.copy(deep=True)
    plot_cluster_scatter(profiles, cluster_centers=centers)
    pd.testing.assert_frame_equal(profiles, p_before)
    pd.testing.assert_frame_equal(centers, c_before)


# --- plot_cluster_centers ----------------------------------------------------

def test_plot_cluster_centers_figure_and_data():
    fig = plot_cluster_centers(_centers_df())
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 3  # 每簇一条
    assert [t.name for t in fig.data] == ["簇 0", "簇 1", "簇 2"]
    assert fig.layout.barmode == "group"
    expected_x = ["平均日收益率", "波动率", "最大回撤"]
    for t in fig.data:
        assert list(t.x) == expected_x
    # 簇 0 中心：mean_return=0.045, volatility=0.11, max_drawdown=-0.065
    assert list(fig.data[0].y) == [0.045, 0.11, -0.065]


def test_plot_cluster_centers_distinct_colors():
    fig = plot_cluster_centers(_centers_df())
    colors = [t.marker.color for t in fig.data]
    assert len(set(colors)) == 3


def test_plot_cluster_centers_colors_match_scatter():
    scatter = plot_cluster_scatter(_profiles_df())
    bars = plot_cluster_centers(_centers_df())
    scatter_color = {
        t.name: t.marker.color for t in scatter.data if t.name != "聚类中心"
    }
    bar_color = {t.name: t.marker.color for t in bars.data}
    for cid in ("簇 0", "簇 1", "簇 2"):
        assert scatter_color[cid] == bar_color[cid]


def test_plot_cluster_centers_missing_column():
    with pytest.raises(DataValidationError):
        plot_cluster_centers(_centers_df().drop(columns=["volatility"]))


def test_plot_cluster_centers_empty():
    with pytest.raises(NoDataError):
        plot_cluster_centers(pd.DataFrame())


def test_plot_cluster_centers_does_not_mutate_input():
    centers = _centers_df()
    before = centers.copy(deep=True)
    plot_cluster_centers(centers)
    pd.testing.assert_frame_equal(centers, before)


# --- plot_cluster_parallel_coordinates ---------------------------------------

def test_plot_cluster_parallel_coordinates_figure_and_dimensions():
    fig = plot_cluster_parallel_coordinates(_profiles_df())
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert isinstance(fig.data[0], go.Parcoords)
    dims = fig.data[0].dimensions
    assert [d.label for d in dims] == ["平均日收益率", "波动率", "最大回撤"]
    assert list(dims[0].values) == [0.05, 0.01, -0.02, 0.04, -0.03]


def test_plot_cluster_parallel_coordinates_colors_by_cluster():
    fig = plot_cluster_parallel_coordinates(_profiles_df())
    trace = fig.data[0]
    # line.color 是 0..k-1 的簇索引，逐股映射到离散 colorscale。
    assert list(trace.line.color) == [0, 1, 2, 0, 2]
    scatter = plot_cluster_scatter(_profiles_df())
    scatter_color = {
        t.name: t.marker.color for t in scatter.data if t.name != "聚类中心"
    }
    actual = [c for _, c in trace.line.colorscale]
    assert actual == [
        scatter_color["簇 0"], scatter_color["簇 1"], scatter_color["簇 2"],
    ]
    # 颜色条（图例）把簇号标出来，用户能对应「哪色=哪簇」。
    assert list(trace.line.colorbar.tickvals) == [0, 1, 2]
    assert list(trace.line.colorbar.ticktext) == ["簇 0", "簇 1", "簇 2"]


def test_plot_cluster_parallel_coordinates_missing_column():
    with pytest.raises(DataValidationError):
        plot_cluster_parallel_coordinates(_profiles_df().drop(columns=["cluster"]))


def test_plot_cluster_parallel_coordinates_empty():
    with pytest.raises(NoDataError):
        plot_cluster_parallel_coordinates(pd.DataFrame())


def test_plot_cluster_parallel_coordinates_does_not_mutate_input():
    profiles = _profiles_df()
    before = profiles.copy(deep=True)
    plot_cluster_parallel_coordinates(profiles)
    pd.testing.assert_frame_equal(profiles, before)
