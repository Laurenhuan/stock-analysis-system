"""Application-facing orchestration for EDA tables and Plotly figures."""

from __future__ import annotations

from typing import TypedDict

from pandas import DataFrame
from plotly.graph_objects import Figure

from src.analysis.insights import EdaInsight, build_eda_insights
from src.analysis.eda import (
    correlation_matrix,
    date_range_summary,
    describe_statistics,
    extreme_returns_summary,
    missing_values_summary,
    return_distribution_summary,
    returns_comparison,
    risk_return_summary,
)
from src.visualization.charts import (
    plot_candlestick,
    plot_correlation_matrix,
    plot_price,
    plot_return_distribution,
    plot_returns_comparison,
    plot_risk_comparison,
    plot_rolling_volatility,
)
from src.utils.exceptions import InsufficientDataError


class EdaDashboard(TypedDict):
    """Stable bundle consumed by the EDA Streamlit page."""

    descriptive_statistics: DataFrame
    date_ranges: DataFrame
    missing_values: DataFrame
    returns: DataFrame
    risk_return: DataFrame
    return_distribution: DataFrame
    extreme_returns: DataFrame
    correlation: DataFrame | None
    insights: list[EdaInsight]
    price_figure: Figure
    candlestick_figure: Figure
    returns_figure: Figure
    risk_figure: Figure
    return_distribution_figure: Figure
    rolling_volatility_figure: Figure
    correlation_figure: Figure | None


def get_analysis_status() -> dict[str, str]:
    """Expose truthful readiness after the D2 integration."""
    return {
        "eda": "ready",
        "classification": "ready",
        "regression": "ready",
        "clustering": "ready",
    }


def build_price_figure(data: DataFrame, *, show_ma: bool = True) -> Figure:
    """Build the shared price figure through Role 3's visualization API."""
    return plot_price(data, show_ma=show_ma, title="收盘价与移动平均")


def build_eda_dashboard(
    data: DataFrame,
    *,
    correlation_method: str = "spearman",
    candlestick_symbol: str | None = None,
) -> EdaDashboard:
    """Coordinate EDA tables, conclusions and reusable figures for one page."""
    risk = risk_return_summary(data)
    try:
        corr = correlation_matrix(data, method=correlation_method)
    except InsufficientDataError:
        corr = None
    candle_symbol = candlestick_symbol or str(sorted(data["symbol"].unique())[0])
    return EdaDashboard(
        descriptive_statistics=describe_statistics(data),
        date_ranges=date_range_summary(data),
        missing_values=missing_values_summary(data).reset_index(names="field"),
        returns=returns_comparison(data),
        risk_return=risk,
        return_distribution=return_distribution_summary(data),
        extreme_returns=extreme_returns_summary(data),
        correlation=corr,
        insights=build_eda_insights(
            data, correlation_method=correlation_method
        ),
        price_figure=plot_price(data, title="多股收盘价对比"),
        candlestick_figure=plot_candlestick(
            data,
            symbols=[candle_symbol],
            title=f"{candle_symbol} K 线图",
        ),
        returns_figure=plot_returns_comparison(
            data, title="多股累计收益率对比"
        ),
        risk_figure=plot_risk_comparison(
            risk, title="波动率与最大回撤对比"
        ),
        return_distribution_figure=plot_return_distribution(
            data, title="日收益率分布"
        ),
        rolling_volatility_figure=plot_rolling_volatility(
            data, title="20 日滚动波动率"
        ),
        correlation_figure=(
            plot_correlation_matrix(
                corr,
                title=f"{correlation_method.title()} 日收益率相关系数",
            )
            if corr is not None
            else None
        ),
    )
