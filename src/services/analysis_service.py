"""Application-facing orchestration for EDA tables and Plotly figures."""

from __future__ import annotations

from typing import TypedDict

import pandas as pd
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


class EdaPresentation(TypedDict):
    """User-reading-order view assembled from existing EDA outputs."""

    core_insights: list[EdaInsight]
    summary_sentences: list[str]
    sections: dict[str, list[EdaInsight]]
    trend_snapshot: DataFrame


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
    presentation: EdaPresentation
    price_figure: Figure
    candlestick_figure: Figure
    returns_figure: Figure
    risk_figure: Figure
    return_distribution_figure: Figure
    rolling_volatility_figure: Figure
    correlation_figure: Figure | None


_CORE_INSIGHT_TITLES = (
    "累计收益最高",
    "回撤最深",
    "波动最大",
    "相关性最高",
    "当前波动率水平",
)


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


def _trend_snapshot(data: DataFrame) -> DataFrame:
    """Create a compact neutral view from already-computed MA columns."""
    columns = ["symbol", "close", "ma5", "ma20", "cumulative_return"]
    if any(column not in data.columns for column in columns):
        return DataFrame(columns=[*columns, "price_vs_ma20", "ma5_vs_ma20"])
    latest = (
        data.sort_values(["symbol", "trade_date"])
        .groupby("symbol", sort=True, as_index=False)
        .tail(1)[columns]
        .reset_index(drop=True)
    )

    def compare(left: float, right: float, label: str) -> str:
        if pd.isna(left) or pd.isna(right):
            return "尚未形成"
        if left > right:
            return f"高于{label}"
        if left < right:
            return f"低于{label}"
        return f"等于{label}"

    latest["price_vs_ma20"] = [
        compare(close, ma20, "MA20")
        for close, ma20 in zip(latest["close"], latest["ma20"])
    ]
    latest["ma5_vs_ma20"] = [
        compare(ma5, ma20, "MA20")
        for ma5, ma20 in zip(latest["ma5"], latest["ma20"])
    ]
    return latest


def _build_presentation(
    data: DataFrame,
    insights: list[EdaInsight],
) -> EdaPresentation:
    sections = {
        category: [
            insight for insight in insights if insight["category"] == category
        ]
        for category in (
            "performance",
            "risk",
            "correlation",
            "trend",
            "distribution",
            "data_quality",
        )
    }
    core: list[EdaInsight] = []
    for title in _CORE_INSIGHT_TITLES:
        match = next(
            (insight for insight in insights if insight["title"] == title),
            None,
        )
        if match is not None:
            core.append(match)
    if not any(item["category"] == "correlation" for item in core):
        correlation_fallback = next(
            (
                insight
                for insight in sections["correlation"]
                if insight["title"]
                in {"相关性", "相关性样本较少", "相关性样本不足"}
            ),
            None,
        )
        if correlation_fallback is not None:
            core.append(correlation_fallback)
    core = core[:5]
    return EdaPresentation(
        core_insights=core,
        summary_sentences=[item["finding"] for item in core[:4]],
        sections=sections,
        trend_snapshot=_trend_snapshot(data),
    )


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
    insights = build_eda_insights(data, correlation_method=correlation_method)
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
        insights=insights,
        presentation=_build_presentation(data, insights),
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
