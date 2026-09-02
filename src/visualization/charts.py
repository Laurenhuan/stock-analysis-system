"""Reusable Plotly figures owned by Role 3 (金融数据分析与可视化工程师).

Every public function returns a ``plotly.graph_objects.Figure``. The module
performs no Streamlit rendering, CSV access, data acquisition or model
training; Role 1 renders the returned Figure in the appropriate page.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import plotly.graph_objects as go
from pandas import DataFrame
from plotly.graph_objects import Figure

from src.utils.exceptions import DataValidationError, NoDataError

# Brand-neutral colour sequence reused across every figure.
_COLOR_SEQUENCE = [
    "#4C78A8", "#F58518", "#54A24B", "#E45756",
    "#72B7B2", "#B279A2", "#FF9DA6", "#9D755D",
]


def _apply_theme(fig: Figure) -> Figure:
    """Apply the shared visual theme (colours, font, margins, hover)."""
    fig.update_layout(
        template="plotly_white",
        colorway=_COLOR_SEQUENCE,
        font=dict(family="Segoe UI, Arial, sans-serif", size=12),
        margin=dict(l=50, r=30, t=60, b=50),
        hovermode="x unified",
    )
    return fig


def _check_input(df, required: tuple[str, ...], *, label: str) -> None:
    """Raise NoDataError on empty input, DataValidationError on missing columns."""
    if not isinstance(df, DataFrame):
        raise DataValidationError(
            f"{label} 需要 pandas.DataFrame，收到 {type(df).__name__}"
        )
    if df.empty:
        raise NoDataError(f"{label} 输入数据为空")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError(f"{label} 缺少必需字段：{missing}")


def _filter_symbols(df: DataFrame, symbols, *, label: str) -> DataFrame:
    """Keep only the requested symbols; raise NoDataError if none remain."""
    if symbols is None:
        return df
    result = df[df["symbol"].isin(symbols)]
    if result.empty:
        raise NoDataError(f"{label} 未找到所选股票：{list(symbols)}")
    return result


def plot_price(
    df: DataFrame,
    *,
    symbols: Sequence[str] | None = None,
    show_ma: bool = False,
    ma_windows: tuple[int, ...] = (5, 20),
    title: str | None = None,
) -> Figure:
    """Close-price line chart.

    Draws one line per symbol when multiple symbols are present. When the input
    holds a single symbol, ``show_ma=True`` overlays the ``ma5`` / ``ma20``
    columns if they exist.

    ``symbols`` restricts which symbols are drawn (useful when the DataFrame
    holds many stocks); by default every symbol is drawn.
    """
    _check_input(df, ("trade_date", "close"), label="plot_price")
    if "symbol" in df.columns and symbols is not None:
        df = _filter_symbols(df, symbols, label="plot_price")
    fig = go.Figure()
    multi = "symbol" in df.columns and df["symbol"].nunique() > 1

    if multi:
        for symbol, group in df.groupby("symbol"):
            fig.add_trace(
                go.Scatter(
                    x=group["trade_date"], y=group["close"],
                    mode="lines", name=str(symbol),
                )
            )
    else:
        fig.add_trace(
            go.Scatter(x=df["trade_date"], y=df["close"], mode="lines", name="收盘价")
        )
        if show_ma:
            for window in ma_windows:
                col = f"ma{window}"
                if col in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df["trade_date"], y=df[col],
                            mode="lines", name=f"MA{window}",
                        )
                    )

    fig.update_layout(title=title, xaxis_title="交易日期", yaxis_title="收盘价")
    return _apply_theme(fig)


def plot_returns_comparison(
    df: DataFrame,
    *,
    symbols: Sequence[str] | None = None,
    title: str | None = None,
) -> Figure:
    """Multi-symbol cumulative-return comparison line chart.

    ``symbols`` restricts which symbols are drawn; by default every symbol is
    drawn.
    """
    _check_input(
        df, ("trade_date", "symbol", "cumulative_return"),
        label="plot_returns_comparison",
    )
    if symbols is not None:
        df = _filter_symbols(df, symbols, label="plot_returns_comparison")
    fig = go.Figure()
    for symbol, group in df.groupby("symbol"):
        fig.add_trace(
            go.Scatter(
                x=group["trade_date"], y=group["cumulative_return"],
                mode="lines", name=str(symbol),
            )
        )
    fig.update_layout(title=title, xaxis_title="交易日期", yaxis_title="累计收益率")
    return _apply_theme(fig)


def plot_risk_comparison(
    df: DataFrame,
    *,
    symbols: Sequence[str] | None = None,
    title: str | None = None,
) -> Figure:
    """Grouped bar chart comparing per-symbol volatility and max drawdown.

    Expects the output of ``risk_return_summary`` (or any DataFrame with the
    ``symbol``, ``volatility`` and ``max_drawdown`` columns). ``symbols``
    restricts which symbols are drawn; by default every symbol is drawn.
    """
    _check_input(
        df, ("symbol", "volatility", "max_drawdown"), label="plot_risk_comparison"
    )
    if symbols is not None:
        df = _filter_symbols(df, symbols, label="plot_risk_comparison")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["symbol"], y=df["volatility"], name="波动率"))
    fig.add_trace(go.Bar(x=df["symbol"], y=df["max_drawdown"], name="最大回撤"))
    fig.update_layout(title=title, barmode="group", xaxis_title="股票", yaxis_title="值")
    return _apply_theme(fig)


def plot_correlation_matrix(
    corr, *, title: str | None = None, decimals: int = 2
) -> Figure:
    """Heatmap of a cross-symbol correlation matrix.

    ``corr`` is the output of ``src.analysis.eda.correlation_matrix``: a square
    DataFrame whose index and columns are symbols. Values are pinned to [-1, 1]
    and rendered with a diverging red/blue scale.
    """
    if isinstance(corr, DataFrame):
        x_labels = [str(x) for x in corr.columns]
        y_labels = [str(x) for x in corr.index]
        arr = corr.to_numpy(dtype=float)
    else:
        arr = np.asarray(corr, dtype=float)
        x_labels = y_labels = [str(i) for i in range(arr.shape[0])]
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] == 0:
        raise DataValidationError("相关系数矩阵必须是 n×n 非空方阵")
    fig = go.Figure(
        go.Heatmap(
            z=arr,
            x=x_labels,
            y=y_labels,
            zmin=-1.0,
            zmax=1.0,
            colorscale="RdBu_r",
            text=[[f"{v:.{decimals}f}" for v in row] for row in arr],
            texttemplate="%{text}",
            textfont=dict(size=11),
            colorbar=dict(title="相关系数"),
        )
    )
    fig.update_layout(title=title, xaxis_title="股票", yaxis_title="股票")
    return _apply_theme(fig)


def plot_confusion_matrix(
    cm, *, labels: tuple[str, ...] = ("0", "1"), title: str | None = None
) -> Figure:
    """Heatmap of a square confusion matrix.

    Rows are true labels, columns are predicted labels. ``labels`` follows the
    fixed ``[0, 1]`` order required by the supervised-learning Contract.
    """
    arr = np.asarray(cm, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] == 0:
        raise DataValidationError("混淆矩阵必须是 n×n 非空方阵")
    fig = go.Figure(
        go.Heatmap(
            z=arr, x=list(labels), y=list(labels),
            texttemplate="%{z}", textfont=dict(size=16),
            colorscale="Blues", showscale=False,
        )
    )
    fig.update_layout(title=title, xaxis_title="预测标签", yaxis_title="真实标签")
    return _apply_theme(fig)


def plot_actual_vs_predicted(df: DataFrame, *, title: str | None = None) -> Figure:
    """Scatter of predicted vs actual values with a y=x reference line.

    Expects the ``y_true`` and ``y_pred`` columns from a regression prediction
    DataFrame.
    """
    _check_input(df, ("y_true", "y_pred"), label="plot_actual_vs_predicted")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=df["y_true"], y=df["y_pred"], mode="markers", name="样本")
    )
    lo = float(min(df["y_true"].min(), df["y_pred"].min()))
    hi = float(max(df["y_true"].max(), df["y_pred"].max()))
    fig.add_trace(
        go.Scatter(
            x=[lo, hi], y=[lo, hi], mode="lines", name="y=x",
            line=dict(dash="dash", color="grey"),
        )
    )
    fig.update_layout(title=title, xaxis_title="实际值 y_true", yaxis_title="预测值 y_pred")
    return _apply_theme(fig)
