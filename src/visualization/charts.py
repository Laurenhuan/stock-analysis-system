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

# Brand-neutral 20-colour categorical palette (Vega "category20"). Sized for
# the 2-20 stock comparison range so every line / bar keeps a distinct colour;
# the legend label (the symbol) still carries the stock identity when colours
# are ambiguous to the reader.
_COLOR_SEQUENCE = [
    "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD",
    "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF",
    "#AEC7E8", "#FFBB78", "#98DF8A", "#FF9896", "#C5B0D5",
    "#C49C94", "#F7B6D2", "#C7C7C7", "#DBDB8D", "#9EDAE5",
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
    if symbols is not None:
        if "symbol" not in df.columns:
            raise DataValidationError(
                "plot_price 指定了 symbols 选股，但输入缺少 symbol 字段"
            )
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
        # 单一股票时保留其 symbol 作为线名（选股后只剩 1 只的情况同样适用）
        name = str(df["symbol"].iloc[0]) if "symbol" in df.columns else "收盘价"
        fig.add_trace(
            go.Scatter(x=df["trade_date"], y=df["close"], mode="lines", name=name)
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


def plot_candlestick(
    df: DataFrame,
    *,
    symbols: Sequence[str] | None = None,
    title: str | None = None,
) -> Figure:
    """OHLC candlestick chart for exactly one stock.

    Candlesticks are only meaningful for a single symbol at a time, so this
    function renders one stock. When the input holds multiple symbols, pass
    ``symbols`` to select a single one; if more than one symbol remains after
    any selection, a ``DataValidationError`` is raised instead of overlaying
    several stocks into one illegible chart.
    """
    _check_input(
        df, ("trade_date", "open", "high", "low", "close"),
        label="plot_candlestick",
    )
    if symbols is not None:
        if "symbol" not in df.columns:
            raise DataValidationError(
                "plot_candlestick 指定了 symbols 选股，但输入缺少 symbol 字段"
            )
        df = _filter_symbols(df, symbols, label="plot_candlestick")
    if "symbol" in df.columns and df["symbol"].nunique() > 1:
        raise DataValidationError(
            "plot_candlestick 仅支持单只股票，请通过 symbols 参数选择一只股票"
        )
    # 单只股票（或选股后只剩 1 只）时保留 symbol 作为图例名
    name = str(df["symbol"].iloc[0]) if "symbol" in df.columns else "价格"
    fig = go.Figure(
        go.Candlestick(
            x=df["trade_date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name=name,
        )
    )
    fig.update_layout(title=title, xaxis_title="交易日期", yaxis_title="价格")
    fig.update_xaxes(rangeslider_visible=False)
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


def plot_return_distribution(
    df: DataFrame,
    *,
    symbols: Sequence[str] | None = None,
    title: str | None = None,
) -> Figure:
    """Box plot of daily returns, one box per symbol.

    Each box shows the median, quartiles and outlying tail days of a stock's
    daily ``return`` — the standard EDA view of return-distribution shape.
    ``symbols`` restricts which symbols are drawn; by default every symbol is
    drawn.
    """
    _check_input(df, ("symbol", "return"), label="plot_return_distribution")
    if symbols is not None:
        df = _filter_symbols(df, symbols, label="plot_return_distribution")
    fig = go.Figure()
    for symbol, group in df.groupby("symbol"):
        fig.add_trace(go.Box(y=group["return"], name=str(symbol)))
    fig.update_layout(title=title, xaxis_title="股票", yaxis_title="日收益率")
    return _apply_theme(fig)


def plot_rolling_volatility(
    df: DataFrame,
    *,
    symbols: Sequence[str] | None = None,
    title: str | None = None,
) -> Figure:
    """Line chart of the rolling 20-day volatility over time, per symbol.

    Expects Role 2's ``volatility_20d`` field (rolling sample std of daily
    returns, ddof=1, not annualized). Leading NaNs before the window matures are
    left as gaps, not filled. ``symbols`` restricts which symbols are drawn.
    """
    _check_input(
        df, ("trade_date", "symbol", "volatility_20d"),
        label="plot_rolling_volatility",
    )
    if symbols is not None:
        df = _filter_symbols(df, symbols, label="plot_rolling_volatility")
    fig = go.Figure()
    for symbol, group in df.groupby("symbol"):
        fig.add_trace(
            go.Scatter(
                x=group["trade_date"], y=group["volatility_20d"],
                mode="lines", name=str(symbol),
            )
        )
    fig.update_layout(title=title, xaxis_title="交易日期", yaxis_title="20 日波动率")
    return _apply_theme(fig)


def plot_correlation_matrix(
    corr, *, title: str | None = None, decimals: int = 2
) -> Figure:
    """Heatmap of a cross-symbol correlation matrix.

    ``corr`` is either the output of ``src.analysis.eda.correlation_matrix`` —
    a square DataFrame whose index and columns hold the symbol labels — or a
    square array-like. Inputs are validated before any index access: axis
    labels must match, values must be finite and within [-1, 1], and the
    diverging colour scale is centred on ``zmid=0``.
    """
    if isinstance(corr, DataFrame):
        if corr.empty:
            raise DataValidationError("相关系数矩阵 DataFrame 不能为空")
        if list(corr.index) != list(corr.columns):
            raise DataValidationError(
                "相关系数矩阵 DataFrame 的 index 与 columns 标签必须一致且顺序相同"
            )
        try:
            arr = corr.to_numpy(dtype=float)
        except ValueError as exc:
            raise DataValidationError(
                f"相关系数矩阵 DataFrame 含非数值内容：{exc}"
            ) from exc
        labels = [str(x) for x in corr.index]
    else:
        arr = np.asarray(corr, dtype=float)
        labels = [str(i) for i in range(arr.shape[0])] if arr.ndim == 2 else []
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] == 0:
        raise DataValidationError("相关系数矩阵必须是 n×n 非空方阵")
    if not np.isfinite(arr).all():
        raise DataValidationError("相关系数矩阵含 NaN/inf，无法渲染热力图")
    if bool(((arr < -1.0) | (arr > 1.0)).any()):
        raise DataValidationError("相关系数取值必须在 [-1, 1] 内")
    fig = go.Figure(
        go.Heatmap(
            z=arr,
            x=labels,
            y=labels,
            zmin=-1.0,
            zmax=1.0,
            zmid=0.0,
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
    """Heatmap of the 2×2 binary confusion matrix.

    Rows are true labels, columns are predicted labels, following the fixed
    ``[0, 1]`` order required by the supervised-learning Contract. Exactly two
    labels must be provided.
    """
    arr = np.asarray(cm, dtype=float)
    if arr.shape != (2, 2):
        raise DataValidationError("按二分类 Contract，混淆矩阵必须是 2×2")
    if len(labels) != 2:
        raise DataValidationError("二分类混淆矩阵必须提供恰好两个标签")
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
    DataFrame; both must be finite numbers.
    """
    _check_input(df, ("y_true", "y_pred"), label="plot_actual_vs_predicted")
    if not np.isfinite(df[["y_true", "y_pred"]].to_numpy(dtype=float)).all():
        raise DataValidationError("y_true/y_pred 含 NaN 或 inf，无法绘图")
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


# Clustering profile features and their Chinese axis labels, shared by the
# cluster scatter plot below. These mirror the Clustering Contract's three
# Stock Profile features (mean_return / volatility / max_drawdown).
_CLUSTER_FEATURES = ("mean_return", "volatility", "max_drawdown")

_FEATURE_AXIS_LABELS = {
    "mean_return": "平均日收益率",
    "volatility": "波动率",
    "max_drawdown": "最大回撤",
}


def _cluster_label(cluster_id) -> str:
    """Format a cluster id as a neutral ``簇 N`` label.

    Cluster ids carry no fixed financial meaning (Clustering Contract §4), so
    the label only states the id; the reader interprets a cluster through its
    centre and members rather than a preset judgement.
    """
    try:
        num = float(cluster_id)
        if num.is_integer():
            return f"簇 {int(num)}"
    except (TypeError, ValueError):
        pass
    return f"簇 {cluster_id}"


def plot_cluster_scatter(
    profiles: DataFrame,
    *,
    cluster_centers: DataFrame | None = None,
    x: str = "volatility",
    y: str = "mean_return",
    title: str | None = None,
) -> Figure:
    """2D scatter of the clustered stock profiles, coloured by cluster.

    Expects Role 5's ``profiles`` output — a DataFrame holding ``symbol``, the
    three profile features (``mean_return``, ``volatility``, ``max_drawdown``)
    and the model-private ``cluster`` column. ``x``/``y`` pick which two of the
    three features form the axes (default: the classic risk/return plane, with
    volatility on x and mean return on y). Each point is one stock, coloured by
    its cluster.

    When ``cluster_centers`` (Role 5's ``cluster_centers`` output) is supplied,
    every cluster's centre is overlaid as a larger ``x`` marker in the matching
    colour so the reader can see where each group sits.
    """
    required = ("symbol", "mean_return", "volatility", "max_drawdown", "cluster")
    _check_input(profiles, required, label="plot_cluster_scatter")
    if x not in _CLUSTER_FEATURES or y not in _CLUSTER_FEATURES:
        raise DataValidationError(
            "plot_cluster_scatter 的 x/y 必须属于聚类特征 "
            f"{list(_CLUSTER_FEATURES)}"
        )
    if x == y:
        raise DataValidationError("plot_cluster_scatter 的 x 与 y 不能相同")

    # Assign one stable colour per cluster id (from the union of profiles and
    # centers) so points and their centre share the same colour.
    cluster_ids = set(profiles["cluster"].dropna())
    if cluster_centers is not None:
        _check_input(
            cluster_centers, ("cluster", x, y),
            label="plot_cluster_scatter 的 cluster_centers",
        )
        cluster_ids |= set(cluster_centers["cluster"].dropna())
    color_by_cluster = {
        cid: _COLOR_SEQUENCE[i % len(_COLOR_SEQUENCE)]
        for i, cid in enumerate(sorted(cluster_ids))
    }

    fig = go.Figure()

    for cluster_id, group in profiles.groupby("cluster", sort=True):
        fig.add_trace(
            go.Scatter(
                x=group[x],
                y=group[y],
                mode="markers",
                name=_cluster_label(cluster_id),
                text=group["symbol"].astype(str),
                marker=dict(
                    size=9,
                    color=color_by_cluster.get(cluster_id, _COLOR_SEQUENCE[0]),
                    opacity=0.75,
                ),
            )
        )

    if cluster_centers is not None:
        center_colors = [
            color_by_cluster.get(cid, _COLOR_SEQUENCE[0])
            for cid in cluster_centers["cluster"]
        ]
        fig.add_trace(
            go.Scatter(
                x=cluster_centers[x],
                y=cluster_centers[y],
                mode="markers",
                name="聚类中心",
                text=[_cluster_label(c) for c in cluster_centers["cluster"]],
                marker=dict(
                    size=16,
                    symbol="x",
                    color=center_colors,
                    line=dict(width=2, color="#333333"),
                ),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title=_FEATURE_AXIS_LABELS[x],
        yaxis_title=_FEATURE_AXIS_LABELS[y],
    )
    _apply_theme(fig)
    # 散点图逐点悬停更自然，覆盖主题默认的按 x 轴统一悬停。
    fig.update_layout(hovermode="closest")
    return fig


def plot_cluster_centers(
    cluster_centers: DataFrame,
    *,
    title: str | None = None,
) -> Figure:
    """Grouped bar chart comparing every cluster's centre across the features.

    Expects Role 5's ``cluster_centers`` output — a DataFrame with ``cluster``
    plus the three profile features (``mean_return``, ``volatility``,
    ``max_drawdown``). Each feature is one x-category; one bar per cluster per
    feature, coloured by cluster so it lines up with ``plot_cluster_scatter``.

    ``max_drawdown`` is ``<= 0``, so its bars extend downward — a shorter bar
    means a shallower (milder) drawdown. Cluster numbers carry no fixed
    financial meaning, so the legend only states the id.
    """
    required = ("cluster", "mean_return", "volatility", "max_drawdown")
    _check_input(cluster_centers, required, label="plot_cluster_centers")

    cluster_ids = sorted(set(cluster_centers["cluster"].dropna()))
    color_by_cluster = {
        cid: _COLOR_SEQUENCE[i % len(_COLOR_SEQUENCE)]
        for i, cid in enumerate(cluster_ids)
    }
    feature_labels = [_FEATURE_AXIS_LABELS[f] for f in _CLUSTER_FEATURES]

    fig = go.Figure()
    for _, row in cluster_centers.iterrows():
        cid = row["cluster"]
        fig.add_trace(
            go.Bar(
                x=feature_labels,
                y=[row[f] for f in _CLUSTER_FEATURES],
                name=_cluster_label(cid),
                marker_color=color_by_cluster.get(cid, _COLOR_SEQUENCE[0]),
            )
        )

    fig.update_layout(
        title=title,
        barmode="group",
        xaxis_title="聚类特征",
        yaxis_title="簇中心值",
    )
    return _apply_theme(fig)


def plot_cluster_parallel_coordinates(
    profiles: DataFrame,
    *,
    title: str | None = None,
) -> Figure:
    """Parallel-coordinates view of the clustered stock profiles.

    Each stock is one polyline crossing the three feature axes (mean return,
    volatility, max drawdown); lines are coloured by cluster, so a cluster reads
    as a bundle of same-coloured lines. This shows every stock's complete
    profile at once — the scatter only shows two of three features, and the
    centre bars aggregate instead of showing individual stocks.
    """
    required = ("symbol", "mean_return", "volatility", "max_drawdown", "cluster")
    _check_input(profiles, required, label="plot_cluster_parallel_coordinates")

    cluster_ids = sorted(set(profiles["cluster"].dropna()))
    k = len(cluster_ids)
    color_by_cluster = {
        cid: _COLOR_SEQUENCE[i % len(_COLOR_SEQUENCE)]
        for i, cid in enumerate(cluster_ids)
    }

    # 用「簇索引 0..k-1 + 离散 colorscale」给每条线着色。这样每个簇精确对应
    # 一个纯色，避免把字符串色值交给 Parcoords 后被当作连续色阶插值混色。
    if k == 1:
        line_values = [0] * len(profiles)
        colorscale = [
            [0.0, color_by_cluster[cluster_ids[0]]],
            [1.0, color_by_cluster[cluster_ids[0]]],
        ]
        cmax = 1.0
    else:
        index_by_cluster = {cid: i for i, cid in enumerate(cluster_ids)}
        line_values = [index_by_cluster.get(c, 0) for c in profiles["cluster"]]
        colorscale = [
            [i / (k - 1), color_by_cluster[cid]]
            for i, cid in enumerate(cluster_ids)
        ]
        cmax = float(k - 1)

    dimensions = [
        dict(label=_FEATURE_AXIS_LABELS[f], values=profiles[f].tolist())
        for f in _CLUSTER_FEATURES
    ]

    fig = go.Figure(
        go.Parcoords(
            line=dict(
                color=line_values,
                colorscale=colorscale,
                cmin=0,
                cmax=cmax,
                showscale=True,
                colorbar=dict(
                    tickvals=list(range(k)),
                    ticktext=[_cluster_label(c) for c in cluster_ids],
                    title="簇",
                ),
            ),
            dimensions=dimensions,
        )
    )
    fig.update_layout(title=title)
    return _apply_theme(fig)
