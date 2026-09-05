"""Multi-stock workspace sharing one query across EDA and clustering."""

from datetime import date

import pandas as pd
import streamlit as st

from app_pages.shared import (
    DEFAULT_START_DATE,
    SOURCE_OPTIONS,
    cached_market_data,
    render_provenance,
    render_stock_search,
)
from src.services import (
    build_eda_dashboard,
    get_clustering_date_diagnostics,
    get_market_metadata,
    get_market_summary,
    get_sample_symbols,
    prepare_symbol_selection,
    run_stock_clustering_dashboard,
)
from src.utils.exceptions import StockAnalysisError


_CORRELATION_METHOD_LABELS = {
    "spearman": "Spearman（秩相关，默认）",
    "pearson": "Pearson（线性相关）",
    "kendall": "Kendall τ（秩一致性）",
}


def _percent_table(
    frame: pd.DataFrame,
    *,
    percentage_columns: tuple[str, ...],
    labels: dict[str, str],
) -> pd.DataFrame:
    display = frame.copy()
    for column in percentage_columns:
        if column in display.columns:
            display[column] = pd.to_numeric(
                display[column], errors="coerce"
            ).mul(100)
    return display.rename(columns=labels)


def _render_insights(insights: list[dict[str, str]]) -> None:
    if not insights:
        st.caption("当前数据未形成这一主题的额外结论。")
        return
    for insight in insights:
        with st.container(border=True):
            st.markdown(f"**{insight['title']}**")
            st.write(insight["finding"])
            with st.expander("查看依据与解释"):
                st.caption(f"数据依据：{insight['evidence']}")
                st.write(insight["interpretation"])


def _render_core_insights(insights: list[dict[str, str]]) -> None:
    if not insights:
        st.info("当前样本不足以形成核心摘要。")
        return
    for offset in range(0, len(insights), 3):
        columns = st.columns(3)
        for column, insight in zip(columns, insights[offset : offset + 3]):
            with column:
                with st.container(border=True):
                    st.caption(insight["category"].replace("_", " ").upper())
                    st.markdown(f"**{insight['title']}**")
                    st.write(insight["finding"])


st.caption("一组多股与日期条件，复用于问题驱动 EDA 和固定 KMeans(k=3) 聚类。")

searched_symbols, searched_names = render_stock_search("multi")
symbol_options = list(dict.fromkeys([*get_sample_symbols(), *searched_symbols]))

with st.form("multi_stock_query"):
    symbols_col, source_col = st.columns([2, 1.2], vertical_alignment="bottom")
    with symbols_col:
        symbol_inputs = st.multiselect(
            "股票代码（2～20 只）",
            options=symbol_options,
            default=get_sample_symbols()[:5],
            format_func=lambda value: (
                f"{value} · {searched_names[value]}"
                if value in searched_names
                else value
            ),
            accept_new_options=True,
            max_selections=20,
            placeholder="选择示例或输入 600519.SH",
            help="EDA 至少 2 只、聚类至少 3 只；支持裸代码并自动补全交易所。",
            key="multi_symbol_inputs",
            persist_state="session",
        )
    with source_col:
        source_label = st.segmented_control(
            "数据来源",
            options=list(SOURCE_OPTIONS),
            default=list(SOURCE_OPTIONS)[0],
            key="multi_source_input",
            persist_state="session",
        )

    start_col, end_col, method_col = st.columns(3, vertical_alignment="bottom")
    with start_col:
        start_date = st.date_input(
            "开始日期",
            value=DEFAULT_START_DATE,
            max_value=date.today(),
            key="multi_start_date",
            persist_state="session",
        )
    with end_col:
        end_date = st.date_input(
            "结束日期",
            value=date.today(),
            max_value=date.today(),
            key="multi_end_date",
            persist_state="session",
        )
    with method_col:
        correlation_method = st.selectbox(
            "相关系数方法",
            options=tuple(_CORRELATION_METHOD_LABELS),
            format_func=_CORRELATION_METHOD_LABELS.__getitem__,
            help=(
                "三种方法均已实现。修改选项后，需要再次点击“加载并比较”"
                "才会按新方法重新计算。"
            ),
            key="multi_correlation_method",
            persist_state="session",
        )
        st.caption("修改相关系数方法后，请重新点击“加载并比较”应用选择。")

    submitted = st.form_submit_button(
        "加载并比较",
        type="primary",
        icon=":material/play_arrow:",
    )

if submitted:
    try:
        symbols = prepare_symbol_selection(
            symbol_inputs, min_count=2, max_count=20
        )
        if source_label not in SOURCE_OPTIONS:
            raise ValueError("请选择数据来源")
    except (StockAnalysisError, ValueError) as error:
        st.error(str(error), icon=":material/error:")
    else:
        st.session_state.multi_query = {
            "symbols": symbols,
            "start_date": start_date,
            "end_date": end_date,
            "source": SOURCE_OPTIONS[source_label],
            "correlation_method": correlation_method,
        }

query = st.session_state.get("multi_query")
if query is None:
    st.info(
        "选择至少 2 只股票后加载。EDA 和聚类将共享同一份历史数据。",
        icon=":material/touch_app:",
    )
    st.stop()

try:
    with st.spinner("正在加载多股历史行情…"):
        market_data = cached_market_data(
            tuple(query["symbols"]),
            query["start_date"],
            query["end_date"],
            query["source"],
        )
    metadata = get_market_metadata(market_data)
    summary = get_market_summary(market_data)
    loaded_symbols = list(
        dict.fromkeys(market_data["symbol"].astype(str).tolist())
    )
    loaded_by_code = {
        symbol.split(".", maxsplit=1)[0]: symbol for symbol in loaded_symbols
    }
    active_symbols = list(
        dict.fromkeys(
            loaded_by_code.get(
                "".join(
                    character
                    for character in requested
                    if character.isdigit()
                ),
                requested,
            )
            for requested in query["symbols"]
        )
    )
    active_symbols.extend(
        symbol for symbol in loaded_symbols if symbol not in active_symbols
    )
except StockAnalysisError as error:
    st.error(f"多股历史行情加载失败：{error}", icon=":material/error:")
    st.stop()

render_provenance(metadata)
st.caption(
    f"当前查询：{len(active_symbols)} 只股票 · "
    f"{query['start_date']} 至 {query['end_date']}"
)
with st.container(horizontal=True, gap="medium"):
    st.metric("有效股票", summary["symbol_count"], border=True)
    st.metric("总记录数", summary["row_count"], border=True)
    st.metric("首个交易日", summary["first_date"], border=True)
    st.metric("最后交易日", summary["last_date"], border=True)

st.warning(
    "所有结论仅描述所选历史区间，不代表未来表现，不构成投资建议。",
    icon=":material/warning:",
)

eda_tab, clustering_tab = st.tabs(
    ["EDA 分析", "股票聚类"],
    key="multi_workspace_tabs",
    on_change="rerun",
)

if eda_tab.open:
    with eda_tab:
        with st.expander("图表设置", icon=":material/tune:"):
            candlestick_symbol = st.selectbox(
                "K 线股票",
                options=active_symbols,
                key="multi_candlestick_symbol",
                persist_state="page",
            )
            st.caption(
                "相关系数："
                f"{_CORRELATION_METHOD_LABELS[query['correlation_method']]}；"
                "K 线仅改变价格图，不改变其他统计结论。"
            )
        try:
            dashboard = build_eda_dashboard(
                market_data,
                correlation_method=query["correlation_method"],
                candlestick_symbol=candlestick_symbol,
            )
        except StockAnalysisError as error:
            st.error(f"EDA 运行失败：{error}", icon=":material/error:")
        else:
            presentation = dashboard["presentation"]
            sections = presentation["sections"]

            st.subheader("先看结论")
            st.caption("从已有统计结果中筛选最多 5 条核心发现；详细依据按主题展开。")
            _render_core_insights(presentation["core_insights"])
            if presentation["summary_sentences"]:
                st.markdown("**本次区间摘要**")
                for sentence in presentation["summary_sentences"]:
                    st.write(f"• {sentence}")

            (
                performance_tab,
                risk_tab,
                relation_tab,
                trend_tab,
                details_tab,
            ) = st.tabs(
                ["收益表现", "风险画像", "股票关系", "趋势状态", "详细统计"]
            )

            with performance_tab:
                st.subheader("谁的区间表现更突出？")
                st.caption(
                    "先读相对结论，再查看累计收益、日均收益、胜率与收益波动的证据。"
                )
                _render_insights(sections["performance"])
                returns = _percent_table(
                    dashboard["returns"],
                    percentage_columns=(
                        "mean_return",
                        "cumulative_return",
                        "win_rate",
                        "std_return",
                    ),
                    labels={
                        "symbol": "股票",
                        "mean_return": "日均收益（%）",
                        "cumulative_return": "累计收益（%）",
                        "win_rate": "上涨日占比（%）",
                        "std_return": "日收益标准差（%）",
                    },
                )
                st.dataframe(returns, hide_index=True)
                st.plotly_chart(dashboard["returns_figure"], width="stretch")

            with risk_tab:
                st.subheader("收益伴随了怎样的风险？")
                st.caption(
                    "波动率衡量日收益离散程度；最大回撤衡量区间内从高点到低点的最深跌幅。"
                    "两者分开绘制，避免正负量纲混在同一坐标轴。"
                )
                _render_insights(sections["risk"])
                risk = _percent_table(
                    dashboard["risk_return"],
                    percentage_columns=(
                        "mean_return",
                        "volatility",
                        "max_drawdown",
                    ),
                    labels={
                        "symbol": "股票",
                        "mean_return": "日均收益（%）",
                        "volatility": "日波动率（%）",
                        "max_drawdown": "最大回撤（%）",
                    },
                )
                st.dataframe(risk, hide_index=True)
                left, right = st.columns(2)
                with left:
                    st.markdown("**日波动率对比（%）**")
                    st.bar_chart(risk.set_index("股票")[["日波动率（%）"]])
                with right:
                    st.markdown("**最大回撤对比（%）**")
                    st.bar_chart(risk.set_index("股票")[["最大回撤（%）"]])
                with st.expander("查看滚动波动率时序图"):
                    st.plotly_chart(
                        dashboard["rolling_volatility_figure"],
                        width="stretch",
                    )

            with relation_tab:
                st.subheader("这些股票的日收益是否同步？")
                st.caption(
                    "相关系数接近 1 表示同向变化较多，接近 -1 表示反向变化较多；"
                    "相关不等于因果，也可能随区间变化。"
                )
                _render_insights(sections["correlation"])
                if dashboard["correlation_figure"] is None:
                    st.info(
                        "股票间重叠交易日不足，相关矩阵暂不展示；"
                        "其他 EDA 结果仍基于各自有效历史区间。"
                    )
                else:
                    st.plotly_chart(
                        dashboard["correlation_figure"], width="stretch"
                    )
                    with st.expander("查看相关系数矩阵"):
                        st.dataframe(dashboard["correlation"])

            with trend_tab:
                st.subheader("最新价格处于怎样的历史位置？")
                st.caption(
                    "这里只比较收盘价、MA5 与 MA20 的相对位置，不把均线状态解释为买卖信号。"
                )
                _render_insights(sections["trend"])
                trend = _percent_table(
                    presentation["trend_snapshot"],
                    percentage_columns=("cumulative_return",),
                    labels={
                        "symbol": "股票",
                        "close": "最新收盘价",
                        "ma5": "MA5",
                        "ma20": "MA20",
                        "cumulative_return": "区间累计收益（%）",
                        "price_vs_ma20": "收盘价相对 MA20",
                        "ma5_vs_ma20": "MA5 相对 MA20",
                    },
                )
                st.dataframe(trend, hide_index=True)

            with details_tab:
                st.subheader("收益分布与极端交易日")
                _render_insights(sections["distribution"])
                distribution = dashboard["return_distribution"].rename(
                    columns={
                        "symbol": "股票",
                        "skewness": "偏度",
                        "kurtosis": "超额峰度",
                        "n": "有效日收益数",
                    }
                )
                extremes = _percent_table(
                    dashboard["extreme_returns"],
                    percentage_columns=("max_return", "min_return"),
                    labels={
                        "symbol": "股票",
                        "max_return": "最高单日收益（%）",
                        "max_return_date": "发生日期",
                        "min_return": "最低单日收益（%）",
                        "min_return_date": "发生日期 ",
                    },
                )
                st.dataframe(distribution, hide_index=True)
                st.dataframe(extremes, hide_index=True)
                st.plotly_chart(
                    dashboard["return_distribution_figure"],
                    width="stretch",
                )
                with st.expander("查看价格、K 线与描述统计"):
                    st.plotly_chart(
                        dashboard["candlestick_figure"], width="stretch"
                    )
                    st.plotly_chart(
                        dashboard["price_figure"], width="stretch"
                    )
                    st.dataframe(dashboard["descriptive_statistics"])

            with st.expander(
                "数据质量与样本范围", icon=":material/fact_check:"
            ):
                _render_insights(sections["data_quality"])
                st.markdown("**分股日期范围**")
                st.dataframe(dashboard["date_ranges"], hide_index=True)
                st.markdown("**缺失值**")
                st.dataframe(dashboard["missing_values"], hide_index=True)

if clustering_tab.open:
    with clustering_tab:
        st.caption(
            "每个点代表一只股票；特征为平均收益率、波动率和最大回撤。"
            "P0 契约固定 KMeans(k=3)。"
        )
        if len(active_symbols) < 3:
            st.warning(
                "固定 k=3 时至少需要 3 只股票；请修改上方共享条件。",
                icon=":material/warning:",
            )
        else:
            try:
                dashboard = run_stock_clustering_dashboard(
                    market_data, random_state=42
                )
            except StockAnalysisError as error:
                diagnostics = (
                    get_clustering_date_diagnostics(market_data)
                    if "比较区间不一致" in str(error)
                    else None
                )
                if diagnostics is None or diagnostics["is_consistent"]:
                    st.error(
                        f"股票聚类运行失败：{error}",
                        icon=":material/error:",
                    )
                else:
                    st.error(
                        "所选股票的有效历史区间不一致，"
                        "暂时无法按相同区间公平聚类。",
                        icon=":material/date_range:",
                    )
                    st.info(
                        "建议把查询区间调整到 "
                        f"{diagnostics['common_start']} 至 "
                        f"{diagnostics['common_end']}，或者移除下面"
                        "历史范围受限的股票后重新加载。"
                    )
                    limiting_ranges = diagnostics[
                        "limiting_ranges"
                    ].copy()
                    limiting_ranges["限制原因"] = limiting_ranges.apply(
                        lambda row: "、".join(
                            reason
                            for limited, reason in (
                                (
                                    row["limits_common_start"],
                                    "起始日期较晚",
                                ),
                                (
                                    row["limits_common_end"],
                                    "截止日期较早",
                                ),
                            )
                            if limited
                        ),
                        axis=1,
                    )
                    st.markdown("**限制共同区间的股票**")
                    st.dataframe(
                        limiting_ranges[
                            [
                                "symbol",
                                "start_date",
                                "end_date",
                                "限制原因",
                            ]
                        ].rename(
                            columns={
                                "symbol": "股票代码",
                                "start_date": "最早有效日期",
                                "end_date": "最晚有效日期",
                            }
                        ),
                        hide_index=True,
                    )
            else:
                result = dashboard["result"]
                interpretation = dashboard["interpretation"]
                labels = interpretation["cluster_label"]
                profiles = result["profiles"].copy()
                centers = result["cluster_centers"].copy()
                profiles["cluster_label"] = profiles["cluster"].map(labels)
                centers["cluster_label"] = centers["cluster"].map(labels)
                profiles["cluster"] = profiles["cluster"].map(
                    lambda value: f"Cluster {value}"
                )
                centers["cluster"] = centers["cluster"].map(
                    lambda value: f"Cluster {value}"
                )
                sample_scope = interpretation["样本范围"]
                st.success(
                    "已根据本次所选股票和历史区间的聚类中心动态生成中文画像。",
                    icon=":material/label:",
                )
                st.caption(
                    f"展示区间：{sample_scope['起始日期']} 至 "
                    f"{sample_scope['截止日期']} · "
                    f"{sample_scope['股票数量']} 只股票。"
                    "Cluster 编号没有固定好坏含义。"
                )
                st.dataframe(profiles, hide_index=True)
                st.subheader("原始尺度聚类中心")
                st.dataframe(centers, hide_index=True)
                with st.expander("查看标签依据"):
                    st.write(interpretation["标签依据"])
                    st.caption(interpretation["免责声明"])
                left_chart, right_chart = st.columns(2)
                with left_chart:
                    st.scatter_chart(
                        profiles.set_index("symbol"),
                        x="volatility",
                        y="mean_return",
                        color="cluster_label",
                    )
                with right_chart:
                    st.scatter_chart(
                        profiles.set_index("symbol"),
                        x="max_drawdown",
                        y="mean_return",
                        color="cluster_label",
                    )

st.divider()
st.subheader("下一步：选择一只关注股票", icon=":material/arrow_forward:")
st.caption(
    "选择后将继承当前股票池的日期和数据来源，进入单股模型分析，无需重复输入。"
)
current_focus = st.session_state.get("focus_symbol")
default_index = (
    active_symbols.index(current_focus)
    if current_focus in active_symbols
    else 0
)
focus_symbol = st.selectbox(
    "关注股票",
    options=active_symbols,
    index=default_index,
    key="multi_focus_selection",
)
if st.button(
    "进入单股模型分析",
    type="primary",
    icon=":material/show_chart:",
    key="multi_to_single",
):
    st.session_state.focus_symbol = focus_symbol
    st.session_state.single_query = {
        "symbol": focus_symbol,
        "start_date": query["start_date"],
        "end_date": query["end_date"],
        "source": query["source"],
    }
    st.switch_page("app_pages/single_stock.py")
