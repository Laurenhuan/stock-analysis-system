"""Multi-stock workspace sharing one query across EDA and clustering."""

from datetime import date

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
    get_market_metadata,
    get_market_summary,
    get_sample_symbols,
    prepare_symbol_selection,
    run_stock_clustering_dashboard,
)
from src.utils.exceptions import StockAnalysisError


st.caption("一组多股与日期条件，复用于 EDA 和固定 KMeans(k=3) 聚类。")

searched_symbols, searched_names = render_stock_search("multi")
symbol_options = list(
    dict.fromkeys([*get_sample_symbols(), *searched_symbols])
)

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
            options=("spearman", "pearson", "kendall"),
            key="multi_correlation_method",
            persist_state="session",
        )

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
        symbol.split(".", maxsplit=1)[0]: symbol
        for symbol in loaded_symbols
    }
    active_symbols = list(
        dict.fromkeys(
            loaded_by_code.get(
                "".join(character for character in requested if character.isdigit()),
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

eda_tab, clustering_tab = st.tabs(
    ["EDA 分析", "股票聚类"],
    key="multi_workspace_tabs",
    on_change="rerun",
)

if eda_tab.open:
    with eda_tab:
        candlestick_symbol = st.selectbox(
            "K 线股票",
            options=active_symbols,
            key="multi_candlestick_symbol",
            persist_state="page",
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
            insight_tab, risk_tab, chart_tab, quality_tab = st.tabs(
                ["分析结论", "收益与风险", "图表", "数据质量"]
            )
            with insight_tab:
                category_names = {
                    "performance": "表现",
                    "risk": "风险",
                    "distribution": "收益分布",
                    "trend": "趋势",
                    "correlation": "相关性",
                    "data_quality": "数据质量",
                }
                for insight in dashboard["insights"]:
                    with st.container(border=True):
                        st.badge(
                            category_names.get(insight["category"], "结论")
                        )
                        st.subheader(insight["title"])
                        st.write(insight["finding"])
                        st.caption(f"依据：{insight['evidence']}")
                        st.write(f"解释：{insight['interpretation']}")
                        st.caption(insight["caveat"])
            with risk_tab:
                st.subheader("区间收益与风险")
                st.dataframe(dashboard["returns"], hide_index=True)
                st.dataframe(dashboard["risk_return"], hide_index=True)
                st.plotly_chart(dashboard["risk_figure"], width="stretch")
                st.subheader("收益分布与极端交易日")
                st.dataframe(
                    dashboard["return_distribution"], hide_index=True
                )
                st.dataframe(dashboard["extreme_returns"], hide_index=True)
                st.plotly_chart(
                    dashboard["return_distribution_figure"],
                    width="stretch",
                )
                st.plotly_chart(
                    dashboard["rolling_volatility_figure"],
                    width="stretch",
                )
            with chart_tab:
                st.plotly_chart(
                    dashboard["candlestick_figure"], width="stretch"
                )
                st.plotly_chart(dashboard["price_figure"], width="stretch")
                st.plotly_chart(
                    dashboard["returns_figure"], width="stretch"
                )
                if dashboard["correlation_figure"] is None:
                    st.info(
                        "股票间重叠交易日不足，相关矩阵暂不展示；"
                        "其他 EDA 结果仍基于各自有效历史区间。"
                    )
                else:
                    st.plotly_chart(
                        dashboard["correlation_figure"], width="stretch"
                    )
            with quality_tab:
                st.subheader("分股日期范围")
                st.dataframe(dashboard["date_ranges"], hide_index=True)
                st.subheader("缺失值")
                st.dataframe(dashboard["missing_values"], hide_index=True)
                st.subheader("描述统计")
                st.dataframe(dashboard["descriptive_statistics"])

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
                st.error(f"股票聚类运行失败：{error}", icon=":material/error:")
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
                st.success(
                    "已根据本次所选股票和历史区间的聚类中心动态生成中文画像。",
                    icon=":material/label:",
                )
                st.caption("Cluster 编号没有固定好坏含义，应结合动态画像和中心值阅读。")
                st.dataframe(profiles, hide_index=True)
                st.subheader("原始尺度聚类中心")
                st.dataframe(centers, hide_index=True)
                with st.expander("查看标签依据与样本范围"):
                    st.write(interpretation["标签依据"])
                    st.json(interpretation["样本范围"])
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

st.warning(
    "所有结论仅描述所选历史区间，不代表未来表现，不构成投资建议。",
    icon=":material/warning:",
)
