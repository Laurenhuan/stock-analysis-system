"""Exploratory analysis page backed only by Role 1 Service entry points."""

import streamlit as st

from src.services import (
    build_eda_dashboard,
    get_market_metadata,
    get_sample_date_bounds,
    get_sample_symbols,
    load_market_data,
)
from src.utils.exceptions import StockAnalysisError


st.set_page_config(page_title="EDA 分析", page_icon="📊", layout="wide")
st.title("📊 EDA 分析")
st.caption("通过 Service 组合 Role 2 公共特征与 Role 3 分析/可视化函数。")

symbols = get_sample_symbols()
first_date, last_date = get_sample_date_bounds()

with st.sidebar:
    st.header("分析条件")
    selected_symbols = st.multiselect(
        "股票（至少 2 只）",
        options=symbols,
        default=symbols[:5],
    )
    start_date = st.date_input("开始日期", value=first_date)
    end_date = st.date_input("结束日期", value=last_date)
    correlation_method = st.selectbox(
        "相关系数方法", options=("spearman", "pearson", "kendall")
    )
    candlestick_symbol = st.selectbox(
        "K 线股票",
        options=selected_symbols or symbols,
    )
    run_analysis = st.button("运行 EDA", type="primary")

if not run_analysis:
    st.info("请选择至少 2 只股票后运行分析。")
    st.stop()
if len(selected_symbols) < 2:
    st.warning("至少需要 2 只股票才能完成多股对比与相关分析。")
    st.stop()

try:
    market_data = load_market_data(
        selected_symbols,
        start_date=start_date,
        end_date=end_date,
        source="sample",
    )
    metadata = get_market_metadata(market_data)
    dashboard = build_eda_dashboard(
        market_data,
        correlation_method=correlation_method,
        candlestick_symbol=candlestick_symbol,
    )
except StockAnalysisError as error:
    st.error(str(error))
    st.stop()

if metadata["is_sample"]:
    st.warning("本页使用 2024 年离线样例快照，结果仅用于课程演示。")

insight_tab, overview_tab, return_tab, chart_tab = st.tabs(
    ("分析结论", "数据质量与描述统计", "收益与风险", "图表与相关性")
)

with insight_tab:
    st.subheader("所选历史区间说明了什么")
    st.caption("以下发现由当前数据动态计算，用于回答表现、风险、趋势和相关性问题。")
    category_names = {
        "performance": "表现",
        "risk": "风险",
        "trend": "趋势",
        "correlation": "相关性",
        "data_quality": "数据质量",
    }
    current_category = None
    for insight in dashboard["insights"]:
        category = insight["category"]
        if category != current_category:
            st.markdown(f"### {category_names.get(category, category)}")
            current_category = category
        st.markdown(f"#### {insight['title']}")
        st.write(insight["finding"])
        st.caption(f"依据：{insight['evidence']}")
        st.write(f"解释：{insight['interpretation']}")
        st.caption(insight["caveat"])

with overview_tab:
    st.subheader("日期范围")
    st.dataframe(dashboard["date_ranges"], width="stretch", hide_index=True)
    st.subheader("缺失值")
    st.caption("收益率和滚动指标的前导空值是窗口预热的预期结果。")
    st.dataframe(dashboard["missing_values"], width="stretch", hide_index=True)
    st.subheader("分股描述统计")
    st.dataframe(dashboard["descriptive_statistics"], width="stretch")

with return_tab:
    st.subheader("收益对比")
    st.dataframe(dashboard["returns"], width="stretch", hide_index=True)
    st.subheader("风险收益摘要")
    st.dataframe(dashboard["risk_return"], width="stretch", hide_index=True)
    st.plotly_chart(dashboard["risk_figure"], width="stretch")

with chart_tab:
    st.plotly_chart(dashboard["candlestick_figure"], width="stretch")
    st.plotly_chart(dashboard["price_figure"], width="stretch")
    st.plotly_chart(dashboard["returns_figure"], width="stretch")
    st.plotly_chart(dashboard["correlation_figure"], width="stretch")
    with st.expander("查看相关系数表"):
        st.dataframe(dashboard["correlation"], width="stretch")
