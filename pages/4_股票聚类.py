"""Stock-clustering page backed only by Role 1 Service entry points."""

import streamlit as st

from src.services import (
    get_sample_date_bounds,
    get_sample_symbols,
    load_market_data,
    run_stock_clustering,
)
from src.utils.exceptions import StockAnalysisError


st.set_page_config(page_title="股票聚类", page_icon="🔬", layout="wide")
st.title("🔬 股票聚类分析")
st.markdown(
    "基于 **平均收益率**、**波动率**、**最大回撤** 构建股票画像，"
    "并按 P0 契约使用固定 **K-Means (k=3)**。"
)

symbols = get_sample_symbols()
first_date, last_date = get_sample_date_bounds()

with st.sidebar:
    st.header("聚类条件")
    selected_symbols = st.multiselect(
        "股票（至少 3 只）", options=symbols, default=symbols
    )
    random_state = st.number_input("随机种子", value=42, min_value=0)
    run_clustering = st.button("运行 K-Means", type="primary")

if not run_clustering:
    st.info("默认选择 10 只离线样例股票，点击按钮后运行聚类。")
    st.stop()
if len(selected_symbols) < 3:
    st.warning("固定 k=3 时至少需要 3 只股票。")
    st.stop()

try:
    market_data = load_market_data(
        selected_symbols,
        start_date=first_date,
        end_date=last_date,
        source="sample",
    )
    result = run_stock_clustering(
        market_data, random_state=int(random_state)
    )
except StockAnalysisError as error:
    st.error(str(error))
    st.stop()

st.warning("使用 2024 年离线样例快照；Cluster 编号没有固定的好坏含义。")

profiles = result["profiles"].copy()
profiles["cluster"] = profiles["cluster"].map(lambda value: f"Cluster {value}")
centers = result["cluster_centers"].copy()
centers["cluster"] = centers["cluster"].map(lambda value: f"Cluster {value}")

k_col, stock_col, seed_col = st.columns(3)
k_col.metric("k", result["k"])
stock_col.metric("股票数", len(profiles))
seed_col.metric("随机种子", int(random_state))

st.subheader("股票画像与聚类结果")
st.dataframe(profiles, width="stretch", hide_index=True)

st.subheader("原始尺度聚类中心")
st.dataframe(centers, width="stretch", hide_index=True)

left_chart, right_chart = st.columns(2)
with left_chart:
    st.markdown("**波动率 vs 平均收益率**")
    st.scatter_chart(
        profiles.set_index("symbol"),
        x="volatility",
        y="mean_return",
        color="cluster",
    )
with right_chart:
    st.markdown("**最大回撤 vs 平均收益率**")
    st.scatter_chart(
        profiles.set_index("symbol"),
        x="max_drawdown",
        y="mean_return",
        color="cluster",
    )

st.subheader("分组查看")
for cluster_name in sorted(profiles["cluster"].unique()):
    group = profiles[profiles["cluster"] == cluster_name]
    with st.expander(f"{cluster_name} ({len(group)} 只股票)"):
        st.dataframe(group, width="stretch", hide_index=True)

st.caption(f"聚类特征：{', '.join(result['features'])}")
st.warning("聚类结果仅用于课程演示，不构成投资建议。")
