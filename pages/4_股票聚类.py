"""Stock clustering page — Role 5 聚类分析页面。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.models.unsupervised.clustering import (
    FEATURE_COLS,
    build_stock_profiles,
    run_clustering,
)

st.set_page_config(page_title="股票聚类", page_icon="🔬")
st.title("🔬 股票聚类分析")
st.markdown("基于 K-Means 的股票风险收益画像聚类。每只股票根据 **平均收益率**、**波动率**、**最大回撤** 三个特征进行分组。")

# ── 侧边栏参数 ──────────────────────────────────────────
st.sidebar.header("聚类参数")
n_clusters = st.sidebar.slider("聚类数 k", min_value=2, max_value=6, value=3)
random_state = st.sidebar.number_input("随机种子", value=42, min_value=0)

# ── 加载数据 ────────────────────────────────────────────
csv_path = Path(__file__).resolve().parent.parent / "data" / "sample" / "sample_market_data.csv"

if not csv_path.exists():
    st.error("❌ 未找到 Sample Data 文件，请先运行数据准备脚本。")
    st.stop()

market_df = pd.read_csv(csv_path)
market_df["trade_date"] = pd.to_datetime(market_df["trade_date"])

st.subheader("📂 原始行情数据")
st.caption(f"共 {market_df['symbol'].nunique()} 只股票，{len(market_df)} 条记录")
st.dataframe(market_df.head(20), use_container_width=True)

# ── 构建 Profile Table ──────────────────────────────────
try:
    profiles = build_stock_profiles(market_df)
except Exception as e:
    st.error(f"构建 Profile 失败: {e}")
    st.stop()

st.subheader("📊 股票画像 (Stock Profile)")
st.caption("每只股票一行，包含三个风险收益特征")
st.dataframe(profiles, use_container_width=True)

# ── 聚类 ────────────────────────────────────────────────
try:
    result = run_clustering(profiles, random_state=random_state)
except Exception as e:
    st.error(f"聚类失败: {e}")
    st.stop()

st.subheader("🔬 K-Means 聚类结果")

# 聚类结果表格
result_df = result["profiles"].copy()
result_df["cluster"] = result_df["cluster"].astype(str)
st.dataframe(result_df, use_container_width=True)

# ── 聚类中心 ────────────────────────────────────────────
st.subheader("📍 聚类中心")
st.caption("每个 cluster 的特征均值（原始尺度）")
centers_df = result["cluster_centers"].copy()
centers_df["cluster"] = centers_df["cluster"].astype(str)
st.dataframe(centers_df, use_container_width=True)

# ── 散点图：mean_return vs volatility ───────────────────
st.subheader("📈 聚类可视化")
fig = px.scatter(
    result_df,
    x="volatility",
    y="mean_return",
    color="cluster",
    hover_name="symbol",
    title="股票聚类散点图 (Volatility vs Mean Return)",
    labels={
        "volatility": "波动率",
        "mean_return": "平均收益率",
        "cluster": "Cluster",
    },
    width=700,
    height=500,
)
st.plotly_chart(fig, use_container_width=True)

# ── 散点图：mean_return vs max_drawdown ─────────────────
fig2 = px.scatter(
    result_df,
    x="max_drawdown",
    y="mean_return",
    color="cluster",
    hover_name="symbol",
    title="股票聚类散点图 (Max Drawdown vs Mean Return)",
    labels={
        "max_drawdown": "最大回撤",
        "mean_return": "平均收益率",
        "cluster": "Cluster",
    },
    width=700,
    height=500,
)
st.plotly_chart(fig2, use_container_width=True)

# ── 按 cluster 分组展示 ────────────────────────────────
st.subheader("📋 按 Cluster 分组")
for c in sorted(result_df["cluster"].unique()):
    group = result_df[result_df["cluster"] == c]
    with st.expander(f"Cluster {c} ({len(group)} 只股票)"):
        st.dataframe(group, use_container_width=True)

# ── 底部说明 ────────────────────────────────────────────
st.divider()
st.markdown(
    """
**注意事项：**
- Cluster 编号（0, 1, 2）没有天然好坏含义，需要根据聚类中心的实际数值来解释。
- 标准化（StandardScaler）是必要的，因为 K-Means 基于距离计算，不同特征的尺度差异会影响聚类结果。
"""
)
