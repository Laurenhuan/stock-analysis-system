"""Streamlit entry point for the D2 integrated application."""

import streamlit as st

from src.services import get_analysis_status


st.set_page_config(
    page_title="证券金融数据分析与可视化系统",
    page_icon="📈",
    layout="wide",
)

st.title("证券金融数据分析与可视化系统")
st.caption("当前版本：D2 跨模块集成")

statuses = get_analysis_status()
ready_count = sum(status == "ready" for status in statuses.values())
st.success(
    f"已接入 {ready_count}/4 个分析模块：EDA、线性回归与 K-Means 聚类可用。"
)
st.info(
    "Decision Tree 分类尚未合并，监督学习页会保留明确的待接入状态，"
    "不伪造分类结果。"
)

st.subheader("可演示功能")
st.markdown(
    """
- **数据概览**：通过正式数据层获取、清洗并计算公共金融指标。
- **EDA 分析**：查看描述统计、缺失值、收益风险对比与相关系数。
- **监督学习**：演示线性回归的 MAE、R² 和实际值/预测值。
- **股票聚类**：使用固定 `KMeans(k=3)` 展示风险收益画像。
"""
)

st.warning(
    "默认使用 Role 2 的 2024 年离线样例快照，非实时行情，不构成投资建议。"
)
