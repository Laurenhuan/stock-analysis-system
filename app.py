"""Streamlit entry point for the stock analysis system."""

import streamlit as st


st.set_page_config(page_title="证券金融数据分析与可视化系统", page_icon="📈")

st.title("证券金融数据分析与可视化系统")
st.write(
    "本项目面向 A 股历史行情数据，计划整合数据处理、探索性分析、"
    "机器学习与结果可视化。"
)

st.info(
    "当前处于 Architecture Bootstrap 阶段。页面与模块仅提供工程骨架，"
    "尚未接入真实数据或业务算法。"
)

st.subheader("分析模块")
st.markdown(
    """
- **数据概览**：后续接入行情获取、清洗与公共金融指标。
- **EDA 分析**：后续接入探索性分析与数据可视化。
- **监督学习**：后续接入决策树分类与线性回归。
- **股票聚类**：后续接入 K-Means 与股票风险收益画像。
"""
)
