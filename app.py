"""Streamlit entry point for the Day 1 integration prototype."""

import streamlit as st


st.set_page_config(page_title="证券金融数据分析与可视化系统", page_icon="📈")

st.title("证券金融数据分析与可视化系统")
st.caption("当前版本：Day 1 Integration Prototype")

st.info(
    "数据概览页已经用 Sample Data 打通 Streamlit → Service → DataFrame → 图表链路。"
    "当前数据和基础价格图仅用于集成验证，不是真实行情或正式分析结果。"
)

st.subheader("当前可演示功能")
st.markdown(
    """
- **数据概览**：选择 Sample Data 中的股票和日期，显示行情表与基础收盘价折线图。

**操作入口：** 请在左侧页面导航中打开“数据概览”。

### 后续模块

- **EDA 分析**：后续接入探索性分析与数据可视化。
- **监督学习**：后续接入决策树分类与线性回归。
- **股票聚类**：后续接入 K-Means 与股票风险收益画像。
"""
)

st.warning("DEMO / SAMPLE ONLY：Day 1 原型不提供投资建议。")
