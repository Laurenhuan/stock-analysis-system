"""Streamlit entry point for the stock-analysis application."""

import streamlit as st


st.set_page_config(
    page_title="证券金融数据分析与可视化系统",
    page_icon=":material/query_stats:",
    layout="wide",
)

# Per-user selections survive page switches. Expensive data stays in bounded
# st.cache_data functions in app_pages.shared rather than session state.
st.session_state.setdefault("single_query", None)
st.session_state.setdefault("multi_query", None)

page = st.navigation(
    [
        st.Page(
            "app_pages/home.py",
            title="项目首页",
            icon=":material/home:",
            default=True,
        ),
        st.Page(
            "app_pages/single_stock.py",
            title="单股研究",
            icon=":material/show_chart:",
        ),
        st.Page(
            "app_pages/multi_stock.py",
            title="多股比较",
            icon=":material/compare_arrows:",
        ),
    ],
    position="top",
)

st.title(page.title, icon=page.icon)
page.run()
