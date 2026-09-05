"""Streamlit entry point for the stock-analysis application."""

import streamlit as st


st.set_page_config(
    page_title="证券数据分析与决策参考平台",
    page_icon=":material/query_stats:",
    layout="wide",
)

st.session_state.setdefault("single_query", None)
st.session_state.setdefault("multi_query", None)
st.session_state.setdefault("focus_symbol", None)

page = st.navigation(
    [
        st.Page(
            "app_pages/home.py",
            title="首页",
            icon=":material/home:",
            default=True,
        ),
        st.Page(
            "app_pages/multi_stock.py",
            title="多股历史研究",
            icon=":material/compare_arrows:",
        ),
        st.Page(
            "app_pages/single_stock.py",
            title="单股模型分析",
            icon=":material/show_chart:",
        ),
        st.Page(
            "app_pages/report.py",
            title="量化分析简报",
            icon=":material/article:",
        ),
        st.Page(
            "app_pages/about.py",
            title="项目介绍",
            icon=":material/info:",
            url_path="about.html",
        ),
    ],
    position="top",
)

if page.title != "首页":
    st.title(page.title, icon=page.icon)
page.run()
