"""Project landing page; intentionally performs no market-data request."""

import streamlit as st

from src.services import get_analysis_status


st.caption("从在线行情到统计分析与机器学习的课程实践项目")
st.markdown(
    "选择一只股票研究行情与监督学习，或选择多只股票完成横向比较与聚类。"
)

statuses = get_analysis_status()
ready_count = sum(status == "ready" for status in statuses.values())

with st.container(horizontal=True, gap="medium"):
    st.metric("已接入分析模块", f"{ready_count}/4", border=True)
    st.metric("历史数据字段", "8 项", border=True)
    st.metric("实时快照字段", "12 项", border=True)
    st.metric("聚类类别", "固定 k=3", border=True)

left, right = st.columns(2, gap="large")
with left:
    with st.container(border=True):
        st.subheader("单股研究", icon=":material/show_chart:")
        st.write("一次选择股票和日期，同时查看行情、决策树分类与线性回归。")
        st.markdown(
            "- 在线历史日线和独立实时快照\n"
            "- K 线、移动平均与标准行情\n"
            "- 次日方向分类与次日收益率回归"
        )
        if st.button(
            "进入单股研究",
            type="primary",
            icon=":material/arrow_forward:",
            key="home_single",
        ):
            st.switch_page("app_pages/single_stock.py")

with right:
    with st.container(border=True):
        st.subheader("多股比较", icon=":material/compare_arrows:")
        st.write("一次选择多只股票和日期，共享同一份历史数据完成 EDA 与聚类。")
        st.markdown(
            "- 表现、风险、趋势与相关性结论\n"
            "- 多股票价格、收益和风险图表\n"
            "- 固定 KMeans(k=3) 股票画像"
        )
        if st.button(
            "进入多股比较",
            icon=":material/arrow_forward:",
            key="home_multi",
        ):
            st.switch_page("app_pages/multi_stock.py")

st.subheader("数据链路", icon=":material/database:")
st.markdown(
    "**AkShare 是 Python 数据接口库**；历史行情由其封装的东方财富接口获取，"
    "失败时回退至腾讯接口。实时快照独立使用东方财富并回退新浪。"
)
st.caption(
    "在线结果不会写入本地 CSV。页面会显示 data_source、provider 和抓取时间，"
    "离线样例也会被明确标注。"
)

st.info(
    "本系统用于课程中的历史数据分析和算法演示。实时快照可能延迟；"
    "所有分析仅描述所选历史区间，不代表未来表现，不构成投资建议。",
    icon=":material/info:",
)
