"""Market overview page backed by the formal Service Layer."""

import streamlit as st

from src.services import (
    build_price_figure,
    get_market_metadata,
    get_sample_date_bounds,
    get_sample_symbols,
    load_market_data,
    load_realtime_quotes,
)
from src.utils.exceptions import StockAnalysisError


SOURCE_OPTIONS = {
    "离线样例（稳定演示）": "sample",
    "自动选择数据源": "auto",
    "AkShare 在线行情": "akshare",
}

st.set_page_config(page_title="数据概览", page_icon="📋", layout="wide")
st.title("📋 数据概览")
st.caption("数据流：Streamlit → Service → Role 2 获取/清洗/公共特征")

symbols = get_sample_symbols()
first_date, last_date = get_sample_date_bounds()

with st.sidebar:
    st.header("查询条件")
    source_label = st.selectbox("数据来源", options=list(SOURCE_OPTIONS))
    selected_symbol = st.selectbox("股票代码", options=symbols)
    start_date = st.date_input("开始日期", value=first_date)
    end_date = st.date_input("结束日期", value=last_date)
    load_data = st.button("加载行情", type="primary")
    load_realtime = st.button("查询实时快照")

if not load_data and not load_realtime:
    st.info("请选择查询条件，再加载日线行情或查询实时快照。")
    st.stop()

if load_realtime:
    try:
        realtime_data = load_realtime_quotes(selected_symbol)
        realtime_metadata = get_market_metadata(realtime_data)
    except StockAnalysisError as error:
        st.error(f"实时行情查询失败：{error}")
    else:
        st.subheader("实时行情快照")
        st.caption(
            "来源："
            f"{realtime_metadata['data_source']} / "
            f"{realtime_metadata['provider'] or 'unknown'}；"
            f"抓取时间：{realtime_metadata['fetched_at'] or '供应商未提供'}。"
            "快照可能存在延迟，不代表交易所逐笔实时行情。"
        )
        st.dataframe(realtime_data, width="stretch", hide_index=True)

if not load_data:
    st.stop()

try:
    market_data = load_market_data(
        selected_symbol,
        start_date=start_date,
        end_date=end_date,
        source=SOURCE_OPTIONS[source_label],
    )
    metadata = get_market_metadata(market_data)
    price_figure = build_price_figure(market_data)
except StockAnalysisError as error:
    st.error(str(error))
    st.stop()

if metadata["is_sample"]:
    st.warning("当前为离线样例快照，不是实时行情。")
else:
    st.success(f"当前数据来源：{metadata['data_source']}")
if metadata["provider"]:
    st.caption(f"数据提供方：{metadata['provider']}")
if metadata["fetched_at"]:
    st.caption(f"抓取时间：{metadata['fetched_at']}")
if metadata["fallback_reason"]:
    st.caption(f"回退原因：{metadata['fallback_reason']}")

left, middle, right = st.columns(3)
left.metric("记录数", len(market_data))
middle.metric("首个交易日", market_data["trade_date"].min().date().isoformat())
right.metric("最后交易日", market_data["trade_date"].max().date().isoformat())

st.subheader("标准行情与公共指标")
st.dataframe(market_data, width="stretch", hide_index=True)

st.subheader("收盘价与移动平均")
st.plotly_chart(price_figure, width="stretch")
st.caption("前 4/19 个交易日的 MA5/MA20 为空是滚动窗口的正常预热期。")
