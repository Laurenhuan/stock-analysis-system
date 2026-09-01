"""Day 1 market overview page using temporary sample data."""

import plotly.express as px
import streamlit as st

from src.services.market_service import (
    get_demo_date_bounds,
    get_demo_symbols,
    get_market_overview,
)
from src.utils.exceptions import DataValidationError, InvalidSymbolError, NoDataError


st.title("数据概览")
st.caption("Day 1 Integration Prototype")
st.warning(
    "DEMO / SAMPLE ONLY：当前使用合成 Sample Data，"
    "仅验证页面、Service、DataFrame 和图表链路。"
)

symbols = get_demo_symbols()

with st.sidebar:
    st.header("查询条件")
    selected_symbol = st.selectbox("股票代码", options=symbols)
    first_date, last_date = get_demo_date_bounds(selected_symbol)
    start_date = st.date_input(
        "开始日期",
        value=first_date,
        min_value=first_date,
        max_value=last_date,
    )
    end_date = st.date_input(
        "结束日期",
        value=last_date,
        min_value=first_date,
        max_value=last_date,
    )
    load_data = st.button("加载 Sample Data", type="primary")

if not load_data:
    st.info("请在左侧选择股票和日期，然后点击“加载 Sample Data”。")
    st.stop()

try:
    overview = get_market_overview(
        symbol=selected_symbol,
        start_date=start_date,
        end_date=end_date,
    )
except DataValidationError as error:
    st.error(str(error))
    st.stop()
except InvalidSymbolError as error:
    st.error(str(error))
    st.stop()
except NoDataError as error:
    st.warning(str(error))
    st.stop()

st.subheader("Sample 行情数据")
st.dataframe(overview, width="stretch", hide_index=True)

st.subheader("基础收盘价图")
# TEMPORARY / DAY-1 PROTOTYPE: Role 3 will replace this demo chart.
price_figure = px.line(
    overview,
    x="trade_date",
    y="close",
    markers=True,
    title=f"{selected_symbol} Sample Close Price",
)
price_figure.update_layout(xaxis_title="交易日期", yaxis_title="收盘价")
st.plotly_chart(price_figure, width="stretch")

st.caption(
    "数据流：Streamlit 输入 → Market Service → Sample CSV → 日期/股票筛选 → "
    "DataFrame → 表格与 Demo 图。"
)
