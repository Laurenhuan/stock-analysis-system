"""Shared presentation helpers for the Streamlit workspaces."""

from __future__ import annotations

from datetime import date

import streamlit as st

from src.services import load_market_data, load_realtime_quotes, search_stocks
from src.utils.exceptions import StockAnalysisError


SOURCE_OPTIONS = {
    "AkShare 在线（东财 → 腾讯）": "akshare",
    "离线样例（稳定演示）": "sample",
}
DEFAULT_START_DATE = date(2024, 1, 2)
HISTORY_CACHE_TTL_SECONDS = 15 * 60
REALTIME_CACHE_TTL_SECONDS = 30
STOCK_DIRECTORY_CACHE_TTL_SECONDS = 6 * 60 * 60


@st.cache_data(
    ttl=HISTORY_CACHE_TTL_SECONDS,
    max_entries=64,
    show_spinner=False,
)
def cached_market_data(
    symbols: tuple[str, ...],
    start_date: date,
    end_date: date,
    source: str,
):
    """Cache an immutable query tuple, not widget or session objects."""
    return load_market_data(
        symbols,
        start_date=start_date,
        end_date=end_date,
        source=source,
        fallback=False,
    )


@st.cache_data(
    ttl=REALTIME_CACHE_TTL_SECONDS,
    max_entries=64,
    show_spinner=False,
)
def cached_realtime_quotes(symbols: tuple[str, ...]):
    """Cache delayed quote snapshots briefly to avoid provider hammering."""
    return load_realtime_quotes(symbols)


@st.cache_data(
    ttl=STOCK_DIRECTORY_CACHE_TTL_SECONDS,
    max_entries=128,
    show_spinner=False,
)
def cached_stock_search(query: str, limit: int = 20):
    """Cache the online code/name directory without persisting a local CSV."""
    return search_stocks(query, limit=limit)


def render_stock_search(key_prefix: str) -> tuple[list[str], dict[str, str]]:
    """Render an optional online lookup and return selectable search results."""
    result_key = f"{key_prefix}_stock_search_results"
    records = st.session_state.get(result_key, [])
    with st.expander(
        "按代码或名称查找股票",
        icon=":material/search:",
    ):
        query_col, button_col = st.columns(
            [4, 1], vertical_alignment="bottom"
        )
        with query_col:
            search_query = st.text_input(
                "股票代码或名称",
                placeholder="例如：600519、贵州茅台或茅台",
                key=f"{key_prefix}_stock_search_query",
            )
        with button_col:
            search_clicked = st.button(
                "查找",
                icon=":material/search:",
                key=f"{key_prefix}_stock_search_button",
                width="stretch",
            )

        if search_clicked:
            if not search_query.strip():
                st.warning("请输入股票代码或名称。")
                records = []
            else:
                try:
                    with st.spinner("正在查询沪深 A 股目录…"):
                        result = cached_stock_search(search_query.strip())
                except StockAnalysisError as error:
                    st.warning(
                        f"在线目录暂不可用：{error}。仍可在下方直接输入代码。"
                    )
                    records = []
                else:
                    records = result.to_dict("records")
                    if not records:
                        st.info("未找到匹配股票，可检查关键词或直接输入标准代码。")
            st.session_state[result_key] = records

        if records:
            st.dataframe(
                records,
                hide_index=True,
                width="stretch",
                column_config={
                    "symbol": "股票代码",
                    "name": "股票名称",
                    "market": "市场",
                },
            )
            st.caption("搜索结果已加入下方股票选择框；在线目录结果不会写入本地 CSV。")

    symbols = [str(record["symbol"]) for record in records]
    names = {
        str(record["symbol"]): str(record["name"])
        for record in records
    }
    return symbols, names


def render_provenance(metadata: dict) -> None:
    """Render provider provenance consistently on both workspaces."""
    if metadata["is_sample"]:
        st.warning(
            "当前使用离线样例快照，不是最新行情。",
            icon=":material/warning:",
        )
    else:
        st.success(
            f"已加载在线历史行情：{metadata['data_source']}",
            icon=":material/cloud_done:",
        )

    details = [f"Provider：{metadata['provider'] or 'unknown'}"]
    if metadata["fetched_at"]:
        details.append(f"抓取时间：{metadata['fetched_at']}")
    if metadata["fallback_reason"]:
        details.append(f"回退原因：{metadata['fallback_reason']}")
    st.caption(" · ".join(details))


def render_realtime_snapshot(quotes, metadata: dict) -> None:
    """Render the independent realtime schema without treating it as history."""
    st.caption(
        "实时快照独立于历史日线，仅用于查看当前行情，不参与 EDA 或模型训练。"
    )
    st.caption(
        f"来源：{metadata['data_source']} / {metadata['provider'] or 'unknown'}；"
        f"抓取时间：{metadata['fetched_at'] or '供应商未提供'}。"
    )
    st.dataframe(
        quotes,
        hide_index=True,
        column_config={
            "price": st.column_config.NumberColumn("最新价", format="%.2f"),
            "change": st.column_config.NumberColumn("涨跌额", format="%.2f"),
            "pct_change": st.column_config.NumberColumn(
                "涨跌幅", format="%.2f%%"
            ),
            "volume": st.column_config.NumberColumn("成交量（股）"),
            "amount": st.column_config.NumberColumn("成交额（元）"),
        },
    )
