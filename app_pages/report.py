"""Structured quantitative brief for the current workflow state."""

import streamlit as st

from app_pages.shared import cached_market_data, render_provenance
from src.services import build_quant_report, get_market_metadata
from src.utils.exceptions import StockAnalysisError


def _percent(value: object) -> str:
    return "—" if value is None else f"{float(value):.2%}"


multi_query = st.session_state.get("multi_query")
if multi_query is None:
    st.info(
        "量化简报需要一个多股股票池。请先完成多股历史研究。",
        icon=":material/route:",
    )
    if st.button(
        "前往多股历史研究",
        type="primary",
        icon=":material/compare_arrows:",
    ):
        st.switch_page("app_pages/multi_stock.py")
    st.stop()

try:
    with st.spinner("正在恢复股票池并生成分析简报…"):
        market_data = cached_market_data(
            tuple(multi_query["symbols"]),
            multi_query["start_date"],
            multi_query["end_date"],
            multi_query["source"],
        )
    metadata = get_market_metadata(market_data)
except StockAnalysisError as error:
    st.error(f"量化简报数据加载失败：{error}", icon=":material/error:")
    st.stop()

symbols = list(dict.fromkeys(market_data["symbol"].astype(str).tolist()))
requested_focus = st.session_state.get("focus_symbol")
focus_symbol = requested_focus if requested_focus in symbols else symbols[0]
focus_symbol = st.selectbox(
    "简报关注股票",
    options=symbols,
    index=symbols.index(focus_symbol),
    key="report_focus_symbol",
)
st.session_state.focus_symbol = focus_symbol

try:
    with st.spinner("正在组合 EDA、聚类和模型证据…"):
        report = build_quant_report(
            market_data,
            focus_symbol=focus_symbol,
            correlation_method=multi_query["correlation_method"],
        )
except StockAnalysisError as error:
    st.error(f"量化简报生成失败：{error}", icon=":material/error:")
    st.stop()

render_provenance(metadata)
scope = report["scope"]
st.caption(
    f"{scope['股票数量']} 只股票 · {scope['开始日期']} 至 {scope['结束日期']} · "
    f"关注 {scope['关注股票']}"
)

st.subheader("先看结论", icon=":material/summarize:")
for finding in report["core_findings"]:
    st.write(f"• {finding}")

st.subheader("关注股票历史画像")
snapshot = report["focus_snapshot"]
columns = st.columns(4)
columns[0].metric("累计收益", _percent(snapshot["累计收益"]))
columns[1].metric("日波动率", _percent(snapshot["日波动率"]))
columns[2].metric("最大回撤", _percent(snapshot["最大回撤"]))
columns[3].metric(
    "最新收盘价",
    "—" if snapshot["最新收盘价"] is None else f"{float(snapshot['最新收盘价']):.2f}",
)
with st.container(border=True):
    st.markdown(f"**趋势状态**：收盘价{snapshot['收盘价相对MA20']}；MA5{snapshot['MA5相对MA20']}。")
    st.markdown(
        f"**K-Means 横向画像**：{report['cluster_label'] or '当前股票池未形成可用画像'}"
    )

st.subheader("下一交易日模型信号")
class_result = report["classification"]
reg_result = report["regression"]
left, right = st.columns(2)
with left:
    with st.container(border=True):
        st.caption("Decision Tree · 方向")
        if class_result is None:
            st.write("当前样本不足，无法形成方向信号。")
        else:
            st.markdown(f"### {class_result['方向信号']}")
            st.metric(
                "历史 Accuracy",
                _percent(class_result["历史准确率"]),
                delta=(
                    f"{float(class_result['相对基线']) * 100:+.2f} 个百分点"
                    " 相对基线"
                ),
                delta_color="off",
            )
            st.caption(
                f"{class_result['基线名称']}："
                f"{_percent(class_result['较强基线'])}"
            )
            st.markdown(f"**{class_result['可靠性标题']}**")
            st.write(class_result["可靠性说明"])

with right:
    with st.container(border=True):
        st.caption("Linear Regression · 收益率")
        if reg_result is None:
            st.write("当前样本不足，无法形成收益率信号。")
        else:
            st.markdown(f"### {_percent(reg_result['预测收益率'])}")
            signal_metrics = st.columns(2)
            signal_metrics[0].metric(
                "最新收盘价", f"{float(reg_result['最新收盘价']):.2f}"
            )
            signal_metrics[1].metric(
                "模型换算价格", f"{float(reg_result['模型换算价格']):.2f}"
            )
            st.caption(
                f"历史 MAE {_percent(reg_result['历史MAE'])}；"
                f"{reg_result['基线名称']} MAE "
                f"{_percent(reg_result['较强基线MAE'])}；"
                f"R² {float(reg_result['历史R²']):.4f}。"
            )
            st.markdown(f"**{reg_result['可靠性标题']}**")
            st.write(reg_result["可靠性说明"])

if report["model_errors"]:
    with st.expander("查看未生成的模型模块"):
        for error in report["model_errors"]:
            st.caption(error)

st.info(
    "模型换算价格不是目标价或实时成交价。模型信号必须与历史样本外表现和简单基线一起阅读，"
    "预测增量有限时应降低参考权重。最终判断由用户自主作出。",
    icon=":material/info:",
)

st.download_button(
    "下载 Markdown 简报",
    data=report["markdown"],
    file_name=f"{focus_symbol}-量化分析简报.md",
    mime="text/markdown",
    icon=":material/download:",
)

with st.expander("查看完整文本简报"):
    st.markdown(report["markdown"])
