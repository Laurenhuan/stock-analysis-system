"""Product landing page for the end-to-end analysis workflow."""

import streamlit as st

from src.services import get_analysis_status


st.markdown(
    """
    <style>
    .product-hero {
        padding: clamp(1.2rem, 3vw, 2.6rem) 0 1.5rem;
        max-width: 960px;
    }
    .product-kicker {
        color: #60A5FA;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
    }
    .product-subtitle {
        color: #CBD5E1;
        font-size: clamp(1.05rem, 2vw, 1.35rem);
        line-height: 1.75;
        max-width: 760px;
        margin-top: 0.8rem;
    }
    .stage-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 2.2rem;
        margin: 1.2rem 0 2.8rem;
    }
    .stage-number {
        color: #60A5FA;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.12em;
    }
    .stage-grid h3 {
        margin: 0.45rem 0;
        font-size: 1.1rem;
    }
    .stage-grid p {
        color: #94A3B8;
        line-height: 1.65;
        margin: 0;
    }
    @media (max-width: 760px) {
        .stage-grid {
            grid-template-columns: 1fr;
            gap: 1.3rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="product-hero">
      <div class="product-kicker">Historical evidence · Model reference · Structured brief</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.title("证券数据分析与决策参考平台")
st.markdown(
    """
    <p class="product-subtitle">
      从多股历史研究，到单股模型研判，再到自动量化分析简报。
      把分散的数据、图表和模型结果组织成一条可理解、可核验的分析路径。
    </p>
    """,
    unsafe_allow_html=True,
)
st.caption("平台提供分析证据和模型参考，不替用户作出买卖决定。")

multi_entry, single_entry = st.columns(2, gap="large")
with multi_entry:
    st.markdown("#### 还没有明确关注股票？")
    st.caption("先比较一组股票的历史收益、风险、趋势、相关性和股票画像。")
    if st.button(
        "开始多股研究",
        type="primary",
        icon=":material/compare_arrows:",
        key="home_multi",
        width="stretch",
    ):
        st.switch_page("app_pages/multi_stock.py")
with single_entry:
    st.markdown("#### 已经有目标股票？")
    st.caption("直接查看单股历史行情、模型判断和样本外可靠性证据。")
    if st.button(
        "直接进入单股研判",
        icon=":material/show_chart:",
        key="home_single",
        width="stretch",
    ):
        st.switch_page("app_pages/single_stock.py")

with st.container(horizontal=True, gap="medium"):
    st.page_link(
        "app_pages/about.py",
        label="项目介绍",
        icon=":material/info:",
    )
    st.page_link(
        "app_pages/report.py",
        label="查看量化分析简报",
        icon=":material/article:",
    )

st.divider()
st.subheader("三步完成一条分析链路", icon=":material/route:")
st.markdown(
    """
    <div class="stage-grid">
      <section>
        <div class="stage-number">01 · EXPLORE</div>
        <h3>多股历史研究</h3>
        <p>比较收益、风险、趋势、相关性与 K-Means 股票画像，先建立横向认识。</p>
      </section>
      <section>
        <div class="stage-number">02 · EXAMINE</div>
        <h3>单股模型研判</h3>
        <p>查看 Decision Tree 方向判断和 Linear Regression 收益率判断，并与 Baseline 对照。</p>
      </section>
      <section>
        <div class="stage-number">03 · SUMMARIZE</div>
        <h3>自动量化分析简报</h3>
        <p>汇总历史画像、模型结果和可靠性证据，形成可下载的结构化总结。</p>
      </section>
    </div>
    """,
    unsafe_allow_html=True,
)

st.subheader("这个平台能帮你做什么？")
past, judgment, summary = st.columns(3, gap="large")
with past:
    st.markdown("### 了解过去")
    st.write(
        "通过多股行情、收益、风险、趋势和相关性，快速认识不同股票在所选历史区间的表现差异。"
    )
with judgment:
    st.markdown("### 辅助判断")
    st.write(
        "通过股票画像和单股监督学习模型提供横向定位与短期参考，同时展示模型过去是否可靠。"
    )
with summary:
    st.markdown("### 整理结论")
    st.write(
        "把分散的指标、图表和模型证据自动组织成量化分析简报，降低手工整理成本。"
    )

st.info(
    "系统负责提供分析证据、横向比较、模型参考与结构化总结；"
    "最终判断由用户自主完成。本平台不是荐股系统。",
    icon=":material/balance:",
)

statuses = get_analysis_status()
ready_count = sum(status == "ready" for status in statuses.values())
with st.expander("技术与方法", icon=":material/code:"):
    st.caption(
        "技术信息放在分析流程之后，便于答辩核验，也避免普通用户在首屏面对大量术语。"
    )
    with st.container(horizontal=True, gap="medium"):
        st.metric("已接入模块", f"{ready_count}/4", border=True)
        st.metric("行情基础字段", "8 项", border=True)
        st.metric("课程 P0 算法", "3 类", border=True)
        st.metric("模型验证", "时间切分 + Baseline", border=True)
    st.markdown(
        "数据链路采用 **AkShare 在线历史行情 + 用户主动选择的离线样例**；"
        "分析层使用 Pandas 与 Plotly；模型限定为 Decision Tree、"
        "Linear Regression 和固定 KMeans(k=3)。"
    )
    st.caption(
        "实时行情仅作快照展示，不参与 EDA 或模型训练；所有模型结论必须与历史样本外指标共同阅读。"
    )
