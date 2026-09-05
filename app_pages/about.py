"""Audience-focused project introduction available at /about.html."""

import streamlit as st


GITEE_REPOSITORY = "https://gitee.com/sp1-2026/25151407"

st.markdown(
    """
    <style>
    .about-lead {
        color: #CBD5E1;
        font-size: 1.12rem;
        line-height: 1.8;
        max-width: 900px;
        margin-bottom: 1.5rem;
    }
    .about-flow {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.8rem;
        margin: 1rem 0 2rem;
    }
    .about-flow div {
        background: rgba(30, 41, 59, 0.65);
        border-top: 2px solid #60A5FA;
        border-radius: 0.45rem;
        padding: 1rem;
        min-height: 7.8rem;
    }
    .about-flow strong {
        display: block;
        margin-bottom: 0.45rem;
    }
    .about-flow span {
        color: #94A3B8;
        font-size: 0.9rem;
        line-height: 1.5;
    }
    .tech-list {
        color: #CBD5E1;
        line-height: 1.9;
    }
    @media (max-width: 900px) {
        .about-flow {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }
    @media (max-width: 560px) {
        .about-flow {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.caption("ABOUT · 软件开发实践 1")
st.markdown(
    """
    <p class="about-lead">
      这是一个面向 A 股历史研究的证券数据分析与决策参考平台。
      它把数据获取、探索性分析、股票画像、监督学习检验和自动简报连接起来，
      让使用者既能看到结论，也能回到指标、图表和样本外表现核验依据。
    </p>
    """,
    unsafe_allow_html=True,
)

st.subheader("面向使用者：从问题出发完成分析")
st.markdown(
    """
    <div class="about-flow">
      <div><strong>① 多股历史研究</strong><span>选择股票池与历史区间，比较收益、风险、趋势和相关性。</span></div>
      <div><strong>② 股票横向画像</strong><span>使用固定 K-Means(k=3) 回答“哪些股票的历史风险收益特征相似”。</span></div>
      <div><strong>③ 单股模型研判</strong><span>Decision Tree 判断短期方向，Linear Regression 估计短期收益率，并同步展示 Baseline。</span></div>
      <div><strong>④ 自动分析简报</strong><span>汇总历史事实、关注股票、模型结果、可靠性证据和方法限制。</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

understand, compare, verify = st.columns(3, gap="large")
with understand:
    st.markdown("### 快速了解")
    st.write("把原始日线转化为收益、波动、回撤、均线和相关性等可阅读证据。")
with compare:
    st.markdown("### 横向比较")
    st.write("在同一股票池和区间中识别表现差异、风险差异与历史画像。")
with verify:
    st.markdown("### 核验模型")
    st.write("预测结果与历史样本外指标、简单 Baseline 同时出现，避免只展示一个预测数字。")

st.info(
    "平台提供历史分析、横向比较、模型参考和结构化总结，"
    "不直接替用户作出买卖决定，所有输出均不构成投资建议。",
    icon=":material/balance:",
)

st.divider()
st.subheader("面向开发者与老师：项目如何工作")
st.markdown(
    """
    <p class="tech-list">
      <strong>技术栈：</strong>Python · Streamlit · Pandas · Plotly ·
      scikit-learn · AkShare<br>
      <strong>分析能力：</strong>数据获取与清洗 · EDA · Decision Tree ·
      Linear Regression · K-Means<br>
      <strong>工程分层：</strong>Domain Modules · Service Layer ·
      Visualization · Streamlit · Report Service
    </p>
    """,
    unsafe_allow_html=True,
)

st.code(
    """AkShare / Sample Data
        ↓
Data：获取、清洗、公共金融特征
        ↓
Analysis / Models：EDA、Decision Tree、Linear Regression、K-Means
        ↓
Services：跨模块编排、输入校验、展示数据
        ↓
Visualization / Streamlit
        ↓
Quantitative Analysis Brief""",
    language="text",
)
st.caption(
    "页面只调用 Service Layer；Domain Modules 不依赖 Streamlit。"
    "算法、字段和时间切分遵循仓库内已冻结的 P0 Contract。"
)

st.subheader("团队协作与贡献概览")
st.write(
    "项目采用一个 Gitee 主仓库和独立 Feature Branch。"
    "下面只列出本地 Git 历史、远程分支和仓库 Ownership 文档能够交叉确认的记录。"
)
st.markdown(
    """
| Role | 已确认账号 / Git 作者 | 主要模块 | 代表性提交 | Feature Branch |
|---|---|---|---|---|
| Role 1 | Laurenhuan | Streamlit、Service、集成 | dc972af · D4 交互工作区 | feat/ui-workspaces |
| Role 2 | juanjuan-he | 数据获取、清洗、公共指标 | f4aa4ae · 动态 A 股搜索 | feat/dynamic-symbols |
| Role 3 | aaaSpringaaa / aaaspringaaa | EDA、Plotly 可视化 | ffbad2d · 动态多股 EDA | feat/eda-dynamic-input |
| Role 4 | zhou_learn_hahaha / zhou-1000 | Decision Tree 分类 | 146a678 · 动态输入与 Contract 修复 | feat/classification-dynamic-range |
| Role 5 | a3556115538qqcom / 冰灼 | Stock Profile、K-Means | 3aced8d · 动态簇画像 | feat/clustering-interpretation |
| Role 6 | lu2160 | Linear Regression、算法 Review | bacfcc8 · 回归实现与测试 | feat/linear-regression |
"""
)

st.markdown("#### 可核验的协作流程")
st.write(
    "统一 Contract → 成员在 Feature Branch 独立开发 → 提交与测试 → "
    "Review / Merge → Role 1 Service 集成 → 跨模块联调 → Main 稳定版本"
)
st.caption(
    "此处不展示贡献比例、活跃度排行榜或推测性统计；"
    "提交内容和合并历史可在中央仓库继续核验。"
)

st.link_button(
    "在 Gitee 查看项目源码",
    GITEE_REPOSITORY,
    icon=":material/code:",
    type="primary",
)
st.caption(
    "中央仓库地址由当前项目 git remote -v 中的 origin 确认。"
    "可进一步查看 README、项目结构、Git 历史和开发记录。"
)
