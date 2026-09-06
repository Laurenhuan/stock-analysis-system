# 证券数据分析与决策参考平台

一个面向课程实践与量化分析学习的 A 股研究应用。平台把在线行情、历史统计、多股画像、监督学习模型和自动简报串成完整流程，帮助用户理解“历史上发生了什么、不同股票有何差异、模型过去是否提供了额外信息”。系统不荐股，最终判断由用户自行完成。

## 核心流程

    选择股票池与历史区间
            ↓
    多股历史研究：EDA / 收益 / 风险 / 趋势 / 相关性
            ↓
    K-Means 股票画像：识别历史风险收益特征相近的股票
            ↓
    选择一只关注股票
            ↓
    Decision Tree 方向判断 + Linear Regression 收益率判断
            ↓
    历史样本外表现 + 简单 Baseline
            ↓
    结构化量化分析简报

页面遵循“先结论 → 再核心数字 → 再图表证据 → 再详细说明”。模型结果必须和历史测试指标、基线一起阅读；低于或接近基线表示当前特征的预测增量有限，不代表程序故障。

## 功能

### 首页

- 说明产品定位、用户价值与三阶段分析流程。
- 提供“开始多股研究”“直接进入单股研判”“生成分析简报”和“项目介绍”入口。

### 多股历史研究

- 支持选择或输入 2～20 只合法 A 股代码，并共享股票池、日期和数据源。
- 展示区间收益、上涨日占比、波动率、最大回撤、K 线、均线、收益分布和极端交易日。
- 支持 Spearman、Pearson、Kendall 三种相关系数；选择后需重新点击“加载并比较”触发计算。
- 固定使用 KMeans(k=3)，根据当前股票池的原始尺度中心生成相对收益、波动和回撤画像。Cluster 编号本身不代表好坏。

### 单股研究

- 数据概览展示历史行情、来源信息与独立实时快照。
- Decision Tree 检验特征对下一交易日上涨/非上涨方向的历史区分能力。
- Linear Regression 检验特征与下一交易日收益率之间的历史线性关系。
- 最早 80% 样本训练、最新 20% 样本测试，不打乱时间；页面同时展示有效样本、测试区间、模型指标、简单基线与多窗口稳定性。

### 量化分析简报

- 复用已有 EDA、风险、趋势、相关性、聚类和监督学习结果。
- 自动组织为面向用户的结构化结论，支持下载 Markdown。
- 不使用额外模型生成买卖建议。

### 项目介绍

- /about.html 从使用者与开发者两个视角介绍产品流程、技术架构和可核验的协作方式。
- 中央课程仓库：[Gitee：sp1-2026/25151407](https://gitee.com/sp1-2026/25151407)。

## 数据来源与口径

- 在线历史日线通过 AkShare 调用公开上游接口，并支持东方财富 → 腾讯的多源回退；在线结果不写入本地 CSV。
- 实时快照使用独立接口，只用于当前行情展示，不直接进入历史 EDA 或模型训练。
- 离线 Sample Data 由用户主动选择，用于无网络环境和课堂稳定演示；页面会明确标注来源、Provider、抓取时间与是否为样例。
- 公共行情字段为 symbol、trade_date、open、high、low、close、volume、amount，派生特征和模型口径详见 [数据与算法 Contract](docs/contracts/)。

上游站点可能限流、断开连接或延迟更新，因此“在线”不等同于交易级实时。历史结果只描述所选区间，不代表未来表现。

## 技术架构

    Streamlit（app.py / app_pages）
            ↓
    Service Layer（src/services）
            ↓
    Data / Analysis / Models / Visualization
            ↓
    Contract-compliant DataFrame 与结果对象

- 页面只调用 Service Layer，不直接依赖领域模块内部实现。
- src/data 负责抓取、清洗与公共特征。
- src/analysis 和 src/visualization 负责统计结论与 Plotly 图表。
- src/models 只包含课程 P0 范围内的 Decision Tree、Linear Regression 和固定 KMeans(k=3)。
- src/services 负责编排用例、诊断样本、组织页面稳定输出和简报。

详细说明见 [架构文档](docs/architecture.md)、[角色边界](docs/role_boundaries.md) 与 [团队规范](docs/team_conventions.md)。

## 在线访问

- Streamlit Cloud：[https://stock-analysis-system-25151407.streamlit.app/](https://stock-analysis-system-25151407.streamlit.app/)
- 项目介绍：[https://stock-analysis-system-25151407.streamlit.app/about.html](https://stock-analysis-system-25151407.streamlit.app/about.html)

在线行情依赖公开上游接口，遇到网络波动或限流时可改用页面中的离线样例完成稳定演示。

## 本地运行

环境建议：Python 3.11+。

    cd /d D:\Codex工作空间\stock-analysis-system-role-boundaries
    python -m venv .venv
    .venv\Scripts\python.exe -m pip install -r requirements.txt
    .venv\Scripts\python.exe -m streamlit run app.py --server.port 8766

默认访问：

- 首页：http://localhost:8766/
- 项目介绍：http://localhost:8766/about.html

## 局域网答辩

Windows 下可直接运行：

    scripts\run_lan.cmd

脚本监听 0.0.0.0:8766 并列出本机 IPv4。同一 Wi-Fi 下分享 http://<本机IPv4>:8766/；项目介绍为 http://<本机IPv4>:8766/about.html。不要分享 127.0.0.1 或虚拟网卡地址。若无法访问，请检查启动窗口、同网条件、访客网络隔离和 Windows 防火墙专用网络权限。

`.streamlit/config.toml` 只保存界面主题，不固定服务器端口。局域网脚本通过命令行使用 8766；Streamlit Community Cloud 可继续使用平台要求的健康检查端口。

## 测试

    .venv\Scripts\python.exe -m pytest -q
    git diff --check

测试覆盖数据契约、在线 Provider 回退、字段与单位、EDA 边界、时间序列泄漏、模型退化输入、聚类可复现性、Service 工作流和 Streamlit AppTest。阶段性输出保存在 docs/test_logs/。

## 项目结构

    app.py                  Streamlit 导航入口
    app_pages/              首页、多股、单股、简报、About 页面
    src/data/               行情抓取、清洗、公共特征
    src/analysis/           EDA 与问题驱动结论
    src/models/             分类、回归、聚类
    src/services/           应用用例与跨模块编排
    src/visualization/      可复用 Plotly 图表
    tests/unit/             各领域模块单元测试
    tests/integration/      Service、分层边界与页面测试
    daily/<学号>/           成员本人日报
    prompts/<学号>/         成员本人 AI 提示词记录
    docs/<学号>/            成员本人课程报告

## 团队分工

| Role | Git 用户 | 主要职责 |
| --- | --- | --- |
| Role 1 | Laurenhuan | 架构、Streamlit、Service、PR 与最终集成 |
| Role 2 | juanjuan-he | 行情获取、清洗、标准数据与公共指标 |
| Role 3 | aaaSpringaaa | EDA、描述统计与 Plotly 可视化 |
| Role 4 | zhou_learn_hahaha | Decision Tree 分类 |
| Role 5 | a3556115538qqcom | 股票画像与 K-Means |
| Role 6 | lu2160 | Linear Regression 与算法 Review |

根目录 README.md 和 todos.md 为团队单一来源。每位成员只维护本人 daily/<学号>/Dn.md、prompts/<学号>/Dn/<工具名>.txt 和 docs/<学号>/，不得猜测或代写他人课程记录。

## 使用边界

本项目用于软件开发实践、金融数据处理和机器学习方法展示。历史表现、聚类画像、相关性和模型预测都存在样本、特征、市场环境与数据源限制，不构成投资建议，也不承诺任何收益或预测准确率。
