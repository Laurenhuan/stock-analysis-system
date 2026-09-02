# 证券金融数据分析与可视化系统

本项目面向 A 股历史行情数据，计划通过 Python 完成数据获取与整理、基础金融指标、探索性数据分析、机器学习和 Streamlit 可视化展示。

> 当前 Repository 处于 Architecture Bootstrap 阶段，部分模块仅包含 Skeleton，不代表业务功能已经完成。

## 当前开发状态

已完成仓库结构、团队责任边界、Review Candidate v0.2 Contracts 和最小 Streamlit 应用。真实行情数据、分析结果及算法均未实现。

## 技术栈

- Web：Streamlit
- 数据处理：pandas、NumPy
- 机器学习：scikit-learn
- 可视化：Matplotlib、Plotly
- 配置：python-dotenv
- 测试：pytest

这些依赖是 V1.0 已明确的基础栈；Bootstrap 页面目前只直接使用 Streamlit，其他依赖将在对应 Role 接入后使用。

## 项目目录

```text
app.py                 Streamlit 首页
pages/                 Streamlit 分析页面
src/data/              数据获取、清洗与公共特征
src/analysis/          EDA
src/models/            监督与无监督学习
src/visualization/     可复用图表
src/services/          UI 与业务模块之间的服务层
src/utils/             公共异常和工具
tests/                 单元及集成测试
data/                  示例、原始和处理后数据目录
docs/                  架构、进度、AI 使用及 Contracts
```

## 安装环境

建议使用 Python 3.11 或 3.12。

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

真实 Token 只能保存在本地 `.env` 中。可复制 `.env.example` 后配置，禁止提交 `.env`。

## 运行 Streamlit

```bash
streamlit run app.py
```

## Day 1 Integration Prototype

“数据概览”页面当前使用 `data/sample/day1_demo_market_data.csv` 中的合成
Sample Data，验证以下最小链路：

```text
Streamlit 输入
→ Service
→ Sample DataFrame
→ 股票与日期筛选
→ 数据表
→ 基础收盘价图
```

该 Sample Data 和基础价格图均为 **TEMPORARY / DAY-1 PROTOTYPE**，不是真实
行情或正式分析模块。后续分别由 Role 2 的数据模块和 Role 3 的可视化模块替换。

## 六人团队 Role

1. Role 1：系统架构与应用集成，负责 Streamlit、Service Layer、接口协调、中央仓库、PR 和最终集成。
2. Role 2：金融数据工程，负责 A 股数据获取、清洗、标准 DataFrame 和公共金融指标。
3. Role 3：金融数据分析与可视化，负责 EDA、描述统计、Plotly 图表和多股票比较。
4. Role 4：监督学习分类，负责 Decision Tree、下一交易日上涨/非上涨、Accuracy 和 Confusion Matrix。
5. Role 5：无监督学习与股票画像，负责 Stock Profile、StandardScaler、K-Means、Cluster 结果和横向股票定位。
6. Role 6：监督学习回归与算法工程，负责 Linear Regression、下一交易日收益率、MAE、R²、Actual-vs-Predicted 数据及算法 Review。

Agent 开始工作前必须阅读根目录 `AGENTS.md`。详细 Ownership、禁止修改范围、分支和交付规则见 `docs/role_boundaries.md`；架构说明见 `docs/architecture.md`。

## 分工

- Role 1（系统架构与集成）：`gitee用户名待填` —— Streamlit、Service Layer、集成测试与工程治理
- Role 2（金融数据工程师）：葛玉娟 —— 数据获取、清洗、公共特征、数据存储与样例数据
- Role 3（数据分析与可视化）：`gitee用户名待填` —— EDA、描述统计与可视化
- Role 4（监督学习）：`gitee用户名待填` —— Decision Tree 分类与 Linear Regression
- Role 5（无监督学习）：`gitee用户名待填` —— K-Means 聚类与风险收益画像

## GitHub 协作原则

- 从 `main` 创建短生命周期功能分支，通过 Pull Request 合并。
- 跨 Ownership 修改先与主要责任人协作，并由相关 Role Code Review。
- 提交前运行测试并检查 `git diff`，禁止提交秘密、缓存或大体积原始数据。
- Contract 变更应由受影响 Role 共同评审，不在单一实现中暗自改变接口。

详细流程见 `docs/github_workflow.md`，路径责任见 `docs/code_ownership.md`。所有成员从同一个受保护的 `main` 开始，通过 Branch → Commit → Pull Request → Review → Merge 协作。

## V1.0 范围

- A 股历史行情数据链路。
- 基础金融指标与 EDA。
- Decision Tree Classification 涨跌方向分类。
- Linear Regression 金融变量与后续收益率分析。
- K-Means 多股票风险收益聚类。
- Streamlit 结果展示。

## 暂不包含

当前阶段不包含真实金融 API、数据清洗、金融特征、EDA、任何模型训练或预测、股票画像、量化策略、投资建议及复杂可视化；也不引入其他 Web 框架、数据库、微服务或深度学习框架。
