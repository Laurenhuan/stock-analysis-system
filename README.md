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

## 团队 Role

1. Role 1：系统架构、Streamlit、Service Layer、集成测试与工程治理。
2. Role 2：金融数据获取、清洗、公共特征与数据存储。
3. Role 3：EDA、描述统计与金融数据可视化。
4. Role 4：Decision Tree Classification 与 Linear Regression。
5. Role 5：K-Means 与股票风险收益画像。

详细 Ownership 见 `docs/architecture.md`。

## GitHub 协作原则

- 从 `main` 创建短生命周期功能分支，通过 Pull Request 合并。
- 跨 Ownership 修改先与主要责任人协作，并由相关 Role Code Review。
- 提交前运行测试并检查 `git diff`，禁止提交秘密、缓存或大体积原始数据。
- Contract 变更应由受影响 Role 共同评审，不在单一实现中暗自改变接口。

## V1.0 范围

- A 股历史行情数据链路。
- 基础金融指标与 EDA。
- Decision Tree Classification 涨跌方向分类。
- Linear Regression 金融变量与后续收益率分析。
- K-Means 多股票风险收益聚类。
- Streamlit 结果展示。

## 暂不包含

当前阶段不包含真实金融 API、数据清洗、金融特征、EDA、任何模型训练或预测、股票画像、量化策略、投资建议及复杂可视化；也不引入其他 Web 框架、数据库、微服务或深度学习框架。
