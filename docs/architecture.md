# Architecture

## Status

Contract Review Candidate v0.2。仓库已经完成 Architecture Bootstrap，共享 Contract 已具备供 Role 2–6 并行实现和人工 Review 的基础。当前六人 Ownership 以 `docs/role_boundaries.md` 为准。

## Layering

```text
Streamlit (app.py / pages)
            ↓
Service Layer (src/services)
            ↓
Domain Modules (Data / Analysis / Models / Visualization)
```

### Streamlit

页面负责收集输入、调用 Service，并展示成功结果、空状态或用户可理解的错误。页面不得直接依赖各 Domain Module 的内部实现。

### Service Layer

Service Layer 是 UI 的稳定应用入口，负责：

- 检查应用层参数；
- 调用对应 Domain Module；
- 协调跨数据、分析、模型和可视化模块的用例；
- 将领域异常转换为 UI 可理解的行为；
- 按共享 Contract 向 UI 返回稳定结果。

Service Layer 不负责：

- 重新实现行情清洗或公共特征；
- 自己训练分类或回归模型；
- 自己实现聚类算法；
- 重复 Role 2–6 已拥有的业务逻辑。

### Domain Modules

- `src/data/`：行情获取、清洗、公共金融特征和数据存储边界。
- `src/analysis/`：EDA 与描述统计。
- `src/models/supervised/`：分类与回归模型。
- `src/models/unsupervised/`：聚类与股票画像算法。
- `src/visualization/`：可复用金融图表，不承载模型训练逻辑。

依赖方向必须从 UI 指向 Service，再指向 Domain Modules。Domain Modules 不得反向导入 Streamlit 页面，也不得通过彼此内部实现形成循环依赖。

## Shared contracts

`src/contracts/` 保存轻量、可导入的共享 Schema 和字段常量。P0 使用 `TypedDict` 表示普通字典结果，并使用 `pandas.DataFrame` 表示表格数据；不引入额外抽象框架。

Contract 文档是语义来源，Python 类型用于减少字段拼写和接口形状分歧，不能替代运行时数据校验。Role 2–6 可以在各自模块内部使用更具体的私有类型，但跨模块输出必须满足共享 Contract。

## Exception boundary

统一领域异常位于 `src/utils/exceptions.py`：

- `InvalidSymbolError`：证券代码格式或支持范围无效；
- `NoDataError`：合法请求没有返回数据；
- `DataValidationError`：数据违反字段、唯一性、单位或质量规则；
- `InsufficientDataError`：有效样本不足以执行指定分析。

Domain Modules 抛出最具体的领域异常。Service Layer 可以记录上下文并转换为应用状态；不得静默吞掉验证错误，也不应把底层堆栈直接展示给最终用户。

## Ownership

| Role | Primary ownership |
| --- | --- |
| Role 1 | `app.py`, `pages/`, `src/services/`, `src/contracts/`, `src/utils/`, `tests/integration/`, `docs/architecture.md`, `docs/contracts/` |
| Role 2 | `src/data/` |
| Role 3 | `src/analysis/`, `src/visualization/` |
| Role 4 | `src/models/supervised/classification.py` 及分类/泄漏单元测试 |
| Role 5 | `src/models/unsupervised/clustering.py` 及画像/聚类/复现单元测试 |
| Role 6 | `src/models/supervised/regression.py` 及回归/泄漏单元测试 |

Ownership 表示主要责任人，不是排他权限。跨模块修改必须协作并经过相关责任人的 Code Review。

Role 1 统一协调 `src/models/supervised/__init__.py`、`src/models/unsupervised/__init__.py`、`requirements.txt`、共享 Fixture 和仓库配置等共享文件。Role 6 可以进行算法层 Review，但对应模块仍由各自 Owner 修改。

## Contract governance

`docs/contracts/` 当前状态为 Review Candidate v0.2。任何字段语义、单位、复权口径、时间切分、模型私有字段或输出 Schema 的变化，都必须提交 `INTERFACE / DATA POLICY CHANGE REQUEST` 并由受影响 Role Review，不能在单一业务实现中暗自改变。
