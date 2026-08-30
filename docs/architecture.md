# Architecture

## Status

Architecture Bootstrap v0.1。当前结构用于明确边界并支持五个 Role 并行开发，不代表业务功能已完成。

## Layering

```text
Streamlit (app.py / pages)
            ↓
Service Layer (src/services)
            ↓
Data / Analysis / Models / Visualization
```

Streamlit 页面只负责交互与呈现。Service Layer 作为应用入口，负责在后续阶段编排数据、分析和模型模块，并把底层异常转换为 UI 可处理的结果。这样可以避免页面直接耦合各团队的内部实现，使独立测试、接口替换和多人并行开发更可控。Bootstrap 阶段的 Service 只暴露明确的占位接口，不执行其他 Role 的业务逻辑。

## Package responsibilities

- `src/data/`：行情获取、清洗、公共金融特征和数据存储边界。
- `src/analysis/`：EDA 与描述统计。
- `src/models/supervised/`：分类与回归模型。
- `src/models/unsupervised/`：聚类与股票画像算法。
- `src/visualization/`：可复用金融图表，不承载模型逻辑。
- `src/services/`：面向 UI 的用例编排和稳定入口。
- `src/utils/`：确实跨模块共享的轻量工具和异常。

依赖方向应从 UI 指向 Service，再指向领域模块。领域模块不得反向导入 Streamlit 页面；Role 模块之间不应通过彼此的内部实现形成循环依赖。

## Ownership

| Role | Primary ownership |
| --- | --- |
| Role 1 | `app.py`, `pages/`, `src/services/`, `src/utils/`, `tests/integration/`, `docs/architecture.md`, `docs/contracts/` |
| Role 2 | `src/data/` |
| Role 3 | `src/analysis/`, `src/visualization/` |
| Role 4 | `src/models/supervised/` |
| Role 5 | `src/models/unsupervised/` |

Ownership 表示主要责任人，不是排他权限。跨模块修改必须协作并经过相关责任人的 Code Review。

## Contract governance

`docs/contracts/` 中的接口当前均为 Draft v0.1。各 Role 开发前应共同确认字段语义、日期与时区、缺失值策略、错误类型以及训练结果的数据结构。确认前不得把草案当作不可变 API。
