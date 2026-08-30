# Market Data Contract

**Contract Status: Draft v0.1**

**Owner: Role 2（需 Role 1、3、4、5 共同评审）**

## Purpose

定义跨数据、分析、模型和展示层传递行情数据的最小概念边界。实现细节尚未冻结。

## Candidate fields

### A. 原始行情字段

| Field | Proposed data type | Meaning |
| --- | --- | --- |
| `symbol` | string | 证券代码；编码规范待确认 |
| `trade_date` | datetime-like | 交易日期；时区与粒度待确认 |
| `open` | numeric | 开盘价 |
| `high` | numeric | 最高价 |
| `low` | numeric | 最低价 |
| `close` | numeric | 收盘价 |
| `volume` | numeric | 成交量；单位待确认 |
| `amount` | numeric | 成交额；币种与单位待确认 |

### B. 公共派生字段候选

`return`, `cumulative_return`, `ma5`, `ma20`, `volatility`, `volume_change`, `drawdown`

这些字段的公式、窗口、复权口径、空值策略和是否进入公共 Contract，必须由 Role 2、3、4、5 共同确认。

### C. 模型私有字段

`label`, `target`, `cluster`, `prediction` 原则上属于模型或具体用例的输出，不应强行写入公共 Market Data Contract。需要共享时应由相应模型 Contract 单独定义，避免基础行情表被任务语义污染。

## Conceptual interfaces

### `fetch_market_data(...)`

- Input：证券范围、日期范围及数据源配置；具体参数待定。
- Output：包含原始行情字段的表格型数据。
- Data Type：预计为 `pandas.DataFrame`，尚待评审。
- Errors：认证失败、限流、网络错误、无数据、参数无效；错误分类待定。
- Owner：Role 2。

### `clean_market_data(...)`

- Input：原始行情表格。
- Output：字段类型、排序和基础质量满足约定的行情表格。
- Data Type：预计为 `pandas.DataFrame`。
- Errors：字段缺失、重复记录、类型转换失败、非法价格关系。
- Owner：Role 2。

### `build_common_features(...)`

- Input：清洗后的行情表格和经确认的特征配置。
- Output：附加已批准公共派生字段的表格。
- Data Type：预计为 `pandas.DataFrame`。
- Errors：历史窗口不足、字段缺失、参数无效。
- Owner：Role 2；字段语义需 Role 3、4、5 评审。
