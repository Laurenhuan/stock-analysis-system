# Market Data Contract

**Contract Status: Review Candidate v0.2**

**Owner: Role 2（需 Role 1、3、4、5、6 Review）**

## 1. Scope

V1.0 只支持 A 股日频历史行情，不支持分钟、高频或人为生成的非交易日数据。

公共数据以 `pandas.DataFrame` 传递。一行表示一只证券的一个真实交易日，按 `symbol` 分组后必须按 `trade_date` 严格升序。

## 2. Base schema

| Field | Data type | Unit / rule |
| --- | --- | --- |
| `symbol` | string | 保留交易所后缀的统一证券代码 |
| `trade_date` | `datetime64[ns]` | timezone-naive，仅代表交易日 |
| `open` | numeric | RMB / share，前复权（qfq） |
| `high` | numeric | RMB / share，前复权（qfq） |
| `low` | numeric | RMB / share，前复权（qfq） |
| `close` | numeric | RMB / share，前复权（qfq） |
| `volume` | numeric | shares |
| `amount` | numeric | RMB |

公共分析 Contract 中的价格字段必须采用同一前复权口径。Role 2 可以另外保存供应商原始数据，但不得把不同复权口径混入公共 DataFrame。供应商存在技术限制时必须提交 `INTERFACE / DATA POLICY CHANGE REQUEST`，不得擅自切换。

供应商返回手、千股、千元或其他单位时，Role 2 必须在数据进入公共层前换算为上述单位。

## 3. Symbol rule

证券代码采用“六位代码 + 交易所后缀”的规范形式，例如：

```text
600519.SH
000001.SZ
```

核心分析模块不得同时混用 `600519`、`sh600519`、`600519.SH` 等表示法。非法或当前不支持的代码应抛出 `InvalidSymbolError`。

## 4. Date and uniqueness rules

- `trade_date` 必须是 pandas datetime、timezone-naive。
- 不创建周末、法定节假日或停市日期记录，也不对这些日期插值。
- 每只股票内部按 `trade_date` 严格升序。
- `symbol + trade_date` 是联合唯一键。
- 完全相同的联合键重复记录可以去重。
- 同一联合键出现数值冲突时，不得静默选择，必须抛出 `DataValidationError`。

## 5. P0 common derived fields

所有计算必须在单一 `symbol` 内按 `trade_date` 升序执行，不能跨股票串联窗口。

| Field | Definition |
| --- | --- |
| `return` | 简单日收益率：`close_t / close_(t-1) - 1` |
| `cumulative_return` | `(1 + return).cumprod() - 1` |
| `ma5` | 最近 5 个交易日 `close` 的滚动平均 |
| `ma20` | 最近 20 个交易日 `close` 的滚动平均 |
| `volatility_20d` | 最近 20 个交易日 `return` 的滚动样本标准差，`ddof=1`，不年化 |
| `volume_change` | `volume_t / volume_(t-1) - 1`；上一日成交量为 0 时结果为 NaN，不产生无穷值 |
| `drawdown` | `close_t / close.cummax() - 1`，通常满足 `drawdown <= 0` |

滚动窗口必须拥有完整的有效观察值后才产生结果。因此 `ma5` 前 4 个有效观察值、`ma20` 和 `volatility_20d` 前 19 个有效观察值可以为 NaN。将来如需年化波动率，应新增字段，不得改变 `volatility_20d` 的语义。

## 6. Missing and invalid data

关键基础字段缺失、非有限数值、单位不明或违反 Contract 时，不得让数据静默进入模型。允许的处理方式是：

- 删除能够明确判定为无效的记录，并在日志或结果说明中记录；
- 抛出 `DataValidationError`；
- 合法请求没有任何数据时抛出 `NoDataError`。

滚动指标自然产生的前置 NaN 是正常现象。算法模块应在建立自己的模型数据集时处理 Feature NaN，不得为了模型需要而修改或补造原始行情历史。

## 7. Model-private fields

以下字段不得进入公共 Market Data Contract：

```text
label
target
next_return
prediction
cluster
```

这些字段由对应模型 Contract 定义和拥有。

## 8. Conceptual interfaces

### `fetch_market_data(...)`

- Input：规范化证券代码、交易日起止范围、数据源配置。
- Output：包含基础行情字段的表格数据。
- Data Type：`pandas.DataFrame`。
- Errors：`InvalidSymbolError`, `NoDataError`，以及供应商访问错误。
- Owner：Role 2。

### `clean_market_data(...)`

- Input：供应商原始行情表格。
- Output：满足基础 Schema、排序、单位、复权和唯一性规则的 DataFrame。
- Data Type：`pandas.DataFrame`。
- Errors：`DataValidationError`, `NoDataError`。
- Owner：Role 2。

### `build_common_features(...)`

- Input：满足基础 Contract 的行情 DataFrame。
- Output：增加全部 P0 公共派生字段的 DataFrame，保留合法的窗口前置 NaN。
- Data Type：`pandas.DataFrame`。
- Errors：`DataValidationError`, `InsufficientDataError`。
- Owner：Role 2；字段为 Role 3–5 的共享输入。
