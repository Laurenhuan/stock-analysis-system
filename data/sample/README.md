# Sample Data (Role 2)

> **DEMO / SAMPLE DATA — NOT real-time market quotes.**
>
> 本目录数据是**合成样例**（脚本随机游走生成），仅用于无 Token、无网络、积分不足时的
> 功能演示与联调，**不代表任何真实历史行情**，不得用于实盘或投资决策。
> 与真实 Tushare 数据严格区分，切勿混用。

## File

| File | Description |
| --- | --- |
| `sample_daily.csv` | 60 只 A 股 × 78 个真实交易日（2024-01-02 ~ 2024-04-30）的合成样例 |

生成脚本见 `scripts/generate_sample_data.py`（固定随机种子，可复现）。

## Composition（板块分布）

| Board | 数量 | 代码段 |
| --- | --- | --- |
| 上海主板 | 25 | `600xxx` / `601xxx` / `603xxx` |
| 深圳主板 | 20 | `000xxx` / `002xxx` |
| 创业板 | 15 | `300xxx` |

覆盖沪深主板 + 创业板，不同板块赋予不同波动率特征，便于 EDA 多股票对比与 K-Means 聚类演示。

## Format

与公共 Market Data Contract 的标准层一致，单位为最终单位，无需再换算：

```text
symbol, trade_date, open, high, low, close, volume, amount
```

| Field | Unit |
| --- | --- |
| `open/high/low/close` | RMB / share（合成数据，口径对齐 qfq 前复权） |
| `volume` | shares |
| `amount` | RMB |

- 日期为真实交易日序列（已排除周末与春节/清明休市），每只股票按 `trade_date` 严格升序。
- 60 只股票使用**统一日期区间**（聚类 Contract 要求相同时间区间才可比）。
- `symbol + trade_date` 唯一。
- 满足 `low <= open/close <= high`。
- 不包含 Token、账号或任何秘密。
