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

## 真实数据（AkShare）

新增 `scripts/fetch_akshare_sample.py`，用 AkShare `stock_zh_a_hist`（前复权 `qfq`）抓取
**10 支真实 A 股**并写回 `sample_daily.csv`，供无 Token 场景用真实行情演示：

```bash
python scripts/fetch_akshare_sample.py
```

- 10 支：600519.SH 贵州茅台、000001.SZ 平安银行、300750.SZ 宁德时代、601318.SH 中国平安、
  000858.SZ 五粮液、600276.SH 恒瑞医药、000333.SZ 美的集团、002594.SZ 比亚迪、
  601899.SH 紫金矿业、000725.SZ 京东方A（沪深两市、行业分散）。
- 日期范围 2024-01-02 ~ 2024-12-31（约 242 个交易日）。
- 成交量「手」×100 换算为「股」；成交额已是「元」，不做换算。
- 抓取需联网，脚本内置重试。运行成功后 `sample_daily.csv` 即替换为真实样例。

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
