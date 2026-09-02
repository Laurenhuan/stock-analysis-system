# Sample Data (Role 2)

> **SAMPLE DATA — 离线演示用，非实时行情。**
>
> 本目录的 `sample_daily.csv` 是**真实历史行情**（AkShare 新浪源、前复权 qfq），
> 用于无 Token、无网络时的离线功能演示与联调；数据为 2024 年全年静态快照，前复权
> 价格随基准日期变化会略有差异，**不代表实时行情，不得用于实盘或投资决策**。

## File

| File | Description |
| --- | --- |
| `sample_daily.csv` | 10 支真实 A 股 × 242 个交易日（2024-01-02 ~ 2024-12-31）的前复权日线 |

生成脚本见 `scripts/fetch_akshare_sample.py`（AkShare 新浪源，前复权 qfq，内置重试）。

> 另保留 `scripts/generate_sample_data.py`：可生成 **60 支合成股票**（固定随机种子、
> 可复现），用于需要更大股票池的离线演示场景；合成数据与真实数据严格区分，切勿混用。

## 真实数据来源（AkShare）

`scripts/fetch_akshare_sample.py` 用 AkShare `stock_zh_a_daily`（新浪源，前复权 `qfq`）
抓取 **10 支真实 A 股**并写回 `sample_daily.csv`：

```bash
python scripts/fetch_akshare_sample.py
```

- 10 支：600519.SH 贵州茅台、000001.SZ 平安银行、300750.SZ 宁德时代、601318.SH 中国平安、
  000858.SZ 五粮液、600276.SH 恒瑞医药、000333.SZ 美的集团、002594.SZ 比亚迪、
  601899.SH 紫金矿业、000725.SZ 京东方A（沪深两市、行业分散）。
- 日期范围 2024-01-02 ~ 2024-12-31（242 个交易日）。
- 成交量已是「股」；成交额已是「元」，不做换算。
- 抓取需联网，脚本内置重试。运行成功后 `sample_daily.csv` 即替换为真实样例。

## Composition（板块分布）

| 板块 | 数量 | 代码段 |
| --- | --- | --- |
| 上海主板 | 4 | `600xxx` / `601xxx` |
| 深圳主板 | 5 | `000xxx` / `002xxx` |
| 创业板 | 1 | `300xxx` |

覆盖沪深主板 + 创业板，行业分散，便于 EDA 多股票对比与 K-Means 聚类演示。

## Format

与公共 Market Data Contract 的标准层一致，单位为最终单位，无需再换算：

```text
symbol, trade_date, open, high, low, close, volume, amount
```

| Field | Unit |
| --- | --- |
| `open/high/low/close` | RMB / share（前复权 qfq） |
| `volume` | shares（股） |
| `amount` | RMB（元） |

- 日期为真实交易日序列，每只股票按 `trade_date` 严格升序。
- 10 支股票使用**统一日期区间**（聚类 Contract 要求相同时间区间才可比）。
- `symbol + trade_date` 唯一。
- 满足 `low <= open/close <= high`。
- 不包含 Token、账号或任何秘密。
