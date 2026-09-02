# Sample Data (Role 2)

> **DEMO / SAMPLE DATA — NOT real-time market quotes.**
>
> 本目录数据仅用于无 Token、无网络、积分不足时的功能演示与联调，
> 由脚本生成（随机游走），**不代表任何真实历史行情**，不得用于实盘或投资决策。

## File

| File | Description |
| --- | --- |
| `sample_daily.csv` | 5 只 A 股 × 78 个真实交易日（2024-01-02 ~ 2024-04-30）的标准行情样例 |

## Symbols

- `600519.SH` 贵州茅台
- `000001.SZ` 平安银行
- `300750.SZ` 宁德时代
- `601318.SH` 中国平安
- `000858.SZ` 五粮液

## Format

与公共 Market Data Contract 的标准层一致，单位为最终单位，无需再换算：

```text
symbol, trade_date, open, high, low, close, volume, amount
```

| Field | Unit |
| --- | --- |
| `open/high/low/close` | RMB / share |
| `volume` | shares |
| `amount` | RMB |

- 日期为真实交易日序列（已排除周末与春节/清明休市），每只股票按 `trade_date` 严格升序。
- `symbol + trade_date` 唯一。
- 满足 `low <= open/close <= high`。
- 不包含 Token、账号或任何秘密。
