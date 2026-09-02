"""Generate the Role 2 sample market data (60 A-share stocks).

Produces ``data/sample/sample_daily.csv`` in the standard 8-column Market Data
Contract format:

    symbol, trade_date, open, high, low, close, volume, amount

The data is **synthetic** (seeded random walk), NOT real quotes. It exists so
Roles 3-5 can develop EDA / clustering against a realistic multi-stock pool
without a Tushare token, points, or network. Reproducible via the fixed seed.

真实样例改用 ``scripts/fetch_akshare_sample.py``（AkShare，10 支真实 A 股，前复权 qfq）；
本脚本保留用于合成 / 离线演示场景。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# A-share holidays excluded from the trading calendar (Spring Festival + Qingming).
_HOLIDAYS = [
    "2024-02-09", "2024-02-12", "2024-02-13", "2024-02-14",
    "2024-02-15", "2024-02-16", "2024-04-04", "2024-04-05",
]

# 60 real A-share symbols with board tags, for sector diversity.
#   sh = 上海主板 (600/601/603)
#   sz = 深圳主板 (000/002)
#   cy = 创业板 (300)
_SYMBOLS: list[tuple[str, str]] = [
    # 上海主板 (25)
    ("600519.SH", "sh"), ("600036.SH", "sh"), ("600900.SH", "sh"),
    ("600030.SH", "sh"), ("600276.SH", "sh"), ("600887.SH", "sh"),
    ("600000.SH", "sh"), ("600028.SH", "sh"), ("600050.SH", "sh"),
    ("600104.SH", "sh"), ("600585.SH", "sh"), ("600690.SH", "sh"),
    ("601318.SH", "sh"), ("601398.SH", "sh"), ("601288.SH", "sh"),
    ("601988.SH", "sh"), ("601166.SH", "sh"), ("601088.SH", "sh"),
    ("601857.SH", "sh"), ("601628.SH", "sh"), ("601668.SH", "sh"),
    ("603288.SH", "sh"), ("603259.SH", "sh"), ("600309.SH", "sh"),
    ("600031.SH", "sh"),
    # 深圳主板 (20)
    ("000001.SZ", "sz"), ("000002.SZ", "sz"), ("000858.SZ", "sz"),
    ("000333.SZ", "sz"), ("000651.SZ", "sz"), ("000725.SZ", "sz"),
    ("000063.SZ", "sz"), ("000568.SZ", "sz"), ("000776.SZ", "sz"),
    ("000100.SZ", "sz"), ("000625.SZ", "sz"), ("000895.SZ", "sz"),
    ("002415.SZ", "sz"), ("002594.SZ", "sz"), ("002714.SZ", "sz"),
    ("002352.SZ", "sz"), ("002475.SZ", "sz"), ("002304.SZ", "sz"),
    ("002142.SZ", "sz"), ("002027.SZ", "sz"),
    # 创业板 (15)
    ("300750.SZ", "cy"), ("300059.SZ", "cy"), ("300015.SZ", "cy"),
    ("300124.SZ", "cy"), ("300760.SZ", "cy"), ("300014.SZ", "cy"),
    ("300122.SZ", "cy"), ("300498.SZ", "cy"), ("300408.SZ", "cy"),
    ("300274.SZ", "cy"), ("300413.SZ", "cy"), ("300033.SZ", "cy"),
    ("300347.SZ", "cy"), ("300433.SZ", "cy"), ("300207.SZ", "cy"),
]

# Per-board volatility so ChiNext is (realistically) more volatile than main board.
_BOARD_VOL = {"sh": 0.012, "sz": 0.016, "cy": 0.024}


def trading_calendar(start: str, end: str) -> pd.DatetimeIndex:
    """Business days minus the fixed A-share holiday list."""
    days = pd.bdate_range(start, end)
    holidays = pd.to_datetime(_HOLIDAYS)
    return days[~days.isin(holidays)]


def generate_stock(symbol: str, board: str, rng: np.random.Generator, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Synthetic OHLCV for one symbol via a seeded geometric random walk."""
    n = len(dates)
    base_price = rng.lognormal(mean=3.0, sigma=0.8)  # ~20 yuan, up to a few hundred
    daily_ret = rng.normal(0.0003, _BOARD_VOL[board], n)
    close = base_price * np.cumprod(1.0 + daily_ret)

    gap = rng.normal(0.0, 0.003, n)
    open_ = np.empty(n)
    open_[0] = close[0] * (1.0 + gap[0])
    open_[1:] = close[:-1] * (1.0 + gap[1:])

    hi = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.005, n)))
    lo = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.005, n)))

    base_volume = rng.lognormal(mean=15.0, sigma=0.5)  # ~ millions of shares
    volume = np.round(base_volume * (1.0 + rng.normal(0.0, 0.15, n)))
    amount = np.round(volume * close, 2)

    return pd.DataFrame({
        "symbol": symbol,
        "trade_date": dates.strftime("%Y%m%d"),
        "open": np.round(open_, 2),
        "high": np.round(hi, 2),
        "low": np.round(lo, 2),
        "close": np.round(close, 2),
        "volume": volume,
        "amount": amount,
    })


def main() -> None:
    rng = np.random.default_rng(42)
    dates = trading_calendar("2024-01-02", "2024-04-30")
    print(f"交易日数量：{len(dates)}")

    frames = [generate_stock(sym, board, rng, dates) for sym, board in _SYMBOLS]
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)

    assert df["symbol"].nunique() == 60, "应当正好 60 支股票"
    assert not df[["symbol", "trade_date"]].duplicated().any(), "symbol+trade_date 必须唯一"
    assert (df["low"] <= df[["open", "close"]].min(axis=1)).all(), "low <= open/close"
    assert (df["high"] >= df[["open", "close"]].max(axis=1)).all(), "high >= open/close"

    out = "data/sample/sample_daily.csv"
    df.to_csv(out, index=False)
    print(f"已生成 {out}：{len(df)} 行，{df['symbol'].nunique()} 支股票")


if __name__ == "__main__":
    main()
