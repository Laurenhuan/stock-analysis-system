"""Fetch 10 real A-shares via AkShare and write them as the Role 2 sample data.

Produces ``data/sample/sample_daily.csv`` in the standard 8-column Market Data
Contract format:

    symbol, trade_date, open, high, low, close, volume, amount

Data is **real** daily history (前复权 qfq)。样例抓取走 AkShare 新浪源
``stock_zh_a_daily``（Provider 代码 ``source="akshare"`` 走 eastmoney
``stock_zh_a_hist``，但 eastmoney 历史端点在本机被网络劫持不可达，故改用同属
AkShare 的新浪源；两源输出同为真实前复权日线，字段与单位一致）。Roles 3-5 可据此
做 EDA / 聚类 / 分类 / 回归联调，无需 Tushare Token。联网时运行一次，
``data/sample/sample_daily.csv`` 即替换为真实样例，供离线回退读取。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# 仓库尚未打包为可安装包，直接运行脚本时把项目根目录加入 sys.path。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.contracts.market_data import BASE_MARKET_COLUMNS
from src.data.clean import clean_market_data
from src.utils.exceptions import NoDataError

# 10 支真实 A 股：沪深两市、行业有差异、2024 全年连续交易的大盘蓝筹。
# 前 5 支同时是数据层单元测试引用的样例，不能随意替换。
SYMBOLS = [
    "600519.SH",  # 贵州茅台 — 白酒
    "000001.SZ",  # 平安银行 — 银行
    "300750.SZ",  # 宁德时代 — 动力电池
    "601318.SH",  # 中国平安 — 保险
    "000858.SZ",  # 五粮液 — 白酒
    "600276.SH",  # 恒瑞医药 — 医药
    "000333.SZ",  # 美的集团 — 家电
    "002594.SZ",  # 比亚迪 — 新能源车
    "601899.SH",  # 紫金矿业 — 有色金属
    "000725.SZ",  # 京东方A — 电子/面板
]

START = "20240102"
END = "20241231"
OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "sample" / "sample_daily.csv"


def _sina_symbol(symbol: str) -> str:
    """契约后缀形式 → 新浪接口的 ``sh/sz`` 前缀形式。"""
    code, suffix = symbol.split(".")
    return ("sh" if suffix == "SH" else "sz") + code


def _fetch_sina(symbol: str) -> pd.DataFrame:
    """抓取单只股票：AkShare 新浪源 ``stock_zh_a_daily``（前复权 qfq）。

    新浪源 ``volume`` 已是「股」、``amount`` 已是「元」，直接映射为标准 8 字段，
    无需像 eastmoney 那样再做 手→股 换算。
    """
    import akshare as ak

    raw = ak.stock_zh_a_daily(
        symbol=_sina_symbol(symbol), start_date=START, end_date=END, adjust="qfq"
    )
    if raw is None or raw.empty:
        raise NoDataError(f"新浪源未返回 {symbol} 在指定区间内的数据")
    df = raw.rename(columns={"date": "trade_date"})
    df["symbol"] = symbol
    return df[list(BASE_MARKET_COLUMNS)].copy()


def fetch_one(symbol: str, retries: int = 3, delay: float = 2.0) -> pd.DataFrame:
    """抓取单只股票，带重试（新浪公开端点偶发抖动）。"""
    for attempt in range(1, retries + 1):
        try:
            return _fetch_sina(symbol)
        except Exception as exc:  # noqa: BLE001 网络异常也重试
            if attempt == retries:
                raise
            print(f"  [重试 {attempt}/{retries}] {symbol} 失败：{exc}")
            time.sleep(delay * attempt)
    raise AssertionError("unreachable")  # pragma: no cover


def main() -> None:
    frames = []
    for symbol in SYMBOLS:
        raw = fetch_one(symbol)
        cleaned = clean_market_data(raw)
        print(f"  {symbol}: {len(cleaned)} 行")
        frames.append(cleaned)

    out = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["symbol", "trade_date"])
        .reset_index(drop=True)
    )

    # 关键校验：10 支、联合键唯一。
    assert out["symbol"].nunique() == len(SYMBOLS), "应当正好 10 支股票"
    assert not out[["symbol", "trade_date"]].duplicated().any(), "symbol+trade_date 必须唯一"

    n_nonpositive_vol = int((out["volume"] <= 0).sum())
    n_nonpositive_amt = int((out["amount"] <= 0).sum())
    bad_ohlc = int(
        (
            (out["low"] > out[["open", "close"]].min(axis=1))
            | (out["high"] < out[["open", "close"]].max(axis=1))
        ).sum()
    )
    if n_nonpositive_vol or n_nonpositive_amt or bad_ohlc:
        print(
            f"  [提示] volume<=0: {n_nonpositive_vol}, amount<=0: {n_nonpositive_amt}, "
            f"OHLC 异常: {bad_ohlc}"
        )

    # Sample CSV 里 trade_date 存成 "YYYYMMDD" 字符串，与 fetch 的字符串区间过滤一致。
    out["trade_date"] = out["trade_date"].dt.strftime("%Y%m%d")
    # 成交量是离散「股」，取整；成交额是「元」，保留小数。
    out["volume"] = out["volume"].round().astype("int64")
    out.to_csv(OUT_PATH, index=False)

    print(f"\n已生成 {OUT_PATH}")
    print(f"总行数：{len(out)}，股票数：{out['symbol'].nunique()}")
    print(f"日期范围：{out['trade_date'].min()} ~ {out['trade_date'].max()}")
    print("各股票交易日数：")
    per = out.groupby("symbol")["trade_date"].count()
    for sym, cnt in per.items():
        print(f"  {sym}: {cnt}")


if __name__ == "__main__":
    main()
