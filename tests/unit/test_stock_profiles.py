"""Stock Profile 计算测试 — Role 5。

覆盖：
- build_stock_profiles 正常计算
- 每只股票唯一
- 三个固定特征列
- NaN / 无穷值检查
- 缺少列、空 DataFrame 异常
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.contracts.clustering import PROFILE_FEATURES
from src.models.unsupervised.clustering import (
    FEATURE_COLS,
    build_stock_profiles,
)
from src.utils.exceptions import DataValidationError


# ── 辅助函数 ──────────────────────────────────────────


def _make_market_data(
    symbols: list[str] | None = None,
    n_days: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """生成模拟行情 DataFrame，包含 symbol, trade_date, close, drawdown。"""
    if symbols is None:
        symbols = ["000001", "000002", "000003", "000004", "000005"]

    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)

    rows = []
    for sym in symbols:
        start_price = rng.uniform(10, 100)
        returns = rng.normal(0.001, 0.02, size=n_days)
        prices = start_price * np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(prices)
        drawdowns = (prices - running_max) / running_max

        for i, date in enumerate(dates):
            rows.append(
                {
                    "symbol": sym,
                    "trade_date": date,
                    "close": round(prices[i], 2),
                    "drawdown": round(drawdowns[i], 6),
                }
            )

    return pd.DataFrame(rows)


# ── 正常计算测试 ──────────────────────────────────────


class TestBuildStockProfiles:
    """build_stock_profiles 正常计算验证。"""

    def test_returns_correct_shape(self) -> None:
        """正常输入应返回每只股票一行、含 4 列的 DataFrame。"""
        df = _make_market_data(symbols=["A", "B", "C", "D", "E"])
        profiles = build_stock_profiles(df)

        assert len(profiles) == 5
        assert list(profiles.columns) == ["symbol", *FEATURE_COLS]

    def test_features_match_contract(self) -> None:
        """特征列必须与 PROFILE_FEATURES 一致。"""
        df = _make_market_data()
        profiles = build_stock_profiles(df)

        assert tuple(profiles.columns[1:]) == PROFILE_FEATURES

    def test_mean_return_is_arithmetic_mean(self) -> None:
        """mean_return 是简单日收益率的算术平均。"""
        # 手动构造：3天，close=[100, 110, 100]
        # daily_return = [0.1, -0.0909...]
        # mean = (0.1 + (-0.0909...)) / 2
        df = pd.DataFrame(
            {
                "symbol": ["X", "X", "X"],
                "trade_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
                "close": [100.0, 110.0, 100.0],
                "drawdown": [0.0, 0.0, -0.090909],
            }
        )
        profiles = build_stock_profiles(df)
        expected_mean = (0.1 + (100 / 110 - 1)) / 2
        assert abs(profiles.loc[0, "mean_return"] - expected_mean) < 1e-10

    def test_volatility_uses_ddof1(self) -> None:
        """volatility 使用 ddof=1 的样本标准差。"""
        # 手动构造：4天，确保可验证
        df = pd.DataFrame(
            {
                "symbol": ["X"] * 4,
                "trade_date": pd.to_datetime(
                    ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
                ),
                "close": [100.0, 105.0, 103.0, 108.0],
                "drawdown": [0.0, 0.0, 0.0, 0.0],
            }
        )
        profiles = build_stock_profiles(df)

        # 手动计算 daily_return 和 std(ddof=1)
        returns = pd.Series([105 / 100 - 1, 103 / 105 - 1, 108 / 103 - 1])
        expected_std = returns.std(ddof=1)
        assert abs(profiles.loc[0, "volatility"] - expected_std) < 1e-10

    def test_max_drawdown_is_min_of_drawdown(self) -> None:
        """max_drawdown 是 drawdown 列的最小值。"""
        df = pd.DataFrame(
            {
                "symbol": ["X", "X", "X"],
                "trade_date": pd.to_datetime(
                    ["2024-01-01", "2024-01-02", "2024-01-03"]
                ),
                "close": [100.0, 90.0, 95.0],
                "drawdown": [0.0, -0.1, -0.05],
            }
        )
        profiles = build_stock_profiles(df)
        assert profiles.loc[0, "max_drawdown"] == -0.1


# ── 唯一性测试 ────────────────────────────────────────


class TestProfileUniqueness:
    """每只股票在 Profile Table 中只出现一次。"""

    def test_symbols_are_unique(self) -> None:
        df = _make_market_data()
        profiles = build_stock_profiles(df)
        assert profiles["symbol"].is_unique

    def test_multiple_stocks_each_appear_once(self) -> None:
        df = _make_market_data(symbols=["SH600519", "SZ000001", "SZ300750"])
        profiles = build_stock_profiles(df)
        assert len(profiles) == 3
        assert profiles["symbol"].is_unique


# ── NaN / 无穷值测试 ──────────────────────────────────


class TestProfileDataQuality:
    """输出的特征必须是有限数值。"""

    def test_no_nan_in_output(self) -> None:
        df = _make_market_data()
        profiles = build_stock_profiles(df)

        for col in FEATURE_COLS:
            assert profiles[col].notna().all(), f"{col} 含有 NaN"

    def test_values_are_finite(self) -> None:
        df = _make_market_data()
        profiles = build_stock_profiles(df)

        for col in FEATURE_COLS:
            assert np.isfinite(profiles[col]).all(), f"{col} 含有非有限值"


# ── 异常输入测试 ──────────────────────────────────────


class TestProfileValidation:
    """异常输入应抛出 DataValidationError。"""

    def test_missing_column_raises_error(self) -> None:
        df = pd.DataFrame({"symbol": ["A"], "close": [10.0]})
        with pytest.raises(DataValidationError):
            build_stock_profiles(df)

    def test_empty_dataframe_raises_error(self) -> None:
        df = pd.DataFrame(columns=["symbol", "trade_date", "close", "drawdown"])
        with pytest.raises(DataValidationError):
            build_stock_profiles(df)

    def test_duplicate_symbol_in_input_ok(self) -> None:
        """输入有重复 symbol 是正常的（多天数据），不应报错。"""
        df = _make_market_data(symbols=["A", "A", "B", "B"])
        profiles = build_stock_profiles(df)
        assert profiles["symbol"].is_unique
