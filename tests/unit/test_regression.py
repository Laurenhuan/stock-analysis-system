"""Role 6 线性回归模块单元测试。

覆盖：
- 正常输入：Contract 键、指标键、predictions 列与日期升序
- 指标有限性、可复现性、单股票约束、样本不足、多股票报错
- 无数据泄漏：predictions 只包含样本外(测试集)
- 端到端：公共特征 DataFrame → RegressionResult
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.contracts.supervised import REGRESSION_PREDICTION_COLUMNS
from src.data.features import build_common_features
from src.models.supervised.regression import FEATURE_COLS, fit_regression
from src.utils.exceptions import DataValidationError, InsufficientDataError


def _make_market_data(symbol: str = "600519.SH", n_days: int = 120, seed: int = 7) -> pd.DataFrame:
    """生成符合公共契约的模拟行情 DataFrame（basis 列）。"""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.02, size=n_days))
    return pd.DataFrame(
        {
            "symbol": symbol,
            "trade_date": dates,
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.randint(100_000, 1_000_000, size=n_days).astype(float),
            "amount": rng.uniform(1e7, 5e7, size=n_days),
        }
    )


def _features(symbol: str = "600519.SH", n_days: int = 120) -> pd.DataFrame:
    return build_common_features(_make_market_data(symbol, n_days=n_days))


class TestFitRegression:
    """fit_regression 函数测试。"""

    def test_returns_contract_keys(self) -> None:
        result = fit_regression(_features())
        assert set(result.keys()) == {"model", "feature_names", "metrics", "predictions"}
        assert set(result["metrics"].keys()) == {"mae", "r2"}

    def test_predictions_columns_and_order(self) -> None:
        pred = fit_regression(_features())["predictions"]
        assert list(pred.columns) == list(REGRESSION_PREDICTION_COLUMNS)
        assert pred["trade_date"].is_monotonic_increasing

    def test_metrics_are_finite(self) -> None:
        m = fit_regression(_features())["metrics"]
        assert np.isfinite(m["mae"])
        assert np.isfinite(m["r2"])

    def test_reproducible(self) -> None:
        r1 = fit_regression(_features())
        r2 = fit_regression(_features())
        pd.testing.assert_frame_equal(r1["predictions"], r2["predictions"])

    def test_single_symbol_only(self) -> None:
        two = pd.concat(
            [_make_market_data("A"), _make_market_data("B")], ignore_index=True
        )
        with pytest.raises(DataValidationError):
            fit_regression(two)

    def test_too_short_raises(self) -> None:
        with pytest.raises(InsufficientDataError):
            fit_regression(_features(n_days=8))

    def test_no_temporal_leakage(self) -> None:
        """predictions 只能是"最新 20% 测试集"的日期，不得包含训练期（无泄漏）。"""
        result = fit_regression(_features())
        pred = result["predictions"]

        # 复现模型内部的有效样本与切分，用于核对预测日期是否为样本外尾部。
        frame = _features().copy()
        frame["next_return"] = frame["return"].shift(-1)
        data = frame.dropna(subset=FEATURE_COLS + ["next_return"]).reset_index(drop=True)
        split_index = int(len(data) * 0.8)
        expected_dates = data["trade_date"].iloc[split_index:].reset_index(drop=True)

        assert len(pred) == len(expected_dates)
        pd.testing.assert_series_equal(
            pred["trade_date"].reset_index(drop=True),
            expected_dates,
            check_names=False,
        )


class TestEndToEnd:
    def test_full_pipeline(self) -> None:
        result = fit_regression(_features("000001.SZ"))
        assert result["model"] is not None
        assert len(result["feature_names"]) == len(FEATURE_COLS)
        assert len(result["predictions"]) > 0

    def test_input_uses_role2_public_fields(self) -> None:
        """Regression 只依赖 Role 2 标准公共字段构建的特征。"""
        df = _features()
        base_ok = all(
            c in df.columns
            for c in ["symbol", "trade_date", "close", "return", "ma5", "ma20", "volatility_20d", "volume_change", "drawdown"]
        )
        assert base_ok
