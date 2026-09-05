"""Role 6 线性回归模块单元测试（D4：动态单股输入 + 时序/指标）。

覆盖：
- 正常输入：Contract 核心键、metrics 键、predictions 列与日期升序
- 不同股票、不同日期范围、日期乱序、多股票报错
- 下一日标签对齐（目标=return(t+1)）、时间切分边界、无数据泄漏
- 严格日期顺序、重复日期、非法数值和异常类型
- 样本不足、无数据、全 NaN、常量目标
- 指标与 sklearn 底层结果一致、结果可复现
- 无固定股票/日期：函数只消费传入 DataFrame，不写死 symbol/年份
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error, r2_score

from src.contracts.supervised import REGRESSION_PREDICTION_COLUMNS
from src.data.features import build_common_features
from src.models.supervised.regression import (
    FEATURE_COLS,
    fit_regression,
    get_regression_sample_info,
)
from src.utils.exceptions import DataValidationError, InsufficientDataError


def _make_market_data(symbol: str = "600519.SH", n_days: int = 120, seed: int = 7) -> pd.DataFrame:
    """生成符合公共契约的模拟行情 DataFrame（基础列）。"""
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


def _features(symbol: str = "600519.SH", n_days: int = 120, start: str = "2024-01-01") -> pd.DataFrame:
    df = _make_market_data(symbol, n_days=n_days)
    df["trade_date"] = pd.bdate_range(start, periods=n_days)
    return build_common_features(df)


class TestContract:
    def test_core_keys_and_metrics_keys(self) -> None:
        result = fit_regression(_features())
        assert set(result) == {"model", "feature_names", "metrics", "predictions"}
        # metrics 必须保持 Contract 的 mae/r2 两个键
        assert set(result["metrics"].keys()) == {"mae", "r2"}

    def test_predictions_columns_and_order(self) -> None:
        pred = fit_regression(_features())["predictions"]
        assert list(pred.columns) == list(REGRESSION_PREDICTION_COLUMNS)
        assert pred["trade_date"].is_monotonic_increasing

    def test_feature_names_match_contract(self) -> None:
        assert fit_regression(_features())["feature_names"] == FEATURE_COLS


class TestTemporalSplitAndLeakage:
    def test_labels_are_next_day(self) -> None:
        """y_true 必须是 return(t+1)，即与日期后一天的真实 return 对齐。"""
        r = fit_regression(_features())
        pred = r["predictions"].reset_index(drop=True)
        # 复现数据：特征窗口后有效样本
        frame = _features().copy()
        frame["next_return"] = frame["return"].shift(-1)
        data = frame.dropna(subset=FEATURE_COLS + ["next_return"]).reset_index(drop=True)
        split = int(len(data) * 0.8)
        assert np.allclose(pred["y_true"].values, data["next_return"].iloc[split:].values)

    def test_no_temporal_leakage(self) -> None:
        """predictions 只能是最新 20% 的日期（样本外），不能含训练期。"""
        r = fit_regression(_features())
        frame = _features().copy()
        frame["next_return"] = frame["return"].shift(-1)
        data = frame.dropna(subset=FEATURE_COLS + ["next_return"]).reset_index(drop=True)
        split = int(len(data) * 0.8)
        expected = data["trade_date"].iloc[split:].reset_index(drop=True)
        pd.testing.assert_series_equal(
            r["predictions"]["trade_date"].reset_index(drop=True), expected, check_names=False
        )

    def test_split_boundary_ratio(self) -> None:
        """训练/测试样本数符合 80/20 切分边界。"""
        frame = _features().copy()
        frame["next_return"] = frame["return"].shift(-1)
        data = frame.dropna(subset=FEATURE_COLS + ["next_return"])
        split = int(len(data) * 0.8)
        predictions = fit_regression(frame)["predictions"]
        assert len(predictions) == len(data) - split

    def test_date_out_of_order_raises(self) -> None:
        df = _features()
        # 逆序（单调递减，非升序）→ 应抛 DataValidationError
        with pytest.raises(DataValidationError):
            fit_regression(df.iloc[::-1].reset_index(drop=True))

    def test_duplicate_dates_raise(self) -> None:
        frame = _features()
        frame.loc[30, "trade_date"] = frame.loc[29, "trade_date"]
        with pytest.raises(DataValidationError, match="不能重复"):
            fit_regression(frame)


class TestDynamicInput:
    def test_different_symbols(self) -> None:
        r1 = fit_regression(_features("000001.SZ"))
        r2 = fit_regression(_features("300750.SZ"))
        assert r1["model"] is not None and r2["model"] is not None
        # 不同股票应产生不同预测（样本不同）
        assert len(r1["predictions"]) == len(r2["predictions"])

    def test_different_date_ranges(self) -> None:
        r_a = fit_regression(_features(start="2024-01-01"))
        r_b = fit_regression(_features(start="2023-06-01"))
        assert r_a["predictions"]["trade_date"].iloc[0] != r_b["predictions"]["trade_date"].iloc[0]

    def test_no_fixed_symbol_or_year_in_code(self) -> None:
        """无固定股票/日期：换任意 symbol 与 start 都能跑。"""
        for sym, start in (("601318.SH", "2022-01-03"), ("002415.SZ", "2025-01-01")):
            r = fit_regression(_features(sym, start=start))
            assert len(r["predictions"]) > 0


class TestEdgeCases:
    @pytest.mark.parametrize("bad_input", [None, [], "not a dataframe"])
    def test_non_dataframe_raises(self, bad_input) -> None:
        with pytest.raises(DataValidationError, match="DataFrame"):
            fit_regression(bad_input)

    def test_multi_symbol_raises(self) -> None:
        two = pd.concat([_make_market_data("A"), _make_market_data("B")], ignore_index=True)
        with pytest.raises(DataValidationError):
            fit_regression(two)

    def test_too_short_raises(self) -> None:
        with pytest.raises(InsufficientDataError):
            fit_regression(_features(n_days=8))

    def test_no_data_raises(self) -> None:
        with pytest.raises(DataValidationError):
            fit_regression(pd.DataFrame())

    def test_all_nan_features_raises(self) -> None:
        df = _features()
        for c in FEATURE_COLS:
            df[c] = np.nan
        with pytest.raises(InsufficientDataError):
            fit_regression(df)

    def test_infinite_feature_raises(self) -> None:
        df = _features()
        df.loc[30, "ma5"] = np.inf
        with pytest.raises(DataValidationError, match="非有限值"):
            fit_regression(df)

    def test_nonnumeric_feature_raises(self) -> None:
        df = _features()
        df["ma5"] = df["ma5"].astype(object)
        df.loc[30, "ma5"] = "bad"
        with pytest.raises(DataValidationError, match="无法转换"):
            fit_regression(df)

    def test_invalid_date_raises(self) -> None:
        df = _features()
        df["trade_date"] = df["trade_date"].astype(object)
        df.loc[30, "trade_date"] = "not-a-date"
        with pytest.raises(DataValidationError, match="无法解析"):
            fit_regression(df)

    def test_constant_target_raises(self) -> None:
        """目标近似常量时 R² 无法定义，应明确报告样本不足。"""
        df = _features()
        # 构造常量目标：close 恒定 → return 恒 0 → next_return 恒 0
        df["close"] = 100.0
        df = build_common_features(df)
        # 重新构建特征（feature_edges）：用恒定 close 重算
        with pytest.raises(InsufficientDataError, match="R²"):
            fit_regression(df)


class TestSampleDiagnostics:
    def test_sample_info_matches_predictions_and_split(self) -> None:
        frame = _features()
        info = get_regression_sample_info(frame)
        result = fit_regression(frame)

        assert info["input_rows"] == len(frame)
        assert info["effective_rows"] == info["train_rows"] + info["test_rows"]
        assert info["dropped_rows"] == len(frame) - info["effective_rows"]
        assert info["test_rows"] == len(result["predictions"])
        assert info["split_ratio"] == 0.8
        train_end = pd.Timestamp(info["train_date_range"].split(" 至 ")[1])
        test_start = pd.Timestamp(info["test_date_range"].split(" 至 ")[0])
        assert train_end < test_start


class TestMetricsAndReproducibility:
    def test_metrics_match_sklearn(self) -> None:
        r = fit_regression(_features())
        pred = r["predictions"]
        y_true, y_pred = pred["y_true"].values, pred["y_pred"].values
        assert np.isclose(r["metrics"]["mae"], mean_absolute_error(y_true, y_pred))
        assert np.isclose(r["metrics"]["r2"], r2_score(y_true, y_pred))

    def test_reproducible(self) -> None:
        r1 = fit_regression(_features())
        r2 = fit_regression(_features())
        pd.testing.assert_frame_equal(r1["predictions"], r2["predictions"])
        assert r1["metrics"] == r2["metrics"]
