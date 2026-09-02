"""Role 5 聚类模块单元测试。

覆盖：
1. build_stock_profiles — 正常输入、缺少列、空 DataFrame、NaN 值、数据不足
2. run_clustering — 正常输入、输出 shape、可复现性、股票不足、symbol 重复、NaN/inf
3. 端到端 — 从原始行情到 ClusteringResult 的完整链路
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.contracts.clustering import (
    CLUSTER_CENTER_COLUMNS,
    CLUSTERING_RESULT_KEYS,
    PROFILE_COLUMNS,
    ClusteringResult,
)
from src.models.unsupervised.clustering import (
    FEATURE_COLS,
    N_CLUSTERS,
    build_stock_profiles,
    run_clustering,
)
from src.utils.exceptions import DataValidationError, InsufficientDataError


# ── 辅助函数：生成测试用的行情数据 ─────────────────────


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


# ── build_stock_profiles 测试 ─────────────────────────


class TestBuildStockProfiles:
    """build_stock_profiles 函数测试。"""

    def test_normal_input_returns_correct_shape(self) -> None:
        """正常输入应返回每只股票一行、含4列的 DataFrame。"""
        df = _make_market_data(symbols=["A", "B", "C", "D", "E"])
        profiles = build_stock_profiles(df)

        assert len(profiles) == 5
        assert list(profiles.columns) == ["symbol", *FEATURE_COLS]

    def test_features_match_contract(self) -> None:
        """特征列必须与 PROFILE_FEATURES 一致。"""
        from src.contracts.clustering import PROFILE_FEATURES

        df = _make_market_data()
        profiles = build_stock_profiles(df)

        assert tuple(profiles.columns[1:]) == PROFILE_FEATURES

    def test_mean_return_is_arithmetic_mean(self) -> None:
        """mean_return 是简单日收益率的算术平均。"""
        df = pd.DataFrame(
            {
                "symbol": ["X", "X", "X"],
                "trade_date": pd.to_datetime(
                    ["2024-01-01", "2024-01-02", "2024-01-03"]
                ),
                "close": [100.0, 110.0, 100.0],
                "drawdown": [0.0, 0.0, -0.090909],
            }
        )
        profiles = build_stock_profiles(df)
        expected_mean = (0.1 + (100 / 110 - 1)) / 2
        assert abs(profiles.loc[0, "mean_return"] - expected_mean) < 1e-10

    def test_volatility_uses_ddof1(self) -> None:
        """volatility 使用 ddof=1 的样本标准差。"""
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

    def test_no_nan_in_output(self) -> None:
        """输出的3个特征列不应有 NaN。"""
        df = _make_market_data()
        profiles = build_stock_profiles(df)

        for col in FEATURE_COLS:
            assert profiles[col].notna().all(), f"{col} 含有 NaN"

    def test_symbols_are_unique(self) -> None:
        """每只股票在 Profile Table 中只出现一次。"""
        df = _make_market_data()
        profiles = build_stock_profiles(df)

        assert profiles["symbol"].is_unique

    def test_values_are_finite(self) -> None:
        """输出的特征值必须是有限数值（非 inf / -inf）。"""
        df = _make_market_data()
        profiles = build_stock_profiles(df)

        for col in FEATURE_COLS:
            assert np.isfinite(profiles[col]).all(), f"{col} 含有非有限值"

    def test_missing_symbol_column_raises_error(self) -> None:
        """缺少 symbol 列应抛出 DataValidationError。"""
        df = pd.DataFrame(
            {"trade_date": [1], "close": [10.0], "drawdown": [0.0]}
        )
        with pytest.raises(DataValidationError):
            build_stock_profiles(df)

    def test_missing_trade_date_column_raises_error(self) -> None:
        """缺少 trade_date 列应抛出 DataValidationError。"""
        df = pd.DataFrame(
            {"symbol": ["A"], "close": [10.0], "drawdown": [0.0]}
        )
        with pytest.raises(DataValidationError):
            build_stock_profiles(df)

    def test_empty_dataframe_raises_error(self) -> None:
        """空 DataFrame 应抛出 DataValidationError。"""
        df = pd.DataFrame(
            columns=["symbol", "trade_date", "close", "drawdown"]
        )
        with pytest.raises(DataValidationError):
            build_stock_profiles(df)

    def test_too_few_days_per_stock_raises_error(self) -> None:
        """每只股票数据不足 3 天应抛出 DataValidationError。"""
        df = pd.DataFrame(
            {
                "symbol": ["A", "A"],
                "trade_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "close": [100.0, 105.0],
                "drawdown": [0.0, 0.0],
            }
        )
        with pytest.raises(DataValidationError, match="数据不足"):
            build_stock_profiles(df)

    def test_null_symbol_in_input_raises_error(self) -> None:
        """输入中存在空 symbol 应抛出 DataValidationError。"""
        df = pd.DataFrame(
            {
                "symbol": ["A", None, "C"],
                "trade_date": pd.to_datetime(
                    ["2024-01-01", "2024-01-02", "2024-01-03"]
                ),
                "close": [100.0, 105.0, 110.0],
                "drawdown": [0.0, 0.0, 0.0],
            }
        )
        with pytest.raises(DataValidationError, match="空的 symbol"):
            build_stock_profiles(df)


# ── run_clustering 测试 ───────────────────────────────


class TestRunClustering:
    """run_clustering 函数测试。"""

    @pytest.fixture()
    def sample_profiles(self) -> pd.DataFrame:
        """5只股票的 Profile Table。"""
        df = _make_market_data()
        return build_stock_profiles(df)

    def test_returns_all_required_keys(self, sample_profiles: pd.DataFrame) -> None:
        """结果必须包含 Contract 约定的4个 key。"""
        result = run_clustering(sample_profiles)

        assert set(result.keys()) == set(CLUSTERING_RESULT_KEYS)

    def test_profiles_has_cluster_column(self, sample_profiles: pd.DataFrame) -> None:
        """profiles 必须包含 cluster 列。"""
        result = run_clustering(sample_profiles)
        profiles = result["profiles"]

        assert "cluster" in profiles.columns
        assert list(profiles.columns) == list(PROFILE_COLUMNS)

    def test_cluster_values_in_valid_range(self, sample_profiles: pd.DataFrame) -> None:
        """cluster 列的值应在 [0, k-1] 范围内。"""
        result = run_clustering(sample_profiles)
        clusters = result["profiles"]["cluster"]

        assert clusters.min() >= 0
        assert clusters.max() < N_CLUSTERS

    def test_cluster_centers_shape(self, sample_profiles: pd.DataFrame) -> None:
        """cluster_centers 应有 k 行，列含 cluster + 3个特征。"""
        result = run_clustering(sample_profiles)
        centers = result["cluster_centers"]

        assert len(centers) == N_CLUSTERS
        assert list(centers.columns) == list(CLUSTER_CENTER_COLUMNS)

    def test_features_and_k_match_contract(self, sample_profiles: pd.DataFrame) -> None:
        """features 和 k 必须与 Contract 一致。"""
        result = run_clustering(sample_profiles)

        assert result["features"] == FEATURE_COLS
        assert result["k"] == N_CLUSTERS

    def test_reproducibility_with_same_seed(self, sample_profiles: pd.DataFrame) -> None:
        """相同 random_state 应产生完全相同的结果。"""
        result1 = run_clustering(sample_profiles, random_state=123)
        result2 = run_clustering(sample_profiles, random_state=123)

        pd.testing.assert_frame_equal(result1["profiles"], result2["profiles"])
        pd.testing.assert_frame_equal(
            result1["cluster_centers"], result2["cluster_centers"]
        )

    def test_different_seed_may_differ(self, sample_profiles: pd.DataFrame) -> None:
        """不同 random_state 可能产生不同分组（不保证不同，但至少不报错）。"""
        result1 = run_clustering(sample_profiles, random_state=0)
        result2 = run_clustering(sample_profiles, random_state=999)

        assert len(result1["profiles"]) == len(result2["profiles"])

    def test_too_few_stocks_raises_error(self) -> None:
        """股票数 < k 应抛出 InsufficientDataError。"""
        tiny = pd.DataFrame(
            {
                "symbol": ["A", "B"],
                "mean_return": [0.01, 0.02],
                "volatility": [0.02, 0.03],
                "max_drawdown": [-0.05, -0.10],
            }
        )
        with pytest.raises(InsufficientDataError):
            run_clustering(tiny)

    def test_duplicate_symbol_raises_error(self) -> None:
        """Profile 中存在重复 symbol 应抛出 DataValidationError。"""
        dupes = pd.DataFrame(
            {
                "symbol": ["A", "A", "B", "C", "D"],
                "mean_return": [0.01, 0.02, 0.03, 0.04, 0.05],
                "volatility": [0.02, 0.03, 0.04, 0.05, 0.06],
                "max_drawdown": [-0.05, -0.10, -0.15, -0.20, -0.25],
            }
        )
        with pytest.raises(DataValidationError, match="重复 symbol"):
            run_clustering(dupes)

    def test_nan_in_profiles_raises_error(self) -> None:
        """Profile 中存在 NaN 应抛出 DataValidationError。"""
        nan_profiles = pd.DataFrame(
            {
                "symbol": ["A", "B", "C", "D", "E"],
                "mean_return": [0.01, float("nan"), 0.03, 0.04, 0.05],
                "volatility": [0.02, 0.03, 0.04, 0.05, 0.06],
                "max_drawdown": [-0.05, -0.10, -0.15, -0.20, -0.25],
            }
        )
        with pytest.raises(DataValidationError, match="NaN"):
            run_clustering(nan_profiles)

    def test_inf_in_profiles_raises_error(self) -> None:
        """Profile 中存在 inf 应抛出 DataValidationError。"""
        inf_profiles = pd.DataFrame(
            {
                "symbol": ["A", "B", "C", "D", "E"],
                "mean_return": [0.01, float("inf"), 0.03, 0.04, 0.05],
                "volatility": [0.02, 0.03, 0.04, 0.05, 0.06],
                "max_drawdown": [-0.05, -0.10, -0.15, -0.20, -0.25],
            }
        )
        with pytest.raises(DataValidationError, match="非有限值"):
            run_clustering(inf_profiles)

    def test_missing_column_raises_error(self) -> None:
        """Profile 缺少必要列应抛出 DataValidationError。"""
        bad = pd.DataFrame(
            {
                "symbol": ["A", "B", "C"],
                "mean_return": [0.01, 0.02, 0.03],
            }
        )
        with pytest.raises(DataValidationError):
            run_clustering(bad)


# ── 端到端集成测试 ────────────────────────────────────


class TestEndToEnd:
    """从原始行情数据到最终 ClusteringResult 的完整链路。"""

    def test_full_pipeline(self) -> None:
        """完整流程：行情 → Profile → 聚类 → Contract 结果。"""
        market_df = _make_market_data(
            symbols=["SH600000", "SH600036", "SH601318", "SZ000001", "SZ002415"],
            n_days=60,
        )

        profiles = build_stock_profiles(market_df)
        assert len(profiles) == 5

        result = run_clustering(profiles)

        assert set(result.keys()) == set(CLUSTERING_RESULT_KEYS)
        assert len(result["profiles"]) == 5
        assert len(result["cluster_centers"]) == N_CLUSTERS
        assert result["features"] == FEATURE_COLS
        assert result["k"] == N_CLUSTERS

        for col in FEATURE_COLS:
            center_min = result["cluster_centers"][col].min()
            center_max = result["cluster_centers"][col].max()
            profile_min = result["profiles"][col].min()
            profile_max = result["profiles"][col].max()
            margin = max(abs(profile_max - profile_min) * 0.5, 1e-6)
            assert center_min >= profile_min - margin
            assert center_max <= profile_max + margin

    def test_all_stocks_assigned_to_a_cluster(self) -> None:
        """每只股票都必须被分配到一个 cluster。"""
        market_df = _make_market_data(symbols=["A", "B", "C", "D", "E"])
        profiles = build_stock_profiles(market_df)
        result = run_clustering(profiles)

        assert result["profiles"]["cluster"].notna().all()
        assert result["profiles"]["cluster"].isin(range(N_CLUSTERS)).all()
