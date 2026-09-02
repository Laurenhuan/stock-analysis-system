"""Role 5 聚类模块单元测试。

覆盖场景：
1. build_stock_profiles — 正常输入、缺少列、空 DataFrame、NaN 值
2. run_clustering — 正常输入、输出 shape、可复现性、股票不足
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
        # 生成随机价格：起始价 10~100，每日涨跌幅 ~N(0, 0.02)
        start_price = rng.uniform(10, 100)
        returns = rng.normal(0.001, 0.02, size=n_days)
        prices = start_price * np.cumprod(1 + returns)

        # drawdown：从区间内最高点算起的回撤
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

    def test_missing_column_raises_error(self) -> None:
        """缺少必要列应抛出 DataValidationError。"""
        df = pd.DataFrame({"symbol": ["A"], "close": [10.0]})
        # 缺少 drawdown 和 trade_date
        with pytest.raises(DataValidationError):
            build_stock_profiles(df)

    def test_empty_dataframe_raises_error(self) -> None:
        """空 DataFrame 应抛出 DataValidationError。"""
        df = pd.DataFrame(columns=["symbol", "trade_date", "close", "drawdown"])
        with pytest.raises(DataValidationError):
            build_stock_profiles(df)

    def test_values_are_finite(self) -> None:
        """输出的特征值必须是有限数值（非 inf / -inf）。"""
        df = _make_market_data()
        profiles = build_stock_profiles(df)

        for col in FEATURE_COLS:
            assert np.isfinite(profiles[col]).all(), f"{col} 含有非有限值"


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

        # 只验证不报错，不强制要求结果不同
        assert len(result1["profiles"]) == len(result2["profiles"])

    def test_too_few_stocks_raises_error(self) -> None:
        """股票数 < k 应抛出 InsufficientDataError。"""
        # 只有2只股票，但 k=3
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

    def test_missing_column_raises_error(self) -> None:
        """Profile 缺少必要列应抛出 DataValidationError。"""
        bad = pd.DataFrame(
            {
                "symbol": ["A", "B", "C"],
                "mean_return": [0.01, 0.02, 0.03],
                # 缺少 volatility 和 max_drawdown
            }
        )
        with pytest.raises(DataValidationError):
            run_clustering(bad)


# ── 端到端集成测试 ────────────────────────────────────


class TestEndToEnd:
    """从原始行情数据到最终 ClusteringResult 的完整链路。"""

    def test_full_pipeline(self) -> None:
        """完整流程：行情 → Profile → 聚类 → Contract 结果。"""
        # 1. 模拟行情数据（5只股票，60天）
        market_df = _make_market_data(
            symbols=["SH600000", "SH600036", "SH601318", "SZ000001", "SZ002415"],
            n_days=60,
        )

        # 2. 构建 Profile Table
        profiles = build_stock_profiles(market_df)
        assert len(profiles) == 5

        # 3. 聚类
        result = run_clustering(profiles)

        # 4. 验证 Contract 完整性
        assert set(result.keys()) == set(CLUSTERING_RESULT_KEYS)
        assert len(result["profiles"]) == 5
        assert len(result["cluster_centers"]) == N_CLUSTERS
        assert result["features"] == FEATURE_COLS
        assert result["k"] == N_CLUSTERS

        # 5. 验证 cluster_centers 的数值在原始尺度的合理范围内
        # 中心点是组内均值，不应超出整体数据的 min/max 范围太多
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
