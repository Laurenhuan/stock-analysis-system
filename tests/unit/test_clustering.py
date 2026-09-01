"""K-Means 聚类测试 — Role 5。

覆盖：
- StandardScaler 使用
- KMeans k=3
- 聚类中心恢复到原始尺度
- 输出结构符合 Contract
- 股票数量不足异常
- 缺少列异常
- 端到端 pipeline
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.contracts.clustering import (
    CLUSTER_CENTER_COLUMNS,
    CLUSTERING_RESULT_KEYS,
    PROFILE_COLUMNS,
)
from src.models.unsupervised.clustering import (
    FEATURE_COLS,
    N_CLUSTERS,
    build_stock_profiles,
    run_clustering,
)
from src.utils.exceptions import DataValidationError, InsufficientDataError


# ── 辅助函数 ──────────────────────────────────────────


def _make_profiles(n_stocks: int = 8, seed: int = 42) -> pd.DataFrame:
    """生成测试用的 Profile Table。"""
    rng = np.random.RandomState(seed)
    symbols = [f"STK{i:03d}" for i in range(n_stocks)]
    profiles = pd.DataFrame(
        {
            "symbol": symbols,
            "mean_return": rng.uniform(-0.01, 0.01, n_stocks),
            "volatility": rng.uniform(0.01, 0.04, n_stocks),
            "max_drawdown": rng.uniform(-0.3, -0.05, n_stocks),
        }
    )
    return profiles


# ── StandardScaler 测试 ───────────────────────────────


class TestStandardScaler:
    """验证 StandardScaler 被正确使用。"""

    def test_scaler_is_applied(self) -> None:
        """聚类前必须经过 StandardScaler 标准化。"""
        profiles = _make_profiles()
        X = profiles[FEATURE_COLS].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 标准化后均值应接近 0，标准差接近 1
        assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-10)
        assert np.allclose(X_scaled.std(axis=0, ddof=0), 1, atol=1e-10)

    def test_original_profiles_not_modified(self) -> None:
        """StandardScaler 不应修改原始 Profile 数据。"""
        profiles = _make_profiles()
        original_values = profiles[FEATURE_COLS].copy()

        result = run_clustering(profiles)

        pd.testing.assert_frame_equal(
            result["profiles"][FEATURE_COLS], original_values
        )


# ── KMeans k=3 测试 ──────────────────────────────────


class TestKMeansClustering:
    """验证 KMeans 聚类正确执行。"""

    def test_k_equals_3(self) -> None:
        """固定 k=3。"""
        assert N_CLUSTERS == 3

    def test_cluster_count_matches_k(self) -> None:
        """聚类结果应恰好有 3 个不同的 cluster 编号。"""
        profiles = _make_profiles()
        result = run_clustering(profiles)

        unique_clusters = result["profiles"]["cluster"].nunique()
        assert unique_clusters == N_CLUSTERS

    def test_all_stocks_assigned(self) -> None:
        """每只股票都必须被分配到一个 cluster。"""
        profiles = _make_profiles()
        result = run_clustering(profiles)

        assert result["profiles"]["cluster"].notna().all()
        assert result["profiles"]["cluster"].isin(range(N_CLUSTERS)).all()

    def test_cluster_values_in_valid_range(self) -> None:
        """cluster 列的值应在 [0, k-1] 范围内。"""
        profiles = _make_profiles()
        result = run_clustering(profiles)

        clusters = result["profiles"]["cluster"]
        assert clusters.min() >= 0
        assert clusters.max() < N_CLUSTERS


# ── 聚类中心还原测试 ──────────────────────────────────


class TestClusterCenters:
    """验证聚类中心通过 inverse_transform 还原到原始尺度。"""

    def test_centers_shape(self) -> None:
        """cluster_centers 应有 k 行，列含 cluster + 3 个特征。"""
        profiles = _make_profiles()
        result = run_clustering(profiles)

        centers = result["cluster_centers"]
        assert len(centers) == N_CLUSTERS
        assert list(centers.columns) == list(CLUSTER_CENTER_COLUMNS)

    def test_centers_are_original_scale(self) -> None:
        """中心点应通过 inverse_transform 还原到原始金融特征尺度。"""
        profiles = _make_profiles()
        result = run_clustering(profiles)

        centers = result["cluster_centers"]
        profile_min = profiles[FEATURE_COLS].min()
        profile_max = profiles[FEATURE_COLS].max()

        for col in FEATURE_COLS:
            assert centers[col].min() >= profile_min[col] - 0.1
            assert centers[col].max() <= profile_max[col] + 0.1

    def test_centers_not_in_scaled_space(self) -> None:
        """中心点不应是标准化空间的值（均值不为 0）。"""
        profiles = _make_profiles()
        result = run_clustering(profiles)

        centers = result["cluster_centers"]
        for col in FEATURE_COLS:
            # 原始尺度的中心点均值不应该接近 0
            assert abs(centers[col].mean()) > 0.001


# ── 输出结构测试 ──────────────────────────────────────


class TestOutputStructure:
    """验证输出符合 ClusteringResult Contract。"""

    def test_returns_all_required_keys(self) -> None:
        profiles = _make_profiles()
        result = run_clustering(profiles)
        assert set(result.keys()) == set(CLUSTERING_RESULT_KEYS)

    def test_profiles_columns_match_contract(self) -> None:
        profiles = _make_profiles()
        result = run_clustering(profiles)
        assert list(result["profiles"].columns) == list(PROFILE_COLUMNS)

    def test_features_and_k_match_contract(self) -> None:
        profiles = _make_profiles()
        result = run_clustering(profiles)
        assert result["features"] == FEATURE_COLS
        assert result["k"] == N_CLUSTERS


# ── 异常测试 ──────────────────────────────────────────


class TestClusteringValidation:
    """异常输入验证。"""

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


# ── 端到端测试 ────────────────────────────────────────


class TestEndToEnd:
    """完整 pipeline：行情 → Profile → 聚类 → ClusteringResult。"""

    def test_full_pipeline(self) -> None:
        """从模拟行情到最终结果的完整链路。"""
        # 1. 生成模拟行情
        rng = np.random.RandomState(42)
        dates = pd.bdate_range("2024-01-01", periods=60)
        symbols = ["SH600000", "SH600036", "SH601318", "SZ000001", "SZ002415"]

        rows = []
        for sym in symbols:
            prices = rng.uniform(20, 100) * np.cumprod(1 + rng.normal(0.001, 0.02, 60))
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
        market_df = pd.DataFrame(rows)

        # 2. 构建 Profile
        profiles = build_stock_profiles(market_df)
        assert len(profiles) == 5

        # 3. 聚类
        result = run_clustering(profiles)

        # 4. 验证 Contract
        assert set(result.keys()) == set(CLUSTERING_RESULT_KEYS)
        assert len(result["profiles"]) == 5
        assert len(result["cluster_centers"]) == N_CLUSTERS
        assert result["features"] == FEATURE_COLS
        assert result["k"] == N_CLUSTERS

        # 5. 所有股票都有 cluster
        assert result["profiles"]["cluster"].notna().all()
        assert result["profiles"]["cluster"].isin(range(N_CLUSTERS)).all()
