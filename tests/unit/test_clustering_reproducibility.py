"""K-Means 可复现性测试 — Role 5。

覆盖：
- 相同 random_state 产生完全相同的结果
- 不同 random_state 可能产生不同分组
- 随机种子和 sklearn 参数记录
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.unsupervised.clustering import (
    DEFAULT_RANDOM_STATE,
    N_CLUSTERS,
    run_clustering,
)


# ── 辅助函数 ──────────────────────────────────────────


def _make_profiles(n_stocks: int = 10, seed: int = 42) -> pd.DataFrame:
    """生成测试用的 Profile Table。"""
    rng = np.random.RandomState(seed)
    symbols = [f"STK{i:03d}" for i in range(n_stocks)]
    return pd.DataFrame(
        {
            "symbol": symbols,
            "mean_return": rng.uniform(-0.01, 0.01, n_stocks),
            "volatility": rng.uniform(0.01, 0.04, n_stocks),
            "max_drawdown": rng.uniform(-0.3, -0.05, n_stocks),
        }
    )


# ── 可复现性测试 ──────────────────────────────────────


class TestReproducibility:
    """验证相同参数产生相同结果。"""

    def test_same_seed_same_result(self) -> None:
        """相同 random_state 应产生完全相同的结果。"""
        profiles = _make_profiles()

        result1 = run_clustering(profiles, random_state=123)
        result2 = run_clustering(profiles, random_state=123)

        pd.testing.assert_frame_equal(result1["profiles"], result2["profiles"])
        pd.testing.assert_frame_equal(
            result1["cluster_centers"], result2["cluster_centers"]
        )

    def test_same_default_seed(self) -> None:
        """使用默认 random_state 也应可复现。"""
        profiles = _make_profiles()

        result1 = run_clustering(profiles)
        result2 = run_clustering(profiles)

        pd.testing.assert_frame_equal(result1["profiles"], result2["profiles"])

    def test_different_seed_may_differ(self) -> None:
        """不同 random_state 可能产生不同分组。"""
        profiles = _make_profiles()

        result1 = run_clustering(profiles, random_state=0)
        result2 = run_clustering(profiles, random_state=999)

        # 只验证不报错，不强制要求结果不同
        assert len(result1["profiles"]) == len(result2["profiles"])

    def test_reproducibility_across_multiple_runs(self) -> None:
        """连续运行 5 次，结果应完全一致。"""
        profiles = _make_profiles()
        results = [run_clustering(profiles, random_state=42) for _ in range(5)]

        for i in range(1, len(results)):
            pd.testing.assert_frame_equal(results[0]["profiles"], results[i]["profiles"])


# ── 参数记录测试 ──────────────────────────────────────


class TestParameterRecording:
    """验证关键参数被正确记录。"""

    def test_default_random_state_is_42(self) -> None:
        """默认随机种子应为 42。"""
        assert DEFAULT_RANDOM_STATE == 42

    def test_n_clusters_is_3(self) -> None:
        """聚类数应固定为 3。"""
        assert N_CLUSTERS == 3

    def test_random_state_can_be_customized(self) -> None:
        """应支持自定义 random_state。"""
        profiles = _make_profiles()

        result = run_clustering(profiles, random_state=999)
        # 不报错即通过，具体值由 sklearn 决定
        assert len(result["profiles"]) == len(profiles)
