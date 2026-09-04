"""单元测试：动态簇标签生成与标签解读输出。

测试目标：
- generate_cluster_labels：根据簇中心特征相对排序生成中文标签
- build_label_interpretation：组装完整标签解读信息供 Role 1 使用

覆盖场景（来自组长 D4 要求）：
1. 3/5/10/20 只股票的聚类标签
2. 改变股票顺序后标签不变
3. 不足 3 只股票时的行为
4. 缺失数据时的行为
5. 极端值（极大收益率/极大波动率/极大回撤）时的标签
6. 特征中心值非常接近时的标签
7. 标签格式一致性：包含"相对""所选历史区间"
8. 标签可复现性：相同输入相同输出
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.contracts.clustering import CLUSTERING_RESULT_KEYS
from src.models.unsupervised.clustering import (
    FEATURE_COLS,
    build_label_interpretation,
    generate_cluster_labels,
    run_clustering,
)


# ── 辅助函数 ──────────────────────────────────────────


def _make_profiles(symbols: list[str], seed: int = 42) -> pd.DataFrame:
    """为指定股票列表生成随机但可复现的 profiles DataFrame。

    每只股票生成随机的 mean_return, volatility, max_drawdown，
    保证各股票特征值有差异以便聚类。
    """
    rng = np.random.RandomState(seed)
    n = len(symbols)
    profiles = pd.DataFrame({
        "symbol": symbols,
        "mean_return": rng.uniform(-0.03, 0.05, n),
        "volatility": rng.uniform(0.01, 0.25, n),
        "max_drawdown": rng.uniform(-0.5, -0.01, n),
    })
    return profiles


def _make_cluster_centers(
    mean_returns: list[float],
    volatilities: list[float],
    max_drawdowns: list[float],
) -> pd.DataFrame:
    """手动构造 cluster_centers DataFrame，用于单元测试标签生成逻辑。"""
    return pd.DataFrame({
        "cluster": list(range(len(mean_returns))),
        "mean_return": mean_returns,
        "volatility": volatilities,
        "max_drawdown": max_drawdowns,
    })


# ── TestGenerateClusterLabels ─────────────────────────


class TestGenerateClusterLabels:
    """测试 generate_cluster_labels 函数。"""

    def test_basic_label_generation(self):
        """3 个簇的基本标签生成。"""
        centers = _make_cluster_centers(
            mean_returns=[0.05, 0.01, -0.02],
            volatilities=[0.10, 0.20, 0.15],
            max_drawdowns=[-0.05, -0.30, -0.15],
        )
        labels = generate_cluster_labels(centers)

        assert len(labels) == 3
        assert set(labels.keys()) == {0, 1, 2}

        # cluster 0: mean_return 最高 → 相对高收益
        assert "相对高收益" in labels[0]
        # cluster 0: volatility 最低 → 相对低波动
        assert "相对低波动" in labels[0]
        # cluster 0: max_drawdown 最高（最接近 0）→ 相对小回撤
        assert "相对小回撤" in labels[0]

        # cluster 1: mean_return 中等 → 相对中等收益
        assert "相对中等收益" in labels[1]
        # cluster 1: volatility 最高 → 相对高波动
        assert "相对高波动" in labels[1]
        # cluster 1: max_drawdown 最低（最负）→ 相对大回撤
        assert "相对大回撤" in labels[1]

        # cluster 2: mean_return 最低 → 相对低收益
        assert "相对低收益" in labels[2]
        # cluster 2: volatility 中等 → 相对中等波动
        assert "相对中等波动" in labels[2]
        # cluster 2: max_drawdown 中等 → 相对中等回撤
        assert "相对中等回撤" in labels[2]

    def test_label_format_contains_required_qualifiers(self):
        """所有标签必须包含'相对'和'所选历史区间'限定词。"""
        centers = _make_cluster_centers(
            mean_returns=[0.05, 0.01, -0.02],
            volatilities=[0.10, 0.20, 0.15],
            max_drawdowns=[-0.05, -0.30, -0.15],
        )
        labels = generate_cluster_labels(centers)

        for cluster_id, label in labels.items():
            assert "相对" in label, f"簇 {cluster_id} 缺少'相对'限定词: {label}"
            assert "所选历史区间" in label, f"簇 {cluster_id} 缺少'所选历史区间': {label}"

    def test_label_ends_with_type_suffix(self):
        """标签以'型'结尾。"""
        centers = _make_cluster_centers(
            mean_returns=[0.05, 0.01, -0.02],
            volatilities=[0.10, 0.20, 0.15],
            max_drawdowns=[-0.05, -0.30, -0.15],
        )
        labels = generate_cluster_labels(centers)

        for label in labels.values():
            assert label.endswith("型"), f"标签未以'型'结尾: {label}"

    def test_label_format_with_five_clusters(self):
        """5 个簇时标签生成（中间值应为'相对中等'）。"""
        centers = _make_cluster_centers(
            mean_returns=[0.10, 0.05, 0.01, -0.01, -0.05],
            volatilities=[0.30, 0.20, 0.15, 0.10, 0.05],
            max_drawdowns=[-0.02, -0.10, -0.20, -0.35, -0.50],
        )
        labels = generate_cluster_labels(centers)

        assert len(labels) == 5

        # cluster 0: 各特征均最高 → 高收益-高波动-小回撤
        assert "相对高收益" in labels[0]
        assert "相对高波动" in labels[0]
        assert "相对小回撤" in labels[0]

        # cluster 4: 各特征均最低 → 低收益-低波动-大回撤
        assert "相对低收益" in labels[4]
        assert "相对低波动" in labels[4]
        assert "相对大回撤" in labels[4]

    def test_label_with_ten_clusters(self):
        """10 个簇时标签生成（中间值应为'相对中等'）。"""
        # 10 个递增的 mean_return 值
        mean_returns = list(np.linspace(0.10, -0.05, 10))
        volatilities = list(np.linspace(0.05, 0.30, 10))
        max_drawdowns = list(np.linspace(-0.02, -0.50, 10))

        centers = _make_cluster_centers(
            mean_returns=mean_returns,
            volatilities=volatilities,
            max_drawdowns=max_drawdowns,
        )
        labels = generate_cluster_labels(centers)

        assert len(labels) == 10
        # 检查标签包含限定词
        for label in labels.values():
            assert "相对" in label
            assert "所选历史区间" in label


class TestLabelConsistency:
    """测试标签一致性：改变输入顺序不影响标签内容。"""

    def test_same_labels_after_shuffling_clusters(self):
        """打乱 cluster_centers 行顺序后，相同簇编号应得到相同标签。"""
        # 原始顺序
        centers_original = _make_cluster_centers(
            mean_returns=[0.05, 0.01, -0.02],
            volatilities=[0.10, 0.20, 0.15],
            max_drawdowns=[-0.05, -0.30, -0.15],
        )
        labels_original = generate_cluster_labels(centers_original)

        # 打乱顺序（cluster 编号不变，只是行顺序不同）
        centers_shuffled = centers_original.iloc[[2, 0, 1]].reset_index(drop=True)
        labels_shuffled = generate_cluster_labels(centers_shuffled)

        # 相同 cluster 编号应有相同标签
        for cluster_id in labels_original:
            assert labels_original[cluster_id] == labels_shuffled[cluster_id]


class TestLabelReproducibility:
    """测试标签可复现性。"""

    def test_same_input_same_labels(self):
        """相同输入多次调用生成相同标签。"""
        centers = _make_cluster_centers(
            mean_returns=[0.05, 0.01, -0.02],
            volatilities=[0.10, 0.20, 0.15],
            max_drawdowns=[-0.05, -0.30, -0.15],
        )
        labels_1 = generate_cluster_labels(centers)
        labels_2 = generate_cluster_labels(centers)

        assert labels_1 == labels_2

    def test_end_to_end_reproducibility(self):
        """完整流程：相同 profiles 和 random_state 产生相同标签。"""
        profiles = _make_profiles(["A", "B", "C", "D", "E"])
        result_1 = run_clustering(profiles, random_state=42)
        result_2 = run_clustering(profiles, random_state=42)

        assert result_1["cluster_label"] == result_2["cluster_label"]


class TestLabelWithExtremeValues:
    """测试极端值时的标签生成。"""

    def test_extreme_positive_return(self):
        """极大正收益率时标签仍正确。"""
        centers = _make_cluster_centers(
            mean_returns=[1.0, 0.01, -0.02],
            volatilities=[0.10, 0.20, 0.15],
            max_drawdowns=[-0.05, -0.30, -0.15],
        )
        labels = generate_cluster_labels(centers)

        # 最高收益 → 相对高收益
        assert "相对高收益" in labels[0]

    def test_extreme_negative_return(self):
        """极大负收益率时标签仍正确。"""
        centers = _make_cluster_centers(
            mean_returns=[0.05, 0.01, -1.0],
            volatilities=[0.10, 0.20, 0.15],
            max_drawdowns=[-0.05, -0.30, -0.15],
        )
        labels = generate_cluster_labels(centers)

        # 最低收益 → 相对低收益
        assert "相对低收益" in labels[2]

    def test_extreme_volatility(self):
        """极大波动率时标签仍正确。"""
        centers = _make_cluster_centers(
            mean_returns=[0.05, 0.01, -0.02],
            volatilities=[100.0, 0.20, 0.15],
            max_drawdowns=[-0.05, -0.30, -0.15],
        )
        labels = generate_cluster_labels(centers)

        # 最高波动 → 相对高波动
        assert "相对高波动" in labels[0]

    def test_extreme_max_drawdown(self):
        """极大回撤时标签仍正确。"""
        centers = _make_cluster_centers(
            mean_returns=[0.05, 0.01, -0.02],
            volatilities=[0.10, 0.20, 0.15],
            max_drawdowns=[-0.05, -0.30, -0.99],
        )
        labels = generate_cluster_labels(centers)

        # 最大回撤（最负） → 相对大回撤
        assert "相对大回撤" in labels[2]

    def test_similar_center_values(self):
        """特征中心值非常接近时标签仍能区分。"""
        centers = _make_cluster_centers(
            mean_returns=[0.0100, 0.0101, 0.0099],
            volatilities=[0.150, 0.151, 0.149],
            max_drawdowns=[-0.200, -0.201, -0.199],
        )
        labels = generate_cluster_labels(centers)

        # 仍然可以排序并生成标签
        assert len(labels) == 3
        for label in labels.values():
            assert "相对" in label
            assert "所选历史区间" in label


class TestLabelWithRealClustering:
    """通过 run_clustering 生成的 cluster_centers 测试标签。"""

    def test_label_from_real_clustering_3_stocks(self):
        """3 只股票聚类后标签格式正确。"""
        profiles = _make_profiles(["A", "B", "C"], seed=10)
        result = run_clustering(profiles, random_state=42)

        labels = result["cluster_label"]
        assert len(labels) == 3
        for cluster_id, label in labels.items():
            assert isinstance(cluster_id, int)
            assert isinstance(label, str)
            assert "相对" in label
            assert "所选历史区间" in label
            assert label.endswith("型")

    def test_label_from_real_clustering_10_stocks(self):
        """10 只股票聚类后标签格式正确。"""
        profiles = _make_profiles([f"S{i:02d}" for i in range(10)], seed=20)
        result = run_clustering(profiles, random_state=42)

        labels = result["cluster_label"]
        assert len(labels) == 3
        for label in labels.values():
            assert "相对" in label
            assert "所选历史区间" in label

    def test_label_from_real_clustering_20_stocks(self):
        """20 只股票聚类后标签格式正确。"""
        profiles = _make_profiles([f"S{i:02d}" for i in range(20)], seed=30)
        result = run_clustering(profiles, random_state=42)

        labels = result["cluster_label"]
        assert len(labels) == 3
        for label in labels.values():
            assert "相对" in label
            assert "所选历史区间" in label

    def test_profiles_order_change_same_labels(self):
        """改变 profiles 行顺序不影响同一簇的标签。"""
        profiles_v1 = _make_profiles(["A", "B", "C", "D", "E"], seed=42)
        result_v1 = run_clustering(profiles_v1, random_state=42)

        # 打乱 profiles 顺序
        profiles_v2 = profiles_v1.iloc[[4, 2, 0, 3, 1]].reset_index(drop=True)
        result_v2 = run_clustering(profiles_v2, random_state=42)

        # 两者的 cluster_label 应相同（特征值不变，聚类不变）
        assert result_v1["cluster_label"] == result_v2["cluster_label"]


class TestBuildLabelInterpretation:
    """测试 build_label_interpretation 函数。"""

    def test_interpretation_has_all_keys(self):
        """解读结果包含所有必要字段。"""
        profiles = _make_profiles(["A", "B", "C", "D", "E"])
        result = run_clustering(profiles, random_state=42)
        interpretation = build_label_interpretation(result)

        expected_keys = {
            "cluster_label", "画像指标", "簇数量",
            "簇中心特征", "标签依据", "样本范围", "免责声明",
        }
        assert set(interpretation.keys()) == expected_keys

    def test_cluster_label_matches(self):
        """解读中的 cluster_label 与聚类结果一致。"""
        profiles = _make_profiles(["A", "B", "C", "D", "E"])
        result = run_clustering(profiles, random_state=42)
        interpretation = build_label_interpretation(result)

        assert interpretation["cluster_label"] == result["cluster_label"]

    def test_profile_metrics_are_feature_cols(self):
        """画像指标即 FEATURE_COLS。"""
        profiles = _make_profiles(["A", "B", "C", "D", "E"])
        result = run_clustering(profiles, random_state=42)
        interpretation = build_label_interpretation(result)

        assert interpretation["画像指标"] == list(FEATURE_COLS)

    def test_cluster_count_matches_k(self):
        """簇数量与 k 一致。"""
        profiles = _make_profiles(["A", "B", "C", "D", "E"])
        result = run_clustering(profiles, random_state=42)
        interpretation = build_label_interpretation(result)

        assert interpretation["簇数量"] == result["k"]

    def test_center_features_count(self):
        """簇中心特征列表长度等于 k。"""
        profiles = _make_profiles(["A", "B", "C", "D", "E"])
        result = run_clustering(profiles, random_state=42)
        interpretation = build_label_interpretation(result)

        assert len(interpretation["簇中心特征"]) == result["k"]

    def test_center_features_have_correct_fields(self):
        """每个簇中心特征包含 cluster 和三个指标。"""
        profiles = _make_profiles(["A", "B", "C", "D", "E"])
        result = run_clustering(profiles, random_state=42)
        interpretation = build_label_interpretation(result)

        for center in interpretation["簇中心特征"]:
            assert "cluster" in center
            assert "mean_return" in center
            assert "volatility" in center
            assert "max_drawdown" in center

    def test_sample_scope_without_date_range(self):
        """无日期范围时样本范围仅含股票数量。"""
        profiles = _make_profiles(["A", "B", "C", "D", "E"])
        result = run_clustering(profiles, random_state=42)
        interpretation = build_label_interpretation(result)

        scope = interpretation["样本范围"]
        assert "股票数量" in scope
        assert scope["股票数量"] == 5
        assert "起始日期" not in scope
        assert "截止日期" not in scope

    def test_sample_scope_with_date_range(self):
        """有日期范围时样本范围包含起止日期。"""
        profiles = _make_profiles(["A", "B", "C", "D", "E"])
        result = run_clustering(profiles, random_state=42)
        interpretation = build_label_interpretation(
            result, date_range=("2025-01-01", "2025-06-30")
        )

        scope = interpretation["样本范围"]
        assert scope["股票数量"] == 5
        assert scope["起始日期"] == "2025-01-01"
        assert scope["截止日期"] == "2025-06-30"

    def test_disclaimer_contains_no_investment_advice(self):
        """免责声明包含不构成投资建议的表述。"""
        profiles = _make_profiles(["A", "B", "C", "D", "E"])
        result = run_clustering(profiles, random_state=42)
        interpretation = build_label_interpretation(result)

        assert "不构成任何投资建议" in interpretation["免责声明"]

    def test_tagline_basis_contains_required_keywords(self):
        """标签依据包含必要的关键词。"""
        profiles = _make_profiles(["A", "B", "C", "D", "E"])
        result = run_clustering(profiles, random_state=42)
        interpretation = build_label_interpretation(result)

        basis = interpretation["标签依据"]
        assert "相对排序" in basis
        assert "相对" in basis
        assert "所选历史区间" in basis

    def test_clustering_result_has_cluster_label_key(self):
        """ClusteringResult 包含 cluster_label 键。"""
        profiles = _make_profiles(["A", "B", "C", "D", "E"])
        result = run_clustering(profiles, random_state=42)

        assert "cluster_label" in result
        assert "cluster_label" in CLUSTERING_RESULT_KEYS
