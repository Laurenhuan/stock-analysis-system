"""单元测试 — generate_cluster_labels / build_label_interpretation。

覆盖组长 D4 要求的全部场景：
- 3/5/10 只股票端到端标签生成
- 打乱行顺序后每个 cluster 的标签不变
- 三个簇中心完全相同
- 并列最高、并列最低
- 空表、缺列、NaN/Inf、重复 cluster ID
- 标签格式与一致性验证
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.contracts.clustering import ClusteringResult
from src.models.unsupervised.clustering import (
    FEATURE_COLS,
    build_label_interpretation,
    generate_cluster_labels,
    run_clustering,
    build_stock_profiles,
)
from src.utils.exceptions import DataValidationError


# ── 工具函数 ──────────────────────────────────────────


def _make_centers(rows: list[tuple[int, float, float, float]]) -> pd.DataFrame:
    """从 (cluster, mean_return, volatility, max_drawdown) 列表构造 centers。"""
    return pd.DataFrame(rows, columns=["cluster", *FEATURE_COLS])


def _make_symbol_data(
    symbol: str,
    returns: list[float],
    drawdowns: list[float],
    dates: list[str] | None = None,
) -> pd.DataFrame:
    """构造单只股票的行情数据。"""
    if dates is None:
        dates = [f"2025-01-0{i}" for i in range(1, len(returns) + 1)]
    return pd.DataFrame(
        {"symbol": symbol, "trade_date": dates, "return": returns, "drawdown": drawdowns}
    )


# ── 标签格式验证 ──────────────────────────────────────


class TestLabelFormat:
    """验证标签的基本格式。"""

    def test_label_format(self):
        centers = _make_centers([
            (0, 0.05, 0.10, -0.05),
            (1, 0.01, 0.20, -0.30),
            (2, -0.02, 0.15, -0.15),
        ])
        labels = generate_cluster_labels(centers)

        assert len(labels) == 3
        for cluster_id, label in labels.items():
            assert label.startswith("[所选历史区间]")
            assert label.endswith("型")
            assert "相对" in label

    def test_label_contains_all_features(self):
        centers = _make_centers([
            (0, 0.05, 0.10, -0.05),
            (1, 0.01, 0.20, -0.30),
            (2, -0.02, 0.15, -0.15),
        ])
        labels = generate_cluster_labels(centers)

        for label in labels.values():
            assert "收益" in label
            assert "波动" in label
            assert "回撤" in label


# ── 3 簇标准场景 ──────────────────────────────────────


class TestThreeClusters:
    """验证 3 簇场景的标签生成。"""

    def test_three_clusters_high_mid_low(self):
        """最高收益得到'高收益'标签，最低得到'低收益'。"""
        centers = _make_centers([
            (0, 0.05, 0.10, -0.05),
            (1, 0.01, 0.20, -0.30),
            (2, -0.02, 0.15, -0.15),
        ])
        labels = generate_cluster_labels(centers)

        # cluster 0: mean_return 最高 → 高收益
        assert "高收益" in labels[0]
        # cluster 2: mean_return 最低 → 低收益
        assert "低收益" in labels[2]
        # cluster 1: mean_return 中间 → 中等收益
        assert "中等收益" in labels[1]

    def test_three_clusters_volatility_ranking(self):
        """波动率最高 → 高波动，最低 → 低波动。"""
        centers = _make_centers([
            (0, 0.05, 0.10, -0.05),
            (1, 0.01, 0.30, -0.30),
            (2, -0.02, 0.15, -0.15),
        ])
        labels = generate_cluster_labels(centers)

        # cluster 1: volatility 最高 → 高波动
        assert "高波动" in labels[1]
        # cluster 0: volatility 最低 → 低波动
        assert "低波动" in labels[0]

    def test_three_clusters_drawdown_ranking(self):
        """回撤最接近 0 → 小回撤，最负 → 大回撤。"""
        centers = _make_centers([
            (0, 0.05, 0.10, -0.02),  # 最接近 0
            (1, 0.01, 0.20, -0.50),  # 最负
            (2, -0.02, 0.15, -0.15),
        ])
        labels = generate_cluster_labels(centers)

        # cluster 0: max_drawdown 最高（-0.02 > -0.50）→ 小回撤
        assert "小回撤" in labels[0]
        # cluster 1: max_drawdown 最低 → 大回撤
        assert "大回撤" in labels[1]


# ── 5 簇和 10 簇 ──────────────────────────────────────


class TestMoreClusters:
    """验证 k>3 时中间簇用'中等'描述。"""

    def test_five_clusters(self):
        centers = _make_centers([
            (0, 0.10, 0.05, -0.01),
            (1, 0.05, 0.10, -0.05),
            (2, 0.01, 0.15, -0.15),
            (3, -0.01, 0.20, -0.30),
            (4, -0.05, 0.25, -0.50),
        ])
        labels = generate_cluster_labels(centers)

        assert len(labels) == 5
        # cluster 0: 收益最高 → 高收益
        assert "高收益" in labels[0]
        # cluster 4: 收益最低 → 低收益
        assert "低收益" in labels[4]
        # 中间簇 → 中等收益
        assert "中等收益" in labels[2]

    def test_ten_clusters(self):
        centers = _make_centers([
            (i, 0.10 - i * 0.02, 0.05 + i * 0.02, -(0.01 + i * 0.05))
            for i in range(10)
        ])
        labels = generate_cluster_labels(centers)

        assert len(labels) == 10
        assert "高收益" in labels[0]
        assert "低收益" in labels[9]
        # 中间簇都应该有"中等"
        for i in range(1, 9):
            assert "中等收益" in labels[i]


# ── 行顺序无关性 ──────────────────────────────────────


class TestLabelConsistency:
    """打乱行顺序后，每个 cluster 的标签不变。"""

    def test_row_order_does_not_affect_labels(self):
        centers_original = _make_centers([
            (0, 0.05, 0.10, -0.05),
            (1, 0.01, 0.20, -0.30),
            (2, -0.02, 0.15, -0.15),
        ])
        labels_original = generate_cluster_labels(centers_original)

        # 打乱行顺序
        centers_shuffled = centers_original.sample(frac=1, random_state=99).reset_index(
            drop=True
        )
        labels_shuffled = generate_cluster_labels(centers_shuffled)

        assert labels_original == labels_shuffled

    def test_row_order_reverse(self):
        centers_original = _make_centers([
            (0, 0.05, 0.10, -0.05),
            (1, 0.01, 0.20, -0.30),
            (2, -0.02, 0.15, -0.15),
        ])
        labels_original = generate_cluster_labels(centers_original)

        # 反转行顺序
        centers_reversed = centers_original.iloc[::-1].reset_index(drop=True)
        labels_reversed = generate_cluster_labels(centers_reversed)

        assert labels_original == labels_reversed

    def test_five_clusters_row_order(self):
        """5 簇场景下打乱行顺序不影响标签。"""
        centers_original = _make_centers([
            (0, 0.10, 0.05, -0.01),
            (1, 0.05, 0.10, -0.05),
            (2, 0.01, 0.15, -0.15),
            (3, -0.01, 0.20, -0.30),
            (4, -0.05, 0.25, -0.50),
        ])
        labels_original = generate_cluster_labels(centers_original)

        centers_shuffled = centers_original.sample(frac=1, random_state=42).reset_index(
            drop=True
        )
        labels_shuffled = generate_cluster_labels(centers_shuffled)

        assert labels_original == labels_shuffled


# ── 完全相同的簇中心 ──────────────────────────────────


class TestIdenticalCenters:
    """三个簇中心完全相同时，所有簇应得到相同的'相对接近'标签。"""

    def test_all_identical_centers(self):
        centers = _make_centers([
            (0, 0.01, 0.15, -0.10),
            (1, 0.01, 0.15, -0.10),
            (2, 0.01, 0.15, -0.10),
        ])
        labels = generate_cluster_labels(centers)

        # 所有簇应得到完全相同的标签
        assert labels[0] == labels[1] == labels[2]
        # 应包含"相对接近"
        assert "相对接近收益" in labels[0]
        assert "相对接近波动" in labels[0]
        assert "相对接近回撤" in labels[0]

    def test_identical_centers_row_order(self):
        """完全相同的簇中心，打乱行顺序不影响标签。"""
        centers_original = _make_centers([
            (0, 0.01, 0.15, -0.10),
            (1, 0.01, 0.15, -0.10),
            (2, 0.01, 0.15, -0.10),
        ])
        labels_original = generate_cluster_labels(centers_original)

        centers_shuffled = centers_original.sample(frac=1, random_state=7).reset_index(
            drop=True
        )
        labels_shuffled = generate_cluster_labels(centers_shuffled)

        assert labels_original == labels_shuffled


# ── 并列值 ────────────────────────────────────────────


class TestTiedValues:
    """并列最高、并列最低的场景。"""

    def test_tied_highest_mean_return(self):
        """两个簇并列最高收益，应得到相同标签。"""
        centers = _make_centers([
            (0, 0.05, 0.10, -0.05),
            (1, 0.05, 0.20, -0.30),  # 并列最高收益
            (2, -0.02, 0.15, -0.15),
        ])
        labels = generate_cluster_labels(centers)

        # cluster 0 和 1 并列最高收益 → 都应包含"高收益"
        assert "高收益" in labels[0]
        assert "高收益" in labels[1]
        # cluster 2 最低 → 低收益
        assert "低收益" in labels[2]

    def test_tied_lowest_mean_return(self):
        """两个簇并列最低收益，应得到相同标签。"""
        centers = _make_centers([
            (0, 0.05, 0.10, -0.05),
            (1, -0.02, 0.20, -0.30),
            (2, -0.02, 0.15, -0.15),  # 并列最低收益
        ])
        labels = generate_cluster_labels(centers)

        # cluster 0 最高 → 高收益
        assert "高收益" in labels[0]
        # cluster 1 和 2 并列最低 → 都应包含"低收益"
        assert "低收益" in labels[1]
        assert "低收益" in labels[2]

    def test_tied_highest_and_lowest(self):
        """两簇并列最高，两簇并列最低。"""
        centers = _make_centers([
            (0, 0.05, 0.10, -0.05),
            (1, 0.05, 0.20, -0.30),
            (2, -0.02, 0.15, -0.15),
            (3, -0.02, 0.25, -0.40),
            (4, 0.01, 0.12, -0.10),
        ])
        labels = generate_cluster_labels(centers)

        # cluster 0,1 并列最高 → 高收益
        assert "高收益" in labels[0]
        assert "高收益" in labels[1]
        # cluster 2,3 并列最低 → 低收益
        assert "低收益" in labels[2]
        assert "低收益" in labels[3]
        # cluster 4 中间 → 中等收益
        assert "中等收益" in labels[4]

    def test_tied_volatility(self):
        """并列波动率值。"""
        centers = _make_centers([
            (0, 0.05, 0.10, -0.05),
            (1, 0.01, 0.10, -0.30),  # 并列最低波动
            (2, -0.02, 0.30, -0.15),
        ])
        labels = generate_cluster_labels(centers)

        # cluster 0 和 1 并列最低波动 → 低波动
        assert "低波动" in labels[0]
        assert "低波动" in labels[1]
        # cluster 2 最高波动 → 高波动
        assert "高波动" in labels[2]


# ── 近似相同的特征值（相对接近）────────────────────────


class TestNearIdenticalValues:
    """当所有簇的某个特征值几乎相同时，应标为'相对接近'。"""

    def test_near_identical_mean_return(self):
        """三个簇的 mean_return 几乎相同。"""
        centers = _make_centers([
            (0, 0.010000, 0.10, -0.05),
            (1, 0.010001, 0.20, -0.30),
            (2, 0.010002, 0.15, -0.15),
        ])
        labels = generate_cluster_labels(centers)

        # mean_return 差异极小 → 都应标为"相对接近收益"
        assert "相对接近收益" in labels[0]
        assert "相对接近收益" in labels[1]
        assert "相对接近收益" in labels[2]

    def test_near_identical_volatility(self):
        """三个簇的 volatility 几乎相同。"""
        centers = _make_centers([
            (0, 0.05, 0.150000, -0.05),
            (1, 0.01, 0.150001, -0.30),
            (2, -0.02, 0.150002, -0.15),
        ])
        labels = generate_cluster_labels(centers)

        assert "相对接近波动" in labels[0]
        assert "相对接近波动" in labels[1]
        assert "相对接近波动" in labels[2]

    def test_mixed_identical_and_different(self):
        """一个特征相同，其他不同。"""
        centers = _make_centers([
            (0, 0.01, 0.10, -0.05),
            (1, 0.01, 0.20, -0.30),
            (2, 0.01, 0.15, -0.15),
        ])
        labels = generate_cluster_labels(centers)

        # mean_return 相同 → 相接近收益
        assert "相对接近收益" in labels[0]
        assert "相对接近收益" in labels[1]
        assert "相对接近收益" in labels[2]
        # volatility 不同 → 有高有低
        assert "高波动" in labels[1]
        assert "低波动" in labels[0]


# ── 端到端：从真实聚类结果生成标签 ────────────────────


class TestEndToEnd:
    """从 build_stock_profiles → run_clustering → generate_cluster_labels 完整流程。"""

    def _make_portfolio(self, n_stocks: int = 10) -> pd.DataFrame:
        """构造 n 只股票的行情数据，保证特征值有明显差异。"""
        rng = np.random.RandomState(42)
        frames = []
        for i in range(n_stocks):
            base_return = 0.01 * (n_stocks - i)  # 0.10, 0.09, ..., 0.01
            base_vol = 0.05 + 0.02 * i
            base_dd = -0.02 - 0.03 * i

            rets = rng.normal(base_return, base_vol, 20).tolist()
            dds = [base_dd + rng.normal(0, 0.01) for _ in range(20)]
            frames.append(_make_symbol_data(f"600{i:03d}.SH", rets, dds))

        return pd.concat(frames, ignore_index=True)

    def test_10_stocks_label_generation(self):
        """10 只股票的端到端标签生成。"""
        portfolio = self._make_portfolio(10)
        profiles = build_stock_profiles(portfolio)
        result = run_clustering(profiles)

        labels = generate_cluster_labels(result["cluster_centers"])
        assert len(labels) == 3
        for label in labels.values():
            assert label.startswith("[所选历史区间]")
            assert "收益" in label
            assert "波动" in label
            assert "回撤" in label

    def test_10_stocks_label_consistency(self):
        """10 只股票，打乱 centers 行顺序不影响标签。"""
        portfolio = self._make_portfolio(10)
        profiles = build_stock_profiles(portfolio)
        result = run_clustering(profiles)

        centers = result["cluster_centers"]
        labels_original = generate_cluster_labels(centers)

        centers_shuffled = centers.sample(frac=1, random_state=99).reset_index(drop=True)
        labels_shuffled = generate_cluster_labels(centers_shuffled)

        assert labels_original == labels_shuffled

    def test_20_stocks_label_generation(self):
        """20 只股票的端到端标签生成。"""
        portfolio = self._make_portfolio(20)
        profiles = build_stock_profiles(portfolio)
        result = run_clustering(profiles)

        labels = generate_cluster_labels(result["cluster_centers"])
        assert len(labels) == 3


# ── 输入校验 ──────────────────────────────────────────


class TestInputValidation:
    """generate_cluster_labels 的输入校验。"""

    def test_empty_dataframe(self):
        centers = pd.DataFrame(columns=["cluster", *FEATURE_COLS])
        with pytest.raises(DataValidationError, match="为空"):
            generate_cluster_labels(centers)

    def test_missing_cluster_column(self):
        centers = pd.DataFrame({
            "mean_return": [0.01],
            "volatility": [0.10],
            "max_drawdown": [-0.05],
        })
        with pytest.raises(DataValidationError, match="缺少必要列"):
            generate_cluster_labels(centers)

    def test_missing_feature_column(self):
        centers = pd.DataFrame({
            "cluster": [0],
            "mean_return": [0.01],
            "volatility": [0.10],
            # 缺少 max_drawdown
        })
        with pytest.raises(DataValidationError, match="缺少必要列"):
            generate_cluster_labels(centers)

    def test_duplicate_cluster_id(self):
        centers = _make_centers([
            (0, 0.05, 0.10, -0.05),
            (0, 0.01, 0.20, -0.30),  # 重复 cluster 0
            (2, -0.02, 0.15, -0.15),
        ])
        with pytest.raises(DataValidationError, match="重复 cluster ID"):
            generate_cluster_labels(centers)

    def test_nan_in_feature(self):
        centers = _make_centers([
            (0, 0.05, 0.10, -0.05),
            (1, float("nan"), 0.20, -0.30),
            (2, -0.02, 0.15, -0.15),
        ])
        with pytest.raises(DataValidationError, match="NaN"):
            generate_cluster_labels(centers)

    def test_inf_in_feature(self):
        centers = _make_centers([
            (0, 0.05, 0.10, -0.05),
            (1, float("inf"), 0.20, -0.30),
            (2, -0.02, 0.15, -0.15),
        ])
        with pytest.raises(DataValidationError, match="inf"):
            generate_cluster_labels(centers)

    def test_non_dataframe_input(self):
        with pytest.raises(DataValidationError, match="DataFrame"):
            generate_cluster_labels("not a dataframe")

    def test_non_numeric_cluster_column(self):
        centers = pd.DataFrame({
            "cluster": ["a", "b", "c"],
            "mean_return": [0.05, 0.01, -0.02],
            "volatility": [0.10, 0.20, 0.15],
            "max_drawdown": [-0.05, -0.30, -0.15],
        })
        with pytest.raises(DataValidationError, match="数值转换失败"):
            generate_cluster_labels(centers)


# ── build_label_interpretation ─────────────────────────


class TestBuildLabelInterpretation:
    """验证 build_label_interpretation 输出完整性。"""

    def _make_result(self) -> ClusteringResult:
        profiles = pd.DataFrame({
            "symbol": ["A", "B", "C"],
            "mean_return": [0.05, 0.01, -0.02],
            "volatility": [0.10, 0.20, 0.15],
            "max_drawdown": [-0.05, -0.30, -0.15],
            "cluster": [0, 1, 2],
        })
        centers = pd.DataFrame({
            "cluster": [0, 1, 2],
            "mean_return": [0.05, 0.01, -0.02],
            "volatility": [0.10, 0.20, 0.15],
            "max_drawdown": [-0.05, -0.30, -0.15],
        })
        return ClusteringResult(
            profiles=profiles,
            cluster_centers=centers,
            features=list(FEATURE_COLS),
            k=3,
        )

    def test_returns_all_fields(self):
        result = self._make_result()
        interp = build_label_interpretation(result)

        required_keys = {
            "cluster_label",
            "画像指标",
            "簇数量",
            "簇中心特征",
            "标签依据",
            "样本范围",
            "免责声明",
        }
        assert required_keys == set(interp.keys())

    def test_cluster_label_is_dict(self):
        result = self._make_result()
        interp = build_label_interpretation(result)

        assert isinstance(interp["cluster_label"], dict)
        assert len(interp["cluster_label"]) == 3
        for cluster_id, label in interp["cluster_label"].items():
            assert isinstance(cluster_id, int)
            assert isinstance(label, str)

    def test_cluster_centers_match(self):
        result = self._make_result()
        interp = build_label_interpretation(result)

        assert len(interp["簇中心特征"]) == 3
        for center in interp["簇中心特征"]:
            assert "cluster" in center
            assert "mean_return" in center
            assert "volatility" in center
            assert "max_drawdown" in center

    def test_with_date_range(self):
        result = self._make_result()
        interp = build_label_interpretation(
            result, date_range=("2025-01-01", "2025-06-30")
        )

        assert interp["样本范围"]["起始日期"] == "2025-01-01"
        assert interp["样本范围"]["截止日期"] == "2025-06-30"

    def test_without_date_range(self):
        result = self._make_result()
        interp = build_label_interpretation(result)

        assert "起始日期" not in interp["样本范围"]
        assert "截止日期" not in interp["样本范围"]

    def test_disclaimer_present(self):
        result = self._make_result()
        interp = build_label_interpretation(result)

        assert "不构成任何投资建议" in interp["免责声明"]
        assert "独立判断" in interp["免责声明"]

    def test_tagline_basis_mentions_relative(self):
        result = self._make_result()
        interp = build_label_interpretation(result)

        assert "相对" in interp["标签依据"]
        assert "所选历史区间" in interp["标签依据"]

    def test_identical_centers_interpretation(self):
        """完全相同的簇中心，解读应包含'相对接近'。"""
        profiles = pd.DataFrame({
            "symbol": ["A", "B", "C"],
            "mean_return": [0.01, 0.01, 0.01],
            "volatility": [0.15, 0.15, 0.15],
            "max_drawdown": [-0.10, -0.10, -0.10],
            "cluster": [0, 1, 2],
        })
        centers = pd.DataFrame({
            "cluster": [0, 1, 2],
            "mean_return": [0.01, 0.01, 0.01],
            "volatility": [0.15, 0.15, 0.15],
            "max_drawdown": [-0.10, -0.10, -0.10],
        })
        result = ClusteringResult(
            profiles=profiles,
            cluster_centers=centers,
            features=list(FEATURE_COLS),
            k=3,
        )
        interp = build_label_interpretation(result)

        for label in interp["cluster_label"].values():
            assert "相对接近" in label
