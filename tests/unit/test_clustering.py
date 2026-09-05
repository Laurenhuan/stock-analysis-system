"""Role 5 聚类模块单元测试。

覆盖场景：
1. build_stock_profiles — 使用 Role 2 的 return 列、数据质量校验
2. run_clustering — StandardScaler、KMeans k=3、输出结构
3. 异常场景 — 类型错误、时间区间不一致、有效收益不足
4. 端到端 — 从行情到 ClusteringResult 的完整链路
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.contracts.clustering import (
    CLUSTER_CENTER_COLUMNS,
    CLUSTERING_RESULT_KEYS,
    PROFILE_COLUMNS,
)
from src.models.unsupervised.clustering import (
    DEFAULT_RANDOM_STATE,
    FEATURE_COLS,
    N_CLUSTERS,
    build_stock_profiles,
    run_clustering,
)
from src.utils.exceptions import DataValidationError, InsufficientDataError


# ── 辅助函数：生成测试用的行情数据（使用 Role 2 的 return 列）──


def _make_market_data(
    symbols: list[str] | None = None,
    n_days: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """生成模拟行情 DataFrame，包含 Role 2 的公共字段。

    输出列：symbol, trade_date, close, return, drawdown
    """
    if symbols is None:
        symbols = ["000001", "000002", "000003", "000004", "000005"]

    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)

    rows = []
    for sym in symbols:
        start_price = rng.uniform(10, 100)
        returns = rng.normal(0.001, 0.02, size=n_days)
        prices = start_price * np.cumprod(1 + returns)

        # 计算 return（Role 2 的公共字段）
        stock_returns = np.concatenate([[np.nan], prices[1:] / prices[:-1] - 1])

        # 计算 drawdown
        running_max = np.maximum.accumulate(prices)
        drawdowns = (prices - running_max) / running_max

        for i, date in enumerate(dates):
            rows.append(
                {
                    "symbol": sym,
                    "trade_date": date,
                    "close": round(prices[i], 2),
                    "return": round(stock_returns[i], 6) if not np.isnan(stock_returns[i]) else np.nan,
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
        """特征列必须与 FEATURE_COLS 一致。"""
        df = _make_market_data()
        profiles = build_stock_profiles(df)

        assert tuple(profiles.columns[1:]) == FEATURE_COLS

    def test_uses_role2_return_column(self) -> None:
        """必须使用 Role 2 的 return 列，不重新计算收益率。"""
        # 构造3只股票（满足 MIN_VALID_PROFILES），使用明确的 return 值
        df = pd.DataFrame(
            {
                "symbol": ["X", "X", "X", "Y", "Y", "Y", "Z", "Z", "Z"],
                "trade_date": pd.to_datetime(
                    ["2024-01-01", "2024-01-02", "2024-01-03"] * 3
                ),
                "close": [100.0, 105.0, 108.0, 200.0, 210.0, 215.0, 50.0, 48.0, 52.0],
                "return": [np.nan, 0.05, 0.028571, np.nan, 0.05, 0.023810, np.nan, -0.04, 0.083333],
                "drawdown": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            }
        )
        profiles = build_stock_profiles(df)

        # mean_return 应该是 return 列的均值（排除 NaN）
        expected_mean = (0.05 + 0.028571) / 2
        assert abs(profiles.loc[0, "mean_return"] - expected_mean) < 1e-4

    def test_volatility_uses_ddof1(self) -> None:
        """volatility 使用 ddof=1 的样本标准差。"""
        df = _make_market_data(symbols=["A", "B", "C"], n_days=10)
        profiles = build_stock_profiles(df)

        for _, row in profiles.iterrows():
            sym = row["symbol"]
            returns = df[df["symbol"] == sym]["return"].dropna()
            expected_std = returns.std(ddof=1)
            assert abs(row["volatility"] - expected_std) < 1e-10

    def test_max_drawdown_is_min_of_drawdown(self) -> None:
        """max_drawdown 是 drawdown 列的最小值。"""
        df = pd.DataFrame(
            {
                "symbol": ["X", "X", "X", "Y", "Y", "Y", "Z", "Z", "Z"],
                "trade_date": pd.to_datetime(
                    ["2024-01-01", "2024-01-02", "2024-01-03"] * 3
                ),
                "close": [100.0, 90.0, 95.0, 200.0, 190.0, 195.0, 50.0, 45.0, 48.0],
                "return": [np.nan, -0.1, 0.055556, np.nan, -0.05, 0.026316, np.nan, -0.1, 0.066667],
                "drawdown": [0.0, -0.1, -0.05, 0.0, -0.05, -0.025, 0.0, -0.1, -0.04],
            }
        )
        profiles = build_stock_profiles(df)
        x_row = profiles[profiles["symbol"] == "X"].iloc[0]
        assert x_row["max_drawdown"] == -0.1

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


# ── 异常测试：类型错误 ────────────────────────────────


class TestTypeValidation:
    """类型错误测试。"""

    def test_non_dataframe_raises_error(self) -> None:
        """非 DataFrame 输入应抛出 DataValidationError。"""
        with pytest.raises(DataValidationError, match="应为 DataFrame"):
            build_stock_profiles("not a dataframe")

    def test_list_input_raises_error(self) -> None:
        """列表输入应抛出 DataValidationError。"""
        with pytest.raises(DataValidationError, match="应为 DataFrame"):
            build_stock_profiles([{"symbol": "A"}])

    def test_none_input_raises_error(self) -> None:
        """None 输入应抛出 DataValidationError。"""
        with pytest.raises(DataValidationError, match="应为 DataFrame"):
            build_stock_profiles(None)


# ── 异常测试：缺少列 ──────────────────────────────────


class TestMissingColumns:
    """缺少列测试。"""

    def test_missing_symbol_column_raises_error(self) -> None:
        df = pd.DataFrame({"trade_date": [1], "return": [0.1], "drawdown": [-0.05]})
        with pytest.raises(DataValidationError, match="缺少必要列"):
            build_stock_profiles(df)

    def test_missing_return_column_raises_error(self) -> None:
        df = pd.DataFrame({"symbol": ["A"], "trade_date": [1], "drawdown": [-0.05]})
        with pytest.raises(DataValidationError, match="缺少必要列"):
            build_stock_profiles(df)

    def test_missing_drawdown_column_raises_error(self) -> None:
        df = pd.DataFrame({"symbol": ["A"], "trade_date": [1], "return": [0.1]})
        with pytest.raises(DataValidationError, match="缺少必要列"):
            build_stock_profiles(df)


# ── 异常测试：空数据 ──────────────────────────────────


class TestEmptyData:
    """空数据测试。"""

    def test_empty_dataframe_raises_error(self) -> None:
        df = pd.DataFrame(columns=["symbol", "trade_date", "return", "drawdown"])
        with pytest.raises(DataValidationError, match="为空"):
            build_stock_profiles(df)


# ── 异常测试：时间区间不一致 ───────────────────────────


class TestInconsistentDateRange:
    """时间区间不一致测试。"""

    def test_different_start_date_raises_error(self) -> None:
        """不同股票起始日期不同应报错。"""
        # A: 01-02~01-03 (2个有效return), B: 01-03~01-04 (2个有效return)
        # start_date 不同
        df = pd.DataFrame(
            {
                "symbol": ["A", "A", "A", "B", "B", "B"],
                "trade_date": pd.to_datetime(
                    ["2024-01-02", "2024-01-03", "2024-01-04",
                     "2024-01-03", "2024-01-04", "2024-01-05"]
                ),
                "close": [100.0, 105.0, 108.0, 200.0, 210.0, 215.0],
                "return": [np.nan, 0.05, 0.028571, np.nan, 0.05, 0.023810],
                "drawdown": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            }
        )
        with pytest.raises(DataValidationError, match="比较区间不一致"):
            build_stock_profiles(df)

    def test_different_end_date_raises_error(self) -> None:
        """不同股票结束日期不同应报错。"""
        # A: 01-02~01-04 (2个有效return), B: 01-03~01-04 (2个有效return)
        # dropna后 A end=01-04, B end=01-04 → 相同，不行
        # 改为：A end=01-05, B end=01-04
        df = pd.DataFrame(
            {
                "symbol": ["A", "A", "A", "A", "B", "B", "B"],
                "trade_date": pd.to_datetime(
                    ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
                     "2024-01-02", "2024-01-03", "2024-01-04"]
                ),
                "close": [100.0, 105.0, 108.0, 112.0, 200.0, 210.0, 215.0],
                "return": [np.nan, 0.05, 0.028571, 0.037037, np.nan, 0.05, 0.023810],
                "drawdown": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            }
        )
        with pytest.raises(DataValidationError, match="比较区间不一致"):
            build_stock_profiles(df)


# ── 异常测试：有效 return 不足 ────────────────────────


class TestInsufficientReturns:
    """有效 return 不足测试。"""

    def test_too_few_returns_per_stock_raises_error(self) -> None:
        """单只股票有效 return 少于 2 个应报错。"""
        # 所有股票时间区间一致，但 A 只有 1 个有效 return
        df = pd.DataFrame(
            {
                "symbol": ["A", "A", "A", "B", "B", "B", "C", "C", "C", "D", "D", "D", "E", "E", "E"],
                "trade_date": pd.to_datetime(
                    ["2024-01-01", "2024-01-02", "2024-01-03"] * 5
                ),
                "close": [100.0, 105.0, 110.0] * 5,
                # A: 第1行有效, 第2行NaN, 第3行有效 → 但第2行NaN被dropna去掉后只剩2行
                # 改为：A只有第1行有效，第2、3行都是NaN
                # B~E: 第1、2、3行都有效
                "return": [0.01, np.nan, np.nan,
                           0.02, 0.03, 0.04,
                           0.02, 0.03, 0.04,
                           0.02, 0.03, 0.04,
                           0.02, 0.03, 0.04],
                "drawdown": [0.0, 0.0, 0.0] * 5,
            }
        )
        with pytest.raises(InsufficientDataError, match="有效 return 仅 1 个"):
            build_stock_profiles(df)


# ── 异常测试：有效 Profile 不足 ───────────────────────


class TestInsufficientProfiles:
    """有效 Profile 不足测试。"""

    def test_too_few_valid_profiles_raises_error(self) -> None:
        """有效 Profile 少于 3 个应报错。"""
        df = _make_market_data(symbols=["A", "B"], n_days=30)
        with pytest.raises(InsufficientDataError, match="有效 Profile 仅 2 个"):
            build_stock_profiles(df)


# ── 异常测试：数值列质量问题 ──────────────────────────


class TestNumericColumnQuality:
    """数值列质量问题测试。"""

    def test_nan_in_return_raises_error(self) -> None:
        """return 列存在 NaN 应报错（非首行）。"""
        df = pd.DataFrame(
            {
                "symbol": ["X", "X", "X"],
                "trade_date": pd.to_datetime(
                    ["2024-01-01", "2024-01-02", "2024-01-03"]
                ),
                "close": [100.0, 105.0, 103.0],
                "return": [np.nan, 0.05, np.nan],
                "drawdown": [0.0, 0.0, 0.0],
            }
        )
        # 只有 1 个有效 return，不足 2 个
        with pytest.raises(InsufficientDataError):
            build_stock_profiles(df)

    def test_inf_in_return_raises_error(self) -> None:
        """return 列存在 inf 应报错。"""
        df = pd.DataFrame(
            {
                "symbol": ["X", "X", "X"],
                "trade_date": pd.to_datetime(
                    ["2024-01-01", "2024-01-02", "2024-01-03"]
                ),
                "close": [100.0, 105.0, 103.0],
                "return": [np.nan, np.inf, -0.019],
                "drawdown": [0.0, 0.0, 0.0],
            }
        )
        with pytest.raises(DataValidationError, match="inf"):
            build_stock_profiles(df)


# ── 输入不可变测试 ────────────────────────────────────


class TestInputImmutability:
    """输入不可变测试。"""

    def test_input_not_modified(self) -> None:
        """build_stock_profiles 不应修改输入 DataFrame。"""
        df = _make_market_data(symbols=["A", "B", "C"])
        original_columns = list(df.columns)
        original_length = len(df)

        build_stock_profiles(df)

        assert list(df.columns) == original_columns
        assert len(df) == original_length


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
        assert "cluster" in result["profiles"].columns
        assert list(result["profiles"].columns) == list(PROFILE_COLUMNS)

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
        assert result["features"] == list(FEATURE_COLS)
        assert result["k"] == N_CLUSTERS

    def test_features_is_list_not_tuple(self, sample_profiles: pd.DataFrame) -> None:
        """features 必须是 list，不是 tuple 或其他类型。"""
        result = run_clustering(sample_profiles)
        assert isinstance(result["features"], list)

    def test_reproducibility_with_same_seed(self, sample_profiles: pd.DataFrame) -> None:
        """相同 random_state 应产生完全相同的结果。"""
        result1 = run_clustering(sample_profiles, random_state=123)
        result2 = run_clustering(sample_profiles, random_state=123)

        pd.testing.assert_frame_equal(result1["profiles"], result2["profiles"])
        pd.testing.assert_frame_equal(
            result1["cluster_centers"], result2["cluster_centers"]
        )

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
        """重复 symbol 应抛出 DataValidationError。"""
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

    def test_non_dataframe_raises_error(self) -> None:
        """非 DataFrame 输入应抛出 DataValidationError。"""
        with pytest.raises(DataValidationError, match="应为 DataFrame"):
            run_clustering("not a dataframe")

    def test_cluster_centers_are_original_scale(self, sample_profiles: pd.DataFrame) -> None:
        """聚类中心应通过 inverse_transform 还原到原始尺度。"""
        result = run_clustering(sample_profiles)

        # 手动验证：中心点应等于对应簇成员的均值
        profiles = result["profiles"]
        centers = result["cluster_centers"]

        for c in range(N_CLUSTERS):
            cluster_members = profiles[profiles["cluster"] == c]
            if len(cluster_members) > 0:
                for col in FEATURE_COLS:
                    expected = cluster_members[col].mean()
                    actual = centers.loc[c, col]
                    assert abs(actual - expected) < 1e-10, (
                        f"Cluster {c} 的 {col} 中心点不等于簇成员均值"
                    )


# ── 端到端测试 ────────────────────────────────────────


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
        assert result["features"] == list(FEATURE_COLS)
        assert result["k"] == N_CLUSTERS

        assert result["profiles"]["cluster"].notna().all()
        assert result["profiles"]["cluster"].isin(range(N_CLUSTERS)).all()

    def test_all_stocks_assigned_to_a_cluster(self) -> None:
        """每只股票都必须被分配到一个 cluster。"""
        market_df = _make_market_data(symbols=["A", "B", "C", "D", "E"])
        profiles = build_stock_profiles(market_df)
        result = run_clustering(profiles)

        assert result["profiles"]["cluster"].notna().all()
        assert result["profiles"]["cluster"].isin(range(N_CLUSTERS)).all()

    def test_with_role2_build_common_features(self) -> None:
        """真实端到端：Role 2 build_common_features → build_stock_profiles → run_clustering。"""
        from pathlib import Path

        from src.data.features import build_common_features

        # 读取 Role 2 提供的样例数据
        sample_path = Path("data/sample/sample_daily.csv")
        assert sample_path.exists(), "共享样例 data/sample/sample_daily.csv 不存在"

        raw = pd.read_csv(sample_path)
        # build_common_features 需要 symbol, trade_date, open, high, low, close, volume
        featured = build_common_features(raw)

        # 验证 Role 2 输出了 return 和 drawdown
        assert "return" in featured.columns
        assert "drawdown" in featured.columns

        # 传给 Role 5 的 build_stock_profiles
        profiles = build_stock_profiles(featured)

        # Profile 应覆盖 Role 2 当前共享样例中的全部股票，不写死数量。
        expected_symbols = set(featured["symbol"].dropna().unique())
        assert len(profiles) == len(expected_symbols)
        assert set(profiles["symbol"]) == expected_symbols
        assert profiles["symbol"].is_unique
        assert tuple(profiles.columns[1:]) == FEATURE_COLS

        # 聚类
        result = run_clustering(profiles)
        assert set(result.keys()) == set(CLUSTERING_RESULT_KEYS)
        assert len(result["profiles"]) == len(expected_symbols)
        assert set(result["profiles"]["symbol"]) == expected_symbols
        assert result["profiles"]["cluster"].notna().all()
        assert result["profiles"]["cluster"].isin(range(N_CLUSTERS)).all()


# ── 10 只股票测试 ─────────────────────────────────────


class TestTenStocks:
    """10 只股票聚类测试 — 符合 Role 2 共享数据要求。"""

    def test_ten_stocks_all_clustered(self) -> None:
        """10 只股票必须全部进入聚类，输出包含 symbol 和 cluster。"""
        symbols = [
            "000001.SZ", "000002.SZ", "000858.SZ", "002415.SZ", "300750.SZ",
            "600519.SH", "600036.SH", "601318.SH", "601888.SH", "601088.SH",
        ]
        market_df = _make_market_data(symbols=symbols, n_days=250)
        profiles = build_stock_profiles(market_df)

        # 恰好 10 行，一行一只股票
        assert len(profiles) == 10
        assert profiles["symbol"].is_unique

        result = run_clustering(profiles)

        # 10 只股票全部进入聚类
        assert len(result["profiles"]) == 10
        assert result["profiles"]["cluster"].notna().all()
        assert result["profiles"]["cluster"].isin(range(N_CLUSTERS)).all()

        # 输出包含 symbol 和 cluster
        assert "symbol" in result["profiles"].columns
        assert "cluster" in result["profiles"].columns

    def test_ten_stocks_reproducible(self) -> None:
        """10 只股票，相同 random_state 结果完全一致。"""
        symbols = [f"STK{i:04d}" for i in range(10)]
        market_df = _make_market_data(symbols=symbols, n_days=250)
        profiles = build_stock_profiles(market_df)

        result1 = run_clustering(profiles, random_state=42)
        result2 = run_clustering(profiles, random_state=42)

        pd.testing.assert_frame_equal(result1["profiles"], result2["profiles"])
        pd.testing.assert_frame_equal(
            result1["cluster_centers"], result2["cluster_centers"]
        )


class TestDegenerateClustering:
    def test_identical_profiles_raise_clear_error(self) -> None:
        profiles = pd.DataFrame(
            {
                "symbol": ["A", "B", "C"],
                "mean_return": [0.01, 0.01, 0.01],
                "volatility": [0.02, 0.02, 0.02],
                "max_drawdown": [-0.1, -0.1, -0.1],
            }
        )

        with pytest.raises(InsufficientDataError, match="可区分画像"):
            run_clustering(profiles)

    def test_only_two_distinct_profiles_cannot_form_three_clusters(self) -> None:
        profiles = pd.DataFrame(
            {
                "symbol": ["A", "B", "C", "D"],
                "mean_return": [0.01, 0.01, 0.03, 0.03],
                "volatility": [0.02, 0.02, 0.04, 0.04],
                "max_drawdown": [-0.1, -0.1, -0.2, -0.2],
            }
        )

        with pytest.raises(InsufficientDataError, match="不足以形成 3 个簇"):
            run_clustering(profiles)
