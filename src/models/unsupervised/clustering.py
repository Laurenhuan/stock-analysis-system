"""K-Means 股票聚类模块 — Role 5 核心实现。

流程：
1. build_stock_profiles: 行情 DataFrame → 每只股票的3个特征（Stock Profile Table）
2. run_clustering: Profile Table → StandardScaler → KMeans(k=3) → ClusteringResult
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.contracts.clustering import (
    CLUSTER_CENTER_COLUMNS,
    CLUSTERING_RESULT_KEYS,
    ClusteringResult,
)
from src.utils.exceptions import DataValidationError, InsufficientDataError

# ── 常量 ──────────────────────────────────────────────

FEATURE_COLS: list[str] = ["mean_return", "volatility", "max_drawdown"]
N_CLUSTERS: int = 3
DEFAULT_RANDOM_STATE: int = 42


# ── 第一步：从行情数据构建 Stock Profile Table ────────


def build_stock_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """将多只股票多天的行情数据，聚合为每只股票一行的 Profile Table。

    输入 df 必须包含的列：symbol, close, drawdown
    （drawdown 由 Role 2 在数据清洗阶段计算好）

    返回的 DataFrame 列：symbol, mean_return, volatility, max_drawdown
    """
    # ── 校验输入 ──
    required = {"symbol", "close", "drawdown"}
    missing = required - set(df.columns)
    if missing:
        raise DataValidationError(f"输入缺少必要列: {missing}")

    if df.empty:
        raise DataValidationError("输入 DataFrame 为空")

    # ── 按股票分组，计算每日收益率 ──
    # 收益率 = (今天收盘价 - 昨天收盘价) / 昨天收盘价
    # groupby + pct_change 会自动按每只股票独立计算
    df = df.sort_values(["symbol", "trade_date"]).copy()
    df["daily_return"] = df.groupby("symbol")["close"].pct_change()

    # 第一天没有"昨天"，收益率为 NaN，丢弃
    df = df.dropna(subset=["daily_return"])

    # ── 按 symbol 聚合出3个特征 ──
    profiles = (
        df.groupby("symbol")
        .agg(
            mean_return=("daily_return", "mean"),       # 平均收益率
            volatility=("daily_return", "std"),          # 收益率标准差（ddof=1）
            max_drawdown=("drawdown", "min"),            # 最大回撤（取最小值）
        )
        .reset_index()
    )

    # ── 校验输出质量 ──
    if profiles["symbol"].duplicated().any():
        raise DataValidationError("Profile 中存在重复 symbol")

    nan_cols = [c for c in FEATURE_COLS if profiles[c].isna().any()]
    if nan_cols:
        raise DataValidationError(f"特征列存在 NaN: {nan_cols}")

    non_finite = [
        c
        for c in FEATURE_COLS
        if not np.isfinite(profiles[c]).all()
    ]
    if non_finite:
        raise DataValidationError(f"特征列存在非有限值: {non_finite}")

    return profiles[["symbol", *FEATURE_COLS]]


# ── 第二步：StandardScaler + KMeans 聚类 ─────────────


def run_clustering(
    profiles: pd.DataFrame,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> ClusteringResult:
    """对 Stock Profile Table 执行 K-Means 聚类，返回 Contract 约定的结果。

    1. 用 StandardScaler 标准化3个特征（私有，不改原始值）
    2. 用 KMeans(n_clusters=3) 聚类
    3. 用 inverse_transform 把中心点还原到原始尺度
    4. 组装成 ClusteringResult 返回
    """
    # ── 校验输入 ──
    missing = {"symbol", *FEATURE_COLS} - set(profiles.columns)
    if missing:
        raise DataValidationError(f"Profile 缺少必要列: {missing}")

    if len(profiles) < N_CLUSTERS:
        raise InsufficientDataError(
            f"股票数量 {len(profiles)} 不足 {N_CLUSTERS}，无法聚类"
        )

    # ── 提取特征矩阵 ──
    X = profiles[FEATURE_COLS].values  # shape: (n_stocks, 3)

    # ── 标准化 ──
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── K-Means 聚类 ──
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # ── 中心点还原到原始尺度 ──
    # kmeans.cluster_centers_ 是标准化空间的中心
    # scaler.inverse_transform 还原回 mean_return/volatility/max_drawdown 的真实数值
    centers_scaled = kmeans.cluster_centers_
    centers_original = scaler.inverse_transform(centers_scaled)

    # ── 组装结果 ──

    # profiles: 原始尺度 + cluster 列
    result_profiles = profiles.copy()
    result_profiles["cluster"] = labels

    # cluster_centers: 还原到原始尺度的中心点
    result_centers = pd.DataFrame(
        centers_original,
        columns=FEATURE_COLS,
    )
    result_centers.insert(0, "cluster", range(N_CLUSTERS))

    return ClusteringResult(
        profiles=result_profiles,
        cluster_centers=result_centers,
        features=FEATURE_COLS,
        k=N_CLUSTERS,
    )
