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

    输入 df 必须包含的列：symbol, trade_date, close, drawdown
    （drawdown 由 Role 2 在数据清洗阶段计算好）

    返回的 DataFrame 列：symbol, mean_return, volatility, max_drawdown

    规则：
    - mean_return：简单日收益率算术平均，不年化
    - volatility：简单日收益率样本标准差，ddof=1，不年化
    - max_drawdown：比较区间 drawdown 最小值
    """
    # ── 校验输入列 ──
    required = {"symbol", "trade_date", "close", "drawdown"}
    missing = required - set(df.columns)
    if missing:
        raise DataValidationError(f"输入缺少必要列: {missing}")

    if df.empty:
        raise DataValidationError("输入 DataFrame 为空")

    # ── 校验输入数据质量 ──
    if df["symbol"].isna().any():
        raise DataValidationError("输入中存在空的 symbol")

    if df["close"].isna().any():
        raise DataValidationError("输入中存在空的 close 值")

    # ── 按股票分组，计算每日收益率 ──
    df = df.sort_values(["symbol", "trade_date"]).copy()
    df["daily_return"] = df.groupby("symbol")["close"].pct_change()

    # ── 校验每只股票的数据量 ──
    # 每只股票至少需要 2 天数据才能计算 1 个收益率
    # 计算 ddof=1 的 std 至少需要 2 个收益率（即 3 天数据）
    days_per_stock = df.groupby("symbol")["trade_date"].count()
    insufficient = days_per_stock[days_per_stock < 3]
    if len(insufficient) > 0:
        symbols = insufficient.index.tolist()
        raise DataValidationError(
            f"以下股票数据不足 3 天，无法计算波动率: {symbols}"
        )

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

    1. 校验输入数据质量
    2. 用 StandardScaler 标准化3个特征（私有，不改原始值）
    3. 用 KMeans(n_clusters=3) 聚类
    4. 用 inverse_transform 把中心点还原到原始尺度
    5. 组装成 ClusteringResult 返回

    参数：
        profiles: 包含 symbol 和三个特征列的 DataFrame
        random_state: 随机种子，默认 42，相同输入和参数可复现

    返回：
        ClusteringResult: 包含 profiles, cluster_centers, features, k

    异常：
        DataValidationError: 缺少列、symbol 重复、NaN/inf 值
        InsufficientDataError: 股票数量不足 3
    """
    # ── 校验输入列 ──
    missing = {"symbol", *FEATURE_COLS} - set(profiles.columns)
    if missing:
        raise DataValidationError(f"Profile 缺少必要列: {missing}")

    # ── 校验 symbol 唯一性 ──
    if profiles["symbol"].duplicated().any():
        dupes = profiles[profiles["symbol"].duplicated()]["symbol"].tolist()
        raise DataValidationError(f"Profile 中存在重复 symbol: {dupes}")

    # ── 校验股票数量 ──
    if len(profiles) < N_CLUSTERS:
        raise InsufficientDataError(
            f"股票数量 {len(profiles)} 不足 {N_CLUSTERS}，无法聚类"
        )

    # ── 校验特征值质量 ──
    nan_cols = [c for c in FEATURE_COLS if profiles[c].isna().any()]
    if nan_cols:
        raise DataValidationError(f"特征列存在 NaN: {nan_cols}")

    non_finite = [
        c for c in FEATURE_COLS if not np.isfinite(profiles[c]).all()
    ]
    if non_finite:
        raise DataValidationError(f"特征列存在非有限值: {non_finite}")

    # ── 提取特征矩阵 ──
    X = profiles[FEATURE_COLS].values  # shape: (n_stocks, 3)

    # ── 标准化 ──
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── K-Means 聚类 ──
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # ── 中心点还原到原始尺度 ──
    centers_scaled = kmeans.cluster_centers_
    centers_original = scaler.inverse_transform(centers_scaled)

    # ── 组装结果 ──
    result_profiles = profiles.copy()
    result_profiles["cluster"] = labels

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
