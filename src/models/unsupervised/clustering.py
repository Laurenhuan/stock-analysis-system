"""K-Means 股票聚类模块 — Role 5 核心实现。

流程：
1. build_stock_profiles: 行情 DataFrame → 每只股票的3个特征（Stock Profile Table）
2. run_clustering: Profile Table → StandardScaler → KMeans(k=3) → ClusteringResult

输入 DataFrame 必须包含 Role 2 输出的公共字段：
- symbol, trade_date, return, drawdown
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

FEATURE_COLS: tuple[str, ...] = ("mean_return", "volatility", "max_drawdown")
N_CLUSTERS: int = 3
DEFAULT_RANDOM_STATE: int = 42
MIN_RETURNS_PER_STOCK: int = 2
MIN_VALID_PROFILES: int = 3


# ── 辅助函数 ──────────────────────────────────────────


def _validate_dataframe(df: pd.DataFrame, context: str) -> None:
    """校验输入是否为 DataFrame。"""
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError(
            f"{context}：输入类型应为 DataFrame，实际为 {type(df).__name__}"
        )


def _validate_numeric_column(
    df: pd.DataFrame, col: str, context: str
) -> None:
    """校验数值列：转换失败、NaN、inf 统一抛 DataValidationError。"""
    if col not in df.columns:
        raise DataValidationError(f"{context}：缺少列 '{col}'")

    try:
        values = pd.to_numeric(df[col], errors="raise")
    except (ValueError, TypeError) as e:
        raise DataValidationError(
            f"{context}：列 '{col}' 数值转换失败: {e}"
        ) from e

    if values.isna().any():
        raise DataValidationError(f"{context}：列 '{col}' 存在 NaN")

    if not np.isfinite(values).all():
        raise DataValidationError(f"{context}：列 '{col}' 存在 inf 值")


# ── 第一步：从行情数据构建 Stock Profile Table ────────


def build_stock_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """将多只股票多天的行情数据，聚合为每只股票一行的 Profile Table。

    输入 df 必须包含 Role 2 输出的公共字段：
    - symbol, trade_date, return, drawdown

    返回的 DataFrame 列：symbol, mean_return, volatility, max_drawdown

    规则：
    - mean_return：简单日收益率算术平均，不年化
    - volatility：简单日收益率样本标准差，ddof=1，不年化
    - max_drawdown：比较区间 drawdown 最小值

    异常：
    - DataValidationError：类型错误、缺少列、空 DataFrame、NaN/inf、时间区间不一致
    - InsufficientDataError：有效 return 不足 2 个、有效 Profile 不足 3 个
    """
    # ── 校验输入类型 ──
    _validate_dataframe(df, "build_stock_profiles")

    # ── 校验输入列 ──
    required = {"symbol", "trade_date", "return", "drawdown"}
    missing = required - set(df.columns)
    if missing:
        raise DataValidationError(f"输入缺少必要列: {missing}")

    if df.empty:
        raise DataValidationError("输入 DataFrame 为空")

    # ── 去掉全 NaN 行 ──
    # Role 2 的 build_common_features 输出中，第一行 return 为 NaN（没有前一天收盘价可算）
    # 这是正常现象，先去掉再做后续校验
    df = df.dropna(subset=["return", "drawdown"])

    # ── 校验数值列 ──
    # 检查 return 和 drawdown 列是否有非法值（NaN/inf）
    # 此时第一行 NaN 已被去掉，如果还有 NaN 说明数据有问题
    _validate_numeric_column(df, "return", "build_stock_profiles")
    _validate_numeric_column(df, "drawdown", "build_stock_profiles")

    # ── 校验每只股票的有效 return 数量 ──
    # 至少需要 2 个有效 return 才能计算标准差（ddof=1 需要至少 2 个数据点）
    # 这个检查放在时间区间检查之前，避免因为数据不足导致误报区间不一致
    grouped_returns = df.groupby("symbol")["return"]
    for sym, ret in grouped_returns:
        valid_count = ret.count()
        if valid_count < MIN_RETURNS_PER_STOCK:
            raise InsufficientDataError(
                f"股票 {sym} 有效 return 仅 {valid_count} 个，"
                f"不足 {MIN_RETURNS_PER_STOCK}，无法计算波动率"
            )

    # ── 校验时间区间一致性 ──
    date_range = df.groupby("symbol")["trade_date"].agg(["min", "max"])
    if date_range["min"].nunique() > 1 or date_range["max"].nunique() > 1:
        raise DataValidationError(
            "各股票比较区间不一致：start_date 或 end_date 不同"
        )

    # ── 按股票分组，聚合特征 ──
    df = df.sort_values(["symbol", "trade_date"]).copy()

    profiles = []
    for symbol, group in df.groupby("symbol"):
        returns = group["return"].dropna()

        mean_return = returns.mean()
        volatility = returns.std(ddof=1)
        max_drawdown = group["drawdown"].min()

        profiles.append(
            {
                "symbol": symbol,
                "mean_return": mean_return,
                "volatility": volatility,
                "max_drawdown": max_drawdown,
            }
        )

    result = pd.DataFrame(profiles)

    # ── 校验有效 Profile 数量 ──
    if len(result) < MIN_VALID_PROFILES:
        raise InsufficientDataError(
            f"有效 Profile 仅 {len(result)} 个，"
            f"不足 {MIN_VALID_PROFILES}，无法进行 K-Means 聚类"
        )

    # ── 校验输出质量 ──
    if result["symbol"].duplicated().any():
        raise DataValidationError("Profile 中存在重复 symbol")

    return result[["symbol", *FEATURE_COLS]]


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
        DataValidationError: 类型错误、缺少列、symbol 重复、NaN/inf 值
        InsufficientDataError: 股票数量不足 3
    """
    # ── 校验输入类型 ──
    _validate_dataframe(profiles, "run_clustering")

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
    for col in FEATURE_COLS:
        _validate_numeric_column(profiles, col, "run_clustering")

    # ── 提取特征矩阵 ──
    # 只取 3 个特征列，转为 numpy 数组供 sklearn 使用
    X = profiles[list(FEATURE_COLS)].values  # shape: (n_stocks, 3)

    # ── 标准化 ──
    # 为什么要做标准化？因为 3 个特征的量纲不同：
    # - mean_return 范围约 -0.05 ~ 0.05
    # - volatility 范围约 0 ~ 0.3
    # - max_drawdown 范围约 -0.5 ~ 0
    # 如果不标准化，数值大的特征会主导聚类结果
    # StandardScaler 把每个特征变成均值=0、标准差=1
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── K-Means 聚类 ──
    # n_init=10 表示用不同初始中心跑 10 次，取最好的结果
    # random_state=42 保证每次运行结果相同（可复现）
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # ── 中心点还原到原始尺度 ──
    # 聚类中心是在标准化后的空间计算的，需要还原到原始尺度才有实际含义
    # 例如：cluster 0 的 mean_return 中心是 0.02，表示该类股票平均日收益约 2%
    centers_scaled = kmeans.cluster_centers_
    centers_original = scaler.inverse_transform(centers_scaled)

    # ── 组装结果 ──
    result_profiles = profiles.copy()
    result_profiles["cluster"] = labels

    result_centers = pd.DataFrame(
        centers_original,
        columns=list(FEATURE_COLS),
    )
    result_centers.insert(0, "cluster", range(N_CLUSTERS))

    return ClusteringResult(
        profiles=result_profiles,
        cluster_centers=result_centers,
        features=list(FEATURE_COLS),
        k=N_CLUSTERS,
    )
