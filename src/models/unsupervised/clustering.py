"""K-Means 股票聚类模块 — Role 5 核心实现。

流程：
1. build_stock_profiles: 行情 DataFrame → 每只股票的3个特征（Stock Profile Table）
2. run_clustering: Profile Table → StandardScaler → KMeans(k=3) → ClusteringResult

输入 DataFrame 必须包含 Role 2 输出的公共字段：
- symbol, trade_date, return, drawdown
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
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
    X = profiles[list(FEATURE_COLS)].to_numpy(dtype=float)
    distinct_profiles = np.unique(np.round(X, decimals=12), axis=0)
    if len(distinct_profiles) < N_CLUSTERS:
        raise InsufficientDataError(
            f"股票数量为 {len(profiles)}，但仅有 {len(distinct_profiles)} 个"
            f"可区分画像，不足以形成 {N_CLUSTERS} 个簇"
        )

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
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            labels = kmeans.fit_predict(X_scaled)
    except ConvergenceWarning as exc:
        raise InsufficientDataError(
            f"有效股票画像无法稳定形成 {N_CLUSTERS} 个不同簇"
        ) from exc
    if len(np.unique(labels)) != N_CLUSTERS:
        raise InsufficientDataError(
            f"聚类结果仅形成 {len(np.unique(labels))} 个有效簇，"
            f"少于固定要求的 {N_CLUSTERS} 个"
        )

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


# ── 第三步：动态簇标签生成 ─────────────────────────────

# 特征值等级 → 中文描述的映射
_LABEL_MAP: dict[str, dict[str, str]] = {
    "mean_return": {
        "high": "相对高收益",
        "mid": "相对中等收益",
        "low": "相对低收益",
        "close": "相对接近收益",
    },
    "volatility": {
        "high": "相对高波动",
        "mid": "相对中等波动",
        "low": "相对低波动",
        "close": "相对接近波动",
    },
    "max_drawdown": {
        "high": "相对小回撤",
        "mid": "相对中等回撤",
        "low": "相对大回撤",
        "close": "相对接近回撤",
    },
}

# 当所有簇的某个特征值几乎相同时的阈值
# 使用特征值范围的 1% 作为判断"接近"的阈值
_CLOSE_THRESHOLD_RATIO: float = 0.01


def _validate_cluster_centers(df: pd.DataFrame) -> None:
    """校验 cluster_centers 输入数据质量。"""
    _validate_dataframe(df, "generate_cluster_labels")

    if df.empty:
        raise DataValidationError("generate_cluster_labels：输入 DataFrame 为空")

    required = {"cluster", *FEATURE_COLS}
    missing = required - set(df.columns)
    if missing:
        raise DataValidationError(
            f"generate_cluster_labels：缺少必要列: {missing}"
        )

    if df["cluster"].duplicated().any():
        dupes = df[df["cluster"].duplicated()]["cluster"].tolist()
        raise DataValidationError(
            f"generate_cluster_labels：存在重复 cluster ID: {dupes}"
        )

    for col in FEATURE_COLS:
        _validate_numeric_column(df, col, "generate_cluster_labels")

    # 校验 cluster 列为数值型
    try:
        pd.to_numeric(df["cluster"], errors="raise")
    except (ValueError, TypeError) as e:
        raise DataValidationError(
            f"generate_cluster_labels：列 'cluster' 数值转换失败: {e}"
        ) from e


def _rank_with_ties(values: np.ndarray) -> np.ndarray:
    """对数组进行排名，相同值获得相同 rank。

    rank 0 = 最高值，rank k-1 = 最低值。
    使用 pandas rank(method='min')：并列值获得相同 rank（取最小位置）。

    返回长度为 len(values) 的整数 rank 数组。
    """
    # pandas rank 默认 ascending=True（值越小 rank 越小）
    # 我们需要降序排列（值越大 rank 越小），所以 ascending=False
    # 然后转换为 0-based index
    series = pd.Series(values)
    # rank(ascending=False): 最高值 rank=1, 次高 rank=2, ...
    # 减 1 转为 0-based: 最高值 rank=0, 次高 rank=1, ...
    ranks = (series.rank(method="min", ascending=False) - 1).astype(int).values
    return ranks


def generate_cluster_labels(
    cluster_centers: pd.DataFrame,
) -> dict[int, str]:
    """根据簇中心特征的相对排序，动态生成中文簇标签。

    标签格式：[所选历史区间]相对X收益-相对Y波动-相对Z回撤型

    排序逻辑：
    - 按每个特征值从高到低排名
    - rank 0 = 最高/最好，rank k-1 = 最低/最差
    - 如果所有簇的某个特征值几乎相同（范围 < 阈值），统一标为"相对接近"
    - 并列值获得相同等级描述

    参数：
        cluster_centers: 包含 cluster 列和 FEATURE_COLS 的 DataFrame

    返回：
        dict[int, str]: 簇编号 → 中文标签

    异常：
        DataValidationError: 空表、缺列、NaN/inf、重复 cluster ID、非数值指标
    """
    _validate_cluster_centers(cluster_centers)

    k = len(cluster_centers)
    labels: dict[int, str] = {}

    # 复制一份，避免修改原始传入的 DataFrame
    centers = cluster_centers.copy()

    # 为每个特征计算 rank
    for col in FEATURE_COLS:
        values = centers[col].values.astype(float)
        ranks = _rank_with_ties(values)
        centers[f"_rank_{col}"] = ranks

        # 计算特征值范围，判断是否所有值都接近
        val_range = values.max() - values.min()
        centers[f"_range_{col}"] = val_range

    # 为每个簇生成复合标签
    for _, row in centers.iterrows():
        cluster_id = int(row["cluster"])

        renditions = []
        for col in FEATURE_COLS:
            val_range = row[f"_range_{col}"]
            values = centers[col].values.astype(float)
            threshold = max(abs(values.max()), abs(values.min()), 1e-10) * _CLOSE_THRESHOLD_RATIO

            # 所有值几乎相同 → 标为"相对接近"
            if val_range <= threshold:
                renditions.append(_LABEL_MAP[col]["close"])
            else:
                # 使用实际极值判断，而非 rank 位置
                # 这样并列值（无论并列最高还是最低）都能得到正确标签
                current_val = float(row[col])
                if current_val == values.max():
                    renditions.append(_LABEL_MAP[col]["high"])
                elif current_val == values.min():
                    renditions.append(_LABEL_MAP[col]["low"])
                else:
                    renditions.append(_LABEL_MAP[col]["mid"])

        compound = "-".join(renditions)
        labels[cluster_id] = f"[所选历史区间]{compound}型"

    return labels


# ── 第四步：标签解读输出 ──────────────────────────────


def build_label_interpretation(
    clustering_result: ClusteringResult,
    date_range: tuple[str, str] | None = None,
) -> dict:
    """组装完整的标签解读信息，供 Role 1 渲染展示。

    直接根据 cluster_centers 调用 generate_cluster_labels() 生成标签。
    cluster_label 不在 ClusteringResult Contract 中，而是由此函数动态生成。

    返回字典包含以下字段：
    - cluster_label: dict[int, str] — 簇编号 → 中文标签
    - 画像指标: list[str] — 聚类使用的特征列名
    - 簇数量: int — 聚类簇数 k
    - 簇中心特征: list[dict] — 每个簇的中心特征值
    - 标签依据: str — 标签生成逻辑说明
    - 样本范围: dict — 聚类涉及的股票数量及可选日期范围
    - 免责声明: str — 免责声明文本

    参数：
        clustering_result: run_clustering 的返回结果
        date_range: 可选的 (start_date, end_date) 元组，用于标注样本时间范围
    """
    centers = clustering_result["cluster_centers"]
    k = clustering_result["k"]

    # 直接从 cluster_centers 生成标签
    cluster_label = generate_cluster_labels(centers)

    # 构建簇中心特征列表
    center_features = []
    for _, row in centers.iterrows():
        center_features.append({
            "cluster": int(row["cluster"]),
            "mean_return": float(row["mean_return"]),
            "volatility": float(row["volatility"]),
            "max_drawdown": float(row["max_drawdown"]),
        })

    # 标签依据说明
    tagline_basis = (
        "标签基于聚类中心特征的相对排序动态生成："
        "比较各簇的平均收益率、波动率、最大回撤的中心值，"
        "按从高到低排序后赋予对应中文描述。"
        "标签中的'相对'表示仅针对当前数据集内各簇的横向比较，"
        "'所选历史区间'表示特征值依赖于输入数据的时间范围。"
        "当所有簇的某个特征值几乎相同时，标为'相对接近'。"
    )

    # 样本范围
    n_stocks = len(clustering_result["profiles"])
    sample_scope: dict = {"股票数量": n_stocks}
    if date_range is not None:
        sample_scope["起始日期"] = date_range[0]
        sample_scope["截止日期"] = date_range[1]

    # 免责声明
    disclaimer = (
        "本标签仅基于历史数据的统计特征生成，不构成任何投资建议。"
        "股票的聚类标签可能随输入数据的时间区间不同而变化，"
        "投资者应独立判断并承担投资风险。"
    )

    return {
        "cluster_label": cluster_label,
        "画像指标": list(clustering_result["features"]),
        "簇数量": k,
        "簇中心特征": center_features,
        "标签依据": tagline_basis,
        "样本范围": sample_scope,
        "免责声明": disclaimer,
    }
