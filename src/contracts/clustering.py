"""Importable output schema for clustering Contract v0.2."""

from typing import TypedDict

from pandas import DataFrame


PROFILE_FEATURES = ("mean_return", "volatility", "max_drawdown")
PROFILE_COLUMNS = ("symbol", *PROFILE_FEATURES, "cluster")
CLUSTER_CENTER_COLUMNS = ("cluster", *PROFILE_FEATURES)
CLUSTERING_RESULT_KEYS = ("profiles", "cluster_centers", "features", "k", "cluster_label")


class ClusteringResult(TypedDict):
    """Stable top-level result returned by the P0 clustering use case."""

    profiles: DataFrame
    cluster_centers: DataFrame
    features: list[str]
    k: int
    cluster_label: dict[int, str]  # 簇编号 → 动态生成的中文标签
