"""Clustering service — Role 5 面向 Streamlit 页面的服务层。

页面只调用这里的函数，不直接 import domain module。
"""

from __future__ import annotations

import pandas as pd

from src.contracts.clustering import ClusteringResult
from src.models.unsupervised.clustering import (
    build_stock_profiles,
    run_clustering,
)


def get_stock_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """将行情 DataFrame 聚合为每只股票一行的 Profile Table。"""
    return build_stock_profiles(df)


def run_stock_clustering(
    df: pd.DataFrame,
    n_clusters: int = 3,
    random_state: int = 42,
) -> ClusteringResult:
    """完整流程：行情数据 → Profile → 聚类 → ClusteringResult。"""
    profiles = get_stock_profiles(df)
    return run_clustering(profiles, random_state=random_state)
