"""Application-facing orchestration for the fixed P0 K-Means workflow."""

from __future__ import annotations

import pandas as pd

from src.contracts.clustering import ClusteringResult
from src.models.unsupervised.clustering import build_stock_profiles, run_clustering


def get_stock_profiles(data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate Contract-compliant market data into stock profiles."""
    return build_stock_profiles(data)


def run_stock_clustering(
    data: pd.DataFrame,
    *,
    random_state: int = 42,
) -> ClusteringResult:
    """Run the P0 workflow with the Contract-fixed ``KMeans(k=3)``."""
    profiles = get_stock_profiles(data)
    return run_clustering(profiles, random_state=random_state)
