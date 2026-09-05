"""Application-facing orchestration for the fixed P0 K-Means workflow."""

from __future__ import annotations

from typing import TypedDict

import pandas as pd

from src.contracts.clustering import ClusteringResult
from src.models.unsupervised.clustering import (
    build_label_interpretation,
    build_stock_profiles,
    run_clustering,
)


class ClusteringDashboard(TypedDict):
    """Presentation bundle that leaves the public clustering Contract intact."""

    result: ClusteringResult
    interpretation: dict[str, object]


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


def run_stock_clustering_dashboard(
    data: pd.DataFrame,
    *,
    random_state: int = 42,
) -> ClusteringDashboard:
    """Run clustering and attach Role 5's dynamic historical interpretation."""
    result = run_stock_clustering(data, random_state=random_state)
    dates = pd.to_datetime(data["trade_date"], errors="coerce").dropna()
    date_range = None
    if not dates.empty:
        date_range = (
            dates.min().date().isoformat(),
            dates.max().date().isoformat(),
        )
    return ClusteringDashboard(
        result=result,
        interpretation=build_label_interpretation(result, date_range),
    )
