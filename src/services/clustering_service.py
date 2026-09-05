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
from src.utils.exceptions import DataValidationError


class ClusteringDashboard(TypedDict):
    """Presentation bundle that leaves the public clustering Contract intact."""

    result: ClusteringResult
    interpretation: dict[str, object]


class ClusteringDateDiagnostics(TypedDict):
    """Comparable date coverage used for actionable page guidance."""

    is_consistent: bool
    common_start: str | None
    common_end: str | None
    ranges: pd.DataFrame
    limiting_ranges: pd.DataFrame


def get_clustering_date_diagnostics(
    data: pd.DataFrame,
) -> ClusteringDateDiagnostics:
    """Describe per-stock valid ranges without changing clustering inputs."""
    required = {"symbol", "trade_date", "return", "drawdown"}
    missing = required - set(data.columns)
    if missing:
        raise DataValidationError(
            f"聚类日期诊断缺少必要列：{sorted(missing)}"
        )

    valid = data.dropna(subset=["return", "drawdown"]).copy()
    valid["trade_date"] = pd.to_datetime(
        valid["trade_date"], errors="coerce"
    )
    valid = valid.dropna(subset=["trade_date"])
    raw_ranges = (
        valid.groupby("symbol", as_index=False)["trade_date"]
        .agg(start_date="min", end_date="max")
        .sort_values("symbol")
        .reset_index(drop=True)
    )

    empty_ranges = pd.DataFrame(
        columns=[
            "symbol",
            "start_date",
            "end_date",
            "limits_common_start",
            "limits_common_end",
        ]
    )
    if raw_ranges.empty:
        return ClusteringDateDiagnostics(
            is_consistent=True,
            common_start=None,
            common_end=None,
            ranges=empty_ranges,
            limiting_ranges=empty_ranges.copy(),
        )

    earliest_start = raw_ranges["start_date"].min()
    latest_end = raw_ranges["end_date"].max()
    raw_ranges["limits_common_start"] = (
        raw_ranges["start_date"] > earliest_start
    )
    raw_ranges["limits_common_end"] = raw_ranges["end_date"] < latest_end
    is_consistent = bool(
        raw_ranges["start_date"].nunique() <= 1
        and raw_ranges["end_date"].nunique() <= 1
    )

    common_start = raw_ranges["start_date"].max()
    common_end = raw_ranges["end_date"].min()
    ranges = raw_ranges.copy()
    for column in ("start_date", "end_date"):
        ranges[column] = ranges[column].dt.date.astype(str)
    limiting_mask = (
        ranges["limits_common_start"] | ranges["limits_common_end"]
    )

    return ClusteringDateDiagnostics(
        is_consistent=is_consistent,
        common_start=common_start.date().isoformat(),
        common_end=common_end.date().isoformat(),
        ranges=ranges,
        limiting_ranges=ranges.loc[limiting_mask].reset_index(drop=True),
    )


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
    interpretation = build_label_interpretation(result, date_range)
    interpretation = {
        **interpretation,
        "cluster_label": {
            cluster: label.removeprefix("[所选历史区间]")
            for cluster, label in interpretation["cluster_label"].items()
        },
    }
    return ClusteringDashboard(result=result, interpretation=interpretation)
