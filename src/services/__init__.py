"""Stable application service entry points owned by Role 1."""

from .analysis_service import (
    build_eda_dashboard,
    build_price_figure,
    get_analysis_status,
)
from .clustering_service import get_stock_profiles, run_stock_clustering
from .market_service import (
    get_market_metadata,
    get_market_overview,
    get_sample_date_bounds,
    get_sample_symbols,
    load_market_data,
    load_realtime_quotes,
)
from .supervised_service import (
    run_classification_dashboard,
    run_regression_dashboard,
)

__all__ = [
    "build_eda_dashboard",
    "build_price_figure",
    "get_analysis_status",
    "get_market_metadata",
    "get_market_overview",
    "get_sample_date_bounds",
    "get_sample_symbols",
    "get_stock_profiles",
    "load_market_data",
    "load_realtime_quotes",
    "run_classification_dashboard",
    "run_regression_dashboard",
    "run_stock_clustering",
]
