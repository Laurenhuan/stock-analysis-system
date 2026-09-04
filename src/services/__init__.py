"""Stable application service entry points owned by Role 1."""

from .analysis_service import (
    build_eda_dashboard,
    build_price_figure,
    get_analysis_status,
)
from .clustering_service import (
    get_stock_profiles,
    run_stock_clustering,
    run_stock_clustering_dashboard,
)
from .market_service import (
    get_market_metadata,
    get_market_overview,
    get_sample_date_bounds,
    get_sample_symbols,
    load_market_data,
    load_realtime_quotes,
    search_stocks,
)
from .supervised_service import (
    run_classification_dashboard,
    run_regression_dashboard,
)
from .workspace_service import (
    get_market_summary,
    get_model_sample_summary,
    prepare_symbol_selection,
)

__all__ = [
    "build_eda_dashboard",
    "build_price_figure",
    "get_analysis_status",
    "get_market_metadata",
    "get_market_overview",
    "get_market_summary",
    "get_model_sample_summary",
    "get_sample_date_bounds",
    "get_sample_symbols",
    "get_stock_profiles",
    "load_market_data",
    "load_realtime_quotes",
    "prepare_symbol_selection",
    "run_classification_dashboard",
    "run_regression_dashboard",
    "run_stock_clustering",
    "run_stock_clustering_dashboard",
    "search_stocks",
]
