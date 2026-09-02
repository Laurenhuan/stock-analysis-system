"""Cross-module tests for Role 1 Service orchestration."""

import pytest
from plotly.graph_objects import Figure

from src.services import (
    build_eda_dashboard,
    get_sample_symbols,
    load_market_data,
    run_regression_dashboard,
    run_stock_clustering,
)
from src.utils.exceptions import NoDataError


@pytest.fixture(scope="module")
def sample_market_data():
    return load_market_data(get_sample_symbols(), source="sample")


def test_eda_service_builds_tables_and_figures(sample_market_data) -> None:
    selected = sample_market_data[
        sample_market_data["symbol"].isin(get_sample_symbols()[:3])
    ]
    dashboard = build_eda_dashboard(selected)

    assert dashboard["date_ranges"].shape[0] == 3
    assert dashboard["risk_return"].shape[0] == 3
    assert dashboard["correlation"].shape == (3, 3)
    assert isinstance(dashboard["price_figure"], Figure)
    assert isinstance(dashboard["correlation_figure"], Figure)


def test_regression_service_runs_role6_and_role3_figure(sample_market_data) -> None:
    symbol = get_sample_symbols()[0]
    dashboard = run_regression_dashboard(sample_market_data, symbol=symbol)
    result = dashboard["result"]

    assert set(result["metrics"]) == {"mae", "r2"}
    assert list(result["predictions"].columns) == [
        "trade_date",
        "y_true",
        "y_pred",
    ]
    assert len(result["predictions"]) >= 2
    assert isinstance(dashboard["actual_vs_predicted_figure"], Figure)


def test_regression_service_rejects_symbol_outside_input(sample_market_data) -> None:
    with pytest.raises(NoDataError, match="未找到股票"):
        run_regression_dashboard(sample_market_data, symbol="999999.SH")


def test_clustering_service_keeps_contract_fixed_k(sample_market_data) -> None:
    result = run_stock_clustering(sample_market_data, random_state=42)

    assert result["k"] == 3
    assert len(result["profiles"]) == 10
    assert set(result["profiles"]["cluster"]).issubset({0, 1, 2})
    assert list(result["cluster_centers"].columns) == [
        "cluster",
        "mean_return",
        "volatility",
        "max_drawdown",
    ]
