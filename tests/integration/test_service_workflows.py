"""Cross-module tests for Role 1 Service orchestration."""

import pandas as pd
import pytest
from plotly.graph_objects import Figure

from src.services import (
    build_eda_dashboard,
    build_quant_report,
    get_clustering_date_diagnostics,
    get_sample_symbols,
    load_market_data,
    run_classification_dashboard,
    run_regression_dashboard,
    run_stock_clustering,
    run_stock_clustering_dashboard,
)
from src.utils.exceptions import NoDataError


@pytest.fixture(scope="module")
def sample_market_data():
    return load_market_data(get_sample_symbols(), source="sample")


def test_eda_service_builds_tables_and_figures(sample_market_data) -> None:
    selected = sample_market_data[
        sample_market_data["symbol"].isin(get_sample_symbols()[:3])
    ]
    candle_symbol = get_sample_symbols()[1]
    dashboard = build_eda_dashboard(
        selected, candlestick_symbol=candle_symbol
    )

    assert dashboard["date_ranges"].shape[0] == 3
    assert dashboard["risk_return"].shape[0] == 3
    assert dashboard["return_distribution"].shape[0] == 3
    assert dashboard["extreme_returns"].shape[0] == 3
    assert dashboard["correlation"].shape == (3, 3)
    assert dashboard["insights"]
    assert all(
        "不构成投资建议" in insight["caveat"]
        for insight in dashboard["insights"]
    )
    assert isinstance(dashboard["price_figure"], Figure)
    assert isinstance(dashboard["candlestick_figure"], Figure)
    assert isinstance(dashboard["return_distribution_figure"], Figure)
    assert isinstance(dashboard["rolling_volatility_figure"], Figure)
    assert dashboard["candlestick_figure"].data[0].name == candle_symbol
    assert isinstance(dashboard["correlation_figure"], Figure)


@pytest.mark.parametrize(
    ("method", "title_prefix"),
    [
        ("spearman", "Spearman"),
        ("pearson", "Pearson"),
        ("kendall", "Kendall"),
    ],
)
def test_eda_service_supports_all_correlation_methods(
    sample_market_data,
    method: str,
    title_prefix: str,
) -> None:
    selected = sample_market_data[
        sample_market_data["symbol"].isin(get_sample_symbols()[:3])
    ]

    dashboard = build_eda_dashboard(selected, correlation_method=method)

    assert dashboard["correlation"].shape == (3, 3)
    assert dashboard["correlation_figure"].layout.title.text.startswith(
        title_prefix
    )


def test_eda_service_degrades_only_correlation_when_dates_do_not_overlap(
    sample_market_data,
) -> None:
    symbols = get_sample_symbols()[:2]
    selected = sample_market_data[
        sample_market_data["symbol"].isin(symbols)
    ].copy()
    shifted = selected["symbol"] == symbols[1]
    selected.loc[shifted, "trade_date"] = pd.Series(
        pd.date_range("2026-01-01", periods=int(shifted.sum()), freq="D"),
        index=selected.index[shifted],
    )

    dashboard = build_eda_dashboard(selected)

    assert dashboard["correlation"] is None
    assert dashboard["correlation_figure"] is None
    assert dashboard["price_figure"].data
    assert any(
        insight["title"] == "相关性样本不足"
        for insight in dashboard["insights"]
    )


def test_classification_service_runs_role4_and_role3_figure(
    sample_market_data,
) -> None:
    symbol = get_sample_symbols()[0]
    dashboard = run_classification_dashboard(
        sample_market_data, symbol=symbol
    )
    result = dashboard["result"]

    assert set(result["metrics"]) == {"accuracy", "confusion_matrix"}
    assert 0.0 <= result["metrics"]["accuracy"] <= 1.0
    assert len(result["metrics"]["confusion_matrix"]) == 2
    assert list(result["predictions"].columns) == [
        "trade_date",
        "y_true",
        "y_pred",
    ]
    assert len(result["predictions"]) >= 2
    sample = dashboard["sample_summary"]
    assert sample["effective_rows"] == sample["train_rows"] + sample["test_rows"]
    assert sample["test_rows"] == len(result["predictions"])
    assert isinstance(dashboard["confusion_matrix_figure"], Figure)


def test_classification_service_rejects_symbol_outside_input(
    sample_market_data,
) -> None:
    with pytest.raises(NoDataError, match="未找到股票"):
        run_classification_dashboard(sample_market_data, symbol="999999.SH")


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
    sample = dashboard["sample_summary"]
    assert sample["effective_rows"] == sample["train_rows"] + sample["test_rows"]
    assert sample["test_rows"] == len(result["predictions"])
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


def test_clustering_date_diagnostics_identifies_limited_stock(
    sample_market_data,
) -> None:
    symbols = get_sample_symbols()[:3]
    selected = sample_market_data[
        sample_market_data["symbol"].isin(symbols)
    ].copy()
    late_symbol = symbols[-1]
    late_start = (
        selected.loc[selected["symbol"] == late_symbol, "trade_date"]
        .sort_values()
        .iloc[30]
    )
    selected = selected[
        (selected["symbol"] != late_symbol)
        | (selected["trade_date"] >= late_start)
    ]

    diagnostics = get_clustering_date_diagnostics(selected)

    assert diagnostics["is_consistent"] is False
    assert diagnostics["common_start"] == late_start.date().isoformat()
    assert diagnostics["common_end"] == "2024-12-31"
    assert diagnostics["limiting_ranges"]["symbol"].tolist() == [late_symbol]
    assert diagnostics["limiting_ranges"][
        "limits_common_start"
    ].tolist() == [True]
    assert diagnostics["limiting_ranges"][
        "limits_common_end"
    ].tolist() == [False]


def test_clustering_date_diagnostics_identifies_early_end(
    sample_market_data,
) -> None:
    symbols = get_sample_symbols()[:3]
    selected = sample_market_data[
        sample_market_data["symbol"].isin(symbols)
    ].copy()
    short_symbol = symbols[1]
    early_end = (
        selected.loc[selected["symbol"] == short_symbol, "trade_date"]
        .sort_values()
        .iloc[-30]
    )
    selected = selected[
        (selected["symbol"] != short_symbol)
        | (selected["trade_date"] <= early_end)
    ]

    diagnostics = get_clustering_date_diagnostics(selected)
    limited = diagnostics["limiting_ranges"].set_index("symbol")

    assert diagnostics["is_consistent"] is False
    assert diagnostics["common_end"] == early_end.date().isoformat()
    assert limited.index.tolist() == [short_symbol]
    assert bool(limited.loc[short_symbol, "limits_common_end"]) is True


def test_clustering_dashboard_attaches_dynamic_labels_and_scope(
    sample_market_data,
) -> None:
    dashboard = run_stock_clustering_dashboard(
        sample_market_data, random_state=42
    )
    result = dashboard["result"]
    interpretation = dashboard["interpretation"]

    assert result["k"] == 3
    assert set(interpretation["cluster_label"]) == {0, 1, 2}
    assert all(
        not label.startswith("[所选历史区间]")
        for label in interpretation["cluster_label"].values()
    )
    assert interpretation["样本范围"] == {
        "股票数量": 10,
        "起始日期": "2024-01-02",
        "截止日期": "2024-12-31",
    }
    assert "不构成任何投资建议" in interpretation["免责声明"]

def test_eda_presentation_scales_to_five_and_ten_symbols(
    sample_market_data,
) -> None:
    for count in (5, 10):
        selected = sample_market_data[
            sample_market_data["symbol"].isin(get_sample_symbols()[:count])
        ]
        dashboard = build_eda_dashboard(selected)
        presentation = dashboard["presentation"]

        assert 1 <= len(presentation["core_insights"]) <= 5
        assert len(presentation["summary_sentences"]) <= 4
        assert set(presentation["sections"]) == {
            "performance",
            "risk",
            "correlation",
            "trend",
            "distribution",
            "data_quality",
        }
        assert presentation["trend_snapshot"].shape[0] == count


def test_supervised_dashboards_compare_models_with_simple_baselines(
    sample_market_data,
) -> None:
    symbol = get_sample_symbols()[0]
    classification = run_classification_dashboard(
        sample_market_data, symbol=symbol
    )
    regression = run_regression_dashboard(sample_market_data, symbol=symbol)

    class_diagnostics = classification["diagnostics"]
    assert class_diagnostics["baseline_name"] == "较强简单基线"
    assert class_diagnostics["validation_windows"]
    assert class_diagnostics["accuracy_delta"] == pytest.approx(
        classification["result"]["metrics"]["accuracy"]
        - class_diagnostics["best_baseline_accuracy"]
    )
    assert classification["assessment"]["level"] in {
        "warning",
        "info",
        "success",
    }

    regression_diagnostics = regression["diagnostics"]
    assert regression_diagnostics["baseline_name"] == "较强简单基线"
    assert regression_diagnostics["validation_windows"]
    assert regression_diagnostics["mae_improvement"] == pytest.approx(
        regression_diagnostics["best_baseline_mae"]
        - regression["result"]["metrics"]["mae"]
    )
    assert regression["assessment"]["level"] in {
        "warning",
        "info",
        "success",
    }

def test_supervised_dashboards_expose_latest_signals(
    sample_market_data,
) -> None:
    symbol = get_sample_symbols()[0]
    classification = run_classification_dashboard(
        sample_market_data, symbol=symbol
    )
    regression = run_regression_dashboard(sample_market_data, symbol=symbol)

    assert classification["forecast"]["direction_label"] in {
        "上涨倾向",
        "非上涨倾向",
    }
    assert classification["forecast"]["as_of_date"] == "2024-12-31"
    assert regression["forecast"]["as_of_date"] == "2024-12-31"
    assert regression["forecast"]["implied_price"] == pytest.approx(
        regression["forecast"]["latest_close"]
        * (1.0 + regression["forecast"]["predicted_return"])
    )

def test_quant_report_combines_existing_outputs(sample_market_data) -> None:
    symbols = get_sample_symbols()[:5]
    selected = sample_market_data[
        sample_market_data["symbol"].isin(symbols)
    ]
    report = build_quant_report(selected, focus_symbol=symbols[0])

    assert report["scope"]["股票数量"] == 5
    assert report["scope"]["关注股票"] == symbols[0]
    assert report["core_findings"]
    assert report["cluster_label"]
    assert report["classification"] is not None
    assert report["regression"] is not None
    assert "下一交易日模型信号" in report["markdown"]
    assert "不构成投资建议" in report["markdown"]
