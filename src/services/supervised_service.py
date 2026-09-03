"""Application-facing orchestration for supervised-learning results."""

from __future__ import annotations

from typing import TypedDict

from pandas import DataFrame
from plotly.graph_objects import Figure

from src.contracts.supervised import RegressionResult
from src.models.supervised.regression import fit_regression
from src.utils.exceptions import NoDataError
from src.visualization.charts import plot_actual_vs_predicted


class RegressionDashboard(TypedDict):
    """Stable regression result and its Role 3 visualization."""

    result: RegressionResult
    actual_vs_predicted_figure: Figure


def run_regression_dashboard(
    data: DataFrame,
    *,
    symbol: str,
) -> RegressionDashboard:
    """Filter one stock, run Role 6 regression and build its result figure."""
    selected = data[data["symbol"] == symbol].copy()
    if selected.empty:
        raise NoDataError(f"未找到股票 {symbol} 的回归输入数据")
    result = fit_regression(selected)
    return RegressionDashboard(
        result=result,
        actual_vs_predicted_figure=plot_actual_vs_predicted(
            result["predictions"], title=f"{symbol} 实际值 vs 预测值"
        ),
    )
