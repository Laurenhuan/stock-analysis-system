"""Application-facing orchestration for supervised-learning results."""

from __future__ import annotations

from typing import TypedDict

from pandas import DataFrame
from plotly.graph_objects import Figure

from src.contracts.supervised import ClassificationResult, RegressionResult
from src.models.supervised.classification import run_classification
from src.models.supervised.regression import fit_regression
from src.utils.exceptions import NoDataError
from src.visualization.charts import plot_actual_vs_predicted, plot_confusion_matrix
from .workspace_service import ModelSampleSummary, get_model_sample_summary


class ClassificationDashboard(TypedDict):
    """Stable classification result and its Role 3 visualization."""

    result: ClassificationResult
    confusion_matrix_figure: Figure
    feature_importance: DataFrame
    sample_summary: ModelSampleSummary


class RegressionDashboard(TypedDict):
    """Stable regression result and its Role 3 visualization."""

    result: RegressionResult
    actual_vs_predicted_figure: Figure
    sample_summary: ModelSampleSummary


def run_classification_dashboard(
    data: DataFrame,
    *,
    symbol: str,
) -> ClassificationDashboard:
    """Filter one stock, run Role 4 classification and build its result figure."""
    selected = data[data["symbol"] == symbol].copy()
    if selected.empty:
        raise NoDataError(f"未找到股票 {symbol} 的分类输入数据")
    result = run_classification(selected)
    feature_importance = DataFrame(
        {
            "feature": result["feature_names"],
            "importance": result["model"].feature_importances_,
        }
    ).sort_values("importance", ascending=False, ignore_index=True)
    return ClassificationDashboard(
        result=result,
        confusion_matrix_figure=plot_confusion_matrix(
            result["metrics"]["confusion_matrix"],
            labels=("非上涨", "上涨"),
            title=f"{symbol} 测试集混淆矩阵",
        ),
        feature_importance=feature_importance,
        sample_summary=get_model_sample_summary(
            selected, result["predictions"]
        ),
    )


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
        sample_summary=get_model_sample_summary(
            selected, result["predictions"]
        ),
    )
