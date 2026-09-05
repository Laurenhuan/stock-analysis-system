"""Application-facing orchestration for supervised-learning results."""

from __future__ import annotations

from typing import Literal, TypedDict

import numpy as np
import pandas as pd
from pandas import DataFrame
from plotly.graph_objects import Figure

from src.contracts.supervised import ClassificationResult, RegressionResult
from src.models.supervised.classification import (
    ClassificationSampleInfo,
    NextDirectionForecast,
    forecast_next_direction,
    get_classification_sample_info,
    run_classification,
)
from src.models.supervised.regression import (
    NextReturnForecast,
    RegressionSampleInfo,
    fit_regression,
    forecast_next_return,
    get_regression_sample_info,
)
from src.utils.exceptions import NoDataError, StockAnalysisError
from src.visualization.charts import plot_actual_vs_predicted, plot_confusion_matrix


class ValidationWindow(TypedDict):
    """One expanding historical evaluation window and its simple baseline."""

    end_date: str
    model_score: float
    baseline_score: float
    delta: float


class ClassificationDiagnostics(TypedDict):
    """Baselines and stability evidence separate from the model Contract."""

    baseline_name: str
    persistence_accuracy: float | None
    majority_accuracy: float
    best_baseline_name: str
    best_baseline_accuracy: float
    accuracy_delta: float
    test_up_rate: float
    predicted_up_rate: float
    validation_windows: list[ValidationWindow]
    validation_mean_delta: float | None


class RegressionDiagnostics(TypedDict):
    """Baselines and stability evidence separate from the model Contract."""

    baseline_name: str
    zero_baseline_mae: float
    training_mean_baseline_mae: float
    best_baseline_name: str
    best_baseline_mae: float
    mae_improvement: float
    validation_windows: list[ValidationWindow]
    validation_mean_delta: float | None


class ModelAssessment(TypedDict):
    """Neutral user-facing conclusion about historical out-of-sample ability."""

    level: Literal["warning", "info", "success"]
    title: str
    detail: str


class ClassificationDashboard(TypedDict):
    """Stable classification result and its Role 3 visualization."""

    result: ClassificationResult
    confusion_matrix_figure: Figure
    feature_importance: DataFrame
    sample_summary: ClassificationSampleInfo
    diagnostics: ClassificationDiagnostics
    assessment: ModelAssessment
    forecast: NextDirectionForecast


class RegressionDashboard(TypedDict):
    """Stable regression result and its Role 3 visualization."""

    result: RegressionResult
    actual_vs_predicted_figure: Figure
    sample_summary: RegressionSampleInfo
    diagnostics: RegressionDiagnostics
    assessment: ModelAssessment
    forecast: NextReturnForecast


def _expanding_prefixes(data: DataFrame) -> list[DataFrame]:
    """Build deterministic 60%/80%/100% prefixes without touching final data."""
    ordered = data.sort_values("trade_date").reset_index(drop=True)
    prefixes: list[DataFrame] = []
    seen_counts: set[int] = set()
    for fraction in (0.6, 0.8, 1.0):
        count = max(1, int(len(ordered) * fraction))
        if count in seen_counts:
            continue
        seen_counts.add(count)
        prefixes.append(ordered.iloc[:count].copy())
    return prefixes


def _persistence_accuracy(
    data: DataFrame,
    result: ClassificationResult,
) -> float | None:
    """Accuracy of the naive rule: tomorrow repeats today's return direction."""
    if "return" not in data.columns:
        return None
    context = data[["trade_date", "return"]].copy()
    context["trade_date"] = pd.to_datetime(context["trade_date"])
    predictions = result["predictions"][["trade_date", "y_true"]].copy()
    predictions["trade_date"] = pd.to_datetime(predictions["trade_date"])
    aligned = predictions.merge(
        context,
        on="trade_date",
        how="left",
        validate="one_to_one",
    ).dropna(subset=["return", "y_true"])
    if aligned.empty:
        return None
    baseline = (aligned["return"].astype(float) > 0).astype(int)
    return float((baseline == aligned["y_true"].astype(int)).mean())


def _majority_accuracy(result: ClassificationResult) -> float:
    """Accuracy of always predicting the majority class seen in training."""
    model = result["model"]
    root_counts = np.asarray(model.tree_.value[0]).reshape(-1)
    majority_class = int(model.classes_[int(np.argmax(root_counts))])
    y_true = result["predictions"]["y_true"].astype(int)
    return float((y_true == majority_class).mean())


def _classification_diagnostics(
    selected: DataFrame,
    result: ClassificationResult,
) -> ClassificationDiagnostics:
    persistence = _persistence_accuracy(selected, result)
    majority = _majority_accuracy(result)
    candidates = [("训练集多数类基线", majority)]
    if persistence is not None:
        candidates.append(("当日方向延续基线", persistence))
    best_name, best_accuracy = max(candidates, key=lambda item: item[1])
    accuracy = float(result["metrics"]["accuracy"])
    predictions = result["predictions"]
    windows: list[ValidationWindow] = []
    for prefix in _expanding_prefixes(selected):
        try:
            window_result = (
                result
                if len(prefix) == len(selected)
                else run_classification(prefix)
            )
        except StockAnalysisError:
            continue
        window_persistence = _persistence_accuracy(prefix, window_result)
        window_baselines = [_majority_accuracy(window_result)]
        if window_persistence is not None:
            window_baselines.append(window_persistence)
        window_baseline = max(window_baselines)
        window_accuracy = float(window_result["metrics"]["accuracy"])
        windows.append(
            ValidationWindow(
                end_date=pd.Timestamp(
                    prefix["trade_date"].iloc[-1]
                ).date().isoformat(),
                model_score=window_accuracy,
                baseline_score=window_baseline,
                delta=window_accuracy - window_baseline,
            )
        )
    mean_delta = (
        float(np.mean([window["delta"] for window in windows]))
        if windows
        else None
    )
    return ClassificationDiagnostics(
        baseline_name="较强简单基线",
        persistence_accuracy=persistence,
        majority_accuracy=majority,
        best_baseline_name=best_name,
        best_baseline_accuracy=best_accuracy,
        accuracy_delta=accuracy - best_accuracy,
        test_up_rate=float(predictions["y_true"].astype(float).mean()),
        predicted_up_rate=float(predictions["y_pred"].astype(float).mean()),
        validation_windows=windows,
        validation_mean_delta=mean_delta,
    )

def _classification_assessment(
    diagnostics: ClassificationDiagnostics,
) -> ModelAssessment:
    delta = diagnostics["accuracy_delta"]
    mean_delta = diagnostics["validation_mean_delta"]
    if delta is None:
        return ModelAssessment(
            level="info",
            title="基线暂不可用",
            detail="缺少可对齐的当日收益率，无法判断模型是否超过简单方向延续规则。",
        )
    if delta <= 0:
        return ModelAssessment(
            level="warning",
            title="当前测试区间未超过简单基线",
            detail=(
                "决策树没有比“次日延续当日涨跌方向”的简单规则更准确；"
                "当前结果不支持把它解释为有效的未来方向预测。"
            ),
        )
    if mean_delta is None or mean_delta <= 0:
        return ModelAssessment(
            level="warning",
            title="仅当前测试区间小幅超过基线",
            detail=(
                "当前区间存在少量提升，但扩展历史窗口没有显示稳定优势；"
                "更合适的结论是预测能力不稳定。"
            ),
        )
    return ModelAssessment(
        level="info",
        title="当前区间与历史窗口均超过简单基线",
        detail=(
            "模型在本次样本中表现出一定历史区分能力，但仍是课程实验结果，"
            "不能据此推断未来表现。"
        ),
    )


def _zero_baseline_mae(result: RegressionResult) -> float:
    y_true = result["predictions"]["y_true"].astype(float)
    return float(y_true.abs().mean())


def _training_mean_baseline_mae(
    selected: DataFrame,
    result: RegressionResult,
) -> float:
    """MAE of a deployable baseline fitted only on the earliest 80%."""
    frame = selected.sort_values("trade_date").reset_index(drop=True).copy()
    frame["_next_return"] = frame["return"].shift(-1)
    frame = frame.dropna(
        subset=[*result["feature_names"], "_next_return"]
    ).reset_index(drop=True)
    split_index = int(len(frame) * 0.8)
    training_mean = float(frame["_next_return"].iloc[:split_index].mean())
    y_true = result["predictions"]["y_true"].astype(float)
    return float((y_true - training_mean).abs().mean())


def _regression_diagnostics(
    selected: DataFrame,
    result: RegressionResult,
) -> RegressionDiagnostics:
    model_mae = float(result["metrics"]["mae"])
    zero_mae = _zero_baseline_mae(result)
    mean_mae = _training_mean_baseline_mae(selected, result)
    if zero_mae <= mean_mae:
        best_name, best_mae = "零收益基线", zero_mae
    else:
        best_name, best_mae = "训练集均值基线", mean_mae
    windows: list[ValidationWindow] = []
    for prefix in _expanding_prefixes(selected):
        try:
            window_result = (
                result if len(prefix) == len(selected) else fit_regression(prefix)
            )
        except StockAnalysisError:
            continue
        window_model_mae = float(window_result["metrics"]["mae"])
        window_baseline = min(
            _zero_baseline_mae(window_result),
            _training_mean_baseline_mae(prefix, window_result),
        )
        windows.append(
            ValidationWindow(
                end_date=pd.Timestamp(
                    prefix["trade_date"].iloc[-1]
                ).date().isoformat(),
                model_score=window_model_mae,
                baseline_score=window_baseline,
                delta=window_baseline - window_model_mae,
            )
        )
    mean_delta = (
        float(np.mean([window["delta"] for window in windows]))
        if windows
        else None
    )
    return RegressionDiagnostics(
        baseline_name="较强简单基线",
        zero_baseline_mae=zero_mae,
        training_mean_baseline_mae=mean_mae,
        best_baseline_name=best_name,
        best_baseline_mae=best_mae,
        mae_improvement=best_mae - model_mae,
        validation_windows=windows,
        validation_mean_delta=mean_delta,
    )

def _regression_assessment(
    result: RegressionResult,
    diagnostics: RegressionDiagnostics,
) -> ModelAssessment:
    r2 = float(result["metrics"]["r2"])
    improvement = diagnostics["mae_improvement"]
    mean_delta = diagnostics["validation_mean_delta"]
    if r2 <= 0 or improvement <= 0:
        return ModelAssessment(
            level="warning",
            title="当前特征未检出有效线性预测能力",
            detail=(
                "线性回归的 R² 不为正，或 MAE 未优于较强的简单基线；"
                "当前结果不应被展示为有效的未来收益预测。"
            ),
        )
    if mean_delta is None or mean_delta <= 0:
        return ModelAssessment(
            level="warning",
            title="当前测试区间优于基线，但跨窗口不稳定",
            detail=(
                "本次测试区间误差较低，但较早历史窗口没有形成一致优势，"
                "只能说明局部样本关系。"
            ),
        )
    return ModelAssessment(
        level="info",
        title="当前区间存在有限线性解释能力",
        detail=(
            "模型在当前测试区间及扩展历史窗口均优于较强简单基线，"
            "但该结论仍仅限所选历史样本。"
        ),
    )


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
    diagnostics = _classification_diagnostics(selected, result)
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
        sample_summary=get_classification_sample_info(selected),
        diagnostics=diagnostics,
        assessment=_classification_assessment(diagnostics),
        forecast=forecast_next_direction(selected),
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
    diagnostics = _regression_diagnostics(selected, result)
    return RegressionDashboard(
        result=result,
        actual_vs_predicted_figure=plot_actual_vs_predicted(
            result["predictions"], title=f"{symbol} 历史测试：实际值 vs 预测值"
        ),
        sample_summary=get_regression_sample_info(selected),
        diagnostics=diagnostics,
        assessment=_regression_assessment(result, diagnostics),
        forecast=forecast_next_return(selected),
    )
