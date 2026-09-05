"""Deterministic quantitative brief assembled from existing domain outputs."""

from __future__ import annotations

from typing import TypedDict

import pandas as pd
from pandas import DataFrame

from .analysis_service import build_eda_dashboard
from .clustering_service import run_stock_clustering_dashboard
from .supervised_service import (
    run_classification_dashboard,
    run_regression_dashboard,
)
from src.utils.exceptions import NoDataError, StockAnalysisError


class QuantReport(TypedDict):
    """Structured facts and prose consumed by the Streamlit report page."""

    scope: dict[str, object]
    core_findings: list[str]
    focus_snapshot: dict[str, object]
    cluster_label: str | None
    classification: dict[str, object] | None
    regression: dict[str, object] | None
    model_errors: list[str]
    markdown: str


def _record_for_symbol(frame: DataFrame, symbol: str) -> dict[str, object]:
    selected = frame[frame["symbol"].astype(str) == symbol]
    return {} if selected.empty else selected.iloc[0].to_dict()


def _percent(value: object) -> str:
    if value is None or pd.isna(value):
        return "暂无"
    return f"{float(value):.2%}"


def build_quant_report(
    data: DataFrame,
    *,
    focus_symbol: str,
    correlation_method: str = "spearman",
) -> QuantReport:
    """Assemble EDA, clustering and supervised evidence without new formulas."""
    if data.empty:
        raise NoDataError("量化简报没有可用历史数据")
    symbols = list(dict.fromkeys(data["symbol"].astype(str).tolist()))
    if focus_symbol not in symbols:
        raise NoDataError(f"关注股票 {focus_symbol} 不在当前股票池中")

    dates = pd.to_datetime(data["trade_date"], errors="coerce").dropna()
    eda = build_eda_dashboard(
        data,
        correlation_method=correlation_method,
        candlestick_symbol=focus_symbol,
    )
    core_findings = list(eda["presentation"]["summary_sentences"])
    returns = _record_for_symbol(eda["returns"], focus_symbol)
    risk = _record_for_symbol(eda["risk_return"], focus_symbol)
    trend = _record_for_symbol(
        eda["presentation"]["trend_snapshot"], focus_symbol
    )
    focus_snapshot = {
        "累计收益": returns.get("cumulative_return"),
        "上涨日占比": returns.get("win_rate"),
        "日波动率": risk.get("volatility"),
        "最大回撤": risk.get("max_drawdown"),
        "最新收盘价": trend.get("close"),
        "收盘价相对MA20": trend.get("price_vs_ma20", "暂无"),
        "MA5相对MA20": trend.get("ma5_vs_ma20", "暂无"),
    }

    cluster_label: str | None = None
    if len(symbols) >= 3:
        try:
            clustering = run_stock_clustering_dashboard(data, random_state=42)
        except StockAnalysisError:
            pass
        else:
            profiles = clustering["result"]["profiles"]
            selected_profile = profiles[
                profiles["symbol"].astype(str) == focus_symbol
            ]
            if not selected_profile.empty:
                cluster = int(selected_profile["cluster"].iloc[0])
                cluster_label = clustering["interpretation"]["cluster_label"].get(
                    cluster
                )

    model_errors: list[str] = []
    classification: dict[str, object] | None = None
    regression: dict[str, object] | None = None
    try:
        class_dashboard = run_classification_dashboard(
            data, symbol=focus_symbol
        )
    except StockAnalysisError as error:
        model_errors.append(f"决策树：{error}")
    else:
        class_result = class_dashboard["result"]
        class_diagnostics = class_dashboard["diagnostics"]
        classification = {
            "方向信号": class_dashboard["forecast"]["direction_label"],
            "信号截至": class_dashboard["forecast"]["as_of_date"],
            "历史准确率": class_result["metrics"]["accuracy"],
            "较强基线": class_diagnostics["best_baseline_accuracy"],
            "基线名称": class_diagnostics["best_baseline_name"],
            "相对基线": class_diagnostics["accuracy_delta"],
            "可靠性标题": class_dashboard["assessment"]["title"],
            "可靠性说明": class_dashboard["assessment"]["detail"],
        }

    try:
        reg_dashboard = run_regression_dashboard(data, symbol=focus_symbol)
    except StockAnalysisError as error:
        model_errors.append(f"线性回归：{error}")
    else:
        reg_result = reg_dashboard["result"]
        reg_diagnostics = reg_dashboard["diagnostics"]
        regression = {
            "预测收益率": reg_dashboard["forecast"]["predicted_return"],
            "最新收盘价": reg_dashboard["forecast"]["latest_close"],
            "模型换算价格": reg_dashboard["forecast"]["implied_price"],
            "信号截至": reg_dashboard["forecast"]["as_of_date"],
            "历史MAE": reg_result["metrics"]["mae"],
            "历史R²": reg_result["metrics"]["r2"],
            "较强基线MAE": reg_diagnostics["best_baseline_mae"],
            "基线名称": reg_diagnostics["best_baseline_name"],
            "可靠性标题": reg_dashboard["assessment"]["title"],
            "可靠性说明": reg_dashboard["assessment"]["detail"],
        }

    lines = [
        f"# {focus_symbol} 量化分析简报",
        "",
        "## 研究范围",
        (
            f"- 股票池：{len(symbols)} 只；历史区间："
            f"{dates.min().date().isoformat()} 至 "
            f"{dates.max().date().isoformat()}。"
        ),
        f"- 关注股票：{focus_symbol}。",
        "",
        "## 多股历史研究摘要",
        *[f"- {finding}" for finding in core_findings],
        "",
        "## 关注股票历史画像",
        f"- 区间累计收益：{_percent(focus_snapshot['累计收益'])}。",
        f"- 日波动率：{_percent(focus_snapshot['日波动率'])}。",
        f"- 最大回撤：{_percent(focus_snapshot['最大回撤'])}。",
        (
            f"- 趋势位置：收盘价{focus_snapshot['收盘价相对MA20']}，"
            f"MA5{focus_snapshot['MA5相对MA20']}。"
        ),
        f"- 聚类画像：{cluster_label or '当前样本未形成可用聚类画像'}。",
        "",
        "## 下一交易日模型信号",
    ]
    if classification is not None:
        lines.extend(
            [
                (
                    f"- 决策树：{classification['方向信号']}；历史 Accuracy "
                    f"{_percent(classification['历史准确率'])}，"
                    f"{classification['基线名称']} "
                    f"{_percent(classification['较强基线'])}。"
                ),
                f"- 分类可靠性：{classification['可靠性标题']}。",
            ]
        )
    if regression is not None:
        lines.extend(
            [
                (
                    f"- 线性回归：预测收益率 "
                    f"{_percent(regression['预测收益率'])}；"
                    f"模型换算价格 {float(regression['模型换算价格']):.2f}。"
                ),
                (
                    f"- 回归可靠性：{regression['可靠性标题']}；"
                    f"历史 R² {float(regression['历史R²']):.4f}。"
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## 使用边界",
            "- 模型换算价格不是目标价或实时成交价。",
            "- 模型信号必须与历史样本外指标及简单基线一起阅读。",
            "- 本简报仅用于课程中的数据分析与方法展示，不构成投资建议。",
        ]
    )
    return QuantReport(
        scope={
            "股票数量": len(symbols),
            "开始日期": dates.min().date().isoformat(),
            "结束日期": dates.max().date().isoformat(),
            "关注股票": focus_symbol,
        },
        core_findings=core_findings,
        focus_snapshot=focus_snapshot,
        cluster_label=cluster_label,
        classification=classification,
        regression=regression,
        model_errors=model_errors,
        markdown="\n".join(lines),
    )
