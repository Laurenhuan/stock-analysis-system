"""Single-stock workspace sharing one query across overview and models."""

from datetime import date

import pandas as pd
import streamlit as st

from app_pages.shared import (
    DEFAULT_START_DATE,
    SOURCE_OPTIONS,
    cached_market_data,
    cached_realtime_quotes,
    render_provenance,
    render_realtime_snapshot,
    render_stock_search,
)
from src.services import (
    build_price_figure,
    get_market_metadata,
    get_market_summary,
    get_sample_symbols,
    prepare_symbol_selection,
    run_classification_dashboard,
    run_regression_dashboard,
)
from src.utils.exceptions import StockAnalysisError


def _percent(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}%}"


def _percentage_points(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:+.2f} 个百分点"


def _render_assessment(assessment: dict[str, str]) -> None:
    """Render weak model evidence as a neutral finding, not a system alert."""
    with st.container(border=True):
        st.caption("历史可靠性结论")
        st.markdown(f"**{assessment['title']}**")
        st.write(assessment["detail"])

def _classification_windows(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "窗口截止日": row["end_date"],
                "决策树 Accuracy": _percent(float(row["model_score"])),
                "较强简单基线": _percent(float(row["baseline_score"])),
                "相对基线": _percentage_points(float(row["delta"])),
            }
            for row in rows
        ]
    )


def _regression_windows(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "窗口截止日": row["end_date"],
                "模型 MAE": _percent(float(row["model_score"])),
                "零收益基线 MAE": _percent(float(row["baseline_score"])),
                "基线误差减去模型误差": _percentage_points(float(row["delta"])),
            }
            for row in rows
        ]
    )


st.caption("一组股票与日期条件，复用于行情概览和两项历史可预测性检验。")

searched_symbols, searched_names = render_stock_search("single")
symbol_options = list(dict.fromkeys([*get_sample_symbols(), *searched_symbols]))

with st.form("single_stock_query"):
    symbol_col, source_col, start_col, end_col = st.columns(
        [1.3, 1.5, 1, 1], vertical_alignment="bottom"
    )
    with symbol_col:
        symbol_input = st.selectbox(
            "股票代码",
            options=symbol_options,
            format_func=lambda value: (
                f"{value} · {searched_names[value]}"
                if value in searched_names
                else value
            ),
            accept_new_options=True,
            placeholder="例如 600519.SH",
            help="可搜索后选择，也可直接输入 600519、600519.SH 或 sh600519。",
            key="single_symbol_input",
            persist_state="session",
        )
    with source_col:
        source_label = st.segmented_control(
            "数据来源",
            options=list(SOURCE_OPTIONS),
            default=list(SOURCE_OPTIONS)[0],
            key="single_source_input",
            persist_state="session",
        )
    with start_col:
        start_date = st.date_input(
            "开始日期",
            value=DEFAULT_START_DATE,
            max_value=date.today(),
            key="single_start_date",
            persist_state="session",
        )
    with end_col:
        end_date = st.date_input(
            "结束日期",
            value=date.today(),
            max_value=date.today(),
            key="single_end_date",
            persist_state="session",
        )
    submitted = st.form_submit_button(
        "加载历史数据",
        type="primary",
        icon=":material/play_arrow:",
    )

if submitted:
    try:
        symbol = prepare_symbol_selection(
            symbol_input or "", min_count=1, max_count=1
        )[0]
        if source_label not in SOURCE_OPTIONS:
            raise ValueError("请选择数据来源")
    except (StockAnalysisError, ValueError) as error:
        st.error(str(error), icon=":material/error:")
    else:
        st.session_state.single_query = {
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "source": SOURCE_OPTIONS[source_label],
        }

query = st.session_state.get("single_query")
if query is None:
    st.info(
        "输入股票和日期后加载。在线模式直接请求 AkShare，不会写入本地 CSV。",
        icon=":material/touch_app:",
    )
    st.stop()

try:
    with st.spinner("正在加载历史行情…"):
        market_data = cached_market_data(
            (query["symbol"],),
            query["start_date"],
            query["end_date"],
            query["source"],
        )
    metadata = get_market_metadata(market_data)
    summary = get_market_summary(market_data)
    active_symbol = str(market_data["symbol"].iloc[0])
except StockAnalysisError as error:
    st.error(f"历史行情加载失败：{error}", icon=":material/error:")
    st.stop()

render_provenance(metadata)
st.caption(
    f"当前查询：{active_symbol} · {query['start_date']} 至 {query['end_date']}"
)

overview_tab, classification_tab, regression_tab = st.tabs(
    ["行情概览", "决策树分类", "线性回归"],
    key="single_workspace_tabs",
    on_change="rerun",
)

if overview_tab.open:
    with overview_tab:
        with st.container(horizontal=True, gap="medium"):
            st.metric("记录数", summary["row_count"], border=True)
            st.metric("首个交易日", summary["first_date"], border=True)
            st.metric("最后交易日", summary["last_date"], border=True)
            st.metric("股票数", summary["symbol_count"], border=True)

        st.plotly_chart(build_price_figure(market_data), width="stretch")
        st.caption("MA5/MA20 前导空值是滚动窗口的正常预热期。")

        with st.container(border=True):
            st.subheader("实时行情快照", icon=":material/speed:")
            st.caption("实时数据不会作为 EDA、分类或回归的训练输入。")
            if st.button(
                "查询实时快照",
                icon=":material/refresh:",
                key="single_realtime",
            ):
                try:
                    quotes = cached_realtime_quotes((active_symbol,))
                    quote_metadata = get_market_metadata(quotes)
                except StockAnalysisError as error:
                    st.error(
                        f"实时行情查询失败：{error}",
                        icon=":material/error:",
                    )
                else:
                    render_realtime_snapshot(quotes, quote_metadata)

        with st.expander(
            "查看标准行情与公共指标", icon=":material/table_chart:"
        ):
            st.dataframe(market_data, hide_index=True)

if classification_tab.open:
    with classification_tab:
        st.subheader("历史方向可预测性检验")
        st.caption(
            "问题：现有特征能否在最新 20% 历史样本中，超过“次日延续当日涨跌方向”的简单规则？"
            "模型仍按 X(t) → y(t+1)、最早 80% 训练、最新 20% 测试且不打乱。"
        )
        try:
            dashboard = run_classification_dashboard(
                market_data, symbol=active_symbol
            )
        except StockAnalysisError as error:
            st.error(f"决策树运行失败：{error}", icon=":material/error:")
        else:
            result = dashboard["result"]
            sample = dashboard["sample_summary"]
            diagnostics = dashboard["diagnostics"]
            forecast = dashboard["forecast"]
            with st.container(border=True):
                st.caption(
                    f"模型使用截至 {forecast['as_of_date']} 的已知历史；"
                    "目标日期为下一交易日。"
                )
                st.subheader("下一交易日方向信号")
                st.metric("决策树判断", forecast["direction_label"])
                st.caption(
                    "这是由完整已知历史重新训练后得到的模型信号，不是确定结果或投资指令。"
                )
            _render_assessment(dashboard["assessment"])

            with st.container(horizontal=True, gap="medium"):
                st.metric(
                    "决策树 Accuracy",
                    _percent(float(result["metrics"]["accuracy"])),
                    border=True,
                )
                st.metric(
                    diagnostics["best_baseline_name"],
                    _percent(diagnostics["best_baseline_accuracy"]),
                    border=True,
                )
                st.metric(
                    "相对基线",
                    _percentage_points(diagnostics["accuracy_delta"]),
                    border=True,
                )
                st.metric(
                    "多窗口平均优势",
                    _percentage_points(diagnostics["validation_mean_delta"]),
                    border=True,
                )
            st.caption(
                f"原始 {sample['input_rows']} 个交易日，形成 {sample['effective_rows']} 个有效样本；"
                f"训练 {sample['train_rows']} 日（{sample['train_date_range']}），"
                f"测试 {sample['test_rows']} 日（{sample['test_date_range']}）。"
                "这些数字是交易日数，不是股票数。"
            )
            st.info(
                f"方向延续基线为 {_percent(diagnostics['persistence_accuracy'])}，"
                f"训练集多数类基线为 {_percent(diagnostics['majority_accuracy'])}，取较强者。"
                "Accuracy 只有稳定超过它才具有增量信息；混淆矩阵用于检查类别偏向。"
            )

            left, right = st.columns([1.4, 1])
            with left:
                st.plotly_chart(
                    dashboard["confusion_matrix_figure"], width="stretch"
                )
            with right:
                st.subheader("模型关注了哪些特征")
                feature_names = {
                    "return_lag1": "前 1 日收益率",
                    "return_lag2": "前 2 日收益率",
                    "ma_diff": "MA5 与 MA20 价差",
                    "volatility_20d": "20 日波动率",
                    "volume_change": "成交量变化率",
                }
                importance = dashboard["feature_importance"].rename(
                    columns={"feature": "特征", "importance": "重要性"}
                )
                importance["特征"] = importance["特征"].map(
                    lambda value: feature_names.get(value, value)
                )
                st.dataframe(
                    importance,
                    hide_index=True,
                    column_config={
                        "重要性": st.column_config.ProgressColumn(
                            "重要性", min_value=0.0, max_value=1.0
                        )
                    },
                )
                st.caption("重要性表示树在历史样本中的分裂贡献，不代表因果关系。")

            with st.expander("查看扩展历史窗口与测试明细"):
                windows = _classification_windows(
                    diagnostics["validation_windows"]
                )
                if windows.empty:
                    st.caption("当前日期范围不足以形成额外历史窗口。")
                else:
                    st.dataframe(windows, hide_index=True)
                st.caption(
                    f"测试集实际上涨占比 {_percent(diagnostics['test_up_rate'])}；"
                    f"模型预测上涨占比 {_percent(diagnostics['predicted_up_rate'])}。"
                )
                st.dataframe(result["predictions"], hide_index=True)

if regression_tab.open:
    with regression_tab:
        st.subheader("次日收益线性关系检验")
        st.caption(
            "问题：现有公开特征与次日收益是否存在稳定线性关系？"
            "模型按 X(t) → return(t+1)、最早 80% 训练、最新 20% 测试且不打乱。"
        )
        try:
            dashboard = run_regression_dashboard(
                market_data, symbol=active_symbol
            )
        except StockAnalysisError as error:
            st.error(f"线性回归运行失败：{error}", icon=":material/error:")
        else:
            result = dashboard["result"]
            sample = dashboard["sample_summary"]
            diagnostics = dashboard["diagnostics"]
            forecast = dashboard["forecast"]
            with st.container(border=True):
                st.caption(
                    f"模型使用截至 {forecast['as_of_date']} 的已知历史；"
                    "目标日期为下一交易日。"
                )
                st.subheader("下一交易日收益率信号")
                signal_columns = st.columns(3)
                signal_columns[0].metric(
                    "预测收益率", _percent(forecast["predicted_return"])
                )
                signal_columns[1].metric(
                    "最新收盘价", f"{forecast['latest_close']:.2f}"
                )
                signal_columns[2].metric(
                    "模型换算价格", f"{forecast['implied_price']:.2f}"
                )
                st.caption(
                    "模型换算价格=最新收盘价×(1+预测收益率)，不是目标价或实时成交价。"
                )
            _render_assessment(dashboard["assessment"])

            with st.container(horizontal=True, gap="medium"):
                st.metric(
                    "模型 MAE",
                    _percent(float(result["metrics"]["mae"])),
                    border=True,
                )
                st.metric(
                    f"{diagnostics['best_baseline_name']} MAE",
                    _percent(diagnostics["best_baseline_mae"]),
                    border=True,
                )
                st.metric(
                    "相对基线误差改善",
                    _percentage_points(diagnostics["mae_improvement"]),
                    border=True,
                )
                st.metric(
                    "R²",
                    f"{float(result['metrics']['r2']):.4f}",
                    border=True,
                )
            st.caption(
                f"原始 {sample['input_rows']} 个交易日，形成 {sample['effective_rows']} 个有效样本；"
                f"训练 {sample['train_rows']} 日（{sample['train_date_range']}），"
                f"测试 {sample['test_rows']} 日（{sample['test_date_range']}）。"
                "这些数字是交易日数，不是股票数。"
            )
            st.info(
                f"零收益基线 MAE 为 {_percent(diagnostics['zero_baseline_mae'])}，"
                f"训练集均值基线 MAE 为 {_percent(diagnostics['training_mean_baseline_mae'])}，"
                "页面取误差较低者。R²≤0 表示线性解释能力较弱；"
                "低分代表预测增量有限，不是系统故障。"
            )
            st.plotly_chart(
                dashboard["actual_vs_predicted_figure"], width="stretch"
            )
            with st.expander("查看扩展历史窗口与测试明细"):
                windows = _regression_windows(diagnostics["validation_windows"])
                if windows.empty:
                    st.caption("当前日期范围不足以形成额外历史窗口。")
                else:
                    st.dataframe(windows, hide_index=True)
                st.dataframe(result["predictions"], hide_index=True)

st.info(
    "以上均为所选历史区间的教学性分析与模型信号，不代表未来表现，不构成投资建议。",
    icon=":material/info:",
)

st.divider()
st.subheader("下一步：生成量化分析简报", icon=":material/article:")
st.session_state.focus_symbol = active_symbol
if st.session_state.get("multi_query") is None:
    st.caption("尚未建立多股股票池；请先完成多股历史研究，再生成完整简报。")
    if st.button("前往多股历史研究", key="single_to_multi"):
        st.switch_page("app_pages/multi_stock.py")
else:
    st.caption("简报将组合股票池 EDA、聚类画像、当前股票模型信号和可靠性证据。")
    if st.button(
        "生成量化分析简报",
        type="primary",
        icon=":material/article:",
        key="single_to_report",
    ):
        st.switch_page("app_pages/report.py")
