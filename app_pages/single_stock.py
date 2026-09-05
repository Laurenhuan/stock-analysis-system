"""Single-stock workspace sharing one query across overview and models."""

from datetime import date

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


st.caption("一组股票与日期条件，复用于行情概览、决策树和线性回归。")

searched_symbols, searched_names = render_stock_search("single")
symbol_options = list(
    dict.fromkeys([*get_sample_symbols(), *searched_symbols])
)

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
        st.caption("X(t) → 次日上涨/非上涨；最早 80% 训练，最新 20% 测试，不打乱。")
        try:
            dashboard = run_classification_dashboard(
                market_data, symbol=active_symbol
            )
        except StockAnalysisError as error:
            st.error(f"决策树运行失败：{error}", icon=":material/error:")
        else:
            result = dashboard["result"]
            sample = dashboard["sample_summary"]
            with st.container(horizontal=True, gap="medium"):
                st.metric(
                    "Accuracy",
                    f"{result['metrics']['accuracy']:.2%}",
                    border=True,
                )
                st.metric("有效样本", sample["effective_rows"], border=True)
                st.metric("训练样本", sample["train_rows"], border=True)
                st.metric("测试样本", sample["test_rows"], border=True)
            st.caption(
                f"原始 {sample['input_rows']} 个交易日，滚动窗口和次日标签共剔除 "
                f"{sample['dropped_rows']} 行；训练区间 {sample['train_date_range']}，"
                f"测试区间 {sample['test_date_range']}。以上样本数量均为交易日数量，"
                "不是股票数量。"
            )
            st.info(
                "Accuracy 是测试集分类正确比例；它不表示未来收益，也应结合混淆矩阵"
                "判断模型是否只偏向某一类别。"
            )
            left, right = st.columns([1.4, 1])
            with left:
                st.plotly_chart(
                    dashboard["confusion_matrix_figure"], width="stretch"
                )
            with right:
                st.subheader("特征重要性")
                st.dataframe(
                    dashboard["feature_importance"],
                    hide_index=True,
                    column_config={
                        "importance": st.column_config.ProgressColumn(
                            "重要性", min_value=0.0, max_value=1.0
                        )
                    },
                )
            with st.expander("查看测试集分类结果"):
                st.dataframe(result["predictions"], hide_index=True)

if regression_tab.open:
    with regression_tab:
        st.caption("X(t) → 次日收益率；最早 80% 训练，最新 20% 测试，不打乱。")
        try:
            dashboard = run_regression_dashboard(
                market_data, symbol=active_symbol
            )
        except StockAnalysisError as error:
            st.error(f"线性回归运行失败：{error}", icon=":material/error:")
        else:
            result = dashboard["result"]
            sample = dashboard["sample_summary"]
            with st.container(horizontal=True, gap="medium"):
                st.metric("MAE", f"{result['metrics']['mae']:.6f}", border=True)
                st.metric("R²", f"{result['metrics']['r2']:.4f}", border=True)
                st.metric("训练样本", sample["train_rows"], border=True)
                st.metric("测试样本", sample["test_rows"], border=True)
            st.caption(
                f"原始 {sample['input_rows']} 个交易日，滚动窗口和次日目标共剔除 "
                f"{sample['dropped_rows']} 行，剩余 {sample['effective_rows']} 个有效样本；"
                f"训练区间 {sample['train_date_range']}，测试区间 "
                f"{sample['test_date_range']}。以上样本数量均为交易日数量，"
                "不是股票数量。"
            )
            st.info(
                "MAE 是测试集次日收益率预测的平均绝对误差，越小表示样本内误差越低；"
                "R² 衡量相对测试集均值基线的解释能力，可能为负，负值表示表现不如均值基线。"
            )
            st.plotly_chart(
                dashboard["actual_vs_predicted_figure"], width="stretch"
            )
            with st.expander("查看测试集预测结果"):
                st.dataframe(result["predictions"], hide_index=True)

st.warning(
    "所有结果仅描述所选历史区间，不代表未来表现，不构成投资建议。",
    icon=":material/warning:",
)
