"""Supervised-learning page with truthful per-algorithm readiness."""

import streamlit as st

from src.services import (
    get_analysis_status,
    get_sample_date_bounds,
    get_sample_symbols,
    load_market_data,
    run_regression_dashboard,
)
from src.utils.exceptions import StockAnalysisError


st.set_page_config(page_title="监督学习", page_icon="🧠", layout="wide")
st.title("🧠 监督学习")
st.caption("统一采用 X(t) → y(t+1)、最早 80% 训练/最新 20% 测试、不打乱。")

statuses = get_analysis_status()
classification_tab, regression_tab = st.tabs(("Decision Tree 分类", "Linear Regression 回归"))

with classification_tab:
    if statuses["classification"] != "ready":
        st.info(
            "Role 4 Decision Tree 分类尚未合并到当前分支。"
            "页面不使用占位指标或伪造预测结果。"
        )
        st.markdown("待接入输出：Accuracy、2×2 Confusion Matrix 与按日期对齐的预测表。")

with regression_tab:
    st.success("Role 6 Linear Regression 已接入。")
    symbols = get_sample_symbols()
    first_date, last_date = get_sample_date_bounds()
    selected_symbol = st.selectbox("股票代码", options=symbols)
    st.caption(f"离线样例区间：{first_date} 至 {last_date}")
    run_regression = st.button("运行线性回归", type="primary")

    if run_regression:
        try:
            market_data = load_market_data(
                selected_symbol,
                start_date=first_date,
                end_date=last_date,
                source="sample",
            )
            dashboard = run_regression_dashboard(
                market_data, symbol=selected_symbol
            )
        except StockAnalysisError as error:
            st.error(str(error))
        else:
            result = dashboard["result"]
            metrics = result["metrics"]
            mae_col, r2_col, sample_col = st.columns(3)
            mae_col.metric("MAE", f"{metrics['mae']:.6f}")
            r2_col.metric("R²", f"{metrics['r2']:.4f}")
            sample_col.metric("测试样本", len(result["predictions"]))

            st.plotly_chart(
                dashboard["actual_vs_predicted_figure"], width="stretch"
            )
            st.subheader("按交易日对齐的测试集预测")
            st.dataframe(result["predictions"], width="stretch", hide_index=True)
            st.caption(f"模型特征：{', '.join(result['feature_names'])}")

st.warning("模型结果仅用于方法演示，不构成投资建议。")
