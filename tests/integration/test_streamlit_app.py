"""Headless Streamlit tests for the D4 workspace navigation."""

from datetime import date
from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "app.py"
SAMPLE_START = date(2024, 1, 2)
SAMPLE_END = date(2024, 12, 31)


def _app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=30).run()


def test_home_renders_without_fetching_market_data() -> None:
    app = _app()

    assert not app.exception
    assert app.title[0].value == "项目首页"
    assert {button.label for button in app.button} >= {
        "进入单股研究",
        "进入多股比较",
    }


def test_single_workspace_waits_for_explicit_submit() -> None:
    app = _app()
    app.switch_page("app_pages/single_stock.py").run()

    assert not app.exception
    assert app.title[0].value == "单股研究"
    assert app.session_state["single_query"] is None
    assert app.info


def test_single_workspace_renders_one_shared_sample_query() -> None:
    app = _app()
    app.session_state["single_query"] = {
        "symbol": "600519.SH",
        "start_date": SAMPLE_START,
        "end_date": SAMPLE_END,
        "source": "sample",
    }
    app.switch_page("app_pages/single_stock.py").run()

    assert not app.exception
    assert not app.error
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["记录数"] == "242"
    assert metrics["股票数"] == "1"
    assert app.warning


def test_single_workspace_accepts_bare_code_and_shows_canonical_symbol() -> None:
    app = _app()
    app.session_state["single_query"] = {
        "symbol": "600519",
        "start_date": SAMPLE_START,
        "end_date": SAMPLE_END,
        "source": "sample",
    }
    app.switch_page("app_pages/single_stock.py").run()

    assert not app.exception
    assert not app.error
    assert any("600519.SH" in item.value for item in app.caption)


def test_single_workspace_surfaces_invalid_custom_symbol() -> None:
    app = _app()
    app.session_state["single_query"] = {
        "symbol": "600519.XY",
        "start_date": SAMPLE_START,
        "end_date": SAMPLE_END,
        "source": "sample",
    }
    app.switch_page("app_pages/single_stock.py").run()

    assert not app.exception
    assert app.error
    assert "无效证券代码" in app.error[0].value


def test_classification_tab_explains_test_samples() -> None:
    app = _app()
    app.session_state["single_query"] = {
        "symbol": "600519.SH",
        "start_date": SAMPLE_START,
        "end_date": SAMPLE_END,
        "source": "sample",
    }
    app.session_state["single_workspace_tabs"] = "决策树分类"
    app.switch_page("app_pages/single_stock.py").run()

    assert not app.exception
    metrics = {metric.label: metric.value for metric in app.metric}
    assert "Accuracy" in metrics
    assert "有效样本" in metrics
    assert "训练样本" in metrics
    assert "测试样本" in metrics
    assert any("不是股票数量" in item.value for item in app.caption)
    assert any("混淆矩阵" in item.value for item in app.info)


def test_regression_tab_explains_samples_and_negative_r2() -> None:
    app = _app()
    app.session_state["single_query"] = {
        "symbol": "600519.SH",
        "start_date": SAMPLE_START,
        "end_date": SAMPLE_END,
        "source": "sample",
    }
    app.session_state["single_workspace_tabs"] = "线性回归"
    app.switch_page("app_pages/single_stock.py").run()

    assert not app.exception
    metrics = {metric.label: metric.value for metric in app.metric}
    assert {"MAE", "R²", "训练样本", "测试样本"}.issubset(metrics)
    assert any("不是股票数量" in item.value for item in app.caption)
    assert any("可能为负" in item.value for item in app.info)


def test_multi_workspace_reuses_one_sample_query_for_eda() -> None:
    app = _app()
    app.session_state["multi_query"] = {
        "symbols": ["600519.SH", "000001.SZ", "000333.SZ"],
        "start_date": SAMPLE_START,
        "end_date": SAMPLE_END,
        "source": "sample",
        "correlation_method": "spearman",
    }
    app.switch_page("app_pages/multi_stock.py").run()

    assert not app.exception
    assert not app.error
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["有效股票"] == "3"
    assert metrics["总记录数"] == "726"
    assert app.selectbox(key="multi_candlestick_symbol").value == "600519.SH"


def test_clustering_tab_keeps_fixed_k_and_renders_dynamic_labels() -> None:
    app = _app()
    app.session_state["multi_query"] = {
        "symbols": ["600519.SH", "000001.SZ", "000333.SZ"],
        "start_date": SAMPLE_START,
        "end_date": SAMPLE_END,
        "source": "sample",
        "correlation_method": "spearman",
    }
    app.session_state["multi_workspace_tabs"] = "股票聚类"
    app.switch_page("app_pages/multi_stock.py").run()

    assert not app.exception
    assert not app.error
    assert any("固定 KMeans(k=3)" in item.value for item in app.caption)
    assert any("动态生成中文画像" in item.value for item in app.success)
    assert any(
        "cluster_label" in dataframe.value.columns
        for dataframe in app.dataframe
    )
