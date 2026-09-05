"""Headless Streamlit tests for the D4 workspace navigation."""

import tomllib
from datetime import date
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from src.services import get_sample_symbols, load_market_data


ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "app.py"
CONFIG = ROOT / ".streamlit" / "config.toml"
LAN_SCRIPT = ROOT / "scripts" / "run_lan.cmd"
SAMPLE_START = date(2024, 1, 2)
SAMPLE_END = date(2024, 12, 31)


def _app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=30).run()


def test_home_renders_without_fetching_market_data() -> None:
    app = _app()

    assert not app.exception
    assert app.title[0].value == "证券数据分析与决策参考平台"
    assert {button.label for button in app.button} >= {
        "开始多股研究",
        "直接进入单股研判",
    }
    assert any(
        item.value == "这个平台能帮你做什么？"
        for item in app.subheader
    )
    assert {link.label for link in app.get("page_link")} >= {
        "项目介绍",
        "查看量化分析简报",
    }


def test_about_page_is_registered_at_exact_html_path() -> None:
    assert 'url_path="about.html"' in APP.read_text(encoding="utf-8")

    app = _app()
    app.switch_page("app_pages/about.py").run()

    assert not app.exception
    assert app.title[0].value == "项目介绍"
    assert {item.value for item in app.subheader} >= {
        "面向使用者：从问题出发完成分析",
        "面向开发者与老师：项目如何工作",
        "团队协作与贡献概览",
    }
    assert any(
        item.proto.url == "https://gitee.com/sp1-2026/25151407"
        for item in app.get("link_button")
    )


def test_lan_delivery_uses_documented_address_and_port() -> None:
    config = tomllib.loads(CONFIG.read_text(encoding="utf-8"))
    script = LAN_SCRIPT.read_text(encoding="utf-8")

    assert "server" not in config
    assert "--server.address 0.0.0.0" in script
    assert "--server.port %STOCK_APP_PORT%" in script
    assert "--server.headless true" in script
    assert "about.html" in script


def test_single_workspace_waits_for_explicit_submit() -> None:
    app = _app()
    app.switch_page("app_pages/single_stock.py").run()

    assert not app.exception
    assert app.title[0].value == "单股模型分析"
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
    assert "决策树 Accuracy" in metrics
    assert {"训练集多数类基线", "当日方向延续基线"} & set(metrics)
    assert "决策树判断" in metrics
    assert "相对基线" in metrics
    assert "多窗口平均优势" in metrics
    assert any("不是股票数" in item.value for item in app.caption)
    assert any("混淆矩阵" in item.value for item in app.info)
    assert any(
        "历史方向可预测性检验" in item.value for item in app.subheader
    )


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
    assert {
        "模型 MAE",
        "相对基线误差改善",
        "R²",
        "预测收益率",
        "模型换算价格",
    }.issubset(metrics)
    assert {"零收益基线 MAE", "训练集均值基线 MAE"} & set(metrics)
    assert any("不是股票数" in item.value for item in app.caption)
    assert any("R²≤0" in item.value for item in app.info)
    assert any("次日收益线性关系检验" in item.value for item in app.subheader)


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
    assert any(
        "重新点击“加载并比较”" in item.value for item in app.caption
    )
    assert any(item.value == "先看结论" for item in app.subheader)
    assert {"收益表现", "风险画像", "股票关系", "趋势状态", "详细统计"}.issubset(
        {tab.label for tab in app.tabs}
    )


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
    labels = [
        str(value)
        for dataframe in app.dataframe
        if "cluster_label" in dataframe.value.columns
        for value in dataframe.value["cluster_label"].dropna()
    ]
    assert labels
    assert all("[所选历史区间]" not in label for label in labels)
    assert any("展示区间：2024-01-02 至 2024-12-31" in item.value for item in app.caption)

def test_clustering_date_mismatch_shows_actionable_guidance() -> None:
    app = _app()
    symbols = get_sample_symbols()[:3]
    market_data = load_market_data(symbols, source="sample")
    late_symbol = symbols[-1]
    late_start = (
        market_data.loc[
            market_data["symbol"] == late_symbol, "trade_date"
        ]
        .sort_values()
        .iloc[30]
    )
    market_data = market_data[
        (market_data["symbol"] != late_symbol)
        | (market_data["trade_date"] >= late_start)
    ].copy()
    app.session_state["multi_query"] = {
        "symbols": symbols,
        "start_date": SAMPLE_START,
        "end_date": SAMPLE_END,
        "source": "sample",
        "correlation_method": "spearman",
    }
    app.session_state["multi_workspace_tabs"] = "股票聚类"

    with patch(
        "app_pages.shared.cached_market_data",
        return_value=market_data,
    ):
        app.switch_page("app_pages/multi_stock.py").run()

    assert not app.exception
    assert any(
        "有效历史区间不一致" in item.value for item in app.error
    )
    assert any(
        late_start.date().isoformat() in item.value for item in app.info
    )
    guidance_tables = [
        table.value
        for table in app.dataframe
        if "限制原因" in table.value.columns
    ]
    assert len(guidance_tables) == 1
    assert guidance_tables[0]["股票代码"].tolist() == [late_symbol]
    assert guidance_tables[0]["限制原因"].tolist() == ["起始日期较晚"]


def test_report_page_combines_current_workflow_state() -> None:
    app = _app()
    app.session_state["multi_query"] = {
        "symbols": ["600519.SH", "000001.SZ", "000333.SZ"],
        "start_date": SAMPLE_START,
        "end_date": SAMPLE_END,
        "source": "sample",
        "correlation_method": "spearman",
    }
    app.session_state["focus_symbol"] = "600519.SH"
    app.switch_page("app_pages/report.py").run()

    assert not app.exception
    assert not app.error
    assert app.title[0].value == "量化分析简报"
    assert any("先看结论" in item.value for item in app.subheader)
    metrics = {metric.label: metric.value for metric in app.metric}
    assert "累计收益" in metrics
    assert "历史 Accuracy" in metrics
    assert "模型换算价格" in metrics
