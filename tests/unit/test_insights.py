"""Unit tests for src.analysis.insights (Role 3).

Coverage: structured insight shape, multi-stock ranking, single stock, ties,
all-NaN / insufficient samples, unformed moving averages, correlation edge
cases, determinism under row shuffling, value consistency with the underlying
EDA tables, input immutability and the absence of investment-advice language.
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis.eda import returns_comparison
from src.analysis.insights import (
    CATEGORY_CORRELATION,
    CATEGORY_DATA_QUALITY,
    CATEGORY_DISTRIBUTION,
    CATEGORY_PERFORMANCE,
    CATEGORY_RISK,
    CATEGORY_TREND,
    DISCLAIMER,
    build_eda_insights,
)
from src.utils.exceptions import DataValidationError, NoDataError

# 组长整改要求的禁用词：结论不得出现任何买卖/推荐/持有措辞。
FORBIDDEN = [
    "建议买入", "未来会上涨", "适合长期持有", "低风险高收益",
    "推荐配置", "买入", "卖出", "推荐", "持有",
]


def _make_df(closes_by_symbol):
    """Build a Contract-shaped DataFrame; ma5/ma20 are NaN until formed."""
    rows = []
    for sym, closes in closes_by_symbol.items():
        dates = pd.date_range("2024-01-01", periods=len(closes))
        for i, c in enumerate(closes):
            prev = closes[i - 1] if i > 0 else None
            rows.append(
                {
                    "symbol": sym,
                    "trade_date": dates[i],
                    "open": c * 0.99,
                    "high": c * 1.01,
                    "low": c * 0.98,
                    "close": c,
                    "volume": 1000.0 + i,
                    "amount": c * 1000.0,
                    "return": c / prev - 1 if prev else np.nan,
                    "drawdown": c / max(closes[: i + 1]) - 1,
                    "ma5": np.nan,
                    "ma20": np.nan,
                }
            )
    df = pd.DataFrame(rows)
    parts = []
    for sym, g in df.groupby("symbol", sort=False):
        g = g.sort_values("trade_date").copy()
        g["ma5"] = g["close"].rolling(5).mean()
        g["ma20"] = g["close"].rolling(20).mean()
        parts.append(g)
    return pd.concat(parts).reset_index(drop=True)


def _bare_row(symbol, date, ret, close):
    return {
        "symbol": symbol, "trade_date": date,
        "open": close * 0.99, "high": close * 1.01, "low": close * 0.98,
        "close": close, "volume": 1000.0, "amount": close * 1000.0,
        "return": ret, "drawdown": 0.0, "ma5": np.nan, "ma20": np.nan,
    }


@pytest.fixture
def multi_df():
    return _make_df(
        {
            "A": [100.0, 110.0, 105.0, 120.0, 115.0],  # 累计收益 15.0%
            "B": [10.0, 9.0, 9.5, 10.0, 10.5],        # 累计收益 5.0%
            "C": [20.0, 22.0, 19.0, 25.0, 24.0],      # 累计收益 20.0%
        }
    )


@pytest.fixture
def single_df(multi_df):
    return multi_df[multi_df["symbol"] == "A"].reset_index(drop=True)


def _titles(insights):
    return [i["title"] for i in insights]


def _by_title(insights, title):
    return next((i for i in insights if i["title"] == title), None)


def _all_text(insights):
    return "".join(v for i in insights for v in i.values())


# --- shape / structure -------------------------------------------------------

def test_returns_list_of_structured_insights(multi_df):
    insights = build_eda_insights(multi_df)
    assert isinstance(insights, list)
    assert insights
    for i in insights:
        assert set(i.keys()) == {
            "category", "title", "finding", "evidence", "interpretation", "caveat",
        }
        assert i["category"] in {
            CATEGORY_PERFORMANCE, CATEGORY_RISK, CATEGORY_TREND,
            CATEGORY_CORRELATION, CATEGORY_DATA_QUALITY, CATEGORY_DISTRIBUTION,
        }
        assert i["caveat"].endswith(DISCLAIMER)


def test_every_insight_carries_disclaimer(multi_df):
    insights = build_eda_insights(multi_df)
    assert insights
    for i in insights:
        assert DISCLAIMER in i["caveat"]


# --- ranking -----------------------------------------------------------------

def test_multi_symbol_ranking_names_extreme_symbols(multi_df):
    insights = build_eda_insights(multi_df)
    top = _by_title(insights, "累计收益最高")
    bottom = _by_title(insights, "累计收益最低")
    assert top is not None and bottom is not None
    assert "C" in top["finding"]
    assert "B" in bottom["finding"]


def test_values_match_underlying_stats(multi_df):
    rc = returns_comparison(multi_df).set_index("symbol")
    top_symbol = rc["cumulative_return"].idxmax()
    top_value = rc.loc[top_symbol, "cumulative_return"]
    insights = build_eda_insights(multi_df)
    top = _by_title(insights, "累计收益最高")
    assert top_symbol in top["finding"]
    assert f"{top_value:.1%}" in top["finding"]


def test_single_stock_uses_overview_not_ranking(single_df):
    insights = build_eda_insights(single_df)
    titles = _titles(insights)
    assert "收益概览" in titles
    assert "风险概览" in titles
    assert "累计收益最高" not in titles
    assert "累计收益最低" not in titles


# --- ties --------------------------------------------------------------------

def test_tied_extremes_name_all_symbols():
    # 两只完全一致的股票 → 最高与最低同时并列。
    df = _make_df({"X": [100.0, 110.0, 105.0], "Y": [100.0, 110.0, 105.0]})
    insights = build_eda_insights(df)
    top = _by_title(insights, "累计收益最高")
    assert top is not None
    assert "X" in top["finding"] and "Y" in top["finding"]


# --- all-NaN / insufficient samples -----------------------------------------

def test_all_nan_returns_reports_insufficient(multi_df):
    df = multi_df.copy()
    df["return"] = np.nan
    insights = build_eda_insights(df)
    titles = _titles(insights)
    assert "样本不足" in titles
    assert "累计收益最高" not in titles  # 不产出误导性排名


def test_single_stock_all_nan_no_overview(single_df):
    df = single_df.copy()
    df["return"] = np.nan
    insights = build_eda_insights(df)
    titles = _titles(insights)
    assert "收益概览" not in titles
    assert "风险概览" not in titles
    assert "收益数据不足" in titles or "样本不足" in titles


def test_all_nan_returns_no_nan_percent(multi_df):
    # 组长复审：多股 return 全 NaN 时，结论不得出现 nan%。
    df = multi_df.copy()
    df["return"] = np.nan
    insights = build_eda_insights(df)
    text = _all_text(insights)
    assert "nan%" not in text
    assert "收益与风险样本不足" in _titles(insights)


def test_three_stocks_two_valid_no_spearman():
    # 组长复审：3 只股票中仅 2 只有效时，不得生成「收益与风险关系」。
    df = _make_df(
        {
            "A": [100.0, 110.0, 105.0, 120.0, 115.0],
            "B": [10.0, 9.0, 9.5, 10.0, 10.5],
            "C": [20.0, 22.0, 19.0, 25.0, 24.0],
        }
    )
    df.loc[df["symbol"] == "C", "return"] = np.nan
    insights = build_eda_insights(df)
    assert "收益与风险关系" not in _titles(insights)


# --- trend / moving averages -------------------------------------------------

def test_unformed_moving_averages_noted_in_caveat():
    df = _make_df({"A": [100.0, 110.0, 105.0]})  # 3 天，MA5/MA20 均未形成
    insights = build_eda_insights(df)
    trend = _by_title(insights, "趋势：A")
    assert trend is not None
    assert "尚未形成" in trend["caveat"]


# --- correlation -------------------------------------------------------------

def test_single_stock_correlation_unavailable(single_df):
    insights = build_eda_insights(single_df)
    corr = _by_title(insights, "无法计算相关性")
    assert corr is not None
    assert corr["category"] == CATEGORY_CORRELATION


def test_correlation_tie_reports_all_equal():
    # 20 个非恒定有效重叠收益，三只股票同序递增 → Spearman 均为 1.0。
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    df = pd.DataFrame(
        [
            {
                "symbol": symbol,
                "trade_date": date,
                "return": float(index * scale),
            }
            for index, date in enumerate(dates)
            for symbol, scale in (("A", 1.0), ("B", 2.0), ("C", 3.0))
        ]
    )
    insights = build_eda_insights(df)
    corr = _by_title(insights, "相关性")
    assert corr is not None
    assert "均为" in corr["finding"]
    assert "20 个重叠交易日" in corr["finding"]


def test_low_overlap_correlation_is_not_ranked():
    df = _make_df(
        {
            "A": [100.0, 101.0, 103.0, 108.0],
            "B": [10.0, 10.5, 11.5, 15.0],
        }
    )

    insights = build_eda_insights(df)
    corr = _by_title(insights, "相关性样本较少")

    assert corr is not None
    assert "不进行最高/最低排名" in corr["finding"]
    assert "3 个重叠交易日" in corr["evidence"]


def test_insufficient_correlation_overlap_is_graceful():
    df = pd.DataFrame(
        [
            _bare_row("A", pd.Timestamp("2024-01-02"), 0.01, 100.0),
            _bare_row("A", pd.Timestamp("2024-01-03"), -0.01, 99.0),
            _bare_row("B", pd.Timestamp("2024-01-02"), 0.02, 50.0),
            _bare_row("B", pd.Timestamp("2024-06-01"), 0.03, 51.0),
        ]
    )
    insights = build_eda_insights(df)  # 不应抛出 InsufficientDataError
    assert _by_title(insights, "相关性样本不足") is not None


# --- determinism / immutability ---------------------------------------------

def test_shuffled_row_order_yields_identical_output(multi_df):
    before = build_eda_insights(multi_df)
    shuffled = multi_df.sample(frac=1, random_state=42).reset_index(drop=True)
    after = build_eda_insights(shuffled)
    assert before == after


def test_does_not_mutate_input(multi_df):
    original = multi_df.copy(deep=True)
    build_eda_insights(multi_df)
    pd.testing.assert_frame_equal(multi_df, original)


# --- forbidden language ------------------------------------------------------

def test_no_investment_advice_words(multi_df, single_df):
    all_nan = multi_df.copy()
    all_nan["return"] = np.nan
    for df in (multi_df, single_df, all_nan):
        text = _all_text(build_eda_insights(df))
        for word in FORBIDDEN:
            assert word not in text, f"出现禁用词：{word}"


def test_comparison_wording_uses_than_not_out(multi_df):
    # 组长要求：「低出……个百分点」改为「低于……个百分点」；对应「高出」一并改为「高于」。
    text = _all_text(build_eda_insights(multi_df))
    assert "低出" not in text
    assert "高出" not in text


# --- validation --------------------------------------------------------------

def test_invalid_correlation_method(multi_df):
    with pytest.raises(DataValidationError):
        build_eda_insights(multi_df, correlation_method="bogus")


def test_missing_required_columns(multi_df):
    with pytest.raises(DataValidationError):
        build_eda_insights(multi_df.drop(columns=["symbol"]))
    with pytest.raises(DataValidationError):
        build_eda_insights(multi_df.drop(columns=["trade_date"]))


def test_empty_input():
    with pytest.raises(NoDataError):
        build_eda_insights(pd.DataFrame())


# --- missing feature columns degrade gracefully ------------------------------

def test_missing_close_skips_trend_and_flags_quality(multi_df):
    df = multi_df.drop(columns=["close"])
    insights = build_eda_insights(df)
    titles = _titles(insights)
    assert "缺少关键字段" in titles
    assert not any(t.startswith("趋势：") for t in titles)


# --- dynamic scale / arbitrary dates / partial validity ----------------------

@pytest.mark.parametrize("n", [2, 3, 8, 15, 20])
def test_variable_symbol_count_is_deterministic(n):
    closes = {f"STK{i:02d}": [100.0 + i + 5 * d for d in range(6)] for i in range(n)}
    df = _make_df(closes)
    insights = build_eda_insights(df)
    assert insights
    shuffled = df.sample(frac=1, random_state=0).reset_index(drop=True)
    assert insights == build_eda_insights(shuffled)


def test_arbitrary_date_range_and_symbols():
    # 非 2024 年、非固定演示股票：区间与股票身份均来自输入。
    base = pd.Timestamp("2021-07-05")
    closes = {"AAA": [50.0, 52.0, 51.0, 53.0], "BBB": [30.0, 31.0, 30.5, 32.0]}
    rows = []
    dates = pd.date_range(base, periods=4)
    for sym, cs in closes.items():
        for i, c in enumerate(cs):
            prev = cs[i - 1] if i > 0 else None
            rows.append({
                "symbol": sym, "trade_date": dates[i],
                "open": c * 0.99, "high": c * 1.01, "low": c * 0.98, "close": c,
                "volume": 1000.0, "amount": c * 1000.0,
                "return": c / prev - 1 if prev else np.nan,
                "drawdown": c / max(cs[: i + 1]) - 1,
                "ma5": np.nan, "ma20": np.nan,
            })
    df = pd.DataFrame(rows)
    text = _all_text(build_eda_insights(df))
    assert "2021-07-05" in text
    assert "2024" not in text
    assert "AAA" in text and "BBB" in text
    assert "600519" not in text and "000001" not in text


def test_two_stocks_one_all_nan_gives_overview_not_ranking():
    # 2 只股票只有 1 只有效收益 → 输出「收益概览」而非把同一只排成最高又最低。
    df = _make_df({"A": [100.0, 110.0, 105.0], "B": [10.0, 9.0, 9.5]})
    df.loc[df["symbol"] == "B", "return"] = np.nan
    df.loc[df["symbol"] == "B", "drawdown"] = np.nan
    insights = build_eda_insights(df)
    titles = _titles(insights)
    assert "收益概览" in titles
    assert "累计收益最高" not in titles
    assert "累计收益最低" not in titles
    assert "nan%" not in _all_text(insights)


def test_partial_all_nan_stocks_not_ranked():
    # 部分股票全 NaN：排名只在有效股票之间进行，且不出现 nan%。
    df = _make_df({
        "A": [100.0, 110.0, 105.0, 120.0, 115.0],
        "B": [10.0, 9.0, 9.5, 10.0, 10.5],
        "C": [20.0, 22.0, 19.0, 25.0, 24.0],
    })
    df.loc[df["symbol"] == "C", "return"] = np.nan
    df.loc[df["symbol"] == "C", "drawdown"] = np.nan
    insights = build_eda_insights(df)
    text = _all_text(insights)
    assert "nan%" not in text
    for title in ("累计收益最高", "累计收益最低", "波动最大", "波动最小",
                  "回撤最深", "回撤最浅"):
        ins = _by_title(insights, title)
        if ins is not None:
            assert "C" not in ins["finding"], f"{title} 不应把全 NaN 的 C 排入"


def test_partial_moving_average_formation():
    # A 25 天（均线均形成），B 仅 3 天（均未形成）。
    df = _make_df({"A": [100.0 + i for i in range(25)], "B": [10.0, 11.0, 12.0]})
    insights = build_eda_insights(df)
    trend_a = _by_title(insights, "趋势：A")
    trend_b = _by_title(insights, "趋势：B")
    assert trend_a is not None and trend_b is not None
    assert "尚未形成" in trend_b["caveat"]
    assert "尚未形成" not in trend_a["caveat"]


def test_no_common_trading_days_graceful():
    df = pd.DataFrame([
        _bare_row("A", pd.Timestamp("2022-01-03"), 0.01, 100.0),
        _bare_row("A", pd.Timestamp("2022-01-04"), -0.01, 99.0),
        _bare_row("B", pd.Timestamp("2022-03-01"), 0.02, 50.0),
        _bare_row("B", pd.Timestamp("2022-03-02"), 0.03, 51.0),
    ])
    insights = build_eda_insights(df)
    assert _by_title(insights, "相关性样本不足") is not None


# --- distribution / extreme moves / volatility regime ------------------------

def test_distribution_insight_names_highest_kurtosis(multi_df):
    insights = build_eda_insights(multi_df)
    assert _by_title(insights, "超额峰度最高") is not None
    assert _by_title(insights, "偏度最明显") is not None


def test_single_stock_distribution_uses_overview(single_df):
    insights = build_eda_insights(single_df)
    titles = _titles(insights)
    assert "收益分布形态" in titles
    assert "超额峰度最高" not in titles
    assert "偏度最明显" not in titles


def test_extreme_insights_name_biggest_day(multi_df):
    insights = build_eda_insights(multi_df)
    up = _by_title(insights, "最大单日收益")
    down = _by_title(insights, "最低单日收益")
    assert up is not None and down is not None
    assert up["category"] == CATEGORY_RISK
    assert "nan%" not in _all_text(insights)


def test_all_nan_returns_distribution_and_extreme_graceful(multi_df):
    df = multi_df.copy()
    df["return"] = np.nan
    insights = build_eda_insights(df)
    titles = _titles(insights)
    assert "分布数据不足" in titles
    assert "极端涨跌数据不足" in titles
    assert "nan%" not in _all_text(insights)


def test_volatility_regime_skipped_without_column(multi_df):
    insights = build_eda_insights(multi_df)
    assert _by_title(insights, "当前波动率水平") is None


def test_volatility_regime_present_with_column():
    base = pd.Timestamp("2024-01-01")
    rows = []
    dates = pd.date_range(base, periods=3)
    for sym, vols in {"A": [0.10, 0.11, 0.12], "B": [0.20, 0.18, 0.16]}.items():
        for i, v in enumerate(vols):
            rows.append({
                "symbol": sym, "trade_date": dates[i],
                "return": 0.01,
                "volatility_20d": v,
            })
    insights = build_eda_insights(pd.DataFrame(rows))
    vol = _by_title(insights, "当前波动率水平")
    assert vol is not None
    assert "A" in vol["finding"] and "B" in vol["finding"]


def test_volatility_regime_deterministic_under_shuffle():
    base = pd.Timestamp("2024-01-01")
    offset = {"A": 0.0, "B": 0.005, "C": -0.003}
    rows = []
    dates = pd.date_range(base, periods=6)
    for sym in ("A", "B", "C"):
        for i in range(6):
            rows.append({
                "symbol": sym,
                "trade_date": dates[i],
                "return": 0.01 * (i % 3) + offset[sym],
                "volatility_20d": 0.1 + 0.01 * i * (1 if sym == "C" else -1),
            })
    df = pd.DataFrame(rows)
    before = build_eda_insights(df)
    shuffled = df.sample(frac=1, random_state=7).reset_index(drop=True)
    assert before == build_eda_insights(shuffled)


# --- regression: review wording fixes ---------------------------------------

def test_all_negative_kurtosis_not_called_fat_tail():
    # 所有股票超额峰度均为负 → 「超额峰度最高」不得固定解释成「厚尾/高于正态的 0」。
    base = pd.Timestamp("2024-01-01")
    rows = []
    dates = pd.date_range(base, periods=5)
    for sym, rets in {"A": [0.01, 0.02, 0.03, 0.04, 0.05],
                      "B": [0.02, 0.04, 0.06, 0.08, 0.10]}.items():
        for i, r in enumerate(rets):
            rows.append({"symbol": sym, "trade_date": dates[i],
                         "return": r})
    insights = build_eda_insights(pd.DataFrame(rows))
    fat = _by_title(insights, "超额峰度最高")
    assert fat is not None
    assert "厚尾" not in fat["finding"]
    assert "高于正态的 0" not in fat["finding"]


def test_volatility_falling_from_high_still_above_median_not_called_rising():
    # 波动率从高位回落、但最新值仍高于历史中位数 → 应称「相对高波动阶段」，而非「上升」。
    base = pd.Timestamp("2024-01-01")
    vols = [0.10, 0.15, 0.20, 0.90, 0.60]
    dates = pd.date_range(base, periods=len(vols))
    rows = [
        {"symbol": "A", "trade_date": dates[i],
         "return": 0.01, "volatility_20d": v}
        for i, v in enumerate(vols)
    ]
    insights = build_eda_insights(pd.DataFrame(rows))
    vol = _by_title(insights, "当前波动率水平")
    assert vol is not None
    assert "相对高波动阶段" in vol["finding"]
    assert "上升" not in vol["finding"]
    assert "下降" not in vol["finding"]


def test_all_positive_returns_min_not_called_drop():
    # 所有有效日收益均为正 → 「最低单日收益」不得解释为「最大单日跌幅」。
    base = pd.Timestamp("2024-01-01")
    dates = pd.date_range(base, periods=4)
    rows = [
        {"symbol": "A", "trade_date": dates[i], "return": r}
        for i, r in enumerate([0.01, 0.02, 0.015, 0.03])
    ]
    insights = build_eda_insights(pd.DataFrame(rows))
    low = _by_title(insights, "最低单日收益")
    assert low is not None
    assert "跌幅" not in low["interpretation"]
    assert "不存在单日下跌日" in low["interpretation"]
