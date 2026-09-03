"""Problem-driven EDA conclusions owned by Role 3 (金融数据分析与可视化工程师).

Whereas :mod:`src.analysis.eda` returns reusable *statistics tables*, this module
composes those tables into *findings*: structured, evidence-backed conclusions
that tell a reader what the data actually shows — the leading / lagging
performer, the riskiest stock, the tightest correlation, the current trend state
and any data-quality issues.

Every insight is an :class:`EdaInsight` dict with a ``category``, a short
``title``, the ``finding``, its ``evidence``, an ``interpretation`` and a
``caveat``. All values (symbols, ranks, figures) are computed dynamically from
the input — nothing is hard-coded — and every group of conclusions carries a
disclaimer that it describes only the selected historical window and is not
investment advice. This module never produces buy / sell / hold language.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import pandas as pd
from pandas import DataFrame

from src.analysis.eda import (
    correlation_matrix,
    date_range_summary,
    returns_comparison,
    risk_return_summary,
)
from src.utils.exceptions import DataValidationError, InsufficientDataError, NoDataError

# Insight categories (used by Role 1 to group the finding cards).
CATEGORY_PERFORMANCE = "performance"
CATEGORY_RISK = "risk"
CATEGORY_TREND = "trend"
CATEGORY_CORRELATION = "correlation"
CATEGORY_DATA_QUALITY = "data_quality"

# Mandatory caveat stamped on every group of conclusions.
DISCLAIMER = "仅描述所选历史区间，不代表未来表现，不构成投资建议。"

# 相关系数方法白名单（模块内自持，不复用 eda 的私有 _CORR_METHODS）。
_CORR_METHODS = ("pearson", "spearman", "kendall")


class EdaInsight(TypedDict):
    category: str
    title: str
    finding: str
    evidence: str
    interpretation: str
    caveat: str


def _pct(value: float) -> str:
    """Format a ratio as a percentage, e.g. 0.234 -> '23.4%'."""
    return f"{value:.1%}"


def _pp(value: float) -> str:
    """Format a ratio difference as percentage points, e.g. 0.082 -> '8.2 个百分点'."""
    return f"{value * 100:.1f} 个百分点"


def _corr(value: float) -> str:
    return f"{value:.2f}"


def _insight(category, title, finding, evidence, interpretation, caveat=DISCLAIMER):
    return {
        "category": category,
        "title": title,
        "finding": finding,
        "evidence": evidence,
        "interpretation": interpretation,
        "caveat": caveat,
    }


def _require_columns(df: DataFrame, required: tuple[str, ...], *, label: str) -> None:
    """模块内输入校验：空输入抛 NoDataError，缺列抛 DataValidationError。

    不复用 ``eda._require_columns``（私有实现），保持本模块自包含。
    """
    if not isinstance(df, DataFrame):
        raise DataValidationError(
            f"{label} 需要 pandas.DataFrame，收到 {type(df).__name__}"
        )
    if df.empty:
        raise NoDataError(f"{label} 输入数据为空")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError(f"{label} 缺少必需字段：{missing}")


def _top_by(df: DataFrame, column: str):
    """Return ``(symbols, value)`` of the maximum of ``column`` (ties included)."""
    valid = df.dropna(subset=[column])
    if valid.empty:
        return None
    value = valid[column].max()
    symbols = sorted(valid.loc[valid[column] == value, "symbol"].tolist())
    return "、".join(symbols), float(value)


def _bottom_by(df: DataFrame, column: str):
    """Return ``(symbols, value)`` of the minimum of ``column`` (ties included)."""
    valid = df.dropna(subset=[column])
    if valid.empty:
        return None
    value = valid[column].min()
    symbols = sorted(valid.loc[valid[column] == value, "symbol"].tolist())
    return "、".join(symbols), float(value)


def _compare_median(value: float, median: float, *, higher: bool) -> str:
    diff = value - median if higher else median - value
    verb = "高于" if higher else "低于"
    return f"同组中位数为 {_pct(median)}，{verb} {_pp(diff)}。"


def _rank_insight(category, title_top, title_bottom, df, column, *, fmt, metric_label,
                  higher_interp, lower_interp):
    """Emit a 'top' and a 'bottom' ranking insight for one numeric column."""
    top = _top_by(df, column)
    bottom = _bottom_by(df, column)
    if top is None or bottom is None:
        return []
    median = df[column].median()
    return [
        _insight(category, title_top,
                 f"{top[0]} {metric_label}最高，为 {fmt(top[1])}。",
                 _compare_median(top[1], median, higher=True),
                 higher_interp, DISCLAIMER),
        _insight(category, title_bottom,
                 f"{bottom[0]} {metric_label}最低，为 {fmt(bottom[1])}。",
                 _compare_median(bottom[1], median, higher=False),
                 lower_interp, DISCLAIMER),
    ]


def build_eda_insights(
    data: DataFrame, *, correlation_method: str = "spearman"
) -> list[EdaInsight]:
    """Compose the EDA statistics tables into structured, problem-driven insights.

    Requires a Contract-compliant market DataFrame carrying at least ``symbol``
    and ``trade_date``. When richer feature columns are present (``return``,
    ``drawdown``, ``close``, ``ma5``, ``ma20``), the corresponding conclusions
    are generated; otherwise the missing columns are reported as data-quality
    insights instead of raising.

    The result is deterministic: input rows are sorted by ``(symbol,
    trade_date)`` and the input DataFrame is never modified.
    """
    _require_columns(data, ("symbol", "trade_date"), label="build_eda_insights")
    if correlation_method not in _CORR_METHODS:
        raise DataValidationError(
            f"build_eda_insights 不支持的相关系数方法：{correlation_method!r}，"
            f"可选 {_CORR_METHODS}"
        )
    # Deterministic ordering; never mutate the caller's DataFrame.
    data = data.sort_values(["symbol", "trade_date"]).reset_index(drop=True)

    has_return = "return" in data.columns
    has_drawdown = "drawdown" in data.columns

    rc = returns_comparison(data) if has_return else None
    rr = risk_return_summary(data) if (has_return and has_drawdown) else None

    insights: list[EdaInsight] = []
    insights.extend(_data_quality_insights(data, rc))
    insights.extend(_performance_insights(data, rc))
    insights.extend(_risk_insights(rc, rr))
    insights.extend(_correlation_insights(data, correlation_method))
    insights.extend(_trend_insights(data, rc))
    return insights


# --- data quality ------------------------------------------------------------

def _data_quality_insights(data: DataFrame, rc) -> list[EdaInsight]:
    insights: list[EdaInsight] = []

    n_symbols = data["symbol"].nunique()
    n_dates = data["trade_date"].nunique()
    insights.append(_insight(
        CATEGORY_DATA_QUALITY, "有效交易日数量",
        f"共 {n_dates} 个有效交易日，覆盖 {n_symbols} 只股票。",
        "以输入数据中去重后的交易日数量为准。",
        "以上为本次分析的样本规模。",
        DISCLAIMER,
    ))

    needed = {
        "return": "收益/风险/相关性结论",
        "drawdown": "回撤结论",
        "close": "趋势结论",
        "ma5": "趋势结论（MA5 短期均线）",
        "ma20": "趋势结论（MA20 长期均线）",
    }
    missing = [c for c in needed if c not in data.columns]
    if missing:
        detail = "；".join(f"缺少 {c}（影响{needed[c]}）" for c in missing)
        insights.append(_insight(
            CATEGORY_DATA_QUALITY, "缺少关键字段",
            f"输入数据{detail}。",
            "缺少的字段会跳过对应结论，仅保留数据质量说明。",
            "需 Role 2 的 build_common_features 补齐契约字段后重算。",
            DISCLAIMER,
        ))
    else:
        insights.append(_insight(
            CATEGORY_DATA_QUALITY, "字段完整",
            "分析所需的契约字段均已具备。",
            "return / drawdown / close / ma5 / ma20 均存在。",
            "可据此生成完整的收益、风险、趋势与相关性结论。",
            DISCLAIMER,
        ))

    rolling = [c for c in ("ma5", "ma20", "volatility_20d") if c in data.columns]
    if rolling:
        insights.append(_insight(
            CATEGORY_DATA_QUALITY, "滚动窗口前置 NaN",
            f"{'、'.join(rolling)} 开头的 NaN 属于滚动窗口尚未成熟，属正常现象，不计为数据缺失。",
            "例如 MA20 需累计 20 个交易日才产生第一个有效值。",
            "这些前置空值是滚动指标的自然产物，不影响其后产生的有效值。",
            DISCLAIMER,
        ))

    dr = date_range_summary(data)
    start = data["trade_date"].min()
    end = data["trade_date"].max()
    if dr["start_date"].nunique() > 1 or dr["end_date"].nunique() > 1:
        odd = dr[(dr["start_date"] != start) | (dr["end_date"] != end)]
        detail = "；".join(
            f"{r['symbol']}（{r['start_date'].date()}~{r['end_date'].date()}）"
            for _, r in odd.iterrows()
        )
        insights.append(_insight(
            CATEGORY_DATA_QUALITY, "日期范围不一致",
            f"以下股票的交易日范围与整体区间不一致：{detail}。",
            "整体区间取各股票日期范围的并集。",
            "日期范围不一致可能影响横向对比的公平性。",
            DISCLAIMER,
        ))

    if rc is not None:
        valid = data["return"].notna().groupby(data["symbol"]).sum().astype(int)
        short = valid[valid < 2]
        if not short.empty:
            detail = "；".join(f"{s}（仅 {int(c)} 个有效收益日）" for s, c in short.items())
            insights.append(_insight(
                CATEGORY_DATA_QUALITY, "样本不足",
                f"以下股票有效收益日不足 2 天，无法可靠计算收益统计：{detail}。",
                "样本过少时对应的收益/波动结论已跳过。",
                "样本不足的股票不纳入排名，以免给出误导性结论。",
                DISCLAIMER,
            ))
    return insights


# --- performance -------------------------------------------------------------

def _performance_insights(data: DataFrame, rc) -> list[EdaInsight]:
    if rc is None:
        return []
    insights: list[EdaInsight] = []

    start = data["trade_date"].min()
    end = data["trade_date"].max()
    insights.append(_insight(
        CATEGORY_PERFORMANCE, "分析区间",
        f"所选区间为 {start.date()} 至 {end.date()}，覆盖 {len(rc)} 只股票。",
        "区间起止以输入数据中的最早和最晚交易日为准。",
        "以上为本次分析的时间范围。",
        DISCLAIMER,
    ))

    if len(rc) == 1:
        row = rc.iloc[0]
        if pd.isna(row["cumulative_return"]):
            insights.append(_insight(
                CATEGORY_PERFORMANCE, "收益数据不足",
                f"{row['symbol']} 有效收益日不足，无法生成收益概览。",
                "—",
                "样本不足的股票不生成收益结论，以免误导。",
                DISCLAIMER,
            ))
            return insights
        insights.append(_insight(
            CATEGORY_PERFORMANCE, "收益概览",
            f"{row['symbol']} 区间累计收益为 {_pct(row['cumulative_return'])}，"
            f"平均日收益 {_pct(row['mean_return'])}，上涨日占比 {_pct(row['win_rate'])}。",
            "仅 1 只股票，无横向对比。",
            "该股票在所选区间的收益表现如上。",
            DISCLAIMER,
        ))
        return insights

    insights.extend(_rank_insight(
        CATEGORY_PERFORMANCE, "累计收益最高", "累计收益最低",
        rc, "cumulative_return", fmt=_pct, metric_label="区间累计收益",
        higher_interp="该股票在所选区间相对表现更强。",
        lower_interp="该股票在所选区间相对表现较弱。",
    ))
    insights.extend(_rank_insight(
        CATEGORY_PERFORMANCE, "平均日收益最高", "平均日收益最低",
        rc, "mean_return", fmt=_pct, metric_label="平均日收益",
        higher_interp="该股票平均每个交易日的收益相对更高。",
        lower_interp="该股票平均每个交易日的收益相对更低。",
    ))
    top_wr = _top_by(rc, "win_rate")
    if top_wr is not None:
        median = rc["win_rate"].median()
        insights.append(_insight(
            CATEGORY_PERFORMANCE, "上涨日占比最高",
            f"{top_wr[0]} 上涨日占比最高，为 {_pct(top_wr[1])}。",
            _compare_median(top_wr[1], median, higher=True),
            "该股票在所选区间内上涨的交易日占比相对最高。",
            DISCLAIMER,
        ))
    return insights


# --- risk --------------------------------------------------------------------

def _risk_insights(rc, rr) -> list[EdaInsight]:
    if rr is None:
        return []
    insights: list[EdaInsight] = []

    if len(rr) == 1:
        row = rr.iloc[0]
        if pd.isna(row["volatility"]):
            insights.append(_insight(
                CATEGORY_RISK, "风险数据不足",
                f"{row['symbol']} 有效收益日不足，无法生成风险概览。",
                "—",
                "样本不足的股票不生成风险结论，以免误导。",
                DISCLAIMER,
            ))
            return insights
        insights.append(_insight(
            CATEGORY_RISK, "风险概览",
            f"{row['symbol']} 日收益标准差为 {_pct(row['volatility'])}（未年化），"
            f"最大回撤为 {_pct(row['max_drawdown'])}。",
            "仅 1 只股票，无横向对比。",
            "该股票在所选区间的风险水平如上。",
            DISCLAIMER,
        ))
        return insights

    v_top = _top_by(rr, "volatility")
    v_bot = _bottom_by(rr, "volatility")
    if v_top is not None and v_bot is not None:
        median = rr["volatility"].median()
        insights.append(_insight(
            CATEGORY_RISK, "波动最大",
            f"{v_top[0]} 日收益波动最大，日收益标准差为 {_pct(v_top[1])}（未年化）。",
            _compare_median(v_top[1], median, higher=True),
            "该股票在所选区间的日收益波动幅度相对最大。",
            DISCLAIMER,
        ))
        insights.append(_insight(
            CATEGORY_RISK, "波动最小",
            f"{v_bot[0]} 日收益波动最小，日收益标准差为 {_pct(v_bot[1])}（未年化）。",
            _compare_median(v_bot[1], median, higher=False),
            "该股票在所选区间的日收益波动幅度相对最小。",
            DISCLAIMER,
        ))

    dd_deep = _bottom_by(rr, "max_drawdown")   # 最负 = 最深
    dd_shallow = _top_by(rr, "max_drawdown")   # 最接近 0 = 最浅
    if dd_deep is not None and dd_shallow is not None:
        insights.append(_insight(
            CATEGORY_RISK, "回撤最深",
            f"{dd_deep[0]} 最大回撤最深，为 {_pct(dd_deep[1])}。",
            "回撤为负值，越接近 -100% 表示从历史峰值回落越深。",
            "该股票在所选区间曾经历相对最深的回撤。",
            DISCLAIMER,
        ))
        insights.append(_insight(
            CATEGORY_RISK, "回撤最浅",
            f"{dd_shallow[0]} 最大回撤最浅，为 {_pct(dd_shallow[1])}。",
            "该股票在所选区间的最大回撤幅度相对最小。",
            "该股票在所选区间的回撤相对最浅。",
            DISCLAIMER,
        ))

    merged = rc[["symbol", "cumulative_return"]].merge(
        rr[["symbol", "volatility", "max_drawdown"]], on="symbol"
    )
    # 剔除累计收益或波动率为 NaN 的股票，避免 nan% 混入结论。
    merged = merged.dropna(subset=["cumulative_return", "volatility"])
    if merged.empty:
        insights.append(_insight(
            CATEGORY_RISK, "收益与风险样本不足",
            "有效收益或波动率不足，无法判断高收益是否伴随高波动。",
            "剔除累计收益/波动率为 NaN 的股票后无有效样本。",
            "样本不足时不生成收益与风险的交叉结论，以免误导。",
            DISCLAIMER,
        ))
        return insights
    med_ret = merged["cumulative_return"].median()
    med_vol = merged["volatility"].median()
    high_high = merged[(merged["cumulative_return"] > med_ret) & (merged["volatility"] > med_vol)]
    if not high_high.empty:
        detail = "；".join(
            f"{r['symbol']}（累计收益 {_pct(r['cumulative_return'])}，波动 {_pct(r['volatility'])}）"
            for _, r in high_high.iterrows()
        )
        insights.append(_insight(
            CATEGORY_RISK, "高收益伴随高波动",
            f"累计收益与波动率均高于同组中位数的股票：{detail}。",
            f"收益中位数 {_pct(med_ret)}，波动率中位数 {_pct(med_vol)}。",
            "这些股票在所选区间收益相对领先，同时历史波动也相对较大，属于高收益伴随高波动。",
            DISCLAIMER,
        ))
    else:
        insights.append(_insight(
            CATEGORY_RISK, "高收益伴随高波动",
            "所选股票中没有同时位于收益与波动率中位数之上的股票。",
            f"收益中位数 {_pct(med_ret)}，波动率中位数 {_pct(med_vol)}。",
            "说明在此区间，收益领先与波动领先并未同时出现。",
            DISCLAIMER,
        ))

    if len(merged) >= 3:
        rho = merged["cumulative_return"].corr(merged["volatility"], method="spearman")
        if pd.notna(rho):
            direction = "正相关" if rho > 0 else ("负相关" if rho < 0 else "无单调关系")
            insights.append(_insight(
                CATEGORY_RISK, "收益与风险关系",
                f"在 {len(merged)} 只有效股票中，累计收益与波动率呈{direction}"
                f"（Spearman ρ={_corr(rho)}）。",
                "统计口径为截面排序相关，非时间序列相关。",
                "说明收益较高的股票是否整体伴随较高波动。",
                f"仅 {len(merged)} 只有效股票，样本较少，此关系仅供参考；{DISCLAIMER}",
            ))
    return insights


# --- correlation -------------------------------------------------------------

def _correlation_pairs(corr: DataFrame) -> list[tuple[str, str, float]]:
    labels = [str(x) for x in corr.index]
    arr = corr.to_numpy(dtype=float)
    n = corr.shape[0]
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):  # 排除对角线
            v = arr[i, j]
            if not np.isnan(v):
                pairs.append((labels[i], labels[j], float(v)))
    return pairs


def _correlation_insights(data: DataFrame, method: str) -> list[EdaInsight]:
    if "return" not in data.columns:
        return []
    if data["symbol"].nunique() < 2:
        return [_insight(
            CATEGORY_CORRELATION, "无法计算相关性",
            "仅 1 只股票，无法计算股票间收益率相关性。",
            "相关性需要至少 2 只股票。",
            "增加股票后即可计算两两相关性。",
            DISCLAIMER,
        )]
    try:
        corr = correlation_matrix(data, method=method)
    except InsufficientDataError as exc:
        return [_insight(
            CATEGORY_CORRELATION, "相关性样本不足",
            "部分股票对的有效重叠收益不足，无法计算相关性。",
            str(exc),
            "样本不足的股票对已被跳过，不给出相关性结论。",
            DISCLAIMER,
        )]

    pairs = _correlation_pairs(corr)
    if not pairs:
        return [_insight(
            CATEGORY_CORRELATION, "无有效相关性",
            "没有可计算的股票对相关性。",
            "所有股票对的有效重叠收益都不足或不可用。",
            "无法给出相关性结论。",
            DISCLAIMER,
        )]

    label = {"pearson": "Pearson", "spearman": "Spearman", "kendall": "Kendall τ"}[method]
    max_v = max(p[2] for p in pairs)
    min_v = min(p[2] for p in pairs)
    if max_v == min_v:
        all_desc = "；".join(f"{a} 与 {b}" for a, b, _ in pairs)
        return [_insight(
            CATEGORY_CORRELATION, "相关性",
            f"所有股票对的 {label} 相关系数均为 {_corr(max_v)}：{all_desc}。",
            "各股票对的历史走势同步程度一致。",
            "在所选区间内，股票两两之间的历史走势同步程度相同。",
            DISCLAIMER,
        )]

    top = [p for p in pairs if p[2] == max_v]
    bottom = [p for p in pairs if p[2] == min_v]
    top_desc = "；".join(f"{a} 与 {b}（{_corr(v)}）" for a, b, v in top)
    bottom_desc = "；".join(f"{a} 与 {b}（{_corr(v)}）" for a, b, v in bottom)
    return [
        _insight(
            CATEGORY_CORRELATION, "相关性最高",
            f"收益率相关性最高的股票对为 {top_desc}，{label} 相关系数 {_corr(max_v)}。",
            "相关系数取值范围 [-1, 1]，越接近 1 表示历史走势同步程度越高。",
            "这些股票的历史日收益走势同步程度相对最高。",
            DISCLAIMER,
        ),
        _insight(
            CATEGORY_CORRELATION, "相关性最低",
            f"收益率相关性最低的股票对为 {bottom_desc}，{label} 相关系数 {_corr(min_v)}。",
            "相关系数越接近 -1 表示历史走势越反向。",
            "这些股票的历史日收益走势同步程度相对最低。",
            DISCLAIMER,
        ),
    ]


# --- trend -------------------------------------------------------------------

def _trend_interpretation(close, ma20, ma5) -> str:
    parts = []
    if pd.notna(close) and pd.notna(ma20):
        if close > ma20:
            parts.append("最新收盘价位于 20 日均线上方，近期价格相对近一月平均水平偏高")
        elif close < ma20:
            parts.append("最新收盘价位于 20 日均线下方，近期价格相对近一月平均水平偏低")
        else:
            parts.append("最新收盘价与 20 日均线持平")
    if pd.notna(ma5) and pd.notna(ma20):
        if ma5 > ma20:
            parts.append("短期均线位于长期均线上方")
        elif ma5 < ma20:
            parts.append("短期均线位于长期均线下方")
    if not parts:
        return "趋势数据不足，无法判断。"
    return "；".join(parts) + "。"


def _trend_insights(data: DataFrame, rc) -> list[EdaInsight]:
    if "close" not in data.columns:
        return []
    insights: list[EdaInsight] = []

    cum_map = {}
    if rc is not None:
        cum_map = dict(zip(rc["symbol"], rc["cumulative_return"]))

    for symbol, group in data.groupby("symbol", sort=True):
        last = group.iloc[-1]
        close = last["close"]
        ma5 = last["ma5"] if "ma5" in data.columns else np.nan
        ma20 = last["ma20"] if "ma20" in data.columns else np.nan
        cum = cum_map.get(symbol, np.nan)

        findings = []
        evidence = []
        insufficient = []

        if pd.isna(close):
            insufficient.append("缺少有效收盘价")
        else:
            if pd.notna(ma20):
                rel = "高于" if close > ma20 else ("低于" if close < ma20 else "持平于")
                findings.append(f"最新收盘价 {close:.2f} {rel} MA20（{ma20:.2f}）")
                evidence.append(f"收盘价 {close:.2f}，MA20 {ma20:.2f}")
            else:
                insufficient.append("MA20 尚未形成（需 20 个交易日）")

            if pd.notna(ma5) and pd.notna(ma20):
                rel5 = "高于" if ma5 > ma20 else ("低于" if ma5 < ma20 else "持平于")
                findings.append(f"MA5（{ma5:.2f}）{rel5} MA20")
            elif pd.isna(ma5):
                insufficient.append("MA5 尚未形成（需 5 个交易日）")
            elif pd.notna(ma5) and pd.isna(ma20):
                insufficient.append("MA20 尚未形成，无法比较 MA5 与 MA20")

        if pd.notna(cum):
            direction = "为正" if cum > 0 else ("为负" if cum < 0 else "为零")
            findings.append(f"区间累计收益{direction}（{_pct(cum)}）")

        finding = f"{symbol}：" + ("；".join(findings) if findings else "趋势数据不足")
        caveat = (("样本不足：" + "；".join(insufficient) + "；") if insufficient else "") + DISCLAIMER

        insights.append(_insight(
            CATEGORY_TREND, f"趋势：{symbol}",
            finding,
            "；".join(evidence) if evidence else "—",
            _trend_interpretation(close, ma20, ma5),
            caveat,
        ))
    return insights
