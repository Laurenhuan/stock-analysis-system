"""Local demo for Role 3's EDA + visualization modules.

Runs every public function in ``src/analysis/eda`` and ``src/visualization/charts``
against ``data/sample/sample_market_data.csv`` (8 symbols x 60 rows) and exports
the figures to HTML so you can eyeball the theme/layout.

NOTE ON BOUNDARIES
------------------
The sample CSV only ships the 8 base columns plus ``drawdown``. The six columns
``return / cumulative_return / ma5 / ma20 / volatility_20d / volume_change`` are
the public indicators owned by Role 2. The block below recomputes them *inline,
for this demo only* so Role 3's functions have valid input to render. It is NOT
part of Role 3's deliverable and must be replaced by Role 2's ``features.py``
output once that lands on the branch.
"""

import sys
from pathlib import Path

# 确保项目根目录在 Python path 中（与 scripts/demo_clustering.py 一致）
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import pandas as pd

from src.analysis.eda import (
    correlation_matrix,
    date_range_summary,
    describe_statistics,
    missing_values_summary,
    returns_comparison,
    risk_return_summary,
)
from src.visualization.charts import (
    plot_correlation_matrix,
    plot_price,
    plot_returns_comparison,
    plot_risk_comparison,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample" / "sample_market_data.csv"
OUT = ROOT / "data" / "sample" / "_demo_output"
OUT.mkdir(exist_ok=True)


def _add_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Temporary stand-in for Role 2's features.py (see module docstring)."""
    out = df.sort_values(["symbol", "trade_date"]).copy()
    out["return"] = out.groupby("symbol")["close"].pct_change()
    out["cumulative_return"] = out.groupby("symbol")["return"].transform(
        lambda r: (1 + r).cumprod() - 1
    )
    out["ma5"] = out.groupby("symbol")["close"].transform(
        lambda c: c.rolling(5).mean()
    )
    out["ma20"] = out.groupby("symbol")["close"].transform(
        lambda c: c.rolling(20).mean()
    )
    out["volatility_20d"] = out.groupby("symbol")["return"].transform(
        lambda r: r.rolling(20).std()
    )
    out["volume_change"] = out.groupby("symbol")["volume"].pct_change()
    # drawdown already present in the sample CSV.
    return out


def main() -> None:
    df = pd.read_csv(SAMPLE, parse_dates=["trade_date"])
    df = _add_feature_columns(df)

    # --- EDA tables -----------------------------------------------------------
    print("== describe_statistics (head) ==")
    print(describe_statistics(df).head())
    print("\n== date_range_summary ==")
    print(date_range_summary(df))
    print("\n== risk_return_summary ==")
    print(risk_return_summary(df).round(4))
    print("\n== returns_comparison ==")
    print(returns_comparison(df).round(4))
    print("\n== correlation_matrix (Spearman, rounded) ==")
    print(correlation_matrix(df).round(2))
    print("\n== missing_values_summary (head) ==")
    print(missing_values_summary(df).head())

    # --- Figures exported to HTML --------------------------------------------
    plot_price(df, title="收盘价走势（全部股票）").write_html(OUT / "price.html")
    plot_returns_comparison(df, title="累计收益率对比（全部）").write_html(
        OUT / "returns.html"
    )
    plot_risk_comparison(
        risk_return_summary(df), title="波动率 vs 最大回撤"
    ).write_html(OUT / "risk.html")

    # 相关系数热力图（新）：直接吃 correlation_matrix() 的输出
    plot_correlation_matrix(
        correlation_matrix(df), title="相关系数热力图（Spearman）"
    ).write_html(OUT / "correlation.html")

    # 选股演示（新）：symbols 参数只画选中的股票，不全显
    pick = ["600519.SH", "000001.SZ", "300750.SZ"]
    plot_price(df, symbols=pick, title="收盘价走势（选 3 只）").write_html(
        OUT / "price_selected.html"
    )
    plot_returns_comparison(
        df, symbols=pick, title="累计收益率（选 3 只）"
    ).write_html(OUT / "returns_selected.html")

    print(f"\n图表已导出到：{OUT}")


if __name__ == "__main__":
    main()
