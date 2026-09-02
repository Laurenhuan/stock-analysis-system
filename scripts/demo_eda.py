"""Local demo for Role 3's EDA + visualization modules.

Runs every public function in ``src.analysis/eda`` and
``src.visualization/charts`` against ``data/sample/sample_market_data.csv``
(60 symbols x 78 trading days). The sample CSV only ships the 8 base Market
Data Contract columns, so Role 2's ``build_common_features`` builds the 7
shared derived fields before Role 3's functions run. Figures are exported as
HTML into ``data/processed/demo_eda/`` (gitignored) so you can eyeball the
theme/layout.
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
from src.data.features import build_common_features
from src.visualization.charts import (
    plot_actual_vs_predicted,
    plot_confusion_matrix,
    plot_correlation_matrix,
    plot_price,
    plot_returns_comparison,
    plot_risk_comparison,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample" / "sample_market_data.csv"
OUT = ROOT / "data" / "processed" / "demo_eda"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(SAMPLE, parse_dates=["trade_date"])
    df = build_common_features(base)

    # --- EDA tables -----------------------------------------------------------
    print("== describe_statistics (head) ==")
    print(describe_statistics(df).head())
    print("\n== date_range_summary ==")
    print(date_range_summary(df))
    print("\n== risk_return_summary (head) ==")
    print(risk_return_summary(df).round(4).head())
    print("\n== returns_comparison (head) ==")
    print(returns_comparison(df).round(4).head())
    print("\n== correlation_matrix (Spearman, 前 10 只) ==")
    print(correlation_matrix(df).round(2).iloc[:10, :10])
    print("\n== missing_values_summary (head) ==")
    print(missing_values_summary(df).head())

    # --- Figures exported to HTML --------------------------------------------
    plot_price(df, title="收盘价走势（全部 60 只）").write_html(OUT / "price.html")
    plot_returns_comparison(df, title="累计收益率对比（全部 60 只）").write_html(
        OUT / "returns.html"
    )
    plot_risk_comparison(
        risk_return_summary(df), title="波动率 vs 最大回撤"
    ).write_html(OUT / "risk.html")
    plot_correlation_matrix(
        correlation_matrix(df), title="相关系数热力图（Spearman）"
    ).write_html(OUT / "correlation.html")

    # 选股演示：symbols 参数只画选中的股票，不全显
    pick = ["600519.SH", "000001.SZ", "300750.SZ"]
    plot_price(df, symbols=pick, title="收盘价走势（选 3 只）").write_html(
        OUT / "price_selected.html"
    )
    plot_returns_comparison(
        df, symbols=pick, title="累计收益率（选 3 只）"
    ).write_html(OUT / "returns_selected.html")

    # 纯绘图函数的演示：Role 3 不训练模型，这里用构造的示例输入
    plot_confusion_matrix(
        [[8, 2], [1, 9]], labels=("0", "1"), title="二分类混淆矩阵（示例）"
    ).write_html(OUT / "confusion.html")
    sample_pred = pd.DataFrame(
        {
            "y_true": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y_pred": [1.1, 1.9, 3.2, 3.8, 5.2],
        }
    )
    plot_actual_vs_predicted(
        sample_pred, title="预测值 vs 实际值（示例）"
    ).write_html(OUT / "actual_vs_predicted.html")

    print(f"\n图表已导出到：{OUT}")


if __name__ == "__main__":
    main()
