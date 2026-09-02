"""Role 5 聚类 Pipeline Demo — 独立可运行的演示脚本。

运行方式：
    python scripts/demo_clustering.py

演示完整流程：
    Sample CSV → build_stock_profiles → StandardScaler → KMeans → Cluster
"""

from __future__ import annotations

import sys
from pathlib import Path

# 确保项目根目录在 Python path 中
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from src.models.unsupervised.clustering import (
    FEATURE_COLS,
    N_CLUSTERS,
    build_stock_profiles,
    run_clustering,
)


def main() -> None:
    # ── 1. 加载 Sample Data ──────────────────────────────
    csv_path = project_root / "data" / "sample" / "sample_market_data.csv"
    print(f"📂 加载数据: {csv_path}")
    market_df = pd.read_csv(csv_path)
    market_df["trade_date"] = pd.to_datetime(market_df["trade_date"])
    print(f"   {len(market_df)} 行, {market_df['symbol'].nunique()} 只股票")
    print()

    # ── 2. 构建 Stock Profile Table ──────────────────────
    print("📊 构建 Stock Profile Table...")
    profiles = build_stock_profiles(market_df)
    print(profiles.to_string(index=False))
    print()

    # ── 3. StandardScaler + KMeans 聚类 ──────────────────
    print(f"🔬 执行 K-Means 聚类 (k={N_CLUSTERS})...")
    result = run_clustering(profiles)

    # ── 4. 输出聚类结果 ──────────────────────────────────
    print("\n✅ 聚类结果:")
    print(result["profiles"].to_string(index=False))

    print("\n📍 聚类中心 (原始尺度):")
    print(result["cluster_centers"].to_string(index=False))

    # ── 5. 按 cluster 分组展示 ───────────────────────────
    print("\n📋 按 Cluster 分组:")
    for c in sorted(result["profiles"]["cluster"].unique()):
        stocks = result["profiles"][result["profiles"]["cluster"] == c]
        print(f"\n  Cluster {c} ({len(stocks)} 只):")
        for _, row in stocks.iterrows():
            print(
                f"    {row['symbol']:>12s}  "
                f"mean_return={row['mean_return']:+.4f}  "
                f"volatility={row['volatility']:.4f}  "
                f"max_drawdown={row['max_drawdown']:.4f}"
            )

    # ── 6. 解释 cluster 含义 ─────────────────────────────
    print("\n💡 解读 cluster 含义 (根据聚类中心):")
    centers = result["cluster_centers"]
    for _, center in centers.iterrows():
        c = int(center["cluster"])
        mr = center["mean_return"]
        vol = center["volatility"]
        dd = center["max_drawdown"]
        print(
            f"  Cluster {c}: "
            f"mean_return={mr:+.4f}, "
            f"volatility={vol:.4f}, "
            f"max_drawdown={dd:.4f}"
        )
        if mr > 0 and vol < centers["volatility"].median():
            print(f"    → 低波动正收益（偏稳健）")
        elif mr < 0 and abs(dd) > abs(centers["max_drawdown"].median()):
            print(f"    → 高回撤负收益（偏风险）")
        else:
            print(f"    → 中间型")

    print("\n🎉 Pipeline 跑通！")


if __name__ == "__main__":
    main()
