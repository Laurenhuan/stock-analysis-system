# 证券金融数据分析与可视化系统

## 项目简介

本项目面向金融数据分析学习与课程展示。系统使用 Python 获取、清洗 A 股日线数据，计算收益率、移动平均、波动率和回撤等指标，通过 Streamlit 与 Plotly 展示行情、EDA 和多股票比较。算法限定为 Decision Tree 涨跌分类、Linear Regression 次日收益率回归和 K-Means 股票画像聚类，不提供投资建议。

当前已完成工程骨架、公共数据契约、六人边界、Sample Data 回退、数据层测试和数据概览原型；其余分析、算法展示及集成仍在推进。安装依赖后执行 `streamlit run app.py`。

## 分工

- `Laurenhuan`：Role 1，系统架构、Streamlit、Service Layer、PR 与最终集成。
- `juanjuan-he`：Role 2，A 股数据获取、清洗、标准 DataFrame 与公共指标。
- `aaaSpringaaa`：Role 3，EDA、描述统计、Plotly 图表与多股票比较。
- `zhou_learn_hahaha`：Role 4，Decision Tree 分类、Accuracy 与 Confusion Matrix。
- `a3556115538qqcom`：Role 5，Stock Profile、StandardScaler、K-Means 与聚类定位。
- `lu2160`：Role 6，Linear Regression、MAE、R²、Actual vs Predicted 与算法 Review。

需求进度见 `todos.md`，日报见 `daily/`，AI 文本记录见 `prompts/`，课程报告见 `docs/`。
