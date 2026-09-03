# 证券金融数据分析与可视化系统

## 项目简介

本项目面向金融数据分析学习与课程展示。系统使用 Python 获取、清洗 A 股日线数据，计算收益率、移动平均、波动率和回撤等指标，通过 Streamlit 与 Plotly 展示行情、EDA 和多股票比较。算法限定为 Decision Tree 涨跌分类、Linear Regression 次日收益率回归和 K-Means 股票画像聚类，不提供投资建议。

当前 P0 核心代码已完成六个 Role 的集成：

- 数据页支持离线 Sample Data、AkShare 在线日线多源回退，以及独立的实时行情快照。AkShare 是统一调用库，页面同时展示其实际上游来源和抓取时间；在线结果不会写入本地 CSV。
- EDA 页除统计表和图表外，还展示由历史区间动态计算的表现、风险、趋势、相关性和数据质量结论，并提供单股 K 线图。
- 监督学习页已接入 Decision Tree 与 Linear Regression 的真实测试集指标、图表和预测明细；聚类页已接入固定 `KMeans(k=3)` 的股票画像和解释。
- Streamlit 页面只调用 Role 1 Service Layer；领域模块不依赖 Streamlit。

离线样例用于稳定课堂演示，在线行情受供应商接口与网络状态影响。安装依赖后执行 `streamlit run app.py` 启动界面。所有分析和模型输出仅用于课程方法演示，不构成投资建议。

## 分工

- `Laurenhuan`：Role 1，系统架构、Streamlit、Service Layer、PR 与最终集成。
- `juanjuan-he`：Role 2，A 股数据获取、清洗、标准 DataFrame 与公共指标。
- `aaaSpringaaa`：Role 3，EDA、描述统计、Plotly 图表与多股票比较。
- `zhou_learn_hahaha`：Role 4，Decision Tree 分类、Accuracy 与 Confusion Matrix。
- `a3556115538qqcom`：Role 5，Stock Profile、StandardScaler、K-Means 与聚类定位。
- `lu2160`：Role 6，Linear Regression、MAE、R²、Actual vs Predicted 与算法 Review。

`README.md` 和 `todos.md` 位于根目录并由全组共用。团队统一规范见 `docs/team_conventions.md`；每位成员的日报、AI 文本记录和三份课程报告分别归档到 `daily/<学号>/Dn.md`、`prompts/<学号>/Dn/<工具名>.txt` 和 `docs/<学号>/`。日报和提示词只能由本人按真实过程补充，期末报告按课程阶段安排完成。
