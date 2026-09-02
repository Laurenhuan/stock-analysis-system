# D2 日报

> 角色：Role 3 金融数据分析与可视化工程师 · 胡梦帆
> 日期：2026-09-02

## 昨日遗留问题的回答

- 数据规模（5 只 vs 60 只）已反馈组长：建议折中 8~15 只真实股票，并附 AkShare 免费前复权替代方案。
- 今日确认：Role 2 数据层采用 Tushare 前复权 + Sample 回退，Sample 为 5 只 × 78 个交易日，后续扩充为 **10 只 × 约 250 个交易日**（字段契约不变）。我的模块已按 10 只规模完成验证，无需改动。

## 今日目标

- 完成 Role 3 P0：EDA 分析函数 + Plotly 可视化函数 + 单元测试
- 与 Role 2 数据层做只读对接验证（不修改数据获取层）
- 支持后续 10 只股票规模：多股票图选股参数 + 相关系数热力图

## 实际进展

- EDA（`src/analysis/eda.py`，6 个函数）：
  - `describe_statistics` / `date_range_summary` / `risk_return_summary` / `returns_comparison` / `correlation_matrix`（Spearman，稳健于日收益率厚尾）/ `missing_values_summary`
  - 统一校验：空数据 / 缺列 / 股票不足分别抛 `NoDataError` / `DataValidationError` / `InsufficientDataError`
- 可视化（`src/visualization/charts.py`，6 个函数）：
  - `plot_price`（单股 + MA 叠加 / 多股）、`plot_returns_comparison`、`plot_risk_comparison`、`plot_confusion_matrix`、`plot_actual_vs_predicted`
  - 统一视觉主题（配色 / 字体 / 边距 / hovermode="x unified"）
- 10 只股票规模适配（今日迭代）：
  - `plot_price` / `plot_returns_comparison` / `plot_risk_comparison` 新增 `symbols` 参数，支持选股显示，避免一次全显所有曲线
  - 新增 `plot_correlation_matrix` 相关系数热力图（发散红蓝色标，验证 10×10）
- 测试：`tests/unit/test_eda.py` 19 个 + `tests/unit/test_charts.py` 20 个 = **39 个全部通过**；补装 scikit-learn 后全仓库 **69 个测试全部通过**
- 只读对接验证：`git fetch` gitee 的 `feat/data-foundation` 分支到临时目录（不动本地工作区、不改 `src/data/` 和 `pages/`），Role 2 的 `clean_market_data` + `build_common_features` 输出直接喂给我的 EDA/图表，**零改动跑通**；字段 / 空值 / 单位均符合契约，无需反馈 Role 2
- 边界自查：`src/analysis/`、`src/visualization/` 内无 AkShare / Tushare / Streamlit 引用，分析只消费 Role 2 的标准 DataFrame
- 本地演示脚本 `scripts/demo_eda.py`：Sample Data 全流程跑通，导出 6 张 HTML 图（含热力图与选股演示）
- 今日提交（分支 `feat/eda-visualization`，作者 `aaaspringaaa`）：
  - `8cc3707` feat(eda): 实现 6 个 EDA 分析函数（描述统计/日期范围/风险收益/收益对比/相关矩阵/缺失值）
  - `4906034` feat(charts): 实现 6 个 Plotly 图表，支持 symbols 选股与相关系数热力图
  - `34accd5` test: 新增 EDA 与图表单元测试 39 个，覆盖 10 只股票规模与选股
  - `docs`: 添加 demo 演示脚本、D2 过程记录与 prompts 导出（随本记录一并提交）

## 遇到的问题与解决

- **问题**：本地 `src/data/` 还是占位符，Role 2 实现在 gitee 分支且未合并进 main。
  **解决**：`git fetch` 只读拉到临时目录验证接口，不修改数据获取层、不动工作区；等合并后再正式对接。
- **问题**：Sample CSV 只有 8 个基础字段 + `drawdown`，缺 `return / ma5 / ma20 / volatility_20d / volume_change`（Role 2 的公共指标）。
  **解决**：demo 脚本内临时计算并在 docstring 标注为 Role 2 Owner 职责；合并后用 `build_common_features` 替换。
- **问题**：导出的 Plotly HTML 图打不开。
  **解决**：文件约 4MB（内联 plotly.js），需在文件资源管理器双击用浏览器打开，聊天里的相对路径链接点不动。
- **问题**：全仓库测试因缺 scikit-learn 收集失败（Role 5 聚类模块依赖）。
  **解决**：安装 scikit-learn（requirements.txt 声明的依赖）后 69 个测试全绿。

## 明日计划

- 等 Role 2 数据层合并进 main 后，替换 demo 里的临时特征计算
- 与 Role 1 对接 EDA 页面（选股 multiselect 传入我的 `symbols` 参数）
- 与 Role 4 确认 `y_true / y_pred` 列名与混淆矩阵标签顺序；与 Role 5 确认 K-Means 输入即 `risk_return_summary` 输出
- 提交 `feat/eda-visualization` 分支并发 PR
- 导出 prompts 记录（`prompts/` 目录）
