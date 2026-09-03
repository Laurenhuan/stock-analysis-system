# 项目需求与待办清单

> 每天按真实完成情况更新。只有代码、测试和页面均达到验收标准后才勾选。

## 工程与协作

- [x] R1 建立 Python、Streamlit、`src/`、`pages/`、`tests/` 工程骨架
- [x] R2 冻结 Market Data、Supervised Learning、Clustering Contract v0.2
- [x] R3 建立六人 Ownership、功能分支和 PR Review 边界
- [x] R4 建立 Gitee 主仓库与 GitHub 备份镜像规则
- [x] R5 建立根目录共用 `README.md` / `todos.md` 和按学号分隔的 `daily/`、`prompts/`、`docs/` 课程材料结构

## 数据与分析

- [x] R6 实现 Sample Data Fallback，使无 Token、无网络时仍可运行
- [x] R7 实现 AkShare 在线日线多源回退、行情清洗、标准字段和公共金融指标
- [x] R8 为数据层补充字段、排序、唯一性、部分成功、单位和多源回退测试
- [x] R9 完成 EDA、描述统计、缺失值检查、多股票比较和问题驱动结论
- [x] R10 完成可复用 Plotly 行情、K 线、收益率、波动率、回撤及模型图表

## 算法

- [x] R11 完成 Decision Tree 次日上涨/非上涨分类及 Accuracy、Confusion Matrix
- [x] R12 完成 Linear Regression 次日收益率回归及 MAE、R²、Actual vs Predicted
- [x] R13 完成 Stock Profile、StandardScaler、K-Means 与 Cluster 解释
- [x] R14 检查时间序列切分、未来信息泄漏、随机种子和可复现性

## 应用与交付

- [x] R15 打通 Streamlit → Service → Sample DataFrame → 筛选 → 表格/价格图原型
- [x] R16 将 Role 2 在线日线、实时快照及来源信息接入数据概览页面
- [x] R17 将 Role 3/4/5/6 结果接入对应 Service 与 Streamlit 页面
- [x] R18 完成六人模块集成测试、异常提示和演示数据检查
- [ ] R19 完成期末课程报告、最终演示材料和交付前回归测试（按课程阶段安排，当前不提前编写）

## D4 产品化与动态数据

- [ ] R20（Role 2 / Role 1 Review）允许用户输入任意合法 A 股代码，完成代码规范化、交易所后缀补全、非法/不存在代码提示；在线数据不写入本地 CSV
- [ ] R21（Role 2 / Role 1）提供可选的股票代码/名称目录或搜索接口并设置合理缓存；若目录接口不可用，保留手工输入标准代码的可靠降级方式
- [ ] R22（Role 1）重构首页和导航为“单股研究”“多股比较”两条用户路径，修正首页仍显示 D2/Decision Tree 待接入的过期文案
- [ ] R23（Role 1）用 `st.navigation` / `st.Page` 和 `st.session_state` 统一股票、日期、数据源状态，进入子页面后不重复选择
- [ ] R24（Role 1 / Role 2）让 EDA、监督学习和聚类均可使用 AkShare 获取到最新交易日的历史日线；实时快照保持独立，不直接作为 EDA 或模型训练输入
- [ ] R25（Role 3）验证 EDA 对用户自选股票数量和日期区间的适应性，补部分缺失、样本不足和不同上市日期的结论/测试
- [ ] R26（Role 4）验证 Decision Tree 对单只用户自选股票和日期区间的适应性，明确最小有效样本、80/20 时间切分和测试集交易日数量
- [ ] R27（Role 6）验证 Linear Regression 对单只用户自选股票和日期区间的适应性，明确滚动窗口、最小样本、训练/测试区间及指标解释
- [ ] R28（Role 5）支持用户自选至少 3 只股票进行聚类，并根据原始尺度中心动态生成每个 Cluster 的中文画像和相对高/中/低解释
- [ ] R29（Role 1 / Role 5 / Role 6 Review）评审 `k` 是否允许调整；在 Contract / P0 范围变更获批前继续固定 `KMeans(k=3)`
- [ ] R30（Role 1）按 Streamlit 1.63 官方技能优化主题、卡片、Material 图标、表单提交、加载状态和移动端布局，优先复用原生组件
- [ ] R31（Role 1）为在线加载设置有 TTL 和容量上限的缓存，补共享筛选状态、任意股票输入、错误提示和页面流程的 AppTest/集成测试
- [ ] R32（Role 1）保存关键 pytest 输出或接入 Gitee CI，让日报中的测试数字可追溯

## 当前验收状态

- Role 2 最新数据链路已通过 PR !18 合入 Gitee `main`；Role 1 的实时行情、EDA 结论/K 线和 Decision Tree 页面已通过 PR !23 完成最终集成。
- 全仓测试：`277 passed`；本轮定向 Service 测试：`16 passed`；页面语法与 Page → Service 分层检查通过。
- 在线行情不落本地 CSV；页面明确展示 Sample/在线状态、实际 provider、抓取时间及延迟说明。
- 两个旧的 `prompts/D1/` 目录说明文件已按组长确认移除，没有修改其他成员按学号归档的课程记录。

## 合并与课程记录待办

- [x] G1 由 Role 1 推送 `feat/akshare-multisource`，在 Gitee 完成 Role 2 代码 PR 合并
- [x] G2 由 Role 1 推送 `feat/role1-final-integration`，在 Gitee 完成最终代码集成 PR 合并
- [x] G3 由 Role 1 在代码进入 `main` 后，如实补充本人的 D3 日报与真实 Codex 提示词记录
- [x] G4 由 Role 2 本人维护并合并其 D3 日报和 prompts PR；Role 1 未代写、未改写
- [ ] G5 Gitee `main` 稳定后同步 GitHub 备份镜像
- [ ] G6 期末阶段由各成员完成本人 `docs/<学号>/` 下的课程报告，不提前虚构
