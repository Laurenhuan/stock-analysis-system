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
- [x] R19 完成项目 README、最终演示材料、课程材料目录核对和交付前回归测试

## D4 产品化与动态数据

- [x] R20（Role 2 / Role 1 Review）允许用户输入任意合法 A 股代码，完成代码规范化、交易所后缀补全、非法/不存在代码提示；在线数据不写入本地 CSV
- [x] R21（Role 2 / Role 1）提供可选的股票代码/名称目录或搜索接口并设置合理缓存；若目录接口不可用，保留手工输入标准代码的可靠降级方式
- [x] R22（Role 1）重构首页和导航为“单股研究”“多股比较”两条用户路径，修正首页仍显示 D2/Decision Tree 待接入的过期文案
- [x] R23（Role 1）用 `st.navigation` / `st.Page` 和 `st.session_state` 统一股票、日期、数据源状态，进入子页面后不重复选择
- [x] R24（Role 1 / Role 2）让 EDA、监督学习和聚类均可使用 AkShare 获取到最新交易日的历史日线；实时快照保持独立，不直接作为 EDA 或模型训练输入
- [x] R25（Role 3）验证 EDA 对用户自选股票数量和日期区间的适应性，补部分缺失、样本不足和不同上市日期的结论/测试
- [x] R26（Role 4）验证 Decision Tree 对单只用户自选股票和日期区间的适应性，明确最小有效样本、80/20 时间切分和测试集交易日数量
- [x] R27（Role 6）验证 Linear Regression 对单只用户自选股票和日期区间的适应性，明确滚动窗口、最小样本、训练/测试区间及指标解释
- [x] R28（Role 5）支持用户自选至少 3 只股票进行聚类，并根据原始尺度中心动态生成每个 Cluster 的中文画像和相对高/中/低解释
- [x] R29（Role 1 / Role 5 / Role 6 Review）评审 `k` 是否允许调整；结论为 Contract / P0 范围变更未获批，继续固定 `KMeans(k=3)`
- [x] R30（Role 1）按 Streamlit 1.63 官方技能优化主题、卡片、Material 图标、表单提交、加载状态和移动端布局，优先复用原生组件
- [x] R31（Role 1）为在线加载设置有 TTL 和容量上限的缓存，补共享筛选状态、任意股票输入、错误提示和页面流程的 AppTest/集成测试
- [x] R32（Role 1）保存关键 pytest 输出或接入 Gitee CI，让日报中的测试数字可追溯

## D5 最终健壮性收尾

- [x] R33（Role 2 / Role 1 Review）严格校验公开日期格式；为股票目录补真实 TTL 与副本隔离；实时行情仅对暂时性网络异常进行有限重试
- [x] R34（Role 3 / Role 1 集成 Review）清理 pandas 日期构造弃用告警；相关性重叠样本少于 20 个交易日时只说明样本不足，不输出最高/最低排名
- [x] R35（Role 4 / Role 1 集成 Review）补分类输入类型、单股票、严格日期和有限数值校验；拒绝无法形成有效分裂的退化决策树，并公开有效样本及 80/20 区间
- [x] R36（Role 5 / Role 1 集成 Review）在固定 `KMeans(k=3)` 前检查可区分画像数量，拒绝少于 3 个实际簇的退化结果
- [x] R37（Role 6 / Role 1 集成 Review）公开回归有效/训练/测试样本及日期区间，页面说明滚动窗口损失、MAE 与可能为负的 R²
- [x] R38（Role 1）统一 Service 与页面诊断展示，忽略本机 Streamlit secrets，并完成跨模块与严格全仓回归测试

## D5 模型可信度与体验升华

- [x] R39（Role 1 / Role 4 / Role 6 Review）核对 X(t) → y(t+1) 与 80/20 时间切分，在不改 P0 模型 Contract 的前提下增加方向延续/零收益基线和 60%/80%/100% 扩展历史窗口；低分按历史证据如实解释
- [x] R40（Role 1 / Role 3 Review）将 EDA 重构为“最多 5 条核心摘要 → 收益/风险/关系/趋势主题 → 详细统计与数据质量”，保留原统计公式并将证据折叠展示
- [x] R41（Role 1 / Role 5 Review）聚类页面集中显示实际样本区间，移除每个标签重复的 `[所选历史区间]` 占位前缀，不改变固定 `KMeans(k=3)` 和画像逻辑
- [x] R42（Role 1 / Role 4 / Role 6 Review）打通多股股票池 → 关注股票 → 单股模型 → 量化简报流程；新增下一交易日模型信号、较强简单基线和 Markdown 简报，不改变现有评估 Contract 或 P0 算法

## D5 首页与答辩访问准备

- [x] R43（Role 1）将首页重构为“产品定位 → 两个主要入口 → 三阶段分析流程 → 用户价值 → 技术与方法”，减少首屏技术后台感
- [x] R44（Role 1）新增使用者/开发者双视角项目介绍，并通过 Streamlit 原生 `url_path="about.html"` 提供真实 `/about.html`
- [x] R45（Role 1）依据 `git remote`、Git 历史、远程分支和 Ownership 文档展示可核验的 Gitee 入口与团队协作概览，不使用贡献比例或猜测数据
- [x] R46（Role 1）配置 `0.0.0.0:8766` 局域网监听，新增 Windows `scripts/run_lan.cmd` 并补同网访问、防火墙和地址选择说明
- [x] R47（Role 1）完成首页/About AppTest、局域网首页与 `/about.html` HTTP 验证及全仓回归测试

## 当前验收状态

- Role 2 动态搜索 PR !28、Role 3 EDA 增强 PR !29、Role 5 聚类画像 PR !27、Role 1 统一工作区 PR !32、Role 4 动态区间分类 PR !33、Role 6 动态区间回归 PR !36 均已合入 Gitee `main`。
- 功能开发、D5 可信度/体验收尾和首页/答辩访问准备 R1–R47 已完成。
- D5 最终健壮性基线为 `465 passed`；本轮模型可信度与 EDA 体验整改结果记录于 `docs/test_logs/D5-model-eda-ux.txt`。
- 首页、`/about.html` 和局域网交付验证结果记录于 `docs/test_logs/D5-home-about-lan.txt`；本机与当前局域网地址的首页和 `/about.html` 均返回 HTTP 200。
- 相关系数交互说明、README 与课程记录收尾后的最终全仓结果为 `480 passed`；三种相关系数均完成独立 Service 回归验证。
- Role 3 的 pandas 弃用告警修复已在最终分支保留原作者归属并通过测试；其新 PR 中超出 800 字的个人日报及未完成评审的预测解读/Wiki 未纳入本轮，Role 1 未改写。
- D4 Role 1 的定向测试、全仓测试、警告说明和在线冒烟结果已保存至 `docs/test_logs/D4-role1.txt`。
- 在线股票目录以“茅台”实测返回 `600519.SH / 贵州茅台`；目录不可用时页面仍允许直接输入代码，历史在线结果不落本地 CSV。
- 在线行情不落本地 CSV；页面明确展示 Sample/在线状态、实际 provider、抓取时间及延迟说明。
- 两个旧的 `prompts/D1/` 目录说明文件已按组长确认移除，没有修改其他成员按学号归档的课程记录。

## 已完成的合并与课程记录

- [x] G1 由 Role 1 推送 `feat/akshare-multisource`，在 Gitee 完成 Role 2 代码 PR 合并
- [x] G2 由 Role 1 推送 `feat/role1-final-integration`，在 Gitee 完成最终代码集成 PR 合并
- [x] G3 由 Role 1 在代码进入 `main` 后，如实补充本人的 D3 日报与真实 Codex 提示词记录
- [x] G4 由 Role 2 本人维护并合并其 D3 日报和 prompts PR；Role 1 未代写、未改写
## PR 合入后的人工收尾（不计入项目开发完成度）

- Gitee `main` 稳定后，由 Role 1 手动同步 GitHub 备份镜像；当前分支尚未合入，不能提前记为完成。
- 各成员只核对本人 `docs/<学号>/` 课程报告；Role 1 不代写、改写或勾选其他成员的个人材料。
