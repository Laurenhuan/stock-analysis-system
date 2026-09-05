# 团队统一开发与集成规范

## 1. 适用范围

本规范用于六个 Role 的功能开发、PR Review 和 Role 1 集成。路径归属与算法边界以 `docs/role_boundaries.md` 为准，字段和数据口径以 `docs/contracts/` 为准；本文档统一日常实现和验收方式，不变更 Contract。

## 2. 依赖方向与公共入口

```text
Streamlit (app.py / app_pages)
    -> src/services
    -> src/data | src/analysis | src/models | src/visualization
```

- 页面只从 `src.services` 调用应用用例，可从 `src.utils.exceptions` 引用统一异常；不直接引用 `src.data`、`src.analysis`、`src.models`、`src.visualization` 或 `src.contracts`。
- Service 负责组合公共函数、应用参数校验和稳定返回形状；不重写清洗、指标、统计或模型公式。
- Domain 模块不引用 Streamlit，不读页面状态，不把模型私有字段写回共享 Market DataFrame。
- 新增或修改公共函数时，PR 必须列明签名、必需列、输出形状、异常和 Role 1 接入示例。

## 3. 数据与口径

- 标准数据顺序为 `fetch_market_data -> clean_market_data -> build_common_features`；页面不自行读 CSV 或补指标。
- 行情页必须显示 `data_source`、`is_sample` 和可用的 `fallback_reason`。Sample Data 必须明示为非实时快照。
- 滚动窗口前导 NaN 是允许的，不得在页面层填造数值。
- 对数据缺列、无数据、样本不足分别使用 `DataValidationError`、`NoDataError`、`InsufficientDataError`；不吞掉程序错误并假装回退成功。

## 4. 算法与演示边界

- P0 只允许 `DecisionTreeClassifier`、`LinearRegression` 和 `KMeans(k=3)`；页面不得提供违反这一范围的算法或 k 值。
- 监督学习统一为 `X(t) -> y(t+1)`，最早 80% 训练、最新 20% 测试，不 shuffle。
- 聚类特征固定为 `mean_return`、`volatility (ddof=1, 不年化)`、`max_drawdown`，Cluster 编号不表示好坏。
- 未合并模块在页面上标为“待接入”，不使用随机数、占位指标或非 Owner 实现代替。

## 5. 测试与验收

每个功能分支至少完成：

1. Owner 模块单元测试；
2. Role 1 跨模块 Service 联调测试；
3. 分层边界与导入循环检查；
4. 完整测试集。

Windows 本地统一命令：

```powershell
.venv\Scripts\python.exe -m pytest -q
```

只有代码、页面、异常状态、测试和记录都与实际结果一致时，对应 `todos.md` 项才能勾选。

## 6. Git、Review 与过程记录

- Gitee `origin` 是主仓库，GitHub 仅为审核后镜像；不直接提交 `main`，不 force-push 已评审历史。
- Review 修复继续推送原分支/原 PR，不新建重复 PR。共享文件和 Contract/数据政策变更按 `docs/role_boundaries.md` 请求所有受影响 Owner Review。
- 根目录 `README.md` 和 `todos.md` 由全组共用，不在成员学号目录中另建个人版本。
- 每位成员只维护自己的 `daily/<学号>/Dn.md`，只记已发生的 commit、测试和阻塞；未提交改动明确写“本地待提交”。
- 每位成员只导出自己的 `prompts/<学号>/Dn/<工具名>.txt`，不代写他人记录，不提交 Token、Cookie、密码或个人身份数据。
- 每位成员的期末文档位于 `docs/<学号>/立项报告.md`、`docs/<学号>/调研报告.md` 和 `docs/<学号>/项目报告.md`。
- 不猜测其他成员学号，不修改、搬运或补写其他成员的日报、提示词和期末文档。
