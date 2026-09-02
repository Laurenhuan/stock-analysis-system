# Role 6 — Linear Regression 记录与交付说明

> 实现 Owner: Role 6　分支: `feat/linear-regression`
> 契约参照: `docs/contracts/supervised_learning.md`（Regression 部分）、`docs/role_boundaries.md`（Role 6）

## 1. 交付文件

| 文件 | 变更 | 说明 |
| --- | --- | --- |
| `src/models/supervised/regression.py` | 实现（原为占位） | `fit_regression(df)`，返回 `RegressionResult` |
| `tests/unit/test_regression.py` | 新增（9 用例） | 契约键、指标、切分、可复现、单股票、样本不足、**无时序泄漏**、端到端 |

（未改动 Contracts、数据层、页面、Service —— 均属其它 Role 职责。）

## 2. 公共函数签名与输入

```python
def fit_regression(df: pd.DataFrame) -> RegressionResult
```

- 输入：**Role 2 标准公共 DataFrame**（单只股票；多股票会抛 `DataValidationError`）。
  必需列：`symbol, trade_date, close, return` + 特征列
  `ma5, ma20, volatility_20d, volume_change, drawdown`。
  （可由 `src.data.features.build_common_features` 从基础列生成。）
- 特征：`X(t) → return(t+1)`，`next_return = return.shift(-1)`（私有，不写回公共表）。
- 切分：按 `trade_date` 升序，`split_index = int(n * 0.8)`，**不随机打乱**。

## 3. 输出 Schema（`RegressionResult`）

```python
{
  "model": LinearRegression,
  "feature_names": ["return","ma5","ma20","volatility_20d","volume_change","drawdown"],
  "metrics": {"mae": float, "r2": float},
  "predictions": DataFrame(columns=["trade_date","y_true","y_pred"])  # 测试集(样本外), 升序
}
```

### 最小示例
```python
from src.data.features import build_common_features
from src.models.supervised.regression import fit_regression
result = fit_regression(build_common_features(stock_frame))
print(result["metrics"])          # {'mae': ..., 'r2': ...}
print(result["predictions"].head())  # trade_date | y_true | y_pred
```

## 4. 异常

- `DataValidationError`：缺字段、非升序、非有限值、多只股票。
- `InsufficientDataError`：处理 NaN 与次日目标后样本不足 / 测试集 < 2（无法定义 R²）。

## 5. 指标记录（当前 Sample Data，**临时性**）

> 数据：`data/sample/sample_daily.csv`（60 只 × 78 天）。**这是 Sample 临时口径**；
> 待 Role 2 合并"10 只真实 A 股 × 约 250 个交易日(qfq)"后需**重新运行并更新本表**。

对每只股票用 `build_common_features → fit_regression` 计算：

| 指标 | 值 |
| --- | --- |
| 股票数 | 60 |
| 每只样本 | 78 个交易日 |
| 每只测试集 | 12 个样本 |
| MAE 中位数 | ≈ 0.0137 |
| MAE 均值 | ≈ 0.0174 |
| R² 中位数 | ≈ −0.254 |
| R² 均值 | ≈ −1.241 |
| R² > 0 的股票数 | 10 / 60 |

部分代表：
- `000001.SZ`：MAE 0.01358，R² +0.071
- `000333.SZ`：MAE 0.00941，R² +0.061
- `000002.SZ`：MAE 0.00945，R² −0.067
- `002304.SZ`：MAE 0.05759，R² −16.224（小样本下极端负 R²）

**解读**：线性回归在 78 天小样本 + 日频收益近似随机游走下，R² 普遍 ≤0（中位数 −0.25），说明基础线性模型对次日收益的解释力有限——**属真实结果，非 bug**；与 Role 4（决策树）的"接近随机"结论一致。

## 6. 测试结果

- `pytest tests/unit/test_regression.py -q` → **9 passed**。
- 全仓套件在 Role 6 改动下：**73 passed**；另有 1 个失败为 **Role 2** 的
  `tests/unit/test_data_layer.py::test_fetch_tushare_uses_qfq_and_marks_source`
  ——原因：本机 `tushare` 为 namespace 包、顶层无 `pro_api`，属**环境/依赖兼容**问题，
  与 Role 6 改动无关；按 Ownership 需 Role 2 处理（不归 Role 6 修改）。

## 7. 已知限制与集成说明

- **数据量**：Sample 仅 78 天；模型对 `ma20/volatility_20d` 各吃掉 20 天，有效样本更少，
  测试集仅 12 点 → 指标波动大、易过拟合。**需真实 250 天数据**。
- **特征选择**：未使用单调的 `cumulative_return`（避免伪相关/共线主导）；`ma5/ma20/return`
  存在共线，线性系数不稳定（P0 可接受，P1 可考虑标准化）。
- **实际-vs-预测**：`predictions` 已按 `trade_date` 升序提供，可供 Role 3 画图、Role 1 集成页面。
- **集成**：Role 1 在 Service Layer 增加调用（如 `run_regression(symbol)`）并把结果渲染到
  `pages/3_监督学习.py`，无需 Role 6 改动页面/Service。
- **合约变更**：无（本 PR `Contract changes: None`）。
- **数据获取**：Role 6 不抓取行情、不调用 AkShare/Tushare；统一消费 Role 2 的标准 DataFrame。
