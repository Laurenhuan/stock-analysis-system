# Supervised Learning Contract

**Contract Status: Review Candidate v0.2**

**Owner: Role 4（由 Role 1 协调，需数据与展示消费者 Review）**

## 1. Shared research frame

监督学习 P0 包含：

1. `DecisionTreeClassifier`
2. `LinearRegression`

共同研究框架为：

```text
X(t) → y(t+1)
```

输入特征只能使用第 t 个交易日或之前已经可获得的信息。禁止使用未来数据构建当前特征。P0 每次模型调用处理一只股票的时间序列；批量股票协调属于 Service Layer，不把多只股票随机混合后切分。

## 2. Eligible samples and temporal split

1. 按 `trade_date` 严格升序。
2. 在模型私有数据集中构造下一交易日目标。
3. 删除无法获得下一交易日目标的最后一个样本。
4. 处理模型所需特征中的 NaN，并保留日期对齐关系。
5. 对最终有效样本使用 `split_index = int(n_samples * 0.8)`。
6. `[:split_index]` 是最早 80% Training Set，`[split_index:]` 是最新 20% Test Set。

禁止随机打乱或使用 `shuffle=True`。训练集或测试集为空、分类训练集不足以形成有效模型、回归测试集不足以计算要求指标时，应抛出 `InsufficientDataError`。

## 3. Classification

### Research question and model

根据历史金融特征，对下一交易日股票方向进行基础分类分析。P0 只使用 `DecisionTreeClassifier`，不增加 Random Forest 等其他模型。

### Private target

在单只股票内部：

```text
next_return = return.shift(-1)
next_return > 0  → label = 1
next_return <= 0 → label = 0
```

`next_return` 和 `label` 均为模型私有字段，不写回公共 Market DataFrame。

### P0 metrics

- `accuracy`：float。
- `confusion_matrix`：固定标签顺序 `[0, 1]` 的 2×2 integer matrix。

`precision`, `recall`, `f1` 属于 P1，不得延误 P0。

### Output Schema

`ClassificationResult` 是普通字典形状，至少包含：

- `model`：已拟合的 `DecisionTreeClassifier` 实例；
- `feature_names`：按训练矩阵列顺序排列的 `list[str]`；
- `metrics`：包含 `accuracy` 和 `confusion_matrix`；
- `predictions`：按 `trade_date` 升序的 `pandas.DataFrame`。

`predictions` 至少包含：

```text
trade_date
y_true
y_pred
```

其中 `y_true`、`y_pred` 只能取 0 或 1。

## 4. Regression

### Research question and model

研究历史金融特征与下一交易日收益率之间是否存在基础线性关系。P0 只使用 `LinearRegression`。

### Private target

```text
next_return = return.shift(-1)
X(t) → return(t+1)
```

`next_return` 是模型私有字段，不写回公共 Market DataFrame。

### P0 metrics

- `mae`：Mean Absolute Error，float。
- `r2`：R²，float；测试集样本不足以定义 R² 时抛出 `InsufficientDataError`。

`mse` 属于 P1，不得延误 P0。

### Output Schema

`RegressionResult` 是普通字典形状，至少包含：

- `model`：已拟合的 `LinearRegression` 实例；
- `feature_names`：按训练矩阵列顺序排列的 `list[str]`；
- `metrics`：包含 `mae` 和 `r2`；
- `predictions`：按 `trade_date` 升序的 `pandas.DataFrame`。

`predictions` 至少包含：

```text
trade_date
y_true
y_pred
```

## 5. Python representation

P0 推荐 `TypedDict + pandas.DataFrame`，定义在 `src/contracts/supervised.py`。这保持普通 Python 字典和 DataFrame 的使用方式，同时固定顶层键、指标键和预测列，不引入复杂结果类或额外框架。

## 6. Errors

- `DataValidationError`：字段、日期顺序、非有限值或目标对齐不满足 Contract。
- `InsufficientDataError`：处理 NaN 和下一日目标后没有足够训练/测试样本。
- Owner：Role 4；Service Layer 负责转换为 UI 可理解状态。
