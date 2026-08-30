# Clustering Contract

**Contract Status: Review Candidate v0.2**

**Owner: Role 5（由 Role 1 协调，需 Role 2、3 Review）**

## 1. Scope and algorithm

P0 使用 `KMeans`，固定 `k = 3`。不增加 DBSCAN、PCA、Hierarchical Clustering、Elbow Method 或 Silhouette Score；这些不属于 P0。

K-Means 不直接对一只股票的逐日行情聚类。输入必须先转换为每行一只股票的 Stock Profile Table。

## 2. Stock Profile Table

P0 输入 Schema：

| Field | Meaning |
| --- | --- |
| `symbol` | 满足 Market Data Contract 的唯一证券代码 |
| `mean_return` | 比较区间内简单日收益率的算术平均，不年化 |
| `volatility` | 比较区间内简单日收益率的样本标准差，`ddof=1`，不年化 |
| `max_drawdown` | 比较区间内 `drawdown` 的最小值，通常 `<= 0` |

每个 `symbol` 必须唯一，三个特征必须是有限数值且不得包含 NaN。参与比较的股票应使用相同时间区间；不能满足时必须在结果说明中明确记录，不得默认为完全可比。

## 3. Scaling and K-Means

Role 5 必须在三个 P0 特征上拟合并应用 `StandardScaler`，再执行 `KMeans(n_clusters=3, ...)`。StandardScaler 属于 Role 5 的私有 Pipeline，不修改公共 Market DataFrame 或原始 Profile 数值。

随机种子和其他不改变 Contract 的训练参数由 Role 5 在实现中显式记录。有效股票数少于 3 或无法形成有效特征矩阵时抛出 `InsufficientDataError`。

## 4. Output Schema

`ClusteringResult` 是普通字典形状，至少包含：

- `profiles`：原始尺度的 Stock Profile DataFrame，并新增模型私有 `cluster` 列；
- `cluster_centers`：通过 scaler `inverse_transform` 返回原始特征尺度的中心 DataFrame；
- `features`：固定为 `['mean_return', 'volatility', 'max_drawdown']`；
- `k`：固定为 `3`。

`profiles` 至少包含：

```text
symbol
mean_return
volatility
max_drawdown
cluster
```

`cluster_centers` 至少包含：

```text
cluster
mean_return
volatility
max_drawdown
```

`cluster` 是模型私有输出，不得写入公共 Market Data Contract。Cluster 编号本身没有固定金融语义，展示层必须根据中心特征解释，不能把编号直接表述为固定的好坏等级。

## 5. Python representation

P0 推荐 `TypedDict + pandas.DataFrame`，定义在 `src/contracts/clustering.py`。这固定顶层结果键和表格列，同时保持实现简单。

## 6. Errors

- `DataValidationError`：Profile 字段、唯一性、数值或可比区间信息不满足 Contract。
- `InsufficientDataError`：股票数量或有效 Profile 不足。
- Owner：Role 5；Service Layer 负责转换为 UI 可理解状态。
