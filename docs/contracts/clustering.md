# Clustering Contract

**Contract Status: Draft v0.1**

**Owner: Role 5（由 Role 1 协调，需 Role 2、3 评审）**

## `run_clustering(...)`

- Input：按股票汇总且定义清晰的风险收益特征表，以及聚类配置。
- Output：股票到聚类标签的映射、用于解释聚类的摘要，以及必要的处理元数据；精确 Schema 待确认。
- Data Type：预计输入为 `pandas.DataFrame`；输出建议使用明确的数据类或字典 Schema，尚未冻结。
- Errors：必要特征缺失、样本不足、非数值数据、非法聚类数、标准化或训练失败。
- Owner：Role 5。

`cluster` 是模型私有输出，不进入公共 Market Data Contract。标准化拟合范围、随机种子、聚类数选择和风险收益画像字段必须在实现前评审。本草案不实现 K-Means、StandardScaler 或股票画像。
