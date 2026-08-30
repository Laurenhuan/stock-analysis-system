# Supervised Learning Contract

**Contract Status: Draft v0.1**

**Owner: Role 4（由 Role 1 协调，需数据与展示消费者评审）**

## `run_classification(...)`

- Input：符合已确认数据 Contract 的特征表、分类目标定义与训练配置。
- Output：结构化的分类结果；指标、预测和模型对象是否拆分返回待确认。
- Data Type：表格型输入；输出建议使用明确的数据类或字典 Schema，尚未冻结。
- Errors：特征/目标缺失、样本不足、非法配置、训练失败。
- Owner：Role 4。

用途为 Decision Tree Classification 的股票涨跌方向分类。本 Contract 不定义标签生成规则，标签属于模型私有字段，必须由 Role 4 与数据使用方共同确认且不得写回公共行情 Contract。

## `run_regression(...)`

- Input：符合已确认数据 Contract 的解释变量、后续收益率目标定义与训练配置。
- Output：结构化的回归结果；系数、指标、预测和模型对象的边界待确认。
- Data Type：表格型输入；输出 Schema 尚未冻结。
- Errors：特征/目标缺失、样本不足、共线性或数值问题、非法配置、训练失败。
- Owner：Role 4。

本草案不固定数据切分方法、评价指标或超参数，以免在业务目标确认前过早固化实现。
