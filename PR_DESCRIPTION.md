# Role 5 聚类模块加固

## 改动概览
- 重写 `build_stock_profiles()`：使用 Role 2 的 public `return` 列，不重新计算收益率
- 重写 `run_clustering()`：增强输入校验，Contract 强制校验
- 重写 `tests/unit/test_clustering.py`：38 个测试，全部通过

## 核心变更
1. **数据源切换**：从 `close` 列计算收益率 → 使用 Role 2 的 `return` 列
2. **输入校验**：DataFrame 类型检查、空值/无穷值检测、时间区间一致性、重复 symbol 检测
3. **数据质量**：每只股票有效 return ≥ 2 个，有效 Profile ≥ 3 个
4. **异常处理**：统一抛出 DataValidationError / InsufficientDataError，不再使用 ValueError/RuntimeError
5. **FEATURE_COLS 改为 tuple**：避免可变全局变量暴露

## 测试覆盖
- build_stock_profiles: 8 个测试（正常、异常、边界）
- run_clustering: 12 个测试（正常、异常、Contract 校验）
- 端到端: 2 个测试
- 10 只股票: 2 个测试
- 总计: 38 个测试，0 个失败

## 验证
- `python -m pytest tests/unit/test_clustering.py` → 38 passed
- `python -m pytest` → 70 passed（全量）

## 文件变更
- `src/models/unsupervised/clustering.py` — 核心实现
- `tests/unit/test_clustering.py` — 单元测试
