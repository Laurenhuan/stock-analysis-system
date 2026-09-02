# 项目待办清单 (Todos)

> 证券金融数据分析与可视化系统 —— 每日按实际完成情况更新勾选状态。

## 数据层（Role 2）

- [x] 数据获取 `fetch.py`（Tushare 前复权 + Sample 兜底 + source 参数）
- [x] 数据清洗 `clean.py`（单位换算、排序、去重、冲突检测、幂等）
- [x] 公共特征 `features.py`（return / ma5 / ma20 / volatility / volume_change / drawdown）
- [x] Sample 样例数据（5 只 A 股 × 78 个真实交易日）
- [x] 数据层单元测试（35 个，全部通过）

## 系统集成（Role 1）

- [ ] Streamlit 首页与页面骨架
- [ ] Service Layer 接入数据层
- [ ] 集成测试

## 数据分析与可视化（Role 3）

- [ ] EDA 描述统计
- [ ] 价格趋势图 / 收益率分布图 / 成交量图

## 监督学习（Role 4）

- [ ] Decision Tree 涨跌方向分类
- [ ] Linear Regression 收益率预测

## 无监督学习（Role 5）

- [x] `build_stock_profiles()` 使用 Role 2 的 return 列（D2 完成）
- [x] `run_clustering()` 加强输入校验（D2 完成）
- [x] FEATURE_COLS 改为 tuple，异常类型统一（D2 完成）
- [x] 39 个单元测试全部通过（D2 完成）
- [x] 真实端到端测试：build_common_features → build_stock_profiles → run_clustering（D2 完成）
- [x] PR #7 已推送并更新（D2 完成）
- [ ] 等待组长 Review，根据意见修改
- [ ] 与 Role 1 联调 Service Layer 接入

## 每日记录

- [ ] daily/ 每日日报（D1…D6）
- [ ] prompts/ 提示词导出（D1…D6）
