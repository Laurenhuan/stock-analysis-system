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

- [ ] K-Means 多股票风险收益画像聚类

## 每日记录

- [ ] daily/ 每日日报（D1…D6）
- [ ] prompts/ 提示词导出（D1…D6）
