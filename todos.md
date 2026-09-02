# 项目需求与待办清单

> 每天按真实完成情况更新。只有代码、测试和页面均达到验收标准后才勾选。

## 工程与协作

- [x] R1 建立 Python、Streamlit、`src/`、`pages/`、`tests/` 工程骨架
- [x] R2 冻结 Market Data、Supervised Learning、Clustering Contract v0.2
- [x] R3 建立六人 Ownership、功能分支和 PR Review 边界
- [x] R4 建立 Gitee 主仓库与 GitHub 备份镜像规则
- [x] R5 建立 `daily/`、`prompts/`、`docs/` 课程过程记录结构

## 数据与分析

- [x] R6 实现 Sample Data Fallback，使无 Token、无网络时仍可运行
- [x] R7 实现行情获取、清洗、标准字段和公共金融指标
- [x] R8 为数据层补充字段、排序、唯一性、冲突和指标测试
- [ ] R9 完成 EDA、描述统计、缺失值检查和多股票比较
- [ ] R10 完成可复用 Plotly 行情、收益率、波动率和回撤图表

## 算法

- [ ] R11 完成 Decision Tree 次日上涨/非上涨分类及 Accuracy、Confusion Matrix
- [ ] R12 完成 Linear Regression 次日收益率回归及 MAE、R²、Actual vs Predicted
- [ ] R13 完成 Stock Profile、StandardScaler、K-Means 与 Cluster 解释
- [ ] R14 检查时间序列切分、未来信息泄漏、随机种子和可复现性

## 应用与交付

- [x] R15 打通 Streamlit → Service → Sample DataFrame → 筛选 → 表格/价格图原型
- [ ] R16 将 Role 2 数据层替换接入正式数据概览页面
- [ ] R17 将 Role 3/4/5/6 结果接入对应 Streamlit 页面
- [ ] R18 完成六人模块集成测试、异常提示和演示数据检查
- [ ] R19 完成调研报告、项目报告、演示材料和最终回归测试
