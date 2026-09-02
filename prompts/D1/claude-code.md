# D1 AI 辅助开发记录

## 工具
Claude Code (VS Code 扩展)

## 对话摘要

### 1. 阅读 Day1 任务文档
- 使用 Claude Code 读取 `Day1.docx` 文件
- 理解 Role 5 的职责：无监督学习与股票画像工程师
- 核心任务：Stock Profile → StandardScaler → K-Means → Cluster

### 2. 探索项目结构
- 通过 Agent 工具探索项目目录结构
- 发现 `src/models/unsupervised/clustering.py` 已有完整实现
- 确认合约文件 `src/contracts/clustering.py` 已定义

### 3. 第一次尝试（被放弃）
- 创建了 sample data、service 层、demo 脚本、更新了页面
- 后来组长更新了要求，明确 Role 5 只负责算法和测试
- 删除了所有不符合要求的文件，重新开始

### 4. 第二次尝试（最终版本）
- 在 `feat/clustering` 分支上重新开发
- 创建了 3 个测试文件：
  - `tests/unit/test_stock_profiles.py`
  - `tests/unit/test_clustering.py`
  - `tests/unit/test_clustering_reproducibility.py`
- 全部 34 个测试通过

### 5. Git 工作流学习
- 理解了 Fork 工作流：fork → clone → branch → commit → push → PR
- 学会了分支管理：创建、切换、删除分支
- 理解了 PR 的原理：只包含两个分支的差异

### 6. PR 提交
- 按组长要求准备了完整的 PR 描述
- 包含：函数签名、输入输出结构、测试结果、AI 使用声明等

## 关键学习点

1. **Git 分支**：未提交的文件在切换分支时会丢失
2. **PR 原理**：只包含分支间的差异，不会包含未修改的文件
3. **测试驱动**：先写测试验证算法，确保代码质量
4. **类型标注**：`-> None` 表示函数不返回值

## 遇到的问题

1. 分支混乱 → 删除多余分支，只保留一个
2. 文件丢失 → 养成及时 git add 的习惯
3. PR 描述不完整 → 按模板逐项补充
4. `/export` 不可用 → 手动创建记录文件

## 使用的 AI 功能

- 文件读取（Read）
- 代码生成（Write）
- Git 操作（Bash）
- 项目探索（Agent）
- 问题解答（对话）
