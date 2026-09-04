# Role 5 — 无监督学习与股票画像工程师

## 一、我是谁

我在六人团队中负责 **K-Means 聚类**，把多只股票按风险收益特征分成 3 类，供后续页面展示和投资参考。

- 姓名：蔡斌
- 学号：25200317
- 分支：`feat/clustering-hardening`

---

## 二、我负责的文件

| 文件 | 作用 |
|------|------|
| `src/models/unsupervised/clustering.py` | 核心代码：构建股票画像 + 聚类 |
| `tests/unit/test_clustering.py` | 39 个单元测试 |

**我不能动的文件：** `app.py`、`pages/`、`src/services/`、`src/contracts/`、`src/data/`、`src/utils/`

---

## 三、输入是什么

我需要两个输入：

### 输入 1：行情数据（来自 Role 2）

Role 2 提供一个 DataFrame，包含以下列：

| 列名 | 含义 | 示例 |
|------|------|------|
| `symbol` | 股票代码 | 600519.SH |
| `trade_date` | 交易日期 | 2024-01-02 |
| `close` | 收盘价 | 1680.0 |
| `return` | 日收益率 | 0.015 |
| `drawdown` | 回撤 | -0.03 |

这是 Role 2 用 `build_common_features()` 函数计算出来的，我直接用，不自己算。

### 输入 2：调用方式

```python
from src.data.features import build_common_features
from src.models.unsupervised.clustering import build_stock_profiles, run_clustering

# 第一步：Role 2 生成特征
featured_df = build_common_features(raw_market_data)

# 第二步：我构建股票画像
profiles = build_stock_profiles(featured_df)

# 第三步：我做聚类
result = run_clustering(profiles)
```

---

## 四、我做了什么（处理流程）

```
原始行情数据
    ↓
Role 2: build_common_features()
    ↓ 输出含 return, drawdown 的 DataFrame
Role 5: build_stock_profiles()
    ↓ 每只股票 → 3 个特征
Role 5: run_clustering()
    ↓ StandardScaler → KMeans(k=3)
ClusteringResult（最终结果）
```

### 第一步：build_stock_profiles()

把多只股票多天的数据，聚合成**每只股票一行**，计算 3 个特征：

| 特征 | 计算方式 | 含义 |
|------|---------|------|
| `mean_return` | 日收益率的算术平均 | 平均每天涨多少 |
| `volatility` | 日收益率的标准差（ddof=1） | 涨跌幅度大不大 |
| `max_drawdown` | 回撤的最小值 | 最大亏损幅度 |

这 3 个特征叫 **Stock Profile（股票画像）**。

### 第二步：run_clustering()

1. 用 **StandardScaler** 把 3 个特征标准化（均值=0，标准差=1）
2. 用 **KMeans(k=3)** 聚成 3 类
3. 把聚类中心通过 **inverse_transform** 还原到原始尺度
4. 输出 ClusteringResult

---

## 五、输出是什么

### 输出 1：ClusteringResult（聚类结果）

```python
{
    "profiles": DataFrame,        # 每只股票一行，含 symbol + cluster 列
    "cluster_centers": DataFrame, # 3 个聚类中心，含 3 个特征的原始尺度值
    "features": ["mean_return", "volatility", "max_drawdown"],
    "k": 3
}
```

- `profiles` 里的 `cluster` 列：0、1、2 代表 3 个类别（数字本身没有好坏含义）
- `cluster_centers`：每个类别的中心点，代表该类别的平均特征

### 输出 2：供 Role 1 使用

Role 1 的 Service Layer 调用我的函数，把结果展示在 Streamlit 页面上。

---

## 六、校验规则（我加了什么保护）

| 检查项 | 异常类型 | 说明 |
|--------|---------|------|
| 输入不是 DataFrame | DataValidationError | 类型错误 |
| 缺少列 | DataValidationError | 必须有 symbol, return, drawdown |
| 空 DataFrame | DataValidationError | 没有数据 |
| return 列有 NaN | DataValidationError | 掉首行后检查 |
| return 列有 inf | DataValidationError | 无穷值 |
| 各股票时间区间不一致 | DataValidationError | 起止日期必须相同 |
| 某只股票有效 return < 2 | InsufficientDataError | 太少算不出波动率 |
| 有效 Profile < 3 | InsufficientDataError | 不够做 K-Means |
| 重复 symbol | DataValidationError | 每只股票只能出现一次 |

---

## 七、测试覆盖

39 个测试全部通过，覆盖：

- 正常输入输出（8 个）
- 类型错误（3 个）
- 缺少列（3 个）
- 空数据（1 个）
- 时间区间不一致（2 个）
- 有效 return 不足（1 个）
- 有效 Profile 不足（1 个）
- 数值质量问题（2 个）
- 输入不可变（1 个）
- run_clustering 正常/异常（12 个）
- 端到端（3 个，含真实调用 Role 2 的 build_common_features）
- 10 只股票（2 个）

---

## 八、与 Role 2 的对接

Role 2 提供 `build_common_features()` 函数，输出含 `return` 和 `drawdown` 列的 DataFrame。

我用 `sample_daily.csv`（5 只 A 股 × 78 个交易日）验证了真实对接：

```python
from src.data.features import build_common_features

raw = pd.read_csv("data/sample/sample_daily.csv")
featured = build_common_features(raw)
profiles = build_stock_profiles(featured)  # 成功
result = run_clustering(profiles)          # 成功
```

---

## 九、今日工作（D2）

- 重写 `build_stock_profiles()`：改用 Role 2 的 return 列
- 加强输入校验：类型、空值、时间区间、重复 symbol
- 重写 39 个单元测试
- 合并 Role 2 分支，写真实端到端测试
- 推送到 Gitee，PR #14 已合并
- 修正提示词格式为【我】标记
