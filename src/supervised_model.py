import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)

# ====================== 特征工程函数 ======================
def build_features(df):
    """
    构造金融特征与标签
    df: 原始数据，至少包含 close, volume
    return: 处理完成数据集，特征+标签
    """
    data = df.copy()
    # 1. 收益率
    data["ret_1d"] = data["close"].pct_change()
    # 特征：历史收益
    data["ret_lag1"] = data["ret_1d"].shift(1)
    data["ret_lag2"] = data["ret_1d"].shift(2)

    # MA移动平均线
    data["ma5"] = data["close"].rolling(window=5).mean()
    data["ma10"] = data["close"].rolling(window=10).mean()
    data["ma_diff"] = data["ma5"] - data["ma10"]

    # 滚动波动率 5日
    data["volatility_5"] = data["ret_1d"].rolling(window=5).std()

    # 成交量变化
    data["vol_change"] = data["volume"].pct_change()

    # -------- 标签构建（关键！shift(-1)取下一日，未来数据） --------
    # 分类标签：下一日涨跌方向
    data["y_class"] = np.where(data["ret_1d"].shift(-1) > 0, 1, 0)
    # 回归标签：下一日收益率数值
    data["y_reg"] = data["ret_1d"].shift(-1)

    # 删除NaN
    data = data.dropna()
    return data


# ====================== A.决策树分类 run_classification ======================
def run_classification(data, train_ratio=0.7):
    """
    决策树分类：预测股票涨跌 Up/Down
    param data: build_features生成数据集
    param train_ratio: 训练集时间占比，时序切分，不shuffle
    return: model, metrics_dict
    """
    feature_cols = ["ret_lag1", "ret_lag2", "ma_diff", "volatility_5", "vol_change"]
    X = data[feature_cols]
    y = data["y_class"]

    # 时间顺序切分，禁止随机打乱！！
    split_idx = int(len(X) * train_ratio)
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    # 建立决策树模型，基础参数
    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # 模型评价
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    metrics = {
        "Accuracy": acc,
        "Confusion Matrix": cm,
        "Precision": prec,
        "Recall": rec,
        "F1": f1
    }

    print("===== 决策树分类模型评价 =====")
    for k, v in metrics.items():
        print(f"{k}:\n{v}\n")

    # 绘制决策树，用于模型可解释性
    plt.figure(figsize=(14, 7), dpi=100)
    plot_tree(model, feature_names=feature_cols, class_names=["Down","Up"], filled=True, fontsize=8)
    plt.title("Decision Tree Classifier")
    plt.tight_layout()
    plt.show()

    return model, metrics


# ====================== B.线性回归 run_regression ======================
def run_regression(data, train_ratio=0.7):
    """
    线性回归：预测下一日收益率
    """
    feature_cols = ["ret_lag1", "ret_lag2", "ma_diff", "volatility_5", "vol_change"]
    X = data[feature_cols]
    y = data["y_reg"]

    # 时序切分
    split_idx = int(len(X)*train_ratio)
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # 评价指标
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "MAE": mae,
        "MSE": mse,
        "R2": r2,
        "coef": dict(zip(feature_cols, model.coef_)),
        "intercept": model.intercept_
    }

    print("===== 线性回归模型评价 =====")
    print(f"MAE:{mae:.6f}")
    print(f"MSE:{mse:.6f}")
    print(f"R2:{r2:.4f}")
    print("\n特征系数：")
    for fname, c in metrics["coef"].items():
        print(f"{fname}: {c:.6f}")
    print(f"截距 intercept: {metrics['intercept']:.6f}")

    # 真实值 vs 预测值对比图
    plt.figure(figsize=(12,4))
    plt.plot(y_test.values, label="真实收益率", alpha=0.7)
    plt.plot(y_pred, label="预测收益率", alpha=0.7)
    plt.legend()
    plt.title("Linear Regression: True vs Predict Return")
    plt.tight_layout()
    plt.show()

    return model, metrics


# ======================== 使用示例 ========================
if __name__ == "__main__":
    # 模拟样例：你替换成真实股票数据（至少需要 close、volume）
    # df = pd.read_csv("stock.csv")
    # df 需要包含列：date, close, volume
    np.random.seed(42)
    n = 500
    price = 100 + np.cumsum(np.random.randn(n)*0.8)
    volume = np.random.randint(1000, 5000, size=n)
    df_sim = pd.DataFrame({"close":price, "volume":volume})

    dataset = build_features(df_sim)
    dt_model, dt_metrics = run_classification(dataset, train_ratio=0.7)
    lr_model, lr_metrics = run_regression(dataset, train_ratio=0.7)
