"""Linear regression module — Role 6 核心实现 (P0, 动态单股输入)。

对照 Supervised Learning Contract v0.2 与 `docs/role_boundaries.md`：
- 研究框架：X(t) → return(t+1)，只用第 t 个交易日及之前可获得的信息。
- 单次调用处理**一只**股票（由调用方按 symbol + 用户日期范围过滤后传入）。
- 私有目标：next_return = return.shift(-1)。
- 切分：按 trade_date 升序，`split_index = int(n * 0.8)`，前 80% 训练 / 后 20% 测试，不 shuffle。
- P0 指标：`mae`、`r2`（保持 Contract 的 metrics 键不变）。

本模块**只使用 `LinearRegression`**，不引入 Ridge/Lasso/RandomForest/LSTM 等其它算法；
代码不依赖固定股票、固定年份，支持任意合法股票与任意日期范围（函数不抓取行情，
只消费传入的标准公共特征 DataFrame）。

返回结果在 Contract 四个核心键之外，额外提供（作为**顶层键**，不改变 metrics 结构）：
- ``rmse``                  : 均方根误差（sklearn 同口径）
- ``n_original``            : 输入原始交易日数
- ``n_valid``               : 有效样本数（剔除特征窗口 NaN、首日 return NaN、末行无次日目标）
- ``window_loss``           : 因滚动窗口/标签对齐丢失的样本数（n_original - n_valid）
- ``n_train`` / ``n_test``  : 训练 / 测试样本数
- ``warning``               : 异常或需要注意的提示（如测试样本过少、目标近似常量），无则 None

私有字段（next_return）只存在于模型内部，不写回公共 Market DataFrame。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.contracts.supervised import REGRESSION_PREDICTION_COLUMNS
from src.utils.exceptions import DataValidationError, InsufficientDataError

# P0 公共特征（t 日可获得；不含单调的 cumulative_return，避免伪相关/共线主导）。
FEATURE_COLS: list[str] = [
    "return",
    "ma5",
    "ma20",
    "volatility_20d",
    "volume_change",
    "drawdown",
]

SPLIT_RATIO: float = 0.8
# 测试样本少于该阈值 → 对 R² 稳定性给出提示（仍会计算）。
_SMALL_TEST_THRESHOLD: int = 5
# 目标近似常量的判定阈值（极差 < 该值视为常量）。
_CONSTANT_EPS: float = 1e-10


def _prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    """校验并规范化输入（单只股票、升序、数值类型），仅做只读校验。"""
    required = ["symbol", "trade_date", "close", "return"] + FEATURE_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError(f"回归输入缺少必要字段：{missing}")
    if df is None or df.empty:
        raise DataValidationError("回归输入 DataFrame 为空（无数据）")

    out = df.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    for c in FEATURE_COLS + ["close", "return"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def fit_regression(df: pd.DataFrame) -> dict:
    """对一只股票（任意日期范围）的公共特征做次日收益率线性回归。

    输入应为 Role 2 标准公共特征 DataFrame，且已过滤为**单只股票**（可由
    ``build_common_features`` 生成，并由调用方按用户选择的股票与日期范围筛选）。

    Returns:
        一个包含 Contract 核心键（``model/fFeature_names/metrics/predictions``）的字典，
        并额外携带 ``rmse``、``n_original``、``n_valid``、``window_loss``、
        ``n_train``、``n_test``、``warning`` 等元信息（见模块文档）。

    Raises:
        DataValidationError: 字段缺失、日期非升序、非有限值、传入多只股票、无数据。
        InsufficientDataError: 处理 NaN 与次日目标后训练/测试样本为空，或测试集
            样本不足以计算 R²（< 2）。
    """
    prepared = _prepare_frame(df)
    if prepared["symbol"].nunique() != 1:
        raise DataValidationError("单次回归调用只处理一只股票")
    if prepared["trade_date"].isna().any():
        raise DataValidationError("trade_date 存在无法解析的日期")
    if not prepared["trade_date"].is_monotonic_increasing:
        raise DataValidationError("trade_date 必须严格升序")

    # 保证后续窗口/移位按时间顺序计算（防御性重排，不影响已校验的升序）。
    frame = prepared.sort_values("trade_date").reset_index(drop=True)

    n_original = len(frame)

    # 私有目标：下一交易日收益率。首行 return 为 NaN，末行 next_return 为 NaN。
    frame["next_return"] = frame["return"].shift(-1)

    cols = FEATURE_COLS + ["next_return"]
    data = frame.dropna(subset=cols).reset_index(drop=True)
    n_valid = len(data)
    window_loss = n_original - n_valid

    if n_valid == 0:
        raise InsufficientDataError(
            "处理滚动窗口/NaN 与次日目标后没有有效样本（数据可能全为 NaN 或样本过少）"
        )

    split_index = int(n_valid * SPLIT_RATIO)
    X = data[FEATURE_COLS]
    y = data["next_return"].astype(float)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    if len(y_train) == 0:
        raise InsufficientDataError("训练集为空（有效样本过少，80% 切分后无训练样本）")
    if len(y_test) == 0:
        raise InsufficientDataError("测试集为空（有效样本过少，不足以构成最新 20%）")
    if len(y_test) < 2:
        raise InsufficientDataError("测试集样本不足以计算 R²（至少需要 2 个样本）")

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test).astype(float)
    y_true = y_test.values.astype(float)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    # 目标近似常量：R² 无统计意义 → 置 NaN 并提示（而不是报错）。
    warnings = []
    if np.ptp(y_true) < _CONSTANT_EPS:
        r2 = float("nan")
        warnings.append("测试集目标近似常量，R² 无统计意义（已置为 NaN）")
    else:
        r2 = float(r2_score(y_true, y_pred))

    if len(y_test) < _SMALL_TEST_THRESHOLD:
        warnings.append(
            f"测试样本较少（n={len(y_test)} < {_SMALL_TEST_THRESHOLD}），R² 可能不稳定"
        )

    warning = "；".join(warnings) if warnings else None

    predictions = pd.DataFrame(
        {
            "trade_date": data["trade_date"].iloc[split_index:].values,
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )[list(REGRESSION_PREDICTION_COLUMNS)]

    result = {
        "model": model,
        "feature_names": list(FEATURE_COLS),
        "metrics": {"mae": mae, "r2": r2},  # 保持 Contract 的 metrics 键
        "predictions": predictions,
        # 附加元信息（顶层键，不改变 Contract 的 metrics 结构）
        "rmse": rmse,
        "n_original": int(n_original),
        "n_valid": int(n_valid),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "window_loss": int(window_loss),
        "warning": warning,
    }
    return result
