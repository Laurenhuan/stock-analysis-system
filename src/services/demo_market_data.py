"""TEMPORARY / DAY-1 PROTOTYPE sample-data adapter.

This module only loads the small synthetic CSV used by Role 1 to prove the
Streamlit integration flow. Role 2 will replace it with the formal data source.
"""

from pathlib import Path

import pandas as pd
from pandas import DataFrame

from src.contracts.market_data import BASE_MARKET_COLUMNS
from src.utils.exceptions import DataValidationError


DEFAULT_DEMO_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "sample"
    / "day1_demo_market_data.csv"
)


def load_demo_market_data(csv_path: Path = DEFAULT_DEMO_PATH) -> DataFrame:
    """Load the small Day 1 CSV and return its standard base columns."""
    data = pd.read_csv(csv_path, parse_dates=["trade_date"])

    missing_columns = set(BASE_MARKET_COLUMNS) - set(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise DataValidationError(f"Demo 数据缺少字段：{missing}")

    return data.loc[:, BASE_MARKET_COLUMNS].sort_values(
        ["symbol", "trade_date"]
    )

