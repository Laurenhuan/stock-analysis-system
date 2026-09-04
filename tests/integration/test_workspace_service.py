"""Integration tests for Role 1 workspace-facing helpers."""

import pandas as pd
import pytest

from src.services import (
    get_market_summary,
    get_model_sample_summary,
    prepare_symbol_selection,
)
from src.utils.exceptions import DataValidationError, NoDataError


def test_prepare_symbol_selection_normalizes_and_deduplicates() -> None:
    result = prepare_symbol_selection(
        [" 600519.sh ", "000001.SZ", "600519.SH"],
        min_count=2,
        max_count=20,
    )

    assert result == ["600519.SH", "000001.SZ"]


def test_prepare_symbol_selection_accepts_common_delimiters() -> None:
    result = prepare_symbol_selection(
        "600519.SH，000001.SZ; 300750.sz",
        min_count=3,
        max_count=3,
    )

    assert result == ["600519.SH", "000001.SZ", "300750.SZ"]


@pytest.mark.parametrize(
    ("values", "min_count", "max_count", "message"),
    [
        ([], 1, 20, "至少需要选择 1 只股票"),
        (["600519.SH"], 2, 20, "至少需要选择 2 只股票"),
        (
            ["600519.SH", "000001.SZ", "300750.SZ"],
            1,
            2,
            "最多只能选择 2 只股票",
        ),
    ],
)
def test_prepare_symbol_selection_enforces_count_bounds(
    values, min_count, max_count, message
) -> None:
    with pytest.raises(DataValidationError, match=message):
        prepare_symbol_selection(
            values,
            min_count=min_count,
            max_count=max_count,
        )


def test_market_summary_reports_loaded_frame_facts() -> None:
    data = pd.DataFrame(
        {
            "symbol": ["600519.SH", "600519.SH", "000001.SZ"],
            "trade_date": [
                "2024-01-02",
                "2024-01-03",
                "2024-01-03",
            ],
        }
    )

    assert get_market_summary(data) == {
        "row_count": 3,
        "symbol_count": 2,
        "first_date": "2024-01-02",
        "last_date": "2024-01-03",
    }


def test_market_summary_rejects_empty_data() -> None:
    with pytest.raises(NoDataError, match="行情数据为空"):
        get_market_summary(pd.DataFrame())


def test_model_sample_summary_does_not_infer_private_training_rows() -> None:
    input_data = pd.DataFrame({"row": range(100)})
    predictions = pd.DataFrame(
        {
            "trade_date": ["2024-10-29", "2024-10-30"],
            "y_true": [0, 1],
            "y_pred": [1, 1],
        }
    )

    assert get_model_sample_summary(input_data, predictions) == {
        "input_rows": 100,
        "test_rows": 2,
        "test_date_range": "2024-10-29 至 2024-10-30",
    }


def test_model_sample_summary_rejects_empty_predictions() -> None:
    with pytest.raises(NoDataError, match="模型未返回"):
        get_model_sample_summary(pd.DataFrame({"row": [1]}), pd.DataFrame())
