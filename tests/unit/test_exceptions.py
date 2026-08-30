"""Tests for the lightweight shared exception hierarchy."""

import pytest

from src.utils.exceptions import (
    DataValidationError,
    InsufficientDataError,
    InvalidSymbolError,
    NoDataError,
    StockAnalysisError,
)


@pytest.mark.parametrize(
    "exception_type",
    [InvalidSymbolError, NoDataError, DataValidationError, InsufficientDataError],
)
def test_domain_exceptions_share_a_common_base(exception_type: type[Exception]) -> None:
    with pytest.raises(StockAnalysisError, match="contract failure"):
        raise exception_type("contract failure")
