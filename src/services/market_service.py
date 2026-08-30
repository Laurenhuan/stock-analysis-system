"""Application-facing market data service contract."""

from src.utils.exceptions import ContractNotImplementedError


def get_market_overview() -> None:
    """Return a future market overview after Role 2 integration.

    Raises:
        ContractNotImplementedError: The data implementation is absent.
    """
    raise ContractNotImplementedError("市场数据模块尚未接入")
