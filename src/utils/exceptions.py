"""Project-specific exception types."""


class StockAnalysisError(Exception):
    """Base exception for expected application-level errors."""


class ContractNotImplementedError(StockAnalysisError):
    """Raised when a draft cross-team contract has no implementation yet."""
