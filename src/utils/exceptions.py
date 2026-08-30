"""Project-specific exception types."""


class StockAnalysisError(Exception):
    """Base exception for expected application-level errors."""


class ContractNotImplementedError(StockAnalysisError):
    """Raised when a shared Contract intentionally has no implementation yet."""


class InvalidSymbolError(StockAnalysisError):
    """Raised when a security symbol is malformed or unsupported."""


class NoDataError(StockAnalysisError):
    """Raised when a valid request produces no market data."""


class DataValidationError(StockAnalysisError):
    """Raised when data violates a shared schema or quality rule."""


class InsufficientDataError(StockAnalysisError):
    """Raised when valid observations are insufficient for an analysis."""
