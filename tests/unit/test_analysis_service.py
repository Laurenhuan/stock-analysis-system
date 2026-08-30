"""Unit tests for the initial service skeleton."""

from src.services.analysis_service import get_analysis_status


def test_all_analysis_modules_are_pending() -> None:
    statuses = get_analysis_status()

    assert statuses
    assert set(statuses.values()) == {"pending"}
