"""Tests for truthful Role 1 module-readiness reporting."""

from src.services.analysis_service import get_analysis_status


def test_analysis_status_matches_integrated_modules() -> None:
    assert get_analysis_status() == {
        "eda": "ready",
        "classification": "pending",
        "regression": "ready",
        "clustering": "ready",
    }
