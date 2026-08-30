"""Integration smoke tests for package boundaries."""

import importlib


MODULES = (
    "src.data.fetch",
    "src.data.clean",
    "src.data.features",
    "src.analysis.eda",
    "src.models.supervised.classification",
    "src.models.supervised.regression",
    "src.models.unsupervised.clustering",
    "src.visualization.charts",
    "src.services.market_service",
    "src.services.analysis_service",
)


def test_project_modules_import_without_cycles() -> None:
    for module_name in MODULES:
        assert importlib.import_module(module_name) is not None
