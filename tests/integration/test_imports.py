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
    "src.contracts.market_data",
    "src.contracts.supervised",
    "src.contracts.clustering",
    "src.services.market_service",
    "src.services.analysis_service",
    "src.services.clustering_service",
    "src.services.supervised_service",
    "src.services.workspace_service",
    "src.utils.exceptions",
)


def test_project_modules_import_without_cycles() -> None:
    for module_name in MODULES:
        assert importlib.import_module(module_name) is not None
