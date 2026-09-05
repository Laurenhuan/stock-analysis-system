"""Static guards for the agreed Streamlit -> Service -> Domain direction."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PAGE_FILES = [ROOT / "app.py", *(ROOT / "app_pages").glob("*.py")]
DOMAIN_DIRS = (
    ROOT / "src" / "data",
    ROOT / "src" / "analysis",
    ROOT / "src" / "models",
    ROOT / "src" / "visualization",
)
FORBIDDEN_PAGE_PREFIXES = (
    "src.data",
    "src.analysis",
    "src.models",
    "src.visualization",
    "src.contracts",
)


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def test_streamlit_pages_do_not_import_domain_modules() -> None:
    violations = {
        path.name: [
            name
            for name in _imports(path)
            if name.startswith(FORBIDDEN_PAGE_PREFIXES)
        ]
        for path in PAGE_FILES
    }

    assert not {path: names for path, names in violations.items() if names}


def test_domain_modules_do_not_import_streamlit() -> None:
    domain_files = [
        path
        for directory in DOMAIN_DIRS
        for path in directory.rglob("*.py")
    ]
    violations = [
        str(path.relative_to(ROOT))
        for path in domain_files
        if "streamlit" in _imports(path)
    ]

    assert violations == []


def test_clustering_page_does_not_offer_variable_k() -> None:
    page = (ROOT / "app_pages" / "multi_stock.py").read_text(
        encoding="utf-8"
    )

    assert "n_clusters" not in page
    assert "slider(" not in page
    assert "KMeans(k=3)" in page
