"""Isolate app-import side effects before collecting unittest or pytest modules.

FastAPI's module-level app constructs its SQLite registry at import time. This
bootstrap deliberately precedes collection, so tests never recover or modify a
developer's real .runtime/documents, Chroma store or document registry.
"""

import os
from pathlib import Path
import shutil
import sys
from uuid import uuid4

import pytest


TEST_ROOT = Path(__file__).resolve().parents[1] / ".runtime" / "tests"
BOOTSTRAP_ROOT = TEST_ROOT / f"pytest-bootstrap-{uuid4()}"
BOOTSTRAP_ROOT.mkdir(parents=True)
_STORAGE_ENV = {
    "DOCUMENT_STORAGE_DIR": str(BOOTSTRAP_ROOT / "documents"),
    "CHROMA_STORAGE_DIR": str(BOOTSTRAP_ROOT / "chroma"),
    "DOCUMENT_REGISTRY_PATH": str(BOOTSTRAP_ROOT / "registry" / "documents.sqlite3"),
}
_ORIGINAL_ENV = {name: os.environ.get(name) for name in _STORAGE_ENV}
os.environ.update(_STORAGE_ENV)


def _remove_owned_root(root):
    # Validate the resolved absolute path before recursive deletion on Windows.
    resolved = root.resolve()
    if root.is_symlink() or resolved != root or resolved.parent != TEST_ROOT.resolve():
        raise RuntimeError(f"Refusing to clean a path outside the isolated test directory: {root}")
    if root.exists():
        shutil.rmtree(root)


@pytest.fixture
def isolated_root():
    """A function-owned Path; close any Chroma clients before fixture teardown."""
    root = TEST_ROOT / f"pytest-{uuid4()}"
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        _remove_owned_root(root)


def pytest_sessionfinish(session, exitstatus):
    try:
        module = sys.modules.get("local_rag.api.app")
        if module is not None:
            module.app.state.index_service.close()
        _remove_owned_root(BOOTSTRAP_ROOT)
    finally:
        for name, original in _ORIGINAL_ENV.items():
            if original is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = original
