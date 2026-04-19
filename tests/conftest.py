from __future__ import annotations

import importlib
from pathlib import Path
import sys

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"

if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

# The shared environment can point editable installs at sibling worktrees.
# Force pytest collection in this checkout to resolve `moe_surgeon` from this
# repo's own `src/` tree instead of whichever editable install was last active.
for module_name in tuple(sys.modules):
    if module_name == "moe_surgeon" or module_name.startswith("moe_surgeon."):
        del sys.modules[module_name]

importlib.invalidate_caches()


def _tiny_fixture_module():
    return importlib.import_module("tests.fixtures.tiny_gemma_like")


@pytest.fixture
def tiny_fixture_state_dict() -> dict[str, object]:
    return _tiny_fixture_module().tiny_state_dict()


@pytest.fixture
def tiny_fixture_topology():
    return _tiny_fixture_module().tiny_topology()


@pytest.fixture
def tiny_fixture_router_states():
    return _tiny_fixture_module().tiny_router_states()


@pytest.fixture
def tiny_fixture_bundle(tiny_fixture_state_dict: dict[str, object]):
    return _tiny_fixture_module().tiny_bundle(state_dict=tiny_fixture_state_dict)


@pytest.fixture
def tiny_fixture_backend(tiny_fixture_state_dict: dict[str, object]):
    return _tiny_fixture_module().TinyMockBackend(state_dict=tiny_fixture_state_dict)


@pytest.fixture
def tiny_fixture_expert_stats():
    return _tiny_fixture_module().tiny_expert_stats()


@pytest.fixture
def tiny_fixture_activation_stats():
    return _tiny_fixture_module().tiny_activation_stats()
