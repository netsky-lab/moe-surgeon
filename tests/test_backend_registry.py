from __future__ import annotations

from dataclasses import dataclass

import pytest

from moe_surgeon.models.backend import BackendSignature
from moe_surgeon.models.errors import BackendMismatchError
from moe_surgeon.models.registry import BackendRegistry, RegisteredBackend

from tests.fixtures.tiny_gemma_like import TinyMockBackend


@dataclass
class _NamedTinyBackend(TinyMockBackend):
    name: str = "tiny"


def test_backend_registry_resolves_shared_tiny_backend_from_mapping() -> None:
    registry = BackendRegistry()
    backend = _NamedTinyBackend(name="tiny-primary")
    registry.register(backend, priority=10)

    resolved = registry.resolve(
        {
            "architectures": ["Gemma4ForConditionalGeneration"],
            "model_type": "gemma4",
        },
        model_id="tests/tiny-gemma-like",
        source_path="/tmp/tiny-gemma-like",
    )

    assert resolved is backend


def test_backend_registry_matching_entries_keep_deterministic_priority_order() -> None:
    registry = BackendRegistry()
    low = _NamedTinyBackend(name="tiny-low")
    high = _NamedTinyBackend(name="tiny-high")
    registry.register(low, priority=5)
    registry.register(high, priority=20)

    matches = registry.matching_entries(
        BackendSignature(model_id="tests/tiny-gemma-like", model_type="gemma4")
    )

    assert tuple(entry.name for entry in matches) == ("tiny-high", "tiny-low")
    assert registry.resolve(BackendSignature(model_id="tests/tiny-gemma-like", model_type="gemma4")) is high


def test_backend_registry_constructor_preserves_resolver_order() -> None:
    low = _NamedTinyBackend(name="tiny-low")
    high = _NamedTinyBackend(name="tiny-high")

    registry = BackendRegistry(
        entries=(
            RegisteredBackend(name=low.name, backend=low, priority=5),
            RegisteredBackend(name=high.name, backend=high, priority=20),
        )
    )

    assert registry.names() == ("tiny-high", "tiny-low")


@dataclass
class _InvalidSupportsTinyBackend(_NamedTinyBackend):
    def supports(self, signature: object) -> str:  # type: ignore[override]
        del signature
        return "yes"


def test_backend_registry_rejects_non_boolean_supports_result() -> None:
    registry = BackendRegistry()
    registry.register(_InvalidSupportsTinyBackend(name="tiny-invalid"), priority=1)

    with pytest.raises(BackendMismatchError, match="supports\\(\\) must return bool"):
        registry.matching_entries(BackendSignature(model_id="tests/tiny-gemma-like", model_type="gemma4"))
