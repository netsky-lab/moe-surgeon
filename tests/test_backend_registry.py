from __future__ import annotations

from dataclasses import dataclass

from moe_surgeon.models.backend import BackendSignature
from moe_surgeon.models.registry import BackendRegistry

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
