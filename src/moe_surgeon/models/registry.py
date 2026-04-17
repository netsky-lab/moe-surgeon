"""Deterministic registry and resolver for model backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from moe_surgeon.models.backend import BackendSignature, ModelBackend
from moe_surgeon.models.errors import BackendMismatchError, UnsupportedModelError


@dataclass(frozen=True)
class RegisteredBackend:
    """Registry entry with deterministic ordering metadata."""

    name: str
    backend: ModelBackend
    priority: int = 0


class BackendRegistry:
    """Stable in-memory backend registry for model-family resolution."""

    def __init__(self, entries: Iterable[RegisteredBackend] | None = None) -> None:
        self._entries: dict[str, RegisteredBackend] = {}
        if entries is not None:
            for entry in entries:
                self.register(entry.backend, priority=entry.priority)

    def register(self, backend: ModelBackend, *, priority: int = 0) -> None:
        """Register a backend adapter under its unique name."""

        name = getattr(backend, "name", None)
        if not isinstance(name, str) or not name.strip():
            raise BackendMismatchError("backend name must be a non-empty string")
        if name in self._entries:
            raise BackendMismatchError(
                "duplicate backend registration",
                backend_name=name,
                details={"existing_priority": self._entries[name].priority, "new_priority": priority},
            )
        self._entries[name] = RegisteredBackend(name=name, backend=backend, priority=int(priority))

    def names(self) -> tuple[str, ...]:
        """Return registered backend names in deterministic resolver order."""

        return tuple(entry.name for entry in self._sorted_entries())

    def resolve(self, signature: BackendSignature) -> ModelBackend:
        """Resolve exactly one backend for the provided signature."""

        matches = self.matching_entries(signature)
        if not matches:
            raise UnsupportedModelError(
                signature.model_id,
                available_backends=self.names(),
                details={
                    "architecture": signature.architecture or "unknown",
                    "model_type": signature.model_type or "unknown",
                },
            )

        winning_priority = matches[0].priority
        top_matches = [entry for entry in matches if entry.priority == winning_priority]
        if len(top_matches) > 1:
            raise BackendMismatchError(
                "ambiguous backend resolution",
                model_id=signature.model_id,
                details={
                    "candidate_backends": ",".join(entry.name for entry in top_matches),
                    "priority": winning_priority,
                },
            )
        return top_matches[0].backend

    def matching_entries(self, signature: BackendSignature) -> tuple[RegisteredBackend, ...]:
        """Return matching backends sorted by deterministic selection rules."""

        matches: list[RegisteredBackend] = []
        for entry in self._sorted_entries():
            supported = entry.backend.supports(signature)
            if not isinstance(supported, bool):
                raise BackendMismatchError(
                    "backend supports() must return bool",
                    model_id=signature.model_id,
                    backend_name=entry.name,
                )
            if supported:
                matches.append(entry)
        return tuple(matches)

    def _sorted_entries(self) -> tuple[RegisteredBackend, ...]:
        return tuple(sorted(self._entries.values(), key=lambda entry: (-entry.priority, entry.name)))


__all__ = ["BackendRegistry", "RegisteredBackend"]
