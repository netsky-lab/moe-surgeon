"""Domain-specific model backend errors and diagnostic helpers.

The contracts in this module are intentionally lightweight so importing backend
protocols and registries does not pull runtime-heavy ML dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

ShapeLike = tuple[int, ...] | list[int] | None


def _format_shape(shape: ShapeLike) -> str:
    if shape is None:
        return "unknown"
    return "x".join(str(dim) for dim in shape) or "empty"


def _format_detail_value(value: object) -> str:
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return str(value)


def _with_legacy_details(
    details: Mapping[str, object] | None,
    legacy_args: tuple[object, ...],
) -> dict[str, object]:
    merged = {} if details is None else dict(details)
    for index, value in enumerate(legacy_args, start=1):
        merged[f"legacy_arg_{index}"] = value
    return merged


def _normalize_shape(shape: Sequence[int] | None) -> tuple[int, ...] | None:
    if shape is None:
        return None
    return tuple(int(dimension) for dimension in shape)


@dataclass(frozen=True)
class DiagnosticContext:
    """Structured diagnostic context for actionable domain errors."""

    model_id: str | None = None
    backend_name: str | None = None
    layer_index: int | None = None
    tensor_key: str | None = None
    expected_shape: tuple[int, ...] | None = None
    actual_shape: tuple[int, ...] | None = None
    details: Mapping[str, object] = field(default_factory=dict)

    def describe(self) -> str:
        parts: list[str] = []
        if self.model_id is not None:
            parts.append(f"model_id={self.model_id}")
        if self.backend_name is not None:
            parts.append(f"backend={self.backend_name}")
        if self.layer_index is not None:
            parts.append(f"layer_index={self.layer_index}")
        if self.tensor_key is not None:
            parts.append(f"tensor_key={self.tensor_key}")
        if self.expected_shape is not None:
            parts.append(f"expected_shape={_format_shape(self.expected_shape)}")
        if self.actual_shape is not None:
            parts.append(f"actual_shape={_format_shape(self.actual_shape)}")
        for key in sorted(self.details):
            parts.append(f"{key}={_format_detail_value(self.details[key])}")
        return ", ".join(parts)


def format_diagnostic_message(message: str, *, context: DiagnosticContext | None = None) -> str:
    """Attach structured context to a human-readable error message."""

    if context is None:
        return message
    description = context.describe()
    if not description:
        return message
    return f"{message} ({description})"


def build_diagnostic_context(
    *,
    model_id: str | None = None,
    backend_name: str | None = None,
    layer_index: int | None = None,
    tensor_key: str | None = None,
    expected_shape: Sequence[int] | None = None,
    actual_shape: Sequence[int] | None = None,
    details: Mapping[str, object] | None = None,
) -> DiagnosticContext:
    """Create normalized diagnostic context for model-domain errors."""

    return DiagnosticContext(
        model_id=model_id,
        backend_name=backend_name,
        layer_index=layer_index,
        tensor_key=tensor_key,
        expected_shape=_normalize_shape(expected_shape),
        actual_shape=_normalize_shape(actual_shape),
        details={} if details is None else dict(details),
    )


class ModelError(ValueError):
    """Base class for model/backend domain errors."""

    error_code = "model_error"

    def __init__(self, message: str, *, context: DiagnosticContext | None = None) -> None:
        self.context = context
        super().__init__(format_diagnostic_message(message, context=context))


class SchemaValidationError(ModelError):
    """Base error for schema and contract violations."""

    error_code = "schema_validation"


class UnsupportedModelError(ModelError):
    """Raised when no registered backend supports a requested model signature."""

    error_code = "unsupported_model_family"

    def __init__(
        self,
        model_id: str,
        *,
        available_backends: tuple[str, ...] = (),
        details: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(
            "unsupported model family",
            context=build_diagnostic_context(
                model_id=model_id,
                details={
                    "available_backends": ",".join(available_backends) if available_backends else "none",
                    **({} if details is None else dict(details)),
                },
            ),
        )


class BackendMismatchError(ModelError):
    """Raised when backend registration or selection invariants are violated."""

    error_code = "backend_mismatch"

    def __init__(
        self,
        message: str,
        *,
        model_id: str | None = None,
        backend_name: str | None = None,
        details: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(
            message,
            context=build_diagnostic_context(
                model_id=model_id,
                backend_name=backend_name,
                details={} if details is None else dict(details),
            ),
        )


class TopologyMismatchError(SchemaValidationError):
    """Raised when topology-level invariants cannot be satisfied."""

    error_code = "topology_mismatch"

    def __init__(
        self,
        message: str,
        *legacy_args: object,
        model_id: str | None = None,
        layer_index: int | None = None,
        tensor_key: str | None = None,
        expected_shape: tuple[int, ...] | None = None,
        actual_shape: tuple[int, ...] | None = None,
        details: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(
            message,
            context=build_diagnostic_context(
                model_id=model_id,
                layer_index=layer_index,
                tensor_key=tensor_key,
                expected_shape=expected_shape,
                actual_shape=actual_shape,
                details=_with_legacy_details(details, legacy_args),
            ),
        )


class ShapeInvariantViolationError(SchemaValidationError):
    """Raised when tensor-like metadata is malformed."""

    error_code = "shape_invariant_violation"

    def __init__(
        self,
        message: str,
        *legacy_args: object,
        model_id: str | None = None,
        layer_index: int | None = None,
        tensor_key: str | None = None,
        expected_shape: tuple[int, ...] | None = None,
        actual_shape: tuple[int, ...] | None = None,
        details: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(
            message,
            context=build_diagnostic_context(
                model_id=model_id,
                layer_index=layer_index,
                tensor_key=tensor_key,
                expected_shape=expected_shape,
                actual_shape=actual_shape,
                details=_with_legacy_details(details, legacy_args),
            ),
        )


__all__ = [
    "BackendMismatchError",
    "DiagnosticContext",
    "ModelError",
    "SchemaValidationError",
    "ShapeInvariantViolationError",
    "TopologyMismatchError",
    "UnsupportedModelError",
    "build_diagnostic_context",
    "format_diagnostic_message",
]
