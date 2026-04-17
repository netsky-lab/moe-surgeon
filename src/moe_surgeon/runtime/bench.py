"""Compatibility wrapper for runtime benchmark profiling."""

from moe_surgeon.runtime.profiler import (
    benchmark,
    BenchmarkResult,
    RouterActivationProfiler,
    RouterActivationRecord,
    RouterCaptureCollector,
)

__all__ = [
    "benchmark",
    "BenchmarkResult",
    "RouterActivationProfiler",
    "RouterActivationRecord",
    "RouterCaptureCollector",
]
