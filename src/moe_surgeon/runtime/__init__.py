"""Runtime package for forward hook profiling and prompt tracing."""

from moe_surgeon.runtime.profiler import (
    benchmark,
    BenchmarkResult,
    iter_prompt_batches,
    PromptBatch,
    RouterActivationProfiler,
    RouterActivationRecord,
    RouterCaptureCollector,
)

__all__ = [
    "benchmark",
    "BenchmarkResult",
    "iter_prompt_batches",
    "PromptBatch",
    "RouterActivationProfiler",
    "RouterActivationRecord",
    "RouterCaptureCollector",
]
