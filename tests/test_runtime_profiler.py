from __future__ import annotations

from dataclasses import dataclass, field
import importlib
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from huggingface_hub import hf_hub_download

from moe_surgeon.models.backend import BackendSignature, LoadedBackendBundle
from moe_surgeon.models.errors import ShapeInvariantViolationError, TopologyMismatchError
from moe_surgeon.models.gemma4 import Gemma4Backend
from moe_surgeon.runtime.bench import RouterActivationProfiler, benchmark, iter_prompt_batches
from moe_surgeon.schemas import LayerTopology, ModelHandle, RouterState

_LIVE_GEMMA4_MODEL_ID = "tiny-random/gemma-4-moe"
_LIVE_GEMMA4_REVISION = "4142709ae44d9bbf3aa363cc4632d4dc4ce4f2a0"


@dataclass
class FakeTensor:
    shape: tuple[int, ...]
    data: object | None = None

    def tolist(self) -> object:
        if self.data is None:
            raise AssertionError("test tensor missing list data")
        return self.data


@dataclass
class FakeHandle:
    module: "FakeRouterModule"
    hook: object
    removed: bool = False

    def remove(self) -> None:
        if self.removed:
            return
        self.removed = True
        self.module.hooks.remove(self.hook)


@dataclass
class FakeRouterModule:
    name: str
    hooks: list[object] = field(default_factory=list)

    def register_forward_hook(self, hook: object) -> FakeHandle:
        self.hooks.append(hook)
        return FakeHandle(module=self, hook=hook)

    def run(self, output: object) -> None:
        for hook in list(self.hooks):
            hook(self, (), output)


class FakeBackend:
    def __init__(self, router_states: dict[int, RouterState], modules: dict[int, FakeRouterModule]) -> None:
        self._router_states = router_states
        self._modules = modules

    def extract_router_state(self, bundle: LoadedBackendBundle, *, layer: LayerTopology) -> RouterState:
        return self._router_states[layer.layer_index]

    def validate_layer(
        self,
        bundle: LoadedBackendBundle,
        *,
        layer: LayerTopology,
        router_state: RouterState | None = None,
    ) -> None:
        active = router_state or self.extract_router_state(bundle, layer=layer)
        if active.top_k != layer.top_k:
            raise AssertionError("test fixture mismatch")

    def resolve_router_module(self, bundle: LoadedBackendBundle, *, layer: LayerTopology) -> FakeRouterModule:
        return self._modules[layer.layer_index]


@dataclass
class FakeTokenizer:
    def __call__(self, prompts: tuple[str, ...], **_: object) -> dict[str, object]:
        max_length = max(len(prompt) for prompt in prompts)
        attention_mask = []
        input_ids = []
        for prompt in prompts:
            token_count = len(prompt)
            padding = max_length - token_count
            attention_mask.append(([1] * token_count) + ([0] * padding))
            input_ids.append(list(range(token_count)) + ([0] * padding))
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def _layer(layer_index: int) -> LayerTopology:
    return LayerTopology(
        layer_index=layer_index,
        layer_name=f"layer-{layer_index}",
        layer_type="fake_moe",
        expert_count=4,
        top_k=2,
        hidden_size=16,
        layer_ref=f"layer_{layer_index}",
        module_paths={"router_proj": f"model.layers.{layer_index}.router.proj.weight"},
    )


def _router_state(layer_index: int) -> RouterState:
    return RouterState(
        layer_index=layer_index,
        num_experts=4,
        top_k=2,
        logits_shape=(0, 4),
        top_k_indices_shape=(1, 2),
        top_k_weights_shape=(1, 2),
        has_raw_logits_capture=True,
    )


def _bundle() -> LoadedBackendBundle:
    return LoadedBackendBundle(
        backend_name="fake",
        model_handle=ModelHandle(model_id="fake-model", backend_name="fake"),
        model=SimpleNamespace(),
        config={},
    )


def _require_live_gemma4_runtime() -> tuple[object, object]:
    try:
        transformers = importlib.import_module("transformers")
        importlib.import_module("transformers.models.gemma4")
    except Exception:
        version = importlib_metadata.version("transformers")
        pytest.skip(
            f"live Gemma4 profiler coverage requires a transformers build with gemma4 support; "
            f"installed version is {version}"
        )

    model_class = getattr(transformers, "Gemma4ForConditionalGeneration", None)
    if model_class is None:
        version = importlib_metadata.version("transformers")
        pytest.skip(
            f"live Gemma4 profiler coverage requires transformers.Gemma4ForConditionalGeneration; "
            f"installed version is {version}"
        )
    return transformers, model_class


def _live_gemma4_signature() -> BackendSignature:
    config_path = hf_hub_download(
        repo_id=_LIVE_GEMMA4_MODEL_ID,
        revision=_LIVE_GEMMA4_REVISION,
        filename="config.json",
    )
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    return BackendSignature.from_mapping(
        config,
        model_id=_LIVE_GEMMA4_MODEL_ID,
        source_path=_LIVE_GEMMA4_MODEL_ID,
    )


def test_router_activation_profiler_collects_outputs_and_detaches_hooks() -> None:
    topology = (_layer(1), _layer(0))
    router_states = {layer.layer_index: _router_state(layer.layer_index) for layer in topology}
    modules = {layer.layer_index: FakeRouterModule(name=layer.layer_name) for layer in topology}
    backend = FakeBackend(router_states=router_states, modules=modules)

    with RouterActivationProfiler(backend=backend, bundle=_bundle(), topology=topology) as profiler:
        assert profiler.attached
        modules[0].run(
            {
                "top_k_indices": FakeTensor((3, 2)),
                "top_k_weights": FakeTensor((3, 2)),
                "router_scores": FakeTensor((3, 4)),
            }
        )
        modules[1].run(
            SimpleNamespace(
                top_k_indices=FakeTensor((2, 2)),
                top_k_weights=FakeTensor((2, 2)),
                router_scores=FakeTensor((2, 4)),
            )
        )
        assert [record.layer_index for record in profiler.records] == [0, 1]

    assert not profiler.attached
    assert all(not module.hooks for module in modules.values())


def test_router_activation_profiler_normalizes_gemma4_tuple_output_order() -> None:
    layer = _layer(0)
    module = FakeRouterModule(name="layer-0")
    backend = FakeBackend(router_states={0: _router_state(0)}, modules={0: module})

    with RouterActivationProfiler(
        backend=backend,
        bundle=_bundle(),
        topology=(layer,),
        include_router_scores=True,
    ) as profiler:
        module.run(
            (
                FakeTensor(shape=(2, 4)),
                FakeTensor(shape=(2, 2), data=[[0.7, 0.3], [0.6, 0.4]]),
                FakeTensor(shape=(2, 2), data=[[1, 3], [0, 2]]),
            )
        )

        assert len(profiler.records) == 1
        record = profiler.records[0]
        assert record.layer_index == 0
        assert record.router_scores is not None
        assert cast(FakeTensor, record.router_scores).shape == (2, 4)
        assert cast(FakeTensor, record.top_k_weights).shape == (2, 2)
        assert cast(FakeTensor, record.top_k_indices).shape == (2, 2)

        stats = profiler.accumulate(attention_mask=[1, 1])

    by_expert = {item.expert_index: item for item in stats if item.layer_index == 0}
    assert by_expert[0].token_count == 1
    assert by_expert[1].token_count == 1
    assert by_expert[2].token_count == 1
    assert by_expert[3].token_count == 1


def test_router_activation_profiler_detaches_hooks_after_failure() -> None:
    layer = _layer(0)
    module = FakeRouterModule(name="layer-0")
    backend = FakeBackend(router_states={0: _router_state(0)}, modules={0: module})
    profiler = RouterActivationProfiler(backend=backend, bundle=_bundle(), topology=(layer,))

    with pytest.raises(RuntimeError, match="boom"):
        with profiler:
            raise RuntimeError("boom")

    assert not profiler.attached
    assert module.hooks == []


def test_router_activation_profiler_rejects_invalid_top_k_output_shape() -> None:
    layer = _layer(0)
    module = FakeRouterModule(name="layer-0")
    backend = FakeBackend(router_states={0: _router_state(0)}, modules={0: module})

    with RouterActivationProfiler(backend=backend, bundle=_bundle(), topology=(layer,)) as profiler:
        with pytest.raises(ShapeInvariantViolationError, match="top_k_indices last dimension mismatch"):
            module.run(
                {
                    "top_k_indices": FakeTensor((2, 3)),
                    "top_k_weights": FakeTensor((2, 2)),
                }
            )

        assert profiler.attached


def test_router_activation_profiler_accepts_sequence_output_without_optional_scores() -> None:
    layer = _layer(0)
    module = FakeRouterModule(name="layer-0")
    backend = FakeBackend(router_states={0: _router_state(0)}, modules={0: module})

    with RouterActivationProfiler(backend=backend, bundle=_bundle(), topology=(layer,)) as profiler:
        module.run(
            (
                [[0.7, 0.3], [0.6, 0.4]],
                [[0, 1], [2, 3]],
            )
        )
        stats = profiler.accumulate(attention_mask=[1, 1])

    by_expert = {item.expert_index: item for item in stats if item.layer_index == 0}
    assert by_expert[0].token_count == 1
    assert by_expert[1].token_count == 1
    assert by_expert[2].token_count == 1
    assert by_expert[3].token_count == 1


def test_router_activation_profiler_rejects_missing_requested_router_scores() -> None:
    layer = _layer(0)
    module = FakeRouterModule(name="layer-0")
    backend = FakeBackend(router_states={0: _router_state(0)}, modules={0: module})

    with RouterActivationProfiler(
        backend=backend,
        bundle=_bundle(),
        topology=(layer,),
        include_router_scores=True,
    ):
        with pytest.raises(
            ShapeInvariantViolationError,
            match=r"router scores capture requested but not present in router output \(model_id=fake-model, layer_index=0\)",
        ):
            module.run(([[0.7, 0.3]], [[0, 1]]))


def test_router_activation_profiler_reports_tuple_length_diagnostics() -> None:
    layer = _layer(0)
    module = FakeRouterModule(name="layer-0")
    backend = FakeBackend(router_states={0: _router_state(0)}, modules={0: module})

    with RouterActivationProfiler(backend=backend, bundle=_bundle(), topology=(layer,)):
        with pytest.raises(
            ShapeInvariantViolationError,
            match=r"router forward output sequence must contain 2 or 3 items \(model_id=fake-model, layer_index=0, output_type=tuple, sequence_length=1\)",
        ):
            module.run(([[0, 1]],))


def test_router_activation_profiler_reports_invalid_output_container_diagnostics() -> None:
    layer = _layer(0)
    module = FakeRouterModule(name="layer-0")
    backend = FakeBackend(router_states={0: _router_state(0)}, modules={0: module})

    with RouterActivationProfiler(backend=backend, bundle=_bundle(), topology=(layer,)):
        with pytest.raises(
            ShapeInvariantViolationError,
            match=r"router forward output must be mapping, object, or sequence \(model_id=fake-model, layer_index=0, output_type=int\)",
        ):
            module.run(7)


def test_router_activation_profiler_aggregates_with_padding_mask() -> None:
    layer = _layer(0)
    module = FakeRouterModule(name="layer-0")
    backend = FakeBackend(router_states={0: _router_state(0)}, modules={0: module})

    with RouterActivationProfiler(backend=backend, bundle=_bundle(), topology=(layer,)) as profiler:
        module.run(
            {
                "top_k_indices": [
                    [[0, 1], [1, 2], [2, 3]],
                    [[3, 0], [0, 2], [1, 3]],
                ],
                "top_k_weights": [
                    [[0.7, 0.3], [0.6, 0.4], [0.5, 0.5]],
                    [[0.9, 0.1], [0.8, 0.2], [0.55, 0.45]],
                ],
            }
        )

        stats = profiler.accumulate(attention_mask=[[1, 1, 0], [1, 0, 0]])

    by_expert = {item.expert_index: item for item in stats if item.layer_index == 0}
    assert by_expert[0].token_count == 2
    assert by_expert[0].weighted_token_count == pytest.approx(0.8)
    assert by_expert[0].weighted_n_tokens == pytest.approx(3.0)
    assert by_expert[0].top1_mass == pytest.approx(0.7)
    assert by_expert[1].token_count == 2
    assert by_expert[2].token_count == 1
    assert by_expert[3].token_count == 1
    assert all(item.n_tokens == 3 for item in by_expert.values())


def test_router_activation_profiler_accumulates_repeated_runs_deterministically() -> None:
    layer = _layer(0)
    module = FakeRouterModule(name="layer-0")
    backend = FakeBackend(router_states={0: _router_state(0)}, modules={0: module})

    with RouterActivationProfiler(backend=backend, bundle=_bundle(), topology=(layer,)) as profiler:
        for _ in range(2):
            module.run(
                {
                    "top_k_indices": [[[0, 1]]],
                    "top_k_weights": [[[0.6, 0.4]]],
                }
            )
            profiler.accumulate(attention_mask=[[1]])

    by_expert = {item.expert_index: item for item in profiler.activation_stats() if item.layer_index == 0}
    assert by_expert[0].token_count == 2
    assert by_expert[1].token_count == 2
    assert by_expert[0].weighted_token_count == pytest.approx(1.2)
    assert by_expert[1].weighted_token_count == pytest.approx(0.8)
    assert all(item.n_tokens == 2 for item in by_expert.values())


def test_router_activation_profiler_aligns_generation_step_to_mask_tail() -> None:
    layer = _layer(0)
    module = FakeRouterModule(name="layer-0")
    backend = FakeBackend(router_states={0: _router_state(0)}, modules={0: module})

    with RouterActivationProfiler(backend=backend, bundle=_bundle(), topology=(layer,)) as profiler:
        module.run(
            {
                "top_k_indices": [
                    [[0, 1]],
                    [[2, 3]],
                ],
                "top_k_weights": [
                    [[0.6, 0.4]],
                    [[0.75, 0.25]],
                ],
            }
        )

        stats = profiler.accumulate(attention_mask=[[1, 1, 1], [1, 1, 0]])

    by_expert = {item.expert_index: item for item in stats if item.layer_index == 0}
    assert all(item.n_tokens == 1 for item in by_expert.values())
    assert all(item.weighted_n_tokens == pytest.approx(1.0) for item in by_expert.values())
    assert by_expert[0].token_count == 1
    assert by_expert[1].token_count == 1
    assert by_expert[2].token_count == 0
    assert by_expert[3].token_count == 0


def test_router_activation_profiler_rejects_out_of_range_expert_indices() -> None:
    layer = _layer(0)
    module = FakeRouterModule(name="layer-0")
    backend = FakeBackend(router_states={0: _router_state(0)}, modules={0: module})

    with RouterActivationProfiler(backend=backend, bundle=_bundle(), topology=(layer,)) as profiler:
        module.run(
            {
                "top_k_indices": [[[0, 4]]],
                "top_k_weights": [[[0.6, 0.4]]],
            }
        )
        with pytest.raises(
            TopologyMismatchError,
            match=r"captured expert index is out of range \(model_id=fake-model, layer_index=0, expert_count=4, expert_index=4\)",
        ):
            profiler.accumulate(attention_mask=[[1]])


def test_benchmark_builds_deterministic_manifest_and_sorted_stats() -> None:
    topology = (_layer(1), _layer(0))
    modules = {layer.layer_index: FakeRouterModule(name=layer.layer_name) for layer in topology}
    backend = FakeBackend(
        router_states={layer.layer_index: _router_state(layer.layer_index) for layer in topology},
        modules=modules,
    )

    with RouterActivationProfiler(backend=backend, bundle=_bundle(), topology=topology) as profiler:
        modules[0].run(
            {
                "top_k_indices": [[[0, 1]]],
                "top_k_weights": [[[0.6, 0.4]]],
            }
        )
        modules[1].run(
            {
                "top_k_indices": [[[2, 3]]],
                "top_k_weights": [[[0.8, 0.2]]],
            }
        )
        profiler.accumulate(attention_mask=[[1]])
        first = benchmark(
            profiler=profiler,
            prompts=("alpha", "beta"),
            profiler_config={"capture_router_scores": False, "batch_size": 2},
        )
        second = benchmark(
            profiler=profiler,
            prompts=("alpha", "beta"),
            profiler_config={"batch_size": 2, "capture_router_scores": False},
        )

    assert first.manifest.run_id == second.manifest.run_id
    assert first.manifest.prompt_set_hash == second.manifest.prompt_set_hash
    assert [layer.layer_index for layer in first.topology] == [0, 1]
    assert [(item.layer_index, item.expert_index) for item in first.activation_stats[:4]] == [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
    ]
    assert first.to_payload()["manifest"]["command"] == "bench"
    assert first.to_payload()["profiler_config"] == {
        "batch_size": 2,
        "capture_router_scores": False,
    }


def test_benchmark_payload_carries_input_and_profiler_hashes_and_writes_json(
    tmp_path: Path,
) -> None:
    layer = _layer(0)
    module = FakeRouterModule(name=layer.layer_name)
    backend = FakeBackend(router_states={0: _router_state(0)}, modules={0: module})

    with RouterActivationProfiler(backend=backend, bundle=_bundle(), topology=(layer,)) as profiler:
        module.run(
            {
                "top_k_indices": [[[0, 1]]],
                "top_k_weights": [[[0.55, 0.45]]],
            }
        )
        profiler.accumulate(attention_mask=[[1]])
        result = benchmark(
            profiler=profiler,
            input_payloads=({"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},),
            profiler_config={"include_router_scores": False, "batch_size": 1},
        )

    payload = result.to_payload()
    assert payload["input_payload_hash"] == result.input_payload_hash
    assert payload["profiler_config"] == {"batch_size": 1, "include_router_scores": False}
    assert result.manifest.input_checksums["input_payloads"] == result.input_payload_hash
    assert result.manifest.prompt_set_hash == result.input_payload_hash
    assert result.manifest.metadata["profiler_config_hash"] == result.manifest.input_checksums["profiler_config"]

    output_path = tmp_path / "bench.json"
    result.write_json(output_path)

    assert output_path.read_text(encoding="utf-8") == result.to_json()


def test_iter_prompt_batches_uses_attention_mask_for_active_token_counts() -> None:
    batches = list(
        iter_prompt_batches(
            tokenizer=FakeTokenizer(),
            prompts=("ab", "c", "def"),
            batch_size=2,
        )
    )

    assert [batch.prompt_indices for batch in batches] == [(0, 1), (2,)]
    assert [batch.active_token_count for batch in batches] == [3, 3]
    assert batches[0].prompt_payload() == {
        "prompts": ["ab", "c"],
        "prompt_indices": [0, 1],
        "active_token_count": 3,
    }


def test_router_activation_profiler_matches_live_gemma4_router_contract() -> None:
    _require_live_gemma4_runtime()

    torch = importlib.import_module("torch")
    backend = Gemma4Backend()
    signature = _live_gemma4_signature()
    bundle = backend.load(signature, dtype="float32", seed=0)
    tokenizer = bundle.tokenizer
    assert tokenizer is not None

    topology = backend.extract_topology(bundle)
    layer_indices = {layer.layer_index for layer in topology}
    assert layer_indices == {0, 1, 2, 3}

    encoded_inputs = tokenizer("router coverage", return_tensors="pt")
    attention_mask = encoded_inputs["attention_mask"]
    model = cast(object, bundle.model)
    getattr(model, "eval")()

    with torch.no_grad():
        with RouterActivationProfiler(
            backend=backend,
            bundle=bundle,
            topology=topology,
            include_router_scores=True,
        ) as profiler:
            getattr(model, "__call__")(
                input_ids=encoded_inputs["input_ids"],
                attention_mask=attention_mask,
                use_cache=False,
            )
            forward_records = tuple(profiler.records)
            assert {record.layer_index for record in forward_records} == layer_indices

            for record in forward_records:
                layer = next(item for item in topology if item.layer_index == record.layer_index)
                router_state = backend.extract_router_state(bundle, layer=layer)
                assert tuple(int(dim) for dim in cast(object, record.top_k_indices).shape)[-1] == router_state.top_k
                assert tuple(int(dim) for dim in cast(object, record.top_k_weights).shape)[-1] == router_state.top_k
                assert record.router_scores is not None
                assert tuple(int(dim) for dim in cast(object, record.router_scores).shape)[-1] == (
                    router_state.num_experts
                )

            forward_stats = profiler.accumulate(attention_mask=attention_mask)
            assert any(item.token_count > 0 for item in forward_stats)
            assert {
                item.layer_index for item in forward_stats if item.token_count > 0
            } == layer_indices

            profiler.reset_aggregation()
            profiler.clear_records()

            generated = getattr(model, "generate")(
                input_ids=encoded_inputs["input_ids"],
                attention_mask=attention_mask,
                max_new_tokens=1,
                do_sample=False,
            )
            generation_records = tuple(profiler.records)
            assert generated.shape[0] == encoded_inputs["input_ids"].shape[0]
            assert len(generation_records) >= len(topology)

            generation_stats = profiler.accumulate(attention_mask=attention_mask)
            assert any(item.token_count > 0 for item in generation_stats)
