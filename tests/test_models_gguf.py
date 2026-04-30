from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from click.testing import CliRunner
from gguf import GGUFWriter

from moe_surgeon.analysis.scan import load_local_scan_bundle, scan_model
from moe_surgeon.cli.main import cli
from moe_surgeon.models.backend import BackendSignature, resolve_backend
from moe_surgeon.models.gguf import Gemma4GgufBackend, open_local_gguf_checkpoint
from moe_surgeon.prune.gguf import inspect_gguf, prune_gguf_static


def _write_tiny_gemma4_gguf(path: Path) -> Path:
    writer = GGUFWriter(path, arch="gemma4")
    writer.add_context_length(16)
    writer.add_block_count(1)
    writer.add_embedding_length(4)
    writer.add_expert_count(3)
    writer.add_expert_used_count(2)
    writer.add_expert_feed_forward_length(2)
    writer.add_feed_forward_length(5)
    writer.add_tensor("blk.0.ffn_gate_inp.weight", np.arange(12, dtype=np.float32).reshape(3, 4))
    writer.add_tensor("blk.0.ffn_gate_inp.scale", np.ones((4,), dtype=np.float32))
    writer.add_tensor("blk.0.ffn_down_exps.scale", np.ones((3,), dtype=np.float32))
    writer.add_tensor("blk.0.ffn_gate_up_exps.weight", np.ones((3, 4, 4), dtype=np.float32))
    writer.add_tensor("blk.0.ffn_down_exps.weight", np.ones((3, 2, 4), dtype=np.float32))
    writer.add_tensor("blk.0.ffn_down.weight", np.ones((5, 4), dtype=np.float32))
    writer.add_tensor("blk.0.ffn_gate.weight", np.ones((4, 5), dtype=np.float32))
    writer.add_tensor("blk.0.ffn_up.weight", np.ones((4, 5), dtype=np.float32))
    writer.add_tensor("blk.0.pre_ffw_norm_2.weight", np.ones((4,), dtype=np.float32))
    writer.add_tensor("blk.0.post_ffw_norm.weight", np.ones((4,), dtype=np.float32))
    writer.add_tensor("blk.0.post_ffw_norm_1.weight", np.ones((4,), dtype=np.float32))
    writer.add_tensor("blk.0.post_ffw_norm_2.weight", np.ones((4,), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


def test_open_local_gguf_checkpoint_reads_gemma4_metadata(tmp_path: Path) -> None:
    checkpoint_path = _write_tiny_gemma4_gguf(tmp_path / "tiny.gguf")

    checkpoint = open_local_gguf_checkpoint(checkpoint_path)

    assert checkpoint.architecture == "gemma4"
    assert checkpoint.fields["gemma4.block_count"] == 1
    assert checkpoint.fields["gemma4.expert_count"] == 3
    assert "blk.0.ffn_gate_inp.weight" in checkpoint.state_keys()


def test_default_registry_resolves_gemma4_gguf_before_hf_backend(tmp_path: Path) -> None:
    checkpoint = open_local_gguf_checkpoint(_write_tiny_gemma4_gguf(tmp_path / "tiny.gguf"))

    backend = resolve_backend(checkpoint.to_backend_signature())

    assert isinstance(backend, Gemma4GgufBackend)


def test_gemma4_gguf_backend_extracts_topology_and_router_state(tmp_path: Path) -> None:
    checkpoint = open_local_gguf_checkpoint(_write_tiny_gemma4_gguf(tmp_path / "tiny.gguf"))
    backend = Gemma4GgufBackend()
    bundle = backend.load(checkpoint.to_backend_signature())

    topology = backend.extract_topology(bundle)
    router_state = backend.extract_router_state(bundle, layer=topology[0])

    assert len(topology) == 1
    assert topology[0].expert_count == 3
    assert topology[0].top_k == 2
    assert topology[0].hidden_size == 4
    assert topology[0].module_paths["router_proj"] == "blk.0.ffn_gate_inp.weight"
    assert router_state.projection_shape == (3, 4)
    assert router_state.per_expert_scale_shape == (3,)


def test_gguf_scan_uses_f32_router_tensors(tmp_path: Path) -> None:
    checkpoint_path = _write_tiny_gemma4_gguf(tmp_path / "tiny.gguf")
    bundle, backend = load_local_scan_bundle(checkpoint_path)

    result = scan_model(bundle, backend=backend)

    assert result.manifest.command == "scan"
    assert result.manifest.model_handle is not None
    assert result.manifest.model_handle.backend_name == "gemma4-gguf"
    assert len(result.layers) == 1
    assert len(result.router_states) == 1
    assert len(result.expert_stats) == 3


def test_gemma4_gguf_backend_requires_gguf_signature_marker() -> None:
    backend = Gemma4GgufBackend()

    assert not backend.supports(BackendSignature(model_id="m", model_type="gemma4"))
    assert backend.supports(
        BackendSignature(
            model_id="m",
            architecture="gemma4",
            model_type="gemma4",
            metadata={"format": "gguf"},
        )
    )


def test_prune_gguf_static_writes_pruned_expert_axis_tensors(tmp_path: Path) -> None:
    checkpoint_path = _write_tiny_gemma4_gguf(tmp_path / "tiny.gguf")
    bundle, backend = load_local_scan_bundle(checkpoint_path)
    scan_result = scan_model(bundle, backend=backend)

    output_path = tmp_path / "tiny-pruned.gguf"
    result = prune_gguf_static(
        checkpoint_path,
        scan_result=scan_result,
        target_experts=2,
        output_path=output_path,
    )
    pruned = open_local_gguf_checkpoint(output_path)

    assert result.target_experts == 2
    assert result.dry_run is False
    assert result.rewritten_tensor_count == 4
    assert result.copied_tensor_count == len(open_local_gguf_checkpoint(checkpoint_path).tensors) - 4
    assert result.source_sha256
    assert result.output_sha256
    assert (tmp_path / "tiny-pruned.gguf.manifest.json").is_file()
    manifest = json.loads((tmp_path / "tiny-pruned.gguf.manifest.json").read_text(encoding="utf-8"))
    assert manifest["output_sha256"] == result.output_sha256
    assert manifest["rewritten_tensor_keys"] == list(result.rewritten_tensor_keys)
    assert pruned.fields["gemma4.expert_count"] == 2
    assert pruned.tensors["blk.0.ffn_gate_inp.weight"].data_shape == (2, 4)
    assert pruned.tensors["blk.0.ffn_down_exps.scale"].data_shape == (2,)
    assert pruned.tensors["blk.0.ffn_gate_up_exps.weight"].shape[-1] == 2
    assert pruned.tensors["blk.0.ffn_down_exps.weight"].shape[-1] == 2


def test_prune_gguf_static_dry_run_reports_without_writing(tmp_path: Path) -> None:
    checkpoint_path = _write_tiny_gemma4_gguf(tmp_path / "tiny.gguf")
    bundle, backend = load_local_scan_bundle(checkpoint_path)
    scan_result = scan_model(bundle, backend=backend)

    result = prune_gguf_static(
        checkpoint_path,
        scan_result=scan_result,
        target_experts=2,
        output_path=None,
        dry_run=True,
    )

    assert result.dry_run is True
    assert result.output_path is None
    assert result.output_sha256 is None
    assert result.rewritten_tensor_count == 4
    assert not (tmp_path / "tiny-pruned.gguf").exists()


def test_inspect_gguf_returns_canonical_inventory(tmp_path: Path) -> None:
    checkpoint_path = _write_tiny_gemma4_gguf(tmp_path / "tiny.gguf")

    result = inspect_gguf(checkpoint_path)

    assert result.architecture == "gemma4"
    assert result.expert_count == 3
    assert result.top_k == 2
    assert result.block_count == 1
    assert result.hidden_size == 4
    assert result.tensor_count == len(open_local_gguf_checkpoint(checkpoint_path).tensors)


def test_cli_gguf_inspect_writes_json(tmp_path: Path) -> None:
    checkpoint_path = _write_tiny_gemma4_gguf(tmp_path / "tiny.gguf")
    output_path = tmp_path / "inspect.json"

    result = CliRunner().invoke(
        cli,
        [
            "--source-path",
            str(checkpoint_path),
            "gguf-inspect",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["architecture"] == "gemma4"
    assert payload["expert_count"] == 3
