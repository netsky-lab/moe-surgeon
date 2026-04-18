from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from safetensors.torch import save_file
import torch

from moe_surgeon.models.backend import LoadedBackendBundle
from moe_surgeon.models.checkpoints import open_local_safetensors_checkpoint
from moe_surgeon.models.errors import ShapeInvariantViolationError, TopologyMismatchError
from moe_surgeon.models.gemma4 import Gemma4Backend
from moe_surgeon.schemas import ModelHandle


def _gemma4_config() -> dict[str, object]:
    return {
        "_name_or_path": "google/gemma-4-27b",
        "_commit_hash": "rev-123",
        "architectures": ["Gemma4ForConditionalGeneration"],
        "model_type": "gemma4",
        "text_config": {
            "num_hidden_layers": 2,
            "hidden_size": 3,
            "enable_moe_block": True,
            "num_experts": 4,
            "top_k_experts": 2,
            "moe_intermediate_size": 2,
            "moe_layer_indices": [0],
        },
    }


def _write_config(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps(_gemma4_config()), encoding="utf-8")


def test_single_file_checkpoint_probe_is_sorted_and_backend_compatible(tmp_path: Path) -> None:
    _write_config(tmp_path)
    save_file(
        {
            "model.language_model.layers.0.router.scale": torch.tensor(1.0, dtype=torch.float32),
            "model.language_model.layers.0.router.per_expert_scale": torch.tensor(
                [0.2, 0.4, 0.1, 0.3],
                dtype=torch.float16,
            ),
            "model.language_model.layers.0.router.proj.weight": torch.arange(
                12, dtype=torch.float32
            ).reshape(4, 3),
            "model.language_model.layers.0.experts.gate_up_proj": torch.ones(
                (4, 4, 3), dtype=torch.float16
            ),
            "model.language_model.layers.0.experts.down_proj": torch.ones(
                (4, 3, 2), dtype=torch.float16
            ),
            "model.language_model.layers.0.mlp.down_proj.weight": torch.ones(
                (3, 6), dtype=torch.float16
            ),
            "model.language_model.layers.0.mlp.gate_proj.weight": torch.ones(
                (6, 3), dtype=torch.float16
            ),
            "model.language_model.layers.0.mlp.up_proj.weight": torch.ones((6, 3), dtype=torch.float16),
            "model.language_model.layers.0.pre_feedforward_layernorm.weight": torch.ones(
                (3,), dtype=torch.float16
            ),
            "model.language_model.layers.0.pre_feedforward_layernorm_2.weight": torch.ones(
                (3,), dtype=torch.float16
            ),
            "model.language_model.layers.0.post_feedforward_layernorm.weight": torch.ones(
                (3,), dtype=torch.float16
            ),
            "model.language_model.layers.0.post_feedforward_layernorm_1.weight": torch.ones(
                (3,), dtype=torch.float16
            ),
            "model.language_model.layers.0.post_feedforward_layernorm_2.weight": torch.ones(
                (3,), dtype=torch.float16
            ),
        },
        str(tmp_path / "model.safetensors"),
    )

    checkpoint = open_local_safetensors_checkpoint(tmp_path)

    assert checkpoint.model_id == "google/gemma-4-27b"
    assert checkpoint.revision == "rev-123"
    assert checkpoint.state_keys() == tuple(sorted(checkpoint.state_keys()))

    metadata = checkpoint.tensor_metadata(
        [
            "model.language_model.layers.0.router.proj.weight",
            "model.language_model.layers.0.router.per_expert_scale",
        ]
    )
    assert [item.tensor_key for item in metadata] == [
        "model.language_model.layers.0.router.per_expert_scale",
        "model.language_model.layers.0.router.proj.weight",
    ]
    assert metadata[0].shape == (4,)
    assert metadata[0].dtype == "F16"
    assert metadata[1].shape == (4, 3)
    assert metadata[1].shard_filename == "model.safetensors"

    backend = Gemma4Backend()
    signature = checkpoint.to_backend_signature()
    assert signature.source_path == str(tmp_path.resolve())
    assert backend.supports(signature)

    bundle = LoadedBackendBundle(
        backend_name="gemma4",
        model_handle=ModelHandle(
            model_id=checkpoint.model_id,
            revision=checkpoint.revision,
            backend_name="gemma4",
            source_path=str(tmp_path.resolve()),
        ),
        model=object(),
        config=checkpoint.config,
        metadata={"state_keys": checkpoint.state_keys()},
    )
    topology = backend.extract_topology(bundle)
    assert [layer.layer_index for layer in topology] == [0]


def test_sharded_checkpoint_loads_requested_router_tensors_only(tmp_path: Path) -> None:
    _write_config(tmp_path)
    first_shard = {
        "model.language_model.layers.0.router.scale": torch.tensor(1.5, dtype=torch.float32),
        "model.language_model.layers.0.router.proj.weight": torch.arange(
            12, dtype=torch.float32
        ).reshape(4, 3),
    }
    second_shard = {
        "model.language_model.layers.0.router.per_expert_scale": torch.tensor(
            [0.2, 0.4, 0.1, 0.3],
            dtype=torch.float16,
        ),
        "unused.tensor": torch.tensor([9.0], dtype=torch.float32),
    }
    save_file(first_shard, str(tmp_path / "model-00001-of-00002.safetensors"))
    save_file(second_shard, str(tmp_path / "model-00002-of-00002.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 0},
                "weight_map": {
                    "model.language_model.layers.0.router.scale": "model-00001-of-00002.safetensors",
                    "model.language_model.layers.0.router.proj.weight": "model-00001-of-00002.safetensors",
                    "model.language_model.layers.0.router.per_expert_scale": "model-00002-of-00002.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )

    checkpoint = open_local_safetensors_checkpoint(tmp_path)
    tensors = checkpoint.load_tensors(
        [
            "model.language_model.layers.0.router.per_expert_scale",
            "model.language_model.layers.0.router.proj.weight",
        ]
    )

    assert list(tensors) == [
        "model.language_model.layers.0.router.per_expert_scale",
        "model.language_model.layers.0.router.proj.weight",
    ]
    assert torch.equal(
        tensors["model.language_model.layers.0.router.proj.weight"],
        first_shard["model.language_model.layers.0.router.proj.weight"],
    )
    assert torch.equal(
        tensors["model.language_model.layers.0.router.per_expert_scale"],
        second_shard["model.language_model.layers.0.router.per_expert_scale"],
    )
    assert "unused.tensor" not in tensors


def test_checkpoint_probe_rejects_missing_shard_file(tmp_path: Path) -> None:
    _write_config(tmp_path)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 0},
                "weight_map": {
                    "model.language_model.layers.0.router.proj.weight": "model-00001-of-00002.safetensors"
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(TopologyMismatchError, match="checkpoint shard file is missing") as exc_info:
        open_local_safetensors_checkpoint(tmp_path)

    message = str(exc_info.value)
    assert "tensor_key=model.language_model.layers.0.router.proj.weight" in message
    assert f"checkpoint_path={tmp_path.resolve()}" in message
    assert "shard_filename=model-00001-of-00002.safetensors" in message


def test_checkpoint_probe_rejects_missing_requested_tensor_name(tmp_path: Path) -> None:
    _write_config(tmp_path)
    save_file({"known.tensor": torch.ones((1,), dtype=torch.float32)}, str(tmp_path / "model.safetensors"))
    checkpoint = open_local_safetensors_checkpoint(tmp_path)

    with pytest.raises(TopologyMismatchError, match="checkpoint tensor key is missing") as exc_info:
        checkpoint.load_tensors(["missing.tensor"])

    message = str(exc_info.value)
    assert "tensor_key=missing.tensor" in message
    assert f"checkpoint_path={tmp_path.resolve()}" in message


def test_checkpoint_load_rejects_missing_shard_file_with_tensor_context(tmp_path: Path) -> None:
    _write_config(tmp_path)
    shard_path = tmp_path / "model-00001-of-00001.safetensors"
    save_file({"known.tensor": torch.ones((1,), dtype=torch.float32)}, str(shard_path))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 0},
                "weight_map": {"known.tensor": "model-00001-of-00001.safetensors"},
            }
        ),
        encoding="utf-8",
    )
    checkpoint = open_local_safetensors_checkpoint(tmp_path)
    shard_path.unlink()

    with pytest.raises(TopologyMismatchError, match="checkpoint shard file is missing") as exc_info:
        checkpoint.load_tensors(["known.tensor"])

    message = str(exc_info.value)
    assert "tensor_key=known.tensor" in message
    assert "shard_filename=model-00001-of-00001.safetensors" in message


def test_checkpoint_probe_rejects_duplicate_tensor_mappings_in_index(tmp_path: Path) -> None:
    _write_config(tmp_path)
    save_file({"known.tensor": torch.ones((1,), dtype=torch.float32)}, str(tmp_path / "model-00001-of-00001.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        """{
  "metadata": {"total_size": 0},
  "weight_map": {
    "known.tensor": "model-00001-of-00001.safetensors",
    "known.tensor": "model-00001-of-00001.safetensors"
  }
}
""",
        encoding="utf-8",
    )

    with pytest.raises(ShapeInvariantViolationError, match="checkpoint index.json is malformed"):
        open_local_safetensors_checkpoint(tmp_path)


def test_checkpoint_probe_rejects_malformed_index_entry_type(tmp_path: Path) -> None:
    _write_config(tmp_path)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 0},
                "weight_map": {"known.tensor": 7},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ShapeInvariantViolationError,
        match="checkpoint index shard filename must be a non-empty string",
    ) as exc_info:
        open_local_safetensors_checkpoint(tmp_path)

    assert "tensor_key=known.tensor" in str(exc_info.value)


def test_checkpoint_probe_rejects_absolute_shard_paths(tmp_path: Path) -> None:
    _write_config(tmp_path)
    shard_path = tmp_path / "model-00001-of-00001.safetensors"
    save_file({"known.tensor": torch.ones((1,), dtype=torch.float32)}, str(shard_path))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 0},
                "weight_map": {"known.tensor": str(shard_path.resolve())},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ShapeInvariantViolationError,
        match="checkpoint index shard filename must be relative",
    ) as exc_info:
        open_local_safetensors_checkpoint(tmp_path)

    message = str(exc_info.value)
    assert "tensor_key=known.tensor" in message
    assert f"checkpoint_path={tmp_path.resolve()}" in message


def test_checkpoint_probe_rejects_escaping_shard_paths(tmp_path: Path) -> None:
    _write_config(tmp_path)
    outside_dir = tmp_path.parent / "outside"
    outside_dir.mkdir(exist_ok=True)
    outside_shard = outside_dir / "escaped.safetensors"
    save_file({"known.tensor": torch.ones((1,), dtype=torch.float32)}, str(outside_shard))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 0},
                "weight_map": {"known.tensor": "../outside/escaped.safetensors"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ShapeInvariantViolationError,
        match="checkpoint index shard path escapes checkpoint directory",
    ) as exc_info:
        open_local_safetensors_checkpoint(tmp_path)

    message = str(exc_info.value)
    assert "tensor_key=known.tensor" in message
    assert "shard_filename=../outside/escaped.safetensors" in message


def test_checkpoint_probe_rejects_indexed_keys_missing_from_existing_shard(tmp_path: Path) -> None:
    _write_config(tmp_path)
    save_file({"other.tensor": torch.ones((1,), dtype=torch.float32)}, str(tmp_path / "model-00001-of-00001.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 0},
                "weight_map": {"known.tensor": "model-00001-of-00001.safetensors"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(TopologyMismatchError, match="checkpoint shard is missing indexed tensor key") as exc_info:
        open_local_safetensors_checkpoint(tmp_path)

    message = str(exc_info.value)
    assert "tensor_key=known.tensor" in message
    assert f"checkpoint_path={tmp_path.resolve()}" in message
    assert "shard_filename=model-00001-of-00001.safetensors" in message


def test_checkpoint_probe_rejects_pickle_only_checkpoint_layout(tmp_path: Path) -> None:
    _write_config(tmp_path)
    (tmp_path / "pytorch_model.bin").write_bytes(b"pickle")

    with pytest.raises(
        TopologyMismatchError,
        match="pickle-only checkpoints are unsupported; expected safetensors weights",
    ) as exc_info:
        open_local_safetensors_checkpoint(tmp_path)

    assert "unsupported_weight_file=pytorch_model.bin" in str(exc_info.value)


def test_checkpoint_probe_import_is_transformers_free_in_fresh_process() -> None:
    probe = """
import sys
import moe_surgeon.models.checkpoints
assert 'transformers' not in sys.modules
print('ok')
"""

    result = subprocess.run(
        [sys.executable, "-c", probe],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert result.stdout.strip() == "ok"
