"""Reproducibility controls and lightweight training trajectory traces."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


BOX_GUIDED_TRAINING_SOURCE_MANIFEST_ID = (
    "rtdetr_box_guided_training_source_v1"
)
BOX_GUIDED_TRAINING_SOURCE_FILES = (
    "main.py",
    "sarfusion/data/__init__.py",
    "sarfusion/data/utils.py",
    "sarfusion/data/wisard.py",
    "sarfusion/experiment/box_guided_alignment.py",
    "sarfusion/experiment/experiment.py",
    "sarfusion/experiment/modality_consistency.py",
    "sarfusion/experiment/run.py",
    "sarfusion/experiment/utils.py",
    "sarfusion/models/__init__.py",
    "sarfusion/models/checkpoints.py",
    "sarfusion/models/detr.py",
    "sarfusion/models/loss.py",
    "sarfusion/models/rtdetr_fusion.py",
    "sarfusion/tracker/abstract_tracker.py",
    "sarfusion/tracker/wandb_tracker.py",
    "sarfusion/utils/general.py",
    "sarfusion/utils/grid.py",
    "sarfusion/utils/metrics.py",
    "sarfusion/utils/reproducibility.py",
    "sarfusion/utils/structures.py",
    "sarfusion/utils/utils.py",
)
_SHA256_HEX_LENGTH = 64


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for block in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_training_source_manifest(
    manifest_id: str = BOX_GUIDED_TRAINING_SOURCE_MANIFEST_ID,
    *,
    repo_root: str | os.PathLike | None = None,
) -> dict:
    """Hash the frozen local source surface for box-guided RT-DETR training.

    The manifest is deliberately an explicit, versioned list rather than a
    recursive source-tree hash.  This keeps generated files and unrelated
    experiments out of the identity while making every file that controls the
    model, data stream, auxiliary loss, validation metric, and checkpoint
    selection part of the scientific contract.
    """
    if manifest_id != BOX_GUIDED_TRAINING_SOURCE_MANIFEST_ID:
        raise ValueError(f"Unknown training-source manifest id: {manifest_id!r}")

    root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[2]
    )
    file_hashes = {}
    for relative_path in BOX_GUIDED_TRAINING_SOURCE_FILES:
        source_path = (root / relative_path).resolve()
        try:
            source_path.relative_to(root)
        except ValueError as error:
            raise RuntimeError(
                f"Training-source path escapes repository root: {relative_path}"
            ) from error
        if not source_path.is_file():
            raise FileNotFoundError(
                f"Training-source manifest file is missing: {source_path}"
            )
        file_hashes[relative_path] = _file_sha256(source_path)

    payload = {
        "schema_version": 1,
        "manifest_id": manifest_id,
        "files": file_hashes,
    }
    return {**payload, "sha256": _canonical_json_sha256(payload)}


def verify_training_source_manifest(
    manifest_id: str | None,
    expected_sha256: str | None,
    *,
    repo_root: str | os.PathLike | None = None,
    required: bool = False,
) -> dict | None:
    """Optionally validate a declared source manifest, failing on half-configs."""
    if manifest_id is None and expected_sha256 is None:
        if required:
            raise RuntimeError("Training-source provenance declaration is required")
        return None
    if manifest_id is None or expected_sha256 is None:
        raise ValueError(
            "training_source_manifest_id and "
            "training_source_manifest_sha256 must be declared together"
        )
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != _SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise ValueError(
            "training_source_manifest_sha256 must be a lowercase SHA-256 digest"
        )

    current = build_training_source_manifest(manifest_id, repo_root=repo_root)
    if current["sha256"] != expected_sha256:
        raise RuntimeError(
            "Training-source manifest changed: expected "
            f"{expected_sha256}, got {current['sha256']}"
        )
    return current


def training_source_runtime_fields(manifest: dict | None) -> dict:
    """Return the exact provenance fields embedded in the first trace event."""
    if manifest is None:
        return {}
    return {
        "training_source_manifest_id": manifest["manifest_id"],
        "training_source_manifest_sha256": manifest["sha256"],
        "training_source_files_sha256": dict(manifest["files"]),
    }


def verify_training_source_runtime_trace(
    trace_path: str | os.PathLike,
    manifest: dict,
) -> dict:
    """Bind a completed run's first runtime event to the current source bytes."""
    trace_path = Path(trace_path)
    if not trace_path.is_file():
        raise RuntimeError("Completed run has no reproducibility_trace.jsonl")
    trace_bytes = trace_path.read_bytes()
    if not trace_bytes:
        raise RuntimeError("Reproducibility trace is empty")
    first_line = trace_bytes.splitlines()[0]
    try:
        first_event = json.loads(first_line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("First reproducibility-trace event is invalid JSON") from error
    if not isinstance(first_event, dict) or first_event.get("event") != "runtime":
        raise RuntimeError("First reproducibility-trace event must be runtime")

    expected_fields = training_source_runtime_fields(manifest)
    observed_fields = {
        key: first_event.get(key) for key in expected_fields
    }
    if observed_fields != expected_fields:
        raise RuntimeError(
            "Runtime trace training-source provenance differs from the "
            "declared/current manifest"
        )
    return {
        **expected_fields,
        "runtime_event_sha256": _canonical_json_sha256(first_event),
        "reproducibility_trace_sha256": hashlib.sha256(trace_bytes).hexdigest(),
    }


def configure_reproducibility(seed: int, deterministic: bool = False, warn_only: bool = False):
    """Configure all RNGs before datasets, models, or CUDA state are created."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        # Required by deterministic CUDA matrix multiplications. It must be set
        # before the first cuBLAS operation in the process.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = bool(deterministic)
    if deterministic and hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if deterministic and hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(bool(deterministic), warn_only=bool(warn_only))

    if deterministic:
        install_deterministic_rtdetr_attention()


def deterministic_grid_sample_bilinear(input_tensor: torch.Tensor, grid: torch.Tensor):
    """Deterministic equivalent of 2-D ``grid_sample`` for RT-DETR.

    This implements the exact RT-DETR combination: bilinear interpolation,
    zero padding and ``align_corners=False``.  Its backward uses ``gather``;
    PyTorch provides a deterministic CUDA implementation for gather backward
    when deterministic algorithms are enabled.  Native CUDA ``grid_sample``
    backward uses atomic additions and is not deterministic.
    """
    if input_tensor.ndim != 4 or grid.ndim != 4 or grid.shape[-1] != 2:
        raise ValueError(
            "Expected input [N,C,H,W] and grid [N,out_h,out_w,2], got "
            f"{tuple(input_tensor.shape)} and {tuple(grid.shape)}"
        )
    if input_tensor.shape[0] != grid.shape[0]:
        raise ValueError("Input and grid batch dimensions must match")

    batch, channels, height, width = input_tensor.shape
    out_height, out_width = grid.shape[1:3]
    x = ((grid[..., 0] + 1) * width - 1) / 2
    y = ((grid[..., 1] + 1) * height - 1) / 2

    x0 = torch.floor(x)
    y0 = torch.floor(y)
    x1 = x0 + 1
    y1 = y0 + 1

    def sample(x_index, y_index):
        valid = (
            (x_index >= 0)
            & (x_index < width)
            & (y_index >= 0)
            & (y_index < height)
        )
        flat_index = (
            y_index.clamp(0, height - 1) * width
            + x_index.clamp(0, width - 1)
        ).to(torch.long)
        flat_index = flat_index.reshape(batch, 1, -1).expand(-1, channels, -1)
        values = input_tensor.flatten(2).gather(2, flat_index)
        values = values.reshape(batch, channels, out_height, out_width)
        return values * valid.unsqueeze(1).to(values.dtype)

    top_left = sample(x0, y0)
    top_right = sample(x1, y0)
    bottom_left = sample(x0, y1)
    bottom_right = sample(x1, y1)

    wx = x - x0
    wy = y - y0
    return (
        top_left * ((1 - wx) * (1 - wy)).unsqueeze(1)
        + top_right * (wx * (1 - wy)).unsqueeze(1)
        + bottom_left * ((1 - wx) * wy).unsqueeze(1)
        + bottom_right * (wx * wy).unsqueeze(1)
    )


def deterministic_multi_scale_deformable_attention(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
):
    """RT-DETR deformable attention without native CUDA ``grid_sample``."""
    batch_size, _, num_heads, hidden_dim = value.shape
    _, num_queries, _, num_levels, num_points, _ = sampling_locations.shape
    split_sizes = [
        int(height.item() * width.item())
        for height, width in value_spatial_shapes
    ]
    value_list = value.split(split_sizes, dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampled_levels = []

    for level_id, (height, width) in enumerate(value_spatial_shapes):
        height, width = int(height.item()), int(width.item())
        value_level = (
            value_list[level_id]
            .flatten(2)
            .transpose(1, 2)
            .reshape(batch_size * num_heads, hidden_dim, height, width)
        )
        sampling_grid = (
            sampling_grids[:, :, :, level_id]
            .transpose(1, 2)
            .flatten(0, 1)
        )
        sampled_levels.append(
            deterministic_grid_sample_bilinear(value_level, sampling_grid)
        )

    attention_weights = attention_weights.transpose(1, 2).reshape(
        batch_size * num_heads,
        1,
        num_queries,
        num_levels * num_points,
    )
    output = (
        (torch.stack(sampled_levels, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(batch_size, num_heads * hidden_dim, num_queries)
    )
    return output.transpose(1, 2).contiguous()


def install_deterministic_rtdetr_attention():
    """Patch the function resolved by Hugging Face RT-DETR forward calls."""
    from transformers.models.rt_detr import modeling_rt_detr

    modeling_rt_detr.multi_scale_deformable_attention = (
        deterministic_multi_scale_deformable_attention
    )


def prepare_rtdetr_model_for_determinism(model: torch.nn.Module):
    """Ensure RT-DETR cannot select its non-deterministic custom CUDA kernel."""
    changed = 0
    for module in model.modules():
        if module.__class__.__name__ == "RTDetrMultiscaleDeformableAttention":
            module.disable_custom_kernels = True
            changed += 1
    return changed


def tensor_digest(tensor: torch.Tensor) -> str:
    """SHA-256 including tensor metadata and every tensor value."""
    tensor = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    # Flatten first: PyTorch cannot reinterpret a scalar tensor to an element
    # type with a different byte width.
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def state_dict_digest(
    state_dict: dict[str, torch.Tensor],
    include_names: Iterable[str] | None = None,
) -> str:
    """Stable SHA-256 for a complete state dict or a selected name subset."""
    selected = set(include_names) if include_names is not None else None
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        if selected is not None and name not in selected:
            continue
        value = state_dict[name]
        if not isinstance(value, torch.Tensor):
            continue
        digest.update(name.encode())
        digest.update(tensor_digest(value).encode())
    return digest.hexdigest()


def model_digests(model: torch.nn.Module) -> dict[str, str]:
    """Hashes that distinguish pretrained backbone and random detection head."""
    state_dict = model.state_dict()
    head_names = [
        name
        for name in state_dict
        if "class_embed" in name or "bbox_embed" in name
    ]
    return {
        "model_sha256": state_dict_digest(state_dict),
        "detection_head_sha256": state_dict_digest(state_dict, head_names),
    }


class ReproducibilityTrace:
    """Append-only JSONL trace stored alongside the run's W&B files."""

    def __init__(self, path: str | os.PathLike | None):
        self.path = Path(path) if path else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self):
        return self.path is not None

    def write(self, event: str, **values):
        if not self.enabled:
            return
        record = {"event": event, **values}
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, sort_keys=True) + "\n")


def runtime_fingerprint() -> dict:
    result = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cudnn": torch.backends.cudnn.version(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }
    if torch.cuda.is_available():
        result["gpu"] = torch.cuda.get_device_name(torch.cuda.current_device())
    return result
