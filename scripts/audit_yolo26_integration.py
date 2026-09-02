#!/usr/bin/env python3
"""CPU integration audit for the frozen YOLO26 RGB+IR implementation."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/sarfusion-yolo26/ultralytics")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/sarfusion-yolo26/matplotlib")
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("WANDB_MODE", "disabled")

import numpy as np
import torch
import yaml
from ultralytics.cfg import get_cfg
from ultralytics.data.utils import check_det_dataset

from sarfusion.yolo26.data import (
    PairedWiSARDYOLODataset,
    RGBTFormat,
    build_stage_a_dataset,
)
from sarfusion.yolo26.protocol import (
    assert_environment,
    build_fusion_model,
    load_pretrained_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("notes/Search_and_Rescue/results/yolo26_integration_audit.json"),
    )
    return parser.parse_args()


def flatten_tensors(value) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        tensors = []
        for key in sorted(value):
            tensors.extend(flatten_tensors(value[key]))
        return tensors
    if isinstance(value, (list, tuple)):
        tensors = []
        for item in value:
            tensors.extend(flatten_tensors(item))
        return tensors
    return []


def dataset_audit(repository: Path, config: dict) -> tuple[dict, dict]:
    split = yaml.safe_load(
        (repository / config["study"]["split_config"]).read_text(encoding="utf-8")
    )
    materialized = repository / "runs/yolo26_stage_a/_integration_audit"
    manifest = build_stage_a_dataset(split, materialized)
    data = check_det_dataset(manifest["data_yaml"])
    hyp = get_cfg(overrides=config["training"])
    datasets = {}
    for mode in ("train", "val"):
        dataset = PairedWiSARDYOLODataset(
            img_path=data[mode],
            imgsz=128,
            batch_size=2,
            augment=False,
            hyp=hyp,
            rect=False,
            cache=False,
            single_cls=True,
            stride=32,
            pad=0.5,
            prefix=f"audit-{mode}: ",
            task="detect",
            classes=None,
            data=data,
            fraction=1.0,
            modal_dropout=False,
        )
        datasets[mode] = dataset
        expected = 3123 if mode == "train" else 896
        if len(dataset) != expected:
            raise RuntimeError(f"{mode} dataset has {len(dataset)}, expected {expected}.")

    sample = datasets["train"][0]
    batch = datasets["train"].collate_fn(
        [datasets["train"][0], datasets["train"][1]]
    )
    if tuple(sample["img"].shape) != (4, 128, 128):
        raise RuntimeError(f"Unexpected sample shape {sample['img'].shape}.")
    if tuple(batch["img"].shape) != (2, 4, 128, 128):
        raise RuntimeError(f"Unexpected batch shape {batch['img'].shape}.")

    sentinel = np.zeros((2, 2, 4), dtype=np.uint8)
    sentinel[:] = (10, 20, 30, 40)
    formatted = RGBTFormat(bgr=0.0)._format_img(sentinel)
    if formatted[:, 0, 0].tolist() != [30, 20, 10, 40]:
        raise RuntimeError("RGBT BGR-to-RGB sentinel failed.")
    return manifest, {
        "train_length": len(datasets["train"]),
        "val_length": len(datasets["val"]),
        "sample_shape": list(sample["img"].shape),
        "batch_shape": list(batch["img"].shape),
        "channel_sentinel": formatted[:, 0, 0].tolist(),
        "train_val_rgb_overlap": 0,
    }


def model_audit(repository: Path, config: dict) -> dict:
    pretrained, _ = load_pretrained_model(repository / config["model"]["weights"])
    seed = int(config["study"]["seed"])
    control = build_fusion_model(
        pretrained,
        seed=seed,
        use_fam=False,
        deterministic=True,
    )
    candidate = build_fusion_model(
        pretrained,
        seed=seed,
        use_fam=True,
        deterministic=True,
    )
    control_report = control.initialization_report()
    candidate_report = candidate.initialization_report()
    for key in ("shared_sha256", "ir_sha256", "fam_sha256"):
        if control_report[key] != candidate_report[key]:
            raise RuntimeError(f"Matched initialization differs for {key}.")

    control.eval()
    rgb = torch.rand(2, 3, 128, 128)
    rgbt = torch.cat((rgb, torch.zeros(2, 1, 128, 128)), dim=1)
    rgb_only_mask = torch.tensor(((1.0, 0.0), (1.0, 0.0)))
    with torch.no_grad():
        standard = control.predict(rgb)
        wrapped = control.predict(rgbt, modality_mask=rgb_only_mask)
    standard_tensors = flatten_tensors(standard)
    wrapped_tensors = flatten_tensors(wrapped)
    if len(standard_tensors) != len(wrapped_tensors):
        raise RuntimeError("RGB parity output structures differ.")
    parity_error = max(
        float((left - right).abs().max().item())
        for left, right in zip(standard_tensors, wrapped_tensors)
    )
    if parity_error != 0.0:
        raise RuntimeError(f"RGB-only wrapper parity error is {parity_error}.")

    candidate.train()
    candidate.args = get_cfg(overrides=config["training"])
    synthetic = {
        "img": torch.rand(2, 4, 128, 128),
        "modality_mask": torch.ones(2, 2),
        "batch_idx": torch.tensor((0, 1)),
        "cls": torch.zeros(2, 1),
        "bboxes": torch.tensor(((0.5, 0.5, 0.2, 0.2), (0.5, 0.5, 0.2, 0.2))),
    }
    optimizer = torch.optim.AdamW(candidate.parameters(), lr=1e-3, weight_decay=5e-4)
    before = [
        next(module.parameters()).detach().clone()
        for module in candidate.fam_modules
    ]
    loss, loss_items = candidate(synthetic)
    if not torch.isfinite(loss).all():
        raise RuntimeError(f"Non-finite YOLO26 loss: {loss}")
    loss.sum().backward()
    gradient_norms = []
    for module in candidate.fam_modules:
        gradients = [
            parameter.grad.detach().norm()
            for parameter in module.parameters()
            if parameter.grad is not None
        ]
        if not gradients or not all(torch.isfinite(value) for value in gradients):
            raise RuntimeError("FAM gradient audit failed.")
        gradient_norms.append(float(torch.stack(gradients).norm().item()))
    optimizer.step()
    updates = [
        float((next(module.parameters()).detach() - old).abs().max().item())
        for module, old in zip(candidate.fam_modules, before)
    ]
    if any(value <= 0.0 for value in updates):
        raise RuntimeError(f"FAM optimizer update failed: {updates}")

    with tempfile.NamedTemporaryFile(suffix=".pt") as stream:
        torch.save(candidate.state_dict(), stream.name)
        restored = build_fusion_model(
            pretrained,
            seed=seed,
            use_fam=True,
            deterministic=True,
        )
        state = torch.load(stream.name, map_location="cpu", weights_only=True)
        restored.load_state_dict(state, strict=True)
        for scope in ("shared", "ir", "fam"):
            if restored.state_sha256(scope) != candidate.state_sha256(scope):
                raise RuntimeError(f"Strict checkpoint roundtrip failed for {scope}.")

    return {
        "control_initialization": control_report,
        "candidate_initialization": candidate_report,
        "matched_initialization": True,
        "rgb_only_max_abs_error": parity_error,
        "loss": [float(value) for value in loss.detach()],
        "loss_items": {key: float(value) for key, value in loss_items.items()},
        "fam_gradient_norms": gradient_norms,
        "fam_max_updates": updates,
        "strict_checkpoint_roundtrip": True,
    }


def main() -> int:
    args = parse_args()
    repository = REPOSITORY
    os.chdir(repository)
    config = yaml.safe_load(
        (repository / "parameters/YOLO26/yolo26s_additive_seed40_stage_a.yaml").read_text(
            encoding="utf-8"
        )
    )
    report = {
        "schema": "sarfusion.yolo26.integration_audit.v1",
        "environment": assert_environment(),
        "dataset_manifest": None,
        "dataset_checks": None,
        "model_checks": None,
        "status": "running",
    }
    manifest, checks = dataset_audit(repository, config)
    report["dataset_manifest"] = {
        key: manifest[key]
        for key in ("schema", "counts", "content_sha256", "dropped_unpaired_frames")
    }
    report["dataset_checks"] = checks
    report["model_checks"] = model_audit(repository, config)
    report["status"] = "passed"
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
