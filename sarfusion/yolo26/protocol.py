"""Frozen construction and audit helpers for YOLO26 Stage A."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
import ultralytics
from ultralytics.nn.tasks import load_checkpoint

from .model import YOLO26FusionDetectionModel


ULTRALYTICS_VERSION = "8.4.138"
TORCH_VERSION = "2.4.0"
TORCHVISION_VERSION = "0.19.0"
YOLO26S_SHA256 = "646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def assert_environment() -> dict[str, str]:
    versions = {
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__.split("+")[0],
        "torchvision": torchvision.__version__.split("+")[0],
    }
    expected = {
        "ultralytics": ULTRALYTICS_VERSION,
        "torch": TORCH_VERSION,
        "torchvision": TORCHVISION_VERSION,
    }
    if versions != expected:
        raise RuntimeError(f"YOLO26 environment mismatch: {versions} != {expected}")
    return versions


def set_construction_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(bool(deterministic), warn_only=False)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = False


def load_pretrained_model(path: str | Path):
    path = Path(path).resolve()
    actual_hash = sha256_file(path)
    if actual_hash != YOLO26S_SHA256:
        raise RuntimeError(
            f"yolo26s.pt SHA256 mismatch: expected {YOLO26S_SHA256}, got {actual_hash}."
        )
    model, checkpoint = load_checkpoint(str(path))
    if model.yaml.get("scale") != "s" or not bool(model.end2end):
        raise RuntimeError("Checkpoint is not the expected end-to-end YOLO26s model.")
    return model, checkpoint


def build_fusion_model(
    pretrained_model,
    *,
    seed: int,
    use_fam: bool,
    deterministic: bool,
    verbose: bool = False,
) -> YOLO26FusionDetectionModel:
    set_construction_seed(seed, deterministic=deterministic)
    model = YOLO26FusionDetectionModel(
        cfg=pretrained_model.yaml,
        nc=1,
        use_fam=use_fam,
        freeze_fam=False,
        spatial_jitter_std=0.0,
        verbose=verbose,
    )
    model.names = {0: "person"}
    model.load_official_pretrained(pretrained_model, verbose=verbose)
    return model


def verify_source_manifest(repository: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    for item in manifest["files"]:
        path = repository / item["path"]
        actual = sha256_file(path) if path.is_file() else None
        if actual != item["sha256"]:
            failures.append(
                {"path": item["path"], "expected": item["sha256"], "actual": actual}
            )
    if failures:
        raise RuntimeError(f"YOLO26 frozen source manifest mismatch: {failures}")
    return manifest
