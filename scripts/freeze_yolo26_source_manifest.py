#!/usr/bin/env python3
"""Regenerate the non-self-referential YOLO26 Stage A source manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


FILES = (
    "requirements-yolo26.txt",
    "sarfusion/yolo26/__init__.py",
    "sarfusion/yolo26/fam.py",
    "sarfusion/yolo26/model.py",
    "sarfusion/yolo26/data.py",
    "sarfusion/yolo26/trainer.py",
    "sarfusion/yolo26/protocol.py",
    "scripts/run_yolo26_stage_a.py",
    "scripts/audit_yolo26_integration.py",
    "scripts/audit_yolo26_stage_a_control.py",
    "scripts/freeze_yolo26_source_manifest.py",
    "tests/test_yolo26.py",
    "parameters/YOLO26/stage_a_split.yaml",
    "parameters/YOLO26/yolo26s_additive_seed40_stage_a.yaml",
    "parameters/YOLO26/yolo26s_fam_seed40_stage_a.yaml",
    "notes/yolo26_fam_stage_a.md",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    items = []
    for relative in FILES:
        path = repository / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        items.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    manifest = {
        "schema": "sarfusion.yolo26.stage_a.source.v1",
        "ultralytics": "8.4.138",
        "ultralytics_tag_commit": "dad7bb4534c95021bc14969ab25d77b77c4efdc3",
        "files": items,
    }
    output = repository / "parameters/YOLO26/stage_a_source_manifest.json"
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)
    print(sha256(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
