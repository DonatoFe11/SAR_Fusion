"""Frozen frame-level temporal splits for paired WiSARD sequences."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def stable_json_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def resolve_manifest_path(path):
    path = Path(path)
    if path.is_absolute():
        return path.resolve()
    if path.is_file():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def _sorted_files(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    return sorted(path for path in directory.iterdir() if path.is_file())


def _row(dataset_root, sequence_id, local_index, paths):
    vis_image, vis_label, ir_image, ir_label = paths
    vis_label_bytes = vis_label.read_bytes()
    ir_label_bytes = ir_label.read_bytes()
    lines = [line for line in vis_label_bytes.decode().splitlines() if line.strip()]
    areas = []
    for line in lines:
        fields = line.split()
        if len(fields) != 5:
            raise RuntimeError(f"Invalid YOLO annotation in {vis_label}: {line!r}")
        areas.append(float(fields[3]) * float(fields[4]))
    return {
        "sequence_id": sequence_id,
        "local_index": local_index,
        "vis_image": str(vis_image.relative_to(dataset_root)),
        "vis_image_size": vis_image.stat().st_size,
        "vis_label": str(vis_label.relative_to(dataset_root)),
        "vis_label_sha256": hashlib.sha256(vis_label_bytes).hexdigest(),
        "ir_image": str(ir_image.relative_to(dataset_root)),
        "ir_image_size": ir_image.stat().st_size,
        "ir_label": str(ir_label.relative_to(dataset_root)),
        "ir_label_sha256": hashlib.sha256(ir_label_bytes).hexdigest(),
        "vis_box_count": len(lines),
        "vis_box_areas": areas,
    }


def _phase_for_index(sequence, index):
    for phase in ("train", "embargo", "val"):
        start, stop = sequence["ranges"][phase]
        if int(start) <= index < int(stop):
            return phase
    raise RuntimeError(
        f"Index {index} is not covered by train/embargo/val ranges for "
        f"{sequence['id']}"
    )


def build_temporal_split_inventory(manifest, dataset_root):
    dataset_root = Path(dataset_root).resolve()
    rows = []
    phase_rows = {"train": [], "embargo": [], "val": []}
    sequence_summaries = []
    for sequence in manifest["sequences"]:
        vis_folder = sequence["vis_folder"]
        ir_folder = sequence["ir_folder"]
        streams = (
            _sorted_files(dataset_root / vis_folder / "images"),
            _sorted_files(dataset_root / vis_folder / "labels"),
            _sorted_files(dataset_root / ir_folder / "images"),
            _sorted_files(dataset_root / ir_folder / "labels"),
        )
        lengths = [len(stream) for stream in streams]
        if lengths[0] != lengths[1] or lengths[2] != lengths[3]:
            raise RuntimeError(
                f"Image/label mismatch in temporal pair {vis_folder}/{ir_folder}: "
                f"{lengths}"
            )
        stream_counts = {
            "vis_images": lengths[0],
            "vis_labels": lengths[1],
            "ir_images": lengths[2],
            "ir_labels": lengths[3],
        }
        if stream_counts != sequence["expected_stream_counts"]:
            raise RuntimeError(
                f"Unexpected stream counts for {sequence['id']}: "
                f"expected {sequence['expected_stream_counts']}, found {stream_counts}"
            )
        paired_frames = min(lengths[0], lengths[2])
        if paired_frames != int(sequence["expected_frames"]):
            raise RuntimeError(
                f"Unexpected frame count for {sequence['id']}: "
                f"expected {sequence['expected_frames']}, found {paired_frames}"
            )
        covered = []
        for phase in ("train", "embargo", "val"):
            start, stop = map(int, sequence["ranges"][phase])
            if not 0 <= start <= stop <= paired_frames:
                raise RuntimeError(f"Invalid {phase} range for {sequence['id']}")
            covered.extend(range(start, stop))
        if covered != list(range(paired_frames)):
            raise RuntimeError(
                f"Ranges for {sequence['id']} must cover every frame once in order"
            )
        sequence_rows = []
        for index, paths in enumerate(zip(*streams)):
            row = _row(dataset_root, sequence["id"], index, paths)
            phase = _phase_for_index(sequence, index)
            row["phase"] = phase
            rows.append(row)
            sequence_rows.append(row)
            phase_rows[phase].append(row)
        sequence_summaries.append(
            {
                "id": sequence["id"],
                "n_frames": len(sequence_rows),
                "stream_counts": stream_counts,
                "unpaired_vis_tail_frames": max(0, lengths[0] - paired_frames),
                "unpaired_ir_tail_frames": max(0, lengths[2] - paired_frames),
                "inventory_sha256": stable_json_hash(sequence_rows),
            }
        )

    phase_summaries = {}
    for phase, selected in phase_rows.items():
        areas = [area for row in selected for area in row["vis_box_areas"]]
        phase_summaries[phase] = {
            "n_frames": len(selected),
            "n_vis_boxes": sum(row["vis_box_count"] for row in selected),
            "n_vis_empty_frames": sum(row["vis_box_count"] == 0 for row in selected),
            "inventory_sha256": stable_json_hash(selected),
            "median_normalized_box_area": _median(areas),
        }
    return {
        "dataset_root": str(dataset_root),
        "n_source_frames": len(rows),
        "source_inventory_sha256": stable_json_hash(rows),
        "sequences": sequence_summaries,
        "phases": phase_summaries,
        "rows": rows,
    }


def _median(values):
    values = sorted(float(value) for value in values)
    if not values:
        return None
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return (values[middle - 1] + values[middle]) / 2


def load_temporal_split_manifest(path, dataset_root, verify=True):
    path = resolve_manifest_path(path)
    with path.open(encoding="utf-8") as input_file:
        manifest = json.load(input_file)
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported temporal split manifest schema")
    if manifest.get("id") != "rtdetr_train_temporal_validation_v1":
        raise ValueError("Unexpected temporal split manifest id")
    if manifest.get("status") != "frozen_before_training":
        raise ValueError("Temporal split must remain frozen_before_training")
    if manifest.get("pairing") != "existing_wisard_sorted_zip":
        raise ValueError("Temporal split must reproduce WiSARD sorted-zip pairing")
    if manifest.get("ground_truth") != "vis":
        raise ValueError("Temporal validation must use VIS ground truth")
    inventory = build_temporal_split_inventory(manifest, dataset_root)
    if verify:
        expected = manifest.get("expected", {})
        actual = {
            "n_source_frames": inventory["n_source_frames"],
            "source_inventory_sha256": inventory["source_inventory_sha256"],
            "phases": inventory["phases"],
            "sequences": inventory["sequences"],
        }
        if actual != expected:
            raise RuntimeError(
                "Temporal split source differs from the frozen manifest:\n"
                + json.dumps({"expected": expected, "actual": actual}, indent=2)
            )
    return manifest, inventory


def manifest_folder_pairs(manifest):
    return [
        (sequence["vis_folder"], sequence["ir_folder"])
        for sequence in manifest["sequences"]
    ]


def _item_key(item, dataset_root):
    item_type, data = item
    if item_type != 2:
        raise RuntimeError("Temporal split received a non-paired WiSARD item")
    (vis_image, vis_label), (ir_image, ir_label) = data
    return tuple(
        str(Path(path).resolve().relative_to(dataset_root))
        for path in (vis_image, vis_label, ir_image, ir_label)
    )


def select_temporal_split_items(items, dataset_root, inventory, phase):
    if phase not in {"train", "val"}:
        raise ValueError("Only train and val may select temporal split items")
    dataset_root = Path(dataset_root).resolve()
    indexed_items = {_item_key(item, dataset_root): item for item in items}
    if len(indexed_items) != len(items):
        raise RuntimeError("Duplicate items in temporal split source dataset")
    source_keys = {
        (row["vis_image"], row["vis_label"], row["ir_image"], row["ir_label"])
        for row in inventory["rows"]
    }
    if set(indexed_items) != source_keys:
        raise RuntimeError("Dataset items differ from the frozen temporal source inventory")
    selected = []
    for row in inventory["rows"]:
        if row["phase"] != phase:
            continue
        key = (row["vis_image"], row["vis_label"], row["ir_image"], row["ir_label"])
        selected.append(indexed_items[key])
    expected_count = inventory["phases"][phase]["n_frames"]
    if len(selected) != expected_count:
        raise RuntimeError(
            f"Temporal {phase} count mismatch: {len(selected)} != {expected_count}"
        )
    return selected
