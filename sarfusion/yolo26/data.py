"""Paired WiSARD data path for upstream Ultralytics 8.4.138."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
from dataclasses import dataclass
from itertools import repeat
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from torchvision.transforms import functional as tvf
from ultralytics.data.augment import Format
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import IMG_FORMATS, exif_size, get_hash, verify_image_label
from ultralytics.utils import LOGGER


FRAME_INDEX = re.compile(r"_(\d+)\.[^.]+$")


class RGBTFormat(Format):
    """Format HWC ``[B,G,R,IR]`` as CHW ``[R,G,B,IR]``."""

    def _format_img(self, img: np.ndarray) -> torch.Tensor:
        if img.ndim < 3:
            img = img[..., None]
        if img.shape[2] != 4:
            raise ValueError(f"RGBTFormat expects four channels, got {img.shape}.")
        chw = img.transpose(2, 0, 1)
        if random.uniform(0.0, 1.0) > self.bgr:
            chw = chw[[2, 1, 0, 3]]
        return torch.from_numpy(np.ascontiguousarray(chw))


def _verify_ir_image(path: str) -> None:
    with Image.open(path) as image:
        image.verify()
        width, height = exif_size(image)
        if width <= 9 or height <= 9:
            raise ValueError(f"image size {(width, height)} <10 pixels")
        if image.format is None or image.format.lower() not in IMG_FORMATS:
            raise ValueError(f"unsupported IR image format {image.format!r}")


def verify_rgb_ir_label(args: tuple) -> list:
    """Run the official RGB/label check and additionally validate paired IR."""
    rgb_file, ir_file, label_file, *official_args = args
    result = list(verify_image_label((rgb_file, label_file, *official_args)))
    if result[0] is None:
        return result
    try:
        _verify_ir_image(ir_file)
    except Exception as error:
        prefix = official_args[0]
        result[0] = None
        result[1] = result[2] = result[3] = result[4] = None
        result[8] = 1
        result[9] = (
            f"{prefix}{rgb_file}: ignoring RGB/IR pair because IR "
            f"{ir_file} is invalid: {error}"
        )
    return result


class PairedWiSARDYOLODataset(YOLODataset):
    """Modern YOLO dataset that loads a VIS path plus its paired IR path."""

    format_class = RGBTFormat
    MODALITY_MASKS = {
        "ir": (0.0, 1.0),
        "rgb": (1.0, 0.0),
        "fusion": (1.0, 1.0),
    }
    MODALITY_CODES = {"ir": 0, "rgb": 1, "fusion": 2}

    def __init__(
        self,
        *args,
        modal_dropout: bool = False,
        modal_dropout_probs=(0.2, 0.2, 0.6),
        **kwargs,
    ) -> None:
        cache = kwargs.get("cache")
        if cache not in {None, False}:
            raise ValueError("YOLO26 paired loading currently requires cache=False.")
        probabilities = [float(value) for value in modal_dropout_probs]
        if len(probabilities) != 3 or any(value < 0.0 for value in probabilities):
            raise ValueError("modal_dropout_probs must be three non-negative values.")
        if not np.isclose(sum(probabilities), 1.0):
            raise ValueError("modal_dropout_probs must sum to 1.0.")
        self.modal_dropout = bool(modal_dropout)
        self.modal_dropout_probs = probabilities
        self.ir_by_rgb: dict[str, str] = {}
        super().__init__(*args, **kwargs)
        missing = [path for path in self.im_files if path not in self.ir_by_rgb]
        if missing:
            raise RuntimeError(f"IR mapping lost for {len(missing)} RGB samples.")

    def get_img_files(self, img_path: str | list[str]) -> list[str]:
        pairs: list[tuple[str, str]] = []
        paths = img_path if isinstance(img_path, list) else [img_path]
        for item in paths:
            pair_list = Path(item)
            if not pair_list.is_file():
                raise FileNotFoundError(
                    f"PairedWiSARDYOLODataset requires a pair-list file, got {item}."
                )
            parent = pair_list.parent
            for line_number, raw_line in enumerate(
                pair_list.read_text(encoding="utf-8").splitlines(), start=1
            ):
                line = raw_line.strip()
                if not line:
                    continue
                pieces = [piece.strip() for piece in line.split(",")]
                if len(pieces) != 2:
                    raise ValueError(
                        f"{pair_list}:{line_number} must contain exactly RGB,IR."
                    )
                resolved = []
                for piece in pieces:
                    path = Path(piece)
                    if not path.is_absolute():
                        path = (parent / path).resolve()
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    if path.suffix.lower().lstrip(".") not in IMG_FORMATS:
                        raise ValueError(f"Unsupported image format: {path}")
                    resolved.append(str(path))
                pairs.append((resolved[0], resolved[1]))

        pairs.sort(key=lambda pair: pair[0])
        rgb_paths = [pair[0] for pair in pairs]
        if len(rgb_paths) != len(set(rgb_paths)):
            raise ValueError("Pair list contains duplicate RGB paths.")
        self.ir_by_rgb = dict(pairs)

        count = (
            self.fraction
            if isinstance(self.fraction, int)
            else round(len(rgb_paths) * self.fraction)
        )
        if count < len(rgb_paths):
            rgb_paths = rgb_paths[:count]
            self.ir_by_rgb = {path: self.ir_by_rgb[path] for path in rgb_paths}
        if not rgb_paths:
            raise FileNotFoundError(f"No valid RGB/IR pairs found in {img_path}.")
        return rgb_paths

    def get_cache_hash(self) -> str:
        ir_files = [self.ir_by_rgb[path] for path in self.im_files]
        return get_hash(self.label_files + self.im_files + ir_files)

    def verify_args(self) -> tuple:
        nkpt, ndim = self.data.get("kpt_shape", (0, 0))
        ir_files = [self.ir_by_rgb[path] for path in self.im_files]
        return verify_rgb_ir_label, zip(
            self.im_files,
            ir_files,
            self.label_files,
            repeat(self.prefix),
            repeat(self.use_keypoints),
            repeat(len(self.data["names"])),
            repeat(nkpt),
            repeat(ndim),
            repeat(self.single_cls),
        )

    @staticmethod
    def adapt_ir_to_rgb_canvas(rgb: np.ndarray, ir: np.ndarray) -> np.ndarray:
        """Replicate the historical height-resize and symmetric x-padding."""
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"Expected BGR image, got {rgb.shape}.")
        if ir.ndim == 3:
            ir = ir[:, :, 0]
        if ir.ndim != 2:
            raise ValueError(f"Expected one-channel IR image, got {ir.shape}.")
        rgb_height, rgb_width = rgb.shape[:2]
        ir_height, ir_width = ir.shape[:2]
        resized_width = int(ir_width * (rgb_height / ir_height))
        padding = rgb_width - resized_width
        if padding < 0 or padding % 2:
            raise ValueError(
                "Historical IR-to-RGB adaptation requires non-negative even "
                f"horizontal padding, got {padding}."
            )
        # Match ``sarfusion.data.wisard.adapt_ir2rgb`` byte-for-byte: that
        # path resized and padded uint8 CHW tensors with torchvision before
        # selecting the first IR channel.  Using OpenCV interpolation here
        # would introduce another small, avoidable preprocessing change.
        tensor = torch.from_numpy(np.ascontiguousarray(ir)).unsqueeze(0)
        tensor = tvf.resize(tensor, (rgb_height, resized_width))
        tensor = tvf.pad(tensor, (padding // 2, 0, padding // 2, 0))
        return tensor.squeeze(0).numpy()

    def load_image(
        self,
        i: int,
        rect_mode: bool = True,
        resize_short: bool = False,
    ) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
        image = self.ims[i]
        if image is None:
            rgb_file = self.im_files[i]
            ir_file = self.ir_by_rgb[rgb_file]
            rgb = cv2.imread(rgb_file, cv2.IMREAD_COLOR)
            # Preserve the historical SARFusion preprocessing exactly.  The
            # source IR JPEGs are visually monochrome but their decoded B/G/R
            # planes are not byte-identical because of JPEG chroma artifacts;
            # legacy experiments decoded BGR and selected channel 0.
            ir_bgr = cv2.imread(ir_file, cv2.IMREAD_COLOR)
            ir = None if ir_bgr is None else ir_bgr[:, :, 0]
            if rgb is None or ir is None:
                raise FileNotFoundError(f"Cannot read RGB/IR pair {rgb_file}, {ir_file}")
            aligned_ir = self.adapt_ir_to_rgb_canvas(rgb, ir)
            image = np.concatenate((rgb, aligned_ir[..., None]), axis=2)

            h0, w0 = image.shape[:2]
            if resize_short:
                ratio = self.imgsz / min(h0, w0)
                if ratio != 1:
                    width, height = (
                        (math.ceil(w0 * ratio), self.imgsz)
                        if h0 < w0
                        else (self.imgsz, math.ceil(h0 * ratio))
                    )
                    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
            elif rect_mode:
                ratio = self.imgsz / max(h0, w0)
                if ratio != 1:
                    width = min(math.ceil(w0 * ratio), self.imgsz)
                    height = min(math.ceil(h0 * ratio), self.imgsz)
                    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
            elif not (h0 == w0 == self.imgsz):
                image = cv2.resize(
                    image,
                    (self.imgsz, self.imgsz),
                    interpolation=cv2.INTER_LINEAR,
                )
            if image.ndim == 2:
                image = image[..., None]

            if self.augment and self.cache != "ram":
                self.ims[i], self.im_hw0[i], self.im_hw[i] = (
                    image,
                    (h0, w0),
                    image.shape[:2],
                )
                self.buffer.append(i)
                if 1 < len(self.buffer) >= self.max_buffer_length:
                    old_index = self.buffer.pop(0)
                    self.ims[old_index] = None
                    self.im_hw0[old_index] = None
                    self.im_hw[old_index] = None
            return image, (h0, w0), image.shape[:2]
        return self.ims[i], self.im_hw0[i], self.im_hw[i]

    def __getitem__(self, index: int) -> dict:
        sample = super().__getitem__(index)
        mode = "fusion"
        if self.augment and self.modal_dropout:
            mode = random.choices(
                ("ir", "rgb", "fusion"),
                weights=self.modal_dropout_probs,
                k=1,
            )[0]
            sample["img"] = sample["img"].clone()
            if mode == "ir":
                sample["img"][:3].zero_()
            elif mode == "rgb":
                sample["img"][3:4].zero_()
        sample["modality_mask"] = torch.tensor(
            self.MODALITY_MASKS[mode], dtype=torch.float32
        )
        sample["modality_code"] = torch.tensor(
            self.MODALITY_CODES[mode], dtype=torch.int64
        )
        sample["sample_index"] = torch.tensor(index, dtype=torch.int64)
        return sample

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        collated = YOLODataset.collate_fn(batch)
        for key in ("modality_mask", "modality_code", "sample_index"):
            collated[key] = torch.stack(tuple(collated[key]), dim=0)
        return collated


@dataclass(frozen=True)
class PairRecord:
    split: str
    sequence: str
    frame_index: int
    rgb: Path
    ir: Path
    label: Path


def _indexed_images(folder: Path) -> dict[int, Path]:
    indexed: dict[int, Path] = {}
    for path in sorted((folder / "images").iterdir()):
        if not path.is_file() or path.suffix.lower().lstrip(".") not in IMG_FORMATS:
            continue
        match = FRAME_INDEX.search(path.name)
        if match is None:
            raise ValueError(f"Cannot extract frame index from {path}.")
        frame_index = int(match.group(1))
        if frame_index in indexed:
            raise ValueError(f"Duplicate frame index {frame_index} in {folder}.")
        indexed[frame_index] = path.resolve()
    return indexed


def discover_split_records(root: Path, split: str, pairs: list[dict]) -> tuple[list[PairRecord], list[dict]]:
    records: list[PairRecord] = []
    dropped: list[dict] = []
    for item in pairs:
        rgb_folder = root / item["rgb"]
        ir_folder = root / item["ir"]
        rgb_images = _indexed_images(rgb_folder)
        ir_images = _indexed_images(ir_folder)
        common = sorted(rgb_images.keys() & ir_images.keys())
        dropped.append(
            {
                "sequence": f"{item['rgb']}+{item['ir']}",
                "rgb_only_frame_indices": sorted(rgb_images.keys() - ir_images.keys()),
                "ir_only_frame_indices": sorted(ir_images.keys() - rgb_images.keys()),
            }
        )
        for frame_index in common:
            rgb = rgb_images[frame_index]
            label = rgb.parent.parent / "labels" / f"{rgb.stem}.txt"
            if not label.is_file():
                raise FileNotFoundError(label)
            records.append(
                PairRecord(
                    split=split,
                    sequence=f"{item['rgb']}+{item['ir']}",
                    frame_index=frame_index,
                    rgb=rgb,
                    ir=ir_images[frame_index],
                    label=label.resolve(),
                )
            )
    records.sort(key=lambda record: (record.sequence, record.frame_index))
    return records, dropped


def _hash_file(digest: Any, role: str, relative_path: str, path: Path) -> None:
    digest.update(role.encode("ascii"))
    digest.update(b"\0")
    digest.update(relative_path.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(path.stat().st_size).encode("ascii"))
    digest.update(b"\0")
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)


def dataset_content_sha256(records: list[PairRecord], root: Path) -> str:
    """Hash paths, sizes and bytes for every RGB, IR and label file."""
    digest = hashlib.sha256()
    for record in records:
        digest.update(record.split.encode("ascii"))
        digest.update(b"\0")
        digest.update(record.sequence.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record.frame_index).encode("ascii"))
        digest.update(b"\0")
        for role, path in (("rgb", record.rgb), ("ir", record.ir), ("label", record.label)):
            _hash_file(digest, role, path.relative_to(root).as_posix(), path)
    return digest.hexdigest()


def build_stage_a_dataset(
    split_config: dict,
    output_dir: Path,
) -> dict[str, Any]:
    """Discover, validate, fully hash and materialize Stage A pair lists."""
    root = Path(split_config["root"]).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[PairRecord] = []
    dropped_by_split: dict[str, list[dict]] = {}
    records_by_split: dict[str, list[PairRecord]] = {}
    for split in ("train", "val"):
        records, dropped = discover_split_records(
            root,
            split,
            split_config[split]["pairs"],
        )
        expected = int(split_config[split]["expected_pairs"])
        if len(records) != expected:
            raise RuntimeError(
                f"Stage A {split} inventory has {len(records)} pairs, expected {expected}."
            )
        records_by_split[split] = records
        dropped_by_split[split] = dropped
        all_records.extend(records)

    train_paths = {record.rgb for record in records_by_split["train"]}
    val_paths = {record.rgb for record in records_by_split["val"]}
    if overlap := train_paths & val_paths:
        raise RuntimeError(f"Train/val RGB overlap detected: {len(overlap)} files.")

    content_hash = dataset_content_sha256(all_records, root)
    expected_hash = split_config.get("expected_content_sha256")
    if expected_hash and content_hash != expected_hash:
        raise RuntimeError(
            "Stage A dataset content hash mismatch: "
            f"expected {expected_hash}, got {content_hash}."
        )

    pair_lists: dict[str, str] = {}
    manifest_records: list[dict[str, Any]] = []
    for split, records in records_by_split.items():
        pair_path = output_dir / f"{split}_pairs.txt"
        lines = []
        for record in records:
            if "," in str(record.rgb) or "," in str(record.ir):
                raise ValueError("Pair-list paths may not contain commas.")
            lines.append(f"{record.rgb},{record.ir}")
            manifest_records.append(
                {
                    "split": split,
                    "sequence": record.sequence,
                    "frame_index": record.frame_index,
                    "rgb": record.rgb.relative_to(root).as_posix(),
                    "ir": record.ir.relative_to(root).as_posix(),
                    "label": record.label.relative_to(root).as_posix(),
                    "rgb_bytes": record.rgb.stat().st_size,
                    "ir_bytes": record.ir.stat().st_size,
                    "label_bytes": record.label.stat().st_size,
                }
            )
        pair_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        pair_lists[split] = str(pair_path.resolve())

    data_yaml = output_dir / "stage_a_data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": pair_lists["train"],
                "val": pair_lists["val"],
                "channels": 4,
                "names": {0: "person"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema": "sarfusion.yolo26.stage_a.dataset.v1",
        "root": str(root),
        "counts": {split: len(records) for split, records in records_by_split.items()},
        "content_sha256": content_hash,
        "dropped_unpaired_frames": dropped_by_split,
        "records": manifest_records,
        "data_yaml": str(data_yaml.resolve()),
    }
    manifest_path = output_dir / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    LOGGER.info(
        "YOLO26 Stage A inventory: %d train, %d val, sha256=%s",
        manifest["counts"]["train"],
        manifest["counts"]["val"],
        content_hash,
    )
    return manifest
