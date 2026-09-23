"""Ultralytics 8.4.138 trainer glue for paired RGB+IR YOLO26."""

from __future__ import annotations

import json
from copy import copy
from pathlib import Path
from typing import Any

import torch
from ultralytics.data.utils import get_split_fraction
from ultralytics.models import yolo
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import colorstr
from ultralytics.utils.torch_utils import unwrap_model

from .data import PairedWiSARDYOLODataset


class YOLO26FusionTrainer(DetectionTrainer):
    """Detection trainer with paired dataset and preregistered selection."""

    MAP50_KEY = "metrics/mAP50(B)"

    def __init__(
        self,
        *args,
        dataset_options: dict[str, Any] | None = None,
        expected_batch: int | None = None,
        checkpoint_min_delta: float = 0.001,
        trace_batches: int = 20,
        **kwargs,
    ) -> None:
        self.dataset_options = dict(dataset_options or {})
        self.expected_batch = expected_batch
        self.checkpoint_min_delta = float(checkpoint_min_delta)
        self.trace_batches = int(trace_batches)
        self._traced_batches = 0
        self.selection_best_epoch: int | None = None
        super().__init__(*args, **kwargs)
        self.trace_path = self.save_dir / "data_trace.jsonl"
        self.selection_path = self.save_dir / "checkpoint_selection.jsonl"
        self.add_callback("on_train_epoch_start", self._assert_frozen_batch)

    def _assert_frozen_batch(self, trainer) -> None:
        del trainer
        if self.expected_batch is None:
            return
        if int(self.batch_size) != int(self.expected_batch):
            raise RuntimeError(
                "Ultralytics attempted to change the preregistered physical "
                f"batch from {self.expected_batch} to {self.batch_size}; "
                "the run is invalid and has been stopped."
            )

    def get_dataset(self):
        data = super().get_dataset()
        # Ultralytics names a collapsed single class "item".  The pretrained
        # head row was explicitly remapped from COCO's "person", so retain the
        # semantically correct name in checkpoints and reports.
        data["names"] = {0: "person"}
        data["nc"] = 1
        return data

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        stride = max(int(unwrap_model(self.model).stride.max()), 32)
        options = dict(self.dataset_options)
        modal_dropout = bool(options.pop("modal_dropout", True)) and mode == "train"
        probabilities = options.pop("modal_dropout_probs", (0.2, 0.2, 0.6))
        if options:
            raise ValueError(f"Unknown paired-dataset options: {sorted(options)}")
        return PairedWiSARDYOLODataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=copy(self.args),
            rect=self.args.rect or mode == "val",
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=stride,
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=get_split_fraction(self.args.fraction, mode),
            modal_dropout=modal_dropout,
            modal_dropout_probs=probabilities,
        )

    def get_validator(self):
        return yolo.detect.DetectionValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )

    def preprocess_batch(self, batch: dict) -> dict:
        batch = super().preprocess_batch(batch)
        if self._traced_batches < self.trace_batches:
            record = {
                "epoch": int(getattr(self, "epoch", -1)),
                "batch": self._traced_batches,
                "im_file": [str(path) for path in batch.get("im_file", ())],
                "sample_index": batch["sample_index"].detach().cpu().tolist(),
                "modality_mask": batch["modality_mask"].detach().cpu().tolist(),
                "modality_code": batch["modality_code"].detach().cpu().tolist(),
            }
            with self.trace_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
            self._traced_batches += 1
        return batch

    def validate(self):
        """Select ``best.pt`` on mAP50 with the frozen minimum delta."""
        metrics = self.validator(self)
        if metrics is None:
            return None, None
        metrics.pop("fitness", None)
        if self.MAP50_KEY not in metrics:
            raise RuntimeError(
                f"Validator did not return required key {self.MAP50_KEY!r}: "
                f"{sorted(metrics)}"
            )
        raw_map50 = float(metrics[self.MAP50_KEY])
        previous_best = self.best_fitness
        improved = (
            previous_best is None
            or raw_map50 > float(previous_best) + self.checkpoint_min_delta
        )
        if improved:
            self.best_fitness = raw_map50
            self.selection_best_epoch = int(self.epoch) + 1
            selection_fitness = raw_map50
        else:
            # Keep save_model() from replacing best.pt on a near-tie while
            # retaining the raw metric in results.csv.
            selection_fitness = min(
                raw_map50,
                float(self.best_fitness) - 1e-12,
            )
        metrics["selection/mAP50"] = raw_map50
        metrics["selection/best_mAP50"] = float(self.best_fitness)
        metrics["selection/best_epoch"] = int(self.selection_best_epoch or 0)
        record = {
            "epoch": int(self.epoch) + 1,
            "raw_mAP50": raw_map50,
            "previous_best_mAP50": previous_best,
            "best_mAP50": float(self.best_fitness),
            "best_epoch": self.selection_best_epoch,
            "min_delta": self.checkpoint_min_delta,
            "improved": improved,
        }
        with self.selection_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        return metrics, selection_fitness
