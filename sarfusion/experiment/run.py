import os
import sys
import shutil
import time
import math
from copy import deepcopy
from safetensors import safe_open
from collections import defaultdict

import torch

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from torch.optim import AdamW
from torchmetrics import MetricCollection
from tqdm import tqdm

from sarfusion.data.wisard import TEST_FOLDERS, TRAIN_FOLDERS, VAL_FOLDERS, generate_wisard_filelist, get_wisard_folders
from sarfusion.data.tile_aggregation import aggregate_tile_predictions
from sarfusion.experiment.yolo import WisardTrainer, install_wandb_empty_curve_guard
from sarfusion.models.yolov10 import YOLOv10WiSARD
from sarfusion.utils.structures import LossOutput, WrapperModelOutput
from sarfusion.utils.logger import get_logger
from sarfusion.data import get_dataloaders
from sarfusion.utils.structures import DataDict
from sarfusion.experiment.utils import WrapperModule
from sarfusion.models.loss import build_loss
from sarfusion.models import build_model
from sarfusion.utils.metrics import DetectionEvaluator, Evaluator, build_evaluator
from sarfusion.utils.reproducibility import (
    ReproducibilityTrace,
    configure_reproducibility,
    model_digests,
    prepare_rtdetr_model_for_determinism,
    runtime_fingerprint,
    tensor_digest,
    training_source_runtime_fields,
    verify_training_source_manifest,
)
from sarfusion.experiment.modality_consistency import (
    matched_detection_consistency_loss,
    modality_consistency_epoch_scale,
    validate_modality_consistency_config,
)
from sarfusion.experiment.box_guided_alignment import (
    box_guided_alignment_epoch_scale,
    box_guided_alignment_loss,
    validate_box_guided_alignment_config,
    validate_box_guided_training_contract,
)
from sarfusion.utils.utils import (
    RunningAverage,
    load_yaml,
    write_yaml,
)

from .utils import (
    SchedulerStepMoment,
    check_nan,
    get_experiment_tracker,
    get_scheduler,
    handle_oom,
    parse_params,
)
from copy import deepcopy

logger = get_logger(__name__)


HEAD_AND_DINO_PARAMETER_NAMES = (
    "mixed_query_content",
    "dn_label_embeddings",
    "enc_output",
    "pos_trans",
    "bbox_embed",
    "class_embed",
)


def partition_optimizer_parameters(named_parameters):
    """Partition parameters once, keeping experimental gates isolated."""
    groups = {
        "backbone": [],
        "new_modules": [],
        "head_and_dino": [],
        "box_guidance": [],
        "reliability_gate": [],
        "alignment_gate": [],
    }
    for name, parameter in named_parameters:
        if any(
            key in name for key in ("common_projection", "guidance_predictor")
        ):
            groups["box_guidance"].append(parameter)
        elif "alignment_gates" in name:
            groups["alignment_gate"].append(parameter)
        elif "reliability_gates" in name:
            groups["reliability_gate"].append(parameter)
        elif any(key in name for key in HEAD_AND_DINO_PARAMETER_NAMES):
            groups["head_and_dino"].append(parameter)
        elif any(key in name for key in ("ir_backbone", "channel_fusion")):
            groups["new_modules"].append(parameter)
        else:
            groups["backbone"].append(parameter)
    return groups


class Run:
    def __init__(self):
        self.params = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.denormalize = None
        self.experiment = None
        self.tracker = None
        self.dataset_params = None
        self.train_params = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.best_metric = None
        self.best_epoch = None
        self.scheduler_step_moment = None
        self.watch_metric = None
        self.train_evaluator: Evaluator = None
        self.val_evaluator: Evaluator = None
        if "." not in sys.path:
            sys.path.extend(".")
        self.global_train_step = 0
        self.global_val_step = 0
        self.validation_json = None
        self.accelerator = None
        self._ended = False
        self.reproducibility = {}
        self.repro_trace = ReproducibilityTrace(None)
        self.training_source_manifest = None
        self.reliability_gate_optimizer_group_index = None
        self.alignment_gate_optimizer_group_index = None
        self.box_guidance_optimizer_group_index = None
        self.modality_consistency_params = validate_modality_consistency_config(None)
        self.box_guided_alignment_params = validate_box_guided_alignment_config(
            None
        )

    def parse_params(self, params: dict):
        self.params = deepcopy(params)

        (
            self.train_params,
            self.dataset_params,
            self.dataloader_params,
            self.model_params,
        ) = parse_params(self.params)

    def init(self, params: dict):
        self.reproducibility = deepcopy(params.get("reproducibility", {}))
        self.training_source_manifest = verify_training_source_manifest(
            self.reproducibility.get("training_source_manifest_id"),
            self.reproducibility.get("training_source_manifest_sha256"),
        )
        if self.training_source_manifest is not None and not bool(
            self.reproducibility.get("trace", False)
        ):
            raise ValueError(
                "A declared training-source manifest requires "
                "reproducibility.trace=true"
            )
        configure_reproducibility(
            params["seed"],
            deterministic=self.reproducibility.get("deterministic", False),
            warn_only=self.reproducibility.get("warn_only", False),
        )
        self.seg_trainer = None
        logger.info("Parameters: ")
        write_yaml(params, file=sys.stdout)
        self.parse_params(params)

        kwargs = [
            DistributedDataParallelKwargs(find_unused_parameters=True),
        ]
        logger.info("Creating Accelerator")
        self.accelerator = Accelerator(
            even_batches=False,
            kwargs_handlers=kwargs,
            split_batches=False,
            mixed_precision=self.train_params.get("precision", None),
            gradient_accumulation_steps=self.train_params.get("gradient_accumulation_steps", 1),
        )
        logger.info("Initiliazing tracker...")
        self.tracker = get_experiment_tracker(self.accelerator, self.params)
        self.url = self.tracker.url
        self.name = self.tracker.name
        trace_enabled = bool(self.reproducibility.get("trace", False))
        trace_path = (
            os.path.join(self.tracker.local_dir, "reproducibility_trace.jsonl")
            if trace_enabled and self.tracker.local_dir
            else None
        )
        self.repro_trace = ReproducibilityTrace(trace_path)
        if self.training_source_manifest is not None and not self.repro_trace.enabled:
            raise RuntimeError(
                "Training-source provenance requires a writable local "
                "reproducibility trace"
            )
        self.repro_trace.write(
            "runtime",
            seed=int(params["seed"]),
            data_seed=int(
                self.reproducibility.get("data_seed", params["seed"])
            ),
            model_seed=self.reproducibility.get("model_seed"),
            training_seed=int(
                self.reproducibility.get("training_seed", params["seed"])
            ),
            repetition=params.get("repetition"),
            **training_source_runtime_fields(self.training_source_manifest),
            **runtime_fingerprint(),
        )
        (self.train_loader, self.val_loader, self.test_loader), self.denormalize = (
            get_dataloaders(
                self.dataset_params,
                self.dataloader_params,
                seed=self.reproducibility.get("data_seed", params["seed"]),
            )
        )
        model_name = self.model_params.get("name")
        logger.info(f"Creating model {model_name}")
        model_seed = self.reproducibility.get("model_seed")
        if model_seed is not None:
            set_seed(int(model_seed))
        self.model = build_model(params=self.model_params)
        if model_seed is not None or "training_seed" in self.reproducibility:
            # Decouple model initialization from stochastic training operators.
            set_seed(
                int(self.reproducibility.get("training_seed", params["seed"]))
            )
        if self.reproducibility.get("deterministic", False):
            attention_modules = prepare_rtdetr_model_for_determinism(self.model)
            logger.info(
                "Deterministic RT-DETR attention enabled for %s modules",
                attention_modules,
            )
        if self.repro_trace.enabled:
            self.repro_trace.write(
                "model_initialized", **model_digests(self.model)
            )
        logger.info("Creating criterion")
        self.model = WrapperModule(self.model, self.criterion)
        self.task = self.params.get("task", None)

        if self.train_params.get("compile", False):
            logger.info("Compiling model")
            self.model = torch.compile(self.model)
        logger.info("Preparing model, optimizer, dataloaders and scheduler")

        self.model = self.accelerator.prepare(self.model)
        
        if self.params.get("train"):
            self._prep_for_training()

        self.compute_val_metrics = lambda: self._compute_metrics(self.val_evaluator)
        if self.val_loader:
            logger.info("Preparing validation dataloader")
            self._prep_for_validation()

        self._load_state()

    def _prep_for_training(self):
        self.criterion = build_loss(self.params["loss"], model=self.model)
        self.modality_consistency_params = validate_modality_consistency_config(
            self.train_params.get("modality_consistency")
        )
        self.box_guided_alignment_params = validate_box_guided_alignment_config(
            self.train_params.get("box_guided_alignment")
        )
        consistency_enabled = self.modality_consistency_params["enabled"]
        dataset_consistency_enabled = bool(
            self.dataset_params.get("paired_consistency", False)
        )
        if consistency_enabled != dataset_consistency_enabled:
            raise ValueError(
                "train.modality_consistency.enabled and "
                "dataset.paired_consistency must be enabled or disabled together"
            )
        if consistency_enabled:
            if not self.dataset_params.get("modal_dropout", False):
                raise ValueError(
                    "mixed consistency training must retain supervised modal dropout"
                )
            if (
                self.dataset_params.get(
                    "modal_dropout_coordinate_contract", "native"
                )
                != "native"
            ):
                raise ValueError(
                    "mixed consistency training must retain native-coordinate "
                    "IR supervision"
                )
        self.box_guided_alignment_params = validate_box_guided_training_contract(
            self.box_guided_alignment_params,
            self.dataset_params,
            self.model_params,
            modality_consistency_enabled=consistency_enabled,
        )
        self.watch_metric = self.train_params["watch_metric"]
        self.greater_is_better = self.train_params.get("greater_is_better", True)
        checkpoint_min_delta = float(
            self.train_params.get("checkpoint_min_delta", 0.0)
        )
        if checkpoint_min_delta < 0:
            raise ValueError("checkpoint_min_delta must be non-negative")
        patience = self.train_params.get("early_stopping_patience")
        if patience is not None:
            if int(patience) < 1:
                raise ValueError("early_stopping_patience must be positive")
            if not self.train_params.get("run_validation", True):
                raise ValueError("Early stopping requires run_validation=true")
        logger.info("Creating optimizer")
        
        backbone_lr = self.train_params.get("backbone_lr", self.train_params["initial_lr"])
        # Keep "dino_lr" as the public configuration key for compatibility.
        # This group also contains detection heads shared by RT-DETR.
        head_and_dino_lr = self.train_params.get(
            "dino_lr", self.train_params["initial_lr"]
        )
        parameter_groups = partition_optimizer_parameters(
            self.model.named_parameters()
        )
        backbone_params = parameter_groups["backbone"]
        new_module_params = parameter_groups["new_modules"]
        head_and_dino_params = parameter_groups["head_and_dino"]
        reliability_gate_params = parameter_groups["reliability_gate"]
        alignment_gate_params = parameter_groups["alignment_gate"]
        box_guidance_params = parameter_groups["box_guidance"]
        box_guidance_lr = float(
            self.train_params.get(
                "box_guidance_lr", self.train_params["initial_lr"]
            )
        )
        if box_guidance_lr <= 0:
            raise ValueError("box_guidance_lr must be positive")
        if "box_guidance_lr" in self.train_params and not box_guidance_params:
            raise ValueError(
                "box_guidance_lr was configured but the model has no "
                "box-guidance parameters"
            )
        reliability_gate_lr = float(
            self.train_params.get(
                "reliability_gate_lr", self.train_params["initial_lr"]
            )
        )
        if reliability_gate_lr <= 0:
            raise ValueError("reliability_gate_lr must be positive")
        if (
            "reliability_gate_lr" in self.train_params
            and not reliability_gate_params
        ):
            raise ValueError(
                "reliability_gate_lr was configured but the model has no "
                "reliability-gate parameters"
            )
        alignment_gate_lr = float(
            self.train_params.get(
                "alignment_gate_lr", self.train_params["initial_lr"]
            )
        )
        if alignment_gate_lr <= 0:
            raise ValueError("alignment_gate_lr must be positive")
        if "alignment_gate_lr" in self.train_params and not alignment_gate_params:
            raise ValueError(
                "alignment_gate_lr was configured but the model has no "
                "residual-alignment-gate parameters"
            )

        frozen = sum(1 for _, p in self.model.named_parameters() if not p.requires_grad)
        logger.info(f"Frozen params: {frozen}")

        logger.info("Sample new_module params:")
        for name, _ in self.model.named_parameters():
            if any(k in name for k in ["ir_backbone", "channel_fusion"]):
                logger.info(f"  NEW: {name}")
                break  # solo il primo per non spammare
        logger.info("Sample backbone params:")
        for name, _ in self.model.named_parameters():
            if not any(k in name for k in ["ir_backbone", "channel_fusion"]):
                logger.info(f"  BACKBONE: {name}")
                break

        total = sum(1 for _, p in self.model.named_parameters())
        trainable = sum(1 for _, p in self.model.named_parameters() if p.requires_grad)
        logger.info(f"Total params tensors: {total}, Trainable: {trainable}")

        logger.info(
            f"New module params: {len(new_module_params)}, "
            f"Detection-head/DINO-specific params: {len(head_and_dino_params)}, "
            f"Box-guidance params: {len(box_guidance_params)}, "
            f"Reliability-gate params: {len(reliability_gate_params)}, "
            f"Residual-alignment-gate params: {len(alignment_gate_params)}, "
            f"Backbone params: {len(backbone_params)}"
        )
        logger.info(
            f"LR for new modules: {self.train_params['initial_lr']}, "
            f"LR for detection-head/DINO-specific modules: {head_and_dino_lr}, "
            f"LR for box guidance: {box_guidance_lr}, "
            f"LR for reliability gate: {reliability_gate_lr}, "
            f"LR for residual alignment gate: {alignment_gate_lr}, "
            f"LR for backbone: {backbone_lr}"
        )

        optimizer_groups = [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": new_module_params, "lr": self.train_params["initial_lr"]},
            {"params": head_and_dino_params, "lr": head_and_dino_lr},
        ]
        self.reliability_gate_optimizer_group_index = None
        self.box_guidance_optimizer_group_index = None
        if box_guidance_params:
            self.box_guidance_optimizer_group_index = len(optimizer_groups)
            optimizer_groups.append(
                {"params": box_guidance_params, "lr": box_guidance_lr}
            )
        if reliability_gate_params:
            self.reliability_gate_optimizer_group_index = len(optimizer_groups)
            optimizer_groups.append(
                {"params": reliability_gate_params, "lr": reliability_gate_lr}
            )
        self.alignment_gate_optimizer_group_index = None
        if alignment_gate_params:
            self.alignment_gate_optimizer_group_index = len(optimizer_groups)
            optimizer_groups.append(
                {"params": alignment_gate_params, "lr": alignment_gate_lr}
            )
        self.optimizer = AdamW(optimizer_groups)

        scheduler_params = self.train_params.get("scheduler", None)
        if scheduler_params:
            self.scheduler, self.scheduler_step_moment = get_scheduler(
                scheduler_params=scheduler_params,
                optimizer=self.optimizer,
                num_training_steps=self.train_params["max_epochs"]
                * len(self.train_loader)
                // self.train_params.get("gradient_accumulation_steps", 1),
            )

        self.train_loader, self.optimizer = self.accelerator.prepare(
            self.train_loader, self.optimizer
        )
        self.scheduler = (
            self.accelerator.prepare(self.scheduler) if self.scheduler else None
        )
        self._init_evaluator(self.params, phase="train")
        
    def _prep_for_validation(self):
        self.val_loader = self.accelerator.prepare(self.val_loader)
        self._init_evaluator(self.params, phase="val")

    def _load_state(self):
        if self.tracker.accelerator_state_dir:
            overwritten = False
            # Merge image_encoder dict with the state dict
            if (
                "checkpoint" in self.model_params
                and self.params["model"]["name"] != "lam_no_vit"
            ):
                if hasattr(self.model, "module"):
                    model = self.model.module.model
                else:
                    model = self.model.model
                shutil.copyfile(
                    self.tracker.accelerator_state_dir + "/pytorch_model.bin",
                    self.tracker.accelerator_state_dir + "/pytorch_model.bin.bak",
                )
                state_dict = torch.load(
                    self.tracker.accelerator_state_dir + "/pytorch_model.bin"
                )
                state_dict = {
                    **{
                        "model.image_encoder." + k: v
                        for k, v in model.image_encoder.state_dict().items()
                    },
                    **state_dict,
                }
                torch.save(
                    state_dict,
                    self.tracker.accelerator_state_dir + "/pytorch_model.bin",
                )
                overwritten = True

            try:
                self.accelerator.load_state(self.tracker.accelerator_state_dir)
                # Ripristinate old state
            finally:
                if (
                    "checkpoint" in self.model_params
                    and self.params["model"]["name"] != "lam_no_vit"
                    and overwritten
                ):
                    shutil.copyfile(
                        self.tracker.accelerator_state_dir + "/pytorch_model.bin.bak",
                        self.tracker.accelerator_state_dir + "/pytorch_model.bin",
                    )
                    os.remove(
                        self.tracker.accelerator_state_dir + "/pytorch_model.bin.bak"
                    )

    def launch(self):
        
        if self.train_params:
            logger.info("Start training loop...")
            # Train the Model
            with self.tracker.train():
                logger.info(
                    f"Running Model Training {self.params.get('experiment').get('name')}"
                )

                patience = self.train_params.get("early_stopping_patience", None)
                epochs_without_improvement = 0

                for epoch in range(self.train_params["max_epochs"]):
                    logger.info(
                        "Epoch: {}/{}".format(epoch + 1, self.train_params["max_epochs"])
                    )
                    self.train_epoch(epoch)

                    metrics = None
                    if (
                        self.train_params.get("run_validation", True)
                        and
                        self.val_loader
                        and epoch % self.train_params.get("val_frequency", 1) == 0
                    ):
                        with self.tracker.validate():
                            logger.info(f"Running Model Validation")
                            metrics = self.validate_epoch(epoch)
                            self._scheduler_step(SchedulerStepMoment.EPOCH, metrics)
                    improved = self._update_best_metric(epoch, metrics)

                    should_stop = False
                    if patience is not None and metrics is not None:
                        if improved:
                            epochs_without_improvement = 0
                        else:
                            epochs_without_improvement += 1
                            logger.info(
                                f"No improvement for {epochs_without_improvement}/{patience} epochs"
                            )
                        should_stop = epochs_without_improvement >= patience

                    save_final_only = self.train_params.get(
                        "save_final_checkpoint_only", False
                    )
                    is_final_epoch = epoch == self.train_params["max_epochs"] - 1
                    if self.train_params.get("save_checkpoints", True):
                        self.save_training_state(
                            epoch,
                            improved=improved,
                            save_latest=(
                                not save_final_only or is_final_epoch or should_stop
                            ),
                        )

                    if should_stop:
                        logger.info(
                            f"Early stopping triggered after {epoch + 1} epochs")
                        break
        else:
            logger.info("No training params, no training")

        if self.test_loader and self.params.get("run_test", True):
            self.test()
        self.end()

    def _metric_is_better(self, metric):
        metric = float(metric)
        if not math.isfinite(metric):
            raise ValueError(
                f"Checkpoint metric {self.watch_metric!r} must be finite, got {metric}"
            )
        if self.best_metric is None:
            return True
        min_delta = float(self.train_params.get("checkpoint_min_delta", 0.0))
        if self.greater_is_better:
            return metric > self.best_metric + min_delta
        return metric < self.best_metric - min_delta

    def _update_best_metric(self, epoch, metrics=None):
        if metrics is None:
            return False
        if self.watch_metric not in metrics:
            raise KeyError(
                f"watch_metric={self.watch_metric!r} is absent from validation "
                f"metrics {sorted(metrics)}"
            )
        metric = float(metrics[self.watch_metric])
        if not self._metric_is_better(metric):
            return False
        previous = self.best_metric
        self.best_metric = metric
        self.best_epoch = int(epoch)
        logger.info(
            "New best checkpoint at epoch %s: %s=%s (previous=%s, min_delta=%s)",
            epoch + 1,
            self.watch_metric,
            metric,
            previous,
            float(self.train_params.get("checkpoint_min_delta", 0.0)),
        )
        return True

    def save_training_state(self, epoch, improved=False, save_latest=True):
        if improved:
            self.tracker.add_summary(
                {
                    "best_epoch": self.best_epoch + 1,
                    f"best_{self.watch_metric}": self.best_metric,
                }
            )
            # Queue metadata before the comparatively slow state write, giving
            # asynchronous trackers time to persist it even on the final epoch.
            self.tracker.log_training_state(epoch=epoch, subfolder="best")
        if save_latest:
            self.tracker.log_training_state(epoch=epoch, subfolder="latest")

    def _get_lr(self):
        if self.scheduler is None:
            return self.train_params["initial_lr"]
        try:
            if hasattr(self.scheduler, "get_lr"):
                return self.scheduler.get_lr()[0]
        except NotImplementedError:
            pass
        if hasattr(self.scheduler, "optimizer"):
            return self.scheduler.optimizer.param_groups[0]["lr"]
        return self.scheduler.optimizers[0].param_groups[0]["lr"]

    def _scheduler_step(self, moment, metrics=None):
        if moment != self.scheduler_step_moment or self.scheduler is None:
            return
        if moment == SchedulerStepMoment.BATCH:
            self.scheduler.step()
        elif moment == SchedulerStepMoment.EPOCH:
            self.scheduler.step(metrics[self.watch_metric])

    def _forward(
        self,
        input_dict: dict,
        epoch: int,
        batch_idx: int,
    ):
        try:
            outputs = self.model(input_dict)
        except RuntimeError as e:
            if "out of memory" in str(e):
                handle_oom(
                    self.model,
                    input_dict,
                    self.optimizer,
                    epoch,
                    batch_idx,
                )
                return e
            raise e
        return outputs

    def _modality_consistency_forward(self, batch_dict, epoch, batch_idx):
        """Run clean stop-gradient teacher and degraded paired student."""
        epoch_scale = modality_consistency_epoch_scale(
            self.modality_consistency_params,
            epoch,
        )
        if epoch_scale == 0.0:
            return None

        required = (
            "consistency_teacher_pixel_values",
            "consistency_student_pixel_values",
            "consistency_pixel_mask",
        )
        missing = [name for name in required if name not in batch_dict]
        if missing:
            raise ValueError(
                "paired consistency batch is missing fields: "
                + ", ".join(missing)
            )
        teacher_input = DataDict(
            pixel_values=batch_dict.consistency_teacher_pixel_values,
            pixel_mask=batch_dict.consistency_pixel_mask,
            labels=None,
        )
        student_input = DataDict(
            pixel_values=batch_dict.consistency_student_pixel_values,
            pixel_mask=batch_dict.consistency_pixel_mask,
            labels=None,
        )

        was_training = self.model.training
        try:
            self.model.eval()
            with torch.no_grad():
                teacher_output = self._forward(
                    teacher_input,
                    epoch,
                    batch_idx,
                )
        finally:
            self.model.train(was_training)
        student_output = self._forward(
            student_input,
            epoch,
            batch_idx,
        )
        loss = matched_detection_consistency_loss(
            teacher_output,
            student_output,
            self.modality_consistency_params,
            epoch_scale=epoch_scale,
        )
        return student_input, student_output, loss

    def _backward(
        self,
        batch_idx,
        input_dict,
        outputs: WrapperModelOutput,
        loss_normalizer,
        auxiliary_loss=None,
    ):
        loss = outputs.loss
        if isinstance(loss, torch.Tensor):
            loss_value = loss
        elif hasattr(loss, "value"):
            loss_value = loss.value
        elif isinstance(loss, dict) and "value" in loss:
            loss_value = loss["value"]
        else:
            raise ValueError(f"Unexpected loss type: {type(loss)}, value: {loss}")

        if auxiliary_loss is not None:
            loss_value = loss_value + auxiliary_loss.value
        loss_value = loss_value / loss_normalizer
        self.accelerator.backward(loss_value)
        check_nan(
            self.model,
            input_dict,
            outputs,
            loss_value,
            batch_idx,
            self.train_params,
        )
        return loss_value

    def _init_evaluator(self, params, phase="train"):
        evaluator = params.get(f"{phase}_evaluation", None)
        evaluator = build_evaluator(
            evaluator, self.task, id2class=self.val_loader.dataset.id2class
        )
        setattr(self, f"{phase}_evaluator", self.accelerator.prepare(evaluator))

    def _update_metrics(
        self,
        evaluator: MetricCollection,
        batch_dict: DataDict,
        result_dict: WrapperModelOutput,
    ):
        with self.accelerator.no_sync(model=evaluator):
            evaluator.update(batch_dict, result_dict)
    
    def _compute_metrics(
        self,
        evaluator: MetricCollection,
    ):
        with self.accelerator.no_sync(model=evaluator):
            metrics_dict = evaluator.compute()
        metrics_dict = {
            k: v.item() if isinstance(v, torch.Tensor) and v.dim() == 0 else v for k, v in metrics_dict.items()
        }
        return metrics_dict

    def _update_val_metrics(
        self,
        batch_dict: DataDict,
        result_dict: WrapperModelOutput,
        tot_steps,
    ):
        result_dict.logits = (
            result_dict.logits.argmax(dim=1)
            if self.task != "detection"
            else result_dict.logits
        )
        self.tracker.log_metric("step", self.global_val_step)
        metrics = (
            self._update_metrics(
                self.val_evaluator, batch_dict, result_dict
            )
            or {}
        )
        return metrics

    def _update_train_metrics(
        self,
        result_dict: torch.tensor,
        batch_dict: torch,
        tot_steps: int,
        step: int,
    ):
        self.tracker.log_metric("step", self.global_train_step)
        metric_values = {}
        if self.train_evaluator is not None:
            self._update_metrics(self.train_evaluator, batch_dict, result_dict)

    def _count_targets(self, batch_dict: DataDict) -> int:
        labels = getattr(batch_dict, "labels", None)
        if not labels:
            return 0
        total = 0
        try:
            for label in labels:
                if label is None:
                    continue
                if isinstance(label, dict) and "boxes" in label:
                    total += len(label["boxes"])
                elif hasattr(label, "boxes"):
                    total += len(label.boxes)
                elif isinstance(label, (list, tuple)):
                    total += len(label)
        except Exception:
            return 0
        return int(total)

    def train_epoch(
        self,
        epoch: int,
    ):
        if epoch > 0:
            training_seed = int(
                self.reproducibility.get("training_seed", self.params["seed"])
            )
            set_seed(training_seed + epoch)
            logger.info(f"Setting seed to {training_seed + epoch}")
        self.tracker.log_metric("start_epoch", epoch)
        self.model.train()
        self.train_evaluator.reset()

        loss_avg = RunningAverage()
        loss_normalizer = 1
        tot_steps = 0

        # tqdm stuff
        bar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            postfix={"loss": 0},
            desc=f"Train Epoch {epoch + 1}/{self.train_params['max_epochs']}",
        )
        metric_values = {}
        probe_every = int(self.train_params.get("time_probe_interval", 0) or 0)
        probe_sync = bool(self.train_params.get("time_probe_sync", True))
        probe_enabled = probe_every > 0
        prev_iter_end = time.perf_counter()
        max_steps = self.train_params.get("max_steps_per_epoch")
        trace_batches = int(self.reproducibility.get("trace_batches", 0) or 0)
        trace_tensor_batches = int(
            self.reproducibility.get("trace_tensor_batches", 0) or 0
        )
        trace_model_steps = int(
            self.reproducibility.get("trace_model_steps", 0) or 0
        )

        for batch_idx, batch_dict in bar:
            if max_steps is not None and batch_idx >= int(max_steps):
                break
            iter_start = time.perf_counter()
            data_time = iter_start - prev_iter_end
            # if batch_idx == 1000:
            #     break
            batch_dict = DataDict(**batch_dict)
            if batch_idx < trace_batches:
                trace_values = {
                    "epoch": epoch,
                    "batch": batch_idx,
                    "global_step": self.global_train_step,
                    "sample_indices": batch_dict.sample_idx.detach().cpu().tolist(),
                    "modality_modes": list(batch_dict.modality_mode),
                    "torch_rng_sha256": tensor_digest(torch.get_rng_state()),
                }
                if "consistency_student_mode" in batch_dict:
                    trace_values["consistency_student_modes"] = list(
                        batch_dict.consistency_student_mode
                    )
                if torch.cuda.is_available():
                    trace_values["cuda_rng_sha256"] = tensor_digest(
                        torch.cuda.get_rng_state()
                    )
                if batch_idx < trace_tensor_batches:
                    trace_values["input_sha256"] = tensor_digest(
                        batch_dict.pixel_values
                    )
                self.repro_trace.write("batch", **trace_values)
            with self.accelerator.accumulate(self.model):
                if probe_enabled and (batch_idx % probe_every == 0):
                    if probe_sync and torch.cuda.is_available():
                        torch.cuda.synchronize()
                compute_start = time.perf_counter()
                self.optimizer.zero_grad()
                result_dict: WrapperModelOutput = self._forward(
                    batch_dict, epoch, batch_idx
                )
                box_guidance_components = {}
                box_guidance_loss = None
                box_guidance_scale = box_guided_alignment_epoch_scale(
                    self.box_guided_alignment_params,
                    epoch,
                )
                if self.box_guided_alignment_params["enabled"]:
                    if "box_alignment_targets" not in batch_dict:
                        raise ValueError(
                            "box-guided training batch is missing "
                            "box_alignment_targets"
                        )
                    box_guidance_loss = box_guided_alignment_loss(
                        self.model,
                        batch_dict.box_alignment_targets,
                        self.box_guided_alignment_params,
                        epoch_scale=box_guidance_scale,
                    )
                    box_guidance_components = box_guidance_loss.components
                supervised_loss = self._backward(
                    batch_idx,
                    batch_dict,
                    result_dict,
                    loss_normalizer,
                    auxiliary_loss=box_guidance_loss,
                )
                consistency_run = self._modality_consistency_forward(
                    batch_dict,
                    epoch,
                    batch_idx,
                )
                consistency_components = {}
                if consistency_run is not None:
                    (
                        consistency_input,
                        consistency_output,
                        consistency_loss,
                    ) = consistency_run
                    normalized_consistency_loss = (
                        consistency_loss.value / loss_normalizer
                    )
                    if not torch.isfinite(normalized_consistency_loss):
                        raise ValueError(
                            "Non-finite modality consistency loss at "
                            f"epoch={epoch} batch={batch_idx}: "
                            f"{normalized_consistency_loss}"
                        )
                    self.accelerator.backward(normalized_consistency_loss)
                    check_nan(
                        self.model,
                        consistency_input,
                        consistency_output,
                        normalized_consistency_loss,
                        batch_idx,
                        self.train_params,
                    )
                    consistency_components = consistency_loss.components
                    loss = (
                        supervised_loss.detach()
                        + normalized_consistency_loss.detach()
                    )
                else:
                    loss = supervised_loss.detach()
                clip_norm = self.train_params.get("gradient_clip_norm", None)
                if clip_norm is not None and clip_norm > 0:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), clip_norm)
                self.optimizer.step()
                self._scheduler_step(SchedulerStepMoment.BATCH)

            if batch_idx < trace_batches:
                loss_components = {}
                if hasattr(result_dict.loss, "components"):
                    loss_components = {
                        name: float(value.detach().cpu().item())
                        if isinstance(value, torch.Tensor) and value.numel() == 1
                        else str(value)
                        for name, value in result_dict.loss.components.items()
                    }
                loss_components.update(
                    {
                        name: float(value.detach().cpu().item())
                        if isinstance(value, torch.Tensor) and value.numel() == 1
                        else str(value)
                        for name, value in consistency_components.items()
                    }
                )
                loss_components.update(
                    {
                        name: float(value.detach().cpu().item())
                        if isinstance(value, torch.Tensor) and value.numel() == 1
                        else str(value)
                        for name, value in box_guidance_components.items()
                    }
                )
                trace_values = {
                    "epoch": epoch,
                    "batch": batch_idx,
                    "global_step": self.global_train_step,
                    "loss": float(loss.detach().cpu().item()),
                    "loss_components": loss_components,
                }
                if batch_idx < trace_model_steps:
                    trace_values.update(model_digests(self.model))
                self.repro_trace.write("optimizer_step", **trace_values)

            if probe_enabled and (batch_idx % probe_every == 0):
                if probe_sync and torch.cuda.is_available():
                    torch.cuda.synchronize()
                compute_time = time.perf_counter() - compute_start
                num_targets = self._count_targets(batch_dict)
                logger.info(
                    "Probe epoch %s step %s: data=%.3fs compute=%.3fs targets=%s",
                    epoch + 1,
                    batch_idx,
                    data_time,
                    compute_time,
                    num_targets,
                )

            loss_avg.update(loss.item())
            self.tracker.log_metric("loss", loss.item())

            if hasattr(result_dict, "loss") and hasattr(result_dict.loss, "components"):
                for k, v in result_dict.loss.components.items():
                    if "cdn" in k:
                        self.tracker.log_metric(k, v.item() if hasattr(v, "item") else v)
            for key, value in consistency_components.items():
                self.tracker.log_metric(
                    key,
                    value.item() if hasattr(value, "item") else value,
                )
            for key, value in box_guidance_components.items():
                self.tracker.log_metric(
                    key,
                    value.item() if hasattr(value, "item") else value,
                )

            self.tracker.log_metric("lr_new_modules", self.optimizer.param_groups[1]["lr"])
            self.tracker.log_metric("lr_backbone", self.optimizer.param_groups[0]["lr"])
            self.tracker.log_metric("lr_dino", self.optimizer.param_groups[2]["lr"])
            if self.box_guidance_optimizer_group_index is not None:
                self.tracker.log_metric(
                    "lr_box_guidance",
                    self.optimizer.param_groups[
                        self.box_guidance_optimizer_group_index
                    ]["lr"],
                )
            if self.reliability_gate_optimizer_group_index is not None:
                self.tracker.log_metric(
                    "lr_reliability_gate",
                    self.optimizer.param_groups[
                        self.reliability_gate_optimizer_group_index
                    ]["lr"],
                )
            if self.alignment_gate_optimizer_group_index is not None:
                self.tracker.log_metric(
                    "lr_alignment_gate",
                    self.optimizer.param_groups[
                        self.alignment_gate_optimizer_group_index
                    ]["lr"],
                )

            self._update_train_metrics(
                result_dict,
                batch_dict,
                tot_steps,
                batch_idx,
            )
            if batch_idx % 100 == 0:
                metric_values = self.train_evaluator.compute()
            bar.set_postfix(
                {
                    **metric_values,
                    "loss": loss.item(),
                    "lr": self._get_lr(),
                }
            )
            tot_steps += 1
            self.global_train_step += 1
            self.tracker.save_experiment_timed()
            prev_iter_end = time.perf_counter()

        logger.info(f"Waiting for everyone")
        self.accelerator.wait_for_everyone()
        logger.info(f"Finished Epoch {epoch+1}")
        logger.info(f"Metrics")
        metric_dict = {
            **self.train_evaluator.compute(),
            "avg_loss": loss_avg.compute(),
        }
        for k, v in metric_dict.items():
            logger.info(f"{k}: {v}")

        self.tracker.log_metrics(
            metrics=metric_dict,
            epoch=epoch,
        )

    def validate_epoch(self, epoch):
        return self.evaluate(self.val_loader, epoch=epoch, phase="val")

    def _evaluation_model_input(self, batch_dict, phase):
        if phase == "val":
            compute_loss = (self.train_params or {}).get(
                "compute_validation_loss", True
            )
        elif phase == "test":
            compute_loss = (self.params or {}).get("compute_test_loss", True)
        else:
            compute_loss = True

        if compute_loss:
            return batch_dict

        # Ground truth remains in batch_dict for mAP. RT-DETR receives no
        # labels, so in eval mode it skips Hungarian matching and auxiliary
        # losses while producing identical logits and boxes.
        model_input = DataDict(**dict(batch_dict))
        model_input.labels = None
        return model_input

    def _prepare_evaluation_memory(self):
        """Release training-only CUDA storage before a large eval batch."""
        optimizer = getattr(self, "optimizer", None)
        if optimizer is not None:
            # The final backward pass leaves gradient tensors resident after
            # optimizer.step(). They are not needed by validation and can push
            # an 8 GB GPU into WSL unified-memory paging.
            optimizer.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _release_evaluation_memory(self):
        """Drop metric state and cached eval allocations before training."""
        evaluator = getattr(self, "val_evaluator", None)
        if evaluator is not None:
            evaluator.reset()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def evaluate(self, dataloader, epoch=None, phase="val"):
        self._prepare_evaluation_memory()
        self.model.eval()
        self.val_evaluator.reset()

        avg_loss = RunningAverage()

        # Rileva se stiamo usando tiling controllando il primo batch
        use_tiling = False
        tile_predictions_buffer = defaultdict(list)
        tile_gt_buffer = {}  # Salva le GT per ogni immagine originale
        
        tot_steps = 0
        eval_cuda_cache_interval = int(
            (self.train_params or {}).get("eval_cuda_cache_interval", 0) or 0
        )
        if eval_cuda_cache_interval < 0:
            raise ValueError(
                "eval_cuda_cache_interval must be non-negative"
            )
        desc = f"{phase} Epoch {epoch+1}" if epoch is not None else f"{phase}"
        bar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            postfix={"loss": 0},
            desc=desc,
            disable=not self.accelerator.is_local_main_process,
        )
        self.tracker.create_image_sequence("predictions", columns=['epoch'])
        
        batch_dict = None
        model_input = None
        result_dict = None
        has_loss = False
        with torch.inference_mode():
            for batch_idx, batch_dict in bar:
                # if batch_idx == 100:
                #     break
                batch_dict = DataDict(**batch_dict)
                
                # Rileva se abbiamo metadati di tiling
                if batch_idx == 0:
                    use_tiling = hasattr(batch_dict, 'original_idx') and hasattr(batch_dict, 'quadrant')
                    if use_tiling:
                        logger.info("Tile aggregation mode activated for evaluation")
                        # DEBUG logging commentato - decommentare se serve debugging
                        # logger.info(f"DEBUG - First batch original_idx: {batch_dict.original_idx}")
                        # logger.info(f"DEBUG - First batch quadrant: {batch_dict.quadrant}")
                    # else:
                    #     logger.info("DEBUG - use_tiling is FALSE - no original_idx/quadrant found")
                    #     logger.info(f"DEBUG - batch_dict keys: {batch_dict.keys() if hasattr(batch_dict, 'keys') else dir(batch_dict)}")
                
                model_input = self._evaluation_model_input(batch_dict, phase)
                result_dict: WrapperModelOutput = self.model(model_input)
                
                if use_tiling:
                    # Modalità con tile aggregation
                    self._accumulate_tile_predictions(
                        batch_dict, 
                        result_dict, 
                        tile_predictions_buffer,
                        tile_gt_buffer,
                        tot_steps
                    )
                else:
                    # Modalità standard senza tiling
                    self._update_val_metrics(batch_dict, result_dict, tot_steps)
                    self.log_predictions(batch_idx, batch_dict, result_dict, epoch)
                
                loss_value = None
                result_loss = getattr(result_dict, "loss", None)
                if result_loss is not None:
                    loss_value = (
                        result_loss.value
                        if isinstance(result_loss, dict)
                        else result_loss
                    )
                    avg_loss.update(loss_value)
                    has_loss = True
                if batch_idx % 100 == 0:
                    bar.set_postfix(
                        {
                            "loss": loss_value,
                        }
                    )

                self.global_val_step += 1
                if (
                    eval_cuda_cache_interval > 0
                    and (batch_idx + 1) % eval_cuda_cache_interval == 0
                ):
                    # Drop only inactive CUDA workspaces. Predictions already
                    # retained by the evaluator remain live and unchanged.
                    batch_dict = model_input = result_dict = None
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # Se usiamo tiling, processa eventuali predizioni rimanenti nel buffer
            if use_tiling:
                logger.info(f"Processing remaining tiles from buffer: {len(tile_predictions_buffer)} images")
                self._process_remaining_tiles(
                    tile_predictions_buffer,
                    tile_gt_buffer,
                    epoch
                )

            metrics_value = self.val_evaluator.compute()
            metrics_dict = dict(metrics_value)
            if has_loss:
                metrics_dict["loss"] = avg_loss.compute()
            
            print(f"DEBUG: Metrics computed for {phase}: {metrics_dict}")

            self.tracker.log_metrics(
                metrics=metrics_dict,
                epoch=epoch,
            )
        self.tracker.add_image_sequence("predictions")
        self.accelerator.wait_for_everyone()

        for k, v in metrics_value.items():
            if epoch is not None:
                logger.info(f"{phase} epoch {epoch+1} - {k}: {v}")
            else:
                logger.info(f"{phase} - {k}: {v}")
        if has_loss:
            logger.info(f"{phase} Loss: {avg_loss.compute()}")
        # Remove the final batch/output references before returning cached
        # blocks to CUDA. Metrics are already materialized as Python values.
        batch_dict = model_input = result_dict = None
        self._release_evaluation_memory()
        return metrics_dict
    
    def _accumulate_tile_predictions(
        self,
        batch_dict: DataDict,
        result_dict: WrapperModelOutput,
        tile_predictions_buffer: dict,
        tile_gt_buffer: dict,
        tot_steps: int,
    ):
        """
        Accumula le predizioni dei tile per ogni immagine originale.
        Quando abbiamo tutti i 4 tile, aggrega e aggiorna le metriche.
        """
        batch_size = len(batch_dict.original_idx)
        
        # DEBUG logging commentato - decommentare se serve debugging dettagliato
        # if not hasattr(self, '_debug_accumulate_called'):
        #     self._debug_accumulate_called = True
        #     logger.info(f"DEBUG _accumulate: batch_size={batch_size}")
        #     logger.info(f"DEBUG _accumulate: has labels_full={hasattr(batch_dict, 'labels_full')}")
        #     if hasattr(batch_dict, 'labels_full'):
        #         logger.info(f"DEBUG _accumulate: labels_full type={type(batch_dict.labels_full)}")
        #         logger.info(f"DEBUG _accumulate: labels_full[0] type={type(batch_dict.labels_full[0])}")
        #         logger.info(f"DEBUG _accumulate: labels_full[0] keys={batch_dict.labels_full[0].keys() if isinstance(batch_dict.labels_full[0], dict) else 'not a dict'}")
        #         if isinstance(batch_dict.labels_full[0], dict) and 'boxes' in batch_dict.labels_full[0]:
        #             logger.info(f"DEBUG _accumulate: labels_full[0]['boxes'] shape={batch_dict.labels_full[0]['boxes'].shape}")
        
        for i in range(batch_size):
            original_idx = batch_dict.original_idx[i].item()
            quadrant = batch_dict.quadrant[i].item()
            
            # Estrai predizioni per questo tile
            tile_pred = {
                'boxes': result_dict.predictions[i]['boxes'],
                'scores': result_dict.predictions[i]['scores'],
                'labels': result_dict.predictions[i]['labels'],
                'quadrant': quadrant,
            }
            
            tile_predictions_buffer[original_idx].append(tile_pred)
            
            # Salva le GT COMPLETE dell'immagine originale (non filtrate per tile)
            if original_idx not in tile_gt_buffer:
                # Usa labels_full se disponibile (GT complete 640x640), altrimenti fallback
                if hasattr(batch_dict, 'labels_full'):
                    tile_gt_buffer[original_idx] = batch_dict.labels_full[i]
                else:
                    # Fallback: usa le GT del tile (comportamento vecchio, potenzialmente errato)
                    logger.warning(f"labels_full not found for image {original_idx}, using tile labels")
                    tile_gt_buffer[original_idx] = batch_dict.labels[i]
            
            # Quando abbiamo tutti i 4 tile, aggrega
            if len(tile_predictions_buffer[original_idx]) == 4:
                self._aggregate_and_update_metrics(
                    original_idx,
                    tile_predictions_buffer[original_idx],
                    tile_gt_buffer[original_idx],
                )
                # Rimuovi dal buffer
                del tile_predictions_buffer[original_idx]
                del tile_gt_buffer[original_idx]
    
    def _aggregate_and_update_metrics(
        self,
        original_idx: int,
        tile_predictions: list,
        ground_truth: dict,
    ):
        """
        Aggrega le predizioni dei 4 tile e aggiorna le metriche.
        """
        # DEBUG logging commentato - decommentare se serve debugging dettagliato
        # if original_idx < 3:
        #     logger.info(f"DEBUG - Image {original_idx}")
        #     # Stampa TUTTI i tile e i loro quadrant
        #     for tile_idx, tile_pred in enumerate(tile_predictions):
        #         q = tile_pred.get('quadrant', 'MISSING')
        #         boxes = tile_pred['boxes']
        #         logger.info(f"DEBUG - tile_idx={tile_idx}, quadrant={q}, num_boxes={len(boxes)}, first_box={boxes[0] if len(boxes) > 0 else 'empty'}")
        #     gt_boxes = ground_truth.get('boxes', None)
        #     logger.info(f"DEBUG - GT boxes: {gt_boxes[:2] if gt_boxes is not None and len(gt_boxes) > 0 else 'empty or None'}")
        
        # Aggrega le predizioni
        # Nota: IoU 0.5 è conservativo (alto recall) - preferibile per SAR dove perdere una persona è critico
        aggregated = aggregate_tile_predictions(tile_predictions, iou_threshold=0.5)
        
        # DEBUG logging commentato
        # if original_idx < 3:
        #     logger.info(f"DEBUG - Aggregated boxes sample (first 2): {aggregated['boxes'][:2] if len(aggregated['boxes']) > 0 else 'empty'}")
        
        # Crea strutture per l'update delle metriche
        fake_batch = DataDict(labels=[ground_truth])
        fake_result = WrapperModelOutput(predictions=[aggregated])
        
        # Aggiorna le metriche con le predizioni aggregate
        self._update_metrics(self.val_evaluator, fake_batch, fake_result)
    
    def _process_remaining_tiles(
        self,
        tile_predictions_buffer: dict,
        tile_gt_buffer: dict,
        epoch,
    ):
        """
        Processa eventuali tile rimanenti nel buffer (immagini incomplete).
        Questo può succedere se il dataset ha un numero di tile non multiplo di 4.
        """
        for original_idx, tile_preds in tile_predictions_buffer.items():
            if len(tile_preds) > 0:
                logger.warning(
                    f"Image {original_idx} has only {len(tile_preds)}/4 tiles. "
                    f"Aggregating available tiles."
                )
                # Aggrega comunque con i tile disponibili
                self._aggregate_and_update_metrics(
                    original_idx,
                    tile_preds,
                    tile_gt_buffer[original_idx],
                )

    def log_predictions(self, batch_idx, batch_dict, result_dict, epoch, sequence_name="predictions"):
        if self.task == "detection":
            self.tracker.log_object_detection(
                batch_idx,
                batch_dict,
                result_dict,
                self.val_loader.dataset.id2class,
                self.denormalize,
                epoch,
                sequence_name=sequence_name,
            )

    def restore_model(self, checkpoint_type="best"):
        if checkpoint_type in (None, "current"):
            logger.info("Testing the model currently in memory")
            return
        if checkpoint_type not in {"best", "latest"}:
            raise ValueError(
                "test_checkpoint must be 'best', 'latest', or 'current', got "
                f"{checkpoint_type!r}"
            )
        try:
            filename = os.path.join(
                self.tracker.local_dir, checkpoint_type, "model.safetensors"
            )
            with safe_open(filename, framework="pt") as f:
                weights = {k: f.get_tensor(k) for k in f.keys()}
            self.model.load_state_dict(weights, strict=False)
            logger.info(f"Checkpoint '{checkpoint_type}' restored from {filename}")

        except FileNotFoundError:
            raise FileNotFoundError(
                f"No '{checkpoint_type}' checkpoint found at {filename}"
            )

    def restore_best_model(self):
        """Backward-compatible alias used by older callers."""
        return self.restore_model("best")

    def test(self):
        self.test_loader = self.accelerator.prepare(self.test_loader)
        self.restore_model(self.params.get("test_checkpoint", "best"))
        with self.tracker.test():
            self.evaluate(self.test_loader, phase="test")

    def end(self):
        if self._ended:
            return
        self._ended = True
        logger.info("Ending run")

        try:
            if self.tracker is not None:
                # Reassert final selection metadata immediately before tracker
                # shutdown. This also covers a best checkpoint selected before
                # the final epoch.
                if self.best_epoch is not None:
                    self.tracker.add_summary(
                        {
                            "best_epoch": self.best_epoch + 1,
                            f"best_{self.watch_metric}": self.best_metric,
                        }
                    )
                self.tracker.end()
        finally:
            # Break the self-referencing lambda before collecting the run.
            self.compute_val_metrics = None

            if self.accelerator is not None:
                (
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    self.train_loader,
                    self.val_loader,
                    self.test_loader,
                ) = self.accelerator.free_memory(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    self.train_loader,
                    self.val_loader,
                    self.test_loader,
                )
            else:
                self.model = None
                self.optimizer = None
                self.scheduler = None
                self.train_loader = None
                self.val_loader = None
                self.test_loader = None

            self.criterion = None
            self.train_evaluator = None
            self.val_evaluator = None
            self.denormalize = None
            self.validation_json = None
            self.tracker = None
            self.accelerator = None

        logger.info("Run ended")


def yolo_train(parameters):
    install_wandb_empty_curve_guard()
    if isinstance(parameters, str):
        args = load_yaml(parameters)
    else:
        args = parameters

    # args['model'] = None
    # model = args.pop("model")
    # model = YOLOv10WiSARD.from_pretrained(**model)
    args.pop("experiment")
    args = {k: (v if v != {} else None) for k, v in args.items()}
    model = args.pop("model")
    args['model'] = None
    
    trainer = WisardTrainer(overrides=args)
    model['params']['nc'] = 1 if args.get("single_cls") else trainer.data["nc"]
    
    trainer.model = build_model(model)
    trainer.train()
    print()


class YoloRun:
    def __init__(self) -> None:
        self.parameters = None

    def init(self, parameters) -> None:
        self.parameters = parameters

    def launch(self) -> None:
        yolo_train(self.parameters)
