import json
from functools import wraps

import numpy as np
import torch

from copy import copy
from ultralytics.models.yolov10.train import (
    YOLOv10DetectionTrainer,
    YOLOv10DetectionValidator,
)
from ultralytics.utils import DEFAULT_CFG, LOGGER
from ultralytics.cfg import cfg2dict, IterableSimpleNamespace
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import LOGGER, TQDM, callbacks, colorstr, emojis
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.ops import Profile
from ultralytics.utils.torch_utils import de_parallel, select_device, smart_inference_mode
from sarfusion.data.wisard import WiSARDYOLODataset
from sarfusion.utils.general import colorstr
from ultralytics.utils.torch_utils import de_parallel, strip_optimizer
from sarfusion.utils.plots import plot_images

def build_yolo_dataset(cfg, img_path, batch, data, mode="train", rect=False, stride=32):
    """Build YOLO Dataset."""
    return WiSARDYOLODataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",  # augmentation
        hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
        rect=cfg.rect or rect,  # rectangular batches
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=int(stride),
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == "train" else 1.0,
        augment_vis_ir=cfg.augment_vis_ir,
        modal_dropout=cfg.modal_dropout if mode == "train" else False,
        modal_dropout_probs=cfg.modal_dropout_probs,
        modal_dropout_strategy=getattr(
            cfg, "modal_dropout_strategy", "feature"
        ),
    )


WISARD_DEFAULT_CFG = IterableSimpleNamespace(
    **{
        **cfg2dict(DEFAULT_CFG),
        "augment_vis_ir": False,
        "modal_dropout": False,
        "modal_dropout_probs": [0.2, 0.2, 0.6],
        "modal_dropout_strategy": "feature",
        "test_checkpoint": "best",
    }
)


def _guard_wandb_plot_curve(plot_curve):
    """Skip malformed metric curves instead of failing a completed training.

    Ultralytics 8.1.34 leaves the precision-recall values empty when plots are
    disabled. Its W&B ``on_train_end`` callback nevertheless tries to
    interpolate that curve, raising a ``ValueError`` after the final checkpoint
    and test metrics have already been saved.
    """
    if getattr(plot_curve, "__dict__", {}).get(
        "_sarfusion_empty_curve_guard", False
    ):
        return plot_curve

    @wraps(plot_curve)
    def guarded(x, y, *args, **kwargs):
        x_array = np.asarray(x)
        y_array = np.asarray(y)
        valid = (
            x_array.ndim == 1
            and x_array.size >= 2
            and y_array.ndim >= 2
            and y_array.size > 0
            and y_array.shape[-1] == x_array.size
        )
        if not valid:
            curve_name = kwargs.get("title") or kwargs.get("id") or "unnamed"
            LOGGER.warning(
                "Skipping empty or malformed W&B metric curve '%s'; "
                "training checkpoints and scalar metrics are unaffected.",
                curve_name,
            )
            return None
        return plot_curve(x, y, *args, **kwargs)

    guarded._sarfusion_empty_curve_guard = True
    return guarded


def install_wandb_empty_curve_guard():
    """Install the local compatibility guard in Ultralytics' W&B callback."""
    try:
        from ultralytics.utils.callbacks import wb as wb_callback
    except ImportError:
        return False

    plot_curve = getattr(wb_callback, "_plot_curve", None)
    if plot_curve is None:
        return False
    if getattr(plot_curve, "__dict__", {}).get(
        "_sarfusion_empty_curve_guard", False
    ):
        return False

    wb_callback._plot_curve = _guard_wandb_plot_curve(plot_curve)
    return True


def build_wisard_validator_args(trainer_args):
    """Remove trainer-only WiSARD options before Ultralytics validation."""
    args = cfg2dict(copy(trainer_args))
    for key in (
        "augment_vis_ir",
        "modal_dropout",
        "modal_dropout_probs",
        "modal_dropout_strategy",
        "test_checkpoint",
    ):
        args.pop(key, None)
    return IterableSimpleNamespace(**args)


class WisardValidator(YOLOv10DetectionValidator):

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None, mode="val"):
        """Supports validation of a pre-trained model if passed or a model being trained if trainer is passed (trainer
        gets priority).
        """
        self.training = trainer is not None
        augment = self.args.augment and (not self.training)
        if self.training:
            self.device = trainer.device
            self.data = trainer.data
            # self.args.half = self.device.type != "cpu"  # force FP16 val during training
            model = trainer.ema.ema or trainer.model
            model = model.half() if self.args.half else model.float()
            # self.model = model
            self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
            self.args.plots &= trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)
            model.eval()
        else:
            callbacks.add_integration_callbacks(self)
            model = AutoBackend(
                weights=model or self.args.model,
                device=select_device(self.args.device, self.args.batch),
                dnn=self.args.dnn,
                data=self.args.data,
                fp16=self.args.half,
            )
            # self.model = model
            self.device = model.device  # update device
            self.args.half = model.fp16  # update half
            stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
            imgsz = check_imgsz(self.args.imgsz, stride=stride)
            if engine:
                self.args.batch = model.batch_size
            elif not pt and not jit:
                self.args.batch = 1  # export.py models default to batch-size 1
                LOGGER.info(f"Forcing batch=1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models")

            if str(self.args.data).split(".")[-1] in ("yaml", "yml"):
                self.data = check_det_dataset(self.args.data)
            elif self.args.task == "classify":
                self.data = check_cls_dataset(self.args.data, split=self.args.split)
            else:
                raise FileNotFoundError(emojis(f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"))

            if self.device.type in ("cpu", "mps"):
                self.args.workers = 0  # faster CPU val as time dominated by inference, not dataloading
            if not pt:
                self.args.rect = False
            self.stride = model.stride  # used in get_dataloader() for padding
            self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)

            model.eval()
            model.warmup(imgsz=(1 if pt else self.args.batch, 3, imgsz, imgsz))  # warmup

        self.run_callbacks("on_val_start")
        dt = (
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
        )
        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        self.init_metrics(de_parallel(model))
        if self.args.single_cls:
            self.nc = 1
        self.jdict = []  # empty before each val
        for batch_i, batch in enumerate(bar):
            self.run_callbacks("on_val_batch_start")
            self.batch_i = batch_i
            # Preprocess
            with dt[0]:
                batch = self.preprocess(batch)

            # Inference
            with dt[1]:
                preds = model(batch["img"], augment=augment)

            # Loss
            with dt[2]:
                if self.training:
                    self.loss += model.loss(batch, preds)[1]

            # Postprocess
            with dt[3]:
                preds = self.postprocess(preds)

            self.update_metrics(preds, batch)
            if self.args.plots and batch_i < 10:
                self.plot_val_samples(batch, batch_i, mode)
                self.plot_predictions(batch, preds, batch_i)

            self.run_callbacks("on_val_batch_end")
        stats = self.get_stats()
        self.check_stats(stats)
        self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt)))
        self.finalize_metrics()
        if not (self.args.save_json and self.is_coco and len(self.jdict)):
            self.print_results()
        self.run_callbacks("on_val_end")
        if self.training:
            model.float()
            if self.args.save_json and self.jdict:
                with open(str(self.save_dir / "predictions.json"), "w") as f:
                    LOGGER.info(f"Saving {f.name}...")
                    json.dump(self.jdict, f)  # flatten and save
                stats = self.eval_json(stats)  # update stats
                stats['fitness'] = stats['metrics/mAP50-95(B)']
            results = {**stats, **trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix="val")}
            return {k: round(float(v), 5) for k, v in results.items()}  # return results as 5 decimal place floats
        else:
            LOGGER.info(
                "Speed: %.1fms preprocess, %.1fms inference, %.1fms loss, %.1fms postprocess per image"
                % tuple(self.speed.values())
            )
            if self.args.save_json and self.jdict:
                with open(str(self.save_dir / "predictions.json"), "w") as f:
                    LOGGER.info(f"Saving {f.name}...")
                    json.dump(self.jdict, f)  # flatten and save
                stats = self.eval_json(stats)  # update stats
            if self.args.plots or self.args.save_json:
                LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
            return stats
        
    def plot_val_samples(self, batch, ni, mode="val"):
        """Plot validation image samples."""
        plot_images(
            batch["img"],
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"{mode}_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(self, batch, preds, ni):
        """Plots predicted bounding boxes using the 4-channel-aware plotter."""
        from sarfusion.utils.plots import output_to_target
        plot_images(
            batch["img"],
            *output_to_target(preds, max_det=self.args.max_det),
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

class WisardTrainer(YOLOv10DetectionTrainer):
    def __init__(self, cfg=WISARD_DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)

    def plot_training_samples(self, batch, ni):
        """Plots training samples with their annotations."""
        im_file = [
            elem[0] if isinstance(elem, list) else elem for elem in batch["im_file"]
        ]
        plot_images(
            images=batch["img"],
            batch_idx=batch["batch_idx"],
            cls=batch["cls"].squeeze(-1),
            bboxes=batch["bboxes"],
            paths=im_file,
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(
            self.args,
            img_path,
            batch,
            self.data,
            mode=mode,
            rect=mode == "val",
            stride=gs,
        )

    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        self.loss_names = "box_om", "cls_om", "dfl_om", "box_oo", "cls_oo", "dfl_oo",
        args = build_wisard_validator_args(self.args)
        return WisardValidator(
            self.test_loader, save_dir=self.save_dir, args=args, _callbacks=self.callbacks
        )
        
    def final_eval(self):
        """Evaluate the predeclared final checkpoint on the WiSARD test split."""
        checkpoint_name = getattr(self.args, "test_checkpoint", "best")
        if checkpoint_name not in {"best", "last"}:
            raise ValueError(
                "test_checkpoint must be 'best' or 'last', got "
                f"{checkpoint_name!r}"
            )

        batch_size = self.batch_size if self.args.task == "obb" else self.batch_size * 2
        test_loader = self.get_dataloader(self.data['test'], batch_size=batch_size, mode="val", rank=-1)
        for f in self.last, self.best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers

        checkpoint = self.last if checkpoint_name == "last" else self.best
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Requested final YOLO checkpoint does not exist: {checkpoint}"
            )

        LOGGER.info(f"\nValidating predeclared {checkpoint_name}.pt: {checkpoint}...")
        self.validator.args.plots = self.args.plots
        self.validator.dataloader = test_loader
        self.metrics = self.validator(model=checkpoint, mode="test")
        self.metrics.pop("fitness", None)
        self.metrics = {
            key.replace("metrics/", "test/"): value
            for key, value in self.metrics.items()
        }
        self.run_callbacks("on_fit_epoch_end")
