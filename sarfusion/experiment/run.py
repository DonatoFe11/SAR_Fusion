import os
import sys
import shutil
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
from sarfusion.experiment.yolo import WisardTrainer
from sarfusion.models.yolov10 import YOLOv10WiSARD
from sarfusion.utils.structures import LossOutput, WrapperModelOutput
from sarfusion.utils.logger import get_logger
from sarfusion.data import get_dataloaders
from sarfusion.utils.structures import DataDict
from sarfusion.experiment.utils import WrapperModule
from sarfusion.models.loss import build_loss
from sarfusion.models import build_model
from sarfusion.utils.metrics import DetectionEvaluator, Evaluator, build_evaluator
from sarfusion.utils.utils import (
    RunningAverage,
    load_yaml,
    make_showable,
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
        self.scheduler = None
        self.criterion = None
        self.best_metric = None
        self.scheduler_step_moment = None
        self.watch_metric = None
        self.train_evaluator: Evaluator = None
        self.val_evaluator: Evaluator = None
        if "." not in sys.path:
            sys.path.extend(".")
        self.global_train_step = 0
        self.global_val_step = 0
        self.validation_json = None

    def parse_params(self, params: dict):
        self.params = deepcopy(params)

        (
            self.train_params,
            self.dataset_params,
            self.dataloader_params,
            self.model_params,
        ) = parse_params(self.params)

    def init(self, params: dict):
        set_seed(params["seed"])
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
        )
        logger.info("Initiliazing tracker...")
        self.tracker = get_experiment_tracker(self.accelerator, self.params)
        self.url = self.tracker.url
        self.name = self.tracker.name
        (self.train_loader, self.val_loader, self.test_loader), self.denormalize = (
            get_dataloaders(
                self.dataset_params,
                self.dataloader_params,
            )
        )
        model_name = self.model_params.get("name")
        logger.info(f"Creating model {model_name}")
        self.model = build_model(params=self.model_params)
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
        self.watch_metric = self.train_params["watch_metric"]
        self.greater_is_better = self.train_params.get("greater_is_better", True)
        logger.info("Creating optimizer")
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            params = self.model.module.get_learnable_params(self.train_params)
        else:
            params = self.model.get_learnable_params(self.train_params)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.train_params["initial_lr"],
        )

        scheduler_params = self.train_params.get("scheduler", None)
        if scheduler_params:
            self.scheduler, self.scheduler_step_moment = get_scheduler(
                scheduler_params=scheduler_params,
                optimizer=self.optimizer,
                num_training_steps=self.train_params["max_epochs"]
                * len(self.train_loader),
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
                for epoch in range(self.train_params["max_epochs"]):
                    logger.info(
                        "Epoch: {}/{}".format(epoch, self.train_params["max_epochs"])
                    )
                    self.train_epoch(epoch)

                    metrics = None
                    if (
                        self.val_loader
                        and epoch % self.train_params.get("val_frequency", 1) == 0
                    ):
                        with self.tracker.validate():
                            logger.info(f"Running Model Validation")
                            metrics = self.validate_epoch(epoch)
                            self._scheduler_step(SchedulerStepMoment.EPOCH, metrics)
                    self.save_training_state(epoch, metrics)
        else:
            logger.info("No training params, no training")

        if self.test_loader:
            self.test()
        self.end()

    def _metric_is_better(self, metric):
        if self.best_metric is None:
            return True
        if self.greater_is_better:
            return metric > self.best_metric
        return metric < self.best_metric

    def save_training_state(self, epoch, metrics=None):
        if metrics:
            if self._metric_is_better(metrics[self.watch_metric]):
                logger.info(
                    f"Saving best model with metric {metrics[self.watch_metric]} as given that metric is greater than {self.best_metric}"
                )
                self.best_metric = metrics[self.watch_metric]
                self.tracker.log_training_state(epoch=epoch, subfolder="best")
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

    def _backward(
        self, batch_idx, input_dict, outputs: WrapperModelOutput, loss_normalizer
    ):
        loss_value = outputs.loss.value if isinstance(outputs.loss, dict) else outputs.loss
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

    def train_epoch(
        self,
        epoch: int,
    ):
        if epoch > 0:
            set_seed(self.params["seed"] + epoch)
            logger.info(f"Setting seed to {self.params['seed'] + epoch}")
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
            desc=f"Train Epoch {epoch}/{self.train_params['max_epochs']-1}",
        )
        metric_values = {}

        for batch_idx, batch_dict in bar:
            # if batch_idx == 1000:
            #     break
            batch_dict = DataDict(**batch_dict)
            self.optimizer.zero_grad()
            result_dict: WrapperModelOutput = self._forward(
                batch_dict, epoch, batch_idx
            )
            loss = self._backward(batch_idx, batch_dict, result_dict, loss_normalizer)
            self.optimizer.step()
            self._scheduler_step(SchedulerStepMoment.BATCH)

            loss_avg.update(loss.item())
            self.tracker.log_metric("loss", loss.item())

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

        logger.info(f"Waiting for everyone")
        self.accelerator.wait_for_everyone()
        logger.info(f"Finished Epoch {epoch}")
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

    def evaluate(self, dataloader, epoch=None, phase="val"):
        self.model.eval()
        self.val_evaluator.reset()

        avg_loss = RunningAverage()

        # Rileva se stiamo usando tiling controllando il primo batch
        use_tiling = False
        tile_predictions_buffer = defaultdict(list)
        tile_gt_buffer = {}  # Salva le GT per ogni immagine originale
        
        tot_steps = 0
        desc = f"{phase} Epoch {epoch}" if epoch is not None else f"{phase}"
        bar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            postfix={"loss": 0},
            desc=desc,
            disable=not self.accelerator.is_local_main_process,
        )
        self.tracker.create_image_sequence("predictions", columns=['epoch'])
        
        with torch.no_grad():
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
                
                result_dict: WrapperModelOutput = self.model(batch_dict)
                
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
                
                loss = result_dict.loss if result_dict.loss is not None else 0
                loss_value = loss.value if isinstance(result_dict.loss, dict) else result_dict.loss
                avg_loss.update(loss_value)
                if batch_idx % 100 == 0:
                    metrics_value = self.val_evaluator.compute()
                    bar.set_postfix(
                        {
                            "loss": loss_value,
                            **make_showable(metrics_value),
                        }
                    )

                self.global_val_step += 1
            
            # Se usiamo tiling, processa eventuali predizioni rimanenti nel buffer
            if use_tiling:
                logger.info(f"Processing remaining tiles from buffer: {len(tile_predictions_buffer)} images")
                self._process_remaining_tiles(
                    tile_predictions_buffer,
                    tile_gt_buffer,
                    epoch
                )

            metrics_dict = {
                **self.val_evaluator.compute(),
                "loss": avg_loss.compute(),
            }
            
            print(f"DEBUG: Metrics computed for {phase}: {metrics_dict}")

            self.tracker.log_metrics(
                metrics=metrics_dict,
                epoch=epoch,
            )
        self.tracker.add_image_sequence("predictions")
        self.accelerator.wait_for_everyone()

        metrics_value = self.val_evaluator.compute()
        for k, v in metrics_value.items():
            if epoch is not None:
                logger.info(f"{phase} epoch {epoch} - {k}: {v}")
            else:
                logger.info(f"{phase} - {k}: {v}")
        logger.info(f"{phase} Loss: {avg_loss.compute()}")
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

    def restore_best_model(self):
        try:
            filename = self.tracker.local_dir + "/best/model.safetensors"
            with safe_open(filename, framework="pt") as f:
                weights = {k: f.get_tensor(k) for k in f.keys()}
            self.model.load_state_dict(weights, strict=False)
            logger.info(f"✅ Best model restored from {filename}")

        except FileNotFoundError:
            logger.warning(f"No best model found in {filename}, ensure you are using a pretrained model")

    def test(self):
        self.test_loader = self.accelerator.prepare(self.test_loader)
        # Restore best model
        self.restore_best_model()
        with self.tracker.test():
            self.evaluate(self.test_loader, phase="test")

    def end(self):
        logger.info("Ending run")
        self.tracker.end()
        logger.info("Run ended")


def yolo_train(parameters):
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
    model['params']['nc'] = trainer.data["nc"]
    
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
