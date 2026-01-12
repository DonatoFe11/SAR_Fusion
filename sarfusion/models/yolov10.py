from copy import deepcopy
import os
import torch
import tempfile

from ultralytics import YOLOv10
from ultralytics.utils.loss import v10DetectLoss
from ultralytics.nn.tasks import guess_model_task, BaseModel

from ultralytics.nn.modules import (
    OBB,
    Detect,
    Pose,
    Segment,
    v10Detect
)
from ultralytics.utils.torch_utils import initialize_weights, scale_img
from ultralytics.utils import LOGGER, RANK, DEFAULT_CFG_DICT

from huggingface_hub import PyTorchModelHubMixin
from ultralytics.models.yolov10.card import card_template_text

from sarfusion.experiment.yolo import WisardTrainer
from sarfusion.models.utils import fusion_pretraining_load, yaml_model_load
# from sarfusion.utils.lossv10 import v10DetectLoss
from sarfusion.utils.general import yaml_save
from sarfusion.utils.structures import ModelOutput
from sarfusion.models.parse import parse_model


class YOLOv10DetectionModel(BaseModel):
    def init_criterion(self):
        return v10DetectLoss(self)
    
    def __init__(self, cfg="yolov8n.yaml", ch=3, nc=None, verbose=True):  # model, input channels, number of classes
        """Initialize the YOLOv8 detection model with the given config and parameters."""
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict

        # Define model
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override YAML value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # default names dict
        self.inplace = self.yaml.get("inplace", True)

        # Build strides
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):  # includes all Detect subclasses like Segment, Pose, OBB, WorldDetect
            s = 256  # 2x min stride
            m.inplace = self.inplace
            forward = lambda x: self.forward(x)[0] if isinstance(m, (Segment, Pose, OBB)) else self.forward(x)
            if isinstance(m, v10Detect):
                # forward = lambda x: self.forward(x).features["one2many"]
                forward = lambda x: self.forward(x)["one2many"]
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
            self.stride = m.stride
            m.bias_init()  # only run once
        else:
            self.stride = torch.Tensor([32])  # default stride for i.e. RTDETR

        # Init weights, biases
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info("")
            
    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference and train outputs."""
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = super().predict(xi)  # forward
            if isinstance(yi, dict):
                yi = yi["one2one"]  # yolov10 outputs
            if isinstance(yi, (list, tuple)):
                yi = yi[0]
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, -1), None  # augmented inference, train

    @staticmethod
    def _descale_pred(p, flips, scale, img_size, dim=1):
        """De-scale predictions following augmented inference (inverse operation)."""
        p[:, :4] /= scale  # de-scale
        x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
        if flips == 2:
            y = img_size[0] - y  # de-flip ud
        elif flips == 3:
            x = img_size[1] - x  # de-flip lr
        return torch.cat((x, y, wh, cls), dim)

    def _clip_augmented(self, y):
        """Clip YOLO augmented inference tails."""
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4**x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[-1] // g) * sum(4**x for x in range(e))  # indices
        y[0] = y[0][..., :-i]  # large
        i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][..., i:]  # small
        return y

    # def forward(self, images):
    #     if self.training:
    #         features = super().forward(x=images)
    #         return ModelOutput(features=features)
    #     else:
    #         result = super().forward(x=images)
    #         if isinstance(result, dict):
    #             features = result["one2one"]
    #         else:
    #             features = result
    #         if isinstance(features, (list, tuple)):
    #             features = features[0]
    #         preds = features.transpose(-1, -2)
    #         boxes, scores, labels = ops.v10postprocess(preds, self.args["max_det"], len(self.names))
    #         bboxes = ops.xywh2xyxy(boxes)
    #         preds = torch.cat([bboxes, scores.unsqueeze(-1), labels.unsqueeze(-1)], dim=-1)
    #         return ModelOutput(logits=preds, features=result)
    
    def get_learnable_params(self, train_params):
        return [{"params": self.model.parameters()}]


class YOLOv10WiSARD(YOLOv10):
    def __init__(self, model="yolov10n.pt", task=None, verbose=False, 
                 names=None, imgsz=None):
        if isinstance(model, dict):
            # Write dict to a temp YAML file
            with tempfile.NamedTemporaryFile(suffix=".yaml") as tmp:
                yaml_save(tmp.name, model)
                super().__init__(model=tmp.name, task=task, verbose=verbose)
        else:
            super().__init__(model=model, task=task, verbose=verbose)
        if names is not None:
            setattr(self.model, 'names', names)
        if imgsz is not None:
            setattr(self.model, 'imgsz', imgsz)
        else:
            if self.ckpt is not None:
                setattr(self.model, 'imgsz', self.ckpt["train_args"]["imgsz"])
    
    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        task_map = super().task_map
        task_map['detect']['model'] = YOLOv10DetectionModel
        task_map['detect']['trainer'] = WisardTrainer
        # task_map['detect']['validator'] = WisardValidator
        return task_map
    
    def _new(self, cfg: str, task=None, model=None, verbose=False) -> None:
        """
        Initializes a new model and infers the task type from the model definitions.

        Args:
            cfg (str): model configuration file
            task (str | None): model task
            model (BaseModel): Customized model.
            verbose (bool): display model info on load
        """
        cfg_dict = yaml_model_load(cfg)
        self.cfg = cfg
        self.task = task or guess_model_task(cfg_dict)
        self.model = (model or self._smart_load("model"))(cfg_dict, verbose=verbose and RANK == -1)  # build model
        self.overrides["model"] = self.cfg
        self.overrides["task"] = self.task

        # Below added to allow export from YAMLs
        self.model.args = {**DEFAULT_CFG_DICT, **self.overrides}  # combine default and model args (prefer model args)
        self.model.task = self.task
        self.model_name = cfg

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, fusion_pretraining=False, cfg=None, **kwargs):
        if fusion_pretraining:
            pretrained_model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
            model = YOLOv10WiSARD(model=cfg, task="detect")
            fusion_pretraining_load(model, pretrained_model.state_dict())
            return model
        if cfg:
            kwargs['model'] = cfg
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)
    
    def push_to_hub(self, repo_name, **kwargs):
        config = kwargs.get('config', {})
        config['names'] = self.names
        config['model'] = self.model.yaml
        config['task'] = self.task
        config['imgsz'] = self.model.imgsz
        kwargs['config'] = config
        PyTorchModelHubMixin.push_to_hub(self, repo_name, **kwargs)
