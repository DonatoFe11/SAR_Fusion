import tempfile

from copy import deepcopy
from enum import StrEnum
from transformers import AutoModel, ViTForImageClassification

from sarfusion.experiment.utils import WrapperModule
from sarfusion.models.experimental import attempt_load
from sarfusion.models.utils import torch_dict_load
from sarfusion.models.utils import nc_safe_load
from sarfusion.models.yolov10 import YOLOv10WiSARD
from sarfusion.models.detr import DeformableDetr, Detr, FusionDetr, RTDetr
from sarfusion.utils.general import yaml_save
from sarfusion.utils.utils import load_yaml


class AdditionalParams(StrEnum):
    PRETRAINED_PATH = "pretrained_path"


def build_model(params):
    """
    Build a model from a yaml file or a dictionary

    Args:
        params (dict or str): Dictionary or path to yaml file containing model parameters
        Additional parameters:
            pretrained_path: The path of the pretrained model

    """
    if isinstance(params, str):
        params = load_yaml(params)
    params = deepcopy(params)
    name = params["name"]
    params = params["params"]
    pretrained_path = params.pop(AdditionalParams.PRETRAINED_PATH, None)

    if name in MODEL_REGISTRY:
        model = MODEL_REGISTRY[name](**params)
    else:
        model = AutoModel.from_pretrained(name)

    if pretrained_path:
        try:
            weights = torch_dict_load(pretrained_path)
            model.load_state_dict(weights)
            print(f"Loaded model from {pretrained_path}")
        except RuntimeError as e:
            print(f"Error loading model from {pretrained_path}: trying to remove 'model.'")
            if list(weights.keys())[0].startswith("model."):
                weights = {k[6:]: v for k, v in weights.items()}
                model.load_state_dict(weights)
    return model


def backbone_learnable_params(self, train_params: dict):
    freeze_backbone = train_params.get("freeze_backbone", False)
    if freeze_backbone:
        for param in self.vit.parameters():
            param.requires_grad = False
        return [
            {"params": [x[1] for x in self.named_parameters() if x[1].requires_grad]}
        ]
    return [{"params": list(self.parameters())}]


def build_vit_classifier(**params):
    params = deepcopy(params)
    labels = params.pop("labels")
    path = params.pop("path")
    num_labels = len(labels)
    id2label = {str(i): c for i, c in enumerate(labels)}
    label2id = {c: str(i) for i, c in enumerate(labels)}
    vit = ViTForImageClassification.from_pretrained(
        path,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
        **params,
    )
    vit.get_learnable_params = backbone_learnable_params.__get__(
        vit, ViTForImageClassification
    )
    return vit


def build_detr(threshold=0.9, id2label=None, path="facebook/detr-resnet-50"):
    return Detr(threshold=threshold, id2label=id2label, pretrained_model_name=path)


def build_rtdetr(threshold=0.9, id2label=None):
    return RTDetr(threshold=threshold, id2label=id2label)


def build_deformable_detr(threshold=0.9, id2label=None):
    return DeformableDetr(threshold=threshold, id2label=id2label)


def build_fusion_detr(threshold=0.9, id2label=None):
    return FusionDetr(threshold=threshold, id2label=id2label)


def build_yolo_v9(cfg, nc=None, checkpoint=None, iou_t=0.2, conf_t=0.001, head={}):
    from sarfusion.models.yolo import Model as YOLOv9

    # if checkpoint:
    #     return attempt_load(checkpoint, head=head, iou_thres=iou_t, conf_thres=conf_t)
    model = YOLOv9(cfg, nc=nc, iou_t=iou_t, conf_t=conf_t)
    nc = model.model[-1].nc
    if checkpoint:
        weights = torch_dict_load(checkpoint)["model"].state_dict()
        nc_safe_load(model.model, weights, nc)

    return model


def build_yolo_v10(
    pretrained_model_name_or_path=None, cfg=None, fusion_pretraining=False, nc=None
):
    if pretrained_model_name_or_path:
        pretrained_model = YOLOv10WiSARD.from_pretrained(
            pretrained_model_name_or_path,
            fusion_pretraining=fusion_pretraining,
            cfg=cfg,
        ).model
        cfg = cfg or pretrained_model.yaml
        # Temporary file to load the model
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml")
        if isinstance(cfg, dict):
            cfg['nc'] = nc
            yaml_save(tmp.name, cfg)
        elif isinstance(cfg, str):
            cfg_dict = load_yaml(cfg)
            cfg_dict['nc'] = nc
            yaml_save(tmp.name, cfg_dict)
        model = YOLOv10WiSARD(model=tmp.name, task="detect").model
        weights = pretrained_model.state_dict()
        nc_safe_load(model, weights, nc)
    else:
        model = YOLOv10WiSARD(cfg, task="detect").model
    return model


MODEL_REGISTRY = {
    "vit_classifier": build_vit_classifier,
    "yolov9": build_yolo_v9,
    "yolov10": build_yolo_v10,
    "detr": build_detr,
    "defdetr": build_deformable_detr,
    "rtdetr": build_rtdetr,
    "fusiondetr": build_fusion_detr,
}
