import tempfile

from copy import deepcopy
from enum import StrEnum
from transformers import AutoModel, ViTForImageClassification

from sarfusion.experiment.utils import WrapperModule
from sarfusion.models.experimental import attempt_load
from sarfusion.models.utils import torch_dict_load
from sarfusion.models.utils import nc_safe_load
from sarfusion.models.yolov10 import YOLOv10WiSARD
from sarfusion.models.detr import DeformableDetr, Detr, FusionDetr, RTDetr, FusionRTDetr, FusionRTDetrCMX
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
            model_state = model.state_dict()
            new_weights = {}

            print(f"DEBUG: Attempting smart load from {pretrained_path}")
            
            # --- LOGICA DI MATCHING PER SUFFISSI ---
            def normalize_key(key):
                """Normalizza una chiave rimuovendo tutti i prefissi comuni"""
                # Rimuovi tutti i prefissi "model."
                while key.startswith("model."):
                    key = key[6:]
                # Rimuovi "decoder." prefix (checkpoint ha decoder.bbox_embed, modello ha bbox_embed)
                if key.startswith("decoder."):
                    key = key[8:]
                return key
            
            # Crea un dizionario inverso: suffix normalizzato -> chiave checkpoint
            ckpt_suffix_to_key = {}
            for k_ckpt in weights.keys():
                suffix = normalize_key(k_ckpt)
                ckpt_suffix_to_key[suffix] = k_ckpt
            
            for k_model in model_state.keys():
                # Prendiamo la parte finale della chiave normalizzata
                suffix = normalize_key(k_model)
                
                if suffix in ckpt_suffix_to_key:
                    k_ckpt = ckpt_suffix_to_key[suffix]
                    # Verifica che le shape corrispondano
                    if weights[k_ckpt].shape == model_state[k_model].shape:
                        new_weights[k_model] = weights[k_ckpt]
                    else:
                        print(f"⚠️ Shape mismatch for {k_model}: ckpt={weights[k_ckpt].shape}, model={model_state[k_model].shape}")
            
            # Carichiamo i pesi rimappati
            model.load_state_dict(new_weights, strict=False)
            print(f"✅ Smart Load Successful: {len(new_weights)}/{len(model_state)} layers matched.")
            
            # Se mancano molte chiavi, stampiamo un avviso
            if len(new_weights) < len(model_state) * 0.9:
                missing = len(model_state) - len(new_weights)
                print(f"⚠️ Warning: {missing} keys not matched. Check naming conventions.")
                # Stampa le prime 5 chiavi mancanti
                matched_keys = set(new_weights.keys())
                missing_keys = [k for k in model_state.keys() if k not in matched_keys]
                print(f"   First 5 missing: {missing_keys[:5]}")
                
        except Exception as e:
            print(f"❌ Error during smart loading: {e}")
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

def build_fusion_rt_detr(threshold=0.9, id2label=None):
    return FusionRTDetr(threshold=threshold, id2label=id2label)

def build_fusion_rt_detr_cmx(threshold=0.9, id2label=None):
    return FusionRTDetrCMX(threshold=threshold, id2label=id2label)


MODEL_REGISTRY = {
    "vit_classifier": build_vit_classifier,
    "yolov9": build_yolo_v9,
    "yolov10": build_yolo_v10,
    "detr": build_detr,
    "defdetr": build_deformable_detr,
    "rtdetr": build_rtdetr,
    "fusiondetr": build_fusion_detr,
    "fusion_rtdetr": build_fusion_rt_detr,
    "fusion_rtdetr_cmx": build_fusion_rt_detr_cmx,
}
