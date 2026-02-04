import torch
import torch.nn as nn

from huggingface_hub import PyTorchModelHubMixin
from transformers import (
    DetrImageProcessor,
    DetrForObjectDetection,
    DeformableDetrForObjectDetection,
    DeformableDetrImageProcessor,
    RTDetrForObjectDetection,
    RTDetrImageProcessor,
)

from sarfusion.utils.structures import LossOutput
from sarfusion.utils.general import xyxy2xywh
from sarfusion.models.detr_fusion import DetrFusionForObjectDetection
from sarfusion.models.rtdetr_fusion import RTDetrFusionForObjectDetection
from sarfusion.models.rtdetr_cmx import RTDetrCMXForObjectDetection
from sarfusion.models.deformable_detr_fusion import DeformableDetrFusionForObjectDetection
from sarfusion.models.dino_fusion import DinoFusionForObjectDetection


def convert_detr_predictions(predictions):
    for i, pred in enumerate(predictions):
        boxes = pred["boxes"]
        predictions[i]["boxes"] = xyxy2xywh(boxes)
    return predictions


class BaseDetr(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        processor_class,
        model_class,
        pretrained_model_name,
        id2label,
        threshold=0.9,
        **model_kwargs,  # Extra kwargs to pass to from_pretrained
    ):
        super(BaseDetr, self).__init__()
        label2id = {c: str(i) for i, c in enumerate(id2label)}
        self.processor = processor_class.from_pretrained(
            pretrained_model_name, id2label=id2label, label2id=label2id
        )
        self.model = model_class.from_pretrained(
            pretrained_model_name,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
            **model_kwargs,  # Pass extra kwargs
        )
        self.threshold = threshold

    # FIX: Aggiunto pixel_mask e passato a self.model
    def forward(self, pixel_values, pixel_mask=None, labels=None, threshold=None):
        outputs = self.model(pixel_values, pixel_mask=pixel_mask, labels=labels)
        if not self.training:
            threshold = threshold if threshold is not None else self.threshold
            outputs["predictions"] = convert_detr_predictions(
                self.processor.post_process_object_detection(
                    outputs, threshold=threshold
                )
            )
        if "loss" in outputs:
            outputs["loss"] = LossOutput(
                value=outputs["loss"], components=outputs["loss_dict"]
            )
        return outputs


class Detr(BaseDetr):
    def __init__(
        self, id2label, threshold=0.9, pretrained_model_name="facebook/detr-resnet-50"
    ):
        super(Detr, self).__init__(
            processor_class=DetrImageProcessor,
            model_class=DetrForObjectDetection,
            pretrained_model_name=pretrained_model_name,
            id2label=id2label,
            threshold=threshold,
        )

    # FIX: Aggiunto pixel_mask anche qui
    def forward(self, pixel_values, pixel_mask=None, labels=None):
        outputs = self.model(pixel_values, pixel_mask=pixel_mask, labels=labels)

        outputs["logits_stripped"] = outputs.logits[:, :, :-1]

        if not self.training:
            outputs["predictions"] = convert_detr_predictions(
                self.processor.post_process_object_detection(
                    outputs, threshold=self.threshold
                )
            )

        if "loss" in outputs:
            outputs["loss"] = LossOutput(
                value=outputs["loss"], components=outputs["loss_dict"]
            )
        return outputs


class DeformableDetr(BaseDetr):
    def __init__(self, id2label, threshold=0.9):
        super(DeformableDetr, self).__init__(
            processor_class=DeformableDetrImageProcessor,
            model_class=DeformableDetrForObjectDetection,
            pretrained_model_name="SenseTime/deformable-detr",
            id2label=id2label,
            threshold=threshold,
        )


class RTDetr(BaseDetr):
    def __init__(self, id2label, threshold=0.9):
        super(RTDetr, self).__init__(
            processor_class=RTDetrImageProcessor,
            model_class=RTDetrForObjectDetection,
            pretrained_model_name="PekingU/rtdetr_r50vd",
            id2label=id2label,
            threshold=threshold,
        )
        
        
class FusionDetr(BaseDetr):
    def __init__(self, id2label, threshold=0.9):
        super(FusionDetr, self).__init__(
            processor_class=DetrImageProcessor,
            model_class=DetrFusionForObjectDetection,
            pretrained_model_name="facebook/detr-resnet-50",
            id2label=id2label,
            threshold=threshold,
        )

class FusionRTDetr(BaseDetr):
    def __init__(self, id2label, threshold=0.9, use_fam=False):
        super(FusionRTDetr, self).__init__(
            processor_class=RTDetrImageProcessor,
            model_class=RTDetrFusionForObjectDetection,
            pretrained_model_name="PekingU/rtdetr_r50vd", 
            id2label=id2label,
            threshold=threshold,
            use_fam=use_fam,  # Pass use_fam to model
        )
        # Force the processor to accept 4 channels
        self.processor.num_channels = 4
        self.use_fam = use_fam

class FusionRTDetrCMX(BaseDetr):
    def __init__(self, id2label, threshold=0.9):
        super(FusionRTDetrCMX, self).__init__(
            processor_class=RTDetrImageProcessor,
            model_class=RTDetrCMXForObjectDetection,
            pretrained_model_name="PekingU/rtdetr_r50vd", 
            id2label=id2label,
            threshold=threshold,
        )
        # Forza il processor ad accettare 4 canali (3 RGB + 1 IR)
        self.processor.num_channels = 4


class FusionDeformableDetr(BaseDetr):
    """Deformable DETR with RGB-IR fusion via token concatenation.
    
    This model is robust to spatial misalignment between RGB and IR modalities
    thanks to deformable attention with learnable offsets and token-level fusion.
    """
    def __init__(self, id2label, threshold=0.9, num_feature_levels=None):
        model_kwargs = {}
        if num_feature_levels is not None:
            model_kwargs['num_feature_levels'] = num_feature_levels
            
        super(FusionDeformableDetr, self).__init__(
            processor_class=DeformableDetrImageProcessor,
            model_class=DeformableDetrFusionForObjectDetection,
            pretrained_model_name="SenseTime/deformable-detr",
            id2label=id2label,
            threshold=threshold,
            **model_kwargs,
        )
        # Force the processor to accept 4 channels (3 RGB + 1 IR)
        self.processor.num_channels = 4


class FusionDino(BaseDetr):
    """DINO (DETR with Improved deNoising anchOr boxes) with RGB-IR fusion.
    
    DINO is SOTA detection transformer with channel concatenation for efficiency.
    Uses Conv + BN + ReLU for expressive cross-modal fusion.
    """
    def __init__(self, id2label, threshold=0.9, num_feature_levels=None, use_fam=False):
        model_kwargs = {}
        if num_feature_levels is not None:
            model_kwargs['num_feature_levels'] = num_feature_levels
        model_kwargs['use_fam'] = use_fam
            
        super(FusionDino, self).__init__(
            processor_class=DeformableDetrImageProcessor,
            model_class=DinoFusionForObjectDetection,
            pretrained_model_name="SenseTime/deformable-detr",  # DINO uses same arch as Deformable DETR
            id2label=id2label,
            threshold=threshold,
            **model_kwargs,
        )
        # Force the processor to accept 4 channels (3 RGB + 1 IR)
        self.use_fam = use_fam
        self.processor.num_channels = 4
