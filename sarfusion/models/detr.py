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
    def __init__(self, id2label, threshold=0.9):
        super(FusionRTDetr, self).__init__(
            processor_class=RTDetrImageProcessor,
            model_class=RTDetrFusionForObjectDetection,
            pretrained_model_name="PekingU/rtdetr_r50vd", 
            id2label=id2label,
            threshold=threshold,
        )
        # Force the processor to accept 4 channels
        self.processor.num_channels = 4