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
from sarfusion.models.rtdetr_fusion_fam import RTDetrFusionForObjectDetection as RTDetrFusionFAMForObjectDetection
from sarfusion.models.rtdetr_cmx import RTDetrCMXForObjectDetection
from sarfusion.models.rtdetr_cmx_hybrid import RTDetrCMXHybridForObjectDetection
from sarfusion.models.deformable_detr_fusion import DeformableDetrFusionForObjectDetection
from sarfusion.models.dino_fusion import DINOFusionForObjectDetection


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
    def __init__(
        self,
        id2label,
        threshold=0.9,
        use_fam=False,
        freeze_fam=False,
        ir_dropout_rate=0.0,
        spatial_jitter_std=0.0,
        fam_variant="current_dcnv2",
        reuse_pretrained_class_head=False,
        use_p2=False,
        use_reliability_gating=False,
        reliability_gate_hidden_channels=16,
        use_residual_alignment_gating=False,
        residual_alignment_hidden_channels=16,
    ):
        super(FusionRTDetr, self).__init__(
            processor_class=RTDetrImageProcessor,
            model_class=RTDetrFusionForObjectDetection,
            pretrained_model_name="PekingU/rtdetr_r50vd", 
            id2label=id2label,
            threshold=threshold,
            use_fam=use_fam,  # Pass use_fam to model
            freeze_fam=freeze_fam, # Pass freeze_fam to model
            ir_dropout_rate=ir_dropout_rate, # Pass ir_dropout_rate to model
            spatial_jitter_std=spatial_jitter_std, # Pass spatial_jitter_std to model
            fam_variant=fam_variant,
            use_p2=use_p2,
            use_reliability_gating=use_reliability_gating,
            reliability_gate_hidden_channels=reliability_gate_hidden_channels,
            use_residual_alignment_gating=use_residual_alignment_gating,
            residual_alignment_hidden_channels=residual_alignment_hidden_channels,
            reuse_pretrained_class_head=reuse_pretrained_class_head,
        )
        # Force the processor to accept 4 channels
        self.processor.num_channels = 4
        self.use_fam = use_fam
        self.freeze_fam = freeze_fam
        self.ir_dropout_rate = ir_dropout_rate
        self.spatial_jitter_std = spatial_jitter_std
        self.fam_variant = fam_variant
        self.use_p2 = use_p2
        self.use_reliability_gating = use_reliability_gating
        self.reliability_gate_hidden_channels = int(
            reliability_gate_hidden_channels
        )
        self.use_residual_alignment_gating = bool(
            use_residual_alignment_gating
        )
        self.residual_alignment_hidden_channels = int(
            residual_alignment_hidden_channels
        )
        self.reuse_pretrained_class_head = reuse_pretrained_class_head


class FusionRTDetrFAM(BaseDetr):
    """RT-DETR FAM implementation (lazy FAM init in backbone forward)."""
    def __init__(self, id2label, threshold=0.9):
        super(FusionRTDetrFAM, self).__init__(
            processor_class=RTDetrImageProcessor,
            model_class=RTDetrFusionFAMForObjectDetection,
            pretrained_model_name="PekingU/rtdetr_r50vd",
            id2label=id2label,
            threshold=threshold,
        )
        # Force the processor to accept 4 channels
        self.processor.num_channels = 4

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

class FusionRTDetrCMXHybrid(BaseDetr):
    def __init__(self, id2label, threshold=0.9):
        super(FusionRTDetrCMXHybrid, self).__init__(
            processor_class=RTDetrImageProcessor,
            model_class=RTDetrCMXHybridForObjectDetection,
            pretrained_model_name="PekingU/rtdetr_r50vd", 
            id2label=id2label,
            threshold=threshold,
        )
        # Forza il processor ad accettare 4 canali
        self.processor.num_channels = 4


class FusionDeformableDetr(BaseDetr):
    """Deformable DETR with RGB-IR fusion + optional FAM alignment."""

    def __init__(
        self,
        id2label,
        threshold=0.9,
        num_feature_levels=None,
        use_fam=False,
        freeze_fam=False,
        ir_dropout_rate=0.0,
        spatial_jitter_std=0.0,
    ):
        model_kwargs = {}
        if num_feature_levels is not None:
            model_kwargs["num_feature_levels"] = num_feature_levels
        model_kwargs.update(
            {
                "use_fam": use_fam,
                "freeze_fam": freeze_fam,
                "ir_dropout_rate": ir_dropout_rate,
                "spatial_jitter_std": spatial_jitter_std,
            }
        )

        super(FusionDeformableDetr, self).__init__(
            processor_class=DeformableDetrImageProcessor,
            model_class=DeformableDetrFusionForObjectDetection,
            pretrained_model_name="SenseTime/deformable-detr",
            id2label=id2label,
            threshold=threshold,
            **model_kwargs,
        )
        self.processor.num_channels = 4
        self.use_fam = use_fam
        self.freeze_fam = freeze_fam
        self.ir_dropout_rate = ir_dropout_rate
        self.spatial_jitter_std = spatial_jitter_std


class FusionDINODeformableDetr(BaseDetr):
    """
    DINO-style Deformable DETR with RGB-IR fusion + optional FAM.
    Adds CDN training and Look-Forward-Twice on top of FusionDeformableDetr.
    """
 
    def __init__(
        self,
        id2label,
        threshold=0.9,
        num_feature_levels=None,
        use_fam=False,
        freeze_fam=False,
        ir_dropout_rate=0.0,
        spatial_jitter_std=0.0,
        num_dn_groups=5,
        label_noise_prob=0.5,
        box_noise_scale=1.0,
        cdn_loss_coef=1.0,
    ):
 
        model_kwargs = {}
        if num_feature_levels is not None:
            model_kwargs["num_feature_levels"] = num_feature_levels
        model_kwargs.update(
            {
                "use_fam": use_fam,
                "freeze_fam": freeze_fam,
                "ir_dropout_rate": ir_dropout_rate,
                "spatial_jitter_std": spatial_jitter_std,
                "num_dn_groups": num_dn_groups,
                "label_noise_prob": label_noise_prob,
                "box_noise_scale": box_noise_scale,
                "cdn_loss_coef": cdn_loss_coef,
            }
        )
 
        super(FusionDINODeformableDetr, self).__init__(
            processor_class=DeformableDetrImageProcessor,
            model_class=DINOFusionForObjectDetection,
            pretrained_model_name="SenseTime/deformable-detr",
            id2label=id2label,
            threshold=threshold,
            **model_kwargs,
        )
        self.processor.num_channels = 4
