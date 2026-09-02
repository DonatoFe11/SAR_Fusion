"""Detection consistency for paired RGB--IR modality degradation.

The clean online teacher and degraded student can reorder RT-DETR queries.
This module therefore matches confident teacher predictions to student
queries before applying classification and localization consistency losses.
"""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.ops import generalized_box_iou

from sarfusion.utils.structures import LossOutput


DEFAULT_MODALITY_CONSISTENCY = {
    "enabled": False,
    "teacher": "online_eval_stop_gradient",
    "start_epoch": 1,
    "warmup_epochs": 2,
    "confidence_threshold": 0.2,
    "max_teacher_queries": 20,
    "matching_class_cost": 2.0,
    "matching_bbox_cost": 5.0,
    "matching_giou_cost": 2.0,
    "classification_weight": 2.0,
    "bbox_weight": 5.0,
    "giou_weight": 2.0,
}


def validate_modality_consistency_config(config):
    """Return a normalized and strictly validated consistency config."""
    normalized = deepcopy(DEFAULT_MODALITY_CONSISTENCY)
    if config:
        unknown = set(config) - set(normalized)
        if unknown:
            raise ValueError(
                "Unknown modality-consistency options: "
                + ", ".join(sorted(unknown))
            )
        normalized.update(config)

    normalized["enabled"] = bool(normalized["enabled"])
    if normalized["teacher"] != "online_eval_stop_gradient":
        raise ValueError(
            "modality consistency teacher must be "
            "'online_eval_stop_gradient'"
        )

    normalized["start_epoch"] = int(normalized["start_epoch"])
    normalized["warmup_epochs"] = int(normalized["warmup_epochs"])
    normalized["max_teacher_queries"] = int(
        normalized["max_teacher_queries"]
    )
    if normalized["start_epoch"] < 0:
        raise ValueError("modality consistency start_epoch must be non-negative")
    if normalized["warmup_epochs"] < 1:
        raise ValueError("modality consistency warmup_epochs must be positive")
    if normalized["max_teacher_queries"] < 1:
        raise ValueError(
            "modality consistency max_teacher_queries must be positive"
        )

    normalized["confidence_threshold"] = float(
        normalized["confidence_threshold"]
    )
    if not 0.0 <= normalized["confidence_threshold"] <= 1.0:
        raise ValueError(
            "modality consistency confidence_threshold must be in [0, 1]"
        )

    weighted_keys = (
        "matching_class_cost",
        "matching_bbox_cost",
        "matching_giou_cost",
        "classification_weight",
        "bbox_weight",
        "giou_weight",
    )
    for key in weighted_keys:
        normalized[key] = float(normalized[key])
        if normalized[key] < 0.0:
            raise ValueError(f"modality consistency {key} must be non-negative")
    if not any(normalized[key] > 0.0 for key in weighted_keys[:3]):
        raise ValueError("at least one consistency matching cost must be positive")
    if not any(normalized[key] > 0.0 for key in weighted_keys[3:]):
        raise ValueError("at least one consistency loss weight must be positive")
    return normalized


def modality_consistency_epoch_scale(config, epoch):
    """Linear warm-up: zero before start, then reach one in fixed epochs."""
    config = validate_modality_consistency_config(config)
    if not config["enabled"] or int(epoch) < config["start_epoch"]:
        return 0.0
    progress = int(epoch) - config["start_epoch"] + 1
    return min(1.0, progress / config["warmup_epochs"])


def _center_to_corners(boxes):
    center_x, center_y, width, height = boxes.unbind(-1)
    return torch.stack(
        (
            center_x - 0.5 * width,
            center_y - 0.5 * height,
            center_x + 0.5 * width,
            center_y + 0.5 * height,
        ),
        dim=-1,
    )


def _weighted_mean(values, weights):
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)


def matched_detection_consistency_loss(
    teacher_output,
    student_output,
    config,
    epoch_scale=1.0,
):
    """Match teacher boxes and return soft classification/box consistency.

    Teacher logits and boxes are always detached. Matching is discrete and is
    computed on detached costs; gradients flow only through matched student
    logits and boxes.
    """
    config = validate_modality_consistency_config(config)
    epoch_scale = float(epoch_scale)
    if not 0.0 <= epoch_scale <= 1.0:
        raise ValueError("consistency epoch_scale must be in [0, 1]")

    teacher_logits = teacher_output.logits.detach()
    teacher_boxes = teacher_output.pred_boxes.detach()
    student_logits = student_output.logits
    student_boxes = student_output.pred_boxes

    if teacher_logits.shape != student_logits.shape:
        raise ValueError(
            "teacher/student logits must have the same shape, got "
            f"{tuple(teacher_logits.shape)} and {tuple(student_logits.shape)}"
        )
    if teacher_boxes.shape != student_boxes.shape:
        raise ValueError(
            "teacher/student boxes must have the same shape, got "
            f"{tuple(teacher_boxes.shape)} and {tuple(student_boxes.shape)}"
        )
    if teacher_logits.ndim != 3 or teacher_boxes.ndim != 3:
        raise ValueError("consistency expects batched [B, Q, C] model outputs")
    if teacher_boxes.shape[-1] != 4:
        raise ValueError("consistency boxes must use normalized cxcywh format")

    classification_terms = []
    bbox_terms = []
    giou_terms = []
    confidence_weights = []
    samples_with_matches = 0

    for batch_index in range(teacher_logits.shape[0]):
        teacher_probabilities = teacher_logits[batch_index].sigmoid()
        teacher_scores, teacher_labels = teacher_probabilities.max(dim=-1)
        selected = torch.nonzero(
            teacher_scores >= config["confidence_threshold"], as_tuple=False
        ).flatten()
        if selected.numel() == 0:
            continue
        if selected.numel() > config["max_teacher_queries"]:
            top_positions = torch.topk(
                teacher_scores[selected],
                k=config["max_teacher_queries"],
                sorted=True,
            ).indices
            selected = selected[top_positions]

        selected_teacher_boxes = teacher_boxes[batch_index, selected]
        selected_teacher_probs = teacher_probabilities[selected]
        selected_teacher_labels = teacher_labels[selected]
        selected_teacher_scores = teacher_scores[selected]
        current_student_boxes = student_boxes[batch_index]
        current_student_probabilities = student_logits[batch_index].sigmoid()

        class_cost = -current_student_probabilities[
            :, selected_teacher_labels
        ].transpose(0, 1)
        bbox_cost = torch.cdist(
            selected_teacher_boxes,
            current_student_boxes,
            p=1,
        )
        giou_cost = -generalized_box_iou(
            _center_to_corners(selected_teacher_boxes),
            _center_to_corners(current_student_boxes),
        )
        matching_cost = (
            config["matching_class_cost"] * class_cost
            + config["matching_bbox_cost"] * bbox_cost
            + config["matching_giou_cost"] * giou_cost
        )
        teacher_indices, student_indices = linear_sum_assignment(
            matching_cost.detach().cpu().numpy()
        )
        teacher_indices = torch.as_tensor(
            teacher_indices,
            dtype=torch.long,
            device=student_logits.device,
        )
        student_indices = torch.as_tensor(
            student_indices,
            dtype=torch.long,
            device=student_logits.device,
        )
        if teacher_indices.numel() == 0:
            continue

        matched_teacher_probs = selected_teacher_probs[teacher_indices]
        matched_teacher_boxes = selected_teacher_boxes[teacher_indices]
        matched_teacher_scores = selected_teacher_scores[teacher_indices]
        matched_student_logits = student_logits[batch_index, student_indices]
        matched_student_boxes = current_student_boxes[student_indices]

        classification_terms.append(
            F.mse_loss(
                matched_student_logits.sigmoid(),
                matched_teacher_probs,
                reduction="none",
            ).mean(dim=-1)
        )
        bbox_terms.append(
            F.l1_loss(
                matched_student_boxes,
                matched_teacher_boxes,
                reduction="none",
            ).mean(dim=-1)
        )
        matched_giou = generalized_box_iou(
            _center_to_corners(matched_teacher_boxes),
            _center_to_corners(matched_student_boxes),
        ).diagonal()
        giou_terms.append(1.0 - matched_giou)
        confidence_weights.append(matched_teacher_scores)
        samples_with_matches += 1

    zero = (student_logits.sum() + student_boxes.sum()) * 0.0
    if not confidence_weights:
        classification_loss = zero
        bbox_loss = zero
        giou_loss = zero
        matched_queries = 0
    else:
        weights = torch.cat(confidence_weights)
        classification_loss = _weighted_mean(
            torch.cat(classification_terms), weights
        )
        bbox_loss = _weighted_mean(torch.cat(bbox_terms), weights)
        giou_loss = _weighted_mean(torch.cat(giou_terms), weights)
        matched_queries = int(weights.numel())

    weighted_classification = config["classification_weight"] * classification_loss
    weighted_bbox = config["bbox_weight"] * bbox_loss
    weighted_giou = config["giou_weight"] * giou_loss
    total = epoch_scale * (
        weighted_classification + weighted_bbox + weighted_giou
    )
    return LossOutput(
        value=total,
        components={
            "consistency_classification": classification_loss,
            "consistency_bbox": bbox_loss,
            "consistency_giou": giou_loss,
            "consistency_total": total,
            "consistency_epoch_scale": torch.as_tensor(
                epoch_scale, device=student_logits.device
            ),
            "consistency_matched_queries": torch.as_tensor(
                matched_queries, device=student_logits.device
            ),
            "consistency_samples_with_matches": torch.as_tensor(
                samples_with_matches, device=student_logits.device
            ),
        },
    )
