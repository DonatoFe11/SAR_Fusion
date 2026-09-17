"""Empirical recall versus false positives per image, with NumPy only.

Matching is performed within each image in descending confidence order. This
differs deliberately from the historical error-analysis IoU-order matcher:
lower-scoring predictions must not change the matches of predictions retained
at a higher confidence threshold. The historical analysis remains unchanged.
All equal-confidence predictions enter a curve together, so every returned
point is attainable by applying ``score >= threshold``.
"""

from __future__ import annotations

import numpy as np


def _boxes(value, name):
    boxes = np.asarray(value, dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"{name} must have shape (N, 4), including (0, 4) when empty")
    if not np.isfinite(boxes).all():
        raise ValueError(f"{name} must contain only finite coordinates")
    if np.any(boxes[:, 2:] <= boxes[:, :2]):
        raise ValueError(f"{name} must contain positive-area XYXY boxes")
    return boxes


def _scores(value, name="scores"):
    scores = np.asarray(value, dtype=np.float64)
    if scores.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError(f"{name} must contain finite confidence values in [0, 1]")
    return scores


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive; a zero denominator is undefined")
    return int(value)


def detection_events(gt_xyxy, pred_xyxy, scores, iou_threshold=0.5):
    """Return descending scores and per-prediction boolean TP flags for one image.

    Each prediction selects the highest-IoU *unused* ground-truth box if that
    IoU reaches ``iou_threshold``. Ties between scores preserve original
    prediction order; ties between ground truths use their original indices.
    Negative coordinates are permitted (the image bounds are not known here),
    but boxes must be finite and have positive area. Images without ground
    truth are valid and all their predictions are false positives.
    """
    ground_truth = _boxes(gt_xyxy, "gt_xyxy")
    predictions = _boxes(pred_xyxy, "pred_xyxy")
    confidence = _scores(scores)
    if len(predictions) != len(confidence):
        raise ValueError("pred_xyxy and scores must have the same length")
    if not np.isscalar(iou_threshold) or isinstance(iou_threshold, (str, bool, np.bool_)):
        raise ValueError("iou_threshold must be finite and in [0, 1]")
    if not np.isfinite(iou_threshold) or not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be finite and in [0, 1]")

    order = np.argsort(-confidence, kind="stable")
    confidence = confidence[order]
    predictions = predictions[order]
    true_positive = np.zeros(len(confidence), dtype=bool)
    if len(ground_truth) == 0 or len(predictions) == 0:
        return confidence, true_positive

    gt_area = np.prod(ground_truth[:, 2:] - ground_truth[:, :2], axis=1)
    available = np.ones(len(ground_truth), dtype=bool)
    for index, prediction in enumerate(predictions):
        if not available.any():
            break
        intersection_size = np.maximum(
            np.minimum(prediction[2:], ground_truth[:, 2:])
            - np.maximum(prediction[:2], ground_truth[:, :2]),
            0.0,
        )
        intersection = np.prod(intersection_size, axis=1)
        pred_area = np.prod(prediction[2:] - prediction[:2])
        union = pred_area + gt_area - intersection
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            iou = intersection / union
        if not np.isfinite(iou).all():
            raise ValueError("box coordinates exceed the numeric range for finite IoU")
        iou[~available] = -1.0
        match = int(np.argmax(iou))
        if iou[match] >= iou_threshold:
            true_positive[index] = True
            available[match] = False
    return confidence, true_positive


def empirical_curve(events, num_images, total_gt):
    """Pool image-local events into an attainable confidence-threshold curve.

    ``num_images`` counts every evaluated frame, including empty frames. The
    event list may omit frames with no predictions, but cannot have more entries
    than frames. ``total_gt`` counts all annotated objects, including missed
    ones, and must be positive because recall otherwise has no denominator.
    The first point rejects all detections (threshold infinity). No smoothing
    or interpolation is applied.
    """
    num_images = _positive_integer(num_images, "num_images")
    total_gt = _positive_integer(total_gt, "total_gt")
    events = list(events)
    if len(events) > num_images:
        raise ValueError("events cannot contain more image entries than num_images")
    score_parts, tp_parts = [], []
    for index, event in enumerate(events):
        if not isinstance(event, (tuple, list)) or len(event) != 2:
            raise ValueError("each image event must be a (scores, tp_flags) pair")
        scores = _scores(event[0], f"events[{index}].scores")
        tp_flags = np.asarray(event[1])
        if tp_flags.ndim != 1 or tp_flags.dtype != np.dtype(bool):
            raise ValueError("tp_flags must be a one-dimensional boolean array")
        if len(scores) != len(tp_flags):
            raise ValueError("event scores and tp_flags must have the same length")
        if np.any(np.diff(scores) > 0):
            raise ValueError("event scores must be sorted in descending order")
        score_parts.append(scores)
        tp_parts.append(tp_flags)

    scores = np.concatenate(score_parts) if score_parts else np.empty(0, dtype=np.float64)
    tp_flags = np.concatenate(tp_parts) if tp_parts else np.empty(0, dtype=bool)
    if int(tp_flags.sum()) > total_gt:
        raise ValueError("true-positive count cannot exceed total_gt")
    order = np.argsort(-scores, kind="stable")
    scores, tp_flags = scores[order], tp_flags[order]
    # Select only the end of each confidence tie group: intermediate points
    # cannot be reached by a confidence threshold shared by all predictions.
    ends = np.flatnonzero(np.r_[scores[:-1] != scores[1:], True]) if len(scores) else []
    cumulative_tp = np.cumsum(tp_flags, dtype=np.int64)
    tp = np.r_[np.int64(0), cumulative_tp[ends]]
    fp = np.r_[np.int64(0), (np.arange(len(scores), dtype=np.int64) + 1 - cumulative_tp)[ends]]
    return {
        "threshold": np.r_[np.inf, scores[ends]],
        "recall": tp.astype(np.float64) / total_gt,
        "fppi": fp.astype(np.float64) / num_images,
        "tp": tp,
        "fp": fp,
    }


def recall_at_budgets(curve, budgets):
    """Return the highest empirical recall satisfying each FP/image budget.

    This uses attainable threshold points only. In particular, a tied score
    group that crosses a budget is excluded in full, without interpolation.
    """
    names = ("threshold", "recall", "fppi", "tp", "fp")
    if not isinstance(curve, dict) or any(name not in curve for name in names):
        raise ValueError("curve must contain threshold, recall, fppi, tp, and fp")
    values = {name: np.asarray(curve[name], dtype=np.float64) for name in names}
    if any(value.ndim != 1 for value in values.values()):
        raise ValueError("curve arrays must be one-dimensional")
    size = len(values["threshold"])
    if size == 0 or any(len(value) != size for value in values.values()):
        raise ValueError("curve arrays must have the same nonzero length")
    threshold = values["threshold"]
    if threshold[0] != np.inf:
        raise ValueError("curve must start with an infinite reject-all threshold")
    _scores(threshold[1:], "curve thresholds")
    if np.any(np.diff(threshold) >= 0):
        raise ValueError("curve thresholds must strictly decrease after reject-all")
    for name in ("recall", "fppi", "tp", "fp"):
        value = values[name]
        if not np.isfinite(value).all() or np.any(value < 0):
            raise ValueError(f"curve {name} must be finite and nonnegative")
        if value[0] != 0 or np.any(np.diff(value) < 0):
            raise ValueError(f"curve {name} must start at zero and be nondecreasing")
    if np.any(values["recall"] > 1):
        raise ValueError("curve recall must be in [0, 1]")
    if any(np.any(values[name] != np.floor(values[name])) for name in ("tp", "fp")):
        raise ValueError("curve tp and fp must be integer counts")

    budgets = np.asarray(budgets, dtype=np.float64)
    if budgets.ndim != 1 or not np.isfinite(budgets).all() or np.any(budgets < 0):
        raise ValueError("budgets must be a one-dimensional array of finite nonnegative values")
    indices = np.searchsorted(values["fppi"], budgets, side="right") - 1
    return values["recall"][indices]
