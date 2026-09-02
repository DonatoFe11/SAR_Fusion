"""Sparse weak supervision for the P3 common-offset guidance branch."""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn.functional as F

from sarfusion.models.rtdetr_fusion import (
    BoxGuidedCommonOffsetFeatureAlignmentModule,
)
from sarfusion.utils.structures import LossOutput


DEFAULT_BOX_GUIDED_ALIGNMENT = {
    "enabled": False,
    "weight": 0.2,
    "start_epoch": 0,
    "warmup_epochs": 2,
    "smooth_l1_beta_cells": 0.25,
}


def validate_box_guided_alignment_config(config):
    """Normalize the frozen weak-guidance configuration."""
    normalized = deepcopy(DEFAULT_BOX_GUIDED_ALIGNMENT)
    if config:
        unknown = set(config) - set(normalized)
        if unknown:
            raise ValueError(
                "Unknown box-guided-alignment options: "
                + ", ".join(sorted(unknown))
            )
        normalized.update(config)

    normalized["enabled"] = bool(normalized["enabled"])
    normalized["weight"] = float(normalized["weight"])
    normalized["start_epoch"] = int(normalized["start_epoch"])
    normalized["warmup_epochs"] = int(normalized["warmup_epochs"])
    normalized["smooth_l1_beta_cells"] = float(
        normalized["smooth_l1_beta_cells"]
    )
    if normalized["weight"] <= 0.0:
        raise ValueError("box-guided-alignment weight must be positive")
    if normalized["start_epoch"] < 0:
        raise ValueError(
            "box-guided-alignment start_epoch must be non-negative"
        )
    if normalized["warmup_epochs"] < 1:
        raise ValueError(
            "box-guided-alignment warmup_epochs must be positive"
        )
    if normalized["smooth_l1_beta_cells"] <= 0.0:
        raise ValueError(
            "box-guided-alignment smooth_l1_beta_cells must be positive"
        )
    return normalized


def validate_box_guided_training_contract(
    config,
    dataset_config,
    model_config,
    *,
    modality_consistency_enabled=False,
):
    """Fail closed when the box-guided ablation is mixed with another one.

    The guidance target is defined only for the standard three-level FAM in
    the historical WiSARD coordinate contract.  In particular, level index
    zero would be P2 rather than P3 in a P2 model, and freezing FAM would also
    freeze the newly introduced predictor.  Rejecting those combinations here
    keeps the scientific intervention represented by the frozen YAML unique.
    """
    normalized = validate_box_guided_alignment_config(config)
    dataset_config = dataset_config or {}
    model_config = model_config or {}
    model_params = model_config.get("params", model_config)

    enabled = normalized["enabled"]
    targets_enabled = bool(dataset_config.get("box_alignment_targets", False))
    if enabled != targets_enabled:
        raise ValueError(
            "train.box_guided_alignment.enabled and "
            "dataset.box_alignment_targets must be enabled or disabled together"
        )
    if not enabled:
        return normalized

    if model_config.get("name", "fusion_rtdetr") != "fusion_rtdetr":
        raise ValueError("box-guided alignment requires model.name='fusion_rtdetr'")
    if model_params.get("use_fam") is not True:
        raise ValueError("box-guided alignment requires use_fam=true")
    if model_params.get("fam_variant") != "box_guided_common_offset_p3":
        raise ValueError(
            "box-guided alignment requires "
            "fam_variant='box_guided_common_offset_p3'"
        )
    if bool(model_params.get("freeze_fam", False)):
        raise ValueError("box-guided alignment requires freeze_fam=false")
    if bool(model_params.get("use_p2", False)):
        raise ValueError("box-guided alignment is defined only for P3--P5, not P2")
    if float(model_params.get("spatial_jitter_std", 0.0)) != 0.0:
        raise ValueError("box-guided alignment cannot be combined with SSJ")
    if float(model_params.get("ir_dropout_rate", 0.0)) != 0.0:
        raise ValueError(
            "box-guided alignment cannot be combined with feature IR dropout"
        )
    incompatible_flags = (
        "use_reliability_gating",
        "use_residual_alignment_gating",
        "use_scalar_residual_alignment",
    )
    active_flags = [
        name for name in incompatible_flags if bool(model_params.get(name, False))
    ]
    if active_flags:
        raise ValueError(
            "box-guided alignment cannot be combined with other alignment "
            f"gates: {', '.join(active_flags)}"
        )
    if modality_consistency_enabled or bool(
        dataset_config.get("paired_consistency", False)
    ):
        raise ValueError(
            "box-guided alignment and modality consistency are separate ablations"
        )
    if not bool(dataset_config.get("modal_dropout", False)):
        raise ValueError(
            "box-guided Stage A requires the historical Modal Dropout stream"
        )
    if dataset_config.get("modal_dropout_coordinate_contract", "native") != "native":
        raise ValueError(
            "box-guided Stage A requires native-coordinate Modal Dropout"
        )
    return normalized


def box_guided_alignment_epoch_scale(config, epoch):
    config = validate_box_guided_alignment_config(config)
    if not config["enabled"] or int(epoch) < config["start_epoch"]:
        return 0.0
    progress = int(epoch) - config["start_epoch"] + 1
    return min(1.0, progress / config["warmup_epochs"])


def find_box_guided_fam(model):
    modules = [
        module
        for module in model.modules()
        if isinstance(module, BoxGuidedCommonOffsetFeatureAlignmentModule)
    ]
    if len(modules) != 1:
        raise ValueError(
            "box-guided alignment expects exactly one guided P3 FAM module, "
            f"found {len(modules)}"
        )
    return modules[0]


def box_guided_alignment_loss(model, targets, config, epoch_scale=1.0):
    """Regress predicted P3 ``(dy, dx)`` at conservative box matches.

    Each target row is ``[vis_x, vis_y, ir_minus_vis_y, ir_minus_vis_x]`` in
    normalized input coordinates.  The predicted displacement is represented
    in feature-map cells, hence normalized target components are multiplied by
    the actual P3 height and width observed in this forward pass.
    """
    config = validate_box_guided_alignment_config(config)
    epoch_scale = float(epoch_scale)
    if not 0.0 <= epoch_scale <= 1.0:
        raise ValueError("box-guided-alignment epoch_scale must be in [0, 1]")
    if not isinstance(targets, (list, tuple)):
        raise ValueError("box_alignment_targets must be a list of tensors")

    module = find_box_guided_fam(model)
    flow = module.last_guidance_flow
    if flow is None:
        raise RuntimeError(
            "guided P3 FAM has no cached flow; compute the detection forward "
            "before the auxiliary loss"
        )
    if len(targets) != flow.shape[0]:
        raise ValueError(
            "box_alignment_targets batch length does not match guidance flow: "
            f"{len(targets)} versus {flow.shape[0]}"
        )

    height, width = flow.shape[-2:]
    prediction_terms = []
    target_terms = []
    samples_with_matches = 0
    for batch_index, sample_targets in enumerate(targets):
        sample_targets = torch.as_tensor(
            sample_targets,
            device=flow.device,
            dtype=flow.dtype,
        )
        if sample_targets.numel() == 0:
            continue
        if sample_targets.ndim != 2 or sample_targets.shape[1] != 4:
            raise ValueError(
                "each box-alignment target tensor must have shape [N, 4], got "
                f"{tuple(sample_targets.shape)}"
            )
        if not torch.isfinite(sample_targets).all():
            raise ValueError("box-alignment targets must be finite")
        positions = sample_targets[:, :2]
        if (positions < 0.0).any() or (positions > 1.0).any():
            raise ValueError(
                "box-alignment VIS centre coordinates must lie in [0, 1]"
            )

        # align_corners=False and normalized image coordinates both map the
        # domain edges to -1/+1.  Output is [1, 2, N, 1].
        sampling_grid = (2.0 * positions - 1.0).view(1, -1, 1, 2)
        predicted = F.grid_sample(
            flow[batch_index : batch_index + 1],
            sampling_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )[0, :, :, 0].transpose(0, 1)
        target_cells = torch.stack(
            (
                sample_targets[:, 2] * height,
                sample_targets[:, 3] * width,
            ),
            dim=-1,
        )
        prediction_terms.append(predicted)
        target_terms.append(target_cells)
        samples_with_matches += 1

    zero = flow.sum() * 0.0
    if prediction_terms:
        predictions = torch.cat(prediction_terms, dim=0)
        target_cells = torch.cat(target_terms, dim=0)
        raw_loss = F.smooth_l1_loss(
            predictions,
            target_cells,
            beta=config["smooth_l1_beta_cells"],
            reduction="mean",
        )
        matched_boxes = predictions.new_tensor(float(predictions.shape[0]))
    else:
        raw_loss = zero
        matched_boxes = zero.detach()

    weighted_loss = raw_loss * config["weight"] * epoch_scale
    result = LossOutput(
        value=weighted_loss,
        components={
            "box_guidance_raw_loss": raw_loss.detach(),
            "box_guidance_weighted_loss": weighted_loss.detach(),
            "box_guidance_matched_boxes": matched_boxes,
            "box_guidance_samples_with_matches": flow.new_tensor(
                float(samples_with_matches)
            ),
            "box_guidance_epoch_scale": flow.new_tensor(epoch_scale),
        },
    )
    # ``weighted_loss`` keeps the graph required by backward.  Releasing the
    # module-side reference prevents a completed batch graph from remaining
    # reachable until the next detector forward.
    module.last_guidance_flow = None
    return result
