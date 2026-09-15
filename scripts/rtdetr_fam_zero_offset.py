"""Opt-in offset-only initialization; the historical training sources stay intact.

Register this factory explicitly before using build_model. It builds the ordinary
pretrained RT-DETR FAM first, then zeros ONLY predictor rows 0:18. No reset is
performed during forward or checkpoint loading. Rows 18:27 (mask logits), DCNv2
filters, parameter identities and random-number-generator state are preserved.
"""

from __future__ import annotations

from pathlib import Path

import torch

from sarfusion.models import MODEL_REGISTRY, build_fusion_rt_detr
from sarfusion.models.rtdetr_fusion import FeatureAlignmentModule
from sarfusion.utils import reproducibility as repro


ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "fusion_rtdetr_zero_offset"
MANIFEST_ID = "rtdetr_fam_zero_offset_training_source_v1"
PROTOCOL_PATH = "parameters/RTDETR/rtdetr_fam_zero_offset_stage_a_protocol.json"
SOURCE_FILES = (*repro.BOX_GUIDED_TRAINING_SOURCE_FILES,
                "scripts/rtdetr_fam_zero_offset.py",
                "scripts/run_rtdetr_fam_zero_offset_stage_a.py",
                "scripts/run_rtdetr_fam_zero_offset_stage_a.sh",
                "scripts/replay_rtdetr_v2_stage_a_validation.py",
                "sarfusion/data/temporal_split.py",
                "parameters/RTDETR/rtdetr_fam_stage_a_five_seed_v2.yaml",
                PROTOCOL_PATH)


def fam_modules(model):
    modules = [(name, module) for name, module in model.named_modules()
               if type(module) is FeatureAlignmentModule]
    if len(modules) != 3:
        raise ValueError("Offset-only ablation requires three standard DCNv2 FAMs")
    for name, module in modules:
        if module.spatial_jitter_std != 0 or module.offset_conv.out_channels != 27:
            raise ValueError(f"Unexpected offset predictor or SSJ in {name}")
        if not all(p.requires_grad for p in module.parameters()):
            raise ValueError(f"FAM must remain trainable: {name}")
    return modules


def unaffected_digest(model, modules):
    state = dict(model.state_dict())
    for name, _ in modules:
        for suffix in ("weight", "bias"):
            key = f"{name}.offset_conv.{suffix}"
            state[key] = state[key][18:]
    return repro.state_dict_digest(state)


def zero_offsets_only(model):
    """Apply the intervention after ALL model/pretrained initialization.

    Returns a serializable audit. This function consumes no random draws and
    never replaces Parameter objects, so optimizer membership is unchanged.
    """
    modules = fam_modules(model)
    initial = repro.model_digests(model)
    unchanged_before = unaffected_digest(model, modules)
    cpu_rng = torch.random.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []
    parameter_ids = [id(p) for p in model.parameters()]
    levels = []
    for name, module in modules:
        predictor = module.offset_conv
        mask_hash = repro.state_dict_digest({
            "weight": predictor.weight[18:], "bias": predictor.bias[18:]})
        levels.append({"name": name,
                       "offset_weight_absmax_before": float(predictor.weight[:18].detach().abs().max()),
                       "mask_parameters_sha256": mask_hash,
                       "deform_conv_sha256": repro.state_dict_digest(module.deform_conv.state_dict())})
        with torch.no_grad():
            predictor.weight[:18].zero_()
            predictor.bias[:18].zero_()
        if torch.count_nonzero(predictor.weight[:18]) or torch.count_nonzero(predictor.bias[:18]):
            raise RuntimeError("Failed to initialize offset predictor to zero")
    unchanged_after = unaffected_digest(model, modules)
    if unchanged_after != unchanged_before:
        raise RuntimeError("Offset reset modified masks, DCNv2 filters or other model state")
    if parameter_ids != [id(p) for p in model.parameters()]:
        raise RuntimeError("Offset reset replaced Parameter objects")
    if not torch.equal(cpu_rng, torch.random.get_rng_state()):
        raise RuntimeError("Offset reset consumed CPU random draws")
    if cuda_rng and any(not torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all())):
        raise RuntimeError("Offset reset consumed CUDA random draws")
    return {"intervention": "offset_rows_0_18_only_after_pretrained_initialization",
            "reference_initialization": initial,
            "candidate_initialization": repro.model_digests(model),
            "unchanged_parameters_sha256": unchanged_after,
            "rng_preserved": True, "parameters_preserved": True,
            "offsets_initially_zero": True, "masks_reset": False,
            "dcnv2_reset": False, "levels": levels}


def build_zero_offset_model(**params):
    required = {"use_fam": True, "freeze_fam": False,
                "fam_variant": "current_dcnv2", "ir_dropout_rate": 0.0,
                "spatial_jitter_std": 0.0, "reuse_pretrained_class_head": True}
    if any(params.get(key) != value for key, value in required.items()):
        raise ValueError("Offset-only experiment requires the unchanged standard FAM recipe")
    forbidden = ("use_p2", "use_reliability_gating", "use_residual_alignment_gating",
                 "use_scalar_residual_alignment")
    if any(params.get(key, False) for key in forbidden):
        raise ValueError("Do not combine offset initialization with another intervention")
    model = build_fusion_rt_detr(**params)
    model.zero_offset_initialization_report = zero_offsets_only(model)
    model.fam_initialization = "zero_offsets_only_after_hf_post_init"
    return model


def register_experiment():
    """Extend existing registries only in this experiment's worker process."""
    existing = MODEL_REGISTRY.get(MODEL_NAME)
    if existing is not None and existing is not build_zero_offset_model:
        raise RuntimeError(f"Model name already registered: {MODEL_NAME}")
    MODEL_REGISTRY[MODEL_NAME] = build_zero_offset_model
    repro.TRAINING_SOURCE_FILES[MANIFEST_ID] = SOURCE_FILES
