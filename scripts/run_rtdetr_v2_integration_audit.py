#!/usr/bin/env python3
"""Offline-capable Gate-0 audit for the pinned RT-DETRv2 integration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from accelerate import Accelerator
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection

from sarfusion.models.rtdetr_v2_fusion import (
    RTDetrV2FusionForObjectDetection,
)
from sarfusion.models.detr import RTDetrV2
from sarfusion.models.checkpoints import complete_shared_state_dict_aliases
from sarfusion.experiment.utils import WrapperModule
from sarfusion.experiment.run import Run
from sarfusion.utils.reproducibility import runtime_fingerprint


CHECKPOINT = "PekingU/rtdetr_v2_r50vd"
REVISION = "282494075698cab9faa1096ae26856890030c817"
EXPECTED_MODEL_SHA256 = (
    "3331d977dbc0c7a6cdae9ec0b0b6ad156eb6720d65b7cf0fa710dcc541d88d71"
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_shared_state_equal(control, candidate):
    control_state = control.state_dict()
    candidate_state = candidate.state_dict()
    shared_keys = sorted(set(control_state) & set(candidate_state))
    if not shared_keys:
        raise RuntimeError("Additive and FAM models have no shared state keys")
    for key in shared_keys:
        torch.testing.assert_close(
            candidate_state[key], control_state[key], atol=0, rtol=0
        )
    return len(shared_keys)


def build_fusion(id2label, *, use_fam):
    return RTDetrV2FusionForObjectDetection.from_pretrained(
        CHECKPOINT,
        id2label=id2label,
        label2id={label: index for index, label in id2label.items()},
        use_fam=use_fam,
        fam_variant="current_dcnv2",
        reuse_pretrained_class_head=(len(id2label) == 1),
        revision=REVISION,
    )


def run_audit(device: str):
    torch.manual_seed(40)
    model_path = hf_hub_download(
        CHECKPOINT, "model.safetensors", revision=REVISION
    )
    model_sha256 = file_sha256(model_path)
    if model_sha256 != EXPECTED_MODEL_SHA256:
        raise RuntimeError(
            "Pinned RT-DETRv2 model SHA-256 mismatch: "
            f"{model_sha256} != {EXPECTED_MODEL_SHA256}"
        )

    processor = RTDetrImageProcessor.from_pretrained(
        CHECKPOINT, revision=REVISION
    )
    base = RTDetrV2ForObjectDetection.from_pretrained(
        CHECKPOINT, revision=REVISION
    )
    if base.config.model_type != "rt_detr_v2":
        raise RuntimeError(f"Unexpected model_type: {base.config.model_type}")
    if base.config.decoder_method != "default":
        raise RuntimeError(
            f"Unexpected decoder_method: {base.config.decoder_method}"
        )

    # Canonical RGB parity is checked without changing the checkpoint label
    # space; reducing to one class changes encoder query selection and would no
    # longer be an architecture-only comparison.
    parity = build_fusion(base.config.id2label, use_fam=False)
    base.eval().to(device)
    parity.eval().to(device)
    # R50 uses 300 encoder top-k queries; 128 px yields 336 pyramid
    # locations, while a 64 px engineering input would be too small.
    audit_size = 128
    pixels_rgb = torch.randn(
        1, 3, audit_size, audit_size, device=device
    )
    pixel_mask = torch.ones(
        1, audit_size, audit_size, dtype=torch.bool, device=device
    )
    with torch.no_grad():
        expected = base(pixels_rgb, pixel_mask=pixel_mask)
        actual = parity(pixels_rgb, pixel_mask=pixel_mask)
    torch.testing.assert_close(actual.logits, expected.logits, atol=0, rtol=0)
    torch.testing.assert_close(
        actual.pred_boxes, expected.pred_boxes, atol=0, rtol=0
    )
    del parity, base, expected, actual

    # Exercise the repository-level standard registry wrapper as a separate
    # integration path; it is engineering parity, not the FAM control arm.
    standard = RTDetrV2(
        id2label={0: "person"},
        threshold=0.01,
        pretrained_model_name=CHECKPOINT,
        pretrained_revision=REVISION,
    )
    standard.model.eval().to(device)
    with torch.no_grad():
        standard_output = standard.model(
            pixels_rgb, pixel_mask=pixel_mask
        )
    if standard_output.logits.shape[-1] != 1:
        raise RuntimeError(
            "The standard RT-DETRv2 wrapper did not expose one class"
        )
    if not torch.isfinite(standard_output.logits).all():
        raise RuntimeError("Non-finite standard RT-DETRv2 logits")
    if standard.pretrained_label_source_indices != [0]:
        raise RuntimeError(
            "The standard wrapper did not reuse the pretrained COCO person row"
        )
    if standard.model.config.label2id != {"person": 0}:
        raise RuntimeError(
            f"Invalid standard label2id: {standard.model.config.label2id}"
        )
    del standard, standard_output

    id2label = {0: "person"}
    torch.manual_seed(40)
    rng_before_control = torch.random.get_rng_state().clone()
    control = build_fusion(id2label, use_fam=False)
    rng_after_control = torch.random.get_rng_state().clone()
    torch.testing.assert_close(rng_after_control, rng_before_control)

    torch.manual_seed(40)
    rng_before_candidate = torch.random.get_rng_state().clone()
    candidate = build_fusion(id2label, use_fam=True)
    rng_after_candidate = torch.random.get_rng_state().clone()
    torch.testing.assert_close(rng_after_candidate, rng_before_candidate)
    torch.testing.assert_close(rng_after_candidate, rng_after_control)

    shared_tensor_count = assert_shared_state_equal(control, candidate)
    fam_modules = candidate.model.backbone.fam_modules
    if fam_modules is None or len(fam_modules) != 3:
        raise RuntimeError("The scientific candidate must expose three FAMs")
    fam_offset_max_abs = [
        float(fam.offset_conv.weight.detach().abs().max())
        for fam in fam_modules
    ]
    if any(value == 0.0 for value in fam_offset_max_abs):
        raise RuntimeError(
            "current_dcnv2 no longer reproduces historical HF post_init"
        )

    candidate.to(device).train()
    pixels_fusion = torch.randn(
        1, 4, audit_size, audit_size, device=device
    )
    features = candidate.model.backbone(pixels_fusion, pixel_mask)
    backbone_loss = sum(feature.square().mean() for feature, _ in features)
    if not torch.isfinite(backbone_loss):
        raise RuntimeError("Non-finite RT-DETRv2 FAM backbone loss")
    backbone_loss.backward()
    fam_gradient_counts = []
    for level, fam in enumerate(fam_modules):
        gradients = [
            parameter.grad
            for parameter in fam.parameters()
            if parameter.requires_grad
        ]
        if not gradients or any(gradient is None for gradient in gradients):
            raise RuntimeError(f"Missing gradients in FAM level {level}")
        if any(not torch.isfinite(gradient).all() for gradient in gradients):
            raise RuntimeError(f"Non-finite gradients in FAM level {level}")
        fam_gradient_counts.append(len(gradients))

    candidate.zero_grad(set_to_none=True)
    candidate.eval()
    with torch.no_grad():
        fusion_output = candidate(pixels_fusion, pixel_mask=pixel_mask)
    if not torch.isfinite(fusion_output.logits).all():
        raise RuntimeError("Non-finite RT-DETRv2 FAM logits")
    if not torch.isfinite(fusion_output.pred_boxes).all():
        raise RuntimeError("Non-finite RT-DETRv2 FAM boxes")

    # Reproduce the operational checkpoint path: nested repository wrapper,
    # Accelerator Safetensors serialization, CPU safe_open, and strict reload.
    model_holder = torch.nn.Module()
    model_holder.add_module("model", candidate)
    wrapped = WrapperModule(model_holder)
    accelerator = Accelerator()
    wrapped = accelerator.prepare(wrapped)
    checkpoint_state_keys = set(wrapped.state_dict())
    parameter_name, parameter = next(iter(wrapped.named_parameters()))
    expected_parameter = parameter.detach().cpu().clone()
    with tempfile.TemporaryDirectory(prefix="rtdetr-v2-roundtrip-") as directory:
        checkpoint_dir = Path(directory) / "best"
        accelerator.save_state(output_dir=checkpoint_dir)
        checkpoint_path = checkpoint_dir / "model.safetensors"
        if not checkpoint_path.is_file():
            raise RuntimeError(
                "Accelerator did not produce the operational model.safetensors"
            )
        checkpoint_sha256 = file_sha256(checkpoint_path)
        with safe_open(checkpoint_path, framework="pt") as source:
            checkpoint_weights = {
                key: source.get_tensor(key) for key in source.keys()
            }
        checkpoint_weights, restored_aliases = (
            complete_shared_state_dict_aliases(wrapped, checkpoint_weights)
        )
        if not restored_aliases:
            raise RuntimeError(
                "Expected RT-DETRv2 tied aliases were not exercised"
            )
        if set(checkpoint_weights) != checkpoint_state_keys:
            raise RuntimeError(
                "Non-alias checkpoint keys changed during serialization"
            )
        with torch.no_grad():
            parameter.zero_()
        run = object.__new__(Run)
        run.tracker = SimpleNamespace(local_dir=directory)
        run.params = {"strict_checkpoint_loading": True}
        run.model = wrapped
        run.restore_model("best")
        torch.testing.assert_close(
            dict(wrapped.named_parameters())[parameter_name].detach().cpu(),
            expected_parameter,
            atol=0,
            rtol=0,
        )
    accelerator.end_training()

    return {
        "status": "passed",
        "checkpoint": CHECKPOINT,
        "revision": REVISION,
        "model_safetensors_sha256": model_sha256,
        "model_type": candidate.config.model_type,
        "decoder_method": candidate.config.decoder_method,
        "processor_size": processor.size,
        "standard_wrapper_status": "passed",
        "device": device,
        "shared_tensor_count": shared_tensor_count,
        "fam_level_count": len(fam_modules),
        "fam_initialization": candidate.fam_initialization,
        "fam_offset_max_abs": fam_offset_max_abs,
        "fam_gradient_tensor_counts": fam_gradient_counts,
        "checkpoint_roundtrip_status": "passed",
        "checkpoint_roundtrip_tensor_count": len(checkpoint_state_keys),
        "checkpoint_roundtrip_restored_alias_count": len(restored_aliases),
        "checkpoint_roundtrip_sha256": checkpoint_sha256,
        "control_transfer_report": control.pretrained_transfer_report,
        "candidate_transfer_report": candidate.pretrained_transfer_report,
        "runtime": runtime_fingerprint(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="out/rtdetr_v2_integration_audit/audit.json",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    result = run_audit(args.device)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
