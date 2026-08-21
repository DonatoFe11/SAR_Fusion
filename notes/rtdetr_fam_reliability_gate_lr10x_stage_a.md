# RT-DETR + FAM + reliability gate LR10x: Stage-A optimization ablation

Status: implementation and real probe complete; scientific seed 40 pending

Defined: 2026-08-21, before observing the complete seed-40 result

## Motivation and isolated variable

The first reliability-gate ablation did not improve FAM reproducibly over five
seeds (`+0.0016 +/- 0.0208` paired validation mAP@50, 3/5 wins, IC95%
`[-0.0242, +0.0274]`). More importantly, its validation-only audit found every
learned weight within `[0.99794, 1.00200]`: the gate remained almost exactly the
neutral additive fusion with which it was initialized.

This follow-up asks a narrower question: was the shared detector learning rate
too small for the 3,174 newly initialized gate parameters? The architecture,
initialization, P3--P5 features, FAM, data, modal dropout, optimizer, batch size,
ten-epoch budget and checkpoint selector are unchanged. The only scientific
variable is:

```text
detector, backbone and FAM LR: 2e-5 (unchanged)
reliability-gate LR:           2e-4 (10x)
```

The implementation isolates all parameters whose names contain
`reliability_gates` in a fourth AdamW parameter group. Models without a
reliability gate retain the historical three optimizer groups. W&B records the
fourth group as `train/lr_reliability_gate`.

## Frozen Stage-A protocol

- train: FHL 0405/0406 plus Baker VIS/IR 1, 3,123 paired frames;
- validation: the complete FHL 0401/0402 video, 896 paired frames;
- seed 40 only in this screening experiment;
- exactly ten epochs, no early stopping;
- direct training batch 4 and validation batch 12;
- `best`: highest validation mAP@50 with `min_delta = 0.001`;
- `latest`: retained at epoch 10 as a diagnostic;
- MtErie and automatic testing disabled.

The frozen FAM seed-40 reference is `0.1521` best validation mAP@50. The LR10x
variant advances to seeds 41--44 only if both conditions below hold:

1. **performance:** best validation mAP@50 is at least `0.1621`, a gain of at
   least `+0.01` over FAM seed 40;
2. **mechanism:** on the complete validation video, the audit finds a maximum
   `mean_abs_delta_one >= 0.01` and both absent-modality responses reach `0.01`:
   at some pyramid level `|mean w_RGB(fusion) - mean w_RGB(IR-only)| >= 0.01`,
   and at some level `|mean w_IR(fusion) - mean w_IR(RGB-only)| >= 0.01`.

These one-percentage-point thresholds distinguish substantive modulation from
the previous sub-`0.0021` numerical response. Failure of either condition
closes this optimization ablation. Passing both permits, but does not itself
establish, a performance claim; that still requires the remaining four seeds.
MtErie is not consulted for this decision.

## Why MtErie and full-data are excluded now

Stage A is the architecture and optimization filter. MtErie has already been
consulted repeatedly in this project and is therefore an internal development
benchmark, not an unused test set. Evaluating every weak variant on it would
allow a favourable test fluctuation to override a failed validation result.
Full-data training removes the held-out sequence needed for checkpoint and
model selection, so it is reserved for a configuration frozen after passing
Stage A. The closed neutral-gate experiment remains a valid negative ablation;
this LR10x run is a new, explicitly declared optimization ablation.

## Operational verification

The checkpoint-free W&B probe `8an9s9vz`, tagged `ExcludeFromCampaign`, passed:

- 12 gate parameter tensors isolated at LR `2e-4`;
- the other 485 trainable tensors retained LR `2e-5`;
- 20/20 batch-4 training steps completed;
- 75/75 validation batches completed in 74 seconds;
- W&B recorded `train/lr_reliability_gate = 0.0002`;
- exit code 0 and no campaign checkpoint produced.

Its validation mAP@50 (`0.0967`) is not a scientific result because it saw only
20 training batches.

## Launch and post-run audit

Launch the single complete scientific seed with:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python main.py experiment \
  --parameters parameters/RTDETR/rtdetr_fam_reliability_gate_lr10x_sequence_validation_seed40.yaml
```

Only if the performance threshold is reached, audit the selected checkpoint
before deciding on additional seeds:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/run_rtdetr_fam_reliability_gate_weight_audit.py \
  --protocol parameters/RTDETR/rtdetr_fam_reliability_gate_lr10x_weight_audit_seed40.yaml
```

## Artifacts and later thesis treatment

- training implementation: `sarfusion/experiment/run.py`;
- scientific configuration:
  `parameters/RTDETR/rtdetr_fam_reliability_gate_lr10x_sequence_validation_seed40.yaml`;
- runtime probe:
  `parameters/RTDETR/rtdetr_fam_reliability_gate_lr10x_runtime_probe.yaml`;
- frozen seed-40 weight audit:
  `parameters/RTDETR/rtdetr_fam_reliability_gate_lr10x_weight_audit_seed40.yaml`;
- regression tests: `tests/test_rtdetr_reliability_gating.py` and
  `tests/test_rtdetr_reliability_gate_weight_audit.py`.

The thesis source is intentionally unchanged. If reported, this belongs after
the neutral-gate result as an optimization ablation. It must not be presented
as a new architecture, because only a parameter-group learning rate changes.
