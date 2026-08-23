# RT-DETR + FAM + scalar residual alignment: Stage-A attribution control

Status: implemented and frozen; runtime probe passed, scientific campaign not
yet trained

Defined: 2026-08-23, after completing RCRA and before observing any scalar
control result

## Scientific question

RCRA is the first new candidate to pass the frozen Stage-A rule: its paired
validation gain over FAM is `+0.0179`, positive in 4/5 seeds. Its alpha audit
also shows an active mechanism. However, the learned fusion behavior is not
uniformly spatial:

- P3 has appreciable within-condition spatial variation;
- P4 behaves mainly as a level-wise suppression of the FAM residual;
- P5 remains close to the original FAM.

This creates an attribution question that must be answered before final
full-data training: **does the improvement require RCRA's local reliability
descriptors, or is a much simpler learned scale for each FAM level sufficient?**

The present model is a negative/positive control for that question. It is not a
new candidate invented after seeing a weak seed and is not another learning-rate
search.

## Frozen architecture

For each level `l` in P3--P5, the control learns one scalar logit `z_l`:

```text
alpha_l    = 2 * sigmoid(z_l)
I_selected = I_aligned + (alpha_l - 1) * (I_aligned - I_raw)
F_fused    = F_RGB + I_selected
```

There are exactly three new parameters in the entire detector. Each alpha is:

- constant over images, modalities, channels and spatial locations;
- bounded to `(0, 2)`;
- exactly one at initialization because `z_l = 0`;
- able to bypass FAM near zero or amplify its residual above one.

The finite-precision expression is the same as RCRA, so `alpha_l = 1` returns
`I_aligned` exactly. The control receives the same training signal and uses the
same residual definition, but cannot inspect RGB, raw IR, aligned IR, agreement,
modality presence or spatial position.

RCRA's 5,283 parameters are replaced by these three scalars. P2 and the old
post-fusion reliability gate remain disabled. The implementation rejects any
attempt to enable more than one of RCRA, scalar alignment and post-fusion
gating together.

## Frozen optimization and Stage-A protocol

The scalar parameters use the same dedicated AdamW group and LR as RCRA:

```text
detector, backbones, FAM, encoder and decoder: 2e-5
three scalar residual logits:                 2e-4
```

Everything else is copied from the RCRA campaign:

- seeds `40--44`;
- train on 3,123 paired FHL 0405/0406 and Baker frames;
- validate on the complete 896-pair FHL 0401/0402 video;
- P3--P5 and input 640;
- direct batch 4, validation batch 12;
- modal dropout `[0.2, 0.2, 0.6]`;
- exactly ten epochs without early stopping;
- `best` selected by validation mAP@50 with `min_delta = 0.001`;
- `latest` retained only as diagnostic;
- no automatic test, MtErie or full-data training.

At initialization, FAM, RCRA and the scalar control implement the same aligned
IR fusion. The scientific variable is whether alpha is constant per level or
predicted locally from reliability descriptors.

## Frozen comparisons and selection rule

Two paired comparisons will be reported:

1. scalar control minus FAM, to determine whether simple residual calibration
   itself passes the existing Stage-A rule;
2. RCRA minus scalar control, to attribute any additional value to conditional
   local prediction.

The existing promotion threshold remains a mean gain of at least `+0.01` and
at least 4/5 paired wins. RCRA has already passed it. Selection is frozen as
follows:

- if the scalar control does not pass the FAM threshold, retain RCRA;
- if the scalar control passes and RCRA exceeds it by at least `+0.01` with
  4/5 wins, select RCRA and treat this as evidence for conditional prediction;
- if the scalar control passes and exceeds RCRA by the same margin and win
  rule, select the scalar model;
- if the scalar control passes but neither wins their direct comparison by
  that margin, prefer the three-parameter scalar control for Stage B on
  parsimony grounds and state that RCRA's additional spatial/reliability
  complexity is unsupported.

A later control does not erase RCRA's data, but model selection must follow the
predeclared evidence and the simpler model wins an inconclusive direct tie.
For every comparison report mean, sample standard deviation, median, paired
95% t interval and number of wins. The interval informs uncertainty but does
not replace the engineering rule.

The three learned alpha values will be extracted from every `best` checkpoint.
This audit requires no validation inference because the coefficients are
input-independent. It is diagnostic and cannot override the performance rule.

## Commands

The checkpoint-free operational probe completed successfully:

- W&B run `3x1v3qfk`, tagged `ExcludeFromCampaign`;
- exactly three scalar parameter tensors isolated at LR `2e-4`;
- all other optimizer groups retained LR `2e-5`;
- 20/20 direct batch-4 training steps completed;
- 75/75 validation batches completed in approximately 70 seconds;
- exit code 0, no OOM and no scientific checkpoint.

Its validation mAP@50 (`0.0970`) is not a scientific result because it follows
only 20 training batches. The command used was:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python main.py experiment \
  --parameters parameters/RTDETR/rtdetr_fam_scalar_alignment_control_runtime_probe.yaml
```

The probe passed. Launch all five scientific seeds with:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python main.py experiment \
  --parameters parameters/RTDETR/rtdetr_fam_scalar_alignment_control_sequence_validation_five_seed.yaml
```

After training, extract the selected scalars with:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/run_rtdetr_fam_scalar_alignment_control_audit.py \
  --protocol parameters/RTDETR/rtdetr_fam_scalar_alignment_control_audit.yaml
```

No probe command remains to be run.

## Results placeholder

No scalar-control performance or learned-alpha result exists at definition
time. Populate this section only after five complete runs.

## Thesis treatment

This is a required architectural ablation, not the main proposed method. It
will determine which claim is defensible:

- RCRA wins clearly: local reliability-conditioned residual selection adds
  value beyond simple FAM rescaling;
- scalar control is equivalent or better: the useful contribution is residual
  calibration, and RCRA's extra complexity should not be credited;
- neither is stable: report the uncertainty and retain FAM as final model.

The thesis source remains intentionally unchanged until this attribution test
and Stage B are complete.

## Versioned artifacts

- implementation: `sarfusion/models/rtdetr_fusion.py`;
- public parameter plumbing: `sarfusion/models/detr.py` and
  `sarfusion/models/__init__.py`;
- five-seed protocol:
  `parameters/RTDETR/rtdetr_fam_scalar_alignment_control_sequence_validation_five_seed.yaml`;
- runtime probe:
  `parameters/RTDETR/rtdetr_fam_scalar_alignment_control_runtime_probe.yaml`;
- frozen scalar audit:
  `parameters/RTDETR/rtdetr_fam_scalar_alignment_control_audit.yaml` and
  `scripts/run_rtdetr_fam_scalar_alignment_control_audit.py`;
- regression tests: `tests/test_rtdetr_scalar_alignment_control.py`.
