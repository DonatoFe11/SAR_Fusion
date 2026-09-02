# RT-DETR + FAM + scalar residual alignment: Stage-A attribution control

Status: completed; the scalar control fails the frozen FAM-promotion rule and
RCRA is retained for Stage B

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

The probe passed. All five scientific seeds were launched with:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python main.py experiment \
  --parameters parameters/RTDETR/rtdetr_fam_scalar_alignment_control_sequence_validation_five_seed.yaml
```

After training, the selected scalars were extracted with:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/run_rtdetr_fam_scalar_alignment_control_audit.py \
  --protocol parameters/RTDETR/rtdetr_fam_scalar_alignment_control_audit.yaml
```

Both commands completed successfully. No command in this section remains to be
run.

## Completed five-seed performance result

All five runs completed exactly ten epochs, recorded the dedicated scalar LR
`train/lr_alignment_gate = 2e-4`, and contain both `best` and `latest`
checkpoints. The primary comparison uses the predefined validation-selected
`best` checkpoint.

| Seed | FAM best | RCRA best | Scalar best (epoch) | Scalar latest | Scalar - FAM | RCRA - scalar |
|---:|---:|---:|---:|---:|---:|---:|
| 40 | 0.1521 | 0.1692 | **0.1750 (1)** | 0.0107 | +0.0228 | -0.0057 |
| 41 | 0.1424 | **0.2243** | 0.1502 (3) | 0.1158 | +0.0078 | +0.0741 |
| 42 | 0.1655 | **0.1848** | 0.1573 (5) | 0.0774 | -0.0081 | +0.0275 |
| 43 | **0.1939** | 0.1562 | 0.1721 (3) | 0.0979 | -0.0218 | -0.0159 |
| 44 | 0.1689 | **0.1778** | 0.1470 (5) | 0.1318 | -0.0219 | +0.0309 |

The scalar control obtains `0.1603 +/- 0.0127` best validation mAP@50. Its
paired scalar-minus-FAM delta is `-0.0043 +/- 0.0195`, positive in only 2/5
seeds, with median `-0.0081` and a two-sided Student-t 95% confidence interval
of `[-0.0284, +0.0199]`. It fails both parts of the frozen promotion rule: the
mean is below `+0.01` and there are fewer than 4/5 wins.

RCRA exceeds the scalar control by `+0.0222 +/- 0.0355` on average, with a
median paired delta of `+0.0275`, 3/5 wins and IC95%
`[-0.0218, +0.0662]`. This direct comparison is uncertain and does not by
itself meet the stronger 4/5-win rule. The predeclared selection rule does not
require that secondary condition here: because the scalar model fails its FAM
threshold while RCRA already passed it, the first frozen branch retains RCRA.

Scalar `latest` obtains `0.0867 +/- 0.0471`. Its paired `best - latest` gain is
`+0.0736 +/- 0.0575`, positive in 5/5 seeds, with median `+0.0742` and IC95%
`[+0.0022, +0.1450]`. This is another independent indication that Stage-A
checkpoint selection cannot be replaced by epoch-10 `latest`.

## Completed scalar-parameter audit

Every selected checkpoint loaded with zero missing and zero unexpected keys.
The control has no input-dependent behavior, so its complete audit consists of
the three learned coefficients from each `best` checkpoint:

| Seed | P3 alpha | P4 alpha | P5 alpha |
|---:|---:|---:|---:|
| 40 | 0.9757 | 0.9909 | 0.9988 |
| 41 | 0.9520 | 0.9718 | 0.9952 |
| 42 | 0.9469 | 0.9687 | 0.9961 |
| 43 | 0.9633 | 0.9814 | 0.9967 |
| 44 | 0.9502 | 0.9810 | 0.9883 |

Across seeds, mean alpha is `0.9576` at P3, `0.9788` at P4 and `0.9950` at
P5. Mean absolute distance from the neutral value is respectively `0.0424`,
`0.0212` and `0.0050`. Thus optimization did move the control consistently:
it learned a modest global suppression of the FAM residual, strongest at P3,
rather than remaining at its identity initialization. Its failed promotion is
therefore not explained by an inactive three-parameter module.

The audit JSON has SHA-256
`c5e4dad9d396c839bb23eb024c6933fe80c1935d91b39abbdb7f2dff78331063`;
the 15-row audit CSV has SHA-256
`58641be66faa97764cd1b8eaf493a1d95a4c87ecef3a9d2ca62192999e6418f1`.
The compact performance CSV has SHA-256
`e07a54324cf21abf9d353722215276c640680d9c4e5bc4ca0020f38943ef44e1`.

## Frozen decision

The result follows the first branch of the rule declared before training: the
scalar control does not pass the comparison with FAM, so RCRA is retained for
Stage B. The subsequent matched full-data protocol is frozen in
`notes/rtdetr_fam_rcra_full_data_stage_b.md`; MtErie was not used to define it.

This control rules out a particularly simple explanation of the RCRA result:
three input-independent per-level residual scales do not reproduce its
Stage-A promotion. It is consistent with conditional/local prediction being
useful, but the claim must remain calibrated. RCRA beats the scalar model in
only 3/5 paired seeds and their direct IC95% crosses zero, so these five runs do
not establish statistical superiority or prove that every RCRA descriptor is
necessary. They justify selecting RCRA under the frozen engineering rule and
reporting the scalar experiment as an attribution ablation.

## Thesis treatment

This is an architectural attribution ablation, not the main proposed method.
The result should be reported as evidence that input-independent per-level
calibration is insufficient under the fixed Stage-A protocol. It supports
carrying RCRA forward, while the uncertain direct comparison prevents a claim
that local conditioning has been statistically proven superior.

Stage B is now complete and retains matched FAM because RCRA's positive mean
gain was observed in only 3/5 seeds. The thesis source remains intentionally
unchanged in this result commit so the method and evaluation can be rewritten
once using the complete frozen result chain.

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
- compact performance table:
  `notes/Search_and_Rescue/results/rtdetr_fam_scalar_alignment_control_stage_a_validation.csv`;
- scalar audit outputs:
  `notes/Search_and_Rescue/results/rtdetr_fam_scalar_alignment_control_audit.json`
  and
  `notes/Search_and_Rescue/results/rtdetr_fam_scalar_alignment_control_audit.csv`;
- regression tests: `tests/test_rtdetr_scalar_alignment_control.py`.
