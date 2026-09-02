# RT-DETR + FAM + reliability gate LR10x: Stage-A optimization ablation

Status: completed and closed; gate activation increased, but the five-seed
experiment provides no evidence of a detection improvement

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

## Seed-40 result and expansion decision

The complete seed-40 run `2d54ipjg` finished all ten epochs in approximately
2 h 13 min. The configured and W&B-recorded learning rates are `2e-5` for
backbone, FAM and detector groups and `2e-4` for the reliability gate.

| Seed-40 configuration | Best validation mAP@50 | Selected epoch | Epoch-10 mAP@50 |
|---|---:|---:|---:|
| FAM baseline | 0.1521 | 1 | 0.0397 |
| gate, shared LR `2e-5` | 0.1662 | 2 | 0.1022 |
| gate, dedicated LR `2e-4` | **0.1747** | 1 | 0.0978 |

LR10x is `+0.0226` above the FAM seed-40 reference, exceeding the frozen
performance threshold of `0.1621`. Its best checkpoint was then audited on all
896 validation pairs in fusion, RGB-only and IR-only conditions. It loaded with
zero missing and zero unexpected state-dict keys.

| Frozen mechanism check | Observed maximum | Required | Result |
|---|---:|---:|---|
| Mean absolute weight deviation from 1 | 0.03776 | >= 0.01 | pass |
| RGB-weight response when RGB is absent | 0.02046 | >= 0.01 | pass |
| IR-weight response when IR is absent | 0.01822 | >= 0.01 | pass |

The largest missing-RGB response occurs at P3 between fusion and IR-only; the
largest missing-IR response also occurs at P3 between fusion and RGB-only. The
gate therefore moved materially farther from identity and reacted to both
presence indicators, unlike the shared-LR version. The modulation is still
mostly level-wise rather than strongly spatial: within-condition standard
deviations remain at most `0.00426`. This is a mechanistic observation, not yet
a multi-seed performance conclusion.

The raw audit JSON has SHA-256
`8d4af0b9c4914979085b94cf497dfe207a6912c5072a432f5d972ae4b8586034`;
the 18-row CSV has SHA-256
`34daa976b90e4f0c3542be23ccaac92d02f71c56349bb51bbcb10c33f0c3b023`.

Both predeclared checks pass, so seeds 41--44 are authorized under the identical
protocol. MtErie remains unconsulted.

## Completed five-seed result

All five runs completed exactly ten epochs with the configured gate LR `2e-4`
and contain both `best` and `latest` checkpoints. The primary comparison uses
the predefined validation-selected `best` checkpoint.

| Seed | FAM best | Shared-LR gate best | LR10x best | LR10x epoch | LR10x latest | LR10x - FAM |
|---:|---:|---:|---:|---:|---:|---:|
| 40 | 0.1521 | 0.1662 | **0.1747** | 1 | 0.0978 | +0.0226 |
| 41 | 0.1424 | **0.1568** | 0.1431 | 9 | 0.1336 | +0.0008 |
| 42 | 0.1655 | **0.1794** | 0.1771 | 2 | 0.1184 | +0.0116 |
| 43 | **0.1939** | 0.1602 | 0.1329 | 1 | 0.0893 | -0.0610 |
| 44 | **0.1689** | 0.1681 | 0.1453 | 4 | 0.0644 | -0.0236 |

Across seeds, FAM obtains `0.1646 +/- 0.0196`, the shared-LR gate obtains
`0.1662 +/- 0.0087`, and LR10x obtains `0.1546 +/- 0.0200` best validation
mAP@50. Relative to FAM, the LR10x paired delta is
`-0.0099 +/- 0.0333`, positive in 3/5 seeds, with a two-sided Student-t 95%
confidence interval of `[-0.0512, +0.0314]`. Its median paired delta is
`+0.0008`.

Relative to the same gate trained at the shared LR, LR10x is
`-0.0115 +/- 0.0147`, positive in only 1/5 seeds, with IC95%
`[-0.0298, +0.0067]`. Thus increasing gate activation did not translate into a
repeatable detection improvement and nominally reduced the mean score.

LR10x `latest` obtains `0.1007 +/- 0.0267`. The paired `best - latest` gain is
`+0.0539 +/- 0.0290`, positive in 5/5 seeds, with IC95%
`[+0.0180, +0.0899]`. This supports the existing Stage-A checkpoint selector;
it does not rescue the architecture comparison, which consistently uses
`best` for every model.

## Stage-A decision

The LR10x optimization ablation is closed. It is not promoted to MtErie or
full-data Stage B. The seed-40 audit already answered its intended mechanistic
question: a dedicated LR can move the gate away from identity and make it
respond to missing-modality indicators. The five-seed result answers the
primary detection question without supporting an improvement. It does not
prove that LR10x is intrinsically worse: the paired confidence interval
contains both benefit and harm. A fifteen-pass five-seed weight audit cannot
change the promotion decision and is therefore not run solely to search for a
favourable secondary explanation.

This is an informative negative result: insufficient gate optimization
explains the neutral weights of the first version, but not its lack of a robust
accuracy gain. Further tuning of this gate's LR on the same validation sequence
would increase selection bias. Before another architectural experiment, its
hypothesis must be frozen separately. The remaining defensible directions are
a higher-resolution FAM control for the tiny-object hypothesis, or a
reliability-conditioned alignment mechanism that is distinct from the already
completed DCNv2, identity, grid-sample and bounded-offset FAM variants. Another
post-hoc gate-LR value is excluded.

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

The audit passed. Launch the remaining four seeds without repeating seed 40:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python main.py experiment \
  --parameters parameters/RTDETR/rtdetr_fam_reliability_gate_lr10x_sequence_validation_five_seed.yaml \
  --start-from-run 1
```

This expansion completed on 2026-08-22. No further command from this section
remains to be run.

## Artifacts and later thesis treatment

- training implementation: `sarfusion/experiment/run.py`;
- scientific configuration:
  `parameters/RTDETR/rtdetr_fam_reliability_gate_lr10x_sequence_validation_seed40.yaml`;
- five-seed expansion:
  `parameters/RTDETR/rtdetr_fam_reliability_gate_lr10x_sequence_validation_five_seed.yaml`;
- runtime probe:
  `parameters/RTDETR/rtdetr_fam_reliability_gate_lr10x_runtime_probe.yaml`;
- frozen seed-40 weight audit:
  `parameters/RTDETR/rtdetr_fam_reliability_gate_lr10x_weight_audit_seed40.yaml`;
- seed-40 audit outputs:
  `notes/Search_and_Rescue/results/rtdetr_fam_reliability_gate_lr10x_weight_audit_seed40.json`
  and `notes/Search_and_Rescue/results/rtdetr_fam_reliability_gate_lr10x_weight_audit_seed40.csv`;
- five-seed validation table:
  `notes/Search_and_Rescue/results/rtdetr_fam_reliability_gate_lr10x_stage_a_validation.csv`;
- regression tests: `tests/test_rtdetr_reliability_gating.py` and
  `tests/test_rtdetr_reliability_gate_weight_audit.py`.

The thesis source is intentionally unchanged. This belongs after the
neutral-gate result as an optimization ablation and must not be presented as a
new architecture, because only a parameter-group learning rate changes. Its
main value is the contrast between a successful mechanistic intervention and
an unsupported detection benefit: stronger non-neutral gating is not by itself
evidence of a better detector. The five-seed result, not seed 40, controls the
conclusion.
