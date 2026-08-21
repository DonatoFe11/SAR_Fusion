# RT-DETR + FAM + reliability gating: Stage-A ablation

Status: completed and closed; no supported mean improvement over FAM, and the
learned gate remained functionally neutral

Defined: 2026-08-20

## Question under test

This ablation asks whether explicitly estimating local modality reliability
improves the existing RT-DETR + FAM detector. The P3--P5 pyramid, FAM alignment,
input resolution, optimizer, data split and direct training batch of four remain
unchanged. P2 is disabled. No new alignment mechanism is introduced, so the
only architectural variable is the reliability gate applied after FAM.

This is a project-specific candidate architecture. Any claim of broader
architectural novelty requires a dedicated literature comparison and is not
assumed by this experiment.

## Reliability-gated fusion

For each of P3, P4 and P5, FAM first produces the aligned IR feature `I_a` from
RGB feature `R` and raw IR feature `I`. A lightweight gate then constructs seven
spatial descriptors:

1. channel mean of `R`;
2. channel RMS of `R`;
3. channel mean of `I_a`;
4. channel RMS of `I_a`;
5. channelwise cosine agreement between `R` and `I_a`;
6. binary RGB-presence indicator;
7. binary IR-presence indicator.

The presence indicators are derived from the exact zero padding already used
by RT-DETR modal dropout. They are inputs to the learned gate, not hard-coded
output masks.

The seven descriptors pass through a 3x3 convolution with 16 hidden channels,
SiLU and a 1x1 convolution producing two spatial logits. Fusion is:

```text
w_R = 2 sigmoid(l_R)
w_I = 2 sigmoid(l_I)
F   = w_R * R + w_I * I_a
```

The modality weights are independent and lie in `(0, 2)`, so either modality
can be suppressed without forcing amplification of the other. The final 1x1
convolution is initialized to exactly zero; therefore `w_R = w_I = 1` and the
initial output is exactly the previous additive FAM fusion. This initialization
remains restored after Hugging Face `post_init` and pretrained transfer.

Each gate has 1,058 trainable parameters. Three levels add 3,174 parameters in
total. The gate uses the same `2e-5` learning rate as the rest of this frozen
protocol.

## Frozen Stage-A protocol and decision rule

- train: FHL 0405/0406 plus Baker VIS/IR 1, 3,123 paired frames;
- validation: complete FHL 0401/0402 video, 896 paired frames;
- ten epochs exactly, without early stopping;
- direct batch size 4, identical to the FAM baseline;
- validation batch size 12;
- primary checkpoint: highest validation mAP@50 with `min_delta = 0.001`;
- MtErie disabled;
- first run: seed 40 only.

The comparison target is FAM seed 40, `0.1521` best validation mAP@50. Before
seeing the gating result, the engineering triage is frozen as:

- gate at least `+0.01` (at least `0.1621`): expand to seeds 41--44;
- gate within `-0.01` to `+0.01` (`0.1421`--`0.1621`): run seed 41 before
  deciding;
- gate below `-0.01` (below `0.1421`): do not expand; inspect optimization and
  learned gate weights first.

This rule controls compute allocation. A scientific performance claim still
requires five seeds. MtErie must not be consulted to decide whether the gate
advances.

## Seed-40 result and expansion decision

The complete seed-40 run `h18ch26u` finished all ten epochs. Its selected
checkpoint is epoch 2.

| Seed-40 configuration | Best validation mAP@50 | Epoch-10 mAP@50 |
|---|---:|---:|
| FAM baseline | 0.1521 | 0.0397 |
| FAM + reliability gate | **0.1662** | 0.1022 |

The gate improves the primary validation score by `+0.0141`, exceeding the
predeclared `+0.01` expansion threshold. Seeds 41--44 must therefore be run
under the identical protocol. The epoch-10 decline is not used to alter the
fixed ten-epoch budget; it reinforces the predefined use of `best` for Stage A.
MtErie remains unconsulted.

## Completed five-seed result

All five runs completed ten epochs and contain both `best` and `latest`
checkpoints. The primary comparison uses the predeclared `best` selector.

| Seed | FAM best | Gate best | Gate latest | Gate - FAM |
|---:|---:|---:|---:|---:|
| 40 | 0.1521 | 0.1662 | 0.1022 | +0.0141 |
| 41 | 0.1424 | 0.1568 | 0.0884 | +0.0145 |
| 42 | 0.1655 | 0.1794 | 0.1284 | +0.0140 |
| 43 | 0.1939 | 0.1602 | 0.0786 | -0.0337 |
| 44 | 0.1689 | 0.1681 | 0.0918 | -0.0008 |

FAM obtains `0.1646 +/- 0.0196`; reliability gating obtains
`0.1662 +/- 0.0087`. The paired delta is `+0.0016 +/- 0.0208`, positive in 3/5
seeds, with a two-sided Student-t 95% confidence interval of
`[-0.0242, +0.0274]`. The median paired delta is `+0.0140`.

The five-seed experiment therefore does not support a mean performance
improvement. The smaller observed standard deviation is potentially useful but
is exploratory: five seeds are insufficient to claim a variance reduction, and
the primary paired interval includes both meaningful harm and benefit. The gate
is not promoted to MtErie or full-data Stage B on this evidence.

Before closure, a validation-only audit was frozen to verify whether the learned
weights actually departed from one and responded coherently to fusion, RGB-only
and IR-only inputs. The audit diagnoses mechanism behavior; it cannot rescue
the unsupported detection-performance claim.

## Learned-weight audit

The five `best` checkpoints were evaluated on all 896 validation pairs in three
conditions: fusion, RGB-only and IR-only. The audit captured both modality
weights at P3, P4 and P5 for every spatial position. All checkpoints loaded with
zero missing and zero unexpected state-dict keys.

The result is a mechanistic null:

- global weight range across seeds, levels and modes: `0.99794`--`1.00200`;
- largest mean absolute deviation from neutral weight 1: `0.00193`;
- largest within-condition spatial standard deviation: `0.00013`;
- fraction of weights below `0.9` or above `1.1`: zero in all 90 audit rows;
- largest mean-weight span between fusion, RGB-only and IR-only: `0.00206`.

The completed audit JSON has SHA-256
`fb95e3a52b00438acf3175f53dac7afb81b5e33030c41fc9d720ef08293f6824`;
the CSV contains one header plus 90 seed/mode/level/modality rows.

The output heads did leave their exact zero initialization, but only enough to
produce sub-percent, nearly spatially constant changes. The gate therefore did
not learn substantive reliability modulation and did not react meaningfully to
an absent modality. The small five-seed mean detection delta cannot be
interpreted as evidence that reliability gating worked.

## Stage-A decision

This gate version is closed. It is not evaluated on MtErie and is not promoted
to full-data Stage B. It remains useful as a negative ablation: neutral
initialization protected the baseline, but the shared `2e-5` learning rate left
the 3,174 new parameters too close to identity to test strong reliability
adaptation. A future follow-up must be declared as a separate optimization
ablation, for example a gate-specific learning rate, rather than being folded
into this result post hoc.

That follow-up was frozen before execution as the separate LR10x Stage-A
ablation documented in `notes/rtdetr_fam_reliability_gate_lr10x_stage_a.md`.

## Verification

Automated tests cover:

- exact additive equivalence at neutral initialization;
- finite gradients and the ability to leave the neutral solution;
- explicit response to modality-presence descriptors;
- three P3--P5 gates and rejection of gating without FAM;
- propagation through the full detector and public model factory;
- unchanged split, batch size, optimizer budget and disabled MtErie;
- a checkpoint-free runtime-probe configuration excluded from the campaign.

The complete repository suite passes 118/118 tests.

The real operational probe is W&B run `6r7eq72h`, tagged
`ExcludeFromCampaign`. It completed:

- 20/20 real batch-4 training iterations in approximately 22 s;
- 75/75 validation batches in 1:13;
- stable late-validation throughput near 0.8 s/batch;
- exit code 0;
- no `best`, `latest`, safetensors or PyTorch checkpoint file.

Its validation mAP is not a scientific result because the model received only
20 training batches.

## Launch

The completed seed 40 was launched with:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python main.py experiment \
  --parameters parameters/RTDETR/rtdetr_fam_reliability_gate_sequence_validation_seed40.yaml
```

After checking seed 40 against the frozen rule, launch only seeds 41--44 from
the versioned five-seed grid with:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python main.py experiment \
  --parameters parameters/RTDETR/rtdetr_fam_reliability_gate_sequence_validation_five_seed.yaml \
  --start-from-run 1
```

Do not run MtErie until all five validation results have been aggregated.

## Artifacts and later thesis changes

- Implementation: `sarfusion/models/rtdetr_fusion.py`
- Public parameter plumbing: `sarfusion/models/detr.py` and
  `sarfusion/models/__init__.py`
- Stage-A configuration:
  `parameters/RTDETR/rtdetr_fam_reliability_gate_sequence_validation_seed40.yaml`
- Five-seed expansion grid:
  `parameters/RTDETR/rtdetr_fam_reliability_gate_sequence_validation_five_seed.yaml`
- Five-seed validation results:
  `notes/Search_and_Rescue/results/rtdetr_fam_reliability_gate_stage_a_validation.csv`
- Learned-weight audit protocol:
  `parameters/RTDETR/rtdetr_fam_reliability_gate_weight_audit.yaml`
- Learned-weight audit runner:
  `scripts/run_rtdetr_fam_reliability_gate_weight_audit.py`
- Learned-weight audit results:
  `notes/Search_and_Rescue/results/rtdetr_fam_reliability_gate_weight_audit.json`
  and `notes/Search_and_Rescue/results/rtdetr_fam_reliability_gate_weight_audit.csv`
- Audit regression tests:
  `tests/test_rtdetr_reliability_gate_weight_audit.py`
- Operational probe:
  `parameters/RTDETR/rtdetr_fam_reliability_gate_runtime_probe.yaml`
- Regression tests: `tests/test_rtdetr_reliability_gating.py`

The thesis source is intentionally unchanged. The methodology should report the
gate descriptors, neutral initialization and controlled P3--P5 comparison. The
evaluation should report the inconclusive five-seed detection delta together
with the mechanistic audit, explicitly stating that this version remained near
identity and did not provide evidence of reliability-aware behavior.
