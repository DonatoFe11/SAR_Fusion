# RT-DETR + FAM at 800x800: Stage-A resolution ablation

Status: completed and closed; 800x800 does not improve the FAM baseline

Defined: 2026-08-22, before any 800x800 scientific run

## Question and motivation

This experiment tests whether increasing input resolution improves detection
of the very small people in FHL without adding P2 or another fusion module. It
retains the `current_dcnv2` FAM baseline and changes the RT-DETR processor from
`640x640` to `800x800`. P2 and reliability gating remain disabled.

The P2 experiment tested a stride-4 feature level and failed over five seeds;
it did not test whether simply preserving more input pixels helps the existing
P3--P5 detector. Resolution is therefore a distinct ablation.

## Frozen protocol

- five seeds `40--44`, run sequentially;
- same 3,123-frame training split and 896-frame FHL validation sequence;
- exactly ten epochs, no early stopping;
- `best` selected by validation mAP@50 with `min_delta = 0.001`;
- input size exactly `800x800`, verified at the processor output;
- training micro-batch 2 with two-step gradient accumulation, effective batch 4;
- validation batch 8;
- same AdamW optimizer and LR `2e-5`;
- no P2, reliability gate, tiling, MtErie or automatic final test.

The smaller micro-batch is required by the 8 GB GPU. A completed 640x640 FAM
control using micro-batch 2 and accumulation 2 obtained `0.1593` on seed 40,
versus `0.1521` for direct-batch-four FAM. This does not establish equivalence
over all seeds, but it provides no evidence that the necessary batching choice
itself causes the large deficits observed with P2.

## Decision rule

The primary comparison is paired against the existing five-seed FAM Stage-A
baseline (`0.1646 +/- 0.0196`). Before observing the new results:

- promote only if the mean paired mAP@50 gain is at least `+0.01` and positive
  in at least 4/5 seeds;
- treat a smaller positive mean or 3/5 wins as inconclusive, not as a claimed
  improvement;
- close if the mean delta is non-positive or fewer than 3/5 seeds improve.

The confidence interval and all `best/latest` pairs will also be reported, but
the interval will not replace the predeclared engineering threshold. MtErie is
not consulted during this decision.

## Launch

The checkpoint-free W&B probe `0tikq6eh`, tagged `ExcludeFromCampaign`, passed:

- production dataloader output `(2, 4, 800, 800)` and mask `(2, 800, 800)`;
- 20/20 training micro-batches with two-step accumulation;
- 112/112 validation batches at evaluation batch 8 in approximately 1:47;
- exit code 0, no OOM and no campaign checkpoint;
- architecture and all optimizer-group LRs identical to FAM.

Its validation mAP@50 (`0.1214`) is not a scientific result because the model
saw only 20 micro-batches. Launch all five scientific seeds with:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python main.py experiment \
  --parameters parameters/RTDETR/rtdetr_fam_800_sequence_validation_five_seed.yaml
```

## Completed five-seed result

All five runs completed exactly ten epochs, 15,620 training micro-batches per
run, input resolution `800x800`, accumulation 2, LR `2e-5`, and contain both
`best` and `latest` checkpoints. Each run required approximately 3 h 23 min.

| Seed | FAM 640 best | FAM 800 best | Selected epoch | FAM 800 latest | 800 - 640 |
|---:|---:|---:|---:|---:|---:|
| 40 | **0.1521** | 0.1251 | 1 | 0.0392 | -0.0270 |
| 41 | 0.1424 | **0.1438** | 7 | 0.1241 | +0.0015 |
| 42 | **0.1655** | 0.1506 | 2 | 0.0796 | -0.0149 |
| 43 | **0.1939** | 0.1380 | 2 | 0.0747 | -0.0559 |
| 44 | **0.1689** | 0.1659 | 6 | 0.0509 | -0.0030 |

FAM 800 obtains `0.1447 +/- 0.0151` best validation mAP@50, versus
`0.1646 +/- 0.0196` for FAM 640. The paired delta is
`-0.0199 +/- 0.0230`, positive in 1/5 seeds, with median `-0.0149` and a
two-sided Student-t 95% confidence interval of `[-0.0484, +0.0087]`.

The `latest` checkpoints obtain `0.0737 +/- 0.0328`. The paired
`best - latest` gain is `+0.0710 +/- 0.0348`, positive in 5/5 seeds, with IC95%
`[+0.0278, +0.1142]`. This again validates the common Stage-A checkpoint rule,
but does not change the resolution comparison because all models are compared
using `best`.

## Stage-A decision

The promotion rule required at least `+0.01` mean paired gain and 4/5 wins.
Observed values are `-0.0199` and 1/5, so both requirements fail. FAM 800 is
closed without MtErie evaluation or full-data Stage B.

The confidence interval crosses zero, so this experiment does not prove that
800x800 is universally harmful. It does show that the fixed 800x800 recipe
provides no evidence of improvement under the frozen development protocol.
Four losses, including a `-0.0559` loss on seed 43, outweigh the negligible
seed-41 win. The completed seed-40 640 micro-batch control (`0.1593`) also makes
the necessary batch-two configuration an implausible explanation for the
seed-40 high-resolution deficit (`0.1251`), although that control was not run
on all five seeds.

No further resolution value is selected post hoc on this same validation
sequence. A new direction must pose a different, predeclared hypothesis.

## Thesis treatment

The thesis source is intentionally unchanged. This experiment should be
reported as a resolution ablation motivated by tiny targets, not as a new
architecture. Its value is to distinguish lack of spatial input detail from
limitations of P2 and reliability gating. The conclusion is that naive global
upsampling does not solve the tiny-target problem for this detector; it is not
that higher resolution can never help under another training recipe.

## Artifacts

- five-seed configuration:
  `parameters/RTDETR/rtdetr_fam_800_sequence_validation_five_seed.yaml`;
- checkpoint-free probe:
  `parameters/RTDETR/rtdetr_fam_800_runtime_probe.yaml`;
- compact five-seed results:
  `notes/Search_and_Rescue/results/rtdetr_fam_800_stage_a_validation.csv`;
- regression tests: `tests/test_rtdetr_fam_high_resolution.py`.
