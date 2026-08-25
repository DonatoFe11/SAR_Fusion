# RT-DETR + FAM versus RCRA: frozen full-data Stage B

Status: completed; RCRA improves the mean but wins only 3/5 paired seeds, so it
fails the frozen confirmation rule and matched FAM is retained

Frozen: 2026-08-24, after the Stage-A RCRA and scalar-control decisions and
before observing any full-data RCRA checkpoint or new MtErie result

## Purpose

Stage A selected RCRA under a video-level development split. RCRA obtained
`0.1825 +/- 0.0257` validation mAP@50 versus `0.1646 +/- 0.0196` for FAM, with
a paired mean gain of `+0.0179` and 4/5 wins. A later three-parameter scalar
control failed to improve FAM, so the predeclared attribution rule retained
RCRA. These results select a candidate but do not show how it behaves when the
held-out FHL sequence is restored to optimization.

Stage B therefore answers one final question: **does the selected RCRA recipe
retain a practically meaningful advantage over a configuration-matched FAM
baseline when both use the complete historical 4,019-frame training set and
the historical fixed-ten-epoch `latest` rule?**

This is a confirmation stage, not another architecture search. P2, resolution,
descriptors, hidden width, learning rates, modal dropout and training length
may not be changed after observing Stage-B results.

## Why FAM is retrained

The historical `RTDETR_FAM_Protocol` campaign is an important secondary
reference and obtained approximately `0.3780 +/- 0.0439` MtErie mAP@50. Its
serialized configuration confirms the intended recipe:

- all 4,019 paired training frames;
- ten epochs and direct batch four;
- AdamW with `2e-5` for the existing detector;
- `current_dcnv2` FAM;
- modal dropout probabilities `[0.2, 0.2, 0.6]`;
- epoch-10 `latest`;
- seeds 40--44.

It was nevertheless trained at Git commit
`8dba641128d647408c21e9e41cc4e65c0f73f722`. Since then the RT-DETR wrapper,
optimizer partition, data plumbing and model implementation have changed to
support the corrected evaluation and new ablations. Even if the baseline
forward path is intended to remain equivalent, reusing an old-code baseline
as the only comparator would leave a preventable implementation confound.

Consequently Stage B retrains FAM with the current code. Historical FAM remains
a sanity check on the absolute scale of the new baseline, but it cannot replace
the matched paired comparison.

## Frozen training recipe

The two configurations differ only in the declared RCRA candidate recipe:

| Property | Matched FAM | RCRA |
|---|---:|---:|
| `use_residual_alignment_gating` | false | true |
| residual-alignment parameters | 0 | 5,283 |
| alignment-gate LR | absent | `2e-4` |
| all existing detector/FAM LRs | `2e-5` | `2e-5` |

Everything else is identical:

- seeds `40, 41, 42, 43, 44` paired by seed;
- train on FHL 0401/0402, FHL 0405/0406 and Baker VIS/IR 1;
- exactly 4,019 paired frames, frozen source-inventory SHA-256
  `5f684ff047c2fd41e5e6440daa4e5ca61a89b34d907f6f156638bb7f740ca8eb`;
- exactly ten complete epochs and 10,050 optimizer steps;
- direct batch 4, no gradient accumulation;
- input 640 and P3--P5 only;
- `current_dcnv2` FAM, unfrozen;
- modal dropout enabled with `[0.2, 0.2, 0.6]`;
- pretrained one-class head reuse enabled;
- no validation, early stopping, checkpoint metric or truncated epoch;
- save only epoch-10 `latest`;
- no automatic MtErie test during training.

The official FHL 0134/0135 pair remains configured as the dataset's validation
phase only to preserve the historical split interface; `run_validation=false`
means it neither selects a checkpoint nor affects optimization.

No new runtime probe is required. Both architectures have already completed
five ten-epoch Stage-A runs at direct batch four. Restoring more training
frames changes the number of batches, not tensor shapes or peak model memory;
disabling validation reduces rather than increases the runtime surface.

## Frozen evaluation and decision rule

After all ten runs existed, the evaluator resolved exactly one `latest`
checkpoint for every configuration and seed. It rejected a run unless its local
W&B config matches the frozen grid, it reached epoch 10 and 10,050 steps, its
logged LRs are correct, and it contains no automatic `test/*` metrics.

Both models are evaluated with:

- the same 708 paired MtErie frames;
- the same VIS annotations: 1,770 boxes and 19 empty frames;
- full VIS+IR input without modality masking;
- confidence threshold `0.01`;
- mAP@50 as the primary metric;
- epoch-10 `latest` only.

The primary paired delta is `RCRA - matched FAM` for equal seeds. RCRA is
confirmed as the final architecture only if both predeclared conditions hold:

1. mean paired mAP@50 gain at least `+0.01`;
2. positive delta in at least 4/5 seeds.

Otherwise FAM remains the final performance baseline. Mean, sample standard
deviation, median, paired Student-t IC95%, wins and exploratory paired tests
will be reported regardless. The interval communicates uncertainty but cannot
override the engineering rule. No seed, threshold or checkpoint may be chosen
from MtErie, and Stage B will not be followed by another architecture rescue.

MtErie has already been consulted repeatedly in this project. The thesis must
therefore call it an internal development benchmark, not a fresh blind test.

## Completed preflight

The checkpoint-free prepare-only validation completed successfully:

- each YAML expands to exactly five runs mapped to seeds 40--44;
- after neutralizing the declared RCRA gate and its dedicated LR, all
  scientific parameters match exactly;
- the locally verified training inventory contains 4,019 frames with the
  frozen hash above;
- the evaluation inventory contains 708 frames with SHA-256
  `d6ceca387baaa22b63c31d762c2cb636b52e78acdaa9d8c33e668948a0718900`;
- no checkpoint was loaded and no inference result was observed.

The preflight command is:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/run_rtdetr_fam_rcra_full_data_stage_b_evaluation.py \
  --prepare-only
```

## Completed training and evaluation commands

The matched FAM campaign was run with:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python main.py experiment \
  --parameters parameters/RTDETR/rtdetr_fam_full_data_stage_b_five_seed.yaml
```

The RCRA campaign was then run with:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python main.py experiment \
  --parameters parameters/RTDETR/rtdetr_fam_rcra_full_data_stage_b_five_seed.yaml
```

These are two five-run grids, for ten scientific trainings in total. Do not
repeat completed runs under the same project names: the evaluator deliberately
fails on duplicate project/seed checkpoints. If a campaign is interrupted,
resume only the missing grid indices with `--start-from-run` and, when useful,
`--max-runs`.

After both campaigns finish, first resolve and audit all checkpoints without
inference:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/run_rtdetr_fam_rcra_full_data_stage_b_evaluation.py \
  --dry-run
```

Only then run the frozen evaluation:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/run_rtdetr_fam_rcra_full_data_stage_b_evaluation.py
```

## Completed run audit

The dry-run passed before any new MtErie inference. It found exactly the ten
expected checkpoints and no duplicate project/seed combination. For every
run, the evaluator verified the frozen configuration, final epoch 10, exactly
10,050 optimizer steps, all learning rates, one `latest` checkpoint and the
absence of automatic `test/*` metrics.

Every checkpoint loaded with zero missing and zero unexpected state-dict keys
after recovering the known shared RT-DETR aliases. Each evaluation used all
708 frozen samples.

## Completed primary Stage-B result

| Seed | Matched FAM mAP@50 | RCRA mAP@50 | RCRA - FAM |
|---:|---:|---:|---:|
| 40 | **0.4291** | 0.3969 | -0.0323 |
| 41 | 0.3498 | **0.3737** | +0.0239 |
| 42 | 0.3233 | **0.4185** | +0.0952 |
| 43 | **0.3308** | 0.2835 | -0.0474 |
| 44 | 0.3513 | **0.4038** | +0.0525 |

Matched FAM obtains `0.3569 +/- 0.0422` mAP@50; RCRA obtains
`0.3753 +/- 0.0538`. The paired RCRA-minus-FAM delta is
`+0.0184 +/- 0.0591`, positive in 3/5 seeds, with median `+0.0239` and a
two-sided Student-t IC95% of `[-0.0550, +0.0918]`.

The exploratory paired t test gives `p = 0.5250`; the exact two-sided Wilcoxon
test gives `p = 0.6250`. With only five seeds neither is evidence of statistical
superiority, and the confidence interval contains practically relevant values
in both directions.

## Frozen Stage-B decision

The predefined rule required both a mean mAP@50 gain of at least `+0.01` and
at least 4/5 paired wins. RCRA passes the mean condition (`+0.0184`) but fails
the stability condition (`3/5`). The frozen decision is therefore
`fail_retain_fam`: RCRA is not confirmed as the final architecture and
configuration-matched FAM remains the final performance baseline.

This should not be paraphrased as “RCRA is worse”. Its mean and median deltas
are positive, and it reaches sizable gains for seeds 42 and 44. The defensible
conclusion is narrower: the gain is not stable enough across seeds to justify
replacing FAM under the rule declared before full-data inference. The negative
seed-43 delta and especially the high dispersion prevent a robust improvement
claim.

Stage A remains a valid development result: RCRA passed its internal
whole-video rule and its mechanism audit. Stage B shows that this advantage
does not transfer consistently to full-data epoch-10 training and the already
used MtErie benchmark. No post-hoc LR, descriptor, threshold or checkpoint
search will be used to rescue the candidate.

After selection was closed, the retained current-code FAM checkpoints received
a descriptive paired modality characterization documented in
`notes/rtdetr_fam_full_data_paired_modality_evaluation.md`. VIS+IR exceeds VIS
in 5/5 seeds by `+0.0373 +/- 0.0223` mAP@50 (IC95%
`[+0.0095, +0.0650]`). This supports sensor complementarity but does not reopen
architecture selection.

## Secondary metrics and computational cost

Secondary metrics are descriptive and cannot override the primary frozen
mAP@50 rule:

| Metric | Matched FAM | RCRA | Mean paired delta | RCRA wins |
|---|---:|---:|---:|---:|
| COCO mAP | 0.1244 +/- 0.0185 | 0.1336 +/- 0.0183 | +0.0091 | 3/5 |
| mAP@50 | 0.3569 +/- 0.0422 | 0.3753 +/- 0.0538 | +0.0184 | 3/5 |
| mAP@75 | 0.0581 +/- 0.0148 | 0.0701 +/- 0.0089 | +0.0121 | 4/5 |

RCRA's more favorable mAP@75 suggests that its alignment selection may improve
localization precision in some runs. Because mAP@75 was not the primary
selection metric and mAP@50 fails the win rule, this is a discussion result,
not a new selection criterion.

Mean W&B training runtime is 8,534 seconds per FAM run (about 2 h 22 min) and
12,482 seconds per RCRA run (about 3 h 28 min). RCRA therefore costs about
46.3% more training time in this implementation despite adding only 5,283
parameters. This is a training-runtime observation; an inference benchmark is
still required before making any real-time deployment claim.

The new matched FAM mean (`0.3569`) is below the historical old-code FAM mean
(`0.3780`) by about `0.0211`. This supports keeping the historical result as a
secondary implementation-era reference. It does not alter the primary
comparison, for which FAM and RCRA were trained and evaluated under the same
current code.

The complete local aggregate JSON has SHA-256
`d9fbaf4d589d557a39437b0a28de18d7168b9fbbfcb7dd56ccf47c2c5fe6c702`;
the evaluator's local ten-row CSV has SHA-256
`af6cf8f03cd266057d0a9ea0d6dd772b16f1718c96e58fc72c3f672193d3ce90`.
The versioned compact five-row paired CSV has SHA-256
`401a0af273818bbd8ce6e3f966f39e24b6567a68a92d7674b7dedb7de0867dd6`.


## Thesis treatment

The thesis source remains intentionally unchanged in this commit. Its final
revision should present:

- Stage A as architecture development on a complete held-out FHL video;
- the scalar model as an attribution control that did not pass the FAM rule;
- Stage B as a separately trained full-data confirmation using `latest`;
- the newly matched FAM result as the primary Stage-B comparator;
- historical FAM only as a secondary implementation-era reference;
- the positive but unstable RCRA Stage-B delta, all five seed pairs and the
  failed 4/5-win condition;
- the secondary mAP@75 behavior and training-runtime overhead without using
  either to overturn the frozen decision;
- paired five-seed deltas and uncertainty without claiming a blind external
  test or statistical certainty unsupported by five seeds.

## Versioned artifacts

- FAM training grid:
  `parameters/RTDETR/rtdetr_fam_full_data_stage_b_five_seed.yaml`;
- RCRA training grid:
  `parameters/RTDETR/rtdetr_fam_rcra_full_data_stage_b_five_seed.yaml`;
- frozen evaluation:
  `parameters/RTDETR/rtdetr_fam_rcra_full_data_stage_b_evaluation.yaml` and
  `scripts/run_rtdetr_fam_rcra_full_data_stage_b_evaluation.py`;
- compact paired result:
  `notes/Search_and_Rescue/results/rtdetr_fam_rcra_full_data_stage_b_evaluation.csv`;
- complete local output:
  `out/rtdetr_fam_rcra_full_data_stage_b_evaluation/`;
- regression tests: `tests/test_rtdetr_fam_rcra_full_data_stage_b.py`.
