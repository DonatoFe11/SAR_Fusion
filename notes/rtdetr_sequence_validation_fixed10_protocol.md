# RT-DETR whole-sequence validation, fixed 10-epoch protocol

Status: completed for RT-DETR + FAM, five seeds and paired checkpoint audit

Defined: 2026-08-14, after retiring the temporal-tail seed-40 pilot
Completed: 2026-08-15

Campaign project: `RTDETR_FAM_SequenceVal_Fixed10_Protocol`

## Campaign completion and results

All five runs completed ten epochs. Validation never stopped training; it only
retained `best`, while `latest` is always epoch 10. The selected epochs were
1, 4, 6, 1 and 1 for seeds 40--44. Thus the decline seen after early epochs
was repeatable across seeds rather than a logging anomaly: decreasing training
loss did not consistently preserve performance on the held-out FHL sequence.

After all runs completed, both checkpoints were evaluated on the frozen paired
MtErie inventory: 708 VIS+IR frames, 1,770 VIS ground-truth boxes, confidence
threshold `0.01` and the same evaluator for every checkpoint.

| Seed | `best` epoch | Val mAP@50 `best` | Val mAP@50 epoch 10 | MtErie `best` | MtErie `latest` | `best - latest` |
|---:|---:|---:|---:|---:|---:|---:|
| 40 | 1 | 0.1521 | 0.0397 | 0.3490 | 0.2132 | +0.1357 |
| 41 | 4 | 0.1424 | 0.1030 | 0.3354 | 0.2887 | +0.0466 |
| 42 | 6 | 0.1655 | 0.0946 | 0.3077 | 0.2740 | +0.0337 |
| 43 | 1 | 0.1939 | 0.0790 | 0.3994 | 0.2876 | +0.1118 |
| 44 | 1 | 0.1689 | 0.1037 | 0.4036 | 0.2811 | +0.1225 |

Across seeds, `best` obtains `0.3590 +/- 0.0416` MtErie mAP@50, while
`latest` obtains `0.2689 +/- 0.0317`. The paired improvement is
`+0.0901 +/- 0.0466`, positive in 5/5 seeds, with a two-sided Student-t 95%
confidence interval of `[+0.0323, +0.1479]`. Mean COCO-style mAP is `0.1295`
for `best` and `0.0990` for `latest`.

This supports the predefined checkpoint selector within the replacement
protocol: the validation-ranked checkpoint also beats epoch 10 on MtErie for
every seed. It does not make MtErie a blind test, nor does it prove that the
new setup improves absolute accuracy over the historical campaign. Historical
FAM `latest` was approximately `0.3780 +/- 0.0439` mAP@50; its nominal value
is above the new `best` mean, but the comparison is not controlled because the
old run optimized all 4,019 paired frames whereas this protocol optimizes
3,123 and reserves an entire 896-frame video for validation. During development,
new architectures must be compared against `0.3590 +/- 0.0416` under this same
split and selector. After selection, the chosen architecture is retrained on all
4,019 paired frames. The older result remains a historical reference.

The five W&B runtimes range from 2 h 38 min 59 s to 2 h 47 min 34 s, with a
mean of about 2 h 42 min per run and about 13 h 30 min in total. Ten training
epochs plus ten full 896-frame validations explain the duration; there is no
evidence that transitions between experiments consumed a material fraction.

## Frozen checkpoint audit and artifacts

The checkpoint comparison is reproduced with:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/run_rtdetr_sequence_checkpoint_evaluation.py
```

The runner validates the MtErie inventory, resolves exactly one local run per
seed, hashes `best` and `latest`, caches each evaluation and writes a paired
aggregate. Its frozen protocol is
`parameters/RTDETR/rtdetr_fam_sequence_validation_checkpoint_evaluation.yaml`.
The versioned compact table is
`notes/Search_and_Rescue/results/rtdetr_fam_sequence_checkpoint_evaluation.csv`.
The complete local JSON is marked `protocol_complete: true` and has SHA-256
`1402142280d299d94bffc8628a756e6d15d42867c260425bcce6c27bfd80357e`.

The thesis source is intentionally unchanged. Its later revision must report
the 5/5 paired wins and confidence interval without calling MtErie a fresh
blind test, and must keep the historical `0.3780` result separate because the
training split and checkpoint rule differ.

## Current position in the experimental plan

The corrected paired-modality evaluation, the train-derived validation, the
fixed checkpoint rule and the RT-DETR + FAM development baseline are complete.
RT-DETR + FAM + P2 at the existing input resolution is now implemented and
documented in `notes/rtdetr_fam_p2_stage_a.md`. Its frozen configuration
contains one complete seed-40 run under this exact split and selector. Local
shape, pretrained-transfer and CUDA memory checks passed before launch. Its
validation result, not MtErie, determines whether it is expanded to additional
seeds according to the predeclared engineering triage. Higher input resolution
and reliability-gated alignment remain
later, separate ablations so their effects are not confounded with P2.

## Two-stage policy for future models

The whole-sequence split is a model-development instrument, not the final
full-data training recipe.

### Stage A: architecture development

- Train on 3,123 frames and validate on the complete 896-frame FHL 0401/0402
  sequence.
- Always complete ten epochs and retain `best` using validation mAP@50.
- Compare variants only under this common split and checkpoint rule.
- Start P2 with one complete seed-40 technical run; expand only variants that
  pass tensor-shape, memory and validation checks.
- Keep P2, higher resolution, reliability gating and alignment as separate
  ablations so their effects remain identifiable.

### Stage B: final full-data retraining

- Freeze the winning architecture and all hyperparameters before retraining.
- Restore all 4,019 paired training frames; do not reserve FHL 0401/0402.
- Train exactly ten epochs with the historical full-data recipe and evaluate
  epoch-10 `latest`.
- Use seeds 40--44 and compare only models trained under this same recipe.
- Reuse historical FAM (`0.3780 +/- 0.0439`) only after confirming optimizer,
  pretrained class head, augmentations, preprocessing, resolution and seed
  handling are identical; otherwise retrain FAM alongside the candidate.
- Consult MtErie for the candidate only after Stage A has frozen the model and
  continue to label it as an internal development benchmark, not a fresh blind test.

## Decision

Every run trains for exactly ten epochs. Validation never triggers early
stopping and therefore never changes the optimization budget. It is used only
to retain the strongest checkpoint observed within those ten epochs. Both
checkpoints are preserved:

- `best`: highest validation `map_50`, requiring an improvement greater than
  `0.001`; this is the predefined primary checkpoint for Stage-A development;
- `latest`: state after epoch 10; this is retained for a paired diagnostic of
  whether checkpoint selection helped.

MtErie is disabled during training. It is evaluated separately only after the
five runs are complete. No run is shortened, irrespective of its validation
curve.

## Why the temporal-tail pilot was retired

The retired protocol split the tail of every training video after a 30-frame
embargo. Seed 40 (`4wkvp82g`) obtained `map_50 = 0.8167` on that validation and
selected epoch 8. Subsequent checks found no byte-identical train/validation
images or evaluator error, but exposed two structural weaknesses:

- the last train frame and first validation frame were only 31 extracted
  frames apart within the same continuous clip;
- Baker contributed 74.3% of validation persons, whose median area was about
  29 times that of the FHL validation persons.

The already-consulted MtErie diagnostic then showed that the selected `best`
was worse than epoch-10 `latest` for this seed:

| Seed-40 checkpoint | Temporal validation `map_50` | MtErie `map_50` | MtErie `map` |
|---|---:|---:|---:|
| epoch 8 `best` | **0.8167** | 0.3967 | 0.1391 |
| epoch 10 `latest` | 0.8046 | **0.4303** | **0.1492** |

One checkpoint pair cannot estimate a general correlation, but it is enough to
reject the assumption that the same-video score is already a trustworthy
selector. Seeds 41--44 of that protocol were never launched. Seed 40 is a
documented pilot and is excluded from the replacement campaign.

The official FHL 0134/0135 validation is independent but not usable as the
primary selector for this detector: it contains 273 frames, only 148 persons,
184 empty frames and a median normalized person area of `0.000133`. Seed-40
`best` and `latest` both produced effectively zero `map_50` (`0.000030` and
`0.000012`), leaving no stable ranking signal.

## Whole-sequence split

The replacement split holds out one complete paired video instead of adjacent
tails from every video.

| Phase | Paired videos | Frames |
|---|---|---:|
| Train | FHL 0405/0406; Baker VIS/IR 1 | 3,123 |
| Validation | FHL 0401/0402 | 896 |
| Test, disabled during training | three MtErie VIS/IR pairs | 708 |

FHL 0401/0402 is chosen before running the replacement campaign because it is
a complete excluded clip, contains roughly 3,319 paired-frame VIS boxes, and
provides a dense tiny-person signal without allowing Baker to dominate the
selector. FHL 0401 and 0405 still belong to the same acquisition campaign, so
this is an internal development split rather than an external generalization
benchmark. Nevertheless, it removes direct same-video temporal adjacency.

The number of optimization frames, 3,123, is almost identical to the retired
pilot's 3,125, keeping epoch runtime and optimization-step count comparable.
All candidate architectures in Stage A must use exactly the same folder split,
seeds and checkpoint rule. Stage-B results must instead share the full-data,
fixed-ten-epoch `latest` recipe and must not be compared directly with Stage-A
scores.

## Fixed training and checkpoint rule

1. Train every seed for all ten epochs; there is no early-stopping parameter.
2. Evaluate the complete 896-frame validation video after every epoch.
3. Use validation `map_50` only to update `best`.
4. Require an improvement strictly greater than `0.001`; a near-tie retains
   the earlier checkpoint.
5. Save `latest` at epoch 10 regardless of validation behaviour.
6. Use `best` as the primary Stage-A checkpoint and evaluate `latest` as a
   paired diagnostic for every seed, not only for favourable cases.
7. Do not inspect MtErie while deciding epochs, hyperparameters or candidates.

Because the previous project already consulted MtErie repeatedly, it must
still be described in the thesis as an internal development benchmark rather
than a newly blinded test set.

## Reproduction and launch history

The configuration expands to five sequential runs, seeds 40--44:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python main.py experiment \
  --parameters parameters/RTDETR/rtdetr_fam_sequence_validation_fixed10_protocol.yaml
```

The completed campaign first ran seed 40 as a technical and scientific pilot
of the replacement split:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python main.py experiment \
  --parameters parameters/RTDETR/rtdetr_fam_sequence_validation_fixed10_protocol.yaml \
  --max-runs 1
```

Unlike the earlier two-epoch smoke, this seed-40 run belongs to the campaign:
it completed with the frozen split, all ten epochs and the checkpoint rule.
After verifying its artifacts, the remaining four seeds were launched without
repeating it:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python main.py experiment \
  --parameters parameters/RTDETR/rtdetr_fam_sequence_validation_fixed10_protocol.yaml \
  --start-from-run 1
```

## Thesis changes to apply later

The thesis source is intentionally unchanged for now. The final revision must:

- describe the temporal-tail experiment as a retired pilot, including why it
  was rejected and the seed-40 `best`/`latest` diagnostic;
- describe the definitive split at video level and state that every run always
  completes ten epochs;
- distinguish checkpoint selection (`best`) from training-budget selection
  (no early stopping);
- report both `best` and `latest` final results across all seeds;
- state that FHL whole-sequence validation is internal and MtErie is a
  previously consulted development benchmark, not a fresh blind test;
- distinguish Stage-A model selection from Stage-B full-data retraining;
- compare architectures within the same stage only;
- report the final full-data candidate against a configuration-matched
  full-data FAM baseline using epoch-10 `latest`.

## Artifacts

- Definitive configuration:
  `parameters/RTDETR/rtdetr_fam_sequence_validation_fixed10_protocol.yaml`
- P2 Stage-A report: `notes/rtdetr_fam_p2_stage_a.md`
- P2 seed-40 configuration:
  `parameters/RTDETR/rtdetr_fam_p2_sequence_validation_seed40.yaml`
- P2 regression tests: `tests/test_rtdetr_p2.py`
- Retired pilot report: `notes/rtdetr_temporal_validation_protocol.md`
- Retired split manifest:
  `parameters/RTDETR/rtdetr_temporal_validation_split.json`
- Phase-override loader: `sarfusion/data/__init__.py`
- Paired-folder normalization: `sarfusion/data/wisard.py`
- Regression tests: `tests/test_data_split_params.py` and
  `tests/test_rtdetr_temporal_validation.py`
