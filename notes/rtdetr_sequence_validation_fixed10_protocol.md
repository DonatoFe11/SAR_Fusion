# RT-DETR whole-sequence validation, fixed 10-epoch protocol

Status: frozen replacement protocol, not yet launched

Defined: 2026-08-14, after retiring the temporal-tail seed-40 pilot

Campaign project: `RTDETR_FAM_SequenceVal_Fixed10_Protocol`

## Decision

Every run trains for exactly ten epochs. Validation never triggers early
stopping and therefore never changes the optimization budget. It is used only
to retain the strongest checkpoint observed within those ten epochs. Both
checkpoints are preserved:

- `best`: highest validation `map_50`, requiring an improvement greater than
  `0.001`; this is the predefined primary checkpoint for final evaluation;
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
All candidate architectures must use exactly the same folder split, seeds and
checkpoint rule.

## Fixed training and checkpoint rule

1. Train every seed for all ten epochs; there is no early-stopping parameter.
2. Evaluate the complete 896-frame validation video after every epoch.
3. Use validation `map_50` only to update `best`.
4. Require an improvement strictly greater than `0.001`; a near-tie retains
   the earlier checkpoint.
5. Save `latest` at epoch 10 regardless of validation behaviour.
6. Use `best` as the primary final checkpoint and evaluate `latest` as a paired
   diagnostic for every seed, not only for favourable cases.
7. Do not inspect MtErie while deciding epochs, hyperparameters or candidates.

Because the previous project already consulted MtErie repeatedly, it must
still be described in the thesis as an internal development benchmark rather
than a newly blinded test set.

## Execution

The configuration expands to five sequential runs, seeds 40--44:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python main.py experiment \
  --parameters parameters/RTDETR/rtdetr_fam_sequence_validation_fixed10_protocol.yaml
```

Before the full campaign, run only seed 40 as a technical and scientific pilot
of the replacement split:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python main.py experiment \
  --parameters parameters/RTDETR/rtdetr_fam_sequence_validation_fixed10_protocol.yaml \
  --max-runs 1
```

Unlike the earlier two-epoch smoke, this seed-40 run belongs to the campaign if
it completes successfully: it uses the final split, all ten epochs and the
frozen checkpoint rule. After verifying its artifacts, continue without
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
- compare new architectures only under this same replacement protocol.

## Artifacts

- Definitive configuration:
  `parameters/RTDETR/rtdetr_fam_sequence_validation_fixed10_protocol.yaml`
- Retired pilot report: `notes/rtdetr_temporal_validation_protocol.md`
- Retired split manifest:
  `parameters/RTDETR/rtdetr_temporal_validation_split.json`
- Phase-override loader: `sarfusion/data/__init__.py`
- Paired-folder normalization: `sarfusion/data/wisard.py`
- Regression tests: `tests/test_data_split_params.py` and
  `tests/test_rtdetr_temporal_validation.py`
