# RT-DETR train-derived temporal validation protocol

Split frozen: 2026-08-14, before training

Protocol finalized: 2026-08-14, before any run was accepted or scored

Split ID: `rtdetr_train_temporal_validation_v1`
Campaign project: `RTDETR_FAM_TemporalVal_Protocol`

Technical smoke project: `RTDETR_FAM_TemporalVal_Smoke`

## Purpose

The previous RT-DETR campaign used ten fixed epochs and the final `latest`
checkpoint because the existing validation set was a single strongly shifted
FHL session and was unsuitable for checkpoint selection. The new campaign needs
a validation signal that can support early stopping without consulting MtErie.

This protocol creates that signal exclusively from the paired sequences already
assigned to training. It does not retroactively invalidate the fixed-epoch
campaign. Results from the old `latest` protocol and the new `best` protocol
answer different experimental questions and must not be mixed as if their
training settings were identical. RT-DETR + FAM is retrained under the new
protocol to provide the comparable baseline for every new candidate.

## Frozen temporal split

Only the three labelled VIS+IR pairs effectively used by the current WiSARD
training loader are eligible. Within each pair, paths are sorted exactly as in
`build_wisard_items` and paired through the existing shortest-stream `zip`.
The last 20% of each sequence is validation. The preceding 30 frames are an
embargo excluded from both train and validation, reducing immediate temporal
adjacency across the boundary.

| Sequence | Paired frames | Train range | Embargo range | Validation range |
|---|---:|---:|---:|---:|
| FHL VIS 0401 / IR 0402 | 896 | `[0, 687)` | `[687, 717)` | `[717, 896)` |
| FHL VIS 0405 / IR 0406 | 943 | `[0, 724)` | `[724, 754)` | `[754, 943)` |
| Baker VIS 1 / IR 1 | 2,180 | `[0, 1714)` | `[1714, 1744)` | `[1744, 2180)` |

The two FHL VIS streams contain respectively one and two trailing frames with no
IR counterpart. These three frames were already silently excluded by the
existing paired loader. Their counts are now recorded explicitly in the frozen
manifest.

| Partition | Frames | VIS boxes | Empty VIS frames | Median normalized box area |
|---|---:|---:|---:|---:|
| Train | 3,125 | 7,851 | 508 | 0.001606 |
| Embargo, unused | 90 | 258 | 30 | 0.000231 |
| Validation | 804 | 1,799 | 178 | 0.003629 |

The manifest freezes folder names, stream counts, ranges, label hashes, image
paths and sizes, phase inventories, box counts and medians. Any change to the
local source inventory causes loading to fail rather than silently changing the
split. No model prediction or validation metric was inspected before freezing
these ranges; only annotation and inventory statistics were used.

An initial command accidentally launched the complete five-seed grid with a
20-epoch cap. Those processes were manually stopped and are excluded from all
analysis. Before inspecting or accepting any validation result, the protocol
was separated into a two-epoch smoke run and a scientific campaign capped at
the historical budget of ten epochs. This procedural correction was motivated
only by runtime and launch scope; the frozen data split was not changed.

### Leakage controls

- MtErie never appears in train or validation.
- The historical FHL validation session is not used for checkpoint selection.
- Paired RGB and IR frames always stay in the same partition.
- Splitting is chronological within a sequence, never random by frame.
- Train and validation item inventories are disjoint.
- Thirty intervening frames per sequence are discarded.
- Modal Dropout remains active only in training; validation always uses complete
  VIS+IR input.

## Fixed checkpoint rule

The following rule applies unchanged to the new FAM baseline and to every model
compared with it:

1. Train for at most 10 epochs, matching the historical RT-DETR budget.
2. Evaluate the full 804-frame temporal validation set after every epoch.
3. Use validation `map_50` as the sole checkpoint-selection metric.
4. Save a new `best` only when `map_50` is strictly more than `0.001` above the
   current best value.
5. A change of at most `0.001`, including an exact tie, retains the earlier
   checkpoint.
6. Stop after five consecutive validation epochs without a qualifying
   improvement.
7. Keep `best` and `latest` as separate checkpoint directories.
8. All final evaluation uses `best`; `latest` is diagnostic only.

Validation does not compute RT-DETR loss. In Hugging Face RT-DETR evaluation
mode, labels are consumed only after logits and boxes have been produced, to
run bipartite matching and auxiliary loss calculations. Ground truth remains
available to the evaluator, so skipping this unused loss leaves predictions
and `map_50` unchanged while reducing checkpoint-selection overhead. The full
COCO metric collection is computed once at the end of each validation epoch;
the loop runs under `torch.inference_mode()`.

The code now computes improvement independently from whether `latest` is due to
be saved. This prevents the earlier `save_final_checkpoint_only` interaction
from triggering early stopping without a valid best checkpoint. The selected
best epoch and metric are also written to the W&B summary. To avoid repeated
full-state writes, `best` is saved whenever it improves, whereas `latest` is
saved only at the natural final epoch or at the epoch that triggers early
stopping. Final selection metadata is written through the concrete W&B run and
reasserted immediately before tracker shutdown, preventing a final-epoch
improvement from leaving a stale local summary.

## Protection of the development benchmark

The executable baseline configuration sets `run_test: false`. Training and
early stopping therefore produce no MtErie metric. MtErie evaluation will be a
separate, frozen step after the baseline and candidate set have been selected
using only temporal validation.

The new validation is not an external generalization benchmark. It uses later
segments of the same three acquisition pairs and is dominated numerically by
Baker. It is appropriate for checkpoint selection and controlled model
comparison, but its absolute score must not be presented as evidence of
cross-site generalization.

## Comparability with the previous campaign

The old five-seed FAM results remain the estimate for the historical protocol:
ten fixed epochs and `latest`. The model architecture itself is unchanged:
both use `fusion_rtdetr`, FAM `current_dcnv2`, 640-pixel preprocessing, AdamW
with learning rate `2e-5`, training batch size 4 and the same modal-dropout
probabilities. The new protocol changes the data available to optimization and
the checkpoint rule:

| Component | Historical RT-DETR + FAM | New temporal-validation baseline |
|---|---|---|
| Optimization frames | all 4,019 paired train frames | 3,125 temporal-head frames |
| Validation | disabled | 804 frozen temporal-tail frames every epoch |
| Temporal embargo | none | 90 frames excluded |
| Maximum epochs | 10 | 10 |
| Selected checkpoint | `latest` | validation `map_50` `best` |
| Early stopping | disabled | patience 5, minimum delta 0.001 |
| MtErie during training run | evaluated after training | disabled |

Consequently, the historical mean `0.3780` and the result of the new baseline
are not a controlled before/after comparison: the new run optimizes on fewer
frames and selects a checkpoint with a new signal. A new architecture may only
be compared against the newly trained FAM baseline.

A useful diagnostic, after the new FAM runs are complete, is to report both
`best` and `latest` for those same runs. This quantifies the effect of checkpoint
selection without crossing protocols. It does not make the old and new
campaigns directly exchangeable.

## Execution order

1. Commit this frozen split and checkpoint rule.
2. Run the separate two-epoch technical smoke configuration. It verifies real
   train/validation execution, distinct `best`/`latest` writes and W&B metadata.
   Its metric is discarded and the run is never included in campaign tables.
   Early-stopping decision logic is covered by the regression test; a two-epoch
   run is not expected to exhaust patience 5.
3. If the smoke is technically valid, launch the complete frozen campaign:
   five fresh runs for seeds 40--44. All five, including a newly trained seed
   40, belong to the scientific campaign.
4. Aggregate validation curves, selected epochs and run durations.
5. Freeze the candidate architecture set.
6. Train candidates with the same split, seed set, maximum epochs, patience,
   minimum delta and checkpoint selector.
7. Only then run the predefined MtErie and paired-modality final evaluations.

To launch the non-scientific technical smoke from the repository root:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python main.py experiment \
  --parameters parameters/RTDETR/rtdetr_fam_temporal_validation_smoke.yaml
```

The command above executes seed 40 for two epochs but does not count it as seed
40 of the campaign. After checking that both checkpoint directories and the
summary metadata exist, launch the complete five-seed campaign from scratch:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python main.py experiment \
  --parameters parameters/RTDETR/rtdetr_fam_temporal_validation_protocol.yaml
```

The campaign uses evaluation batch size 12 while preserving training batch
size 4. This only groups more validation samples into each forward pass; it
does not change model optimization or the computed metrics. If batch 12 causes
an out-of-memory error on the 8 GB GPU during the smoke, reduce only
`evaluation_batch_size` before freezing and launching the campaign.

## Thesis changes to apply later

The thesis `.tex` files are intentionally unchanged for now. In the final
revision:

- describe the old fixed-epoch campaign and the new validation-based campaign
  as separate protocols;
- state that the temporal validation is train-derived and not an external
  holdout;
- report the exact temporal ranges, 30-frame embargo and frozen inventory;
- explain that all new model comparisons use a retrained FAM baseline;
- report selected epoch distributions rather than only the final test scores;
- state that MtErie was not evaluated during checkpoint or architecture
  selection in the new campaign;
- retain the limitation that MtErie remains a previously consulted internal
  development benchmark, not a newly blinded test set.

## Artifacts

- Split manifest: `parameters/RTDETR/rtdetr_temporal_validation_split.json`
- Baseline configuration:
  `parameters/RTDETR/rtdetr_fam_temporal_validation_protocol.yaml`
- Technical smoke configuration:
  `parameters/RTDETR/rtdetr_fam_temporal_validation_smoke.yaml`
- Split implementation: `sarfusion/data/temporal_split.py`
- Loader integration: `sarfusion/data/__init__.py`
- Checkpoint implementation: `sarfusion/experiment/run.py`
- Regression tests: `tests/test_rtdetr_temporal_validation.py`
