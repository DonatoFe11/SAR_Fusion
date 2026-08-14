# RT-DETR train-derived temporal validation protocol

Date frozen: 2026-08-14  
Split ID: `rtdetr_train_temporal_validation_v1`  
Baseline project: `RTDETR_FAM_TemporalVal_Protocol`

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

1. Train for at most 20 epochs.
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

The code now computes improvement independently from whether `latest` is due to
be saved. This prevents the earlier `save_final_checkpoint_only` interaction
from triggering early stopping without a valid best checkpoint. The selected
best epoch and metric are also written to the W&B summary.

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
ten fixed epochs and `latest`. The new campaign changes both the effective
training set and checkpoint rule, so a new architecture may only be compared
against the newly trained FAM baseline.

A useful diagnostic, after the new FAM runs are complete, is to report both
`best` and `latest` for those same runs. This quantifies the effect of checkpoint
selection without crossing protocols. It does not make the old and new
campaigns directly exchangeable.

## Execution order

1. Commit this frozen split and checkpoint rule.
2. Run one technical FAM pilot, seed 40, to verify a complete train/validation
   cycle, checkpoint restoration and early stopping metadata. Do not modify the
   split or rule in response to its score; only implementation failures may be
   fixed.
3. If the pilot is technically valid, resume seeds 41--44 unchanged.
4. Aggregate validation curves, selected epochs and run durations.
5. Freeze the candidate architecture set.
6. Train candidates with the same split, seed set, maximum epochs, patience,
   minimum delta and checkpoint selector.
7. Only then run the predefined MtErie and paired-modality final evaluations.

To launch only the technical pilot from the repository root:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python main.py experiment \
  --parameters parameters/RTDETR/rtdetr_fam_temporal_validation_protocol.yaml \
  --start-from-run 0 \
  --max-runs 1
```

The command above executes seed 40 only. After the pilot is accepted as
technically valid, launch the remaining frozen seeds with `--start-from-run 1`
and no `--max-runs` override.

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
- Split implementation: `sarfusion/data/temporal_split.py`
- Loader integration: `sarfusion/data/__init__.py`
- Checkpoint implementation: `sarfusion/experiment/run.py`
- Regression tests: `tests/test_rtdetr_temporal_validation.py`
