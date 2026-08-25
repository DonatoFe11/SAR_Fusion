# Current-code FAM paired modality characterization

Status: completed; VIS+IR outperforms the paired VIS intervention in 5/5 seeds

Frozen: 2026-08-25, after Stage B retained configuration-matched FAM and before
observing any new VIS-only or paired masked-IR result from its five full-data
checkpoints

## Question

Stage B selected RT-DETR + FAM as the final performance baseline because RCRA
did not pass the predeclared 4/5-win confirmation condition. The remaining
question is descriptive: how much does the selected current-code FAM rely on
VIS and IR when the evaluated population is held fixed?

This analysis cannot reopen model, checkpoint, threshold or seed selection.
MtErie is an already-used internal benchmark, not a newly blind test set.

## Frozen protocol

The five seeds 40--44 use their final epoch-10 `latest` checkpoints from
`RTDETR_FAM_FullData_StageB_FiveSeed`. Every condition uses the same 708 paired
MtErie frames, the same 1,770 VIS boxes, the same sample order and the same
confidence threshold `0.01`:

- `VIS+IR`: keep the complete four-channel tensor;
- `VIS`: zero only the IR channel;
- `masked IR / VIS GT`: zero the three VIS channels only after constructing the
  paired tensor; retain the geometrically adapted IR channel and VIS labels;
- keep VIS annotations for all three interventions.

The VIS+IR values must reproduce the completed Stage-B measurements within
`0.0002` mAP@50. This is a reconstruction check, not a fitted tolerance.

Primary descriptive quantities are reported per seed and as mean, sample
standard deviation and two-sided Student-t IC95% over the five checkpoints:

- VIS+IR, VIS and paired masked-IR/VIS-GT mAP@50;
- `VIS+IR - VIS`;
- `VIS+IR - paired masked IR`;
- `VIS+IR - best paired intervention` and the number of positive paired
  differences.

No pass/fail rule is attached to these quantities because architecture
selection is already closed.

## Reused compute characterization

The existing FAM latency and complexity result in
`notes/rtdetr_compute_benchmark.md` applies to this selected model: its
architecture, tensor shapes and operators are unchanged by the new checkpoint
weights. A redundant latency rerun is therefore not part of this protocol.
That benchmark measures detector-forward latency only, not an end-to-end SAR
pipeline.

## Commands

Preflight without inference:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/run_rtdetr_fam_full_data_paired_modality_evaluation.py --dry-run
```

Complete evaluation:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/run_rtdetr_fam_full_data_paired_modality_evaluation.py
```

Post-hoc native-IR coordinate diagnostic:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/run_rtdetr_fam_full_data_native_ir_coordinate_diagnostic.py
```

## Results

All 15 evaluations completed on 708 samples. The five VIS+IR reruns reproduce
their Stage-B values at the stored precision, confirming checkpoint, model,
preprocessing and evaluator reconstruction.

| Seed | VIS+IR mAP@50 | VIS mAP@50 | Masked IR / VIS GT mAP@50 | Fusion - VIS |
|---:|---:|---:|---:|---:|
| 40 | 0.4291 | 0.3764 | 0.0213 | +0.0527 |
| 41 | 0.3498 | 0.3054 | 0.0333 | +0.0444 |
| 42 | 0.3233 | 0.3174 | 0.0170 | +0.0059 |
| 43 | 0.3308 | 0.3077 | 0.0093 | +0.0232 |
| 44 | 0.3513 | 0.2911 | 0.0265 | +0.0602 |

Across seeds, mAP@50 is:

- VIS+IR: `0.3569 +/- 0.0422`;
- VIS: `0.3196 +/- 0.0331`;
- paired masked IR with VIS ground truth: `0.0215 +/- 0.0091`.

VIS is the better single-modality intervention for every seed. Consequently,
`VIS+IR - best single` equals `VIS+IR - VIS`: `+0.0373 +/- 0.0223`, positive in
5/5 seeds, with median `+0.0444` and two-sided Student-t IC95%
`[+0.0095, +0.0650]`. This is evidence, on this fixed paired internal
population, that the thermal channel contributes information beyond VIS for
the selected FAM checkpoints.

The `0.0215` value is not native IR detector performance. FAM is asymmetric:
its offset predictor consumes concatenated RGB and IR features and warps IR
toward RGB. This intervention removes the RGB reference while still requesting
boxes in the VIS coordinate system. Moreover, `adapt_ir2rgb` performs only the
existing resize/pad adaptation, not a calibrated image-level registration.

The training Modal Dropout does not reproduce this exact intervention. When it
samples IR-only, `WiSARDDataset` uses the native IR image and native IR labels;
the paired evaluator instead masks RGB after constructing an adapted paired
sample and keeps VIS labels. Thus `0.0215` measures missing-RGB robustness under
the VIS-coordinate fusion contract, not whether the thermal backbone retained
useful information.

## Post-hoc native-IR coordinate diagnostic

The unexpectedly low paired masked-IR value motivated a separate diagnostic.
Because it was defined after observing that value, it is explicitly marked
`post_hoc_diagnostic` and cannot be used for model or checkpoint selection.

The diagnostic uses the exact same 708 IR image counterparts from MtErie but
changes the coordinate contract to the one used by native IR samples during
training:

- no `adapt_ir2rgb`;
- the IR image occupies its native input canvas with zero RGB channels;
- the 1,824 native IR boxes are used instead of the 1,770 VIS boxes;
- there are 3 empty IR-labelled frames rather than 19 VIS-labelled frames;
- checkpoint, seed, threshold and evaluator remain unchanged.

| Seed | Paired masked IR / VIS GT | Native IR / IR GT diagnostic |
|---:|---:|---:|
| 40 | 0.0213 | 0.6389 |
| 41 | 0.0333 | 0.5841 |
| 42 | 0.0170 | 0.5180 |
| 43 | 0.0093 | 0.5360 |
| 44 | 0.0265 | 0.5322 |

Native-coordinate IR obtains `0.5618 +/- 0.0498` mAP@50, with IC95%
`[0.5001, 0.6236]`. This rules out a collapsed or numerically zero IR branch.
It does not show that IR is better than fusion: native IR and VIS+IR use
different coordinate systems and different ground truth, so their scores are
not direct competitors.

The historical FAM native-IR mean around `0.2028` was measured over 3,310
frames from a broader and different test population. The higher post-hoc value
on the 708 MtErie counterparts therefore cannot replace that deployment
benchmark; it isolates the coordinate-contract explanation on this smaller
subset.

Secondary means support a mainly detection/recall benefit rather than a clear
high-IoU localization gain. COCO mAP changes from `0.1144` VIS to `0.1244`
VIS+IR and mAR@100 from `0.2396` to `0.2617`, whereas mAP@75 is effectively
unchanged (`0.0580` for both at four decimals). These observations are
descriptive and do not alter the closed Stage-B selection.

## Artifacts and thesis treatment

The compact versioned result is
`notes/Search_and_Rescue/results/rtdetr_fam_full_data_paired_modality_evaluation.csv`.
The complete local output is under
`out/rtdetr_fam_full_data_paired_modality_evaluation/`.

The post-hoc native-IR diagnostic is reproduced by
`parameters/RTDETR/rtdetr_fam_full_data_native_ir_coordinate_diagnostic.yaml`
and `scripts/run_rtdetr_fam_full_data_native_ir_coordinate_diagnostic.py`. Its
compact result is
`notes/Search_and_Rescue/results/rtdetr_fam_full_data_native_ir_coordinate_diagnostic.csv`;
the complete local output is under
`out/rtdetr_fam_full_data_native_ir_coordinate_diagnostic/`.

The complete aggregate JSON has SHA-256
`28b767752de3b744e529dd7d281178a91cf08b5834414d772e706fda0012ddb5`;
the local 15-row evaluator CSV has SHA-256
`ca6836149bc6a986c87e8504291aa27f44f3b5b5f16d8d3d1f01943669f9cf99`;
the compact versioned CSV has SHA-256
`ac25727b34fbd929cb316dbdb2f3607c8d9bee3a345ba40ede1cc9e7a980a152`.

The native-IR diagnostic aggregate JSON has SHA-256
`7e0fce162d298f303a4bb602379d2b78df7ac420f65335df5851d0f85eaca034`;
its local evaluator CSV has SHA-256
`7a1719d9afb12245f9199b6f33c43283ee8e180bd39d877473d204bbe64b5c8d`;
its compact versioned CSV has SHA-256
`610b83758a52c2610b972a5544873756accf6e60683966c89f6c25a8e3f68a5c`.

The thesis should report this as a post-selection paired sensor ablation of the
current-code FAM baseline, distinct from the historical native-sensor tables
that use different test populations. It should include all five seed deltas,
the 5/5 fusion wins and the confidence interval. The `0.0215` condition must be
named paired masked-IR/VIS-GT robustness rather than general IR-only
performance. The post-hoc native-coordinate result may be used as a diagnostic
showing that the IR branch did not collapse, but not as a direct competitor to
VIS+IR or as a new selection result. MtErie must remain labelled as an
already-used internal benchmark.
