# Current-code FAM paired modality characterization

Status: completed; VIS+IR outperforms the paired VIS intervention in 5/5 seeds

Frozen: 2026-08-25, after Stage B retained configuration-matched FAM and before
observing any new VIS-only or IR-only result from its five full-data checkpoints

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
- `IR`: zero the three VIS channels;
- keep VIS annotations for all three interventions.

The VIS+IR values must reproduce the completed Stage-B measurements within
`0.0002` mAP@50. This is a reconstruction check, not a fitted tolerance.

Primary descriptive quantities are reported per seed and as mean, sample
standard deviation and two-sided Student-t IC95% over the five checkpoints:

- VIS+IR, VIS and IR mAP@50;
- `VIS+IR - VIS`;
- `VIS+IR - IR`;
- `VIS+IR - max(VIS, IR)` and the number of positive paired differences.

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

## Results

All 15 evaluations completed on 708 samples. The five VIS+IR reruns reproduce
their Stage-B values at the stored precision, confirming checkpoint, model,
preprocessing and evaluator reconstruction.

| Seed | VIS+IR mAP@50 | VIS mAP@50 | IR mAP@50 | Fusion - VIS |
|---:|---:|---:|---:|---:|
| 40 | 0.4291 | 0.3764 | 0.0213 | +0.0527 |
| 41 | 0.3498 | 0.3054 | 0.0333 | +0.0444 |
| 42 | 0.3233 | 0.3174 | 0.0170 | +0.0059 |
| 43 | 0.3308 | 0.3077 | 0.0093 | +0.0232 |
| 44 | 0.3513 | 0.2911 | 0.0265 | +0.0602 |

Across seeds, mAP@50 is:

- VIS+IR: `0.3569 +/- 0.0422`;
- VIS: `0.3196 +/- 0.0331`;
- IR: `0.0215 +/- 0.0091`.

VIS is the better single-modality intervention for every seed. Consequently,
`VIS+IR - best single` equals `VIS+IR - VIS`: `+0.0373 +/- 0.0223`, positive in
5/5 seeds, with median `+0.0444` and two-sided Student-t IC95%
`[+0.0095, +0.0650]`. This is evidence, on this fixed paired internal
population, that the thermal channel contributes information beyond VIS for
the selected FAM checkpoints.

The interpretation of the very low IR-only value is narrower. FAM is
asymmetric: RGB supplies the reference features and IR is aligned toward that
stream. Zeroing RGB therefore measures robustness of this fusion detector to a
missing reference modality; it is not an estimate of the best attainable
performance of a separately trained native IR detector. The valid sensor
complementarity quantity is the paired fusion-minus-VIS delta, not a claim that
the IR sensor itself is intrinsically weak.

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

The complete aggregate JSON has SHA-256
`28b767752de3b744e529dd7d281178a91cf08b5834414d772e706fda0012ddb5`;
the local 15-row evaluator CSV has SHA-256
`ca6836149bc6a986c87e8504291aa27f44f3b5b5f16d8d3d1f01943669f9cf99`;
the compact versioned CSV has SHA-256
`8f66c04e5b68ede039e531b1ce98e4cc18486a036874ca3a7eb244215f43a0d0`.

The thesis should report this as a post-selection paired sensor ablation of the
current-code FAM baseline, distinct from the historical native-sensor tables
that use different test populations. It should include all five seed deltas,
the 5/5 fusion wins, the confidence interval, the asymmetric-architecture
caveat for IR-only and the fact that MtErie was already used internally.
