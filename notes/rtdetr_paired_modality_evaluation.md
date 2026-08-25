# RT-DETR paired modality evaluation on MtErie

Date: 2026-08-14  
Protocol: `rtdetr_paired_modality_evaluation_v1`

## Question and protocol

This evaluation estimates the contribution of each sensor to the existing
four-channel detector. It is distinct from the previous native-sensor
benchmarks, which remain useful for estimating performance on all available VIS
or IR test streams but cannot support a paired claim about sensor fusion.

All conditions use exactly the same three paired MtErie streams, 708 frames,
1,770 VIS ground-truth boxes and the VIS coordinate system. A single paired
loader is traversed once per checkpoint. For every batch:

- `VIS+IR` keeps all four preprocessed channels;
- `VIS` zeros only the IR channel;
- `IR` zeros the three VIS channels;
- labels, sample order, pixel mask, threshold and checkpoint remain unchanged.

The protocol was frozen before inference. It covers six configurations, seeds
40--44 and the final `latest` checkpoint: 90 evaluations in total. The maximum
absolute difference between the 30 unmasked VIS+IR reruns and the historical
VIS+IR values was `0.000050` mAP@50.

## Why the previous table was not a paired comparison

The previous RT-DETR modality campaign deliberately selected `folders: vis`,
`folders: ir`, and `folders: vis_ir`. In the WiSARD phase filtering used by the
runner, those selectors produce three different test populations:

- VIS: 2,989 frames from ten folders;
- IR: 3,310 frames from six folders;
- VIS+IR: 708 paired frames from the three MtErie folder pairs.

Those results are not erroneous measurements. They are native-sensor benchmarks
over all test streams available for each input condition. The error is using
them as if every checkpoint had been evaluated three times on the same images
and annotations. Consequently, the historical VIS/IR/VIS+IR table can describe
deployment on the three native test populations, but it cannot estimate
`VIS+IR − best single` or establish sensor complementarity.

The corrected intervention starts from the paired four-channel tensor after the
existing `adapt_ir2rgb` preprocessing. This is important: it changes only the
available channels and not the population, geometry, labels, or evaluator.

## Results

Values are mean ± sample standard deviation over five checkpoint/seeds.
`Fusion − best single` is paired within each checkpoint; its confidence interval
is a two-sided 95% Student-t interval over the five signed deltas.

| Configuration | VIS+IR mAP@50 | VIS mAP@50 | IR mAP@50 | Fusion − best single | 95% CI | Fusion wins |
|---|---:|---:|---:|---:|---:|---:|
| Additive | 0.3081 ± 0.0677 | 0.2468 ± 0.0623 | 0.0269 ± 0.0080 | +0.0613 ± 0.0102 | [+0.0486, +0.0740] | 5/5 |
| FAM | 0.3780 ± 0.0439 | 0.3601 ± 0.0421 | 0.0206 ± 0.0055 | +0.0179 ± 0.0138 | [+0.0008, +0.0350] | 5/5 |
| FAM + IR Dropout | 0.3870 ± 0.0466 | 0.3651 ± 0.0552 | 0.0157 ± 0.0041 | +0.0219 ± 0.0146 | [+0.0038, +0.0400] | 5/5 |
| FAM + SSJ | 0.3751 ± 0.0185 | 0.3465 ± 0.0200 | 0.0153 ± 0.0064 | +0.0286 ± 0.0176 | [+0.0068, +0.0505] | 5/5 |
| Identity DCNv2 | 0.3281 ± 0.0942 | 0.2922 ± 0.1020 | 0.0202 ± 0.0083 | +0.0359 ± 0.0286 | [+0.0004, +0.0714] | 4/5 |
| Grid Sample | 0.3701 ± 0.0345 | 0.3469 ± 0.0270 | 0.0244 ± 0.0029 | +0.0232 ± 0.0127 | [+0.0073, +0.0390] | 5/5 |

For the main comparison, FAM minus Additive remains positive in VIS+IR:
`+0.0699 ± 0.0532`, 5/5 wins, 95% CI `[+0.0039, +0.1360]`, paired
t-test `p=0.0423`. On the paired VIS intervention the difference is
`+0.1133 ± 0.0505`, 5/5 wins, 95% CI `[+0.0507, +0.1760]`.

### Per-checkpoint mAP@50

| Configuration | Seed | VIS+IR | VIS | IR | Fusion − VIS |
|---|---:|---:|---:|---:|---:|
| Additive | 40 | 0.2566 | 0.1895 | 0.0152 | +0.0672 |
| Additive | 41 | 0.4029 | 0.3326 | 0.0259 | +0.0703 |
| Additive | 42 | 0.2960 | 0.2477 | 0.0345 | +0.0483 |
| Additive | 43 | 0.3476 | 0.2792 | 0.0342 | +0.0684 |
| Additive | 44 | 0.2373 | 0.1850 | 0.0247 | +0.0523 |
| FAM | 40 | 0.3780 | 0.3378 | 0.0223 | +0.0402 |
| FAM | 41 | 0.4334 | 0.4123 | 0.0184 | +0.0211 |
| FAM | 42 | 0.3130 | 0.3026 | 0.0295 | +0.0104 |
| FAM | 43 | 0.3964 | 0.3837 | 0.0163 | +0.0127 |
| FAM | 44 | 0.3693 | 0.3643 | 0.0164 | +0.0050 |
| FAM + IR Dropout | 40 | 0.4087 | 0.3925 | 0.0150 | +0.0162 |
| FAM + IR Dropout | 41 | 0.3234 | 0.2808 | 0.0180 | +0.0425 |
| FAM + IR Dropout | 42 | 0.4444 | 0.4300 | 0.0199 | +0.0144 |
| FAM + IR Dropout | 43 | 0.3988 | 0.3682 | 0.0162 | +0.0305 |
| FAM + IR Dropout | 44 | 0.3598 | 0.3541 | 0.0092 | +0.0057 |
| FAM + SSJ | 40 | 0.3670 | 0.3185 | 0.0084 | +0.0485 |
| FAM + SSJ | 41 | 0.3737 | 0.3362 | 0.0106 | +0.0375 |
| FAM + SSJ | 42 | 0.3527 | 0.3471 | 0.0227 | +0.0057 |
| FAM + SSJ | 43 | 0.4031 | 0.3668 | 0.0132 | +0.0363 |
| FAM + SSJ | 44 | 0.3792 | 0.3640 | 0.0214 | +0.0152 |
| Identity DCNv2 | 40 | 0.4339 | 0.4152 | 0.0182 | +0.0186 |
| Identity DCNv2 | 41 | 0.2117 | 0.1529 | 0.0066 | +0.0588 |
| Identity DCNv2 | 42 | 0.2928 | 0.2985 | 0.0254 | -0.0057 |
| Identity DCNv2 | 43 | 0.2861 | 0.2382 | 0.0260 | +0.0479 |
| Identity DCNv2 | 44 | 0.4161 | 0.3561 | 0.0251 | +0.0599 |
| Grid Sample | 40 | 0.4050 | 0.3773 | 0.0279 | +0.0277 |
| Grid Sample | 41 | 0.3479 | 0.3438 | 0.0234 | +0.0041 |
| Grid Sample | 42 | 0.4084 | 0.3690 | 0.0211 | +0.0394 |
| Grid Sample | 43 | 0.3326 | 0.3104 | 0.0227 | +0.0222 |
| Grid Sample | 44 | 0.3566 | 0.3341 | 0.0268 | +0.0225 |

## Interpretation

The corrected experiment supports a smaller and more precise claim than the old
cross-dataset comparison: fusion improves over the best masked single-sensor
condition in 29/30 checkpoints, not 30/30. The exception is Identity DCNv2 seed
42, where VIS+IR is `0.2928` and VIS is `0.2985`.

The marginal benefit of fusion is configuration-dependent. For standard FAM it
is only `+0.0179` mAP@50 on average and ranges from `+0.0050` to `+0.0402`.
SSJ has a larger mean fusion-versus-VIS margin (`+0.0286`) but does not improve
the final VIS+IR score over FAM: SSJ minus FAM is `-0.0029`, with 3/5 wins and a
95% CI of `[-0.0484, +0.0426]`.

The very low masked IR score must not be presented as general IR detector
performance. It is measured against VIS annotations after the fusion pipeline's
existing geometric adaptation, and therefore quantifies how much the trained
four-channel detector can rely on IR alone in the VIS evaluation coordinate
system. The historical IR-native benchmark answers a different question.

A later post-selection diagnostic on the current-code Stage-B FAM checkpoints
made this distinction explicit on the same 708 IR counterparts. Native IR
preprocessing with native IR ground truth obtains `0.5618 +/- 0.0498` mAP@50,
whereas paired masked IR with VIS ground truth obtains `0.0215 +/- 0.0091`.
This rules out IR-branch collapse but is a post-hoc coordinate-contract check,
not a direct metric comparison or a model-selection result; see
`notes/rtdetr_fam_full_data_paired_modality_evaluation.md`.

Machine-readable results are in
`out/rtdetr_paired_modality_evaluation/rtdetr_paired_modality_evaluation.json`
and the checkpoint table is in the adjacent CSV file.

## Thesis changes to apply later

No thesis source is changed as part of this correction. During the final thesis
revision, the following edits are required.

1. In `pages/04_background.tex`, qualify the statement that modality conditions
   retain a common dataset and annotation set. That statement is true for this
   paired masking protocol and for the final YOLO evaluator, but not for the
   historical RT-DETR native-sensor campaign.
2. In `pages/06_experimental_evaluation.tex`, subsection
   *Missing-modality evaluation*, replace the current RT-DETR modality table
   with the paired results above. The historical table may be retained only as
   a separately labelled native-sensor benchmark with its three sample counts.
3. Remove the claim that VIS+IR exceeds the strongest single modality in all 30
   checkpoints and that its margin is 0.1067--0.1454. Replace it with 29/30
   overall and the configuration-specific paired margins in this report.
4. Replace the old unimodal FAM-minus-Additive deltas. Under the paired
   intervention they are `+0.1133` for VIS (5/5) and `-0.0063` for IR (1/5).
   The IR result must retain the VIS-coordinate-system qualification.
5. In `pages/07_discussion_conclusions.tex`, subsection *RQ2*, replace the
   30/30 statement with the narrower conclusion: fusion is positive in 29/30
   checkpoints, while standard FAM has a mean paired fusion margin of only
   `+0.0179` and an IC95% of `[+0.0008, +0.0350]`.
6. Preserve the main FAM-versus-Additive conclusion. Its VIS+IR delta remains
   approximately `+0.0700`, is positive in 5/5 seeds, and is reproduced by the
   corrected evaluation.
7. State explicitly that masked IR against VIS ground truth is an ablation of
   reliance within the fusion system, not a replacement for the IR-native
   deployment benchmark.
8. Regenerate any modality figure and update cross-references only after the
   new train-derived-validation campaign is complete, avoiding two successive
   thesis rewrites.

## Reproduction

The frozen protocol is
`parameters/RTDETR/rtdetr_paired_modality_evaluation.yaml`; the resumable runner
is `scripts/run_rtdetr_paired_modality_evaluation.py`. From the repository root:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/run_rtdetr_paired_modality_evaluation.py
```

The runner validates the frozen 708-frame inventory, resolves all local
checkpoints, rejects incompatible cached results, checks the historical VIS+IR
reproduction tolerance, and aggregates only after all requested interventions
have completed.
