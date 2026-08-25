# RT-DETR + FAM: paired-VIS modality-dropout probe

Status: completed; pure paired-VIS replacement rejected by the frozen rule

Defined: 2026-08-25

Completed: 2026-08-26

## Why this experiment exists

The selected full-data FAM checkpoints obtained only `0.0215 +/- 0.0091`
mAP@50 when RGB was masked on the 708 paired MtErie samples while retaining
VIS ground truth. An explicitly post-hoc native-coordinate diagnostic
subsequently obtained `0.5618 +/- 0.0498` on the same 708 IR counterparts with
native IR annotations. The two values are not competing scores: they measure
different coordinate contracts.

Code inspection identified a matching training/evaluation discrepancy. The
historical RT-DETR Modal Dropout uses, for an IR-only draw:

- the native IR image;
- the native IR canvas;
- native IR annotations.

The paired sensor intervention instead constructs the normal fusion sample,
adapts IR to the VIS canvas, masks the three RGB channels and retains VIS
annotations. The old augmentation therefore never trained the exact missing-RGB
condition measured by the paired evaluator.

This experiment tests whether removing that mismatch improves paired
missing-RGB robustness without materially damaging fusion or native-IR use. It
does **not** reinterpret `0.0215` as a detector bug and does not invalidate the
historical training recipe: native-coordinate dropout remains a coherent
choice when the intended input is a standalone native IR stream.

## Implemented intervention

`WiSARDDataset` now accepts:

```yaml
modal_dropout_coordinate_contract: paired_vis
```

For an IR-only draw, the loader creates the same four-channel sample used in
fusion mode:

1. adapt IR to the VIS input canvas with the existing `adapt_ir2rgb`;
2. preprocess VIS and adapted IR exactly as in the current fusion branch;
3. retain the processed VIS annotations;
4. apply the sampled intervention only by zeroing channels.

Fusion follows the existing branch unchanged. RGB-only also retains its
existing VIS preprocessing and zero IR channel, avoiding unnecessary IR
preprocessing because its tensor and target contract were already identical.

Consequently:

| Draw | RGB channels | IR channel | Training ground truth |
|---|---|---|---|
| fusion | present | adapted to VIS canvas | VIS |
| RGB-only | present | zero | VIS |
| IR-only | zero | adapted to VIS canvas | VIS |

The default is deliberately `native`. Every existing YAML that omits the new
field therefore retains its previous behaviour. Regression tests verify that
fusion and RGB-only tensors/targets are unchanged between the two contracts,
that paired-VIS IR-only retains VIS labels and that the historical default
continues to use native IR labels.

## Frozen seed-40 training screen

The training configuration is
`parameters/RTDETR/rtdetr_fam_paired_vis_modal_dropout_sequence_validation_seed40.yaml`.
It expands to exactly one run. After removing tracker metadata and the one new
dataset field, an automated test requires its flattened scientific
configuration to equal seed 40 of
`rtdetr_fam_sequence_validation_fixed10_protocol.yaml` exactly.

The matched settings are:

- RT-DETR + current DCNv2 FAM, P3--P5, input 640;
- seed 40, batch 4, AdamW and learning rate `2e-5`;
- train on FHL 0405/0406 and Baker pair 1, 3,123 paired frames;
- validate on the complete FHL 0401/0402 pair;
- probabilities `[0.2, 0.2, 0.6]` in IR-only, RGB-only, fusion order;
- exactly ten epochs, without early stopping;
- validation at every epoch and primary checkpoint `best` by fusion
  validation mAP@50 with `min_delta = 0.001`;
- no automatic test and no MtErie inference.

This is a scientific Stage-A screening run rather than an operational smoke
test. Under the frozen rule, passing would have allowed it to serve as seed 40
of a later five-seed expansion; it must not be repeated selectively.

## Frozen evaluation and promotion rule

After training, the baseline and candidate seed-40 `best` checkpoints were
evaluated on the same frozen 896-pair FHL validation inventory, at threshold
`0.01`, using:

1. paired VIS+IR with VIS ground truth;
2. paired VIS-only by masking the IR channel, with the same VIS ground truth;
3. paired masked-IR by masking RGB, with the same VIS ground truth;
4. native IR on the same 896 IR counterparts with their native IR ground
   truth, reported as a separate coordinate-contract comparison.

The paired inventory is the exact existing sorted-zip loader output. VIS has
897 files and IR has 896; the only excluded VIS item is the terminal frame
`00896`. All 896 retained pairs have matching temporal indices, so there is no
one-frame shift. The frozen inventory contains 3,319 VIS boxes, 93 empty VIS
frames, 3,122 IR boxes and 100 empty IR frames. Its full paired-row SHA-256 is
`47e2f348ebdc202cb749b1bbf2741fc868d8681eb24ffa4356cf4f584dfa4ec4`.

Expansion to seeds 41--44 requires **all** of these engineering conditions:

- candidate minus baseline paired masked-IR/VIS-GT mAP@50 `>= +0.03`;
- candidate minus baseline fusion mAP@50 `>= -0.01`;
- candidate minus baseline native-IR/IR-GT mAP@50 `>= -0.03`.

VIS-only is reported as a diagnostic but is not a fourth gate. These are
single-seed screening tolerances, not confidence bounds or evidence of
superiority. Passing only authorizes the unchanged five-seed experiment;
failing closes this pure replacement and may motivate a separately declared
mixed native/paired dropout design. No threshold may be changed after seeing
the seed-40 results.

MtErie is excluded from this screening and from any decision about expansion.
It may be consulted only after a five-seed candidate has passed the same
Stage-A development protocol and a full-data Stage-B comparison has been
frozen.

## Pre-training checks completed

- 8 targeted unit/configuration tests pass;
- a real cached RT-DETR preprocessor check loaded 943 paired FHL 0405/0406
  items and produced a `[4, 640, 640]` IR-only tensor;
- its RGB channels were exactly zero;
- its retained IR channel was exactly equal to the fusion sample's adapted IR
  channel;
- its processed targets were exactly equal to the fusion sample's VIS targets.

## Results

The candidate completed all ten epochs and 7,810 optimization steps. Its
fusion-selected `best` is epoch 3 with validation mAP@50 `0.156899`; the
matched baseline `best` is epoch 1 with `0.152148`. The standalone evaluator
reproduced both logged fusion values exactly, confirming checkpoint, loader,
preprocessor and metric reconstruction.

| FHL condition | Baseline | Paired-VIS dropout | Delta | Frozen requirement | Pass |
|---|---:|---:|---:|---:|:---:|
| paired VIS+IR / VIS GT | 0.152148 | 0.156899 | +0.004752 | >= -0.010 | yes |
| paired VIS / VIS GT | 0.151107 | 0.143206 | -0.007901 | diagnostic | -- |
| paired masked-IR / VIS GT | 0.002519 | 0.002836 | +0.000317 | >= +0.030 | **no** |
| native IR / IR GT | 0.263709 | 0.027067 | -0.236642 | >= -0.030 | **no** |

The pure replacement therefore fails two of the three required checks and is
closed. Seeds 41--44 must not be launched, and this candidate must not be
evaluated on MtErie or promoted to full-data Stage B.

The failure is not explained by absent IR-only sampling. The reproducibility
trace contains 800 training samples from the first 20 batches of every epoch:
146 IR-only (`18.25%`), 187 RGB-only (`23.38%`) and 467 fusion (`58.38%`),
close to the configured `20/20/60` distribution.

The small fusion mAP@50 increase is not evidence of a better detector at one
seed. Fusion COCO mAP changes from `0.07987` to `0.07771`, mAP@75 from
`0.07233` to `0.06634` and mAR@100 from `0.18111` to `0.13902`. More
importantly, the intervention it was designed to improve remains effectively
zero, while native-IR robustness is largely destroyed.

The most plausible interpretation is a coordinate-contract trade-off rather
than a software failure. Replacing native IR/IR-GT examples removes the
supervision that supported native-IR operation, but adapted IR alone still
cannot reliably predict VIS-coordinate boxes when the RGB reference needed by
the asymmetric FAM alignment is absent. This is an inference from the matched
results, not proof of a unique causal mechanism.

This result supports retaining historical native-coordinate Modal Dropout for
the selected FAM baseline. A future mixed native/paired strategy would be a
new experiment, not a continuation of this one; the negligible paired gain
provides little justification for launching it during the current campaign.

## Reproduction and artifacts

The fail-closed evaluation is defined by
`parameters/RTDETR/rtdetr_fam_paired_vis_modal_dropout_probe_evaluation.yaml`
and executed with:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/run_rtdetr_fam_paired_vis_modal_dropout_probe_evaluation.py
```

The compact versioned table is
`notes/Search_and_Rescue/results/rtdetr_fam_paired_vis_modal_dropout_probe_evaluation.csv`.
The complete aggregate remains under
`out/rtdetr_fam_paired_vis_modal_dropout_probe_evaluation/` and is marked
`protocol_complete: true`.

The thesis source is intentionally unchanged. This experiment can be reported
as a coordinate-contract ablation and a paired robustness/native-IR trade-off,
but seed 40 alone is not a final performance comparison.

## Training command used

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python main.py experiment \
  --parameters parameters/RTDETR/rtdetr_fam_paired_vis_modal_dropout_sequence_validation_seed40.yaml
```
