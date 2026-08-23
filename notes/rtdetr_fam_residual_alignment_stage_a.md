# RT-DETR + FAM + Reliability-Conditioned Residual Alignment: Stage A

Status: completed; passes the frozen Stage-A promotion rule and the independent
mechanism audit

Defined: 2026-08-22

## Motivation

The completed experiments rule out several simple explanations for the current
FAM baseline:

- adding P2 reduced validation performance in 5/5 seeds;
- increasing the input from 640 to 800 reduced the mean score and won only 1/5
  seed pairs;
- the post-FAM reliability gate at the shared learning rate remained almost
  exactly neutral;
- increasing only that gate's learning rate activated it, but did not yield a
  repeatable detection gain;
- the historical identity, grid-sample and bounded-offset FAM variants did not
  improve the current DCNv2 FAM.

The next candidate therefore changes neither resolution nor feature-pyramid
levels and does not repeat modality weighting after fusion. It tests a narrower
hypothesis: **FAM's spatial correction is useful in some locations but harmful
in others, so the detector should be allowed to retain or bypass the correction
locally instead of always replacing raw IR with aligned IR.**

The working name is Reliability-Conditioned Residual Alignment (RCRA). It is a
project-specific architectural proposal, not a claim that reliability-aware
alignment is unprecedented. Related work already combines deformable alignment
and adaptive fusion in UAV RGB-IR detection (OAFA, CVPR 2024), confidence and
alignment in multispectral pedestrian detection (AR-CNN, ICCV 2019), and
uncertainty-conditioned residual fusion for unaligned RGB-T salient-object
detection (UMFNet, CVPR 2026). The exact raw-IR/FAM-residual selector used here
and its insertion into this RT-DETR + FAM implementation are the proposed
design contribution. A broader novelty claim would require a dedicated
systematic literature review.

Primary references:

- [OAFA: Weakly Misalignment-free Adaptive Feature Alignment for UAVs-based Multimodal Object Detection](https://openaccess.thecvf.com/content/CVPR2024/html/Chen_Weakly_Misalignment-free_Adaptive_Feature_Alignment_for_UAVs-based_Multimodal_Object_Detection_CVPR_2024_paper.html)
- [AR-CNN: Weakly Aligned Cross-Modal Learning for Multispectral Pedestrian Detection](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhang_Weakly_Aligned_Cross-Modal_Learning_for_Multispectral_Pedestrian_Detection_ICCV_2019_paper.pdf)
- [UMFNet: Uncertainty-Aware Modality Fusion for Unaligned RGB-T Salient Object Detection](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Uncertainty-Aware_Modality_Fusion_for_Unaligned_RGB-T_Salient_Object_Detection_CVPR_2026_paper.html)

## Frozen architecture

At each of P3, P4 and P5, let `I_raw` be the IR-backbone feature and `I_aligned`
the output of the existing `current_dcnv2` FAM. RCRA predicts one spatial scalar
per feature-map location:

```text
alpha      = 2 * sigmoid(g(descriptors))
I_selected = I_aligned + (alpha - 1) * (I_aligned - I_raw)
F_fused    = F_RGB + I_selected
```

The interpretation is:

- `alpha = 0`: bypass FAM and recover raw IR;
- `alpha = 1`: retain the existing FAM output;
- `alpha > 1`: moderately amplify the learned FAM correction;
- the interval is bounded to `(0, 2)`.

The last `1x1` logit convolution is zero-initialized. Consequently
`alpha == 1` at every location and the finite-precision expression returns
`I_aligned` exactly at initialization. This is important: the candidate starts
from the already trained recipe's architecture, not from an arbitrary mixture
of raw and aligned IR.

The predictor uses twelve channel-compressed spatial descriptors:

1. local channel mean and RMS of RGB;
2. local channel mean and RMS of raw IR;
3. local channel mean and RMS of aligned IR;
4. channel-wise cosine agreement of RGB with raw IR;
5. channel-wise cosine agreement of RGB with aligned IR;
6. RMS magnitude of the alignment residual;
7. agreement gain, `agreement_aligned - agreement_raw`;
8. the two image-level modality-presence indicators expanded spatially.

These descriptors feed `Conv3x3(12,16) -> SiLU -> Conv1x1(16,1)`. There is one
independent predictor per pyramid level: 1,761 parameters per level and 5,283
parameters in total. The descriptor cost scales linearly with spatial area and
does not contain a full-channel concatenation convolution.

RCRA is applied immediately after FAM and before the existing optional IR
dropout; dropout is zero in this campaign. The previous post-fusion reliability
gate is explicitly disabled, and the implementation rejects enabling both
gates together so that the ablation remains identifiable.

## Optimization choice

The neutral output layer only begins to learn through its last convolution.
The earlier gate experiment established that the shared `2e-5` detector LR can
leave such a neutral module effectively unchanged in ten epochs, whereas a
dedicated `2e-4` LR produces measurable movement. RCRA therefore uses a frozen
dedicated AdamW group at `2e-4`; all backbone, FAM, encoder, decoder and head
groups retain `2e-5`.

This means the Stage-A result estimates the complete **RCRA candidate recipe**,
not a pure architecture-only effect at a shared LR. If the candidate passes,
a later shared-LR ablation may separate architecture from optimization; it must
not be used to rescue a failed five-seed result by post-hoc tuning.

W&B records the fourth group as `train/lr_alignment_gate`. Configuring this LR
without any `alignment_gates` parameters is treated as an error.

## Frozen Stage-A protocol

- seeds: `40, 41, 42, 43, 44`;
- train: FHL 0405/0406 and Baker VIS/IR 1, 3,123 paired frames;
- validation: complete FHL 0401/0402, 896 paired frames;
- P3--P5, input 640, direct batch 4, validation batch 12;
- exactly ten epochs, no early stopping;
- `best` selected by validation mAP@50 with `min_delta = 0.001`;
- `latest` retained only as a diagnostic;
- modal dropout probabilities `[0.2, 0.2, 0.6]` unchanged;
- automatic test, MtErie and full-data Stage B disabled.

The primary reference is the existing paired five-seed FAM Stage-A baseline,
`0.1646 +/- 0.0196` best validation mAP@50.

## Frozen decision rule

RCRA advances only if both engineering conditions hold:

1. mean paired best-mAP@50 gain of at least `+0.01` over FAM;
2. a positive paired delta in at least 4/5 seeds.

A smaller positive mean or 3/5 wins is inconclusive and cannot support a better
model claim. A non-positive mean or fewer than 3/5 wins closes the candidate.
The paired standard deviation, median, 95% t interval, `best` epochs and
`latest` scores will be reported regardless.

The alpha audit is diagnostic and cannot override the performance rule. It is
run on every selected `best` checkpoint in fusion, RGB-only and IR-only modes.
It reports alpha distributions at P3--P5 and modality-absence responses. To
describe the mechanism as active rather than neutral, the across-seed audit
must show:

- mean absolute deviation `|alpha - 1| >= 0.01` at at least one pyramid level
  in fusion mode;
- for both absent-modality comparisons, an absolute change of mean alpha of at
  least `0.01` at some level in at least 3/5 seeds.

Failing these mechanism criteria changes the interpretation, not the frozen
performance result. MtErie must not be inspected before the Stage-A decision.

## Commands

The checkpoint-free operational probe has completed successfully:

- W&B run `1bbxubjk`, tagged `ExcludeFromCampaign`;
- exactly 3 alignment gates and 12 optimizer tensors at LR `2e-4`;
- all other optimizer groups retained LR `2e-5`;
- 20/20 direct batch-4 training steps, with stable steps near 1.1 seconds after
  warm-up;
- 75/75 validation batches completed in approximately 71 seconds;
- exit code 0, no OOM and no scientific checkpoint.

Its validation mAP@50 (`0.0847`) is not a scientific result because the model
saw only 20 training batches. The command used was:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python main.py experiment \
  --parameters parameters/RTDETR/rtdetr_fam_residual_alignment_runtime_probe.yaml
```

The probe passed. The complete five-seed campaign was launched with:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python main.py experiment \
  --parameters parameters/RTDETR/rtdetr_fam_residual_alignment_sequence_validation_five_seed.yaml
```

After all five `best` checkpoints existed, alpha was audited without consulting
MtErie:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/run_rtdetr_fam_residual_alignment_alpha_audit.py \
  --protocol parameters/RTDETR/rtdetr_fam_residual_alignment_alpha_audit.yaml
```

Both commands completed. No command in this section remains to be run.

## Completed five-seed performance result

All five scientific runs completed exactly ten epochs (`7,810` optimizer
steps), recorded `train/lr_alignment_gate = 2e-4`, and contain both `best` and
`latest` checkpoints. The primary comparison uses the predefined
validation-selected `best` checkpoint.

| Seed | FAM best | RCRA best | RCRA epoch | RCRA latest | RCRA - FAM |
|---:|---:|---:|---:|---:|---:|
| 40 | 0.1521 | **0.1692** | 5 | 0.1177 | +0.0171 |
| 41 | 0.1424 | **0.2243** | 1 | 0.0422 | +0.0820 |
| 42 | 0.1655 | **0.1848** | 5 | 0.1210 | +0.0193 |
| 43 | **0.1939** | 0.1562 | 1 | 0.1006 | -0.0377 |
| 44 | 0.1689 | **0.1778** | 1 | 0.1029 | +0.0089 |

FAM obtains `0.1646 +/- 0.0196`; RCRA obtains `0.1825 +/- 0.0257` best
validation mAP@50. The paired RCRA-minus-FAM delta is
`+0.0179 +/- 0.0427`, positive in 4/5 seeds, with median `+0.0171` and a
two-sided Student-t 95% confidence interval of `[-0.0350, +0.0709]`.

The frozen promotion rule required a mean gain of at least `+0.01` and at least
4/5 wins. RCRA reaches `+0.0179` and exactly 4/5, so both requirements pass.
The confidence interval nevertheless includes both signs because five seeds
are few and seed-to-seed variance is substantial. The correct conclusion is
that RCRA is the first candidate to pass the predeclared engineering rule, not
that statistical superiority has been established.

RCRA `latest` obtains `0.0969 +/- 0.0318`. The paired `best - latest` gain is
`+0.0856 +/- 0.0547`, positive in 5/5 seeds, with IC95%
`[+0.0177, +0.1535]`. This again supports validation-based checkpoint selection
in Stage A and shows why the epoch-10 result cannot replace `best` in this
comparison.

## Completed alpha-mechanism audit

Every `best` checkpoint loaded with zero missing and zero unexpected keys. The
audit evaluated all 896 FHL validation pairs in fusion, RGB-only and IR-only
modes, without consulting MtErie.

| Seed | P3 mean / MAD | P4 mean / MAD | P5 mean / MAD | max response: RGB absent | max response: IR absent |
|---:|---:|---:|---:|---:|---:|
| 40 | 0.9762 / 0.0299 | 0.9315 / 0.0685 | 0.9849 / 0.0151 | 0.4401 | 0.0822 |
| 41 | 0.9177 / 0.0823 | 0.9077 / 0.0923 | 0.9686 / 0.0314 | 0.1899 | 0.0704 |
| 42 | 0.9937 / 0.0310 | 0.7876 / 0.2124 | 0.9982 / 0.0018 | 0.4820 | 0.0954 |
| 43 | 0.9318 / 0.0682 | 0.9527 / 0.0473 | 0.9957 / 0.0043 | 0.2231 | 0.0788 |
| 44 | 0.9151 / 0.0849 | 0.9600 / 0.0400 | 0.9961 / 0.0039 | 0.1478 | 0.0892 |

`MAD` is the spatial mean of `|alpha - 1|` in fusion mode. Across seeds, the
mean fusion alpha is `0.9469` at P3, `0.9079` at P4 and `0.9887` at P5; mean
MAD is respectively `0.0593`, `0.0921` and `0.0113`. Thus the learned behavior
mainly suppresses part of the FAM correction at P3/P4 and leaves P5 close to
the original FAM. It is not a uniform return to raw IR: P3 retains meaningful
spatial variation (mean within-condition standard deviation `0.0260`), while
P4 and especially P5 act more like level-wise calibration.

When RGB is removed, the maximum mean-alpha response exceeds `0.01` in 5/5
seeds and always occurs at P3 (`0.1478--0.4820`). The gate therefore moves
toward raw IR when the RGB guide required by FAM is absent. Removing IR also
changes mean alpha by at least `0.01` in 5/5 seeds (`0.0704--0.0954`, again at
P3). Both frozen mechanism conditions required only 3/5 wins and therefore
pass decisively.

The failed seed 43 is not explained by a neutral or obviously pathological
gate: its P3/P4 alpha and both missing-modality responses are within the range
of the other seeds. With only five observations there is no defensible simple
relationship between gate magnitude and paired performance. The audit supports
that RCRA learned the intended conditional mechanism, but not that stronger
modulation necessarily produces higher mAP.

The audit JSON has SHA-256
`bd675be09c1f4c8ad29f50f495819f663b50e541bb08fb8953e18b83e5a6161d`;
the 45-row CSV has SHA-256
`36815e5358f9b4e4444002babda3d5a03e02abbc7293efe996cb62e06a404adf`.
The compact five-seed performance CSV has SHA-256
`ab688933790ef8cba496c1f968eb28f917f56ed9a12f0945d191359e03419a6c`.

## Stage-A decision and next control

RCRA passes Stage A on both performance and mechanism and is therefore retained
as the first candidate for final training. MtErie has still not been consulted.

Before spending the final five-seed full-data budget, one attribution question
remains valuable: much of P4/P5 behaves like level-wise residual scaling. A
three-parameter control with one learned scalar alpha per pyramid level, the
same exact-neutral initialization and the same dedicated LR can test whether
the gain requires local reliability descriptors or merely rescaling FAM. This
control should use the same Stage-A split and five paired seeds. It cannot
invalidate RCRA's observed result, but it determines whether the thesis can
attribute the gain to reliability-conditioned spatial selection rather than a
much simpler per-level calibration. The control is now implemented and frozen
in `notes/rtdetr_fam_scalar_alignment_control_stage_a.md`; its runtime probe is
complete (`3x1v3qfk`) and the five-seed campaign is the next operation.

After that control, freeze the selected architecture and proceed to Stage B:
full 4,019-frame training, ten epochs, five seeds, epoch-10 `latest`, and a
matched FAM comparison before the already-used MtErie benchmark.

## Thesis treatment after results

The thesis source remains unchanged for now. If reported, this experiment
belongs after the negative P2, resolution and post-fusion-gate ablations. The
method section should distinguish:

- established ideas: adaptive alignment, confidence-aware fusion and residual
  modulation;
- this implementation's contribution: exact-neutral, spatial selection between
  raw IR and the current FAM correction inside RT-DETR;
- experimental evidence: five paired seeds under the fixed Stage-A protocol;
- limitations: only one dataset, one held-out acquisition, ten epochs, and no
  claim of global novelty without a fuller literature review.

## Versioned artifacts

- implementation: `sarfusion/models/rtdetr_fusion.py`;
- optimizer isolation: `sarfusion/experiment/run.py`;
- five-seed configuration:
  `parameters/RTDETR/rtdetr_fam_residual_alignment_sequence_validation_five_seed.yaml`;
- runtime probe:
  `parameters/RTDETR/rtdetr_fam_residual_alignment_runtime_probe.yaml`;
- frozen alpha audit:
  `parameters/RTDETR/rtdetr_fam_residual_alignment_alpha_audit.yaml` and
  `scripts/run_rtdetr_fam_residual_alignment_alpha_audit.py`;
- performance table:
  `notes/Search_and_Rescue/results/rtdetr_fam_residual_alignment_stage_a_validation.csv`;
- alpha audit outputs:
  `notes/Search_and_Rescue/results/rtdetr_fam_residual_alignment_alpha_audit.json`
  and `notes/Search_and_Rescue/results/rtdetr_fam_residual_alignment_alpha_audit.csv`;
- regression tests: `tests/test_rtdetr_residual_alignment.py` and
  `tests/test_rtdetr_residual_alignment_alpha_audit.py`.
