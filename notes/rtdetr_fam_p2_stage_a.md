# RT-DETR + FAM + P2: Stage-A ablation

Status: five P2 seeds completed; P2 recipe underperformed FAM on validation;
matched batch-2 FAM control pending

Defined: 2026-08-15

## Question under test

This ablation asks whether exposing a stride-4 feature level improves detection
of tiny people without changing input resolution or the RGB--IR fusion rule.
The intended control is the completed RT-DETR + FAM whole-sequence baseline.
Reliability gating, a new alignment module and higher input resolution are
intentionally excluded.

P2 did not fit the 8 GB GPU with the baseline's direct batch of four, so its
training used micro-batch two with two-step gradient accumulation. The effective
batch and number of optimizer updates are preserved, but RT-DETR's hybrid
encoder contains train-mode BatchNorm layers. Consequently, the completed
comparison tests the full P2 training recipe; a matched FAM batch-2 control is
required before the deficit can be attributed specifically to P2.

## Architecture

The existing model consumes backbone levels P3--P5, with strides 8, 16 and 32.
With `use_p2: true`, the model instead consumes P2--P5:

| Level | Backbone stride | R50-vd channels | Fusion |
|---|---:|---:|---|
| P2 | 4 | 256 | FAM-align IR to RGB, then additive fusion |
| P3 | 8 | 512 | unchanged FAM fusion |
| P4 | 16 | 1,024 | unchanged FAM fusion |
| P5 | 32 | 2,048 | unchanged FAM fusion |

The RT-DETR hybrid encoder, PAN/FPN path and decoder deformable attention are
expanded from three to four feature levels. Transformer encoding remains on
the deepest level, now index 3 (P5), matching the previous use of P5. The input
remains 640 px and FAM remains `current_dcnv2` at every level.

The public model flag defaults to `false`, so existing RT-DETR configurations
retain their original three-level graph.

## Pretrained initialization

Adding a level changes the semantic meaning of numeric module indices. A
shape-only load would incorrectly treat old P3 tensors as new P2 tensors. The
loader therefore remaps the COCO-pretrained checkpoint explicitly:

- pretrained P3--P5 projection, PAN and decoder tensors move to levels 1--3;
- the new P2 encoder projection starts as an exact channel identity;
- new P2 decoder/PAN operations start from the closest pretrained P3 operation;
- the extra P3-to-P2 top-down operation starts from the pretrained P4-to-P3
  operation;
- decoder deformable-attention parameters are expanded to four levels, with
  P2 initialized from the old P3 slice and the old P3--P5 slices shifted;
- R50-vd stage-1 weights are transferred to both modality backbones; the IR
  stem keeps the established channel-average initialization.

FAM has no COCO counterpart and follows the same initialization already used
by the baseline, including the newly introduced P2 FAM.

## Frozen Stage-A training protocol

Every retained P2 run is a complete Stage-A experiment, not a shortened smoke
test:

- train: FHL 0405/0406 plus Baker VIS/IR 1, 3,123 paired frames;
- validation: complete FHL 0401/0402 video, 896 paired frames;
- ten epochs exactly, without early stopping;
- primary checkpoint: highest validation mAP@50 with `min_delta = 0.001`;
- MtErie test disabled during training;
- seeds 40--44;
- batch size 2 and two-step gradient accumulation, preserving the baseline's
  effective batch size of 4 and approximately the same optimizer-step count;
- validation batch size 12, exactly as in the baseline;
- gradients and inactive training cache released before validation;
- inactive CUDA workspace cache released after every validation batch with
  `eval_cuda_cache_interval: 1`, while predictions retained by the evaluator
  remain live and unchanged;
- evaluator state and inactive cache released before returning to training.

All remaining optimizer, augmentation, modal-dropout, preprocessing and class
head settings are copied from the baseline configuration.

The predeclared engineering triage compares validation `best` with baseline
seed 40 (`0.1521` mAP@50):

- P2 at least `+0.01`: expand to seeds 41--44;
- P2 within `-0.01` to `+0.01`: run seed 41 before deciding;
- P2 below `-0.01`: do not expand immediately; first inspect optimization,
  per-size metrics and the P2/FAM activations.

This rule controls compute allocation, not the final scientific claim. Although
seed 40 fell below the stop threshold, seeds 41--44 were launched before the
aggregate analysis and therefore constitute a documented deviation from the
compute-triage rule. They are nevertheless complete runs under the same P2
protocol and provide the required five-seed estimate. MtErie was not consulted.

## Initial launch, failure and correction

The first pre-launch memory check measured an isolated validation forward. It
did not retain AdamW state, gradients or repeated CUDA workspaces and was
therefore insufficient. The initial run `vhusi1io` exposed the error:

| Phase | Elapsed | Throughput |
|---|---:|---:|
| Epoch-1 train | 18:53 | 1.38 batch/s |
| Epoch-1 validation | 35:54 | 28.73 s/batch |
| Epoch-2 train | 42:14 | 1.62 s/batch |

The run was interrupted during epoch-2 validation. It is an operational probe,
not a campaign result: it is incomplete, its epoch-1 validation mAP@50 of
`0.0841` must not enter any comparison, and its checkpoints remain only as
diagnostic evidence.

A realistic CUDA lifecycle probe then kept the trained model and AdamW state
resident. The relevant measurements were:

- one batch-2 train step: 5.26 GiB peak allocated and 6.63 GiB reserved;
- after clearing completed-step gradients: 1.47 GiB allocated and 2.42 GiB
  reserved before validation;
- one batch-12 validation forward: 1.32 s, 6.81 GiB peak allocated and 7.11
  GiB reserved;
- repeated batch-12 validation without per-batch cleanup: severe slowdown and
  9.75 GiB reserved after ten batches, beyond physical VRAM under WSL;
- 25 consecutive batch-12 validations with per-batch inactive-cache cleanup:
  300 images in 28.00 s, 2.42 GiB reserved after cleanup and no growth;
- mAP state computation after those batches: 0.16 s;
- a subsequent batch-2 train step after validation: 0.79 s, with 1.47 GiB
  allocated and 2.42 GiB reserved before it.

A `nvidia-smi` reading near 7.8 GiB is not, by itself, proof of a problem:
other workloads may use that amount normally. Here the diagnosis rests on the
paired evidence of throughput collapse and PyTorch's reserved-memory counter
growing to 9.75 GiB. Releasing only inactive cached workspaces after each batch
removes that growth without modifying model state, predictions or evaluator
state.

Batch 4 was also tested and was faster, but not adopted. RT-DETR's marginal
top-k query set showed small numerical batch-shape sensitivity; retaining batch
12 preserves the exact validation setting used by the baseline.

## Verification status

The corrected implementation now covers:

- four-level feature shapes and full detector forward;
- explicit pretrained level remapping;
- unchanged default P3--P5 path;
- real R50-vd model construction;
- train-to-validation-to-train CUDA lifecycle;
- repeated batch-12 inference with the real mAP evaluator;
- unit checks for gradient, evaluator-state and CUDA-cache cleanup.

The real P2 graph contains 123,608,847 parameters. The replacement runs started
from scratch; the aborted `vhusi1io` run is excluded.

## Completed five-seed validation result

All five retained runs completed ten epochs. The table compares the selected
validation checkpoint against the already completed FAM baseline under the same
whole-sequence split and checkpoint selector.

| Seed | FAM best epoch | FAM best mAP@50 | P2 best epoch | P2 best mAP@50 | P2 latest mAP@50 | P2 - FAM |
|---:|---:|---:|---:|---:|---:|---:|
| 40 | 1 | 0.1521 | 5 | 0.1014 | 0.0649 | -0.0508 |
| 41 | 4 | 0.1424 | 2 | 0.1226 | 0.0230 | -0.0197 |
| 42 | 6 | 0.1655 | 2 | 0.0835 | 0.0592 | -0.0820 |
| 43 | 1 | 0.1939 | 7 | 0.1145 | 0.0214 | -0.0794 |
| 44 | 1 | 0.1689 | 1 | 0.0733 | 0.0416 | -0.0956 |

FAM obtains `0.1646 +/- 0.0196`; P2 obtains `0.0991 +/- 0.0207`. The paired
delta is `-0.0655 +/- 0.0304`, negative in 5/5 seeds, with a two-sided Student-t
95% confidence interval of `[-0.1032, -0.0278]`. Thus the P2 recipe fails Stage
A and must not be evaluated on MtErie or promoted to full-data Stage B.

This is not yet a clean estimate of the P2 architectural effect because FAM
used batch four while P2 used batch two plus accumulation. The frozen follow-up
is one FAM seed-40 control with P2 disabled and the P2 batch/accumulation recipe:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python main.py experiment \
  --parameters parameters/RTDETR/rtdetr_fam_sequence_validation_batch2_control_seed40.yaml
```

The decision is fixed before seeing that result:

- if matched FAM exceeds P2 seed 40 by more than `0.02` mAP@50, the substantial
  deficit remains P2-specific and P2 is closed;
- if matched FAM lies within `+/-0.02` of P2 seed 40, micro-batch effects are a
  plausible explanation and a second matched control is required;
- an unexpected matched-FAM score more than `0.02` below P2 triggers a pipeline
  audit rather than model promotion.

## Launch order

The operational probe is reproducible with:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python main.py experiment \
  --parameters parameters/RTDETR/rtdetr_fam_p2_runtime_probe.yaml
```

It performs 20 training micro-batches and the complete 75-batch validation,
saves no checkpoint and is tagged `ExcludeFromCampaign`. Its predeclared
acceptance conditions were:

- exactly 20 training iterations and 75 validation iterations;
- validation wall time below ten minutes;
- no progressive multi-second-to-tens-of-seconds slowdown across validation;
- clean process termination and release of the CUDA context.

The real probe `4d6r2p0f` passed all four conditions on 2026-08-15:

- training: 20/20 micro-batches in 19 s;
- validation: 75/75 batches in 3:18, 2.65 s/batch overall;
- late batches remained near 2.44--2.6 s, excluding progressive degradation;
- exit code 0, no `best/` or `latest/` checkpoint, no orphan worker, and 0 MiB
  VRAM after exit.

The probe's mAP is not a performance result because it trained for only 20
micro-batches. The retained seed-40 campaign was launched from scratch with:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python main.py experiment \
  --parameters parameters/RTDETR/rtdetr_fam_p2_sequence_validation_seed40.yaml
```

The versioned campaign YAML contains seed 40. Seeds 41--44 were run from
temporary copies that changed only the seed value. Their retained W&B run IDs
are `lbrr41te`, `61wyomy4`, `aflizsb0` and `y4q7b3sz`; seed 40 is `k0uugy3n`.

## Artifacts and thesis changes to apply later

- Training configuration:
  `parameters/RTDETR/rtdetr_fam_p2_sequence_validation_seed40.yaml`
- Operational probe configuration:
  `parameters/RTDETR/rtdetr_fam_p2_runtime_probe.yaml`
- Matched batch-2 FAM control:
  `parameters/RTDETR/rtdetr_fam_sequence_validation_batch2_control_seed40.yaml`
- Five-seed validation table:
  `notes/Search_and_Rescue/results/rtdetr_fam_p2_stage_a_validation.csv`
- Implementation: `sarfusion/models/rtdetr_fusion.py`,
  `sarfusion/models/detr.py` and `sarfusion/models/__init__.py`
- Regression tests: `tests/test_rtdetr_p2.py`

The thesis source is intentionally unchanged. Its later revision should report
P2 as a negative Stage-A ablation, including the five paired validation deltas
and the matched-batch control outcome. It must not report the aborted run or the
runtime probe as performance evidence, and must keep any later full-data Stage-B
comparison separate.
