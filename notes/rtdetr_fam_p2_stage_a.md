# RT-DETR + FAM + P2: Stage-A ablation

Status: corrected and verified by real runtime probe; initial seed-40 launch
aborted and excluded, replacement launch pending

Defined: 2026-08-15

## Question under test

This ablation asks whether exposing a stride-4 feature level improves detection
of tiny people without changing input resolution or the RGB--IR fusion rule.
The control is the completed RT-DETR + FAM whole-sequence baseline. P2 is the
only architectural variable; reliability gating, a new alignment module and
higher input resolution are intentionally excluded.

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

The seed-40 run is a complete Stage-A experiment, not a shortened smoke test:

- train: FHL 0405/0406 plus Baker VIS/IR 1, 3,123 paired frames;
- validation: complete FHL 0401/0402 video, 896 paired frames;
- ten epochs exactly, without early stopping;
- primary checkpoint: highest validation mAP@50 with `min_delta = 0.001`;
- MtErie test disabled during training;
- seed 40 only;
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

This rule controls compute allocation, not the final scientific claim. Any
performance claim requires the same five seeds for P2 and the baseline. MtErie
must not be consulted to choose whether P2 advances.

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

The real P2 graph contains 123,608,847 parameters. The replacement seed-40 run
must start from scratch; resuming `vhusi1io` is forbidden.

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
micro-batches. The replacement seed-40 campaign may now start from scratch:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python main.py experiment \
  --parameters parameters/RTDETR/rtdetr_fam_p2_sequence_validation_seed40.yaml
```

The campaign YAML contains only seed 40, so this launches exactly one complete
run. Do not resume `vhusi1io`, do not add `--max-runs 1`, and do not run a
separate MtErie evaluation yet.

## Artifacts and thesis changes to apply later

- Training configuration:
  `parameters/RTDETR/rtdetr_fam_p2_sequence_validation_seed40.yaml`
- Operational probe configuration:
  `parameters/RTDETR/rtdetr_fam_p2_runtime_probe.yaml`
- Implementation: `sarfusion/models/rtdetr_fusion.py`,
  `sarfusion/models/detr.py` and `sarfusion/models/__init__.py`
- Regression tests: `tests/test_rtdetr_p2.py`

The thesis source is intentionally unchanged. If P2 reaches the five-seed
campaign, the methodology must document the stride-4 path, four-level FAM,
checkpoint remapping and controlled Stage-A comparison. The evaluation must
report P2 against the new FAM baseline under the identical split and selector,
then keep the later full-data Stage-B comparison separate.
