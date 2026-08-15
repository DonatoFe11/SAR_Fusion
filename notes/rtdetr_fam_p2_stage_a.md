# RT-DETR + FAM + P2: Stage-A ablation

Status: implemented and locally verified; seed-40 training not yet launched

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
- validation batch size 12, as in the baseline.

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

## Local verification

The following checks completed successfully before launch:

- unit forward through a four-level miniature detector;
- feature shapes at strides 4, 8, 16 and 32;
- pretrained level-remapping and deformable-attention expansion;
- regression check that the default P3--P5 path is unchanged;
- offline construction of the real `PekingU/rtdetr_r50vd` graph;
- one 640-px CUDA training step including backward and AdamW update;
- 640-px CUDA evaluation including post-processing.

The real P2 graph contains 123,608,847 parameters. On the local RTX 4070 8 GB,
the synthetic batch-2 training step peaked at about 5.26 GiB allocated and
6.63 GiB reserved. Batch-12 evaluation peaked at about 5.84 GiB allocated and
6.34 GiB reserved. These are smoke-test measurements, not guarantees for every
dataset batch, but they leave enough measured margin for the frozen settings.

## Launch

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python main.py experiment \
  --parameters parameters/RTDETR/rtdetr_fam_p2_sequence_validation_seed40.yaml
```

The YAML contains only seed 40, so this command launches exactly one run. Do
not add `--max-runs 1`, and do not run a separate MtErie evaluation yet.

## Artifacts and thesis changes to apply later

- Training configuration:
  `parameters/RTDETR/rtdetr_fam_p2_sequence_validation_seed40.yaml`
- Implementation: `sarfusion/models/rtdetr_fusion.py`,
  `sarfusion/models/detr.py` and `sarfusion/models/__init__.py`
- Regression tests: `tests/test_rtdetr_p2.py`

The thesis source is intentionally unchanged. If P2 reaches the five-seed
campaign, the methodology must document the stride-4 path, four-level FAM,
checkpoint remapping and controlled Stage-A comparison. The evaluation must
report P2 against the new FAM baseline under the identical split and selector,
then keep the later full-data Stage-B comparison separate.
