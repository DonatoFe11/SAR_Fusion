# RT-DETR + FAM at 800x800: Stage-A resolution ablation

Status: protocol and operational verification complete; five scientific seeds
pending

Defined: 2026-08-22, before any 800x800 scientific run

## Question and motivation

This experiment tests whether increasing input resolution improves detection
of the very small people in FHL without adding P2 or another fusion module. It
retains the `current_dcnv2` FAM baseline and changes the RT-DETR processor from
`640x640` to `800x800`. P2 and reliability gating remain disabled.

The P2 experiment tested a stride-4 feature level and failed over five seeds;
it did not test whether simply preserving more input pixels helps the existing
P3--P5 detector. Resolution is therefore a distinct ablation.

## Frozen protocol

- five seeds `40--44`, run sequentially;
- same 3,123-frame training split and 896-frame FHL validation sequence;
- exactly ten epochs, no early stopping;
- `best` selected by validation mAP@50 with `min_delta = 0.001`;
- input size exactly `800x800`, verified at the processor output;
- training micro-batch 2 with two-step gradient accumulation, effective batch 4;
- validation batch 8;
- same AdamW optimizer and LR `2e-5`;
- no P2, reliability gate, tiling, MtErie or automatic final test.

The smaller micro-batch is required by the 8 GB GPU. A completed 640x640 FAM
control using micro-batch 2 and accumulation 2 obtained `0.1593` on seed 40,
versus `0.1521` for direct-batch-four FAM. This does not establish equivalence
over all seeds, but it provides no evidence that the necessary batching choice
itself causes the large deficits observed with P2.

## Decision rule

The primary comparison is paired against the existing five-seed FAM Stage-A
baseline (`0.1646 +/- 0.0196`). Before observing the new results:

- promote only if the mean paired mAP@50 gain is at least `+0.01` and positive
  in at least 4/5 seeds;
- treat a smaller positive mean or 3/5 wins as inconclusive, not as a claimed
  improvement;
- close if the mean delta is non-positive or fewer than 3/5 seeds improve.

The confidence interval and all `best/latest` pairs will also be reported, but
the interval will not replace the predeclared engineering threshold. MtErie is
not consulted during this decision.

## Launch

The checkpoint-free W&B probe `0tikq6eh`, tagged `ExcludeFromCampaign`, passed:

- production dataloader output `(2, 4, 800, 800)` and mask `(2, 800, 800)`;
- 20/20 training micro-batches with two-step accumulation;
- 112/112 validation batches at evaluation batch 8 in approximately 1:47;
- exit code 0, no OOM and no campaign checkpoint;
- architecture and all optimizer-group LRs identical to FAM.

Its validation mAP@50 (`0.1214`) is not a scientific result because the model
saw only 20 micro-batches. Launch all five scientific seeds with:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python main.py experiment \
  --parameters parameters/RTDETR/rtdetr_fam_800_sequence_validation_five_seed.yaml
```

## Thesis treatment

The thesis source is intentionally unchanged. This experiment should be
reported as a resolution ablation motivated by tiny targets, not as a new
architecture. Its value is to distinguish lack of spatial input detail from
limitations of P2 and reliability gating.
