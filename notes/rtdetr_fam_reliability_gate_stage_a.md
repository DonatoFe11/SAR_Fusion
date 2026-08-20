# RT-DETR + FAM + reliability gating: Stage-A ablation

Status: implemented, regression-tested and verified by real runtime probe;
complete seed-40 Stage-A training pending

Defined: 2026-08-20

## Question under test

This ablation asks whether explicitly estimating local modality reliability
improves the existing RT-DETR + FAM detector. The P3--P5 pyramid, FAM alignment,
input resolution, optimizer, data split and direct training batch of four remain
unchanged. P2 is disabled. No new alignment mechanism is introduced, so the
only architectural variable is the reliability gate applied after FAM.

This is a project-specific candidate architecture. Any claim of broader
architectural novelty requires a dedicated literature comparison and is not
assumed by this experiment.

## Reliability-gated fusion

For each of P3, P4 and P5, FAM first produces the aligned IR feature `I_a` from
RGB feature `R` and raw IR feature `I`. A lightweight gate then constructs seven
spatial descriptors:

1. channel mean of `R`;
2. channel RMS of `R`;
3. channel mean of `I_a`;
4. channel RMS of `I_a`;
5. channelwise cosine agreement between `R` and `I_a`;
6. binary RGB-presence indicator;
7. binary IR-presence indicator.

The presence indicators are derived from the exact zero padding already used
by RT-DETR modal dropout. They are inputs to the learned gate, not hard-coded
output masks.

The seven descriptors pass through a 3x3 convolution with 16 hidden channels,
SiLU and a 1x1 convolution producing two spatial logits. Fusion is:

```text
w_R = 2 sigmoid(l_R)
w_I = 2 sigmoid(l_I)
F   = w_R * R + w_I * I_a
```

The modality weights are independent and lie in `(0, 2)`, so either modality
can be suppressed without forcing amplification of the other. The final 1x1
convolution is initialized to exactly zero; therefore `w_R = w_I = 1` and the
initial output is exactly the previous additive FAM fusion. This initialization
remains restored after Hugging Face `post_init` and pretrained transfer.

Each gate has 1,058 trainable parameters. Three levels add 3,174 parameters in
total. The gate uses the same `2e-5` learning rate as the rest of this frozen
protocol.

## Frozen Stage-A protocol and decision rule

- train: FHL 0405/0406 plus Baker VIS/IR 1, 3,123 paired frames;
- validation: complete FHL 0401/0402 video, 896 paired frames;
- ten epochs exactly, without early stopping;
- direct batch size 4, identical to the FAM baseline;
- validation batch size 12;
- primary checkpoint: highest validation mAP@50 with `min_delta = 0.001`;
- MtErie disabled;
- first run: seed 40 only.

The comparison target is FAM seed 40, `0.1521` best validation mAP@50. Before
seeing the gating result, the engineering triage is frozen as:

- gate at least `+0.01` (at least `0.1621`): expand to seeds 41--44;
- gate within `-0.01` to `+0.01` (`0.1421`--`0.1621`): run seed 41 before
  deciding;
- gate below `-0.01` (below `0.1421`): do not expand; inspect optimization and
  learned gate weights first.

This rule controls compute allocation. A scientific performance claim still
requires five seeds. MtErie must not be consulted to decide whether the gate
advances.

## Verification

Automated tests cover:

- exact additive equivalence at neutral initialization;
- finite gradients and the ability to leave the neutral solution;
- explicit response to modality-presence descriptors;
- three P3--P5 gates and rejection of gating without FAM;
- propagation through the full detector and public model factory;
- unchanged split, batch size, optimizer budget and disabled MtErie;
- a checkpoint-free runtime-probe configuration excluded from the campaign.

The complete repository suite passes 118/118 tests.

The real operational probe is W&B run `6r7eq72h`, tagged
`ExcludeFromCampaign`. It completed:

- 20/20 real batch-4 training iterations in approximately 22 s;
- 75/75 validation batches in 1:13;
- stable late-validation throughput near 0.8 s/batch;
- exit code 0;
- no `best`, `latest`, safetensors or PyTorch checkpoint file.

Its validation mAP is not a scientific result because the model received only
20 training batches.

## Launch

After committing the implementation, launch exactly one complete seed with:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python main.py experiment \
  --parameters parameters/RTDETR/rtdetr_fam_reliability_gate_sequence_validation_seed40.yaml
```

Do not run MtErie and do not launch seeds 41--44 until the seed-40 validation
result has been checked against the frozen rule.

## Artifacts and later thesis changes

- Implementation: `sarfusion/models/rtdetr_fusion.py`
- Public parameter plumbing: `sarfusion/models/detr.py` and
  `sarfusion/models/__init__.py`
- Stage-A configuration:
  `parameters/RTDETR/rtdetr_fam_reliability_gate_sequence_validation_seed40.yaml`
- Operational probe:
  `parameters/RTDETR/rtdetr_fam_reliability_gate_runtime_probe.yaml`
- Regression tests: `tests/test_rtdetr_reliability_gating.py`

The thesis source is intentionally unchanged. If the gate reaches five seeds,
the methodology should report its descriptors, neutral initialization,
per-level spatial weights and controlled P3--P5 comparison. The evaluation
should include both detection performance and a post-hoc audit showing whether
the learned weights respond meaningfully to fusion, RGB-only and IR-only input.
