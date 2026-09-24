# Configurazioni RT-DETR

Tengo una configurazione di riferimento per ciascun esperimento di training o valutazione. Per cambiare seed, dropout o learning rate modifico il relativo template.

Stage A usa la separazione per acquisizione tra training e validation. Stage B usa tutti i dati di training. Li mantengo separati perché cambia il protocollo, anche quando il modello è lo stesso. I nomi dei file conservano il riferimento agli esperimenti originali.

## Preparazione degli ambienti

Eseguo questi comandi dalla root del repository, con Conda disponibile.

### `sarfusion`: RT-DETR v1

Le dipendenze complete sono in [environment.yml](../../environment.yml), che specifica Python, PyTorch/CUDA e i pacchetti Conda e pip. La creazione dell'ambiente installa anche la sezione `pip:` del file, quindi non serve un secondo `requirements.txt` per questo ambiente. Il file descrive l'ambiente Linux usato nel progetto.

```bash
conda env create --file environment.yml
conda activate sarfusion
```

Se `sarfusion` è già stato creato e configurato, posso passare direttamente all'attivazione.

### `sarfusion-rtdetrv2`: RT-DETR v2

[requirements-rtdetrv2.txt](../../requirements-rtdetrv2.txt) contiene le versioni aggiuntive/sostitutive necessarie a RT-DETR v2, tra cui `transformers==4.51.3`. Va installato sopra una copia dell'ambiente completo `sarfusion`:

```bash
conda create --name sarfusion-rtdetrv2 --clone sarfusion
conda activate sarfusion-rtdetrv2
python -m pip install -r requirements-rtdetrv2.txt
```

Se la copia esiste già, salto `conda create` e uso i due comandi successivi. Questo requirements è un aggiornamento della base: da solo non installa tutte le dipendenze del progetto.

Prima del training o della valutazione attivo `sarfusion` per RT-DETR v1 oppure `sarfusion-rtdetrv2` per RT-DETR v2.

## Training

| Esperimento | Configurazione |
| --- | --- |
| Protocollo finale: Additive, FAM e ablation di dropout/SSJ/inizializzazione | [rtdetr_protocol.yaml](rtdetr_protocol.yaml) — le opzioni sono nell'intestazione |
| FAM con offset limitati | [rtdetr_fam_bounded4_protocol.yaml](rtdetr_fam_bounded4_protocol.yaml) |
| CMX | [fusion_rtdetr_cmx.yaml](fusion_rtdetr_cmx.yaml) |
| CMX hybrid | [fusion_rtdetr_cmx_hybrid.yaml](fusion_rtdetr_cmx_hybrid.yaml) |
| Tiling | [fusion_rtdetr_tile.yaml](fusion_rtdetr_tile.yaml) |
| FAM, Stage A | [rtdetr_fam_stage_a_five_seed_v2.yaml](rtdetr_fam_stage_a_five_seed_v2.yaml) |
| FAM, immagini 800 × 800 | [rtdetr_fam_800_sequence_validation_five_seed.yaml](rtdetr_fam_800_sequence_validation_five_seed.yaml) |
| FAM con livello P2 | [rtdetr_fam_p2_sequence_validation_seed40.yaml](rtdetr_fam_p2_sequence_validation_seed40.yaml) |
| FAM box-guided | [rtdetr_fam_box_guided_stage_a_five_seed_v2.yaml](rtdetr_fam_box_guided_stage_a_five_seed_v2.yaml) |
| FAM con mixed consistency | [rtdetr_fam_mixed_consistency_stage_a_five_seed_v2.yaml](rtdetr_fam_mixed_consistency_stage_a_five_seed_v2.yaml) |
| FAM con reliability gating | [rtdetr_fam_reliability_gate_lr10x_sequence_validation_five_seed.yaml](rtdetr_fam_reliability_gate_lr10x_sequence_validation_five_seed.yaml) |
| FAM con RCRA, Stage A | [rtdetr_fam_residual_alignment_sequence_validation_five_seed.yaml](rtdetr_fam_residual_alignment_sequence_validation_five_seed.yaml) |
| FAM con controllo scalare | [rtdetr_fam_scalar_alignment_control_sequence_validation_five_seed.yaml](rtdetr_fam_scalar_alignment_control_sequence_validation_five_seed.yaml) |
| FAM, Stage B | [rtdetr_fam_full_data_stage_b_five_seed.yaml](rtdetr_fam_full_data_stage_b_five_seed.yaml) |
| FAM con RCRA, Stage B | [rtdetr_fam_rcra_full_data_stage_b_five_seed.yaml](rtdetr_fam_rcra_full_data_stage_b_five_seed.yaml) |
| FAM con inizializzazione degli offset a zero | [rtdetr_fam_zero_offset_stage_a_five_seed.yaml](rtdetr_fam_zero_offset_stage_a_five_seed.yaml) |
| RT-DETR v2 Additive | [rtdetr_v2_additive_sequence_validation_five_seed_v2.yaml](rtdetr_v2_additive_sequence_validation_five_seed_v2.yaml) |
| RT-DETR v2 FAM | [rtdetr_v2_fam_sequence_validation_five_seed_v2.yaml](rtdetr_v2_fam_sequence_validation_five_seed_v2.yaml) |

Dalla root, nell'ambiente `sarfusion`:

```bash
python main.py experiment --parameters parameters/RTDETR/rtdetr_fam_stage_a_five_seed_v2.yaml
```

Prima di un nuovo esperimento aggiorno `experiment.name`, `group` e i tag W&B. Nei template con `other_grids`, ogni seed ha i corrispondenti `data_seed`, `model_seed` e `training_seed`: li modifico insieme. I manifest delle sorgenti presenti nei template verificano la versione del codice richiesta dal protocollo.

RT-DETR v2 richiede l'ambiente `sarfusion-rtdetrv2`. L'esperimento zero-offset usa invece il runner dedicato, che registra il modello e gestisce training e replay:

```bash
bash scripts/run_rtdetr_fam_zero_offset_stage_a.sh --dry-run
```

## Valutazione

Per un checkpoint locale uso [fusion_rtdetr_grid_test.yaml](fusion_rtdetr_grid_test.yaml): imposto `pretrained_path` e i parametri dell'architettura del checkpoint. Il template valuta VIS, IR e VIS+IR con soglia 0.01 e non esegue training.

```bash
python main.py experiment --parameters parameters/RTDETR/fusion_rtdetr_grid_test.yaml
```

Per la valutazione dei checkpoint bounded uso allo stesso modo [rtdetr_fam_bounded4_modality_evaluation.yaml](rtdetr_fam_bounded4_modality_evaluation.yaml).

I protocolli seguenti hanno runner dedicati in `scripts/`. Ciascun runner usa il proprio YAML predefinito e descrive le opzioni con `--help`.

| Valutazione | Configurazione | Runner |
| --- | --- | --- |
| Modalità sul protocollo finale | [rtdetr_paired_modality_evaluation.yaml](rtdetr_paired_modality_evaluation.yaml) | `run_rtdetr_paired_modality_evaluation.py` |
| Best e latest dello Stage A | [rtdetr_fam_sequence_validation_checkpoint_evaluation.yaml](rtdetr_fam_sequence_validation_checkpoint_evaluation.yaml) | `run_rtdetr_sequence_checkpoint_evaluation.py` |
| FAM e RCRA nello Stage B | [rtdetr_fam_rcra_full_data_stage_b_evaluation.yaml](rtdetr_fam_rcra_full_data_stage_b_evaluation.yaml) | `run_rtdetr_fam_rcra_full_data_stage_b_evaluation.py` |
| Modalità della FAM finale | [rtdetr_fam_full_data_paired_modality_evaluation.yaml](rtdetr_fam_full_data_paired_modality_evaluation.yaml) | `run_rtdetr_fam_full_data_paired_modality_evaluation.py` |
| Carnation | [rtdetr_carnation_stress_test.yaml](rtdetr_carnation_stress_test.yaml) | `run_rtdetr_carnation_stress_test.py` |
| Acquisizioni aggiuntive | [rtdetr_unused_acquisition_confirmation.yaml](rtdetr_unused_acquisition_confirmation.yaml) | `run_rtdetr_unused_acquisition_confirmation.py` |
| Perturbazioni geometriche | [rtdetr_synthetic_geometric_stress.yaml](rtdetr_synthetic_geometric_stress.yaml) | `run_rtdetr_synthetic_geometric_stress.py` |
| Recall/FPPI | [rtdetr_recall_fppi.yaml](rtdetr_recall_fppi.yaml) | `run_rtdetr_recall_fppi.py` |
| Costo computazionale | [rtdetr_additive_fam_compute_benchmark.yaml](rtdetr_additive_fam_compute_benchmark.yaml) | `run_rtdetr_compute_benchmark.py` |


## File di supporto

- [rtdetr_error_analysis_manifest.json](rtdetr_error_analysis_manifest.json): selezione dei frame per `scripts/run_rtdetr_error_analysis.py`.
- [rtdetr_temporal_validation_split.json](rtdetr_temporal_validation_split.json): inventario verificato delle immagini, ancora usato nella valutazione Stage B e nei test dello split temporale.
- [rtdetr_fam_zero_offset_stage_a_protocol.json](rtdetr_fam_zero_offset_stage_a_protocol.json): controlli di inizializzazione e confronti usati dal runner zero-offset.
