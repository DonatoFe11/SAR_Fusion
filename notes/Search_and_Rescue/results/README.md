# Artefatti sperimentali versionati

Questa directory contiene risultati compatti destinati alla tesi. Gli output
grezzi completi restano in `out/`, che è ignorata da Git, per evitare di
versionare decine di megabyte di predizioni frame-level.

## RT-DETR + FAM: selezione `best` contro `latest`

Il file
[`rtdetr_fam_sequence_checkpoint_evaluation.csv`](rtdetr_fam_sequence_checkpoint_evaluation.csv)
contiene 10 righe: cinque seed e i due checkpoint `best` e `latest`. Tutti i
checkpoint sono valutati sui medesimi 708 frame VIS+IR MtErie con ground truth
VIS. `best` è selezionato esclusivamente dalla validation FHL 0401/0402; MtErie
è usato solo a posteriori e non modifica la selezione.

Questa è la baseline di sviluppo per scegliere l'architettura. Il modello
selezionato verrà poi ritrainato su tutti i 4.019 frame per dieci epoche e
confrontato tramite `latest` con una baseline FAM full-data configurata allo
stesso modo; i risultati delle due fasi non saranno confrontati direttamente.

Il risultato primario `best` è `0.3590 +/- 0.0416` mAP@50, contro
`0.2689 +/- 0.0317` per `latest`. Il delta appaiato è `+0.0901 +/- 0.0466`,
positivo in 5/5 seed, IC95% `[+0.0323, +0.1479]`. Protocollo, limiti e
confrontabilità con la campagna storica sono documentati in
[`../../rtdetr_sequence_validation_fixed10_protocol.md`](../../rtdetr_sequence_validation_fixed10_protocol.md).

L'aggregato JSON locale completo è marcato `protocol_complete: true` e ha
SHA-256:

```text
1402142280d299d94bffc8628a756e6d15d42867c260425bcce6c27bfd80357e
```

## RT-DETR + FAM full-data: caratterizzazione appaiata delle modalità

Il file
[`rtdetr_fam_full_data_paired_modality_evaluation.csv`](rtdetr_fam_full_data_paired_modality_evaluation.csv)
contiene i cinque checkpoint FAM Stage-B selezionati e le tre condizioni
VIS+IR, VIS mascherato e IR mascherato con ground truth VIS, valutate sugli
stessi 708 frame e sulle stesse annotazioni VIS. La fusione supera VIS in 5/5
seed: il delta appaiato mAP@50 è
`+0.0373 +/- 0.0223`, IC95% `[+0.0095, +0.0650]`.

Questa è una caratterizzazione post-selezione e non riapre la scelta del
modello. Il valore IR mascherato `0.0215` misura robustezza senza il riferimento
RGB nel sistema di coordinate VIS e non va chiamato prestazione IR nativa. Il
protocollo e le statistiche complete sono documentati in
[`../../rtdetr_fam_full_data_paired_modality_evaluation.md`](../../rtdetr_fam_full_data_paired_modality_evaluation.md).
L'aggregato JSON locale contiene 15 unità sperimentali, è marcato
`protocol_complete: true` e ha SHA-256:

```text
28b767752de3b744e529dd7d281178a91cf08b5834414d772e706fda0012ddb5
```

Il controllo post-hoc
[`rtdetr_fam_full_data_native_ir_coordinate_diagnostic.csv`](rtdetr_fam_full_data_native_ir_coordinate_diagnostic.csv)
usa le stesse 708 controparti IR con preprocessing e 1.824 box IR nativi. La
media `0.5618 +/- 0.0498` dimostra che il ramo termico non è collassato, ma non
è direttamente confrontabile con VIS+IR perché cambia sistema di coordinate e
ground truth. Il relativo aggregato locale è marcato
`protocol_status: post_hoc_diagnostic`, `protocol_complete: true` e ha SHA-256:

```text
7e0fce162d298f303a4bb602379d2b78df7ac420f65335df5851d0f85eaca034
```

## RT-DETR + FAM: probe Modal Dropout paired-VIS

Il file
[`rtdetr_fam_paired_vis_modal_dropout_probe_evaluation.csv`](rtdetr_fam_paired_vis_modal_dropout_probe_evaluation.csv)
confronta i `best` seed 40 della baseline e del training con IR-only adattato
al canvas VIS. Le quattro condizioni usano gli stessi 896 frame FHL: tre
interventi paired con ground truth VIS e un controllo IR nativo con ground
truth IR.

Il candidato mantiene la fusion (`+0.00475` mAP@50), ma il masked-IR paired
migliora di appena `+0.00032` e l'IR nativo crolla di `-0.23664`. Fallisce
quindi due dei tre criteri congelati e il pure paired-VIS replacement viene
chiuso senza altri seed, MtErie o Stage B. Protocollo e interpretazione sono in
[`../../rtdetr_fam_paired_vis_modal_dropout_probe.md`](../../rtdetr_fam_paired_vis_modal_dropout_probe.md).
L'aggregato locale è marcato `protocol_complete: true` e ha SHA-256:

```text
4c93cc43f4bb3c4fcf8e6156d947981ac3ebd0ae89d97f76e23542630c87ac03
```

Il CSV compatto versionato ha SHA-256:

```text
4eca3eda8bb7113c896b801ae55fa27bee01b357b6b51fc5d70f695f4a692f68
```

## RT-DETR Additive--FAM error analysis

Il file
[`rtdetr_error_analysis_checkpoints.csv`](rtdetr_error_analysis_checkpoints.csv)
contiene 50 righe: due configurazioni, cinque seed e cinque soglie di
confidenza. Ogni riga è un riepilogo su tutti i 708 frame MtErie. Le definizioni
delle metriche e l'interpretazione sono documentate in
[`../../rtdetr_error_analysis.md`](../../rtdetr_error_analysis.md).

La campagna è stata prodotta dal protocollo
`rtdetr_additive_fam_error_analysis_v1` mediante
[`../../../scripts/run_rtdetr_error_analysis.py`](../../../scripts/run_rtdetr_error_analysis.py)
e il manifest GT-only
[`../../../parameters/RTDETR/rtdetr_error_analysis_manifest.json`](../../../parameters/RTDETR/rtdetr_error_analysis_manifest.json).
L'aggregato locale completo contiene 35.400 righe frame/soglia, è marcato
`protocol_complete: true` e ha SHA-256:

```text
21d13ff4d80e51835ce2f0593a50fa158492454760b811e9e0f64e7877226be0
```

Le figure versionate in `../images/` sono:

- `rtdetr_additive_fam_error_summary.png`: confronto appaiato alla soglia
  primaria `0.01`;
- `rtdetr_additive_fam_threshold_sensitivity.png`: medie e deviazioni standard
  sulle cinque soglie predefinite;
- `rtdetr_error_qualitative_conf_001.jpg` e
  `rtdetr_error_qualitative_conf_025.jpg`: tutti i sei frame selezionati dal
  manifest, senza selezione post-hoc basata sulle predizioni.

Nelle due tavole qualitative, ogni riga corrisponde a una sequenza MtErie e
mostra il frame small-target e il frame vuoto fissati prima dell'inferenza. Le
figure individuali alla risoluzione originale restano nell'output locale.

## YOLOv10 Additive--FAM per modalità

Il file
[`yolo_final_modality_evaluation.csv`](yolo_final_modality_evaluation.csv)
contiene 30 righe: due configurazioni, cinque seed e tre condizioni VIS+IR,
VIS-only e IR-only. Tutti i valori sono stati calcolati con lo stesso evaluator
standalone sui 708 frame MtErie. La figura
`../images/yolov10_final_modality_paired_map50.png` visualizza i confronti
appaiati per seed.

La campagna usa il protocollo `yolov10_final_modality_evaluation_v1`; dettagli,
risultati e audit dello scarto marginale fra evaluator sono in
[`../../yolo_final_modality_evaluation.md`](../../yolo_final_modality_evaluation.md).
L'aggregato locale completo contiene 30 unità sperimentali, è marcato
`protocol_complete: true` e ha SHA-256:

```text
bcfc3a16f27e32f8d42da9c48b0ee2c87ef1eccfc8e93bd144f38ba73eea98b7
```

## RT-DETR Additive--FAM: stress test Carnation

Il file
[`rtdetr_carnation_stress_test.csv`](rtdetr_carnation_stress_test.csv)
contiene 30 righe: due configurazioni, cinque seed e tre condizioni VIS+IR,
VIS-only e IR-only. Ogni valutazione usa gli stessi 739 identificatori di
frame comuni alle sequenze Carnation VIS 0023 e IR 0024. La figura
`../images/rtdetr_carnation_paired_map50.png` mostra i confronti appaiati.

Il protocollo `rtdetr_carnation_stress_test_v1` è stato congelato e versionato
prima dell'inferenza. Carnation è un caso mirato di forte mismatch di scala,
non una sostituzione del benchmark interno MtErie e non una sorgente di tuning.
Dettagli e interpretazione sono in
[`../../rtdetr_carnation_stress_test.md`](../../rtdetr_carnation_stress_test.md).

L'aggregato locale completo contiene 30 unità sperimentali, è marcato
`protocol_complete: true` e ha SHA-256:

```text
00916ece2484062b9c591602dd9ccf9129546965213465cbf084b1a12e7484c4
```

## RT-DETR Additive--FAM: costo computazionale

Il file [`rtdetr_compute_benchmark.csv`](rtdetr_compute_benchmark.csv)
contiene il riepilogo delle due configurazioni finali: parametri, dimensione
dello stato, proxy GFLOPs, latenza, throughput e memoria CUDA. Ogni latenza
deriva da tre trial in processi isolati, 100 forward per trial, batch 1 e input
FP32 `[1, 4, 640, 640]` su RTX 4070 Laptop GPU. Preprocessing e postprocessing
sono esclusi.

Il protocollo `rtdetr_additive_fam_compute_benchmark_v1` è stato congelato e
pushato prima della misura. Metodo, limiti del conteggio DCNv2 e interpretazione
sono documentati in
[`../../rtdetr_compute_benchmark.md`](../../rtdetr_compute_benchmark.md). Il
JSON locale completo è marcato `protocol_complete: true` e ha SHA-256:

```text
cb942c8876f763d17b14bfc40a0e3371efd10c907266ab8cd9b2c41ca5902cbb
```
