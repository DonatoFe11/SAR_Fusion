# Artefatti sperimentali versionati

Questa directory contiene risultati compatti destinati alla tesi. Gli output
grezzi completi restano in `out/`, che è ignorata da Git, per evitare di
versionare decine di megabyte di predizioni frame-level.

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
