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

## RT-DETR: conferma su acquisizioni WiSARD inutilizzate

Il file
[`rtdetr_unused_acquisition_confirmation.csv`](rtdetr_unused_acquisition_confirmation.csv)
contiene le 50 valutazioni congelate su Carnation 0025/0026 e FHL 0407/0408.
Le acquisizioni non erano state usate nelle campagne conservate e l'autore ha
attestato di non averle visionate manualmente prima del protocollo. FAM storico
supera Additive in 5/5 seed su entrambe: delta mAP@50 medio `+0.0960` su
Carnation e `+0.1554` su FHL. Le diagnostiche fusion--VIS e RCRA--FAM non sono
uniformi e non riaprono la selezione.

Protocollo, audit del pairing, statistiche e limiti sono in
[`../../rtdetr_unused_acquisition_confirmation.md`](../../rtdetr_unused_acquisition_confirmation.md).
Il CSV versionato ha SHA-256:

```text
a314bbf1d5eb7ffce296945a4f892231c2cbb713ba1436295c9941a73b4233c8
```

## RT-DETR: stress geometrico sintetico controllato

Il file
[`rtdetr_synthetic_geometric_stress.csv`](rtdetr_synthetic_geometric_stress.csv)
contiene 600 punti: 40 identità riusate e 560 inferenze nelle quali soltanto il
canale IR è traslato o riscalato. La figura
[`../images/rtdetr_synthetic_geometric_stress_curves.png`](../images/rtdetr_synthetic_geometric_stress_curves.png)
mostra le curve medie sulle quattro direzioni e la macro-media a peso uguale
delle due acquisizioni.

La risposta con segno non conferma una tolleranza geometrica universalmente
superiore di FAM. Su FHL varie perturbazioni migliorano accidentalmente i
modelli, soprattutto Additive, segnalando che lo stress può compensare un
mismatch nativo e non equivale a una calibrazione. RCRA non ottiene un
vantaggio stabile su FAM. Metodo, contrasti appaiati e vincoli interpretativi
sono in
[`../../rtdetr_synthetic_geometric_stress.md`](../../rtdetr_synthetic_geometric_stress.md).

SHA-256 degli artefatti versionati:

```text
CSV:    f37b984c61e9aab51afa7706d57407d898048c9feb9691a5fb5e40d3405613ef
Figura: c94eb817972c06abbfe2c742c69432c7c140d8343988947424bf11b2808b5854
```

## RT-DETR FAM: screen mixed consistency seed 40

Il file
[`rtdetr_fam_mixed_consistency_probe_evaluation.csv`](rtdetr_fam_mixed_consistency_probe_evaluation.csv)
contiene le otto valutazioni del gate Stage A: FAM baseline e candidato seed
40 nelle condizioni fusion, VIS-only, paired masked-IR con ground truth VIS e
IR nativa con ground truth IR, sempre sugli stessi 896 frame FHL.

Il candidato fallisce tutti i gate congelati: delta mAP@50 fusion `-0,026884`,
paired masked-IR `+0,001829` e IR nativa `-0,050581`. È chiuso dopo un seed;
non sono autorizzati seed 41--44, Stage B o valutazioni MtErie. Tempi di
training, checkpoint e interpretazione sono in
[`../../rtdetr_fam_mixed_consistency_stage_a.md`](../../rtdetr_fam_mixed_consistency_stage_a.md).

Il CSV versionato ha SHA-256:

```text
5cb36e63d7a5758215e9fa4cf431e3ad4e56d7b285259dd00466d7627b608d65
```

## RT-DETR FAM Box-Guided P3: inventario e probe tecnico

La variante `box_guided_common_offset_p3` aggiunge a P3 un campo comune
`(dy, dx)` debolmente supervisionato dai box appaiati e lascia P4/P5 come FAM
storico. Il ramo aggiunge 53.410 parametri. Un fix precedente al training rende
bit-identici, a parità di seed, sia tutti i pesi FAM condivisi sia lo stato RNG
globale rispetto al FAM di controllo.

I quattro YAML scientifici dichiarano il manifest fail-closed
`rtdetr_box_guided_training_source_v1`: 22 file critici, SHA-256 aggregato
`b06ea1328be206a9f7c64b3412f64ed7bb95b884da591c476584a90403592412`.
Il training verifica i byte prima di partire e registra hash aggregato e
per-file nel primo evento della trace; gli audit legano la trace al checkpoint,
alla configurazione e ai sorgenti correnti. Il probe tecnico, eseguito prima di
questo vincolo, non è una run scientifica provenance-bound.

L'inventario congelato del train Stage A comprende 3.123 frame, 5.209 match e
2.306 frame con almeno un match. La distribuzione match/frame è `0:817`,
`1:801`, `2:543`, `3:526`, `4:436`. Lo SHA-256 della serializzazione canonica
dei path VIS/IR e dei tensori `float32 [x_VIS, y_VIS, dy, dx]` è:

```text
d519574962e81ae5b492248113247cca20d7ef15b2d189d1e3b58aebf218f3c0
```

Il probe tecnico non scientifico è completato nella run `j37qaj8r`, directory
locale
[`wandb/run-20260831_122623-j37qaj8r`](../../../wandb/run-20260831_122623-j37qaj8r/).
Ha eseguito 20 step più la validation completa di 896 frame in 132 secondi:
loss media `17,44285`, valori finiti, ultima loss guida raw `1,18184`, pesata
`0,11818`, scala `0,5`; la validation mAP@50 `0,07828` è esclusa da ogni
confronto scientifico.

Il controfattuale validation è stato congelato prima dell'inferenza sui soli
896 frame appaiati FHL 0401/0402. Gli SHA-256 di inventario storico, inventario
forte comprensivo dei byte delle immagini e ordine dei campioni sono,
rispettivamente, `47e2f348ebdc202cb749b1bbf2741fc868d8681eb24ffa4356cf4f584dfa4ec4`,
`6c7748af3be2761a3a466b548af64aae925b693fbca795edf695072e28f17141` e
`49415f065575c869087c78f842591096b74a0ea3a16ca2e4ce765e26958badcd`.
Lo screen scientifico seed 40 è completato. Il matched control `2fx2ozwm`
ottiene `0,147388741` mAP@50 al best epoch 3; la candidata `2jvqs9mr` ottiene
`0,155485332` al best epoch 1. Il delta `+0,008096591` è positivo ma inferiore
al gate preregistrato `+0,01`, quindi la candidata è chiusa e seed 41--44 e
Stage B non sono autorizzati.

L'[`audit meccanicistico`](rtdetr_fam_box_guided_mechanism_audit_v1.json)
passa tutti i controlli: la Smooth L1 è `0,437972` contro `1,006123` di zero e
`0,884058` del miglior vettore costante; la correlazione centrata guida--target
è `0,786072`, senza saturazione e con cancellazione residua minima. Il
[`controfattuale active-vs-zero`](rtdetr_fam_box_guided_counterfactual_v1.json)
riproduce la best W&B, ma misura `0,155485332` con guida attiva e `0,155478507`
con guida azzerata: `+0,000006825`, formalmente non degradante ma
prestazionalmente trascurabile. Il campo ha dunque appreso i target train, ma
non fornisce un beneficio diretto misurabile sulla validation.

Il primo tentativo dell'audit non ha pubblicato output ed è fallito per un
controllo runtime impossibile: chiedeva l'hash dei file sorgente a una
rappresentazione composta soltanto da identificatori e target dei batch. La
correzione separa il preflight dei file dal replay dei target senza modificare
protocollo o metriche; è documentata nella
[`nota dettagliata`](../../rtdetr_fam_box_guided_stage_a.md).

SHA-256 degli artefatti:

```text
mechanism JSON:      fb85e3e885836be64c2bc26377bd6aa33172b6c91d365e5df80a9b547e0cec9c
mechanism CSV:       1f9616fdca680c61c69546467e288c35d7aee47100c999a326a2e8271b58f116
counterfactual JSON: f1aecdad12c36160ad9b15c1c0730fff68a9a72d86cb340958e192673b9b76b2
counterfactual CSV:  3089ed1717b75a9075a831dae0c967cb15543619240512f62db8457ac534981e
```

La regressione completa del repository passa 269/269 test; la compilazione
Python dei file coinvolti e `git diff --check` sono puliti. Poiché il
meccanismo geometrico apprende ma non migliora la detection, il fallback
cost-volume non viene attivato; la direzione successiva indicata dal piano è
RT-DETRv2 + FAM.
