# Artefatti sperimentali versionati

Questa directory contiene risultati compatti destinati alla tesi. Gli output
grezzi completi restano in `out/`, che è ignorata da Git, per evitare di
versionare decine di megabyte di predizioni frame-level.

Nei nomi storici degli artefatti e nei campi macchina, `additive` identifica la
configurazione chiamata **Base** nella tesi: entrambe le configurazioni terminano
con un'addizione, mentre soltanto FAM trasforma prima le feature IR.

## Recall a parità di falsi positivi per immagine

Il protocollo `rtdetr_historical_recall_fppi_v1` è completo: 30 valutazioni,
due configurazioni storiche (Base/Additive e FAM), cinque `latest` seed 40--44
e tre popolazioni, senza nuovi training. MtErie contiene 708 frame paired;
Carnation 0025/0026 ne contiene 1.313 e FHL 0407/0408 ne contiene 1.035.
Tutti i confronti usano fusion VIS+IR e le medesime annotazioni VIS.

- [`rtdetr_recall_fppi_summary.csv`](rtdetr_recall_fppi_summary.csv): nove
  riepiloghi ai budget 0.1, 0.5 e 1 FP/immagine, con media, SD e vittorie;
- [`rtdetr_recall_fppi_budgets.csv`](rtdetr_recall_fppi_budgets.csv): 90 punti
  per checkpoint e budget, con soglia descrittiva, FPPI raggiunta, TP e FP;
- [`rtdetr_recall_fppi_paired.csv`](rtdetr_recall_fppi_paired.csv): 45 delta
  appaiati (budget e popolazioni riusano gli stessi checkpoint, non sono
  repliche indipendenti);
- [`rtdetr_recall_fppi_grid.csv`](rtdetr_recall_fppi_grid.csv): 1.806 righe,
  301 budget comuni per ciascuna combinazione configurazione/popolazione;
- [`rtdetr_recall_fppi.json`](rtdetr_recall_fppi.json): protocollo, completezza,
  hash, provenienza e percorsi delle cache;
- [`rtdetr_recall_fppi.pdf`](../images/rtdetr_recall_fppi.pdf): figura a tre
  pannelli inclusa nella sottosezione `sec:recall-fppi` della tesi.

A 0.5 FPPI il delta medio di recall è +0.1102 su MtErie, +0.0981 su Carnation
e +0.1479 su FHL, con 5/5 vittorie per popolazione. L'unica sconfitta ai budget
prefissati è Carnation seed 43 a 0.1 FPPI (-0.0057). L'analisi è post-hoc e
non seleziona soglie operative; il matching in ordine di confidenza differisce
dal precedente error analysis in ordine di IoU. Le bande mostrano SD tra seed,
non intervalli di confidenza sui frame. Metodo completo e risultati sono nel
[report](../../rtdetr_recall_fppi.md).

Rigenerazione offline dalle cache complete, nell'ambiente `sarfusion`:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/run_rtdetr_recall_fppi.py --summarize-only
```

## Confronti Stage A a cinque seed: allineamento e trasferimento

Gli artefatti usati dalle tabelle aggregate della tesi sono
[`stage_a_five_seed_comparison.csv`](stage_a_five_seed_comparison.csv) e
[`stage_a_five_seed_comparison.json`](stage_a_five_seed_comparison.json).
Contengono 35 repliche, cinque per ciascuna configurazione: FAM standard
RT-DETR, mixed consistency, box-guided P3, RT-DETRv2 Additive/FAM e YOLO26
Additive/FAM. Il CSV conserva i singoli seed come provenienza verificabile;
il testo della tesi presenta confronti aggregati, non pilot o run isolate.

Il protocollo comune usa 3.123 coppie train, 896 coppie validation FHL
0401/0402, seed 40--44, nessun early stopping e best validation mAP@50 come
primario (`min_delta=0.001`). Le ultime epoche sono una diagnostica distinta:
10 per RT-DETR/RT-DETRv2, 50 per YOLO26. Il JSON contiene statistiche
campionarie e contrasti appaiati, con IC t al 95% e numero di vittorie.

| Contrasto sui best | Delta medio mAP@50 | Vittorie | Stage B |
|---|---:|---:|---|
| Mixed − FAM standard | -0,055126 | 0/5 | non attivato |
| Box-guided − FAM standard | +0,007717 | 2/5 | non attivato |
| RT-DETRv2 FAM − Additive | +0,022036 | 3/5 | non attivato |
| YOLO26 FAM − Additive | -0,023418 | 1/5 | non attivato |

Sono richiesti sia un delta medio almeno `+0.01` sia almeno quattro vittorie.
RT-DETRv2 supera la soglia media, ma non quella di stabilità: non ha un effetto
medio negativo. Le cinque baseline FAM RT-DETR sono riutilizzate; configurazioni
e inizializzazioni sono state confrontate, senza attestare identità completa
del codice e runtime storici. Ogni detector usa il proprio controllo Additive,
e le mAP non costituiscono una classifica fra recipe e budget diversi.

Verifica del 14 settembre 2026: history complete per le 25 run RT-DETR/RT-DETRv2,
dieci run YOLO26 da 50 epoche e replay dei best YOLO26 coerenti. La run RT-DETRv2
Additive seed 40 interrotta è esclusa e sostituita dalla replica completa dello
stesso seed, non conteggiata due volte. Non sono state eseguite nuove inferenze
sul test né le diagnostiche multiseed mixed sulle modalità isolate o box-guided
sul campo geometrico; queste ultime non sono desumibili dagli esiti dei pilot.

Le sezioni archiviate sui pilot mixed, box-guided e YOLO26 conservano soltanto
la provenienza dei vecchi artefatti: i loro divieti sugli altri seed e i loro
risultati non descrivono le campagne aggregate riportate qui.

## RT-DETR + FAM: inizializzazione dei soli offset a zero, Stage A

Gli artefatti
[`rtdetr_fam_zero_offset_stage_a.csv`](rtdetr_fam_zero_offset_stage_a.csv) e
[`rtdetr_fam_zero_offset_stage_a.json`](rtdetr_fam_zero_offset_stage_a.json)
documentano cinque nuovi training sui seed 40--44 e il confronto appaiato con
i cinque controlli FAM standard Stage A già disponibili. Il CSV contiene
cinque righe con ID delle run, epoche selezionate, metriche `best`/`latest`,
delta e replay del best; il JSON conserva protocollo, statistiche e provenienza.
Queste cinque candidate sono aggiuntive alle 35 repliche del precedente
artefatto Stage A; i controlli sono riutilizzati e non contati come nuovi training.

L'intervento azzera soltanto pesi e bias delle righe `0:18` di `offset_conv`
a P3/P4/P5 dopo l'inizializzazione completa del detector pretrained. Le maschere,
i filtri DCNv2 e gli altri parametri restano quelli del controllo, così come
lo stato RNG; gli offset rimangono apprendibili. Non è un confronto con Base
senza FAM, né una variante `identity_dcnv2` o con offset congelati.

Il protocollo mantiene 3.123 coppie train, 896 coppie validation, dieci epoche,
AdamW `2e-5`, batch 4 e Modal Dropout nativo 20/20/60. Tutti i seed completano
il budget, senza filtro sul seed 40. Il primario è `best` sulla validation
mAP@50 con `min_delta=0.001`; `latest` all'epoca 10 è diagnostico.
Le deviazioni standard riportate sono campionarie.

| Checkpoint | FAM standard, media ± DS | Offset-zero, media ± DS | Delta appaiato, media ± DS | IC t 95% del delta | Vittorie |
|---|---:|---:|---:|---|---:|
| `best` (primario) | 0,164563 ± 0,019554 | 0,152080 ± 0,016686 | −0,012482 ± 0,025718 | [−0,044416; +0,019451] | 1/5 |
| `latest` (diagnostico) | 0,084014 ± 0,026683 | 0,095808 ± 0,024515 | +0,011794 ± 0,018057 | [−0,010628; +0,034215] | 3/5 |

La regola congelata richiede delta medio `best >= +0.01` e almeno quattro
seed positivi: **nessuna promozione allo Stage B e nessuna valutazione test**.
Il vantaggio medio diagnostico di `latest` non sostituisce il criterio primario.
Entrambi gli intervalli comprendono zero; il risultato descrive questa
campagna e non dimostra un peggioramento universale dell'inizializzazione a zero.
Questi valori di validation non vanno confrontati direttamente con la media
test storica `0.3780`.

Verifica del 16 settembre 2026: statistiche ricalcolate dai cinque record;
selezione best e latest ricostruite dalle dieci metriche live per seed;
inizializzazioni ricostruite coerenti con i controlli e con le trace candidate;
29 file del manifest coerenti con gli hash registrati. I cinque replay già
salvati riproducono esattamente il best live (errore assoluto `0.0`, tolleranza
`0.0002`) con caricamento stretto: 1.051 tensori serializzati e 48 alias esatti
ricostruiscono i 1.099 tensori di stato. Questo export non esegue training,
inferenze o nuovi replay; conserva gli hash checkpoint registrati.

La lettura su CPU dei soli parametri offset nei dieci checkpoint verifica
30 predittori: tutte le 540 righe dei pesi sono non nulle e le 18 righe di
ciascun predittore sono distinte; tutti i 540 bias sono non nulli e i valori
controllati sono finiti. Si verifica quindi l'apprendimento dei parametri,
non la qualità dell'allineamento o gli offset in pixel.

Il riuso dei controlli verifica configurazione e inizializzazione, ma non
certifica uno snapshot sorgente, un ambiente o una traiettoria numerica storica
interamente identici. Non sono stati ripetuti i replay dei controlli storici.
Metodo e limiti sono documentati nella
[`nota sperimentale`](../../rtdetr_fam_zero_offset_stage_a.md).

SHA-256 della fonte numerica locale
[`decision.json`](../../../out/rtdetr_fam_zero_offset_stage_a/decision.json):

```text
0fbc3e12bf235c498069b33f6e889590a605acf85fde49e9b00da65ee0ed9650
```

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
Le acquisizioni non erano state usate nelle campagne conservate; ho inoltre
registrato di non averle visionate manualmente prima del protocollo. FAM storico
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
Figura: 55f49b8847175f825671718c6ed1cd129ca978b817e92520674f0a6b4796a13d
```

## Archivio — RT-DETR FAM: screen mixed consistency seed 40

Il file
[`rtdetr_fam_mixed_consistency_probe_evaluation.csv`](rtdetr_fam_mixed_consistency_probe_evaluation.csv)
contiene le otto valutazioni del gate Stage A: FAM baseline e candidato seed
40 nelle condizioni fusion, VIS-only, paired masked-IR con ground truth VIS e
IR nativa con ground truth IR, sempre sugli stessi 896 frame FHL.

Il candidato fallisce tutti i gate congelati: delta mAP@50 fusion `-0,026884`,
paired masked-IR `+0,001829` e IR nativa `-0,050581`. È chiuso dopo un seed;
nel pilot v1 non ho eseguito seed 41--44, Stage B o valutazioni MtErie. Tempi di
training, checkpoint e interpretazione sono in
[`../../rtdetr_fam_mixed_consistency_stage_a.md`](../../rtdetr_fam_mixed_consistency_stage_a.md).

Il CSV versionato ha SHA-256:

```text
5cb36e63d7a5758215e9fa4cf431e3ad4e56d7b285259dd00466d7627b608d65
```

## Archivio — RT-DETR FAM Box-Guided P3: inventario e probe tecnico

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
al gate preregistrato `+0,01`: ho quindi chiuso il pilot prima dei seed
41--44 e dello Stage B.

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

## Archivio — YOLO26 dual-backbone: pilot e repair Additive seed 40

YOLO26s ufficiale è stato integrato con due backbone RGB/IR e fusione
P3/P4/P5, congelando Additive e FAM prima dello Stage A. Il pilot Additive ha
completato 50 epoche ma ha raggiunto soltanto `0,06472` mAP@50 all'epoca 2,
collassando a `0,00001` all'epoca 50. L'audit ha individuato una recipe
correggibile: con AdamW esplicito il learning rate di warmup dei bias era
rimasto a `0,1`.

Dopo il pilot ho provato un solo repair, cambiando esclusivamente
`warmup_bias_lr` a zero. Il repair completa 50/50 epoche e passa integrità,
optimizer e replay checkpoint, ma ottiene un best complessivo di `0,04353`
all'epoca 1. Nel tratto preregistrato 4--50 il massimo è `0,01535`, contro la
soglia di vitalità `0,10`; lo stato finale è
`control_integrity_passed_vitality_failed`.

La linea YOLO26 è quindi chiusa senza eseguire FAM, altri seed o Stage B. Il
risultato non misura il delta FAM--Additive e non dimostra un limite assoluto
del task: mostra che questo controllo dual-backbone e questa recipe non hanno
prodotto una baseline vitale. Dettagli e interpretazione sono nella
[`nota post-run`](../../yolo26_stage_a_outcome.md); il riepilogo versionato è
[`yolo26_additive_seed40_stage_a_repair_v1.json`](yolo26_additive_seed40_stage_a_repair_v1.json).

SHA-256 principali:

```text
source manifest: f6ef8616ce45b834cdb8bb7a777ecb6c89c545f7fa028580366d053fc981e48f
audit repair:    5612f1378948f2be9f3bf3071063f38f8330f15dd105443f5bdac3b343f8ee5e
results CSV:     b6a31271f82c0da7accefb67e78b77bf108f4d7d71616f71028c9c2295a81818
best.pt:         4592fa6c053dd648230f6e3731cbc1fbe59980c06a54592ae15803bc8d5ad2ff
last.pt:         de9e5158de94592dc6ee355accaac7ab9e03d7eada550be028b9cc6724367f5b
```
