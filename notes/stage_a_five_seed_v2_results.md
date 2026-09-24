# Stage A a cinque seed v2: risultati finali

## Stato al 13 settembre 2026

Ho completato **tutti i 20 nuovi training** previsti, senza filtro
prestazionale sul seed 40: cinque box-guided, cinque mixed consistency,
cinque YOLO26 Additive e cinque YOLO26 FAM. La campagna avviata il 10 settembre
è terminata il **13 settembre 2026 alle 03:55 CEST**. Il log principale è
[`stage_a_20_training_20260910_105454.log`](../stage_a_20_training_20260910_105454.log).
Non risultano training ancora in corso alla verifica finale.

Il controllo RT-DETR v1 + FAM standard è costituito dalle **cinque run Stage A
già complete**, che ho scelto di riutilizzare: 25 run analizzate,
ma solo 20 nuovi training. Non è RT-DETRv2 e non è una baseline RGB-only.

**Nessuna candidata supera il criterio prestazionale aggregato Stage A;
nessuno Stage B viene attivato.** Non sono stati avviati nuovi training o
nuove valutazioni sul test durante questa verifica. Protocollo, configurazioni
e comando eseguito sono nella [nota Stage A/B v2](stage_a_b_five_seed_v2.md).
I vecchi pilot seed 40 sono archiviati separatamente e non entrano nelle
statistiche di questa campagna.

## Dati, checkpoint e provenienza

- Seed appaiati: **40, 41, 42, 43, 44**.
- Training: 3.123 coppie FHL 0405/0406 + Baker 1; validation: 896 coppie FHL
  0401/0402. Nessuna nuova valutazione sulle 708 coppie MtErie.
- Budget completo, senza early stopping: 10 epoche RT-DETR, 50 epoche YOLO26.
- Primario: checkpoint **best**, selezionato sulla validation mAP@50 con
  miglioramento minimo `0,001`. Il valore riportato è quello del checkpoint
  selezionato, non necessariamente il massimo grezzo in presenza di questa
  tolleranza. Le epoche nelle tabelle sono numerate da 1.
- Secondario: **latest** RT-DETR all'epoca 10 e **last.pt** YOLO26 all'epoca 50.
  Non sono confusi con il best o usati per modificare a posteriori la decisione.

### Run RT-DETR

| Seed | FAM standard riutilizzato | Box-guided v2 | Mixed consistency v2 |
|---|---|---|---|
| 40 | [m7xjslb6](../wandb/run-20260814_203405-m7xjslb6/files/) | [ian5nn87](../wandb/run-20260910_105509-ian5nn87/files/) | [yki5nvtj](../wandb/run-20260910_230237-yki5nvtj/files/) |
| 41 | [2kil4xq9](../wandb/run-20260814_232554-2kil4xq9/files/) | [navxhwvc](../wandb/run-20260910_131826-navxhwvc/files/) | [gyrqzo9j](../wandb/run-20260911_053447-gyrqzo9j/files/) |
| 42 | [fu87g1i2](../wandb/run-20260815_020741-fu87g1i2/files/) | [qhrxfz1f](../wandb/run-20260910_153459-qhrxfz1f/files/) | [kuoev8xa](../wandb/run-20260911_125947-kuoev8xa/files/) |
| 43 | [pz5tzni4](../wandb/run-20260815_051325-pz5tzni4/files/) | [7r65y3xw](../wandb/run-20260910_174502-7r65y3xw/files/) | [sxund2gf](../wandb/run-20260911_222244-sxund2gf/files/) |
| 44 | [398272cv](../wandb/run-20260815_081411-398272cv/files/) | [ezqfhmhb](../wandb/run-20260910_202519-ezqfhmhb/files/) | [syu6yilp](../wandb/run-20260912_043431-syu6yilp/files/) |

Progetti W&B, rispettivamente:

- `RTDETR_FAM_SequenceVal_Fixed10_Protocol`;
- `RTDETR_FAM_BoxGuided_StageA_FiveSeed_V2`;
- `RTDETR_FAM_MixedConsistency_StageA_FiveSeed_V2`.

Le fonti sono `config.yaml`, `wandb-summary.json`, la history locale W&B,
`output.log` e le directory `best/` e `latest/` di ciascuna run.

Il riuso della baseline del 14--15 agosto è sostenuto dal controllo della
configurazione comune di modello, training, dati e split, e dalla corrispondenza
degli hash di inizializzazione ricostruiti per tutti i cinque seed. **Non
attesta uno snapshot sorgente o un ambiente storico integralmente identico**:
le vecchie trace non registrano il manifest sorgente corrente. Le due nuove
campagne RT-DETR attestano invece il manifest
`rtdetr_box_guided_training_source_v1`, SHA-256
`3320b24d060aaacec316290c114933db1dce8d71eaaf455f1ccb6448183f76e5`.

Per mixed, i sorteggi aggiuntivi del percorso student possono inoltre cambiare
le successive realizzazioni del Modal Dropout. L'appaiamento per seed non è
una garanzia di identità di tutti gli ingressi intermedi: si valuta l'intero
intervento di training, con questo limite di attribuzione.

### Run YOLO26

Artefatti in [`runs/yolo26_stage_a_five_seed_v2/`](../runs/yolo26_stage_a_five_seed_v2/),
directory `additive_seed40`--`additive_seed44` e `fam_seed40`--`fam_seed44`.
Ogni directory contiene `completion.json`, `run_manifest.json`, `results.csv`,
`checkpoint_selection.jsonl`, `data_trace.jsonl` e `weights/{best,last}.pt`.
I best sotto riportati sono le metriche live di selezione registrate in
`completion.json`; il replay serializzato è un controllo separato, non una
sostituzione selettiva della metrica primaria.

## Risultati primari: best sulla validation

Tutte le mAP sono sulla scala **0--1**, non percentuali. I delta sono candidato
meno controllo con lo stesso seed; `+0,01` equivale a un punto percentuale.
Le cifre sono arrotondate per la presentazione, dopo il calcolo delle statistiche.

### RT-DETR v1: box-guided e mixed contro FAM standard

| Seed | FAM best (epoca) | Box-guided best (epoca) | Delta box | Mixed best (epoca) | Delta mixed |
|---|---:|---:|---:|---:|---:|
| 40 | 0,152148 (1) | 0,202981 (1) | +0,050834 | 0,144840 (1) | -0,007307 |
| 41 | 0,142360 (4) | 0,174408 (2) | +0,032049 | 0,057599 (5) | -0,084761 |
| 42 | 0,165466 (6) | 0,138806 (5) | -0,026661 | 0,122612 (1) | -0,042854 |
| 43 | 0,193932 (1) | 0,179987 (2) | -0,013944 | 0,101216 (1) | -0,092715 |
| 44 | 0,168908 (1) | 0,165216 (3) | -0,003692 | 0,120917 (1) | -0,047991 |

### YOLO26: FAM contro Additive

| Seed | Additive best (epoca) | FAM best (epoca) | Delta FAM meno Additive |
|---|---:|---:|---:|
| 40 | 0,043530 (1) | 0,018900 (2) | -0,024630 |
| 41 | 0,048190 (1) | 0,009880 (1) | -0,038310 |
| 42 | 0,078390 (1) | 0,047560 (1) | -0,030830 |
| 43 | 0,035570 (1) | 0,036640 (1) | +0,001070 |
| 44 | 0,047750 (1) | 0,023360 (1) | -0,024390 |

### Aggregazione a cinque seed

DS è la deviazione standard **campionaria** (`ddof=1`). Gli IC sui delta
usano `media ± 2,776445 × DS / sqrt(5)` (t di Student, 4 gradi di libertà).
Sono descrittivi della variabilità fra seed sullo split fissato, non della
generalizzazione a nuove acquisizioni; con cinque osservazioni dipendono dalle
assunzioni del modello t. Non sono la regola di promozione.

| Modello | Best mAP@50, media ± DS | Ultima epoca mAP@50, media ± DS |
|---|---:|---:|
| RT-DETR FAM standard, riutilizzato | 0,164563 ± 0,019554 | 0,084014 ± 0,026683 |
| RT-DETR FAM box-guided | 0,172280 ± 0,023329 | 0,084549 ± 0,026338 |
| RT-DETR FAM mixed consistency | 0,109437 ± 0,032839 | 0,029120 ± 0,015367 |
| YOLO26 Additive | 0,050686 ± 0,016295 | 0,001262 ± 0,002154 |
| YOLO26 FAM | 0,027268 ± 0,014893 | 0,000092 ± 0,000112 |

| Confronto sui best | Delta medio ± DS | Mediana delta | IC t 95% del delta medio | Vittorie | Gate Stage A |
|---|---:|---:|---|---:|---|
| Box-guided − FAM standard | +0,007717 ± 0,032528 | -0,003692 | [-0,032672; +0,048106] | 2/5 | fallito |
| Mixed − FAM standard | -0,055126 ± 0,034566 | -0,047991 | [-0,098046; -0,012206] | 0/5 | fallito |
| YOLO26 FAM − Additive | -0,023418 ± 0,014822 | -0,024630 | [-0,041822; -0,005014] | 1/5 | fallito |

La regola fissata richiede **entrambi**: delta medio `>= +0,01` e almeno
**4/5 delta strettamente positivi**. Nessun risultato del solo seed 40 ha
interrotto la campagna o determinato la decisione.

## Diagnostica obbligatoria: ultima epoca

| Seed | RT-DETR FAM | Box-guided | Mixed | YOLO26 Additive | YOLO26 FAM |
|---|---:|---:|---:|---:|---:|
| 40 | 0,039738 | 0,052475 | 0,017873 | 0,005110 | 0,000010 |
| 41 | 0,103029 | 0,125297 | 0,011573 | 0,000290 | 0,000130 |
| 42 | 0,094595 | 0,086820 | 0,027050 | 0,000140 | 0,000050 |
| 43 | 0,078981 | 0,082274 | 0,040703 | 0,000420 | 0,000270 |
| 44 | 0,103729 | 0,075877 | 0,048400 | 0,000350 | 0,000000 |

Il valore YOLO26 FAM seed 44 è zero **alla precisione del CSV**, non prova
che la metrica non arrotondata sia esattamente nulla.

| Confronto sull'ultima epoca | Delta medio ± DS | Mediana delta | IC t 95% del delta medio | Vittorie |
|---|---:|---:|---|---:|
| Box-guided − FAM standard | +0,000534 ± 0,019389 | +0,003294 | [-0,023540; +0,024608] | 3/5 |
| Mixed − FAM standard | -0,054894 ± 0,026758 | -0,055329 | [-0,088119; -0,021670] | 0/5 |
| YOLO26 FAM − Additive | -0,001170 ± 0,002199 | -0,000160 | [-0,003901; +0,001561] | 0/5 |

Box-guided non conserva un vantaggio medio rilevante a fine training; mixed
resta inferiore al controllo in tutti i seed. Entrambi i bracci YOLO26
mostrano prestazioni finali molto basse. La scelta frequente di best precoci
non deriva da un confronto imposto all'epoca 1: **tutte le epoche previste sono
state eseguite**, e il checkpoint è stato selezionato lungo l'intero budget.
Per mixed, quattro best sono all'epoca 1, prima dell'attivazione della
consistency, mentre il seed 41 seleziona l'epoca 5.

## Controlli effettuati e audit ancora non eseguiti

Controlli conclusi sugli artefatti locali:

- RT-DETR: tutte le dieci nuove run e i cinque controlli storici risultano
  complete con 10 epoche e 7.810 optimizer step. Per le nuove run, history,
  summary e selezione best sono coerenti; best/latest e gli stati di training
  attesi sono presenti.
- RT-DETR: sui 20 nuovi checkpoint best/latest sono stati verificati tensori
  finiti e caricamento stretto, ricostruendo gli alias dei pesi condivisi.
  Questo controllo non è un nuovo replay delle metriche sul dataset.
- YOLO26: tutte le dieci run hanno stato `completed`, 50 righe di risultati,
  50 record di selezione e `test_evaluated=false`; verificati manifest,
  ambiente, sorgenti, pesi pretrained e preflight.
- YOLO26: inventario e hash del contenuto dei dati, inizializzazione e prime
  20 trace di batch corrispondono tra Additive e FAM per ciascun seed.
  I percorsi dei rispettivi YAML sono specifici della run e non costituiscono
  una differenza del dataset.
- YOLO26: tutti i 20 checkpoint best/last sono caricabili strettamente e
  contengono tensori finiti. I tensori FAM del ramo Additive restano compatibili
  con l'inizializzazione entro la tolleranza di serializzazione FP16; nel ramo
  FAM risultano aggiornati tutti i 12 tensori, per best e last su tutti i seed.
  Questo verifica aggiornamenti effettivi, non un allineamento utile.
- YOLO26: tutti i dieci replay del best già eseguiti dal runner passano;
  massimo errore assoluto live/serializzato `0,0002525`, contro tolleranza
  `0,003`. Non è stato lanciato un nuovo replay durante la verifica finale.

**Non sono state eseguite sui nuovi cinque seed** le valutazioni mixed
masked-IR/VIS-GT e IR nativa/IR-GT, né gli audit box-guided geometrici e
controfattuali guida attiva/azzerata. Non se ne dichiara un esito aggregato:
gli audit del vecchio pilot non si trasferiscono automaticamente alle nuove
run. Questi controlli sarebbero necessari per la promozione e per conclusioni
sul meccanismo; il criterio fusion primario, necessario, è però già fallito.

## Decisione e interpretazione

1. **Box-guided:** piccolo incremento medio dei best (+0,77 punti percentuali),
   ma solo 2/5 vittorie, mediana negativa e IC che include zero. Non è un
   miglioramento abbastanza consistente secondo il criterio fissato.
2. **Mixed consistency:** regressione fusion in tutti i seed, sia best sia
   latest. La recipe valutata non migliora il riferimento; il risultato non
   identifica da solo una causa geometrica o un effetto sulle modalità isolate.
3. **YOLO26 FAM:** inferiore ad Additive in 4/5 seed sui best; il collasso finale
   riguarda entrambi i bracci. È un risultato negativo per questo port
   dual-backbone e questa recipe, non una dimostrazione generale sull'architettura
   YOLO26 o sull'impossibilità di usarvi FAM.

Lo **Stage B non viene attivato** per nessuna delle tre candidate.
RT-DETR v1 + FAM standard resta il riferimento della tesi, senza sostituirlo con il
singolo seed migliore di un'ablation. Un eventuale nuovo esperimento richiede
una modifica motivata e un protocollo distinto, non altri seed scelti per
rivalutare soltanto gli esiti favorevoli.

I valori YOLO26 e RT-DETR qui affiancati sono descrittivi: recipe, budget
(50 contro 10 epoche) e pipeline di valutazione sono diversi. Non costituiscono
una classifica controllata tra famiglie, né vanno confrontati direttamente
con le mAP Stage B su MtErie. La conclusione delle nuove ablation RT-DETR
mantiene inoltre il limite del controllo storico riutilizzato sopra dichiarato.

Questo aggiornamento riguarda i Markdown; non aggiorna la tesi `.tex` né
riscrive i risultati e le decisioni dei vecchi pilot.
