# Error analysis finale RT-DETR Additive--FAM

## Stato

Il protocollo è stato bloccato il 12 agosto 2026 prima dell'inferenza completa.
La campagna è terminata regolarmente: dieci checkpoint, 35.400 righe
frame/soglia, 50 riepiloghi checkpoint e dodici figure qualitative. L'output
aggregato è marcato `protocol_complete: true`. Il manifest qualitativo è stato
generato usando soltanto le annotazioni.

Durante il primo smoke tecnico, la soglia preliminare `0.50` non produceva
alcuna detection Additive sul primo frame; `0.25` produceva predizioni, ma un
solo batch non permette di dimostrare che non sia troppo selettiva sul resto
del test. Per evitare di tarare una nuova soglia su MtErie, prima di calcolare
risultati aggregati la soglia primaria è stata quindi ricondotta a `0.01`, cioè
il valore già fissato e usato nelle valutazioni finali VIS, IR e VIS+IR.
`0.05`, `0.10`, `0.25` e `0.50` sono riportate tutte come sensitivity analysis,
senza sceglierne retroattivamente una come migliore.

## Domanda e protocollo congelato

L'analisi non serve a ricalcolare la mAP, già riportata nel protocollo finale,
ma a caratterizzare come cambia il comportamento operativo del detector:

- checkpoint `latest` Additive e FAM `current_dcnv2` per i seed `40--44`;
- input VIS+IR completo sui 708 frame MtErie;
- soglia di confidenza primaria `0.01`, identica alla valutazione finale;
- sensitivity analysis completa a `0.05`, `0.10`, `0.25` e `0.50`;
- matching uno-a-uno per IoU decrescente, con match valido a `IoU >= 0.50`;
- aggregazione prima sull'intero checkpoint e poi fra i cinque seed;
- differenze appaiate FAM meno Additive per seed;
- nessuna selezione di checkpoint, soglia o frame dopo aver visto i risultati.

Per ogni checkpoint vengono riportati precision, recall, falsi positivi per
immagine, frazione di frame senza predizioni, frazione di frame vuoti con
almeno un falso positivo, frazione di frame non vuoti con almeno un falso
negativo e IoU media dei match. La recall è inoltre stratificata usando le
categorie COCO alla risoluzione di evaluation `640 x 640`:

- small: area minore di `32^2` pixel;
- medium: area fra `32^2` e `96^2` pixel;
- large: area almeno `96^2` pixel.

Il benchmark contiene 708 frame, 1.770 oggetti e 19 frame vuoti. Gli oggetti
sono 1.451 small, 301 medium e 18 large; la classe large è quindi riportata
con particolare cautela.

## Risultati multi-soglia

Le medie e deviazioni standard campionarie fra cinque checkpoint sono:

| Soglia | Modello | Precision | Recall | FP/immagine | Frame non vuoti con almeno un FN |
|---:|---|---:|---:|---:|---:|
| 0.01 | Additive | 0.0418 ± 0.0175 | 0.7677 ± 0.0490 | 51.006 ± 21.591 | 0.4508 ± 0.0777 |
| 0.01 | FAM | 0.0443 ± 0.0132 | **0.8044 ± 0.0328** | 47.623 ± 17.873 | **0.4009 ± 0.0524** |
| 0.05 | Additive | 0.2713 ± 0.0909 | 0.5932 ± 0.1134 | 4.844 ± 3.199 | 0.6476 ± 0.1155 |
| 0.05 | FAM | 0.3010 ± 0.0940 | **0.6539 ± 0.0972** | 4.336 ± 2.191 | **0.5835 ± 0.0775** |
| 0.10 | Additive | 0.4721 ± 0.0923 | 0.4582 ± 0.1535 | 1.513 ± 1.115 | 0.7466 ± 0.1223 |
| 0.10 | FAM | **0.5194 ± 0.1148** | **0.5507 ± 0.1333** | 1.447 ± 0.792 | **0.6554 ± 0.0725** |
| 0.25 | Additive | 0.7290 ± 0.0866 | 0.2340 ± 0.1468 | **0.249 ± 0.218** | 0.8682 ± 0.0879 |
| 0.25 | FAM | **0.7715 ± 0.0742** | **0.3330 ± 0.1506** | 0.276 ± 0.168 | **0.7858 ± 0.0525** |
| 0.50 | Additive | 0.9140 ± 0.0536 | 0.1088 ± 0.0705 | **0.029 ± 0.033** | 0.9257 ± 0.0588 |
| 0.50 | FAM | **0.9350 ± 0.0504** | **0.1627 ± 0.0663** | 0.036 ± 0.037 | **0.8790 ± 0.0359** |

Il risultato più consistente è la recall. FAM supera Additive in 4/5 seed a
`0.01`, 3/5 a `0.05` e **5/5** sia a `0.10`, sia a `0.25`, sia a `0.50`. I
delta medi FAM meno Additive sono rispettivamente `+0.0367`, `+0.0607`,
`+0.0925`, `+0.0990` e `+0.0539`. La quota di frame non vuoti con almeno un
target mancato diminuisce in 4/5 seed a `0.01` e `0.05` e in **5/5** da `0.10`
a `0.50`.

La maggiore recall non deriva soltanto dagli oggetti grandi. A `0.10` FAM
migliora in media la recall small di `+0.0866`, medium di `+0.1183` e large di
`+0.1444`; a `0.25` i delta sono `+0.0815`, `+0.1781` e `+0.1889`. Gli oggetti
large sono però soltanto 18 e non supportano una conclusione autonoma robusta.

Gli FP non mostrano invece una riduzione appaiata universale. In media FAM ne
produce leggermente meno a `0.01--0.10`, ma leggermente più a `0.25--0.50`; i
segni variano fra seed. Sui soli 19 frame vuoti, a `0.01` almeno un FP compare
quasi sempre per entrambi i modelli (`0.968` Additive, `1.000` FAM). A `0.10`
la frazione scende a `0.221` e `0.137`, mentre a `0.25` vale `0.042` e `0.021`.
Il campione di frame vuoti è piccolo: queste percentuali sono diagnostiche,
non una stima precisa del comportamento in scenari senza persone.

La soglia `0.01` è quindi coerente con la raccolta di predizioni usata dalla
valutazione mAP, ma non è operativamente utilizzabile senza un ulteriore
criterio di filtraggio: genera circa 48--51 FP per immagine. La sensitivity
analysis non identifica una soglia deployment ottimale, perché MtErie è il
benchmark interno già consultato; mostra però che il vantaggio di recall del
FAM persiste lungo tutto l'intervallo predefinito.

## Figure qualitative predefinite

Le figure usano esclusivamente il seed 43, perché il delta FAM--Additive di
quel seed coincide con la mediana dei cinque delta mAP@50. Non rappresenta né
la migliore né la peggiore run.

Il manifest versionato è
[`rtdetr_error_analysis_manifest.json`](../parameters/RTDETR/rtdetr_error_analysis_manifest.json).
Per ciascuna delle tre sequenze sceglie, dalle sole annotazioni, il frame con
target small più vicino a un terzo della sequenza e il frame vuoto più vicino
a due terzi:

| Sequenza | Small target | Frame vuoto |
|---|---:|---:|
| MtErie VIS 0003 | 88 | 202 |
| MtErie VIS 0005 | 354 | 352 |
| MtErie VIS 0007 | 593 | 641 |

Gli stessi sei frame, lo stesso IoU e lo stesso limite di venti predizioni
visibili sono usati per entrambi i modelli. Per ciascun frame vengono generate
due versioni predefinite: `0.01`, coerente con la valutazione finale, e `0.25`,
utile a mostrare separatamente le predizioni con confidenza maggiore. Non si
sceglie la soglia dopo avere visto quale modello appare migliore.

Ogni figura mostra RGB con ground truth, IR, predizioni Additive e predizioni
FAM. I colori sono: ground truth matched verde, ground truth missed arancione,
predizione TP blu e falso positivo rosso. Se un modello produce più di venti
box, i conteggi nel titolo includono tutte le predizioni alla soglia indicata,
mentre il pannello mostra le venti con score maggiore per restare leggibile.
Le figure restano esempi illustrativi; le conclusioni devono derivare
dall'aggregazione sui cinque seed.

L'ispezione di tutti i dodici pannelli conferma il ruolo illustrativo della
soglia. A `0.01` le scene sono sovraccariche di box a bassa confidenza: nei sei
frame Additive produce da 42 a 141 FP e FAM da 5 a 68. A `0.25` i tre frame
vuoti non producono box con entrambi i modelli; nei frame small predefiniti FAM
recupera uno dei quattro target del campione 88, ma entrambi i modelli mancano
i target dei campioni 354 e 593. Questi tre esempi non vengono generalizzati
oltre quanto supportato dalle statistiche multi-seed.

## Implementazione e controlli

Il runner è
[`run_rtdetr_error_analysis.py`](../scripts/run_rtdetr_error_analysis.py). È
resumable a livello di checkpoint e rifiuta risultati incompatibili per
checkpoint, manifest, protocollo o modalità smoke. Produce:

- `out/rtdetr_error_analysis/raw/{additive,fam}_seed_*.json`;
- `out/rtdetr_error_analysis/rtdetr_error_analysis.json`;
- `out/rtdetr_error_analysis/rtdetr_error_analysis_checkpoints.csv`;
- dodici figure qualitative, sei a `0.01` e sei a `0.25`, in
  `out/rtdetr_error_analysis/figures/qualitative/`;
- il grafico appaiato
  `out/rtdetr_error_analysis/figures/rtdetr_additive_fam_error_summary.png`;
- il grafico multi-soglia
  `out/rtdetr_error_analysis/figures/rtdetr_additive_fam_threshold_sensitivity.png`.

Lo smoke test sui due checkpoint seed 43 aveva verificato l'intero percorso su
un batch e marcato correttamente l'output come `protocol_complete: false`. La
campagna finale ha risolto un solo checkpoint `latest` per configurazione/seed,
caricato ogni modello senza chiavi mancanti o inattese e prodotto dieci JSON
completi.

## Comando usato

La campagna completa è stata eseguita con:

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
MPLCONFIGDIR=/tmp/matplotlib-rtdetr-error-analysis \
PYTHONUNBUFFERED=1 \
/home/donato/miniconda3/envs/sarfusion/bin/python \
  scripts/run_rtdetr_error_analysis.py \
  --device cuda \
  2>&1 | tee rtdetr_error_analysis.log
```

Il runner resta resumable: i JSON compatibili già completi vengono saltati. Non
usare `--force` salvo dopo avere ispezionato manualmente gli output esistenti.

## Conclusione

L'error analysis rafforza, senza sostituire, il risultato mAP@50 multi-seed.
FAM non garantisce una riduzione dei falsi positivi a ogni soglia e seed, ma
sposta consistentemente il compromesso verso una recall maggiore, riducendo
la frequenza di frame non vuoti con almeno una persona mancata. In Search and
Rescue questo è un vantaggio pertinente, pur richiedendo calibrazione della
soglia e controllo degli FP prima di un impiego operativo. Il confronto resta
limitato al benchmark interno MtErie e non dimostra generalizzazione a nuove
acquisizioni.
