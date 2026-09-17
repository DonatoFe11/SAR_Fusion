# RT-DETR storico: recall rispetto ai falsi positivi per immagine

## Stato e obiettivo

Analisi completata: 30/30 unità. Protocollo `rtdetr_historical_recall_fppi_v1`.
Si confrontano Additive e FAM storici in fusion VIS+IR sui checkpoint
`latest` dei cinque seed 40–44, senza nuovo training. Non sono inclusi
i checkpoint FAM current-code Stage B, RCRA o altre varianti.

Questa è una caratterizzazione post-hoc dei checkpoint esistenti:
non seleziona un nuovo modello, un checkpoint o una soglia operativa
da applicare in deployment. Le acquisizioni sono già state valutate
nelle campagne precedenti e non vengono presentate come nuovi holdout ciechi.

## Protocollo

| Acquisizione | Frame paired | Box VIS | Frame vuoti |
|---|---:|---:|---:|
| MtErie | 708 | 1770 | 19 |
| Carnation 0025/0026 | 1313 | 5238 | 100 |
| FHL 0407/0408 | 1035 | 2022 | 239 |

Si riutilizzano gli inventari e la ground truth VIS congelati:
per MtErie l'associazione paired storica (sorted zip); per Carnation
0025/0026 e FHL 0407/0408 l'intersezione degli ID numerici comuni.
Anche i frame senza persone rientrano nel denominatore FPPI.

Un solo forward per immagine e checkpoint raccoglie tutte le predizioni
native del modello (soglia di raccolta 0, nessuna nuova NMS). Le curve
si ricavano offline variando la confidenza: una predizione è conservata
se `score >= soglia`; score uguali entrano insieme. È incluso il punto
che rifiuta tutte le predizioni. Il JSON conserva gli hash dei checkpoint
e i percorsi delle cache delle predizioni.

Il matching è uno-a-uno a IoU ≥ 0.50, in ordine decrescente di confidenza,
con scelta della GT non ancora assegnata a IoU maggiore. Le predizioni
duplicate sono falsi positivi. Questo criterio mantiene coerenti i
prefissi della curva; differisce dal precedente error analysis che
ricalcolava un matching greedy globale per IoU a ogni soglia. Piccole
differenze rispetto ai vecchi punti non indicano un cambiamento del modello.

`Recall = TP / box GT`; `FPPI = FP / numero totale di frame`.
A ogni budget FPPI si prende la massima recall empiricamente raggiungibile
senza superarlo, senza interpolazione o frazionamento di score uguali.
Ogni seed viene campionato sulla stessa griglia logaritmica 0.01–10
FPPI; solo dopo si calcolano media e deviazione standard campionaria.
Non si media a una medesima confidenza, che può corrispondere a budget
di falsi positivi differenti nei diversi checkpoint. Le curve si fermano
all'endpoint minimo raccolto tra i seed e le due configurazioni del pannello:
non si estrapolano punti oltre il supporto osservato.

Le bande descrivono la variabilità tra cinque training seed, non sono
intervalli di confidenza sui frame né prove di generalizzazione SAR.
Le tre acquisizioni sono riportate separatamente; i frame non vengono
trattati come repliche statistiche indipendenti.
Gli eventuali budget oltre l'endpoint di un checkpoint sono marcati
non disponibili: il CSV conserva esplicitamente il numero di seed
disponibili per ogni statistica, senza imputazione.

## Risultati ai budget prefissati

I budget 0.1, 0.5 e 1 FP/immagine sono punti descrittivi prefissati.
Ogni delta è appaiato sullo stesso seed; `vittorie` conta solo delta
strettamente positivi. Un vantaggio a un budget non implica una dominanza
su tutta la curva o su tutte le acquisizioni.

| Acquisizione | FPPI ≤ | Recall Additive (media ± SD) | Recall FAM (media ± SD) | Δ FAM−Additive (media ± SD) | Vittorie |
|---|---:|---:|---:|---:|---:|
| MtErie | 0.1 | 0.1610 ± 0.0493 | 0.2217 ± 0.0338 | 0.0607 ± 0.0393 | 5/5 |
| MtErie | 0.5 | 0.3266 ± 0.0739 | 0.4367 ± 0.0885 | 0.1102 ± 0.0615 | 5/5 |
| MtErie | 1 | 0.4296 ± 0.0662 | 0.5368 ± 0.0778 | 0.1072 ± 0.0600 | 5/5 |
| Carnation 0025/0026 | 0.1 | 0.1797 ± 0.0496 | 0.2670 ± 0.0579 | 0.0873 ± 0.0550 | 4/5 |
| Carnation 0025/0026 | 0.5 | 0.3688 ± 0.0745 | 0.4669 ± 0.0531 | 0.0981 ± 0.0318 | 5/5 |
| Carnation 0025/0026 | 1 | 0.4507 ± 0.0757 | 0.5467 ± 0.0542 | 0.0960 ± 0.0388 | 5/5 |
| FHL 0407/0408 | 0.1 | 0.1632 ± 0.0604 | 0.2777 ± 0.0666 | 0.1145 ± 0.0946 | 5/5 |
| FHL 0407/0408 | 0.5 | 0.3804 ± 0.0884 | 0.5283 ± 0.0664 | 0.1479 ± 0.1255 | 5/5 |
| FHL 0407/0408 | 1 | 0.4903 ± 0.0895 | 0.6310 ± 0.0567 | 0.1407 ± 0.1112 | 5/5 |

## Endpoint e tracciabilità

| Acquisizione | Endpoint minimo FPPI (10 checkpoint) |
|---|---:|
| Carnation 0025/0026 | 296.3085 |
| FHL 0407/0408 | 298.1865 |
| MtErie | 297.8164 |

I CSV conservano i risultati per seed ai budget, i delta appaiati,
le statistiche riassuntive e la griglia comune aggregata. Il JSON
conserva protocollo, checkpoint, percorsi delle cache ed endpoint,
hash degli inventari e dell'implementazione e versioni software.
Il CSV per seed riporta anche FPPI effettivamente raggiunta, soglia,
TP e FP del punto empirico: la soglia descrive quel checkpoint su
quell'acquisizione e non è una soglia ottimizzata per il deployment.

- [Curve PDF](Search_and_Rescue/images/rtdetr_recall_fppi.pdf)
- [Curve PNG](Search_and_Rescue/images/rtdetr_recall_fppi.png)
- [Riepilogo CSV](Search_and_Rescue/results/rtdetr_recall_fppi_summary.csv)
- [Budget per seed](Search_and_Rescue/results/rtdetr_recall_fppi_budgets.csv)
- [Delta appaiati](Search_and_Rescue/results/rtdetr_recall_fppi_paired.csv)
- [Griglia aggregata](Search_and_Rescue/results/rtdetr_recall_fppi_grid.csv)
- [Metadati JSON](Search_and_Rescue/results/rtdetr_recall_fppi.json)

## Riproduzione

Dalla root del repository, nell'ambiente `sarfusion`:

```bash
python scripts/run_rtdetr_recall_fppi.py --device cuda
```

Il comando riutilizza le cache compatibili già completate e riparte
in sicurezza dalle unità mancanti; cache con provenienza incompatibile
vengono rifiutate. Non è necessario un nuovo training.
Per rigenerare soltanto i riepiloghi da tutte le cache già presenti:

```bash
python scripts/run_rtdetr_recall_fppi.py --summarize-only
```

Configurazione: [`rtdetr_recall_fppi.yaml`](../parameters/RTDETR/rtdetr_recall_fppi.yaml).
Le predizioni dense sono conservate in `out/rtdetr_recall_fppi/predictions/`;
le curve esatte e gli altri risultati sono in `out/rtdetr_recall_fppi/`.

## Inserimento successivo nella tesi

La figura e la tabella possono integrare l'error analysis spiegando se
il guadagno di recall persiste a parità di budget di falsi positivi.
Occorre esplicitare matching, media sui seed, acquisizioni separate e
carattere post-hoc. Non sostituiscono mAP, non costituiscono una nuova
selezione architetturale e non autorizzano una soglia di deployment
ottimizzata sui set di test. I file `.tex` non sono stati modificati.
