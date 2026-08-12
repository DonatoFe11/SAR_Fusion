# Stress test Carnation RT-DETR Additive--FAM

## Stato

Il protocollo `rtdetr_carnation_stress_test_v1` è stato congelato e versionato
il 12 agosto 2026 **prima di eseguire inferenza o osservare metriche su
Carnation**. La campagna è terminata regolarmente: 30/30 valutazioni, 739
campioni ciascuna, dieci checkpoint caricati senza chiavi mancanti o inattese e
aggregato marcato `protocol_complete: true`. Lo stress test non modifica il
modello principale, la soglia, i checkpoint o il protocollo MtErie.

Il file sorgente del protocollo è
[`rtdetr_carnation_stress_test.yaml`](../parameters/RTDETR/rtdetr_carnation_stress_test.yaml).
Il runner resumable è
[`run_rtdetr_carnation_stress_test.py`](../scripts/run_rtdetr_carnation_stress_test.py).

## Domanda sperimentale

La coppia Carnation presenta una differenza di scala fra RGB e IR più forte
rispetto alle acquisizioni del benchmark interno. La domanda è quindi:

> il vantaggio medio del FAM standard rispetto ad Additive persiste, si annulla
> o cambia segno quando i dieci checkpoint finali vengono trasferiti senza
> tuning a questa acquisizione con forte mismatch geometrico?

Un risultato negativo descriverà un limite di generalizzazione sotto scale
diverse fra sensori. Non confuterà il confronto appaiato MtErie e non potrà
essere usato per scegliere retroattivamente un'altra variante FAM.

## Dati bloccati

Le sole sequenze ammesse sono:

- `210529_Carnation_Enterprise_VIS_0023`;
- `210529_Carnation_Enterprise_IR_0024`.

Il ramo VIS contiene 739 frame, mentre il ramo IR contiene un frame finale
aggiuntivo (`1045`). Il protocollo usa l'intersezione degli identificatori
numerici, ottenendo gli stessi **739 frame** in VIS+IR, VIS e IR. Nessun frame
viene scelto sulla base delle predizioni.

Prima di ogni inferenza il runner verifica:

- hash dei 739 identificatori comuni;
- inventario di immagini e annotazioni;
- 1.942 box e quattro frame vuoti nel ramo VIS;
- 1.713 box e quattro frame vuoti nel ramo IR;
- esclusione del solo frame IR `1045`.

VIS+IR viene costruito dal loader esistente tramite `adapt_ir2rgb` e valutato
nel sistema di coordinate e con le annotazioni VIS. VIS-only usa le annotazioni
VIS; IR-only usa le annotazioni IR. Nelle modalità singole i canali mancanti
sono azzerati, coerentemente con la valutazione finale RT-DETR. Modal Dropout è
sempre disattivato in inferenza.

## Modelli e protocollo bloccato

Sono ammesse soltanto due configurazioni:

| Configurazione | Progetto checkpoint | FAM |
|---|---|---:|
| Additive | `RTDETR_Protocol` | no |
| FAM standard | `RTDETR_FAM_Protocol` | `current_dcnv2` |

Per entrambe:

- seed appaiati `40--44`;
- checkpoint finale `latest` del protocollo fisso a 10 epoche;
- soglia di confidenza `0.01`;
- seed di evaluation `42`;
- modalità VIS+IR, VIS e IR;
- nessun training, early stopping, tuning o selezione di seed;
- unità sperimentale: checkpoint/seed.

La campagna contiene quindi 2 configurazioni × 5 seed × 3 modalità = **30
valutazioni**. Per ogni modalità verranno riportati i valori per seed, media,
mediana, deviazione standard campionaria, min--max, IC t al 95%, delta appaiati
FAM meno Additive, vittorie su cinque seed e test appaiati esplorativi. La
mAP@50 resta la metrica primaria; mAP COCO, mAP@75, mAR e metriche per
dimensione vengono conservate come secondarie.

## Vincoli interpretativi predefiniti

I risultati Carnation non possono essere usati per:

- selezionare modello, seed, checkpoint o soglia;
- progettare e provare un'altra correzione sullo stesso stress set;
- decidere un nuovo training sulla base della sola metrica Carnation;
- presentare Carnation come campione rappresentativo della distribuzione SAR;
- sostituire MtErie nella conclusione sperimentale principale.

Carnation viene consultata una sola volta dopo il congelamento e il versionamento
di questi vincoli.

## Sequenza operativa

Verifica dell'inventario, senza caricare checkpoint o calcolare metriche:

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
/home/donato/miniconda3/envs/sarfusion/bin/python \
  scripts/run_rtdetr_carnation_stress_test.py \
  --prepare-only
```

Risoluzione dei dieci checkpoint, ancora senza inferenza:

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
/home/donato/miniconda3/envs/sarfusion/bin/python \
  scripts/run_rtdetr_carnation_stress_test.py \
  --dry-run
```

Dopo commit e push del protocollo, smoke tecnico su un solo batch e una coppia
di checkpoint, in una directory separata:

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
MPLCONFIGDIR=/tmp/matplotlib-rtdetr-carnation-smoke \
/home/donato/miniconda3/envs/sarfusion/bin/python \
  scripts/run_rtdetr_carnation_stress_test.py \
  --device cuda \
  --configurations additive fam \
  --seeds 43 \
  --modalities vis_ir \
  --max-batches 1 \
  --output-dir out/rtdetr_carnation_stress_test_smoke
```

Campagna completa resumable:

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
MPLCONFIGDIR=/tmp/matplotlib-rtdetr-carnation \
PYTHONUNBUFFERED=1 \
/home/donato/miniconda3/envs/sarfusion/bin/python \
  scripts/run_rtdetr_carnation_stress_test.py \
  --device cuda \
  2>&1 | tee rtdetr_carnation_stress_test.log
```

Il completamento richiede 30 JSON compatibili, 739 campioni in ciascuno e
`protocol_complete: true` nell'aggregato. Non usare `--force` salvo dopo avere
ispezionato manualmente un output incompatibile.

## Risultati mAP@50

I valori grezzi per seed sono:

| Seed | Additive VIS+IR | Additive VIS | Additive IR | FAM VIS+IR | FAM VIS | FAM IR |
|---:|---:|---:|---:|---:|---:|---:|
| 40 | 0.0701 | 0.0707 | 0.2097 | 0.2327 | 0.2890 | 0.1944 |
| 41 | 0.1368 | 0.1855 | 0.1460 | 0.2597 | 0.3953 | 0.1056 |
| 42 | 0.0953 | 0.0995 | 0.0394 | 0.2805 | 0.3241 | 0.0582 |
| 43 | 0.2517 | 0.2274 | 0.1923 | 0.1801 | 0.2572 | 0.1104 |
| 44 | 0.1688 | 0.1943 | 0.0566 | 0.2507 | 0.3325 | 0.0212 |

L'aggregazione tratta il checkpoint come unità sperimentale:

| Modalità | Additive, media ± SD | FAM, media ± SD | Δ FAM − Additive, media ± SD | IC 95% Δ | Vittorie FAM |
|---|---:|---:|---:|---:|---:|
| VIS+IR | 0.1445 ± 0.0709 | **0.2408 ± 0.0380** | +0.0962 ± 0.1017 | [−0.0301, +0.2225] | 4/5 |
| VIS | 0.1555 ± 0.0669 | **0.3196 ± 0.0518** | +0.1641 ± 0.0828 | [+0.0614, +0.2669] | 5/5 |
| IR | **0.1288 ± 0.0776** | 0.0979 ± 0.0652 | −0.0308 ± 0.0368 | [−0.0766, +0.0149] | 1/5 |

I test appaiati, riportati soltanto come esplorativi con `n=5`, producono:

| Modalità | p t-test appaiato | p Wilcoxon esatto |
|---|---:|---:|
| VIS+IR | 0.1020 | 0.1250 |
| VIS | 0.0114 | 0.0625 |
| IR | 0.1346 | 0.1875 |

Il risultato VIS+IR conserva quindi la direzione osservata su MtErie, ma non la
stessa uniformità: FAM migliora Additive in quattro seed su cinque, aumenta la
mAP@50 media di `+0.0962` e presenta una deviazione standard inferiore. Il
delta cambia però segno nel seed 43 e il suo intervallo t al 95% attraversa lo
zero; non va descritto come superiorità universale su ogni inizializzazione.

## La fusione non supera VIS-only

Il confronto più importante dello stress test non è soltanto FAM contro
Additive. Con Additive, VIS+IR supera VIS-only in 1/5 seed e il delta medio
VIS+IR meno VIS è `−0.0109`; rispetto alla migliore modalità singola il delta
medio è `−0.0387`, ancora con una sola vittoria.

Con FAM, VIS è la migliore modalità in tutti i seed. VIS+IR resta inferiore a
VIS-only in **5/5 checkpoint**, con delta medio `−0.0789 ± 0.0353` e IC 95%
`[−0.1227, −0.0351]`. La fusione FAM supera invece IR-only in 5/5 seed, con
delta medio `+0.1428`.

La conclusione corretta è pertanto duplice:

1. rispetto alla fusione additiva, FAM attenua in modo sostanziale il danno
   causato dal mismatch geometrico Carnation;
2. il FAM corrente non compensa abbastanza tale mismatch da rendere utile
   aggiungere IR al forte ramo VIS: la configurazione FAM VIS-only rimane
   superiore alla stessa configurazione VIS+IR in tutti i seed.

Questo esito non dimostra una registrazione fisica e non contraddice la
diagnostica interna. È coerente con il FAM come trasformazione
geometrico-funzionale capace anche di filtrare un contributo IR dannoso, ma non
come correttore generale di differenze di scala arbitrarie.

## Metriche secondarie e confronto descrittivo con MtErie

Su VIS+IR, la mAP COCO media passa da `0.0536` Additive a `0.0821` FAM e la
mAR@100 da `0.2025` a `0.2828`; entrambe migliorano in 4/5 seed. Su VIS-only,
la mAP passa da `0.0571` a `0.1102` e la mAR@100 da `0.2090` a `0.3231`, con
cinque miglioramenti su cinque. Su IR-only, mAP e mAR@100 scendono invece da
`0.0263` a `0.0210` e da `0.1662` a `0.1587`. Tutte le istanze Carnation
risultano nella categoria COCO small alla risoluzione di evaluation; mAP
medium e large sono quindi non definite e non vengono mediate come valori
reali.

Rispetto alle valutazioni MtErie eseguite con lo stesso schema di modalità, la
mAP@50 VIS+IR media scende descrittivamente da `0.3081` a `0.1445` per Additive
e da `0.3780` a `0.2408` per FAM. Il calo è più contenuto per FAM, ma su
Carnation viene meno il risultato più stabile di MtErie, dove VIS+IR superava
la migliore modalità singola in ogni checkpoint. Il confronto tra sessioni
non è un test statistico di dominio: differiscono acquisizione, geometria e
annotazioni e Carnation è stato scelto appositamente come caso estremo.

## Cautela sulle modalità singole

La valutazione RT-DETR mono-modale mantiene il contratto a quattro canali:
l'input mancante viene azzerato, ma entrambe le backbone vengono eseguite. Nel
modello FAM anche il modulo di allineamento resta attivo sulle feature prodotte
dal ramo con input nullo, che possono essere non nulle per bias e
normalizzazioni. Il grande vantaggio FAM in VIS-only non è quindi evidenza di
allineamento tra due segnali reali durante quell'inferenza; descrive la
robustezza complessiva appresa dal modello sotto il regime di Modal Dropout.
Analogamente, il risultato IR-only include il comportamento del FAM quando il
riferimento RGB reale è assente.

## Artefatti

Gli output completi e i 30 JSON restano in
`out/rtdetr_carnation_stress_test/`. Gli artefatti compatti versionati sono:

- [`rtdetr_carnation_stress_test.csv`](Search_and_Rescue/results/rtdetr_carnation_stress_test.csv),
  con le 30 unità checkpoint/modalità;
- [`rtdetr_carnation_paired_map50.png`](Search_and_Rescue/images/rtdetr_carnation_paired_map50.png),
  con i confronti appaiati per seed.

L'aggregato completo ha SHA-256:

```text
00916ece2484062b9c591602dd9ccf9129546965213465cbf084b1a12e7484c4
```

Lo stress set è ora chiuso. In accordo con il protocollo predefinito, non si
avviano nuove varianti, tuning o training sulla base di questi risultati.
