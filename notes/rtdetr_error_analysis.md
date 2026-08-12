# Error analysis finale RT-DETR Additive--FAM

## Stato

Il protocollo è stato bloccato il 12 agosto 2026 prima dell'inferenza completa.
Il manifest qualitativo è stato generato usando soltanto le annotazioni e lo
smoke test ha verificato caricamento, inferenza, matching e aggregazione per
Additive e FAM. I risultati dello smoke non costituiscono evidenza
sperimentale e sono salvati soltanto in una cartella `out` ignorata da Git.

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
  `out/rtdetr_error_analysis/figures/rtdetr_additive_fam_error_summary.png`.

Lo smoke test sui due checkpoint seed 43 ha verificato l'intero percorso su un
batch e ha marcato correttamente l'output come `protocol_complete: false`.

## Comando completo

Prima dell'esecuzione vanno versionati runner, test, manifest e questa nota.
La campagna completa si avvia con:

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

L'esecuzione completa risolve dieci checkpoint e non esegue training. In caso
di interruzione si rilancia lo stesso comando: i JSON compatibili già completi
vengono saltati. Non usare `--force` salvo dopo avere ispezionato manualmente
gli output esistenti.

Monitoraggio da un secondo terminale:

```bash
tail -f rtdetr_error_analysis.log
```

Al termine devono esistere dieci JSON grezzi e il file aggregato deve avere
`protocol_complete: true`. Solo allora si interpretano i risultati e si
aggiornano i Markdown della tesi.
