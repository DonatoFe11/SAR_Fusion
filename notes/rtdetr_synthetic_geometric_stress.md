# RT-DETR: stress geometrico sintetico controllato

## Stato

**Protocollo congelato il 30 agosto 2026 prima di qualunque inferenza con IR
sinteticamente perturbata. Campagna completata il 31 agosto 2026: 560/560
nuove inferenze e 600/600 punti curva includendo le identità riusate.**

Protocollo: `rtdetr_synthetic_geometric_stress_v1`.

La configurazione eseguibile è
[`rtdetr_synthetic_geometric_stress.yaml`](../parameters/RTDETR/rtdetr_synthetic_geometric_stress.yaml).
L'hash SHA-256 canonico del blocco scientifico è
`dd3d92151a3a11cdf78cb4c25edd2744e04c4c6792730202d77ce8117543077f`.

## Domanda sperimentale

La conferma sulle acquisizioni WiSARD precedentemente inutilizzate ha mostrato
che FAM storico supera Additive in tutti i seed, ma anche che la modalità IR
non migliora uniformemente lo stesso FAM Stage B. Lo stress controllato chiede:

> quanto decade la mAP@50 di ciascun checkpoint quando soltanto la modalità IR
> viene traslata o riscalata con intensità fissate a priori?

L'obiettivo è una curva di sensibilità rispetto all'identità, non trovare una
trasformazione che aumenti la metrica né simulare una calibrazione fisica.

Lo stress Carnation 0023/0024 già completato rimane distinto: misura un
mismatch nativo di una specifica coppia di sensori. Qui le stesse perturbazioni
note sono applicate a due acquisizioni, mantenendo invariati RGB e ground truth
VIS.

## Sorgenti bloccate

Si riusano esattamente gli inventari corretti della conferma:

| Acquisizione | Coppie | Ground truth | Hash inventario |
|---|---:|---|---|
| Carnation 0025/0026 | 1.313 | VIS | `29eb9ff1…641` |
| FHL 0407/0408 | 1.035 | VIS | `dd4a2c81…9a9d` |

Il pairing resta l'intersezione degli identificativi numerici. Il protocollo
della conferma è vincolato dall'hash scientifico
`b36e950aa451f7861faad2e1615dd94471c823ca25e0b2eebacd80cb1c3dce96`.
L'identità non viene ricalcolata: si riusano i 40 risultati VIS+IR pertinenti
del CSV già versionato, SHA-256
`a314bbf1d5eb7ffce296945a4f892231c2cbb713ba1436295c9941a73b4233c8`.

## Trasformazioni bloccate

La trasformazione agisce soltanto sul quarto canale del tensore normalizzato
`[4, 640, 640]`, dopo il preprocessing comune. RGB, `pixel_mask` e annotazioni
VIS non cambiano. Interpolazione bilineare, riempimento neutro `0.0` e centro
dell'immagine sono fissati.

Le traslazioni sono valutate a magnitudine 8, 16 e 32 pixel nelle quattro
direzioni cardinali: destra, sinistra, basso e alto. Per ogni seed la curva per
magnitudine usa la media a peso uguale delle quattro direzioni; tutti i valori
direzionali restano comunque disponibili.

Le scale sono `0.9` e `1.1` attorno al centro. Traslazione e scala costituiscono
due famiglie indipendenti con la stessa identità: non vengono composte in una
grid. Ne risultano 15 condizioni per checkpoint/acquisizione:

- una identità riusata dalla conferma;
- dodici traslazioni, quattro direzioni per tre magnitudini;
- due scale.

## Checkpoint e confronti

Restano congelati gli stessi venti checkpoint `latest`, seed 40--44:

| Famiglia | Riferimento | Candidato | Confronto ammesso |
|---|---|---|---|
| storico bloccato | Additive | FAM | tolleranza FAM rispetto ad Additive |
| current-code Stage B | FAM | RCRA | tolleranza RCRA rispetto a FAM |

Le accuratezze assolute fra famiglie non vengono usate come classifica perché
i checkpoint appartengono a campagne e training set differenti. Il confronto
di robustezza usa, per ogni checkpoint e acquisizione:

```text
drop relativo = (mAP@50 identità - mAP@50 perturbata) / mAP@50 identità
```

Un valore positivo indica degradazione; un valore negativo indica un aumento
accidentale sotto la perturbazione e viene riportato senza trasformarlo in una
nuova configurazione. Si conserva anche il delta assoluto
`perturbata - identità`.

L'identità più quattordici perturbazioni per quattro configurazioni, cinque
seed e due acquisizioni producono 600 punti curva complessivi, dei quali 560
richiedono nuova inferenza.

## Vincoli interpretativi

I risultati non possono essere usati per:

- selezionare trasformazione, direzione o intensità;
- scegliere modello, checkpoint, seed o soglia;
- modificare FAM o RCRA e rivalutarli sulle stesse curve;
- classificare direttamente checkpoint storici e Stage B per accuratezza;
- dichiarare calibrazione fisica o tolleranza a ogni disallineamento 2D;
- presentare le acquisizioni come holdout esterno.

Le direzioni e le scale sono state fissate senza consultare predizioni
perturbate. Risultati negativi e aumenti accidentali devono essere riportati.

## Sequenza operativa

1. validare protocollo sorgente, CSV identità, inventari e venti checkpoint;
2. testare matematicamente che soltanto il canale IR venga trasformato;
3. eseguire un smoke test tecnico separato, non scientifico, su un solo batch;
4. eseguire la campagna resumable salvando una perturbazione alla volta;
5. verificare 600 punti complessivi e generare curve per acquisizione e macro;
6. versionare CSV, figure e risultati nel presente Markdown.

L'integrazione nella tesi avverrà soltanto dopo il completamento, insieme alla
conferma sulle acquisizioni inutilizzate.

## Preflight completato

Prima della campagna scientifica sono stati completati:

- quattro test automatici su protocollo, trasformazione del solo canale IR e
  requisito dei 600 punti complessivi;
- verifica dei due inventari e dei loro hash;
- verifica dei venti checkpoint e dei relativi SHA-256;
- verifica byte-per-byte del CSV identità e dei 40 baseline VIS+IR riusati;
- espansione deterministica delle 15 condizioni senza prodotti cartesiani;
- dry-run di 40 identità riusate e 560 nuove inferenze;
- smoke test GPU su un batch Carnation, FAM seed 40 e traslazione di 8 pixel a
  destra, con tutte le chiavi del checkpoint corrispondenti.

Lo smoke ha usato `--max-batches 1` e una directory separata in `/tmp`. La sua
metrica su dodici campioni non è un risultato scientifico e non può essere
caricata dalla campagna completa.

## Esecuzione completata

La campagna salva atomicamente un JSON per ogni combinazione acquisizione,
configurazione, seed e perturbazione. La prima parte dell'esecuzione è stata
fermata operativamente dopo **366/560** risultati completi. Il riavvio del 31
agosto ha rivalidato sorgenti e checkpoint, ha saltato quelle 366 unità e ha
calcolato le 194 rimanenti. L'unità interrotta prima della scrittura è stata
ricalcolata integralmente. Le pause non hanno modificato protocollo,
trasformazioni, checkpoint o metriche e nessun aggregato parziale è stato usato
per scegliere la prosecuzione.

L'aggregato finale attesta `protocol_complete: true`, 560 nuove inferenze e
600 punti complessivi. L'unità sperimentale resta il checkpoint/seed entro
acquisizione: i frame non sono trattati come repliche indipendenti.

Comando della campagna completa resumable:

```bash
MPLCONFIGDIR=/tmp/rtdetr_synthetic_stress_mpl \
YOLO_CONFIG_DIR=/tmp/rtdetr_synthetic_stress_yolo \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
conda run --no-capture-output -n sarfusion python -u \
  scripts/run_rtdetr_synthetic_geometric_stress.py --device cuda
```

## Risultati: risposta con segno

La tabella riporta il `drop relativo` medio sui cinque seed, in percentuale.
Per le traslazioni è prima calcolata, entro seed, la media a peso uguale delle
quattro direzioni. Valori positivi indicano perdita rispetto all'identità;
valori negativi indicano un miglioramento accidentale sotto perturbazione.

| Acquisizione | Configurazione | 8 px | 16 px | 32 px | scala 0,9 | scala 1,1 |
|---|---|---:|---:|---:|---:|---:|
| Carnation | Additive storico | -0,07% | -0,40% | +0,66% | +5,46% | -0,77% |
| Carnation | FAM storico | -0,07% | -0,24% | -0,13% | +1,76% | -0,54% |
| Carnation | FAM Stage B | -0,05% | +0,15% | +0,68% | +2,03% | -1,84% |
| Carnation | RCRA Stage B | -0,17% | -0,11% | +1,34% | +0,09% | +0,69% |
| FHL | Additive storico | -3,15% | -6,38% | -6,93% | -2,31% | -2,13% |
| FHL | FAM storico | -0,57% | -0,63% | -0,56% | -1,25% | +0,65% |
| FHL | FAM Stage B | -0,45% | -1,92% | -2,46% | -0,11% | +0,73% |
| FHL | RCRA Stage B | -1,13% | -2,03% | -2,14% | -2,50% | +0,56% |

La risposta non è monotona e dipende fortemente dall'acquisizione. Su
Carnation le traslazioni producono variazioni medie piccole fino a 16 pixel;
la scala 0,9 penalizza soprattutto Additive. Su FHL, invece, quasi tutte le
trasformazioni aumentano la mAP@50, in particolare per Additive. Questo mostra
che l'identità del tensore preprocessato non coincide necessariamente con un
ottimo geometrico per la detection: una perturbazione sintetica può compensare
in parte un mismatch nativo oppure cambiare favorevolmente le feature, senza
costituire una calibrazione valida.

## Confronto appaiato di robustezza

Il contrasto congelato è `drop riferimento - drop candidato`: un valore
positivo favorisce il candidato perché perde meno; un valore negativo significa
che il riferimento ha una risposta con segno più favorevole. Gli intervalli
sono IC t al 95% su cinque differenze appaiate e sono necessariamente incerti.

### FAM storico rispetto ad Additive

| Acquisizione | Condizione | Vantaggio FAM | IC95% | Seed favorevoli |
|---|---|---:|---:|---:|
| Carnation | traslazione 8 px | -0,00 pp | [-1,27; +1,26] | 2/5 |
| Carnation | traslazione 16 px | -0,17 pp | [-2,37; +2,04] | 2/5 |
| Carnation | traslazione 32 px | +0,79 pp | [-5,27; +6,86] | 2/5 |
| Carnation | scala 0,9 | +3,70 pp | [-2,00; +9,40] | 3/5 |
| Carnation | scala 1,1 | -0,23 pp | [-4,83; +4,36] | 1/5 |
| FHL | traslazione 8 px | -2,58 pp | [-3,75; -1,41] | 0/5 |
| FHL | traslazione 16 px | -5,75 pp | [-8,40; -3,09] | 0/5 |
| FHL | traslazione 32 px | -6,38 pp | [-10,50; -2,25] | 0/5 |
| FHL | scala 0,9 | -1,06 pp | [-7,56; +5,44] | 3/5 |
| FHL | scala 1,1 | -2,78 pp | [-11,80; +6,24] | 1/5 |

Carnation non mostra un vantaggio uniforme di FAM; l'indicazione favorevole a
scala 0,9 è variabile e il relativo intervallo include zero. Su FHL il contrasto
con segno favorisce Additive per tutte le magnitudini di traslazione e in 5/5
seed, ma la causa è un **aumento** anomalo di Additive rispetto alla propria
identità, non un maggiore crollo di FAM. Non è quindi corretto trasformare
questo risultato nella conclusione che Additive sia intrinsecamente più
robusto. La conclusione ammessa è più limitata: la metrica primaria
pre-specificata non conferma una superiore tolleranza geometrica generale di
FAM e rivela un mismatch nativo o una sensibilità direzionale di FHL.

### RCRA Stage B rispetto a FAM Stage B

| Acquisizione | Condizione | Vantaggio RCRA | IC95% | Seed favorevoli |
|---|---|---:|---:|---:|
| Carnation | traslazione 8 px | +0,12 pp | [-1,27; +1,50] | 3/5 |
| Carnation | traslazione 16 px | +0,27 pp | [-2,24; +2,77] | 3/5 |
| Carnation | traslazione 32 px | -0,67 pp | [-8,24; +6,91] | 2/5 |
| Carnation | scala 0,9 | +1,94 pp | [-0,25; +4,12] | 4/5 |
| Carnation | scala 1,1 | -2,53 pp | [-10,69; +5,62] | 2/5 |
| FHL | traslazione 8 px | +0,67 pp | [-0,60; +1,94] | 4/5 |
| FHL | traslazione 16 px | +0,11 pp | [-2,08; +2,30] | 4/5 |
| FHL | traslazione 32 px | -0,31 pp | [-4,78; +4,15] | 2/5 |
| FHL | scala 0,9 | +2,40 pp | [-0,13; +4,92] | 4/5 |
| FHL | scala 1,1 | +0,16 pp | [-5,45; +5,78] | 2/5 |

Tutti gli intervalli includono zero e il segno cambia con intensità,
acquisizione e seed. Lo stress non fornisce quindi evidenza di un vantaggio
stabile di RCRA rispetto a FAM. Questo è coerente con la decisione precedente
di non promuovere RCRA, ma non costituisce una nuova selezione del modello.

## Conclusione scientifica

Lo stress risponde negativamente alla formulazione forte «FAM è sempre più
tollerante al disallineamento». FAM mantiene il vantaggio di accuratezza su
Additive dimostrato dalla conferma sulle acquisizioni inutilizzate, ma tale
vantaggio non può essere attribuito in generale a una curva di degrado
geometrico più favorevole. RCRA non corregge questa conclusione in modo
riproducibile.

Il risultato più informativo è metodologico: traslazioni e scale sintetiche
possono migliorare una sequenza già disallineata e gli effetti sono anisotropi.
Per sostenere una proprietà fisica di tolleranza servirebbero calibrazione nota,
ground truth geometrica o acquisizioni controllate, non la ricerca post-hoc di
una trasformazione favorevole. Nessuna trasformazione osservata viene quindi
adottata come preprocessing e la scelta finale del FAM standard non cambia.

## Artefatti e integrità

- suite finale di regressione: 23/23 test superati, includendo split dei dati,
  Modal Dropout paired-VIS, Stage B FAM/RCRA, conferma inutilizzata e stress
  sintetico;
- output grezzi e aggregato JSON locale:
  `out/rtdetr_synthetic_geometric_stress/`;
- 560 JSON grezzi ordinati, hash composito SHA-256
  `a5f2d1e69ad043b61fed108c6fa00b697989418e7689d7e9df6a24900af7f3f8`;
- aggregato JSON completo, SHA-256
  `96d3ac9be143452cbbdb2ce359169aa8b4d9f39928ed9e5628651279cec7e2bd`;
- [CSV versionato con i 600 punti](Search_and_Rescue/results/rtdetr_synthetic_geometric_stress.csv),
  SHA-256 `f37b984c61e9aab51afa7706d57407d898048c9feb9691a5fb5e40d3405613ef`;
- [curve versionate](Search_and_Rescue/images/rtdetr_synthetic_geometric_stress_curves.png),
  SHA-256 `c94eb817972c06abbfe2c742c69432c7c140d8343988947424bf11b2808b5854`.
