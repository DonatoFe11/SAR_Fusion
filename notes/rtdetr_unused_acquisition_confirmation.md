# RT-DETR: conferma su acquisizioni WiSARD precedentemente inutilizzate

## Stato

**Campagna completata: protocollo, inventari e checkpoint sono stati congelati
prima dell'inferenza, l'attestazione dell'autore è stata registrata il 30 agosto
2026 e tutte le 50 unità previste sono state calcolate. FAM storico supera
Additive in tutti i dieci confronti acquisizione/seed; l'apporto della modalità
IR nei checkpoint Stage B non è invece uniforme e RCRA resta instabile.**

Protocollo: `rtdetr_unused_acquisition_confirmation_v1`.

Data di congelamento scientifico: 30 agosto 2026.

La configurazione eseguibile è
[`rtdetr_unused_acquisition_confirmation.yaml`](../parameters/RTDETR/rtdetr_unused_acquisition_confirmation.yaml)
e il runner fail-closed è
[`run_rtdetr_unused_acquisition_confirmation.py`](../scripts/run_rtdetr_unused_acquisition_confirmation.py).

## Motivazione

MtErie è stato consultato ripetutamente durante lo sviluppo e non costituisce
un holdout cieco. Anche la validazione Stage A FHL 0401/0402, pur separando un
video completo, appartiene alla stessa campagna di parte dei dati di training.
La tesi ha quindi bisogno soprattutto di verificare se i risultati principali
persistono su sequenze alle quali i checkpoint non sono stati applicati durante
la campagna archiviata.

Il repository locale contiene due coppie sincronizzate con annotazioni VIS ma
senza annotazioni IR che non compaiono nei parametri versionati né negli output
locali delle campagne:

| Acquisizione | Coppie | Box VIS | Frame VIS vuoti | Label IR |
|---|---:|---:|---:|---:|
| Carnation 0025/0026 | 1.313 | 5.238 | 100 | assenti |
| FHL 0407/0408 | 1.035 | 2.022 | 239 | assenti |

I conteggi sono stati ottenuti senza eseguire modelli. FHL ha gli stessi 1.035
identificativi numerici nei due flussi. Carnation contiene invece 1.315 immagini
per stream ma soltanto 1.313 identificativi comuni: VIS `654` e `983` non hanno
la controparte con lo stesso ID in IR, mentre IR `1454` e `1455` non hanno la
controparte VIS. Questi quattro file sono esclusi esplicitamente.

Il normale `sorted zip` avrebbe conservato 1.315 righe, ma avrebbe associato
ID differenti in 661 posizioni dopo i primi buchi. Il protocollo usa pertanto
soltanto l'intersezione degli identificativi numerici, senza stimare o cercare
uno shift temporale. Questo audit corregge il conteggio preliminare di 1.315
"coppie" e impedisce una pseudo-replica temporalmente disallineata. Percorsi,
dimensioni, contenuti completi delle immagini, annotazioni VIS e identificativi
sono riassunti da hash SHA-256 congelati nel protocollo.

FHL 0126/0127 non entra nel risultato principale. Ha 1.006 frame VIS e 1.008
frame IR; due annotazioni VIS (`00000211` e `00000212`) sono file vuoti già
registrati in `MISSING_ANNOTATIONS`. Potrà essere oggetto di un audit separato,
ma non verrà aggiunta dopo avere osservato i risultati delle due acquisizioni
primarie.

## Attestazione necessaria prima dell'esecuzione

Attestazione registrata il 30 agosto 2026: l'autore ha dichiarato di non avere
mai visionato manualmente Carnation 0025/0026 e FHL 0407/0408 prima del
congelamento. L'audit del repository e degli archivi locali non ha rilevato un
loro uso nelle configurazioni versionate o nelle run conservate. È stata quindi
registrata la condizione
`no_prior_model_or_manual_experimental_use`, senza modificare il blocco
scientifico già congelato.

Il repository e gli archivi locali possono escludere soltanto usi registrati.
Prima dell'inferenza l'autore deve indicare una delle due condizioni:

1. non ha consultato queste sequenze in prove manuali o esperimenti non
   archiviati che abbiano influenzato modello, checkpoint, soglia o protocollo;
2. esiste un uso precedente oppure non è possibile escluderlo con sicurezza.

Nel primo caso la tesi userà l'espressione *previously unused internal
acquisition-level confirmation sets*. Nel secondo verranno descritte soltanto
come *additional internal acquisitions*. In nessun caso saranno chiamate
holdout esterno o campione rappresentativo della popolazione SAR.

L'attestazione è conservata fuori dal blocco scientifico `protocol` del file
YAML, così la registrazione successiva non modifica le scelte sperimentali già
congelate. Il runner rifiuta l'inferenza finché lo stato resta `pending`.

## Checkpoint congelati

Si usano esclusivamente checkpoint `latest` già esistenti, seed 40--44:

| Blocco | Configurazioni | Ruolo |
|---|---|---|
| storico | `RTDETR_Protocol` Additive e `RTDETR_FAM_Protocol` FAM | replica del contrasto centrale storico |
| current-code Stage B | `RTDETR_FAM_FullData_StageB_FiveSeed` e `RTDETR_FAM_RCRA_FullData_StageB_FiveSeed` | verifica descrittiva del vantaggio RCRA instabile |

Gli SHA-256 dei venti checkpoint sono parte del protocollo. Il runner fallisce
se la risoluzione locale seleziona un file diverso. Additive storico non viene
confrontato direttamente con FAM current-code: i due appartengono a epoche
implementative diverse.

Il preflight del 30 agosto 2026 ha risolto univocamente tutti i venti file e ne
ha verificato gli hash senza costruire o eseguire i modelli. Per i dieci
checkpoint current-code ha anche ricontrollato progetto, seed, configurazione,
epoca finale, numero di optimizer step e assenza di metriche test automatiche.
La decisione Stage B viene ricalcolata dal CSV versionato e deve continuare a
essere `fail_retain_fam`.

## Condizioni e confronti

Ogni valutazione usa il medesimo inventario paired e la ground truth VIS. La
condizione VIS-only deriva dallo stesso tensore fusion azzerando soltanto il
canale IR adattato; Modal Dropout è disabilitato.

I tre contrasti predefiniti sono:

1. FAM storico meno Additive storico in VIS+IR;
2. RCRA Stage B meno FAM Stage B in VIS+IR;
3. FAM Stage B VIS+IR meno lo stesso FAM Stage B VIS-only.

La metrica primaria descrittiva è mAP@50 con soglia di preprocessing delle
predizioni `0.01`. Sono conservate anche COCO mAP, mAP@75, mAP per dimensione e
mAR. Per ogni acquisizione si riportano tutti i seed, media, deviazione standard
campionaria, mediana, intervallo t al 95%, numero di delta positivi e test
appaiati esplorativi.

Le due acquisizioni non vengono concatenate fingendo che i frame siano
repliche indipendenti. Un eventuale riepilogo comune è la macro-media a peso
uguale delle due acquisizioni, calcolata per seed e dichiarata descrittiva.

## Vincoli interpretativi congelati

I risultati non possono essere usati per:

- scegliere modello, checkpoint, seed o soglia;
- modificare FAM o RCRA;
- decidere da soli un nuovo training;
- omettere un'acquisizione o un seed sfavorevole;
- sostenere generalizzazione a un holdout esterno o alla popolazione SAR.

FAM resta il modello selezionato dalla regola Stage B già chiusa. Un eventuale
vantaggio RCRA su queste sequenze è nuova evidenza descrittiva, non una
selezione retroattiva. Risultati negativi devono essere riportati integralmente.

## Sequenza operativa

1. costruire gli inventari e congelarne gli hash senza caricare modelli;
2. risolvere e congelare i venti checkpoint senza inferenza;
3. registrare l'attestazione dell'autore;
4. eseguire una sola campagna completa e resumable;
5. verificare 50 unità acquisizione/configurazione/seed/condizione;
6. versionare la tabella compatta e aggiornare la tesi;
7. soltanto dopo, avviare il protocollo separato di stress geometrico
   sintetico, senza usare questa conferma per il tuning.

Comandi previsti:

```bash
MPLCONFIGDIR=/tmp/rtdetr_confirmation_mpl \
YOLO_CONFIG_DIR=/tmp/rtdetr_confirmation_yolo \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
conda run --no-capture-output -n sarfusion python -u \
  scripts/run_rtdetr_unused_acquisition_confirmation.py --inventory-only

MPLCONFIGDIR=/tmp/rtdetr_confirmation_mpl \
YOLO_CONFIG_DIR=/tmp/rtdetr_confirmation_yolo \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
conda run --no-capture-output -n sarfusion python -u \
  scripts/run_rtdetr_unused_acquisition_confirmation.py --dry-run

MPLCONFIGDIR=/tmp/rtdetr_confirmation_mpl \
YOLO_CONFIG_DIR=/tmp/rtdetr_confirmation_yolo \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
conda run --no-capture-output -n sarfusion python -u \
  scripts/run_rtdetr_unused_acquisition_confirmation.py
```

Il primo comando non risolve né carica checkpoint. Il secondo risolve e valida
inventari e checkpoint ma non carica modelli. Il terzo rimane bloccato finché
l'attestazione non è registrata.

## Preflight completato

Prima di qualunque inferenza sono stati completati:

- hash del contenuto di tutte le 4.696 immagini incluse nei due stream paired,
  oltre a percorsi, dimensioni e annotazioni VIS;
- rilevamento e congelamento dei quattro ID Carnation esclusi;
- costruzione reale di un campione `[4, 640, 640]` per ciascuna acquisizione,
  con ground truth VIS e IR priva di annotazioni;
- risoluzione e SHA-256 dei venti checkpoint;
- ricostruzione della decisione Stage B che mantiene FAM;
- dry-run delle 50 unità previste;
- 19 test mirati su loader, Modal Dropout, Stage B e nuovo protocollo, tutti
  superati.

Il loader conteneva già il sentinel `""` per una cartella IR senza label, ma
`load_annotations` tentava comunque di aprirlo come file. Il comportamento è
stato completato in modo conservativo: soltanto il percorso esplicitamente
vuoto restituisce zero annotazioni; un percorso non vuoto ma mancante continua
a produrre errore. Questa correzione non cambia alcun dataset delle campagne
precedenti ed è coperta dai test.

## Esecuzione e ripresa

La campagna è stata eseguita su GPU senza modificare i parametri congelati. La
prima sessione ha raggiunto il limite temporale dell'esecutore dopo avere
salvato 42 delle 50 unità. Il secondo avvio ha rivalidato protocollo, inventari
e checkpoint, ha verificato la compatibilità dei 42 file grezzi, li ha saltati
e ha calcolato soltanto le otto unità RCRA mancanti. Il completamento finale è
avvenuto con exit code zero e il runner ha attestato
`Protocol complete: all 50 frozen confirmation results are present`.

L'interruzione non ha cambiato l'ordine logico, il batch size, la soglia, i
checkpoint o le metriche. Ogni file grezzo viene scritto soltanto al termine
dell'intera unità acquisizione/configurazione/seed/condizione, quindi non sono
stati riutilizzati risultati parziali.

## Risultati principali

La tabella seguente riporta la media sui cinque seed della mAP@50. I delta sono
sempre `candidato - riferimento`; la deviazione standard e l'intervallo t al
95% riguardano i cinque delta appaiati, non i singoli frame.

| Contrasto | Acquisizione | Riferimento | Candidato | Delta medio ± SD | IC95% delta | Vittorie |
|---|---|---:|---:|---:|---:|---:|
| FAM storico − Additive storico | Carnation | 0,3945 | 0,4905 | +0,0960 ± 0,0595 | [+0,0221; +0,1699] | 5/5 |
| FAM storico − Additive storico | FHL | 0,2953 | 0,4507 | +0,1554 ± 0,0909 | [+0,0426; +0,2682] | 5/5 |
| FAM Stage B fusion − VIS | Carnation | 0,4316 | 0,4336 | +0,0019 ± 0,0402 | [−0,0480; +0,0518] | 3/5 |
| FAM Stage B fusion − VIS | FHL | 0,4064 | 0,3808 | −0,0256 ± 0,0170 | [−0,0467; −0,0044] | 1/5 |
| RCRA Stage B − FAM Stage B | Carnation | 0,4336 | 0,4151 | −0,0184 ± 0,1149 | [−0,1611; +0,1242] | 2/5 |
| RCRA Stage B − FAM Stage B | FHL | 0,3808 | 0,4174 | +0,0366 ± 0,1201 | [−0,1126; +0,1857] | 3/5 |

I delta mAP@50 per seed 40--44, conservati integralmente anche nel CSV, sono:

| Contrasto | Acquisizione | 40 | 41 | 42 | 43 | 44 |
|---|---|---:|---:|---:|---:|---:|
| FAM storico − Additive storico | Carnation | +0,0927 | +0,1188 | +0,1735 | +0,0092 | +0,0857 |
| FAM storico − Additive storico | FHL | +0,2466 | +0,0526 | +0,1196 | +0,1026 | +0,2557 |
| FAM Stage B fusion − VIS | Carnation | +0,0179 | +0,0561 | −0,0157 | −0,0526 | +0,0039 |
| FAM Stage B fusion − VIS | FHL | +0,0018 | −0,0389 | −0,0379 | −0,0328 | −0,0202 |
| RCRA Stage B − FAM Stage B | Carnation | −0,0789 | +0,0766 | −0,0298 | −0,1710 | +0,1108 |
| RCRA Stage B − FAM Stage B | FHL | +0,0057 | −0,0452 | +0,1647 | −0,1007 | +0,1584 |

La macro-media descrittiva a peso uguale tra acquisizioni conferma lo stesso
quadro:

| Contrasto | Delta macro medio ± SD | IC95% | Vittorie seed |
|---|---:|---:|---:|
| FAM storico − Additive storico | +0,1257 ± 0,0521 | [+0,0610; +0,1904] | 5/5 |
| FAM Stage B fusion − VIS | −0,0118 ± 0,0228 | [−0,0401; +0,0164] | 2/5 |
| RCRA Stage B − FAM Stage B | +0,0091 ± 0,1028 | [−0,1186; +0,1368] | 3/5 |

I test appaiati sono soltanto esplorativi con `n=5`. Per FAM contro Additive,
il test t fornisce `p=0,0227` su Carnation e `p=0,0187` su FHL, mentre il test
esatto di Wilcoxon a due code raggiunge in entrambi i casi soltanto il minimo
possibile `p=0,0625` con cinque differenze tutte positive. Per FAM fusion contro
VIS, i valori t sono `p=0,9203` su Carnation e `p=0,0283` su FHL; i rispettivi
Wilcoxon sono `p=0,8125` e `p=0,1250`. Per RCRA contro FAM nessun test è vicino
alla soglia convenzionale (`p_t=0,7378` su Carnation e `p_t=0,5333` su FHL).
Questi test non trasformano le cinque inizializzazioni in evidenza a livello di
frame e non sostituiscono la lettura dei segni e delle ampiezze dei delta.

## Interpretazione

La conferma centrale riesce: il vantaggio della FAM storica rispetto alla
fusione Additive persiste su entrambe le acquisizioni precedentemente
inutilizzate, in tutti i seed, con un delta macro medio di +0,1257 mAP@50. Il
risultato rafforza l'evidenza sulla scelta architetturale senza nuovo training e
senza selezionare checkpoint dopo avere visto le sequenze.

Questo risultato non implica però che l'IR aiuti sempre. Nel confronto
controllato sullo stesso FAM Stage B, azzerare il canale IR lascia Carnation
sostanzialmente invariata e migliora in media FHL di 0,0256 mAP@50. FAM può
quindi essere un meccanismo di fusione migliore di Additive pur non garantendo
che ogni checkpoint sfrutti utilmente la termica su ogni nuova acquisizione.
La distinzione tra confronto architetturale storico e ablation di modalità
Stage B deve rimanere esplicita.

RCRA alterna miglioramenti e perdite molto grandi tra seed. La macro-media è
vicina a zero e il suo intervallo è ampio; la conferma non giustifica una
promozione retroattiva. Resta valida la decisione Stage B già chiusa:
`fail_retain_fam`.

Le due sequenze rimangono confirmation set interni, non un holdout esterno. La
generalizzazione fuori da WiSARD e alla popolazione SAR reale non è dimostrata.

## Artefatti prodotti

- risultati completi locali:
  `out/rtdetr_unused_acquisition_confirmation/`;
- aggregato JSON locale, SHA-256
  `8f7bee0e9597897ebf3ed339a98e2627c10ff7350f8c31b68b576460c020677c`;
- insieme ordinato dei 50 file grezzi, hash composito SHA-256
  `3cff6bf475ab304d012fed4001d6084b6206ed972e4251905d2d07779ec175cf`;
- [tabella compatta versionata](Search_and_Rescue/results/rtdetr_unused_acquisition_confirmation.csv),
  identica byte per byte al CSV prodotto dal runner, SHA-256
  `a314bbf1d5eb7ffce296945a4f892231c2cbb713ba1436295c9941a73b4233c8`;
- integrazione nei capitoli di metodologia, valutazione e discussione da
  effettuare insieme al successivo stress test geometrico, evitando di
  frammentare la revisione della tesi.
