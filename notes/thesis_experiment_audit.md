# Audit sperimentale per la tesi

Ultimo aggiornamento: 7 agosto 2026.

## Decisione generale

I report `report_dl.tex` e `report_cv.tex` documentano correttamente la fase in
cui sono stati prodotti, ma non devono essere usati come se contenessero il
protocollo sperimentale finale della tesi. Molti risultati in quei report sono
singole run e alcune conclusioni sono state superate dalla successiva analisi di
riproducibilità.

La tesi deve separare esplicitamente due fasi:

1. **fase esplorativa**, nella quale singole run sono state usate per sviluppare
   Modal Dropout, RT-DETR fusion, FAM, SSJ e le estensioni ad altre
   architetture;
2. **fase a protocollo bloccato**, nella quale le ipotesi principali su RT-DETR
   sono state rivalutate con cinque seed appaiati, processi isolati, epoca
   finale fissata e statistiche aggregate.

Questa distinzione è un risultato metodologico della tesi, non un difetto da
nascondere. La formulazione consigliata non è "a differenza di lavori
precedenti", che potrebbe sembrare un confronto con la letteratura, ma:

> Nelle sperimentazioni preliminari del progetto i modelli erano confrontati
> mediante singole esecuzioni. La replica delle configurazioni ha evidenziato
> una variabilità run-to-run sufficientemente ampia da cambiare l'ordinamento
> dei metodi. È stato quindi definito, prima della campagna finale, un
> protocollo con seed appaiati, orizzonte di training e checkpoint fissati a
> priori e reporting della distribuzione dei risultati.

I file `.tex` non vanno ancora aggiornati con nuove tabelle definitive. Prima
si completano le attività obbligatorie elencate sotto; poi `main.tex` viene
riscritto usando i Markdown come fonte unica. I due report di corso possono
restare come documenti storici, eventualmente con una breve nota che rimandi
alla tesi finale.

## Che cosa ha mostrato l'indagine di riproducibilità

Questa parte merita una sezione autonoma nella tesi perché modifica
sostanzialmente l'interpretazione dei risultati preliminari.

- Il Modal Dropout veniva applicato per errore anche a validation e test. Il
  loader è stato corretto e sono stati aggiunti test di regressione: il dropout
  resta attivo solo nel training.
- La validation effettiva contiene una sola sessione FHL, 273 frame, 184 frame
  vuoti e 148 box. Gli oggetti hanno un'area mediana circa dodici volte più
  piccola rispetto al train. Le sue metriche quasi nulle non sono quindi un
  criterio rappresentativo per scegliere l'epoca sul benchmark MtErie.
- Il confronto tra checkpoint `best` e `latest` ha escluso che la forte
  variabilità fosse spiegata soltanto da una selezione errata del `best`.
- Il trasferimento della riga COCO `person` alle teste single-class riduce la
  loss iniziale e l'amplificazione delle perturbazioni, ma non elimina il
  non-determinismo CUDA.
- A parità di seed, input, ordine dei batch e decisioni di Modal Dropout, il
  percorso CUDA nativo diverge al primo backward. Il percorso deterministico
  sostitutivo di RT-DETR produce invece trace, metriche e hash del checkpoint
  identici su repliche complete.
- Il FAM `current_dcnv2` non può usare la modalità strict perché il backward di
  `torchvision.ops.DeformConv2d` non dispone di un'implementazione CUDA
  deterministica. Per il confronto finale è stato quindi scelto un protocollo
  statistico nativo, identico per tutte le varianti.
- L'esecuzione di ogni seed in un nuovo processo evita il rallentamento
  progressivo osservato nelle grid eseguite nello stesso processo.

I dettagli tecnici, gli identificativi delle run e i risultati completi sono in
[`rtdetr_reproducibility.md`](rtdetr_reproducibility.md).

## Stato delle evidenze presenti nel progetto

| Linea sperimentale | Evidenza disponibile | Uso corretto nella tesi | Azione |
|---|---|---|---|
| DETR + Modal Dropout | sweep preliminare a singola run | motivazione e scelta esplorativa del 20/20/60 | non ripetere tutta la grid; non chiamare 20/20/60 "ottimo" in senso statistico |
| RT-DETR Additive e FAM preliminari | singole run, `best`, protocollo precedente | storia dello sviluppo; non usare 0.357/0.396 come stime finali | sostituiti dalla campagna multi-seed |
| RT-DETR finale: Additive, FAM, IR Dropout, SSJ, Identity, Grid Sample | 6 configurazioni x 5 seed, checkpoint finale, 90 valutazioni di modalità | evidenza quantitativa principale | completa |
| Lazy FAM, Frozen FAM e Spatial Dropout | singole run; il Lazy contiene un bug | analisi storica che ha motivato i controlli successivi | non ripetere il bug né il Frozen Random |
| Ablation Identity DCNv2 e Grid Sample | cinque seed ciascuna nel protocollo finale | ablation quantitativa valida sul benchmark interno | completa |
| Diagnostica degli offset RT-DETR | un checkpoint FAM e uno SSJ del protocollo storico | solo risultato preliminare/meccanistico | ripetere sui checkpoint finali |
| Analisi qualitativa Lazy vs SSJ | due checkpoint storici scelti dopo il training | illustrazione storica, non evidenza finale | sostituire le figure principali con checkpoint finali |
| Tiling, CMX e CMX ibrido | singole run | studi di fattibilità negativi della specifica implementazione | non ripetere; evitare spiegazioni causali non misurate |
| Deformable DETR | cinque run per variante, protocollo precedente | evidenza esplorativa di trasferibilità e instabilità | non ripetere salvo che si voglia sostenere una superiorità quantitativa cross-architettura |
| DINO completo | cinque run base, prestazioni molto basse | risultato negativo della configurazione valutata | non ripetere e non avviare FAM/SSJ senza una nuova ipotesi |
| YOLO fusion-only, input dropout e feature gating | una run per configurazione, `best.pt`, early stopping variabile | sviluppo esplorativo | ripetere il confronto essenziale se YOLO resta nelle conclusioni |

## Evidenza finale già valida su RT-DETR

La campagna principale ha confrontato cinque seed appaiati, `40–44`, usando
CUDA nativa, testa `person` pretrained, Modal Dropout solo nel train, dieci
epoche fisse, processi isolati e checkpoint finale `latest`.

Le conclusioni supportate sono:

- FAM `current_dcnv2` migliora Additive di `+0.0700` mAP@50 medio e vince in
  tutti i cinque seed;
- IR Dropout ha la media grezza più alta, ma il guadagno appaiato su FAM è
  soltanto `+0.0091` e cambia segno tra seed;
- SSJ non migliora la media di FAM (`0.3749` contro `0.3780`), anche se mostra
  una dispersione descrittiva inferiore;
- Identity DCNv2 non risolve l'instabilità e non è la variante principale;
- Grid Sample è vicino a FAM, ma non lo supera in modo consistente;
- VIS+IR supera la migliore modalità singola in tutti i 30 checkpoint.

Il modello principale della tesi è pertanto **RT-DETR con FAM
`current_dcnv2`, senza IR Dropout e senza SSJ**. La scelta privilegia il
guadagno consistente rispetto ad Additive e non il massimo numerico osservato
in una media o in una singola run.

Questa campagna è confermativa rispetto al confronto interno pre-specificato,
ma non rende automaticamente il test MtErie un holdout incontaminato: lo stesso
test era già stato consultato durante lo sviluppo. Questa limitazione deve
essere dichiarata.

## Validation, checkpoint e riuso del test

### Validation

La validation non è "inutile" in assoluto: dimostra un forte domain shift e può
essere riportata come stress test fuori distribuzione. Non deve però scegliere
il checkpoint nel protocollo corrente. Il checkpoint finale a un numero di
epoche fissato prima del training evita sia la selezione rumorosa tramite FHL
sia la selezione indiretta sul test.

Una futura validation multi-sessione sarebbe necessaria per tuning ed early
stopping ordinari, ma cambiare ora soltanto la validation di una famiglia
renderebbe i confronti ancora meno omogenei. Non va costruita scegliendo le
sequenze in base ai risultati già osservati.

### Benchmark MtErie

Le tre sequenze MtErie sono state consultate ripetutamente per confrontare
architetture, epoche e regolarizzazioni. Nella tesi vanno chiamate **benchmark
di sviluppo comune** o **test interno**, non holdout cieco. Le metriche restano
utili per confronti appaiati bloccati, ma possono essere ottimistiche rispetto
alla generalizzazione a una nuova acquisizione.

### Sequenza Carnation non usata

Nel repository esiste una coppia etichettata esclusa da train, validation e
test:

- `210529_Carnation_Enterprise_VIS_0023`;
- `210529_Carnation_Enterprise_IR_0024`.

Contiene 739 coppie utilizzabili dal `zip` del loader; il ramo VIS contiene
1.942 box. La camera RGB è zoomata rispetto all'IR, motivo per cui la sequenza
non è rappresentativa dello stesso regime geometrico degli split correnti.
Non va usata come nuova validation né come sostituto silenzioso del test.

Può essere usata per una **singola valutazione opzionale di stress**, dichiarata
come tale, dei cinque checkpoint Additive e dei cinque FAM già congelati. Non è
un prerequisito per la tesi né un test rappresentativo della distribuzione
principale. Se viene eseguita, protocollo e configurazioni devono essere fissati
prima di calcolare le metriche e nessuna scelta successiva del modello può
basarsi sul suo esito. Un risultato negativo sarebbe informativo sulla
generalizzazione a una forte differenza di scala fra sensori, non una
confutazione del confronto MtErie.

## Esperimenti da completare prima di nuove architetture

### 1. Diagnostica FAM sui checkpoint finali

È necessaria perché le tabelle correnti sugli offset usano una sola run FAM e
una sola run SSJ del vecchio protocollo. Non serve alcun nuovo training.

Il protocollo da bloccare è:

- checkpoint finali FAM dei seed `40–44`;
- checkpoint finali FAM + SSJ dei seed `40–44`;
- checkpoint finali Grid Sample dei seed `40–44`, adattando la diagnostica ai
  due canali di offset e all'assenza di mask;
- gli stessi 10 campioni MtErie, 10 FHL e 10 Baker già documentati;
- esportazione JSON per campione, livello, seed e configurazione;
- aggregazione prima per checkpoint e poi tra i cinque seed: le migliaia di
  celle spaziali non devono essere trattate come repliche statistiche
  indipendenti;
- figure qualitative prodotte con seed 40 per FAM e SSJ, scelto prima
  dell'analisi perché il FAM seed 40 coincide con la mediana e lo SSJ seed 40 è
  vicino alla propria mediana;
- eventuali correlazioni tra offset e mAP descritte come esplorative, dato
  `n=5`.

La diagnostica può stabilire che il modulo è attivo e caratterizzarne il campo;
non può da sola dimostrare che ogni offset corrisponda al vero spostamento
fisico fra sensori.

### 2. Error analysis e figure finali RT-DETR

Il confronto qualitativo attuale fra Lazy FAM e SSJ non va ripetuto: il primo
modello contiene un bug e il secondo non è la configurazione finale. Per la
tesi va sostituito da Additive contro FAM standard.

Il protocollo raccomandato non richiede training:

- usare tutti i cinque seed per misure aggregate di recall, falsi positivi per
  immagine e frazione di immagini senza predizioni a una soglia fissata prima
  dell'analisi;
- se possibile, stratificare i risultati per dimensione del target e presenza
  o assenza di annotazioni, senza creare i gruppi dopo aver visto quale favorisce
  FAM;
- usare il seed 43 per le sole figure illustrative Additive/FAM: è la coppia
  il cui miglioramento FAM − Additive (`+0.0488`) coincide con il delta mediano
  dei cinque seed, quindi rappresenta l'effetto tipico senza scegliere la run
  dal risultato massimo;
- scegliere in anticipo indici distribuiti nel test, includendo casi con target
  piccoli e frame vuoti sulla base delle annotazioni, non sulla bontà delle
  predizioni;
- mostrare gli stessi frame, la stessa soglia e lo stesso limite di detection
  per entrambi i modelli;
- descrivere le figure come esempi e fondare le conclusioni sui risultati
  aggregati.

La vecchia osservazione secondo cui il Lazy FAM non produceva predizioni su
circa il 38% delle immagini può restare nella storia del debugging, ma non deve
essere una conclusione operativa sul modello finale.

### 3. Replica essenziale di YOLO

YOLO deve essere ritrainato **solo se deve comparire nelle conclusioni come
verifica della trasferibilità del FAM**. Le run esistenti non sono adatte a
questa conclusione finale perché usano un solo seed, selezionano `best.pt` su
una validation non rappresentativa e terminano a epoche diverse per early
stopping.

Il protocollo minimo raccomandato contiene soltanto:

1. YOLOv10 dual-backbone senza FAM, feature gating 20/20/60;
2. YOLOv10 dual-backbone con FAM standard, feature gating 20/20/60.

Per entrambe:

- seed appaiati `40–44`;
- un nuovo processo per seed;
- 200 epoche fisse, perché è l'orizzonte e la schedulazione già dichiarati
  negli YAML storici;
- validation disabilitata ai fini di early stopping e selezione;
- nessun early stopping;
- test del solo `last.pt`;
- valutazione VIS, IR e VIS+IR con lo stesso checkpoint;
- statistiche e confronti appaiati analoghi a RT-DETR.

FAM + SSJ non è obbligatorio nel blocco confermativo: RT-DETR non mostra un
guadagno medio di SSJ e la vecchia grid YOLO può restare come indagine
esplorativa. Va aggiunto come terza configurazione a cinque seed soltanto se la
tesi mantiene una domanda esplicita sulla trasferibilità di SSJ.

Non occorre ripetere le grid YOLO fusion-only e input-level dropout. Il feature
gating è l'implementazione coerente con una modalità realmente assente e
risponde alla domanda finale più rilevante.

### 4. Valutazione di stress Carnation, opzionale

Dopo aver bloccato codice e protocollo, valutare una sola volta Additive e FAM
finali sulla sequenza Carnation, separando VIS, IR e VIS+IR. Non fare tuning,
selezione di seed o scelta di checkpoint su questi risultati.

### 5. Aggiornamento della tesi

Solo dopo le attività precedenti:

- sostituire nel `main.tex` le tabelle RT-DETR a singola run con i risultati
  aggregati;
- spostare Lazy/Frozen/Spatial Dropout, CMX, tiling e gli attuali YOLO in una
  sezione di sviluppo esplorativo;
- sostituire la conclusione "SSJ è il migliore" con la scelta del FAM standard;
- inserire una sezione su variabilità, determinismo, validazione e protocollo;
- aggiornare abstract, riepilogo e sviluppi futuri;
- eliminare l'affermazione storica secondo cui RT-DETR nativo sarebbe
  pienamente deterministico;
- distinguere sempre risultati del benchmark interno da eventuale stress test
  Carnation.

## Esperimenti che non è necessario ripetere

- l'intera grid DETR per dimostrare nuovamente 20/20/60;
- il modello Lazy con bug;
- Frozen FAM casuale e Spatial Dropout, salvo una nuova ipotesi precisa;
- tiling, CMX puro e CMX ibrido;
- le campagne Deformable DETR e DINO, se restano risultati esplorativi e non
  vengono usate per dichiarare una classifica quantitativa definitiva;
- tutte le varianti YOLO storiche.

Se si volesse invece sostenere nella tesi che FAM è superiore in modo
statisticamente dimostrato **su ogni architettura**, allora Deformable DETR e
YOLO andrebbero entrambi riallenati col medesimo protocollo di checkpoint e
seed; i dati attuali non supportano una frase così forte. La scelta consigliata
è una tesi più focalizzata: dimostrazione principale su RT-DETR, trasferibilità
su YOLO verificata dal confronto essenziale, altre architetture come studi
esplorativi.

## Ordine operativo

1. Non avviare nuove architetture.
2. Estendere `fam_alignment_check.py` per JSON, repliche e Grid Sample. **Fatto:**
   runner `scripts/run_rtdetr_fam_diagnostics.py`, test automatici e output
   aggregato per checkpoint/seed.
3. Eseguire la diagnostica sui 15 checkpoint RT-DETR già disponibili con
   `python scripts/run_rtdetr_fam_diagnostics.py`.
4. Eseguire l'error analysis Additive/FAM e produrre le figure predefinite.
5. Preparare e verificare con smoke test i due YAML YOLO finali.
6. Eseguire le 10 run YOLO e le 30 valutazioni di modalità.
7. Se si decide di includere lo stress test opzionale, preparare lo split
   Carnation e valutare i 10 checkpoint RT-DETR Additive/FAM una sola volta.
8. Consolidare risultati e figure nei Markdown.
9. Aggiornare `notes/Search_and_Rescue/main.tex`.

Ogni deviazione da questo ordine o dal protocollo va annotata prima di vedere
le metriche interessate.

## Reporting statistico

Per ogni configurazione finale vanno mostrati:

- i cinque valori grezzi per seed;
- media, mediana, deviazione standard campionaria e min--max;
- intervallo di confidenza t al 95% della media, dichiarando che con `n=5` è
  molto incerto;
- differenze appaiate per seed rispetto al riferimento;
- numero di vittorie sui cinque seed;
- test t appaiato e Wilcoxon esatto solo come analisi esplorative, senza usare
  `p < 0.05` come unico criterio;
- mAP COCO, mAP@50, mAP@75 e mAR, mantenendo mAP@50 come metrica primaria già
  dichiarata;
- costo computazionale e tempo di inferenza se si usa il termine "real-time".

Non vanno riportati soltanto il seed migliore o la media senza dispersione. Le
unità sperimentali sono le run/checkpoint, non i frame, le box o le celle delle
feature map.
