# Audit sperimentale per la tesi

Ultimo aggiornamento: 8 agosto 2026.

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
| Diagnostica degli offset RT-DETR | FAM, SSJ e Grid Sample, cinque seed e 30 campioni per checkpoint; ablation fattoriale P3/P4/P5; variante bounded-offset su cinque seed | evidenza meccanicistica principale; identifica il failure mode P5 e verifica una correzione mirata | completa; il bounding elimina il collasso osservato ma non migliora la detection |
| Error analysis e figure Additive--FAM | dieci checkpoint, cinque soglie e sei frame GT-only mostrati a due soglie | evidenza finale sui failure mode di detection; sostituisce il confronto storico Lazy--SSJ | completa |
| Tiling, CMX e CMX ibrido | singole run | studi di fattibilità negativi della specifica implementazione | non ripetere; evitare spiegazioni causali non misurate |
| Deformable DETR | cinque run per variante, protocollo precedente | evidenza esplorativa di trasferibilità e instabilità | non ripetere salvo che si voglia sostenere una superiorità quantitativa cross-architettura |
| DINO completo | cinque run base, prestazioni molto basse | risultato negativo della configurazione valutata | non ripetere e non avviare FAM/SSJ senza una nuova ipotesi |
| YOLO storico: fusion-only, input dropout e feature gating | una run per configurazione, `best.pt`, early stopping variabile | sviluppo esplorativo | non usare per una conclusione multi-seed |
| YOLO finale Additive/FAM | 2 configurazioni x 5 seed, 200 epoche, `last.pt`; 30 valutazioni VIS+IR/VIS/IR | esito finale di trasferibilità e robustezza, compatibile con degradazione tardiva | completa; nessun retraining senza una nuova validation predefinita |

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

### 1. Diagnostica FAM sui checkpoint finali — completata

La replica ha analizzato FAM, FAM + SSJ e Grid Sample sui seed `40–44`, usando
gli stessi 10 campioni MtErie, 10 FHL e 10 Baker fissati prima dell'analisi.
Sono disponibili 1.350 osservazioni campione/livello e l'aggregazione rispetta
il checkpoint come unità sperimentale.

I risultati principali sono:

- il percorso geometrico del FAM è attivo a P3/P4 in tutti i seed;
- offset e attivazioni mostrano una variabilità sostanziale fra seed;
- il FAM seed 41 presenta una degenerazione completa di P5, con offset medi
  di circa 5.030 pixel e uscita costante nello spazio, nonostante ottenga la
  mAP@50 FAM più alta (`0.4335`);
- escludendo quel seed, FAM continua a superare Additive in 4/4 repliche con
  delta medio `+0.0799`, quindi il confronto prestazionale principale non è
  prodotto dall'anomalia;
- SSJ aumenta in modo ripetibile offset e smoothing a P3, ma gli effetti a P4
  sono variabili e il confronto P5 è contaminato dal collasso;
- le proxy sulla stessa cella non aumentano per DCNv2, che effettua anche
  filtraggio e mixing dei canali: l'evidenza supporta una trasformazione
  geometrico-funzionale utile al task, non una registrazione fisica verificata;
- Grid Sample è più stabile internamente e preserva maggiormente le feature,
  ma non supera FAM in modo consistente nella detection.

La figura P5, le 40 valutazioni fattoriali (otto condizioni per cinque seed) e
l'audit interno sono completati. Il FAM completo resta il migliore con mAP@50
media `0.3780`; P3+P5 è la rimozione più vicina (`0.3648`), mentre P3+P4 scende
a `0.3221`. Nel seed patologico togliere P5 riduce la mAP@50 da `0.4335` a
`0.2399`: il decoder si è adattato al ramo degenerato, quindi una rimozione
statica post-hoc non è una correzione.

L'audit mostra che nel seed 41 la scala è già patologica nelle feature P5
pre-FAM di entrambe le modalità (std media RGB `17.30`, IR `20.18`, contro
rispettivamente `0.17--0.69` e `0.056--0.101` negli altri seed). I pesi del
predittore hanno invece scala ordinaria. Gli offset medi raggiungono circa
`5057 px` d'immagine, sono quasi identici fra campioni (correlazione maggiore
di `0.999`) e portano la DCNv2 fuori mappa, riducendo l'uscita IR P5 al bias
appreso.

È stata quindi valutata una sola variante `bounded_dcnv2_4`, che applica
`4*tanh(raw/4)` agli offset in celle della feature map. Non introduce gate,
residui, normalizzazioni o modifiche al backbone. Il limite 4 è stato fissato
prima del training usando la distribuzione dei quattro checkpoint non
patologici; il protocollo è in
[`rtdetr_fam_bounded4_protocol.yaml`](../parameters/RTDETR/rtdetr_fam_bounded4_protocol.yaml).
Sui seed `40--44` ottiene `0.3496 ± 0.0685` mAP@50 VIS+IR, contro
`0.3780 ± 0.0439` del FAM corrente: delta appaiato medio `-0.0284`, due
vittorie su cinque. L'audit sugli stessi 30 campioni per checkpoint conferma
che tutti gli offset effettivi restano sotto quattro celle e che non ricompare
alcuna uscita P5 spazialmente costante. Il rimedio elimina dunque il failure
mode osservato nei cinque nuovi training, ma non migliora accuratezza o
dispersione; `current_dcnv2` resta il modello principale.

Poiché la variante nasce da un'analisi post-hoc di MtErie, il confronto sullo
stesso benchmark resta esplorativo; cinque seed misurano la robustezza ma non
sostituiscono un nuovo holdout indipendente. Non si avvia una grid di ulteriori
rimedi sullo stesso test. Documentazione e tabelle complete sono in
[`verifica_allineamento_FAM.md`](verifica_allineamento_FAM.md).

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

Il protocollo è stato ora congelato in
[`rtdetr_error_analysis.md`](rtdetr_error_analysis.md) e implementato in
[`run_rtdetr_error_analysis.py`](../scripts/run_rtdetr_error_analysis.py).
Usa la soglia primaria `0.01` già fissata per le valutazioni finali,
sensitivity analysis `0.05/0.10/0.25/0.50`, matching uno-a-uno a IoU `0.50` e
stratificazione COCO small/medium/large. Il manifest GT-only seleziona sei
frame prima dell'inferenza: un target small e un frame vuoto per ciascuna
sequenza MtErie. La campagna è completa su dieci checkpoint. FAM aumenta la
recall in 5/5 seed alle soglie `0.10`, `0.25` e `0.50`; a `0.25` il delta medio
è `+0.0990` e la quota di frame non vuoti con almeno un FN scende di `0.0824`.
Gli FP medi non migliorano in modo uniforme fra soglie e seed. Le dodici
figure predefinite sono state generate e ispezionate; i dettagli sono nella
nota dedicata.

La vecchia osservazione secondo cui il Lazy FAM non produceva predizioni su
circa il 38% delle immagini può restare nella storia del debugging, ma non deve
essere una conclusione operativa sul modello finale.

### 3. Replica essenziale di YOLO — completa, linea in standby

Il confronto finale è stato eseguito su cinque seed appaiati per
configurazione. Tutte le run hanno raggiunto 200 epoche e il test VIS+IR del
`last.pt`; Additive ottiene `0.2485 ± 0.0256` mAP@50 e FAM
`0.2197 ± 0.0443`. FAM − Additive vale `−0.0288 ± 0.0528` e FAM vince in
2/5 seed. Le valutazioni standalone finali confermano il risultato VIS+IR
(`0.2498` contro `0.2213`), non mostrano un vantaggio VIS (`0.1964` contro
`0.1929`) e mostrano un piccolo incremento IR in 5/5 seed (`0.0277` contro
`0.0369`). Quest'ultimo resta troppo basso per essere operativo e, poiché il
FAM è bypassato in IR-only, descrive un effetto del training sui pesi appresi.

Il protocollo eseguito contiene:

1. YOLOv10 dual-backbone senza FAM, feature gating 20/20/60;
2. YOLOv10 dual-backbone con FAM standard, feature gating 20/20/60.

Per entrambe sono stati usati:

- seed appaiati `40–44`;
- un nuovo processo per seed;
- 200 epoche fisse, perché è l'orizzonte e la schedulazione già dichiarati
  negli YAML storici;
- validation disabilitata ai fini di early stopping e selezione;
- nessun early stopping;
- test del solo `last.pt`;
- valutazione VIS, IR e VIS+IR con lo stesso checkpoint;
- statistiche e confronti appaiati analoghi a RT-DETR.

Il confronto storico a seed 42 aveva selezionato checkpoint molto precedenti
(epoca 16 per Additive e 89 per FAM) e valori test superiori di circa 0.10. Le
loss di training continuano a scendere fino a 200; per Additive la traiettoria
vecchia e nuova coincide esattamente fino all'arresto storico. Il risultato è
compatibile con degradazione tardiva della generalizzazione, ma FHL non offre
una curva sufficientemente stabile per scegliere un nuovo checkpoint.

YOLO resta quindi in standby. Non si fissa retroattivamente un orizzonte di 100
epoche dopo avere visto MtErie. Un eventuale nuovo protocollo richiede prima
una validation rappresentativa costruita dal solo training e separata per
sessione o blocchi temporali. FAM + SSJ non è giustificato dal risultato
RT-DETR e non va aggiunto come nuova grid. I dettagli sono in
[`yolov10_fam_integrazione.md`](yolov10_fam_integrazione.md).

### 4. Valutazione di stress Carnation — protocollo congelato, inferenza da eseguire

Il protocollo `rtdetr_carnation_stress_test_v1` è stato congelato prima di
osservare metriche. Valuta una sola volta Additive e FAM finali sulla sequenza
Carnation, separando VIS, IR e VIS+IR e usando gli stessi 739 identificatori di
frame nelle tre modalità. Sono vietati tuning, selezione di seed, scelta di
checkpoint e nuove varianti basate sul risultato. Configurazione, inventario,
vincoli e comandi sono in
[`rtdetr_carnation_stress_test.md`](rtdetr_carnation_stress_test.md).

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
su YOLO valutata dal confronto essenziale, altre architetture come studi
esplorativi.

## Ordine operativo

1. Non avviare nuove architetture.
2. Estendere `fam_alignment_check.py` per JSON, repliche e Grid Sample. **Fatto:**
   runner `scripts/run_rtdetr_fam_diagnostics.py`, test automatici e output
   aggregato per checkpoint/seed.
3. Eseguire e documentare la diagnostica sui 15 checkpoint RT-DETR. **Fatto:**
   30 JSON, 1.350 righe e aggregati multi-seed verificati.
4. Generare la figura P5 seed 41, verificare parametri/attivazioni del ramo
   patologico, confrontare direttamente i tensori di offset fra campioni e
   implementare sui cinque checkpoint l'ablation in inferenza fattoriale delle
   otto combinazioni FAM attivo/disattivo a P3, P4 e P5. **Fatto.**
5. Bloccare dai controlli una sola variante correttiva. **Fatto:**
   `bounded_dcnv2_4`, limite fissato a quattro celle, 23 test mirati e smoke
   test superati. **Training completato:** cinque `latest`, zero crash, mAP@50
   VIS+IR `0.3495 ± 0.0686`, delta appaiato `-0.0285` rispetto al FAM corrente
   e 2/5 vittorie. **Valutazioni completate:** 15/15 combinazioni; VIS
   `0.2252 ± 0.0544`, IR `0.1943 ± 0.0272`, VIS+IR `0.3496 ± 0.0685`; il
   VIS+IR ripetuto coincide con il test automatico entro `0.00038`.
   **Diagnostica completata:** offset effettivi inferiori a quattro celle in
   ogni livello e seed, nessun collasso P5 nei 150 casi osservati. La variante
   corregge il failure mode ma non migliora la detection; non viene promossa a
   modello finale.
6. Eseguire l'error analysis Additive/FAM e produrre le figure predefinite.
   **Fatto:** dieci checkpoint, cinque soglie, stratificazione per dimensione,
   35.400 righe e dodici figure GT-only predefinite. CSV, grafici aggregati e
   due tavole contenenti tutti gli esempi sono stati copiati negli artefatti
   versionati della tesi.
7. Preparare e verificare con smoke test i due YAML YOLO finali. **Fatto.**
8. Eseguire le 10 run YOLO. **Fatto:** 200 righe e `last.pt` per tutti i seed;
   test VIS+IR automatici e 30 valutazioni standalone VIS+IR/VIS/IR
   completati. Il sanity check fra evaluator ha uno scarto massimo `0.002077`,
   marginalmente oltre la tolleranza `0.002`, ed è documentato senza cambiare
   la soglia. L'eventuale ridisegno del protocollo di checkpoint resta separato
   e non autorizza una selezione post-hoc a 100 epoche.
9. Eseguire il protocollo Carnation già congelato sui 10 checkpoint finali,
   per 30 valutazioni complessive VIS+IR/VIS/IR, e documentarne l'esito senza
   avviare tuning successivo sullo stress set.
10. Consolidare risultati e figure nei Markdown.
11. Aggiornare `notes/Search_and_Rescue/main.tex`.

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
