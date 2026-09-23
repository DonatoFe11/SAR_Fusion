# Audit sperimentale per la tesi

Ultimo aggiornamento: 31 agosto 2026.

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
| RT-DETR FAM box-guided P3 | matched control e candidata seed 40 completi; audit su 3.123 frame train e controfattuale su 896 frame validation | studio negativo informativo: la geometria viene appresa, ma il contributo diretto alla detection è trascurabile | chiusa: delta `+0,00810 < +0,01`; niente seed 41--44 o Stage B |
| Error analysis e figure Additive--FAM | dieci checkpoint, cinque soglie e sei frame GT-only mostrati a due soglie | evidenza finale sui failure mode di detection; sostituisce il confronto storico Lazy--SSJ | completa |
| Stress test Carnation Additive--FAM | dieci checkpoint, tre modalità e 739 frame comuni; protocollo congelato prima dell'inferenza | test esterno mirato al forte mismatch di scala, non stima rappresentativa di generalizzazione | completo; FAM attenua il danno in fusione ma VIS-only resta superiore a VIS+IR in 5/5 checkpoint FAM |
| Conferma WiSARD su Carnation 0025/0026 e FHL 0407/0408 | quattro configurazioni, cinque seed, 50 valutazioni; inventari, attestazione e checkpoint congelati prima dell'inferenza | conferma su acquisizioni interne precedentemente inutilizzate, non holdout esterno | completa; FAM storico supera Additive in 10/10 confronti, l'IR Stage B non aiuta uniformemente e RCRA resta instabile |
| Costo RT-DETR Additive--FAM | parametri, proxy GFLOPs, tre trial di latenza e memoria in processi isolati | quantifica il trade-off del modello principale sul solo forward del detector | completo; FAM migliora la detection ma aggiunge 77,5% parametri e 17,9% latenza sulla GPU misurata |
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

### 4. Valutazione di stress Carnation — completata

Il protocollo `rtdetr_carnation_stress_test_v1` è stato congelato e versionato
prima di osservare metriche. Le 30 valutazioni Additive/FAM sui cinque seed e
nelle tre modalità sono complete sugli stessi 739 identificatori di frame.

In VIS+IR, FAM ottiene `0.2408 ± 0.0380` mAP@50 contro
`0.1445 ± 0.0709` di Additive: delta appaiato medio `+0.0962`, quattro vittorie
su cinque. Il vantaggio non basta però a rendere la fusione preferibile al
solo VIS sotto il forte mismatch di scala. Nei checkpoint FAM, VIS-only ottiene
`0.3196 ± 0.0518` e supera VIS+IR in 5/5 seed, con delta fusion meno VIS
`−0.0789`. FAM peggiora inoltre IR-only in 4/5 seed (`0.0979` contro `0.1288`).

Carnation supporta quindi una conclusione circoscritta: FAM attenua la
fragilità dell'Additive fusion, ma non corregge abbastanza una differenza di
scala estrema da rendere sempre utile la seconda modalità. Non si avviano
nuove varianti o tuning sullo stress set. Configurazione, risultati e cautele
sono in [`rtdetr_carnation_stress_test.md`](rtdetr_carnation_stress_test.md).

### 5. Costo computazionale RT-DETR — completato

Il protocollo `rtdetr_additive_fam_compute_benchmark_v1` è stato congelato e
pushato prima delle misure. Su RTX 4070 Laptop GPU, batch 1 e input FP32
`[1, 4, 640, 640]`, FAM passa da 66,20 a 117,49 milioni di parametri e da
208,37 a 304,54 GFLOPs nella proxy che corregge il mancato conteggio DCNv2 del
profiler. La latenza del solo detector passa da `66,91 ± 0,30` a
`78,87 ± 0,37` ms sui tre trial (`+17,9%`); il picco CUDA passa da 436,26 a
712,18 MiB.

Ogni trial/configurazione è stato eseguito in un processo isolato. Un primo
output in-process, nel quale l'allocazione cresceva cumulativamente fra modelli,
è stato rigettato come artefatto dell'implementazione e non usato. Il conteggio
FLOPs resta una proxy convenzionale e la latenza esclude preprocessing e
postprocessing. Risultati e limiti sono in
[`rtdetr_compute_benchmark.md`](rtdetr_compute_benchmark.md).

### 6. Conferma su acquisizioni WiSARD inutilizzate — completata

Il protocollo `rtdetr_unused_acquisition_confirmation_v1` è stato congelato il
30 agosto 2026 prima dell'inferenza. L'autore ha attestato di non avere mai
visionato manualmente Carnation 0025/0026 e FHL 0407/0408; l'audit non ne ha
rilevato l'uso nei parametri versionati o nelle run conservate. Sono state
incluse 1.313 coppie Carnation e 1.035 coppie FHL con ground truth VIS. L'audit
del pairing ha escluso quattro ID asimmetrici Carnation che avrebbero prodotto
661 associazioni posizionali errate.

Le 50 valutazioni congelate sono complete. FAM storico supera Additive in
5/5 seed su entrambe le acquisizioni: delta mAP@50 medio `+0.0960` su
Carnation e `+0.1554` su FHL. La macro-media descrittiva a peso uguale è
`+0.1257`, ancora positiva in 5/5 seed.

La diagnostica Stage B limita l'interpretazione: FAM fusion meno VIS-only è
`+0.0019` su Carnation ma `−0.0256` su FHL; RCRA meno FAM è `−0.0184` e
`+0.0366`, con deviazioni standard rispettivamente `0.1149` e `0.1201` e
frequenti inversioni fra seed. La conferma rafforza quindi FAM rispetto ad
Additive ma non dimostra che l'IR aiuti sempre né riapre la decisione RCRA.
Risultati, test esplorativi, hash e limiti sono in
[`rtdetr_unused_acquisition_confirmation.md`](rtdetr_unused_acquisition_confirmation.md).

### 7. Stress geometrico sintetico controllato — completato

Lo stress Carnation 0023/0024 già chiuso misura un mismatch di scala nativo e
non sostituisce una perturbazione sintetica controllata. Il passo successivo è
un protocollo separato che trasformi soltanto il canale IR, lasciando RGB e
ground truth VIS invariati, e riporti il calo rispetto all'identità anziché
cercare la trasformazione migliore. Traslazioni, scale, direzioni, gestione dei
bordi, famiglie di checkpoint e regole di aggregazione sono stati fissati nel
protocollo `rtdetr_synthetic_geometric_stress_v1`. Il runner, quattro test
automatici, il preflight completo e uno smoke GPU su un solo batch in `/tmp`
sono stati completati prima della campagna. L'esecuzione resumable ha prodotto
560/560 inferenze perturbate e 600/600 punti includendo le identità congelate.

La metrica con segno non conferma una superiore tolleranza generale di FAM.
Su Carnation i contrasti FAM--Additive per le traslazioni restano piccoli e
incerti; su FHL il contrasto favorisce Additive a 8, 16 e 32 pixel in 0/5 seed
per FAM, ma perché Additive **migliora** rispetto alla propria identità fino al
6,93%, non perché FAM subisca un crollo. Le perturbazioni stanno quindi
probabilmente compensando parte di un mismatch nativo o modificando
favorevolmente le feature. RCRA non mostra un vantaggio stabile su FAM: tutti
gli IC95% dei contrasti aggregati includono zero. Il risultato non modifica la
scelta del FAM standard e vieta di attribuirne il vantaggio di accuratezza a
una robustezza geometrica universale. CSV, figura, hash e limiti sono in
[`rtdetr_synthetic_geometric_stress.md`](rtdetr_synthetic_geometric_stress.md).

### 8. FAM con Modal Dropout misto e consistency training — chiuso

Dopo la chiusura delle valutazioni naturali e sintetiche, l'unico nuovo
candidato autorizzato è una modifica del training FAM. Il percorso
supervisionato conserva esattamente il Modal Dropout storico, inclusi esempi IR
nativi con ground truth IR; un percorso aggiuntivo abbina mediante Hungarian
matching un teacher fusion pulito senza gradiente e uno student paired-VIS con
RGB oppure IR mascherata.

Il seed 40 `hghkalag` ha completato dieci epoch e 7.810 step senza errori. La
prima epoca è durata `11:15`, mentre le successive circa 34--40 minuti: il
salto è previsto, perché la prima salta interamente la consistency e dalla
seconda ogni batch aggiunge teacher e student. Il `best` è rimasto all'epoch 1
con fusion mAP@50 `0,125263`; dopo l'attivazione della loss la validation è
scesa a `0,065633` e poi `0,036052` nei due epoch successivi.

Il gate congelato sulle 896 coppie FHL è fallito in tutti i requisiti: delta
fusion `-0,026884` contro limite `-0,01`, delta paired masked-IR `+0,001829`
contro minimo `+0,03` e delta IR nativa `-0,050581` contro limite `-0,03`.
Il candidato è quindi chiuso fail-closed dopo un seed; seed 41--44, Stage B e
MtErie non vengono eseguiti. Il FAM standard resta il modello principale.
Dettagli, tabella e limiti interpretativi sono in
[`rtdetr_fam_mixed_consistency_stage_a.md`](rtdetr_fam_mixed_consistency_stage_a.md).

### 9. FAM Box-Guided Common-Offset P3 — screen seed 40 chiuso

La candidata successiva aggiunge soltanto a P3 un campo comune `(dy, dx)`
supervisionato debolmente dai centri dei box VIS/IR mutual-nearest. Il campo è
sommato ai nove offset residui del FAM storico; P4 e P5 restano
`current_dcnv2`. Il ramo nuovo usa una proiezione condivisa a 32 canali e
aggiunge esattamente **53.410 parametri**, senza modificare decoder e teste. La
loss usa Smooth L1 sui match conservativi, con soglia normalizzata `0,05`,
`lambda=0,2`, warm-up di due epoche e limite `±4` celle applicato soltanto alla
guida comune.

L'inventario Stage A è stato ricostruito indipendentemente sui 3.123 frame di
train. Contiene 5.209 match distribuiti su 2.306 frame; 817 frame non hanno
target. La distribuzione per numero di match è `0:817`, `1:801`, `2:543`,
`3:526`, `4:436`. Ordinando frame e righe in modo canonico e serializzando i
tensori `[x_VIS, y_VIS, dy, dx]` come `float32` little-endian, lo SHA-256 è:

```text
d519574962e81ae5b492248113247cca20d7ef15b2d189d1e3b58aebf218f3c0
```

È stato corretto prima del training un confondente RNG: l'allocazione del ramo
nuovo consumava estrazioni casuali e avrebbe cambiato l'inizializzazione dei
FAM condivisi successivi. Il costruttore usa ora un RNG fork locale e impedisce
al `post_init` Hugging Face di reinizializzare il subtree nuovo. Il test di
regressione conferma, per lo stesso seed, pesi di tutti i FAM condivisi
bit-identici e stato RNG globale bit-identico fra baseline e candidata.

Le quattro configurazioni scientifiche della campagna dichiarano inoltre il
manifest sorgente `rtdetr_box_guided_training_source_v1`, composto da 22 file
critici, con SHA-256 aggregato:

```text
b06ea1328be206a9f7c64b3412f64ed7bb95b884da591c476584a90403592412
```

Il training è fail-closed: prima della run ricalcola il manifest, richiede una
trace attiva e scrivibile e registra hash aggregato e per-file nel primo evento.
Audit meccanicistico e controfattuale legano poi checkpoint, configurazione
W&B/YAML, trace originale e sorgenti correnti. Il probe `j37qaj8r`, precedente
a questo vincolo, resta soltanto ingegneristico. Il manifest non sostituisce il
fingerprint delle dipendenze o gli inventari del dataset.

Il probe tecnico, escluso da ogni confronto scientifico, è completato nella
run `j37qaj8r`, directory
[`wandb/run-20260831_122623-j37qaj8r`](../wandb/run-20260831_122623-j37qaj8r/).
Ha eseguito 20 step e la validation completa di 896 frame in 132 secondi, con
loss media `17,44285` e valori finiti. All'ultimo step la loss guida raw è
`1,18184`, quella pesata `0,11818` e la scala di warm-up `0,5`. La validation
mAP@50 `0,07828` è una diagnostica non scientifica e non entra nei gate.

Il matched control `2fx2ozwm` completa dieci epoche/7.810 step e ottiene best
mAP@50 `0,147388741` all'epoch 3. La candidata `2jvqs9mr` completa lo stesso
protocollo e ottiene `0,155485332` all'epoch 1. Il delta appaiato
`+0,008096591` è positivo, ma manca di `0,001903409` il gate preregistrato
`+0,01`; il vecchio FAM seed 40 `0,152148` resta soltanto descrittivo.

L'audit meccanicistico sui 3.123 frame/5.209 match passa tutti i gate. La
Smooth L1 della guida è `0,437972`, contro `1,006123` di zero e `0,884058` del
miglior vettore costante; i miglioramenti relativi sono `56,47%` e `50,46%`.
La correlazione centrata guida--target è `0,786072`, la saturazione è nulla e
la quota di vettori cancellati almeno a metà dagli offset residui è soltanto
`0,006911`.

Il controfattuale sugli stessi 896 frame FHL riproduce esattamente la best W&B
e misura mAP@50 `0,155485332` con guida attiva contro `0,155478507` con guida
azzerata: delta `+0,000006825`. Il vincolo non negativo passa formalmente, ma
l'effetto diretto è trascurabile; AP50:95 e AP75 risultano persino lievemente
inferiori con la guida attiva. Il risultato indica quindi che la regressione
geometrica è apprendibile, ma non fornisce un vantaggio di detection sufficiente.

La candidata è chiusa fail-closed: seed 41--44, authorization manifest, audit
aggregato e Stage B non vengono eseguiti. Protocollo, artefatti, hash e il fix
non scientifico del controllo runtime dell'inventario sono in
[`rtdetr_fam_box_guided_stage_a.md`](rtdetr_fam_box_guided_stage_a.md).

Come controllo interpretativo non selettivo, l'audit confronta anche `g` col
miglior vettore costante `(dy, dx)` che minimizza esattamente la Smooth L1
`beta=0,25` sui medesimi target train congelati. Il confronto non modifica i
gate: se la candidata vince in mAP ma la guida non migliora strettamente questa
costante, il modello resta promuovibile secondo il protocollo prestazionale ma
il meccanismo va descritto come calibrazione globale, non come allineamento
input-conditioned. Se la batte, supporta soltanto un fit non globale sui target
train, non esclude una mappa spaziale o una memoria dell'acquisizione e non
prova generalizzazione. Zero e costante misurano fit sul train; il
controfattuale validation misura invece l'effetto prestazionale fuori dal
train della loss.

Era stato congelato anche l'audit Stage A v2 per l'eventuale espansione: combina
seed 40 dai progetti dello screen e seed 41--44 dai progetti condizionali,
verifica le quattro sorgenti YAML, indici del grid, metadati di lancio,
manifest/trace e ambiente comune delle dieci run, richiede i due risultati seed
40 già passati e applica automaticamente delta seed 40, delta medio, 4/5
vittorie e meccanismo 5/5. Il `best_map_50` W&B resta la metrica primaria del
protocollo storico; soltanto la candidata seed 40 dispone già del replay
validation indipendente. Resta come traccia del protocollo condizionale, ma non
viene eseguito perché lo screen seed 40 non ha autorizzato l'espansione.

### 10. Aggiornamento della tesi

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

I punti 1--17 ricostruiscono il piano precedente: il loro vincolo era chiudere
la campagna allora attiva prima di aprire nuove architetture, ed è stato
soddisfatto. I punti 18 e successivi costituiscono l'estensione ora autorizzata.

1. Chiudere la campagna allora attiva prima di avviare nuove architetture.
   **Fatto.**
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
9. Eseguire il protocollo Carnation già congelato sui 10 checkpoint finali.
   **Fatto:** 30/30 valutazioni su 739 frame; FAM migliora Additive in fusion
   ma non supera VIS-only sotto il mismatch di scala. Lo stress set è chiuso
   senza tuning successivo.
10. Eseguire il benchmark computazionale Additive--FAM già congelato.
    **Fatto:** tre trial e 300 forward per configurazione in processi isolati;
    risultato completo e riepilogo versionato.
11. Confermare Additive/FAM e diagnosticare fusion/RCRA sulle due acquisizioni
    WiSARD inutilizzate. **Fatto:** 50/50 unità, FAM storico favorevole in
    10/10 confronti acquisizione/seed; fusion Stage B e RCRA non uniformi.
12. Congelare ed eseguire lo stress geometrico sintetico, senza usare i
    risultati della conferma per scegliere le perturbazioni. **Fatto:**
    protocollo congelato, preflight e smoke completati, 560/560 inferenze e
    600/600 punti curva; nessuna trasformazione selezionata post-hoc.
13. Consolidare risultati e figure nei Markdown. **Fatto:** CSV e curva dello
    stress versionati; risultati, hash e limiti interpretativi documentati.
14. Implementare e congelare il candidato FAM con Modal Dropout misto e
    consistency training. **Fatto:** configurazioni seed 40/cinque seed, loss
    matched, suite completa di 206 test, probe GPU di due batch senza OOM o NaN
    e valutatore automatico seed 40 con inventario FHL verificato.
15. Eseguire soltanto il seed 40 Stage A e applicare i tre gate congelati.
    **Fatto:** run completa; tutti i gate falliti, candidato chiuso.
16. Se e solo se lo screen passa, eseguire i seed 41--44 senza ripetere il 40
    e applicare la regola Stage A a cinque seed. **Non applicabile:** lo screen
    è fallito; l'espansione è vietata dal protocollo.
17. Preparare ed eseguire Stage B soltanto dopo una promozione Stage A. **Non
    applicabile:** nessuna promozione Stage A.
18. Implementare e congelare il FAM Box-Guided Common-Offset P3. **Fatto:**
    architettura, pseudo-target, loss, fix RNG, test, inventario canonico e
    probe tecnico `j37qaj8r` completati; manifest fail-closed di 22 sorgenti
    applicato ai quattro YAML scientifici.
19. Eseguire per primo il nuovo FAM matched control seed 40. **Fatto:** run
    `2fx2ozwm`, best epoch 3, mAP@50 `0,147388741`.
20. Solo dopo il controllo, eseguire la candidata box-guided seed 40 con la
    configurazione già congelata. **Fatto:** run `2jvqs9mr`, best epoch 1,
    mAP@50 `0,155485332`; delta `+0,008096591`, inferiore al gate `+0,01`.
21. Dopo il training candidato, eseguire l'audit meccanicistico sul checkpoint
    `best` e applicare il gate candidata meno matched control `>= +0,01` insieme
    ai gate del campo guida e della cancellazione residua; riportare anche il
    confronto descrittivo col miglior predittore costante. **Fatto:** tutti i
    gate meccanicistici passano; Smooth L1 `0,437972`, miglioramento `56,47%`
    contro zero e `50,46%` contro la migliore costante. Il gate prestazionale
    fra run fallisce.
22. Eseguire il controfattuale validation active-vs-zero sullo stesso
    checkpoint, con delta mAP@50 non negativo. **Fatto:** active
    `0,155485332`, zero `0,155478507`, delta `+0,000006825`; formalmente passa,
    ma l'effetto diretto è trascurabile.
23. Congelare un protocollo meccanicistico multi-run che unisca il seed 40 del
    progetto screen ai quattro seed del progetto di espansione. **Fatto:** il
    v2 lega dieci run, prerequisiti seed 40, ambiente, checkpoint e gate Stage A.
24. Se tutti i gate seed 40 passano, scrivere prima del lancio un authorization
    manifest con gli hash dei due JSON seed 40; solo allora eseguire i seed
    41--44 insieme ai
    matched control FAM freschi 41--44 già congelati; non usare le run storiche
    come riferimento primario e applicare l'audit aggregato a tutti e cinque i
    checkpoint. **Non applicabile:** gate primario seed 40 fallito; espansione
    vietata.
25. Preparare Stage B soltanto dopo una promozione Stage A, congelando e
    addestrando anche un nuovo FAM matched control full-data con stesso codice,
    manifest e ambiente. **Non applicabile:** nessuna promozione Stage A.
26. Se la guida non apprende la geometria, passare a un candidato P3 distinto
    con proiezione condivisa 32 canali, normalizzazione L2, cost volume `9×9`
    a raggio quattro e regressione/soft-argmax, lasciando libero il residuo FAM.
    Se invece la geometria è appresa ma la mAP non migliora, chiudere questa
    classe di interventi e valutare RT-DETRv2 + FAM; D-FINE con FDR/GO-LSD resta
    la terza scelta detector-level. **Decisione applicata:** la guida apprende
    la geometria ma non migliora abbastanza la detection; il cost-volume non è
    giustificato e la prossima linea da valutare è RT-DETRv2 + FAM.
27. Aggiornare `notes/Search_and_Rescue/main.tex` dopo la chiusura della nuova
    campagna oppure dopo il suo arresto fail-closed. **Da fare:** RT-DETRv2 e
    YOLO26 sono ora entrambi chiusi, quindi non restano altri training necessari
    prima del consolidamento della tesi.
28. Integrare e valutare RT-DETRv2 + FAM sul solo seed 40 Stage A. **Fatto e
    chiuso:** Additive `0,181978` e FAM `0,158466` best validation mAP@50; il
    delta primario non passa il gate e non autorizza altri seed o Stage B.
29. Integrare YOLO26s ufficiale con dual backbone e congelare Additive/FAM.
    **Fatto:** port 4-canali, loss E2E nativa, dataset appaiato, inizializzazione,
    trace e checkpoint auditati. Il pilot Additive è integro ma collassa dopo
    il best `0,06472` dell'epoca 2; FAM v1 non viene eseguito.
30. Applicare un solo repair post-pilot alla recipe YOLO26, cambiando
    esclusivamente `warmup_bias_lr` da `0,1` a zero. **Fatto e chiuso:** 50/50
    epoche, audit optimizer e replay passati; best complessivo `0,04353`, best
    eleggibile epoche 4--50 `0,01535` contro soglia `0,10`. Il gate di vitalità
    fallisce e vieta FAM, seed 41--44, Stage B e una terza taratura.

Nota operativa per gli eventuali fallback: l'ambiente locale `sarfusion` usa
Transformers `4.43.3` e non contiene classi o moduli RT-DETRv2/D-FINE. Un loro
screen richiederà un ambiente separato o un port esplicito, non una modifica
trasparente del costruttore attuale.

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
