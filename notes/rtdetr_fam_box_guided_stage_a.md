# RT-DETR + FAM: Box-Guided Common-Offset a P3

## Stato

**Screen scientifico seed 40 completato e candidato chiuso.** Il matched
control ottiene `0,147389` mAP@50 e la candidata `0,155485`: il delta
`+0,008097` è positivo ma inferiore al gate preregistrato `+0,01`. Il ramo
supera tutti i gate meccanicistici, mentre il controfattuale active-vs-zero
misura soltanto `+0,00000682` mAP@50, cioè un effetto diretto trascurabile.
Seed 41--44 e Stage B non sono autorizzati. La regressione del repository passa
**269/269 test**; `py_compile` e `git diff --check` sono puliti.

File principali:

- [probe tecnico](../parameters/RTDETR/rtdetr_fam_box_guided_runtime_probe.yaml);
- [FAM matched control seed 40](../parameters/RTDETR/rtdetr_fam_box_guided_matched_control_seed40.yaml);
- [screen scientifico seed 40](../parameters/RTDETR/rtdetr_fam_box_guided_sequence_validation_seed40.yaml);
- [Stage A a cinque seed, condizionale](../parameters/RTDETR/rtdetr_fam_box_guided_sequence_validation_five_seed.yaml);
- [FAM matched control freschi seed 41--44, condizionali](../parameters/RTDETR/rtdetr_fam_box_guided_matched_control_seeds41_44.yaml);
- [protocollo dell'audit meccanicistico](../parameters/RTDETR/rtdetr_fam_box_guided_mechanism_audit_seed40.yaml)
  e [runner](../scripts/run_rtdetr_fam_box_guided_mechanism_audit.py);
- [protocollo controfattuale validation active-vs-zero](../parameters/RTDETR/rtdetr_fam_box_guided_counterfactual_seed40.yaml)
  e [runner](../scripts/run_rtdetr_fam_box_guided_counterfactual.py);
- [risultato meccanicistico JSON](Search_and_Rescue/results/rtdetr_fam_box_guided_mechanism_audit_v1.json)
  e [CSV](Search_and_Rescue/results/rtdetr_fam_box_guided_mechanism_audit_v1.csv);
- [risultato controfattuale JSON](Search_and_Rescue/results/rtdetr_fam_box_guided_counterfactual_v1.json)
  e [CSV](Search_and_Rescue/results/rtdetr_fam_box_guided_counterfactual_v1.csv);
- [protocollo Stage A aggregato a cinque seed, condizionale](../parameters/RTDETR/rtdetr_fam_box_guided_stage_a_five_seed_audit_v2.yaml)
  e [runner](../scripts/run_rtdetr_fam_box_guided_stage_a_five_seed_audit.py).

## Perché questa candidata

Il mixed consistency training appena concluso ha mostrato che imporre alle
predizioni IR mascherate di imitare box nelle coordinate VIS è un obiettivo
troppo indiretto e conflittuale. Il risultato locale distingue però semantica e
geometria:

- il modello FAM Stage B ottiene circa `0,5618 ± 0,0498` mAP@50 su IR nativa
  con ground truth IR;
- lo stesso ingresso IR, adattato e valutato con ground truth VIS, ottiene
  soltanto circa `0,0215`;
- FAM migliora Additive soprattutto in recall e su target piccoli;
- il FAM corrente stima gli offset direttamente dalla concatenazione delle
  feature RGB e IR, senza una misura esplicita del displacement fra sensori;
- limitare tutti gli offset a quattro celle ha eliminato il collasso P5 ma ha
  peggiorato la detection, quindi non va nuovamente vincolato l'intero campo
  task-adaptive.

WiSARD contiene già annotazioni separate VIS e IR nei video usati per il
training. Esse non sono una calibrazione fisica e non forniscono identità degli
oggetti, ma possono produrre pseudo-corrispondenze conservative. La nuova
candidata usa questa informazione soltanto per fornire una **guida geometrica
debole** a P3; i 18 offset FAM originali restano residui liberi ottimizzati dal
task di detection.

## Audit delle annotazioni appaiate

L'audit è stato eseguito prima del nuovo training. Dopo la trasformazione
esistente di `adapt_ir2rgb`, l'IR `640×512` diventa `1350×1080` ed è centrata
con padding orizzontale di 285 pixel sul canvas VIS `1920×1080`. I box sono
associati soltanto se i centri sono mutual-nearest e la distanza euclidea
normalizzata non supera `0,05`.

| Coppia | Box VIS / IR | Match conservati | Copertura VIS / IR | Shift mediano a 640 px, IR−VIS |
|---|---:|---:|---:|---:|
| FHL 0405/0406 train | 577 / 1.534 | 510 | 88,4% / 33,3% | circa `(-9,5; -1,9)` px |
| Baker 1 train | 6.012 / 5.261 | 4.699 | 78,2% / 89,3% | circa `(-4,3; -5,3)` px |
| FHL 0401/0402 validation, sola verifica strutturale | 3.319 / 3.122 | 2.412 | 72,7% / 77,3% | circa `(-6,0; -6,8)` px |

La loss di training usa **soltanto i 5.209 match del train**. La coppia FHL
0401/0402 non entra nella loss: resta il video di validation Stage A. Il suo
audit precedente serve unicamente a verificare che ordine di grandezza e segno
non siano peculiari a una singola acquisizione.

### Inventario target Stage A congelato

La ricostruzione indipendente sui 3.123 frame paired del train ha verificato
bit per bit i tensori prodotti dal dataset. Sono presenti 5.209 match in 2.306
frame; 817 frame hanno tensor target vuoto. La distribuzione del numero di
match per frame è:

| Match nel frame | 0 | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| Frame | 817 | 801 | 543 | 526 | 436 |

La media è `1,667947` match per tutti i frame e `2,258890` nei soli 2.306
frame non vuoti; il massimo è quattro. FHL 0405/0406 contribuisce 510 match in
395/943 frame, Baker 1 contribuisce 4.699 match in 1.911/2.180 frame.

L'inventario canonico ordina i record per coppia di path relativi VIS/IR,
ordina le righe di ciascun tensor per `(x_VIS, y_VIS, dy, dx)`, serializza i
valori come `float32` little-endian e include identificatori e numero di righe
con lunghezze `uint32` big-endian. Il payload inizia con l'header ASCII esatto
seguente, seguito dal numero di frame come `uint32` big-endian:

```text
WISARD_BOX_ALIGNMENT_TARGETS_V1
mutual_nearest<=0.05
float32_le[x,y,dy,dx]
```

Lo SHA-256 congelato è:

```text
d519574962e81ae5b492248113247cca20d7ef15b2d189d1e3b58aebf218f3c0
```

L'inventario dei file sorgente (path, dimensioni e contenuto delle label) ha
SHA-256 `f889b5a54115f0267e0d5c087e6c3673bd2c65f63607e0c01df063620ea76a1e`.
L'audit ricalcola sia questo digest sia il digest dei target anche dai batch
effettivamente attraversati; non si limita a fidarsi del dataset costruito in
memoria.

Un match Baker è appena sotto la soglia (`0,0499999262`): il replay deve usare
la stessa aritmetica `float32` e la stessa regola `<= 0,05`. Un conteggio o hash
diverso blocca l'audit invece di essere accettato silenziosamente.

Le associazioni restano pseudo-label rumorose: non esistono object ID comuni,
FHL 0405 ha molte più annotazioni IR che VIS e l'IoU dei box fra sensori può
essere nullo anche per centri plausibilmente corrispondenti. Per questo non si
supervisionano né tutti i nove punti DCNv2 né P4/P5.

## Architettura congelata

La variante si chiama `box_guided_common_offset_p3`. P4 e P5 sono esattamente
`current_dcnv2`. Soltanto a P3 viene aggiunto un predittore compatto:

1. una proiezione `1×1`, condivisa fra RGB e IR, porta 512 canali a 32;
2. `GroupNorm(8)` e SiLU riducono la differenza di scala fra modalità;
3. il predittore riceve `RGB`, `IR`, `RGB−IR` e `RGB⊙IR` nello spazio comune;
4. produce un solo campo `g(p)=(dy,dx)` per posizione;
5. `g` viene sommato a ciascuno dei nove offset residui del FAM storico.

Il solo ramo nuovo aggiunge esattamente **53.410 parametri**: 16.448 nella
proiezione condivisa con GroupNorm, 36.896 nella convoluzione `3x3` del
predittore e 66 nell'uscita a due canali. P4, P5, decoder e teste non ricevono
nuovi parametri.

Per il punto `k` del kernel deformabile:

```text
offset_totale_k(p) = offset_FAM_k(p) + g(p)
```

Il ramo finale che produce `g` è inizializzato a zero. Prima del primo update
la candidata coincide quindi col FAM corrente, a parità dei pesi condivisi. Il
campo guida è limitato dolcemente a `±4` celle P3, mentre
`offset_FAM_k` **non è limitato**. Questo distingue la candidata
dall'ablation bounded-offset già fallita.

È stato inoltre corretto un confondente di inizializzazione: costruire il ramo
aggiuntivo consumava numeri casuali e avrebbe modificato i FAM P4/P5 e lo stato
RNG successivo pur usando lo stesso seed. La costruzione ora avviene in un RNG
fork locale e il subtree nuovo non viene reinizializzato dal `post_init` Hugging
Face. Il test di regressione verifica, a parità di seed, tutti i pesi FAM
condivisi bit-identici e lo stato RNG globale bit-identico fra FAM e candidata.
La neutralità funzionale iniziale e questa equivalenza RNG sono entrambe
necessarie: nessuna delle due sostituisce però un controllo addestrato nello
stesso ambiente.

### Provenienza del codice di training

Le quattro configurazioni scientifiche già preparate — controllo e candidata
seed 40, controlli seed 41--44 e candidata a cinque seed — dichiarano il
manifest `rtdetr_box_guided_training_source_v1`. Il manifest contiene 22 file
critici per costruzione del grid, dataset e target, modello e loss, training,
checkpoint, metrica e tracciamento. Il suo SHA-256 aggregato è:

```text
b06ea1328be206a9f7c64b3412f64ed7bb95b884da591c476584a90403592412
```

Prima di creare una run, il codice ricalcola gli hash dei 22 file e interrompe
il training se il manifest dichiarato non coincide o se la trace non è attiva
e scrivibile. Il primo evento della trace registra digest aggregato e hash di
ogni file. Gli audit post-training verificano poi la catena completa
configurazione W&B/YAML → manifest dichiarato → primo evento della trace →
sorgenti correnti. Una trace o un sorgente manomesso fa fallire l'audit.

Questo contratto copre la superficie sorgente esplicitamente necessaria al
training, non l'intero ambiente Conda, le dipendenze binarie o la cache Hugging
Face. Versioni runtime e inventari del dataset restano controlli separati. Il
probe tecnico `j37qaj8r` precede l'introduzione del manifest e non contiene
questi campi: resta valido soltanto come prova ingegneristica, mai come run
scientifica provenance-bound.

Quando Modal Dropout rende assente RGB o IR, `g` viene moltiplicato per zero per
quel campione. La guida modifica quindi soltanto la fusion; i percorsi nativi
RGB-only e IR-only continuano a usare il FAM storico.

L'implementazione è ispirata al principio di offset guidance e common subspace
di OAFA, ma **non è OAFA**: non implementa il suo decoupling
common/specific, le tre loss DML, le teste semantiche ausiliarie né il training
50+100 epoche. Riferimenti: [OAFA, CVPR
2024](https://openaccess.thecvf.com/content/CVPR2024/html/Chen_Weakly_Misalignment-free_Adaptive_Feature_Alignment_for_UAVs-based_Multimodal_Object_Detection_CVPR_2024_paper.html)
e [materiale
supplementare](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Chen_Weakly_Misalignment-free_Adaptive_CVPR_2024_supplemental.pdf).

## Loss ausiliaria congelata

Per ogni match, il target è il displacement del centro IR adattato rispetto al
centro VIS:

```text
target = (y_IR_adattato - y_VIS, x_IR_adattato - x_VIS)
```

Il predittore usa l'ordine `(dy, dx)` richiesto dagli offset interleaved di
`torchvision.ops.DeformConv2d`; un test sintetico blocca questa convenzione. Il
target normalizzato viene convertito nelle **dimensioni P3 osservate nel
forward**, non moltiplicato a priori per 640. La predizione è campionata nel
centro del box VIS e ottimizzata con Smooth L1:

```text
L = L_detection + lambda * warmup(epoch) * SmoothL1(g_P3, target_P3)
```

Valori bloccati prima del training:

- matching mutual-nearest, distanza massima normalizzata `0,05`;
- solo P3;
- 32 canali comuni;
- limite del solo campo guida `4` celle;
- `lambda = 0,2`;
- `beta Smooth L1 = 0,25` celle;
- warm-up lineare: scala `0,5` all'epoca 1 e `1,0` dall'epoca 2;
- learning rate del ramo guida `1e-4`;
- learning rate del resto del modello invariato a `2e-5`.

Non verrà svolta una grid su `lambda`, soglia di matching, numero di canali,
livelli o limite osservando ripetutamente FHL 0401/0402.

## Protocollo: Stage A e Stage B restano quelli comuni

Non viene introdotto un nuovo split sperimentale:

- Stage A train: FHL 0405/0406 + Baker 1, 3.123 coppie;
- Stage A validation: intero FHL 0401/0402, 896 coppie;
- dieci epoche fisse, AdamW, Modal Dropout nativo `20/20/60`;
- `best` selezionato solo dalla validation mAP@50 con `min_delta=0,001`;
- nuovo FAM matched control seed 40, poi candidata seed 40 e audit
  meccanicistico più controfattuale del suo checkpoint `best`;
- seed 40--44 soltanto in caso di promozione dello screen;
- MtErie escluso da Stage A.

L'eventuale Stage B userà tutte le 4.019 coppie e checkpoint `latest`, come il
protocollo già concordato. Prima di avviarlo verranno congelati sia la
candidata sia un **nuovo FAM matched control Stage B**, con stesso codice,
manifest e ambiente; non sarà sufficiente confrontare la candidata soltanto
con checkpoint o numeri storici non appaiati.

## Gate congelati

### Probe tecnico

Il probe da 20 step è completato nella run W&B `j37qaj8r`, directory locale
[`wandb/run-20260831_122623-j37qaj8r`](../wandb/run-20260831_122623-j37qaj8r/)
([summary](../wandb/run-20260831_122623-j37qaj8r/files/wandb-summary.json),
[log](../wandb/run-20260831_122623-j37qaj8r/files/output.log)). Ha eseguito 20
step di training e la validation completa di 896 frame in 132 secondi. La loss
media è `17,44285`; tutte le loss e i gradienti osservati sono finiti. All'ultimo
step la loss guida raw è `1,18184`, quella pesata `0,11818` con scala di warm-up
`0,5`. La validation mAP@50 è `0,07828`.

Questi valori dimostrano soltanto che dataset, collate, backward, memoria e
validation funzionano. La mAP@50 del probe è **non scientifica**, perché deriva
da 20 step e dalla validation già nota; non entra in alcun gate o confronto.
Il probe ha verificato:

- dataset e collate dei target variabili;
- valori finiti di detection loss e loss ausiliaria;
- gradienti non nulli sul nuovo ramo;
- memoria compatibile col batch 4;
- validation completa senza target ausiliari.

### Controfattuale validation congelato

Prima del training è stato congelato anche il replay active-vs-zero. Usa solo
FHL 0401/0402: inventario VIS 897, inventario IR 896, 896 coppie trattenute e
il solo frame VIS terminale `...00896.jpg` escluso dallo `zip` storico. Le GT
VIS contengono 3.319 box e 93 frame vuoti; le label IR 3.122 box e 100 frame
vuoti. I digest congelati sono:

- inventario storico path/dimensioni/label:
  `47e2f348ebdc202cb749b1bbf2741fc868d8681eb24ffa4356cf4f584dfa4ec4`;
- inventario forte che aggiunge i byte di tutte le immagini:
  `6c7748af3be2761a3a466b548af64aae925b693fbca795edf695072e28f17141`;
- ordine dei 896 campioni:
  `49415f065575c869087c78f842591096b74a0ea3a16ca2e4ce765e26958badcd`.

I due YAML seed 40 hanno SHA-256
`811c5c15ede894337b7dee2fe862bf50a9fd24fb31026fdf2b6c5c61eac4fef6`
per il matched control e
`16c1d58a4926bb6f6d1c018eb96600b52279116ca04a42d35286d63fb12ea647`
per la candidata. Questi hash dei file YAML sono distinti dal digest del
manifest sorgente.
Il runner accetta soltanto una run W&B completa da 10 epoche/7.810 step, priva
di metriche test e con configurazione scientifica identica al YAML. Carica
`best` in modo stretto, valuta active, azzera peso e bias della sola
convoluzione finale a due canali, valuta zero e ripristina esattamente i
tensori. I JSON/CSV sono stati pubblicati soltanto dopo entrambi i passaggi e
tutte le verifiche d'identità; gli esiti sono riportati nella sezione seguente.

### Screen seed 40

Il confronto primario usa il nuovo FAM matched control seed 40 addestrato prima
della candidata nello stesso codice e ambiente. Il valore FAM storico
`0,152148` resta descrittivo e non definisce più una soglia assoluta. Si passa
ai cinque seed solo se valgono tutte le condizioni:

1. delta appaiato `mAP@50_best(candidata) - mAP@50_best(matched control)`
   almeno `+0,01`;
2. sull'audit meccanicistico del train, errore Smooth L1 del campo appreso
   almeno 20% inferiore al predittore neutro `g=0`;
3. campo non degenere: ampiezza media assoluta maggiore di `0,05` celle e meno
   dell'1% delle componenti a `|g| >= 3,9` celle;
4. il campo comune totale, ottenuto sommando `g` alla media dei nove offset
   residui, non deve avere errore peggiore di `g=0`; il rapporto medio di
   cancellazione positiva e la quota di vettori guida cancellati almeno a metà
   devono essere entrambi al massimo `0,50`;
5. sul controfattuale della validation, lo stesso checkpoint e gli stessi 896
   frame devono dare `mAP@50(active) - mAP@50(g=0) >= 0`; il replay active deve
   riprodurre la best mAP@50 W&B entro `0,0002`;
6. nessun valore non finito o crash.

#### Esito osservato

| Run seed 40 | ID W&B | Best epoch | Best mAP@50 |
|---|---|---:|---:|
| FAM matched control | `2fx2ozwm` | 3 | `0,147388741` |
| Box-Guided P3 | `2jvqs9mr` | 1 | `0,155485332` |

Il delta appaiato è `+0,008096591`: manca il gate `+0,01` di
`0,001903409`. Il valore storico FAM seed 40 `0,152148` resta descrittivo e
non modifica questa decisione.

L'audit sui 3.123 frame e 5.209 match train mostra che il meccanismo ha
imparato un campo non banale:

- Smooth L1 `0,437972` contro `1,006123` di `g=0`, miglioramento `56,47%`;
- Smooth L1 `0,884058` per il miglior vettore costante
  `(-0,653662, -0,608873)` celle; la guida lo migliora del `50,46%`;
- correlazione centrata guida--target `0,786072`;
- ampiezza media assoluta `0,905823` celle e saturazione `0%`;
- rapporto medio di cancellazione `0,043572` e soltanto `0,006911` dei
  vettori cancellati almeno a metà;
- errore del campo comune totale `0,468253`, migliore del `53,46%` rispetto a
  zero.

Tutti i gate meccanicistici passano. Il controfattuale sugli stessi 896 frame
di validation riproduce esattamente la best W&B: `0,155485332` con guida
attiva e `0,155478507` con la sola uscita della guida azzerata. Il delta
`+0,000006825` passa formalmente il vincolo non negativo, ma è troppo piccolo
per sostenere un vantaggio diretto in inferenza. Anche AP50:95 e AP75 sono
leggermente inferiori con la guida attiva (`0,084657` contro `0,085045` e
`0,082546` contro `0,083776`), perciò non vanno usate come salvataggio
post-hoc del candidato.

Il primo tentativo dell'audit ha inoltre rivelato un bug fail-closed nel solo
runner: il replay runtime, che ricostruisce identificatori e target dai batch,
veniva erroneamente obbligato a fornire anche l'hash dei byte dei file già
verificato dal preflight indipendente. Il controllo è stato separato senza
alterare dati, soglie o accumulatori: il preflight continua a verificare i
file, il replay verifica popolazione e digest dei target. Nessun output
parziale era stato pubblicato. Il runner corretto ha SHA-256
`d0d868f75a4de4c064b61c774dc9d8056dd82f8bd8f977b41f1822e2b9dcf14d`.

SHA-256 degli artefatti scientifici:

```text
mechanism JSON:      fb85e3e885836be64c2bc26377bd6aa33172b6c91d365e5df80a9b547e0cec9c
mechanism CSV:       1f9616fdca680c61c69546467e288c35d7aee47100c999a326a2e8271b58f116
counterfactual JSON: f1aecdad12c36160ad9b15c1c0730fff68a9a72d86cb340958e192673b9b76b2
counterfactual CSV:  3089ed1717b75a9075a831dae0c967cb15543619240512f62db8457ac534981e
```

I primi tre controlli del meccanismo, i due controlli di cancellazione e il
controllo controfattuale sono implementati come gate fail-closed nei runner,
non lasciati a interpretazione manuale. Il controfattuale isola soltanto il
contributo **diretto in inferenza** del ramo già addestrato: un delta esattamente
zero significa "non degradante", non dimostra un contributo positivo. Se un
requisito fallisce, non si cambia `lambda` post-hoc: la candidata viene chiusa
e si passa alla successiva direzione preregistrata.

L'audit produce inoltre una diagnostica **descrittiva e non selettiva** contro
il miglior vettore costante `(dy, dx)`: è il minimizzatore esatto, calcolato per
bisezione componente per componente, della stessa Smooth L1 con `beta=0,25`
sui 5.209 target train congelati. Vengono salvati loss e vettore costante,
media e deviazione standard di guida e target, correlazione centrata e quota
del campo spiegata dal bias medio. Questo confronto non modifica nessun gate
di promozione già congelato. Se la guida non migliora **strettamente** il
predittore costante, un eventuale risultato positivo può essere descritto al
massimo come calibrazione globale appresa, non come allineamento dipendente
dall'input. Anche se lo batte, dimostra soltanto un fit non globale sui target
train: potrebbe riflettere struttura spaziale o specifica delle acquisizioni e
non prova generalizzazione input-conditioned. Sia `g=0` sia il baseline
costante sono valutati sul train usato dalla loss e misurano il fit del
meccanismo, non la generalizzazione; il controfattuale sui 896 frame di
validation resta il controllo prestazionale fuori dal train.

### Stage A a cinque seed

**Non eseguito e non autorizzato:** il gate primario seed 40 è fallito. Quanto
segue conserva il protocollo preregistrato per tracciabilità, non costituisce
un piano di lancio corrente.

La promozione allo Stage B richiede, contro i FAM appaiati seed 40--44:

- delta medio della best validation mAP@50 almeno `+0,01`;
- almeno 4/5 delta positivi;
- audit del meccanismo valido in tutti i checkpoint candidati.

AP50:95, AP75, recall e deviazione standard sono secondarie. Possono spiegare
il risultato ma non sostituiscono il gate primario.

Il seed 40 usa il matched control dedicato dello screen. Se e solo se lo screen
passa, i seed 41--44 usano quattro nuovi controlli FAM addestrati nello stesso
codice e ambiente mediante il YAML già congelato; le run FAM storiche non sono
più il riferimento primario. Per la candidata a cinque seed si salta il seed 40
già completato con `--start-from-run 1`.

Il protocollo meccanicistico seed 40 resta intenzionalmente separato. È già
congelato anche il protocollo aggregato v2: risolve il seed 40 dai due progetti
dello screen e i seed 41--44 dai due progetti di espansione, verificando hash
YAML, ordine e indice del grid, `start_from_run`, manifest/trace, identità dei
seed e un fingerprint runtime comune su tutte le dieci run. Prima di poter
promuovere richiede inoltre che i JSON seed 40 dell'audit e del controfattuale
esistano, siano completi e passati e puntino allo stesso checkpoint; impone di
nuovo delta seed 40 `>= +0,01`, delta medio `>= +0,01`, 4/5 vittorie e
meccanismo 5/5. I cinque checkpoint di controllo vengono caricati in modo
stretto, mentre sui cinque candidati viene ripetuto il replay del meccanismo.
Il runner non sovrascrive output scientifici esistenti.

La metrica primaria dell'aggregato resta il `best_map_50` registrato nella
summary W&B co-localizzata e legata al checkpoint, coerentemente col protocollo
storico. Il controfattuale riproduce indipendentemente il valore della candidata
seed 40; i valori 41--44 non vanno invece descritti come “replayed” finché non
si congela un eventuale integrity replay validation separato.

Il v2 impedisce una promozione con prerequisiti assenti o falliti, ma un audit
post-hoc non può dimostrare da solo l'ordine temporale. Era quindi previsto un
authorization manifest con gli hash dei due JSON seed 40 prima di lanciare
41--44. Poiché il gate primario è fallito, il manifest non viene creato e i
YAML condizionali restano definitivamente inattivi per questa candidata.

## Decisione dopo questa candidata

Il failure mode previsto è ora osservato: il campo apprende bene la geometria,
ma il suo contributo diretto alla validation è praticamente nullo e il delta
fra run non raggiunge il gate. Non si attiva quindi il fallback cost-volume
`9×9`, che era motivato soltanto dall'eventuale incapacità del predittore di
apprendere gli shift. Una correlazione più esplicita risolverebbe un problema
che questo audit non ha mostrato.

Con circa un mese disponibile, la prossima linea coerente è una modifica
detector-level separata: **RT-DETRv2 + FAM**, iniziando da un singolo seed
Stage A con controllo appaiato e senza riutilizzare la validation per una grid.
Se il port risulta impraticabile o non promettente, la terza scelta resta
D-FINE con FDR/GO-LSD, più onerosa ma più direttamente orientata alla qualità
della localizzazione. Box-Guided, cost-volume, RCRA e consistency non vanno
combinati con questo primo screen detector-level.

D-FINE è interessante perché FDR e GO-LSD migliorano più famiglie DETR e non
richiedono un teacher su input degradati; il port nella versione Hugging Face
locale modifica però decoder, regressione e loss, quindi ha rischio e costo di
attribuzione maggiori. Riferimenti: [D-FINE, ICLR
2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/6cf58a87e3097e7d1f9be3e8693a93de-Paper-Conference.pdf)
e [repository ufficiale](https://github.com/Peterande/D-FINE). RT-DETRv2
rimane più vicino al codice corrente, ma sul ResNet-50 il guadagno COCO
ufficiale è modesto; la parte più interessante per WiSARD è la recipe dinamica
di augmentazione. Riferimenti: [RT-DETRv2](https://arxiv.org/abs/2407.17140)
e [repository ufficiale](https://github.com/lyuwenyu/RT-DETR).

Audit locale del 31 agosto 2026: l'ambiente `sarfusion` usa Transformers
`4.43.3` e non espone né `RTDetrV2ForObjectDetection` né
`DFineForObjectDetection` (non sono presenti neppure i relativi moduli). Queste
due linee richiederebbero quindi un ambiente isolato/upgrade o un port dal
repository ufficiale. Questo non le esclude nel mese disponibile, ma impedisce
di trattarle come una semplice sostituzione di una classe nel codice corrente.

## Comandi e ordine operativo

Il probe tecnico seguente è già completato e non va contato né ripetuto come
run scientifica:

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
MPLCONFIGDIR=/tmp/matplotlib-rtdetr-box-guided \
YOLO_CONFIG_DIR=/tmp/yolo-rtdetr-box-guided \
PYTHONUNBUFFERED=1 \
conda run --no-capture-output -n sarfusion python main.py experiment \
  --parameters parameters/RTDETR/rtdetr_fam_box_guided_runtime_probe.yaml
```

Il matched control seed 40 è stato eseguito con il comando seguente, conservato
per riproducibilità ma da non rilanciare:

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
MPLCONFIGDIR=/tmp/matplotlib-rtdetr-box-guided-control \
YOLO_CONFIG_DIR=/tmp/yolo-rtdetr-box-guided-control \
PYTHONUNBUFFERED=1 \
conda run --no-capture-output -n sarfusion python main.py experiment \
  --parameters parameters/RTDETR/rtdetr_fam_box_guided_matched_control_seed40.yaml
```

La candidata seed 40 è stata poi eseguita con:

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
MPLCONFIGDIR=/tmp/matplotlib-rtdetr-box-guided \
YOLO_CONFIG_DIR=/tmp/yolo-rtdetr-box-guided \
PYTHONUNBUFFERED=1 \
conda run --no-capture-output -n sarfusion python main.py experiment \
  --parameters parameters/RTDETR/rtdetr_fam_box_guided_sequence_validation_seed40.yaml
```

Il comando effettivamente usato per le due run, mantenendo l'ordine controllo
→ candidata e interrompendo la sequenza se il primo processo restituisce
errore, è stato:

```bash
set -e
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MPLCONFIGDIR=/tmp/matplotlib-rtdetr-box-guided-overnight
export YOLO_CONFIG_DIR=/tmp/yolo-rtdetr-box-guided-overnight
export PYTHONUNBUFFERED=1

for configuration in \
  parameters/RTDETR/rtdetr_fam_box_guided_matched_control_seed40.yaml \
  parameters/RTDETR/rtdetr_fam_box_guided_sequence_validation_seed40.yaml
do
  conda run --no-capture-output -n sarfusion python main.py experiment \
    --parameters "$configuration"
done
```

Le due run hanno richiesto circa due ore ciascuna (`7.435` e `7.477` secondi
registrati da W&B). A
differenza del mixed consistency, questa candidata non attiva forward
aggiuntivi dalla seconda epoca: il ramo guida è eseguito da subito e cambia
soltanto il peso di warm-up (`0,5` poi `1,0`), perciò non è atteso il precedente
salto sistematico da 11 a 35--40 minuti per epoca.

Le directory locali `wandb/run-*` prodotte da queste run non vanno cancellate
o spostate: gli audit richiedono insieme checkpoint, `config.yaml`, summary e
`reproducibility_trace.jsonl`. In caso di crash non si rilancia alla cieca lo
stesso progetto/seed, perché due checkpoint candidati renderebbero ambigua la
risoluzione; prima si ispeziona e si documenta la run incompleta.

L'audit meccanicistico è stato eseguito sul checkpoint `best` con:

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
MPLCONFIGDIR=/tmp/matplotlib-rtdetr-box-guided-audit \
YOLO_CONFIG_DIR=/tmp/yolo-rtdetr-box-guided-audit \
PYTHONUNBUFFERED=1 \
conda run --no-capture-output -n sarfusion python \
  scripts/run_rtdetr_fam_box_guided_mechanism_audit.py \
  --protocol parameters/RTDETR/rtdetr_fam_box_guided_mechanism_audit_seed40.yaml
```

Subito dopo è stato eseguito il controfattuale sullo stesso checkpoint `best`. Il
runner attraversa FHL 0401/0402 prima con il campo appreso e poi azzerando
temporaneamente soltanto la convoluzione finale del ramo guida; verifica
ordine, ground truth e ripristino bit-identico dei pesi:

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
MPLCONFIGDIR=/tmp/matplotlib-rtdetr-box-guided-counterfactual \
YOLO_CONFIG_DIR=/tmp/yolo-rtdetr-box-guided-counterfactual \
PYTHONUNBUFFERED=1 \
conda run --no-capture-output -n sarfusion python \
  scripts/run_rtdetr_fam_box_guided_counterfactual.py \
  --protocol parameters/RTDETR/rtdetr_fam_box_guided_counterfactual_seed40.yaml
```

Il file a cinque seed resta versionato per tracciabilità, ma **non va
eseguito**: il gate seed 40 è fallito. I comandi condizionali originariamente
preregistrati erano:

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
MPLCONFIGDIR=/tmp/matplotlib-rtdetr-box-guided-five-seed \
YOLO_CONFIG_DIR=/tmp/yolo-rtdetr-box-guided-five-seed \
PYTHONUNBUFFERED=1 \
conda run --no-capture-output -n sarfusion python main.py experiment \
  --parameters parameters/RTDETR/rtdetr_fam_box_guided_matched_control_seeds41_44.yaml

HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
MPLCONFIGDIR=/tmp/matplotlib-rtdetr-box-guided-five-seed \
YOLO_CONFIG_DIR=/tmp/yolo-rtdetr-box-guided-five-seed \
PYTHONUNBUFFERED=1 \
conda run --no-capture-output -n sarfusion python main.py experiment \
  --parameters parameters/RTDETR/rtdetr_fam_box_guided_sequence_validation_five_seed.yaml \
  --start-from-run 1
```

Il seguente audit aggregato sarebbe stato eseguito soltanto dopo dieci run
autorizzate; nella campagna corrente non è applicabile:

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
MPLCONFIGDIR=/tmp/matplotlib-rtdetr-box-guided-five-seed-audit \
YOLO_CONFIG_DIR=/tmp/yolo-rtdetr-box-guided-five-seed-audit \
PYTHONUNBUFFERED=1 \
conda run --no-capture-output -n sarfusion python \
  scripts/run_rtdetr_fam_box_guided_stage_a_five_seed_audit.py \
  --protocol parameters/RTDETR/rtdetr_fam_box_guided_stage_a_five_seed_audit_v2.yaml
```

## Vincoli interpretativi

- I match di box sono weak labels, non ground truth di registrazione.
- FHL 0401/0402 è una validation comparabile ma già consultata da molte
  varianti; non è un holdout puro.
- MtErie, Carnation, le confirmation acquisition e lo stress sintetico sono
  già stati osservati e non possono essere usati per scegliere iperparametri.
- Un guadagno dimostrerebbe l'utilità di una guida geometrica debole per il
  detector; non dimostrerebbe una calibrazione fisica perfetta delle camere.
