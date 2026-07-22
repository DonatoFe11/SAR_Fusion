# Verifica dell'allineamento del FAM — riepilogo del lavoro

## Obiettivo

Prima di estendere il FAM (Feature Alignment Module) a una nuova architettura (YOLO), verificare se il FAM stia davvero realizzando un allineamento spaziale RGB-IR, e non solo migliorando la mAP in modo indiretto/non attribuibile con certezza al meccanismo di allineamento. Fino a questo punto, tutta l'evidenza a favore del FAM era indiretta (mAP che sale, con range statistici che si sovrappongono tra baseline e FAM).

Metodologia suggerita dal dottorando supervisore: visualizzazione PCA delle feature map (stile DINOv2/DINOv3 — un estrattore di feature self-supervised di Meta AI, da non confondere con il DINO usato nel progetto, che è invece "DETR with Improved DeNoising Anchor Boxes", Zhang et al. ICLR 2023, una tecnica di training per detection completamente diversa nonostante il nome uguale).

## Strumento sviluppato

Script `fam_alignment_check.py`, autonomo e parametrico, che:
- Carica un modello fusion (`fusion_rtdetr` o `fusion_defdetr`) da un file yaml di run (formato "grid search" di `sarfusion.experiment.Experimenter`) + un checkpoint `.safetensors`
- Intercetta via forward hook (agganciati per **nome di classe**, `FeatureAlignmentModule`, non per path fisso — quindi robusto a differenze tra le implementazioni RT-DETR/Deformable DETR) le feature RGB e IR immediatamente prima e dopo ogni istanza del FAM nella backbone, più il campo di offset grezzo predetto da `offset_conv`
- Proietta le feature con **PCA a base condivisa** (fittata su RGB+IR+FAM(IR) messi in comune, non indipendentemente per ciascuna — dettaglio cruciale, vedi sotto) per renderle visualizzabili e confrontabili a colori
- Converte l'ampiezza del campo di offset da pixel di feature map a **pixel dell'immagine originale**, tenendo conto della stride reale di ciascun livello della piramide (8/16/32 per i 3 livelli usati)
- Supporta l'analisi aggregata su più campioni in un solo lancio (`--sample-idx 0 1 2 ...`), con un riepilogo statistico finale (mean/median/p90/max dell'offset in pixel, pooled su tutti i campioni)

Il codice riusa direttamente `sarfusion.utils.grid.make_grid`, `sarfusion.data.get_dataloaders`, `sarfusion.models.build_model` — nessuna reimplementazione della logica esistente del progetto.

## Percorso seguito (cronologia sintetica)

1. **Impostazione iniziale**: script che carica un checkpoint, registra gli hook, gira inferenza su un campione, produce 4 pannelli (PCA RGB, PCA IR, PCA FAM(IR), overlay pre/post) + campo di offset.
2. **Bug di processo**: primo script creato ma mai reso scaricabile — corretto.
3. **Bug del formato yaml**: i file di run non sono "flat" ma nel formato grid-search (`parameters:` con ogni valore terminale wrappato in una lista). Prima reimplementato a mano l'unwrap, poi sostituito con l'uso diretto di `sarfusion.utils.grid.make_grid` (la stessa funzione usata dal codice di produzione) una volta recuperato il file `grid.py`, per eliminare ogni scostamento dal comportamento reale (es. il caso `None -> {}` per le sezioni vuote).
4. **Primo giro sul modello sbagliato**: la prima verifica è stata fatta sul checkpoint RT-DETR+FAM+**SSJ** (spatial jitter). Osservazione dell'utente: per isolare "il FAM funziona?" dalla domanda "cosa fa la SSJ?", va testato prima il modello con **solo FAM** (senza spatial jitter in training), altrimenti i pesi di `offset_conv`/`deform_conv` riflettono l'interazione tra i due meccanismi, non il FAM in isolamento. Chiarito che `spatial_jitter_std` nel config non influisce sull'inferenza (`model.eval()` disattiva l'iniezione di rumore comunque), quindi l'unica variabile è il checkpoint.
5. **Bug metodologico nella PCA**: la prima versione fittava una PCA indipendente per ciascuna feature map (RGB, IR, FAM(IR)). Basi PCA indipendenti hanno rotazione/segno arbitrari — colori uguali in pannelli diversi non indicano la stessa struttura. Corretto fittando **una base condivisa** su tutte e tre le feature map insieme.
6. **Ipotesi di artefatto di scala**: dopo la correzione, `PCA(FAM(IR))` appariva quasi uniforme/saturo a tutti i livelli. Ipotesi: `deform_conv` non ha normalizzazione a valle (a differenza di RGB/IR che escono da un backbone con BatchNorm), quindi potrebbe avere una scala di attivazione diversa e dominare/schiacciare il fit PCA condiviso. Aggiunta stampa di statistiche grezze (mean/std/min/max) per ogni feature map, e standardizzazione (z-score) di ciascuna prima del fit PCA condiviso. **Risultato: l'ipotesi di puro artefatto di scala non era supportata** (std comparabili tra le tre feature map) — il pattern osservato (FAM(IR) più liscio/a bassa frequenza di IR grezzo) è reale, non un artefatto di normalizzazione.
7. **Reinterpretazione**: il comportamento osservato non è "collasso a zero informazione" ma **smoothing/filtraggio passa-basso** — spiegabile meccanicisticamente con l'interpolazione bilineare della deformable conv e l'inizializzazione della mask a 0.5 (miscelazione morbida sui 9 tap, non un tap netto).
8. **Correzione di scala per l'offset**: il quiver plot mostrava l'offset in pixel di *feature map*, non di immagine originale — a stride diverse (8/16/32) lo stesso numero grezzo corrisponde a spostamenti fisici molto diversi. Aggiunta la conversione esplicita, che ha rivelato spostamenti medi di alcuni pixel e code fino a ~100 px — molto più significativi di quanto sembrasse dal solo quiver plot.
9. **Confronto quantitativo Modello B (solo FAM) vs Modello E (FAM+SSJ)**: prima su 1 campione, poi esteso a 3 campioni del test set con riepilogo aggregato, per verificare che il pattern non fosse specifico di una singola scena.

## Risultati numerici finali

Offset del FAM, **in pixel dell'immagine originale**, aggregati su 3 campioni del test set (n = 21.600–345.600 vettori di offset per livello):

| livello | stride | Modello B (solo FAM): mean / median / p90 / max | Modello E (FAM+SSJ): mean / median / p90 / max | max/mean B | max/mean E |
|---|---|---|---|---|---|
| 0 | 8 | 2.78 / 2.53 / 5.25 / 20.70 px | 4.02 / 3.14 / 9.11 / 29.26 px | 7.4× | 7.3× |
| 1 | 16 | 2.72 / 2.08 / 5.78 / 56.97 px | 4.34 / 3.43 / 9.48 / 43.16 px | 20.9× | 9.9× |
| 2 | 32 | 3.95 / 3.08 / 7.80 / 94.55 px | 4.90 / 4.03 / 9.53 / 99.66 px | 23.9× | 20.3× |

Checkpoint usati: `fusion_rtdetr_fam_correct.safetensors` (Modello B), `spatial_jitter.safetensors` (Modello E). Config comune: `parameters/RTDETR/fusion_rtdetr.yaml`.

## Scoperte principali

1. **Il FAM non è un modulo "morto"**: produce campi di offset non banali, di ampiezza fisicamente plausibile per correggere una parallasse RGB-IR reale (ordine di pochi pixel medi, code fino a ~100 px).
2. **L'ipotesi originale sulla SSJ era imprecisa**: non è vero che la SSJ regolarizza il FAM verso offset più piccoli — anzi la mean/median è sistematicamente **più alta** nel Modello E. Quello che la SSJ fa chiaramente, in modo coerente su tutti e 3 i livelli e su tutte le scene testate, è **ridurre drasticamente il rapporto max/mean**: sposta il FAM da poche correzioni-outlier isolate e molto estreme verso una correzione più diffusa, di ampiezza tipica maggiore ma meno dipendente da singoli punti estremi. Caratterizzazione più precisa da riportare in tesi: *"SSJ regolarizza verso l'uniformità spaziale della correzione, non verso una minore ampiezza media"*.
3. **FAM(IR) mostra smoothing rispetto a IR grezzo**: comportamento spiegabile dall'interpolazione bilineare della deformable conv, non un difetto o un collasso dell'informazione (confermato non essere un artefatto di scala/normalizzazione).
4. **Il campo di offset è quasi identico tra scene diverse — confermato anche cross-sessione**: già nei 3 campioni del test set (stessa sessione di volo, MtErie) l'offset risultava pressoché identico. Verifica successiva estesa a campioni di **sessioni di volo diverse** (FHL 210924 vs Baker 220109, dal train set): mean offset per livello 2.58–2.59 px / 2.79–2.81 px / 3.84–3.85 px in entrambe le sessioni, sostanzialmente indistinguibili tra loro e dai valori già osservati sulla sessione MtErie. **Conclusione consolidata** (non più solo ipotesi): il FAM ha imparato una correzione sistematica/di calibrazione tra le due camere, la cui ampiezza tipica non dipende né dalla scena né dalla sessione di volo — non un allineamento dinamico content-adaptive. Da riportare in tesi come caratterizzazione precisa del comportamento appreso dal modulo, non come limite dell'analisi.
5. **Artefatto di preprocessing individuato**: le bande nere verticali visibili a livello 0 in `PCA(IR)` derivano dal padding introdotto da `adapt_ir2rgb` (letterboxing per compensare l'aspect ratio IR/RGB), non da un problema del FAM — utile da sapere per non fraintendere le visualizzazioni.
6. **Nota tecnica sul checkpoint usato**: `missing=48, unexpected=0` nel caricamento — inizialmente interpretato come possibile export parziale (solo backbone+FAM, senza teste). Chiarito successivamente che i file in `checkpoints/` sono copie dirette e complete di `best/model.safetensors` dalle cartelle di tracking wandb, quindi le teste **sono** state allenate. La causa reale identificata: `RTDetrFusionForObjectDetection` tiene `class_embed`/`bbox_embed` sia come attributi top-level sia aliasati dentro `model.decoder` (stesso oggetto Python, non una copia — vedi `rtdetr_fusion.py`); `safetensors` non può salvare due chiavi che condividono la stessa memoria, quindi nel file sopravvive un solo path e l'altro risultava "missing" nello script pur essendo lo stesso peso allenato. Corretto lo script con un passo di recupero per identità di oggetto (non per nome indovinato): per ogni parametro "missing" si cercano automaticamente i suoi alias nell'albero dei moduli, e si recupera il valore dal checkpoint se presente sotto uno di quei nomi alternativi. **Confermato con successo**: rilanciando lo script, tutti i 48 pesi vengono recuperati correttamente (`Recuperati 48/48 ... missing=0 unexpected=0`). I numeri di offset restano identici a prima del fix (atteso: le teste di detection non influenzano le feature intercettate dagli hook), quindi tutta l'analisi precedente resta valida — questo era un problema cosmetico nel logging del caricamento, non nei risultati. Effetto collaterale utile: ora che il modello carica per intero, è diventato possibile anche verificare l'effetto del FAM sui box predetti reali (post-processing delle predizioni), non solo sulle feature intermedie.

## Conclusione: il FAM funziona?

Sì, in un senso preciso ma più modesto dell'ipotesi di partenza: il modulo apprende una correzione geometrica reale, di ampiezza plausibile, e risponde in modo meccanicisticamente sensato a un cambiamento del regime di training (SSJ). Non è invece confermato — anzi i dati vanno chiaramente in direzione opposta, confermato anche cross-sessione — che si tratti di un allineamento dinamico e content-adaptive per ogni scena: è una correzione di calibrazione sistematica tra le due camere, la cui ampiezza tipica non varia né tra scene né tra sessioni di volo diverse. Non abbiamo inoltre una conferma visiva diretta e pulita (tipo "la sagoma si sposta e combacia"); la verifica sull'effetto sui box predetti reali è ora tecnicamente possibile (risolto il problema di caricamento del checkpoint) ma non ancora eseguita.

## Limiti dell'analisi attuale

- Nessuna verifica su predizioni/box reali (lo script ora recupera anche i pesi delle teste grazie al fix sugli alias condivisi — questa verifica è diventata fattibile, semplicemente non ancora fatta)
- Analisi limitata a RT-DETR (Modelli B ed E); non ripetuta su Deformable DETR/DINO, ma la priorità concordata è di non investire ulteriore tempo lì, dato che restano comunque sotto RT-DETR in ogni scenario
- La verifica cross-sessione è stata fatta solo sul Modello B (solo FAM); non ancora ripetuta sul Modello E (FAM+SSJ) — da considerare solo se emergono elementi che lo rendano interessante

## Prossimi passi

1. **Estensione a YOLO + FAM** (prossimo blocco sperimentale, ora la priorità principale essendosi risolta la verifica cross-sessione): implementare una backbone dual (RGB + IR) con inserimento del FAM ai livelli P3/P4/P5, riusando la classe `FeatureAlignmentModule` così com'è (è già disaccoppiata dall'architettura). Inquadramento scientifico: terzo test, dopo RT-DETR, dell'ipotesi "SSJ+FAM stabilizza su architetture ad attenzione non-deformabile" — YOLO è un'architettura CNN pura, senza deformable attention né matcher Ungherese discontinuo, quindi è il data point più pulito per generalizzare (o confutare) questa scoperta oltre RT-DETR.
2. Una volta pronto YOLO+FAM, ripetere la stessa identica diagnostica (`fam_alignment_check.py` è già riusabile, indipendente dall'architettura) per confrontare il comportamento del FAM su YOLO con quello osservato qui su RT-DETR.
3. (Opzionale) Ora che lo script recupera correttamente anche i pesi delle teste (fix sugli alias condivisi), visualizzare i box predetti reali (post-processing delle predizioni sovrapposto a RGB/IR) come verifica qualitativa complementare a quella quantitativa sugli offset — utile ma non prioritario, dato che l'evidenza quantitativa già raccolta è più solida di un confronto visivo dei box.
4. (Opzionale) Ripetere la verifica cross-sessione anche sul Modello E (FAM+SSJ), per vedere se anche lì la correzione risulta indipendente da scena/sessione, o se la SSJ introduce una qualche dipendenza dal contenuto che il solo FAM non ha.
