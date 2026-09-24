# Strategia di Tiling (quadranti fissi 2×2)

> **Stato storico/esplorativo.** I risultati di questa nota provengono da
> singole run e valutano soltanto la specifica strategia a quadranti fissi
> implementata nel progetto. Sono sufficienti per motivarne l'abbandono nello
> sviluppo, ma non dimostrano che ogni metodo di tiling sia inferiore né vanno
> confrontati come stime finali con la campagna RT-DETR multi-seed.

Ho provato a dividere l’input in quattro quadranti per aumentare la scala apparente dei target. L’implementazione richiede tre passaggi: ritagliare immagini e label nel dataset, rimappare le predizioni e aggregarle prima delle metriche.

> L'implementazione non è una sliding window generica con stride e sovrapposizione: divide ogni immagine ridimensionata a $640\times640$ in quattro quadranti fissi e non sovrapposti da $320\times320$.

## Fase 1: Generazione e Caricamento dei Tile (Dataset Layer)

**File di riferimento**: [sarfusion/data/wisard.py](../sarfusion/data/wisard.py)

L'immagine e le annotazioni vengono manipolate "alla base" del framework, durante la chiamata `__getitem__` del Dataset. L'obiettivo è lavorare su patch di dimensioni ridotte (es. $320 \times 320$) a partire dall'immagine ad alta risoluzione (es. $640 \times 640$).

### Dinamiche di Training vs Testing
Il tiling si comporta diversamente a seconda della fase di esecuzione:
- **Training** (`use_tiling=True`): Il metodo `__getitem__` seleziona casualmente un solo quadrante per l'immagine (`quadrant = random.randint(0, 3)`). Questo modifica la distribuzione degli input e può agire da data augmentation, ma riduce anche il contesto disponibile e la probabilità che un target cada nel tile selezionato.
- **Testing/Evaluation** (`test_all_tiles=True`): Il dataset esplode le istanze (`expanded_items`). Per ciascuna immagine originale, il dataloader restituisce sistematicamente 4 item separati, uno per ciascun quadrante (0, 1, 2, 3). In output vengono esplicitamente allegati i metadati `original_idx` e `quadrant`, fondamentali per ricostruire l'immagine in Fase 3.

### La funzione di ritaglio: `_get_tile`
Questo metodo gestisce la matematica necessaria a eseguire il crop delle immagini e la ricalibrazione dei target (Ground Truths).

1. **Ritaglio Immagine**:
   Calcola l'offset a partire dall'indice del quadrante:
   ```python
   # 640 // 2 = 320 -> Dimensione tile
   x_off = (quadrant % 2) * tile_size
   y_off = (quadrant // 2) * tile_size
   ```
   Così otteniamo:
   - Quadrante 0: Alto-Sinistra (x=0, y=0)
   - Quadrante 1: Alto-Destra (x=320, y=0)
   - Quadrante 2: Basso-Sinistra (x=0, y=320)
   - Quadrante 3: Basso-Destra (x=320, y=320)

2. **Traslazione e Clipping dei Bounding Box**:
   Oltre ai pixel, bisogna spostare i box annotati:
   - **Assegnazione logica**: Viene calcolato il centro $(X_{center}, Y_{center})$ di ogni box originario. Se questo centro cade entro il quadrante, l'oggetto è preservato.
   - **Traslazione**: Essendo l'immagine originariamente ancorata in $(0,0)$, per allineare i box occorre sottrarre gli offset (`x_min - x_off`, `y_min - y_off`).
   - **Clipping**: Un oggetto potrebbe avere il centro in un tile ma invadere lo spazio di un altro. Tramite costrutti del tipo `min(bbox_width, tile_size - new_x_min)`, il riquadro viene contenuto nel tile. **Limite del codice attuale:** dopo `new_x_min = max(0, x_min - x_off)` la larghezza originale non viene ridotta della porzione tagliata a sinistra; lo stesso vale in alto per l’altezza. Non è quindi un’intersezione geometrica esatta. Per esempio, con `x_off=320`, `x_min=300`, `width=80`, il codice restituisce `x=0, width=80`, mentre l’intersezione corretta avrebbe larghezza 60. Questa limitazione va considerata nell’interpretazione degli esperimenti storici.

### Resize quadrato (`resize((640, 640))`)
Per garantire il corretto funzionamento geometrico di questo approccio, durante il `__getitem__` delle singole modalità (sia RGB che IR), l'immagine in PIL originale viene ridimensionata forzatamente a $640 \times 640$ prima di subire le operazioni descritte da `_get_tile`. Questo rende intere le dimensioni dei quadranti, ma **non preserva l’aspect ratio** se l’immagine iniziale non è quadrata.

## Fase 2: Geometria e Fusione delle Predizioni (Logic Layer)

**File di riferimento**: [sarfusion/data/tile_aggregation.py](../sarfusion/data/tile_aggregation.py)

Una volta che il modello ha processato i frammenti (tile) e generato le predizioni (con coordinate relative ai $320 \times 320$), è necessario ricostruire l'ingombro totale per valutare correttamente l'immagine a risoluzione intera ($640 \times 640$).

### 1. Rimappatura delle Coordinate (`remap_tile_boxes_to_original`)
Le predizioni in output sono normalizzate rispetto al tile (i valori $x, y, w, h$ vanno da 0 a 1). La funzione di rimappatura ripristina queste coordinate rispetto all'immagine originaria:
- Poiché la larghezza/altezza del tile è esattamente la metà, le larghezze e le altezze predette vengono scalate per un fattore $0.5$.
- Anche i *centri* dei box vengono scalati per $0.5$, ma in base al quadrante originario subiscono anche una "traslazione normalizzata" di $0$ o $0.5$. Ad esempio, per un oggetto trovato nel tile in basso a destra (quadrante 3), al centro $X$ e al centro $Y$ predetti scalati viene aggiunto $0.5$, riposizionando l'oggetto nella metà inferiore e destra dell'immagine.

### 2. Eliminare i duplicati con la NMS (`aggregate_tile_predictions`)
Un oggetto vicino al bordo può generare predizioni in due tile adiacenti.
Dopo averle riportate sulla stessa griglia, applico la NMS:

1. ordino i box per confidenza decrescente;
2. conservo quello con score maggiore;
3. elimino i box rimasti che hanno IoU superiore a 0.5 con quello conservato;
4. ripeto sui box ancora disponibili.

Il confronto è globale, indipendente dalla classe.

In [sarfusion/data/tile_aggregation.py](../sarfusion/data/tile_aggregation.py), i box vengono prima uniti, convertiti dal formato centrale (cx, cy, w, h) al formato a vertici (x1, y1, x2, y2), mantenendo le coordinate normalizzate rispetto all’immagine intera, ed infine passati alla funzione integrata `torchvision.ops.nms` per produrre il set compatto di rilevamenti aggregati. La NMS è globale, senza separazione per classe: questo è coerente con il caso a classe singola, ma non implementa una NMS distinta per classe.

## Fase 3: Orchestrazione e Accumulo nel Loop (Pipeline Layer)

**File di riferimento**: [sarfusion/experiment/run.py](../sarfusion/experiment/run.py)

Il dataloader fornisce i dati in **batch** (blocchi) casuali o sequenziali. Con il parametro `test_all_tiles=True`, una singola immagine genera 4 tile distinti. Dipendendo dalla dimensione del batch (es. batch size = 8 o 16), è possibile che i 4 frammenti di un'immagine vengano processati nello stesso batch, ma anche che vengano "spaccati" a cavallo tra un batch e l'altro.

Per questo motivo, la Pipeline di test (`evaluate`) non può calcolare subito le metriche, ma deve appoggiarsi a un buffer persistente tra batch.

### 1. I Buffer di Accumulo (`_accumulate_tile_predictions`)
All'inizio della valutazione vengono inizializzati due dizionari:
- `tile_predictions_buffer = defaultdict(list)`
- `tile_gt_buffer = {}`

Al termine della predizione di ogni batch, la funzione `_accumulate_tile_predictions` entra in gioco:
1. Itera sugli elementi del batch corrente.
2. Legge il metadato `original_idx` e `quadrant` (impostato nella Fase 1).
3. Salva la predizione (boxes, scores, labels) nella lista del `tile_predictions_buffer` usando `original_idx` come chiave.
4. Salva `labels_full`, il ground truth dell’immagine intera, in `tile_gt_buffer`. Se manca, il codice emette un warning e ripiega sulle label del tile: quel fallback non garantisce una valutazione corretta sulla griglia intera.

### 2. Esecuzione Trigger e Calcolo Metriche (`_aggregate_and_update_metrics`)
Sempre all'interno della stessa funzione, non appena il buffer per uno specifico `original_idx` raggiunge **4 elementi** (ovvero abbiamo elaborato tutto il mosaico dell'immagine), avviene uno "scarico":
```python
if len(tile_predictions_buffer[original_idx]) == 4:
    self._aggregate_and_update_metrics(original_idx, ...)
```

In questo metodo finale:
1. Viene invocata la funzione `aggregate_tile_predictions(tile_predictions, iou_threshold=0.5)` presentata nella Fase 2.
2. I risultati "ripuliti" e ricalcolati sulla mappa intera vengono finalmente passati a `self.val_evaluator.update(...)`, che si occupa di aggiornare le metriche (come la mAP) paragonandole al Ground Truth a griglia intera.
3. Il buffer per quell' `original_idx` viene svuotato (`del tile_predictions_buffer[original_idx]`) in modo da liberare la memoria occupata dalle predizioni.

### 3. Safety Flush finale (`_process_remaining_tiles`)
Se per caso il dataset contiene immagini incomplete (es. bug nel caricamento e un array si ferma a 3 quadranti), il ciclo principale finirebbe lasciandoli pendenti. La funzione `_process_remaining_tiles` è garantita per girare dopo la chiusura del loader, forzando l'aggregazione per ogni eventuale immagine frammentata rimasta nel buffer.

## Esito sperimentale: il tiling non ha migliorato RT-DETR

La procedura è stata implementata e valutata, ma non ha prodotto il miglioramento atteso per target piccoli. A soglia 0.01, il riferimento RT-DETR su immagine intera addestrato per 10 epoche ha raggiunto **0.357 mAP@50** in fusione, mentre la corrispondente configurazione con tiling ha ottenuto **0.151**. Anche dopo 25 epoche il tiling resta sotto il modello integrale confrontabile (0.302 contro 0.321).

| Configurazione | Epoche | $D_{RGB-IR}$ mAP@50 |
| --- | ---: | ---: |
| RT-DETR, immagine intera | 10 | 0.357 |
| RT-DETR, tiling 2×2 | 10 | 0.151 |
| RT-DETR, immagine intera | 25 | 0.321 |
| RT-DETR, tiling 2×2 | 25 | 0.302 |

Il confronto include l'inferenza sui quattro quadranti e la successiva rimappatura/NMS: il calo non dipende quindi dal fatto che in valutazione siano state ignorate parti dell'immagine.

Le spiegazioni più plausibili sono:

- **Perdita di contesto globale.** Ogni predizione vede solo un quarto della scena; relazioni con ambiente, scala e porzioni del soggetto oltre il bordo del tile vengono rimosse.
- **Minore frequenza di supervisione utile dei target.** In training viene processato un solo quadrante casuale per immagine. Un target il cui centro cade in uno specifico quadrante compare quindi soltanto quando viene estratto quel quadrante; rispetto all'immagine intera riceve meno esposizioni positive per immagine elaborata.
- **Effetti ai confini.** La regola di assegnazione per centro e il clipping possono fornire esempi parziali vicino ai bordi; inoltre training su un solo tile e inferenza aggregata su quattro tile non coincidono perfettamente.

I risultati supportano soprattutto l'ipotesi che, in questa configurazione, la perdita di contesto e la minore frequenza utile dei target superino il beneficio dell'aumento di scala apparente. Non isolano quantitativamente il contributo di ciascun fattore, né escludono varianti diverse del tiling (tile sovrapposti, campionamento target-aware o addestramento multi-tile).
