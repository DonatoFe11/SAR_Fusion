# Strategia di Tiling (Sliding Window Inference)

Questo documento spiega l'approccio di tiling (o patch splitting) implementato nell'architettura. Questa strategia è divisa in livelli logici che coprono il ciclo di vita del dato, dalla lettura per il dataloader fino all'aggregazione finale.

## Fase 1: Generazione e Caricamento dei Tile (Dataset Layer)

**File di riferimento**: [sarfusion/data/wisard.py](../sarfusion/data/wisard.py)

L'immagine e le annotazioni vengono manipolate "alla base" del framework, durante la chiamata `__getitem__` del Dataset. L'obiettivo è lavorare su patch di dimensioni ridotte (es. $320 \times 320$) a partire dall'immagine ad alta risoluzione (es. $640 \times 640$).

### Dinamiche di Training vs Testing
Il tiling si comporta diversamente a seconda della fase di esecuzione:
- **Training** (`use_tiling=True`): Il metodo `__getitem__` seleziona casualmente un solo quadrante per l'immagine (`quadrant = random.randint(0, 3)`). Questo agisce come **Data Augmentation**, costringendo il modello a specializzarsi nell'identificazione di piccoli dettagli ed evitando l'overfitting.
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
   - **Traslazione**: Essendo l'immagine originariemente ancorata in $(0,0)$, per allineare i box occorre sottrarre gli offset (`x_min - x_off`, `y_min - y_off`).
   - **Clipping**: Un oggetto potrebbe avere il centro in un tile ma invadere lo spazio di un altro. Tramite costrutti del tipo `min(bbox_width, tile_size - new_x_min)`, il riquadro viene clippato perfettamente per non eccedere le dimensioni limitate del quadrante.

### Resize Conservativo (`resize((640, 640))`)
Per garantire il corretto funzionamento geometrico di questo approccio, durante il `__getitem__` delle singole modalità (sia RGB che IR), l'immagine in PIL originale viene ridimensionata forzatamente a $640 \times 640$ prima di subire le operazioni descritte da `_get_tile`. Questo assicura proporzioni ideali, in modo che la divisione per due (320x320) non produca mai resti frazionari.

## Fase 2: Geometria e Fusione delle Predizioni (Logic Layer)

**File di riferimento**: [sarfusion/data/tile_aggregation.py](../sarfusion/data/tile_aggregation.py)

Una volta che il modello ha processato i frammenti (tile) e generato le predizioni (con coordinate relative ai $320 \times 320$), è necessario ricostruire l'ingombro totale per valutare correttamente l'immagine a risoluzione intera ($640 \times 640$).

### 1. Rimappatura delle Coordinate (`remap_tile_boxes_to_original`)
Le predizioni in output sono normalizzate rispetto al tile (i valori $x, y, w, h$ vanno da 0 a 1). La funzione di rimappatura ripristina queste coordinate rispetto all'immagine originaria:
- Poiché la larghezza/altezza del tile è esattamente la metà, le larghezze e le altezze predette vengono scalate per un fattore $0.5$.
- Anche i *centri* dei box vengono scalati per $0.5$, ma in base al quadrante originario subiscono anche una "traslazione normalizzata" di $0$ o $0.5$. Ad esempio, per un oggetto trovato nel tile in basso a destra (quadrante 3), al centro $X$ e al centro $Y$ predetti scalati viene aggiunto $0.5$, riposizionando l'oggetto nella metà inferiore e destra dell'immagine.

### 2. Eliminare i duplicati con la NMS (`aggregate_tile_predictions`)
Durante un approccio a ritagli (specialmente nel testing), se un oggetto reale si trova fisicamente a cavallo del confine tra due quadranti, il modello individuerà una parte dell'oggetto nel "Tile Top" e l'altra parte dell'oggetto nel "Tile Bottom". Ri-mappando i due box assieme nell'immagine originale, ci ritroveremmo con due predizioni sovrapposte che denotano lo stesso target empirico.

Per risolvere e fondere le inferenze utilizziamo la **NMS**.

**Cos'è la Non-Maximum Suppression (NMS)?**
La NMS è un algoritmo standard della computer vision che serve per **filtrare doppi rilevamenti dello stesso oggetto**. Ecco come lavora:
1. Prende tutte le predizioni sull'immagine intera e le ordina partendo da quella con il **punteggio di confidenza più alto** (quella di cui la rete è più sicura).
2. Seleziona questa predizione top come "Definitiva" (es. confidenza $0.98$).
3. Misura l'**IoU** (Intersection over Union, l'area di sovrapposizione geometrica) tra questo box definitivo e tutti gli altri box dello stesso tipo.
4. Se uno degli altri box ha una profonda sovrapposizione spaziale (es. un'IoU maggiore di una soglia prefissata, tipo $0.5$), l'algoritmo deduce che stanno guardando lo stesso medesimo oggetto e "sopprime" la scatola con confidenza minore.
5. Si passa al prossimo box con confidenza maggiore rimasto e si ripete fino a pulire del tutto la mappa.

In [sarfusion/data/tile_aggregation.py](../sarfusion/data/tile_aggregation.py), i box vengono prima uniti, convertiti da formati centrali (cx, cy, w, h) in formati assoluti (x1, y1, x2, y2) richiesti da PyTorch, ed infine passati alla funzione integrata `torchvision.ops.nms` per produrre il set compatto di rilevamenti aggregati.

## Fase 3: Orchestrazione e Accumulo nel Loop (Pipeline Layer)

**File di riferimento**: [sarfusion/experiment/run.py](../sarfusion/experiment/run.py)

Il Datalaoder fornisce i dati in **batch** (blocchi) casuali o sequenziali. Con il parametro `test_all_tiles=True`, una singola immagine genera 4 tile distinti. Dipendendo dalla dimensione del batch (es. batch size = 8 o 16), è possibile che i 4 frammenti di un'immagine vengano processati nello stesso batch, ma anche che vengano "spaccati" a cavallo tra un batch e l'altro.

Per questo motivo, la Pipeline di test (`evaluate`) non può calcolare subito le metriche, ma deve appoggiarsi a un sistema di "parcheggio" asincrono.

### 1. I Buffer di Accumulo (`_accumulate_tile_predictions`)
All'inizio della valutazione vengono inizializzati due dizionari:
- `tile_predictions_buffer = defaultdict(list)`
- `tile_gt_buffer = {}`

Al termine della predizione di ogni batch, la funzione `_accumulate_tile_predictions` entra in gioco:
1. Itera sugli elementi del batch corrente.
2. Legge il metadato `original_idx` e `quadrant` (impostato nella Fase 1).
3. Salva la predizione (boxes, scores, labels) nella lista del `tile_predictions_buffer` usando `original_idx` come chiave.
4. Salva il Ground Truth originale dell'immagine intera (e non dei singoli patch) in `tile_gt_buffer`.

### 2. Esecuzione Trigger e Calcolo Metriche (`_aggregate_and_update_metrics`)
Sempre all'interno della stessa funzione, non appena il buffer per uno specifico `original_idx` raggiunge **4 elementi** (ovvero abbiamo elaborato tutto il mosaico dell'immagine), avviene uno "scarico":
```python
if len(tile_predictions_buffer[original_idx]) == 4:
    self._aggregate_and_update_metrics(original_idx, ...)
```

In questo metodo finale:
1. Viene invocata la funzione `aggregate_tile_predictions(tile_predictions, iou_threshold=0.5)` presentata nella Fase 2.
2. I risultati "ripuliti" e ricalcolati sulla mappa intera vengono finalmente passati a `self.val_evaluator.update(...)`, che si occupa di aggiornare le metriche (come la mAP) paragonandole al Ground Truth a griglia intera.
3. Il buffer per quell' `original_idx` viene svuotato (`del tile_predictions_buffer[original_idx]`) in modo da liberare la preziosa memoria RAM/VRAM della macchina.

### 3. Safety Flush finale (`_process_remaining_tiles`)
Se per caso il dataset contiene immagini incomplete (es. bug nel caricamento e un array si ferma a 3 quadranti), il ciclo principale finirebbe lasciandoli pendenti. La funzione `_process_remaining_tiles` è garantita per girare dopo la chiusura del loader, forzando l'aggregazione per ogni eventuale immagine frammentata rimasta nel buffer.
