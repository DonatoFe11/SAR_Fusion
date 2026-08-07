# Integrazione YOLOv10 + FAM

> **Stato.** Tutti i risultati YOLO attualmente riportati sono singole run del
> protocollo storico, con `best.pt` scelto dalla validation ed early stopping a
> epoche diverse. Restano validi come sviluppo esplorativo, ma non possono
> stabilire se FAM migliori YOLO. Se YOLO rimane nelle conclusioni della tesi è
> necessario il confronto finale a cinque seed definito in fondo a questa nota.

## Contesto

Il progetto studia fusione RGB-IR per object detection in scenari SAR (Search
And Rescue). La campagna finale RT-DETR a cinque seed ha mostrato che FAM
standard migliora Additive di `+0.0700` mAP@50 medio, vincendo in tutti i seed.
SSJ non ha invece migliorato la media di FAM. I precedenti valori RT-DETR
`0.438` e `0.396` erano singole run e non sono più il riferimento finale.

I precedenti tentativi YOLO (`FusionConv` in `cfg/yolofusion-s.yaml`, `FusionTransformer` in `cfg/yolotrans-s.yaml`) fondevano RGB e IR al primissimo layer dello stem, dentro un'unica backbone. Tutte le convoluzioni successive operavano su feature già miscelate ma spazialmente disallineate, amplificando l'errore di parallasse a ogni downsampling.

L'architettura proposta replica il pattern vincente di RT-DETR: **dual backbone + FAM a ogni livello della piramide + fusione additiva dopo allineamento**.

## Decisioni architetturali

| Decisione | Scelta | Motivazione |
|-----------|--------|-------------|
| Versione YOLO | YOLOv10s | Già integrato con trainer, validator e modello custom. Il contributo studiato è FAM + dual backbone, non un confronto fra versioni YOLO. |
| Integrazione neck | Standalone forward | Ricostruito come forward separato per avere controllo sul routing delle feature fuse. Più pulito degli hook. |
| Tipo fusione | Additive (`rgb + ir_aligned`) | È il percorso con cui FAM ha migliorato Additive in tutti i cinque seed RT-DETR. Preserva i canali originali attesi dal neck YOLO. |
| FAM source | `FeatureAlignmentModule` da `rtdetr_fusion.py` | YOLO riusa direttamente la classe RT-DETR; Deformable DETR possiede un'implementazione separata con la stessa logica generale. |
| Dataset | `wisards_vis_ir.yaml` | Stesso `vis_ir` di RT-DETR (solo coppie sincronizzate). I file YAML di dataset stanno in `parameters/YOLO_datasets/`; il trainer cerca dalla root, quindi servono copie/symlink. |
| Classi | **1 classe (persona)** | `single_cls: True`, `nc=1`. Correzione rispetto al precedente setup multi-classe errato. |

## Architettura

```
Input (4ch: RGB[0:3] + IR[3])
       ↓                    ↓
RGB Backbone            IR Backbone
(YOLOv10-s, 3ch,        (YOLOv10-s, 1ch,
 pesi COCO)              primo Conv adattato da RGB)
       ↓                    ↓
  feat[4]  (P3, 128ch)   feat[4]  (P3, 128ch)
  feat[6]  (P4, 256ch)   feat[6]  (P4, 256ch)
  feat[10] (P5, 512ch)   feat[10] (P5, 512ch)
       │         FAM          │
       └──→ ir_aligned ←──────┘
                ↓
    Fused = rgb + ir_aligned  (additive fusion)
                ↓
         YOLO Neck (FPN+PAN)
         (Upsample → Concat → C2f → Conv → SCDown → Concat)
                ↓
         v10Detect Head
         (one2one + one2many)
```

### Due YAML, due responsabilità

`parameters/YOLO/30.yolov10-fam.yaml` è la configurazione dell'**esperimento**: seleziona il registry entry `yolov10_fusion_fam`, indica dataset, epoche, batch, ottimizzatore e la griglia dei parametri. Il riferimento

```yaml
model.params.cfg: cfg/yolov10-fusion-fam-s.yaml
```

porta invece al YAML di **architettura** `cfg/yolov10-fusion-fam-s.yaml`, che descrive il grafo YOLOv10-s: layer della backbone, neck FPN/PAN e testa `v10Detect`.

### Perché i punti di fusione sono `[4, 6, 10]`

Gli indici sono posizioni dei layer nel grafo della backbone, non il numero dei livelli di piramide. Il layer 3 effettua il downsampling a P3/8 e il `C2f` al layer 4 raffina la feature mantenendo la stessa risoluzione; analogamente 5→6 produce/raffina P4/16 e 7→10 produce/raffina P5/32. Per questo si fondono le uscite raffinate `4`, `6` e `10` (rispettivamente 128, 256 e 512 canali), che sono esattamente le tre feature laterali lette dal neck.

`C2f` è il blocco CSP (*Cross Stage Partial*) compatto di YOLO: raffina le feature senza cambiare la loro risoluzione spaziale. In `P3/8`, il valore 8 è lo stride rispetto all'immagine di input; per esempio, con input 640×640, P3/8 è 80×80.

### Neck FPN/PAN e routing delle feature fuse

Il neck è posto tra backbone e detection head. La parte FPN propaga contesto dall'alto verso il basso (P5 → P4 → P3) con `Upsample + Concat + C2f`; la parte PAN riporta le feature arricchite dal basso verso l'alto (P3 → P4 → P5) con `Conv/SCDown + Concat + C2f`.

Nel forward custom, `y[4]`, `y[6]` e `y[10]` contengono rispettivamente `RGB + IR_allineato`; il neck legge quindi feature già fuse nei suoi `Concat`, ad esempio `[-1, 6]` significa “output precedente e P4 fusa salvata al layer 6”, concatenati lungo i canali. Non somma una seconda volta RGB originale e feature fusa.

## File creati

### `sarfusion/models/yolo_fusion_fam.py` (~360 linee)

Classe `YOLOv10FusionFAM(nn.Module)`:

- **Costruttore**: carica cfg YAML (`yaml_model_load`), crea RGB backbone (ch=3) e IR backbone (ch=1, primo conv adattato via media canali), scopre i canali delle feature map ai livelli P3/P4/P5, costruisce 3 moduli FAM con supporto per freeze/SSJ/dropout, calcola lo stride della testa di detection
- **Forward**:
  - `forward(x)`: dispatcher — se `x` è dict chiama `loss()`, se è tensor chiama `predict()`
  - `predict(x, modality_mask)`: divide input 4ch in RGB + IR e usa una maschera esplicita `[RGB, IR]`. In full-fusion applica FAM e somma additiva a P3/P4/P5; in RGB-only usa soltanto le feature RGB; in IR-only usa soltanto le feature IR e bypassa FAM
  - i campioni privi di una modalità vengono esclusi dal relativo backbone, evitando che input azzerati aggiornino le statistiche BatchNorm
  - `loss(batch)`: calcola loss via `v10DetectLoss` compatibile con trainer ultralytics
- **Compatibilità**: si comporta come `BaseModel` ultralytics — `model.args` come `IterableSimpleNamespace`, `model.stride`, `model.names`, `model.task`, `model.loss()`

### `sarfusion/models/__init__.py`

Aggiunto `build_yolo_fusion_fam` al `MODEL_REGISTRY` con chiave `"yolov10_fusion_fam"`.

### `cfg/yolov10-fusion-fam-s.yaml`

Config YAML standard YOLOv10s (nc=1, backbone + head completi). Usato come blueprint dal costruttore per determinare struttura dei layer e canali. Il dual backbone gestisce internamente i canali di input.

### `parameters/YOLO/30.yolov10-fam.yaml`

Parametri esperimento con grid search 2 × 3 = 6 run:

| `use_fam` | `spatial_jitter_std` | Descrizione |
|-----------|---------------------|-------------|
| True | 0.0 | FAM senza SSJ |
| True | 0.5 | FAM con SSJ moderato |
| True | 1.0 | FAM con SSJ forte |
| False | 0.0 | Baseline: dual backbone senza allineamento |
| False | 0.5 | Duplicato funzionale della baseline: SSJ non viene applicato |
| False | 1.0 | Duplicato funzionale della baseline: SSJ non viene applicato |

Iperparametri: lr0=1e-4, lrf=1e-5, optimizer=AdamW, 200 epoche, patience=30, batch=16, imgsz=640, single_cls=True, `model.params.pretrained=True`, `parameters.pretrained=False`, augment_vis_ir=False, mosaic=0.0. Il primo flag abilita l'inizializzazione interna COCO del modello custom; il secondo disabilita il caricamento standard del trainer Ultralytics, evitando un secondo tentativo di pretraining.

## Metriche del modello

| Variante | Parametri | FAM params |
|----------|-----------|------------|
| No FAM | ~11.96M | 0 |
| Con FAM | ~15.49M | ~3.53M |

Stride: `[8, 16, 32]` — standard YOLOv10s.

## Bugfix durante execution test

### `sarfusion/data/wisard.py` (line 1015)

`adapt_ir2rgb()` restituisce una tupla `(rgb, new_ir)` ma `WiSARDYOLODataset.load_image()` chiamava `.permute()` direttamente sulla tupla. La pipeline YOLO via `WiSARDYOLODataset` non aveva mai caricato coppie sincronizzate prima di `vis_ir`, quindi il bug non si era mai manifestato. RT-DETR usa `WiSARDDataset`, una classe diversa con `__getitem__` separato — nessun impatto.

**Fix**: decomporre la tupla e concatenare RGB + IR lungo il canale prima di permute:
```python
im_vis, im_ir = adapt_ir2rgb(im_vis, im_ir)
im_ir = im_ir[:1] if im_ir.dim() == 3 else im_ir
im = torch.cat([im_vis, im_ir], dim=0).permute(1, 2, 0).numpy()
```

### `sarfusion/utils/plots.py` — plot_images() path tuple

Il validator chiama `plot_images` con path sotto forma di tuple `(vis_path, ir_path)` per il dataset `vis_ir`, ma il plotter si aspettava stringhe. Causava TypeError non fatali nei thread di plotting.

**Fix**: estrarre il primo elemento se path è una lista/tupla:
```python
p = paths[i][0] if isinstance(paths[i], (list, tuple)) else paths[i]
```

### `sarfusion/models/yolo_fusion_fam.py` — predict() (3ch warmup)

`final_eval()` in `yolo.py` carica il modello via `AutoBackend` che fa warmup con input 3ch `(1, 3, 640, 640)`. `predict()` si aspettava sempre 4ch e crashava su `x[:, 3:4]` (tensore con 0 canali → Conv IR falliva).

**Fix**: `predict()` controlla `x.shape[1]`; se 3 canali, reindirizza a `_forward_single()` che esegue forward standard single-modality. Gli input a 4 canali seguono invece il percorso dual-backbone, eventualmente controllato dalla maschera esplicita delle modalità.

## Risultati grid — Single Class (SarYOLOSingleClass)

Mappatura cartelle → parametri (ordine di esecuzione grid search):

| Cartella | `use_fam` | `spatial_jitter_std` | Epoche | mAP50 |
|----------|-----------|---------------------|--------|-------|
| `YOLOv10-sc-FAM-Grid` | True | 0.0 | 40 | **0.4074** |
| `YOLOv10-sc-FAM-Grid2` | True | 0.5 | 59 | **0.2982** |
| `YOLOv10-sc-FAM-Grid3` | True | 1.0 | 68 | **0.3331** |
| `YOLOv10-sc-FAM-Grid4` | False | 0.0 | 52 | **0.3776** |
| `YOLOv10-sc-FAM-Grid5` | False | 0.5 | 52 | **0.3776** |
| `YOLOv10-sc-FAM-Grid6` | False | 1.0 | 52 | **0.3776** |

### Nota su Grid4/5/6

Grid4, Grid5 e Grid6 hanno file `results.csv` **byte-per-byte identici**. Questo non è un bug del grid search engine: con `seed: 42` fisso e `deterministic: True` nel config, e `spatial_jitter_std` che non ha alcun effetto quando `use_fam=False` (`fam_modules` sono `nn.Identity`, il parametro non viene mai letto), tre run con la stessa architettura effettiva, lo stesso seed e lo stesso ordine di dati producono deterministicamente la stessa traiettoria di training — nessuna sorpresa. Lo stesso fenomeno, identico nella causa, era già stato osservato nel blocco multi-classe precedente (Grid11/12/13 identici in ogni metrica). La griglia `use_fam=False` per SSJ 0.0/0.5/1.0 è quindi rappresentata correttamente da un'unica run indipendente effettiva (Grid4); Grid5/Grid6 non aggiungono informazione e possono essere ignorate nei confronti successivi senza perdita.

### Analisi risultati

Il modello funziona in single-class e raggiunge performance comparabili all'esperimento multi-classe precedente:

- **Miglior risultato: Grid (FAM, SSJ=0.0) con 0.4074 mAP50**
- **No FAM (Grid4): 0.3776 mAP50** — superiore a Grid2 e Grid3 con FAM
- Grid2 (FAM+SSJ=0.5, 0.2982) e Grid3 (FAM+SSJ=1.0, 0.3331) sono inferiori a Grid, suggerendo che SSJ su single-class peggiora la convergenza del FAM
- Il modello senza FAM (0.3776) batte RT-DETR base (0.357) di circa 0.02

**Confronto con l'esperimento precedente (multi-classe):**

| Variante | Multi-classe mAP50 | Single-class mAP50 |
|----------|-------------------|-------------------|
| FAM, SSJ=0.0 | 0.348 | **0.4074** |
| FAM, SSJ=0.5 | 0.385 | 0.2982 |
| FAM, SSJ=1.0 | 0.396 | 0.3331 |
| No FAM | 0.413 | 0.3776 |

La variante FAM+SSJ=0.0 in single-class (0.4074) supera la corrispondente multi-classe (0.348), suggerendo che l'addestramento su una sola classe ben definita aiuta l'allineamento FAM. Il No FAM scende da 0.413 a 0.3776. Entrambi gli esperimenti caricavano internamente i pesi COCO di `jameslahm/yolov10s`; il `pretrained=False` presente al livello principale dei parametri riguardava soltanto il trainer Ultralytics. La differenza va quindi attribuita al diverso task/dataset o alla variabilità delle singole run, non alla presenza o assenza del pretraining COCO.

## Verifiche superate

1. **Forward pass (eval)**: `torch.randn(1, 4, 640, 640)` → `{'one2one': Tensor, 'one2many': Tensor}` corretto
2. **Forward pass (train)**: output `{'one2many': [3 liste feature], 'one2one': [3 liste feature]}` con shape corrette
3. **Loss computation**: `model(batch_dict)` → `(loss_tensor, loss_items_vector)` senza errori
4. **build_model() via registry**: funzionante sia per variante FAM che no-FAM
5. **No NaN**: verificato con `spatial_jitter_std=0.5` in training mode
6. **IR backbone adaptation**: primo Conv adattato da 3ch→1ch via media canali

## Come eseguire

```bash
# Singola run
python main.py experiment --parameters parameters/YOLO/30.yolov10-fam.yaml --yolo

# Solo creazione file di configurazione (senza esecuzione)
python main.py experiment --parameters parameters/YOLO/30.yolov10-fam.yaml --yolo --only-create

# Feature-level modal gating: 20% IR-only / 20% RGB-only / 60% fusion
python main.py experiment --parameters parameters/YOLO/31.yolov10-fam-modal-dropout.yaml --yolo
```

La strategia è selezionabile nel file dei parametri:

```yaml
modal_dropout: true
modal_dropout_strategy: feature  # feature | input
modal_dropout_probs: [0.2, 0.2, 0.6]
```

Con `feature`, la maschera campionata esclude la backbone non disponibile e
bypassa FAM in IR-only. Con `input`, i canali della modalità assente vengono
azzerati ma la maschera passata al modello resta `[1, 1]`: entrambe le backbone
e l'eventuale FAM rimangono quindi attivi, riproducendo il comportamento
input-level storico.

## Verifica diagnostica del FAM (fam_alignment_check.py) sui checkpoint single-class

Eseguita su Grid, Grid2, Grid3 (le uniche 3 run FAM indipendenti), 3 campioni del val set, stessa metodologia già validata sui checkpoint multi-classe (hook per nome-classe `FeatureAlignmentModule`, PCA condivisa, offset convertiti in pixel-immagine reali per stride).

### Offset in pixel immagine (mean/median/p90/max, aggregati su 3 campioni)

| livello | stride | Grid (ssj=0.0) | Grid2 (ssj=0.5) | Grid3 (ssj=1.0) |
|---|---|---|---|---|
| 0 (P3) | 8 | 9.5 / 9.3 / 16.3 / 34.9 px | 10.4 / 9.5 / 19.5 / 28.6 px | 12.4 / 10.9 / 24.5 / 51.6 px |
| 1 (P4) | 16 | 21.9 / 15.3 / 55.2 / 114.5 px | 38.2 / 30.3 / 89.9 / 148.0 px | 29.7 / 23.1 / 59.7 / 149.4 px |
| 2 (P5) | 32 | 42.7 / 37.2 / 82.3 / 151.7 px | 55.2 / 35.7 / 133.0 / 248.1 px | 70.6 / 58.8 / 151.0 / 291.9 px |

L'ampiezza cresce con SSJ in modo più consistente qui che sui checkpoint multi-classe (dove l'andamento era invertito) — direzionalmente più vicino a quanto documentato su RT-DETR (SSJ aumenta l'ampiezza media), anche se qui l'effetto è più marcato.

### Collasso della varianza spaziale (`std_spaziale(fam(ir))` vs `std_spaziale(ir)` pre-FAM, livello P5)

| | Grid (ssj=0.0) | Grid2 (ssj=0.5) | Grid3 (ssj=1.0) |
|---|---|---|---|
| std_spaziale ir | 0.326 | 0.337 | 0.267 |
| std_spaziale fam(ir) | 0.073 | 0.054 | 0.013 |
| rapporto ir/fam | **4.5×** | **6.3×** | **20.1×** |

Il collasso peggiora monotonicamente con SSJ, confermato anche visivamente nella PCA (`PCA(FAM(IR))` uniforme/piatta a tutti e 3 i checkpoint, stesso pattern già documentato sui checkpoint multi-classe).

### Uniformità spaziale dell'offset (`uniformity_ratio`, livello P5)

| Grid (ssj=0.0) | Grid2 (ssj=0.5) | Grid3 (ssj=1.0) |
|---|---|---|
| 0.246 | 0.338 | 0.274 |

Nessun andamento monotono chiaro con SSJ su questa metrica specifica.

### Interpretazione: collasso e beneficio netto non sono in contraddizione

Un punto importante da chiarire esplicitamente: **il checkpoint con mAP migliore (Grid, 0.4074) mostra comunque un collasso spaziale sostanziale (4.5× a P5)** — il collasso non implica automaticamente un effetto dannoso. La `mean` di `fam(ir)` è sistematicamente diversa da zero e coerente a tutti i livelli (-0.33, -0.41, -0.25 per Grid) — un bias sistematico sommato al ramo RGB può contribuire utilmente anche portando poca variazione spaziale punto-per-punto, funzionando più come un termine di calibrazione/normalizzazione che come un contributo content-adaptive ricco.

Allo stesso tempo, **la severità del collasso non spiega da sola l'ordinamento del mAP tra le 3 run FAM**: Grid3 (collasso peggiore, 20×) supera Grid2 (collasso minore, 6.3×) in mAP (0.3331 vs 0.2982). Con una sola run per configurazione (nessun seed multiplo, a differenza del protocollo a 5 seed usato per DefDETR/DINO), non è possibile distinguere con sicurezza una relazione causale da rumore di singola inizializzazione tra Grid2 e Grid3 — da trattare come limite aperto, non da forzare in una narrativa pulita che i dati non supportano fino in fondo.

**Ipotesi candidata per spiegare perché la SSJ danneggia il FAM su YOLO invece di aiutarlo (come su RT-DETR)**: YOLO è un'architettura interamente convoluzionale, che aggrega informazione da un vicinato spaziale fisso e rigido — un rumore che disallinea quel vicinato (SSJ) danneggia direttamente la corrispondenza locale su cui si basa la fusione. RT-DETR/DefDETR usano attenzione (deformable o meno), che non ha lo stesso vincolo di corrispondenza pixel-per-pixel e può quindi assorbire meglio un disallineamento indotto in training. Ipotesi architetturalmente motivata, coerente con l'inversione di segno dell'effetto SSJ osservata tra le due famiglie di modelli, ma non dimostrata sperimentalmente in modo diretto in questo progetto — da trattare come spiegazione candidata plausibile, non come conclusione stabilita.



## Test di robustezza mono-modale sui checkpoint single-class (vis/ir)

Eseguito con `eval_yolo_modalities.py` sui 4 checkpoint indipendenti (Grid, Grid2, Grid3, Grid4 — Grid5/Grid6 esclusi, duplicati di Grid4), split test, stesso protocollo già validato (`rect=True`, nc derivato dall'head reale non dall'attributo bookkeeping `model.nc`). Sanity check ripetuto anche su `vis_ir` in questo stesso giro: `map_50` ricalcolato (0.4073/0.2993/0.3335/0.3774) coincide con i numeri ufficiali di training (0.4074/0.2982/0.3331/0.3776) a meno di arrotondamenti floating point, confermando l'affidabilità dei numeri su vis/ir riportati sotto.

### Risultati (mAP50, tutte e 3 le modalità)

| Grid | FAM | SSJ | vis_ir | vis | ir |
|---|---|---|---|---|---|
| Grid | True | 0.0 | 0.4074 | 0.1713 | 0.0 |
| Grid2 | True | 0.5 | 0.2982 | 0.1955 | 0.0 |
| Grid3 | True | 1.0 | 0.3331 | 0.1538 | ~2.8×10⁻⁶ |
| Grid4 | False | 0.0 | 0.3776 | **0.0782** | ~3.1×10⁻⁸ |

### Scoperte

1. **Su VIS-only, tutte e 3 le configurazioni FAM battono il baseline in modo netto e consistente** (0.15-0.20 vs 0.078, un fattore ~2×), indipendentemente dalla SSJ — un pattern più pulito e robusto di quello osservato in modalità fusion, dove solo Grid (ssj=0.0) batteva chiaramente il baseline.

2. **Interpretazione che riconcilia questo risultato con il collasso spaziale già documentato, senza contraddirlo**: in VIS-only il canale IR è azzerato in input; la backbone IR, per via di bias/BatchNorm, produce comunque un output non-zero ma privo di informazione reale sulla scena (un "rumore di fondo" strutturato, content-independent). Senza FAM questo contributo degenere viene sommato **grezzo e senza filtro** al ramo RGB, disturbandolo in modo sostanziale (mAP crolla a 0.078). Con FAM, lo stesso segnale passa attraverso la deformable conv che — coerentemente con lo smoothing/collasso spaziale già misurato in fusion (vedi sezione precedente) — tende ad attenuarlo/appiattirlo prima della somma. La stessa tendenza del FAM a comprimere la varianza spaziale dell'IR agisce quindi da **filtro protettivo quando l'IR in ingresso è degenere**, mentre in fusion (dove l'IR porta informazione reale) la stessa compressione **limita** quanto contributo utile arriva al resto della rete. Non è un comportamento contraddittorio del FAM — è lo stesso meccanismo, con effetto netto opposto a seconda che l'input IR sia informativo o rumoroso.

3. **IR-only resta sostanzialmente inutilizzabile per tutti e 4 i checkpoint** (0.0 per Grid/Grid2, dell'ordine di 10⁻⁶/10⁻⁸ per Grid3/Grid4, trascurabile in tutti i casi), confermando quanto già osservato sui checkpoint multi-classe: l'assenza di modal dropout nel regime di training YOLO produce zero robustezza appresa alla mancanza del canale RGB, indipendentemente da FAM o SSJ.

## Modal dropout input-level 60/20/20 — esperimento completato

Questa sezione documenta la **prima implementazione** del modal dropout YOLO, precedente al feature-level gating. Durante il training, per ogni coppia RGB-IR veniva campionata una delle tre condizioni:

- 20% IR-only: canali RGB azzerati;
- 20% RGB-only: canale IR azzerato;
- 60% full-fusion: entrambe le modalità inalterate.

La modalità assente veniva azzerata soltanto nell'immagine di input. Entrambe le backbone continuavano quindi a essere eseguite e le loro feature venivano comunque combinate. I valori riportati sotto descrivono specificamente questo regime **input-level** e non il feature-gating implementato successivamente.

### Confronto con il training senza modal dropout

Valori mAP50 sul test set:

| Variante | Fusion senza → con dropout | RGB-only senza → con dropout | IR-only senza → con dropout |
|---|---:|---:|---:|
| FAM, SSJ=0.0 | 0.407 → 0.329 | 0.171 → 0.158 | 0.000 → 0.054 |
| FAM, SSJ=0.5 | 0.298 → 0.379 | 0.196 → 0.179 | 0.000 → 0.018 |
| FAM, SSJ=1.0 | 0.333 → 0.334 | 0.154 → 0.166 | ~0.000 → 0.013 |
| No FAM | 0.378 → 0.334 | 0.078 → 0.131 | ~0.000 → 0.023 |

### Interpretazione

Il modal dropout input-level non ha prodotto una robustezza mono-modale sufficiente a compensare il costo sulla modalità completa:

1. **IR-only migliora tecnicamente rispetto allo zero**, ma rimane nell'intervallo 0.013–0.054 mAP50, troppo basso per rappresentare una capacità operativa utile.
2. **RGB-only non mostra un vantaggio convincente** nelle varianti FAM: SSJ=0.0 e 0.5 peggiorano leggermente, mentre SSJ=1.0 migliora solo marginalmente. Il miglioramento No FAM da 0.078 a 0.131 non colma il divario rispetto alle configurazioni FAM senza dropout.
3. **La prestazione fusion diminuisce** per FAM SSJ=0.0 e No FAM. SSJ=1.0 resta sostanzialmente invariato. Il guadagno di SSJ=0.5, da 0.298 a 0.379, parte invece da una baseline senza dropout particolarmente debole e non costituisce da solo evidenza di un beneficio generale.

La spiegazione più plausibile è che azzerare una modalità in input non equivalga a rimuoverne il ramo. Convoluzioni, bias e BatchNorm possono generare feature non nulle anche da un tensore azzerato. In IR-only, inoltre, il FAM riceve come riferimento feature RGB prive di informazione reale e tenta comunque di allineare l'IR rispetto a esse. Il neck viene quindi addestrato su fusioni contenenti contributi artificiali dei sensori formalmente assenti.

La conclusione sperimentale è pertanto:

> Il modal dropout input-level 60/20/20 non trasferisce automaticamente a YOLO i benefici di robustezza osservati con RT-DETR.

### Follow-up: feature-level gating

Il nuovo esperimento mantiene le stesse probabilità 20/20/60 ma applica una maschera esplicita alle feature:

| Condizione | Percorso verso il neck |
|---|---|
| RGB-only | sole feature RGB; backbone IR esclusa |
| IR-only | sole feature IR; backbone RGB esclusa e FAM bypassato |
| Full-fusion | `RGB + FAM(IR, RGB)`, invariato rispetto alla fusione originale |

Il training delle quattro configurazioni è stato completato. Le nuove run sono `Grid5`–`Grid8`, perché `Grid`–`Grid4` nella stessa directory appartengono all'esperimento input-level precedente.

#### Risultati completi

Le valutazioni standalone sono state eseguite sullo stesso test set in full-fusion, RGB-only e IR-only. Gli output completi sono salvati in `modal_eval_feature_gating/`.

| Run | Variante | Fusion | RGB-only | IR-only |
|---|---|---:|---:|---:|
| `Grid5` | FAM, SSJ=0.0 | 0.2965 | 0.0750 | 0.0356 |
| `Grid6` | FAM, SSJ=0.5 | 0.3335 | 0.1454 | 0.0308 |
| `Grid7` | FAM, SSJ=1.0 | 0.2358 | 0.0638 | 0.0295 |
| `Grid8` | No FAM | **0.3469** | **0.1756** | **0.0616** |

`Grid8` domina tutte le altre configurazioni feature-gated in ognuna delle tre modalità. Nel regime con gating corretto, nessuna variante FAM fornisce quindi un vantaggio misurabile; SSJ=1.0 è la configurazione più penalizzata.

#### Confronto tra i tre regimi

Ogni cella riporta `senza dropout / input-level / feature-gating`:

| Variante | Fusion | RGB-only | IR-only |
|---|---:|---:|---:|
| FAM, SSJ=0.0 | **0.407** / 0.329 / 0.297 | **0.171** / 0.158 / 0.075 | ~0.000 / **0.054** / 0.036 |
| FAM, SSJ=0.5 | 0.298 / **0.379** / 0.334 | **0.196** / 0.179 / 0.145 | ~0.000 / 0.018 / **0.031** |
| FAM, SSJ=1.0 | 0.333 / **0.334** / 0.236 | 0.154 / **0.166** / 0.064 | ~0.000 / 0.013 / **0.029** |
| No FAM | **0.378** / 0.334 / 0.347 | 0.078 / 0.131 / **0.176** | ~0.000 / 0.023 / **0.062** |

Il risultato più informativo è il confronto No FAM:

- rispetto al dropout input-level, il feature-gating migliora fusion di 0.013, RGB-only di 0.045 e IR-only di 0.039;
- rispetto al training senza dropout, sacrifica 0.031 in fusion ma guadagna 0.098 in RGB-only e 0.062 in IR-only.

Questo conferma che eliminare realmente il ramo assente è tecnicamente preferibile al semplice azzeramento dell'immagine, almeno nella variante senza FAM. Il miglioramento non è però sufficiente a raggiungere una robustezza IR utile: il massimo YOLO è 0.0616 mAP50, ancora molto distante dai valori IR-only circa 0.18–0.26 documentati per le varianti RT-DETR con FAM correttamente ottimizzato.

La conclusione finale è quindi duplice:

1. **La modifica strutturale era corretta**: nel caso No FAM migliora contemporaneamente tutte le modalità rispetto al dropout input-level.
2. **Il limite sostanziale di YOLO rimane**: la capacità IR-only resta troppo bassa e il guadagno non compensa pienamente la perdita rispetto al miglior modello fusion senza dropout.

Limitatamente alle singole run storiche, la massima accuratezza fusion è stata
osservata con FAM senza dropout (`0.407`), mentre il miglior compromesso
feature-gated è stato `Grid8` senza FAM (`0.347 / 0.176 / 0.062`). Questi valori
non giustificano una scelta finale del modello: le due conclusioni provengono
da protocolli diversi e non includono variabilità tra seed.

Come per la grid precedente, ogni configurazione è rappresentata da una sola run con seed 42. Le differenze più ampie e il dominio di `Grid8` sulle tre modalità sono evidenze interne a questa ablation, non una stima della variabilità tra seed.

> **Nota di riproducibilità.** La modalità predefinita `--modality auto` dell'attuale `eval_yolo_modalities.py` usa i nuovi percorsi feature-gated. Per riprodurre sui checkpoint storici il comportamento mono-modale input-level della tabella precedente occorre forzare `--modality fusion` sul dataset RGB-only o IR-only, così che entrambi i rami vengano eseguiti nonostante il padding a zero.

## Validazione e selezione del checkpoint

La verifica standalone di `Grid8` sullo split `val` ha prodotto `mAP50-95=0.00353`
e `mAP50=0.01563`, valori coerenti con quelli registrati durante il training. La
bassa mAP di validazione non è quindi dovuta a un errore del validator YOLO.

L'origine principale della discrepanza tra validation e test è il contenuto dello
split. In `wisard.py`, `VAL_VIS_IR` parte dagli indici `[4, 5, 6, 8]`, ma le
sequenze 4, 5 e 6 vengono eliminate dal filtro `NO_LABELS`. Rimane soltanto la
sessione FHL all'indice 8: **273 immagini, 184 background e 148 bounding box**.
Il test set usa invece tre sequenze MtErie. La validation è dunque piccola,
sbilanciata verso i background, limitata a una singola sessione e appartenente a
un dominio diverso dal test. I conteggi riportati nei commenti dei file YAML non
descrivono più il contenuto effettivo dopo il filtraggio.

Questa criticità **non è specifica di YOLO**. Anche RT-DETR e Deformable DETR
hanno mostrato metriche basse sul medesimo validation set; ciò indica un limite
del protocollo condiviso, non una sensibilità dimostrata esclusivamente per
YOLO. Il confronto tra architetture rimane internamente coerente perché usa gli
stessi split, ma la selezione dell'epoca migliore può essere rumorosa e poco
rappresentativa per tutti i modelli.

Esiste comunque una differenza nel criterio di selezione:

- YOLO salva `best.pt` ed esegue l'early stopping sulla fitness
  `0.1 * mAP50 + 0.9 * mAP50-95`, calcolata soltanto in modalità full-fusion;
- RT-DETR e Deformable DETR monitorano `map`, cioè la mAP50-95.

Per le run feature-gated YOLO, RGB-only e IR-only non partecipano quindi alla
scelta del checkpoint. Per `Grid8` il miglior fitness è stato raggiunto intorno
all'epoca 16 e il training è terminato all'epoca 46 con patience 30; anche le
altre run mostrano arresti coerenti con lo stesso criterio.

Non conviene sostituire retroattivamente lo split soltanto per YOLO. I risultati
storici sul test comune restano documentabili come esplorativi, dichiarando la
validation ridotta e il checkpoint `best`. Per il confronto finale non si
costruisce una nuova validation ad hoc: si usa un orizzonte fissato a priori,
senza early stopping, e si valuta `last.pt`, come nel protocollo RT-DETR.

## Protocollo finale YOLO da eseguire

La domanda finale non è quale delle vecchie grid abbia prodotto il massimo
numero, ma se il FAM standard migliori una baseline YOLO multimodale sotto lo
stesso regime di robustezza alle modalità mancanti.

### Confronto obbligatorio se YOLO resta nelle conclusioni

| Configurazione | FAM | SSJ | Modal Dropout | Run |
|---|---:|---:|---|---:|
| YOLO dual-backbone Additive | no | 0 | feature gating 20/20/60 | 5 |
| YOLO dual-backbone + FAM | sì | 0 | feature gating 20/20/60 | 5 |

Entrambe devono usare:

- seed appaiati `40–44`;
- processi Python separati;
- 200 epoche fisse, coerenti con l'orizzonte e la schedulazione degli YAML
  originari;
- validation non usata per early stopping o scelta del checkpoint;
- `patience` disattivata;
- test del solo `last.pt`;
- valutazione dello stesso checkpoint in VIS, IR e VIS+IR;
- tabella dei valori per seed, media, mediana, deviazione standard, min–max, IC
  95% e differenze appaiate.

Il protocollo è ora implementato in due configurazioni separate:

- `parameters/YOLO/yolov10_additive_protocol.yaml`;
- `parameters/YOLO/yolov10_fam_protocol.yaml`.

Entrambe espandono i seed `40–44` in processi isolati, disabilitano early
stopping con `patience: 0`, mantengono 200 epoche fisse e impostano
`test_checkpoint: last`. `WisardTrainer.final_eval()` accetta esplicitamente
`best` o `last` e valuta sul test split soltanto la scelta dichiarata. Un test
automatico verifica sia il selettore sia l'equivalenza del protocollo fra i
due YAML, salvo l'attivazione del FAM.

### SSJ e vecchie grid

Non è necessario ripetere fusion-only, input-level dropout o SSJ 1.0. FAM +
SSJ 0.5 diventa una terza configurazione a cinque seed soltanto se la tesi
mantiene una domanda esplicita sulla trasferibilità di SSJ. La campagna finale
RT-DETR non mostra un vantaggio medio di SSJ, quindi aggiungerlo automaticamente
al protocollo YOLO non è giustificato.

La diagnostica interna YOLO esistente usa inoltre tre soli campioni di
validation. Va ripetuta sui cinque checkpoint FAM finali e sugli stessi trenta
campioni/sessioni usati per RT-DETR soltanto dopo il training confermativo. Le
statistiche devono essere aggregate per checkpoint, evitando di trattare celle
spaziali come repliche indipendenti.

Se il confronto finale confermerà una robustezza IR insufficiente, eventuali
loss ausiliarie mono-modali o teste specifiche costituiscono un nuovo lavoro,
non un'aggiustamento da scegliere guardando gli attuali risultati di test.

## Note tecniche

- **Caricamento pesi COCO**: il costruttore carica `jameslahm/yolov10s` via `YOLOv10WiSARD.from_pretrained()` quando `model.params.pretrained=True`. Di 1478 chiavi totali: 847 vengono caricate (backbone RGB + IR + neck parziale); le 631 mancanti appartengono principalmente alla testa `v10Detect`, perché il checkpoint COCO usa `nc=80` mentre l'esperimento usa `nc=1`. Il flag separato `parameters.pretrained=False` appartiene al trainer Ultralytics e non disabilita questo caricamento interno.
- **Inizializzazione FAM**: il checkpoint COCO non contiene moduli FAM, quindi `offset_conv` e `deform_conv` non ricevono pesi preaddestrati. `offset_conv` è inizializzato a zero, per cui gli offset iniziali sono nulli e la mask iniziale vale 0.5; `deform_conv` mantiene invece la sua inizializzazione standard. Offset nulli non rendono dunque il FAM un'identità matematica perfetta: preservano la geometria iniziale, ma la feature IR viene già filtrata dalla deformable convolution.
- La loss `v10DetectLoss` usa `TaskAlignedAssigner` e `BboxLoss` con DFL. Durante training combina entrambi i rami `one2many` e `one2one`; `one2many` viene selezionato da `_compute_stride()` solo per misurare gli stride `[8, 16, 32]`, non per escludere l'altro ramo dalla loss.
- Se la GPU ha poca VRAM, ridurre `batch` da 16 a 8.
- Compatibile con `torch.cuda.amp`.
