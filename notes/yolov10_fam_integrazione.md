# Integrazione YOLOv10 + FAM

## Contesto

Il progetto studia fusione RGB-IR per object detection in scenari SAR (Search And Rescue). I modelli migliori sono basati su RT-DETR con Feature Alignment Module (FAM), che ha raggiunto **0.438 mAP@50** sul dataset fusion e ha dimostrato di produrre offset di allineamento reali (2-5 px medi, verificati cross-sessione via PCA).

I precedenti tentativi YOLO (`FusionConv` in `cfg/yolofusion-s.yaml`, `FusionTransformer` in `cfg/yolotrans-s.yaml`) fondevano RGB e IR al primissimo layer dello stem, dentro un'unica backbone. Tutte le convoluzioni successive operavano su feature già miscelate ma spazialmente disallineate, amplificando l'errore di parallasse a ogni downsampling.

L'architettura proposta replica il pattern vincente di RT-DETR: **dual backbone + FAM a ogni livello della piramide + fusione additiva dopo allineamento**.

## Decisioni architetturali

| Decisione | Scelta | Motivazione |
|-----------|--------|-------------|
| Versione YOLO | YOLOv10s | Già integrato (WisardTrainer, WisardValidator, YOLOv10WiSARD). Il contributo scientifico è FAM + dual backbone, non la versione YOLO. Upgrade a v11/v26 banale dopo validazione. |
| Integrazione neck | Standalone forward | Ricostruito come forward separato per avere controllo sul routing delle feature fuse. Più pulito degli hook. |
| Tipo fusione | Additive (`rgb + ir_aligned`) | Ha performato meglio su RT-DETR (0.438). Preserva i canali originali attesi dal neck YOLO. |
| FAM source | `FeatureAlignmentModule` da `rtdetr_fusion.py` | Classe identica tra RT-DETR e Deformable DETR, senza dipendenze architettura-specifiche. |
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
  - `predict(x)`: divide input 4ch in RGB + IR, esegue backbone duali, applica FAM e additive fusion a P3/P4/P5, forward del neck FPN+PAN + v10Detect head
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

**Fix**: `predict()` ora controlla `x.shape[1]`; se 3 canali, reindirizza a `_forward_single()` che esegue forward standard single-modality. Training e inferenza 4ch non sono toccati.

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

# Ripetizione della stessa grid con modal dropout 20% IR / 20% RGB / 60% fusion
python main.py experiment --parameters parameters/YOLO/31.yolov10-fam-modal-dropout.yaml --yolo
```

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

## Prossimi passi

1. **Diagnosticare il bug della val mAP** (`WisardTrainer`/`WisardValidator`, discrepanza nota tra val e test sullo stesso checkpoint) — resta aperto e non affrontato, rilevante per la fiducia nella selezione del checkpoint "best" usato in tutte le analisi sopra.

## Note tecniche

- **Caricamento pesi COCO**: il costruttore carica `jameslahm/yolov10s` via `YOLOv10WiSARD.from_pretrained()` quando `model.params.pretrained=True`. Di 1478 chiavi totali: 847 vengono caricate (backbone RGB + IR + neck parziale); le 631 mancanti appartengono principalmente alla testa `v10Detect`, perché il checkpoint COCO usa `nc=80` mentre l'esperimento usa `nc=1`. Il flag separato `parameters.pretrained=False` appartiene al trainer Ultralytics e non disabilita questo caricamento interno.
- **Inizializzazione FAM**: il checkpoint COCO non contiene moduli FAM, quindi `offset_conv` e `deform_conv` non ricevono pesi preaddestrati. `offset_conv` è inizializzato a zero, per cui gli offset iniziali sono nulli e la mask iniziale vale 0.5; `deform_conv` mantiene invece la sua inizializzazione standard. Offset nulli non rendono dunque il FAM un'identità matematica perfetta: preservano la geometria iniziale, ma la feature IR viene già filtrata dalla deformable convolution.
- La loss `v10DetectLoss` usa `TaskAlignedAssigner` e `BboxLoss` con DFL. Durante training combina entrambi i rami `one2many` e `one2one`; `one2many` viene selezionato da `_compute_stride()` solo per misurare gli stride `[8, 16, 32]`, non per escludere l'altro ramo dalla loss.
- Se la GPU ha poca VRAM, ridurre `batch` da 16 a 8.
- Compatibile con `torch.cuda.amp`.
