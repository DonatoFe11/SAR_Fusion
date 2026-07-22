# DINO Fusion per RGB-IR (versione original)

Questo documento descrive l'implementazione del modello DINO (DETR with Improved deNoising anchOr boxes) adattato per scenari multi-modali (RGB + Infrarosso) nella sua versione base.
Il codice di riferimento è: [sarfusion/models/dino_fusion_original.py](../sarfusion/models/dino_fusion_original.py).

---

## 1. Relazione tra DINO e Deformable DETR

Leggendo il file ci si accorge subito di un dettaglio interessante: **DINO eredita direttamente le classi di `DeformableDetr` da Hugging Face.** Non ci sono classi `DinoModel` nel codice, bensì `DeformableDetrModel`.

**Perché questa scelta?**
DINO non è un'architettura completamente nuova rispetto a Deformable DETR. DINO introduce principalmente tre miglioramenti che si applicano però *esclusivamente durante l'addestramento*:

1. **Contrastive DeNoising Training**: aggiunta di rumore controllato ai bounding box reali (Ground Truth) da far predire alla rete, per accelerare la convergenza.
2. **Mixed Query Selection**: inizializzazione non casuale delle *object query* per renderle più forti fin dall'inizio.
3. **Look Forward Twice**: miglioramento sull'aggiornamento dei gradienti tra i layer del decoder.

Siccome in fase di inference questi artifici spariscono, **l'architettura fisica di DINO è esattamente identica a quella di Deformable DETR**. Di conseguenza è possibile incapsulare i framework l'uno dentro l'altro, beneficiando però di eventuali pesi pre-addestrati con ricette DINO.

---

## 2. Il problema: DINO accetta solo RGB

Il dataset **WiSARD** fornisce immagini a doppio canale — **RGB** (visibile, 3 canali) e **IR** (infrarosso termico, 1 canale). DINO standard accetta solo immagini RGB. L'obiettivo di questo file è costruire una versione "fusion" che accetti entrambe le modalità e le combini in un'unica rappresentazione.

---

## 3. Architettura: doppio backbone + fusione

### Il problema dei canali di input

Il primo layer Conv2d di una ResNet ha shape `[out_channels, in_channels, kH, kW]`. Per RGB servono 3 canali di input, per IR ne basta 1. Gli strati successivi sono identici — cambia solo il primo layer perché è l'unico che tocca i pixel raw.

```
RGB backbone — primo layer:
  input:   [B,  3, H, W]
  kernel:  [out_ch, 3, kH, kW]   ← tre "finestre" R, G, B

IR backbone — primo layer:
  input:   [B,  1, H, W]
  kernel:  [out_ch, 1, kH, kW]   ← una sola finestra di intensità termica
```

Questo si ottiene creando due configurazioni separate con `copy.deepcopy(config)` e impostando `backbone_kwargs["in_chans"]` a 3 e a 1 rispettivamente, prima di istanziare i due `DeformableDetrConvEncoder`.

### Output del backbone a più scale

Entrambi i backbone producono feature map a 4 livelli di scala (multi-scale detection, utile per oggetti di dimensioni diverse). Per un'immagine `800×600`:

```
Livello 0 (C3):  [B,  512, 100, 75]   stride 8
Livello 1 (C4):  [B, 1024,  50, 38]   stride 16
Livello 2 (C5):  [B, 2048,  25, 19]   stride 32
Livello 3 (extra):[B,  256,  13, 10]  proiettato da C5
```

### Positional embeddings

I positional embedding del backbone sono **sinusoidali e deterministici**: calcolati da formule fisse sulla griglia spaziale `[H/l, W/l]`, senza parametri appresi. `rgb_pos` e `ir_pos` sono quindi numericamente identici — dipendono solo dalla forma della feature map, non dal contenuto. Nel forward si usa `rgb_pos` per convenzione.

`self.position_embedding = position_embeddings` viene salvato come attributo per compatibilità con il framework HuggingFace, che si aspetta che qualsiasi backbone esponga quell'attributo.

---

## 4. La fusione: concatenazione canale + proiezione espressiva

### Il modulo `channel_fusion`

```python
self.channel_fusion = nn.ModuleList([
    nn.Sequential(
        nn.Conv2d(channels * 2, channels, kernel_size=1),
        nn.GroupNorm(32, channels),
        nn.ReLU(inplace=True)
    )
    for channels in self.intermediate_channel_sizes
])
```

Un blocco separato per ciascuno dei 4 livelli di scala. Ogni blocco fa tre cose:

1. **Conv2d 1×1** — comprime da `2C` a `C` canali. Impara come pesare e combinare le due modalità canale per canale.
2. **GroupNorm(32, channels)** — normalizza dividendo i canali in 32 gruppi indipendenti. Stabile con `batch_size=1`, al contrario della BatchNorm che richiede batch grandi per stimare correttamente media e varianza.
3. **ReLU** — introduce non-linearità.

### Il forward nel caso RGB+IR (4 canali)

```python
concat_feat = torch.cat([rgb_feat, ir_feat], dim=1)  # [B, 2C, H, W]
fused_feat  = self.channel_fusion[level_idx](concat_feat)  # [B, C, H, W]
```

Per ogni livello di scala, le due feature map vengono concatenate lungo la dimensione dei canali e poi proiettate a `C`. Esempio per il livello 0:

```
rgb_feat:   [1,  512, 100, 75]
ir_feat:    [1,  512, 100, 75]
concat:     [1, 1024, 100, 75]   ← cat(dim=1)
fused:      [1,  512, 100, 75]   ← Conv1×1 + GN + ReLU
```

Il forward gestisce anche i casi in cui una sola modalità è disponibile (modal dropout durante il training):

```python
if num_channels == 1:   # solo IR → ir_backbone
if num_channels == 3:   # solo RGB → rgb_backbone
if num_channels == 4:   # entrambi → fusione
```

---

## 5. Gerarchia delle classi

```
DeformableDetrForObjectDetection   (HuggingFace)
└── DinoFusionForObjectDetection   (questo file)
      │
      └── model: DeformableDetrModel   (HuggingFace)
            └── DinoFusionModel        (questo file)
                  │
                  └── backbone: DinoFusionBackbone   (questo file)
                        ├── rgb_backbone  (ResNet50, in_chans=3)
                        ├── ir_backbone   (ResNet50, in_chans=1)
                        └── channel_fusion × 4 livelli
```

Le teste di classificazione (quale classe?) e di regressione (dove si trova?) appartengono a `DinoFusionForObjectDetection` — ereditano da HuggingFace e lavorano sugli output del decoder, non sulle feature map del backbone.

---

## 6. Caricamento dei pesi pre-addestrati

### Cosa viene caricato

`from_pretrained` scarica un checkpoint `SenseTime/deformable-detr` pre-addestrato su COCO e copia i pesi dove le chiavi coincidono (`strict=False` ignora le mismatch).

```
Caricato da pretrained:
  rgb_backbone.conv_encoder    ← pesi ResNet50 ImageNet/COCO
  ir_backbone.conv_encoder     ← pesi adattati (vedi sotto)
  encoder, decoder             ← pesi transformer COCO
  input_proj                   ← proiezioni backbone→transformer

Inizializzato random e appreso da zero:
  channel_fusion               ← non esiste nel pretrained
  class_labels_classifier      ← reinizializzata (1 classe invece di 80)
```

### Adattamento dei pesi per l'IR backbone

Non esistono modelli IR pre-addestrati su larga scala — tutti i grandi checkpoint usano ImageNet che è RGB. La soluzione è fare la media dei pesi RGB:

```python
for key, value in rgb_backbone_state.items():
    if value.dim() == 4 and value.shape[1] == 3:
        ir_backbone_state[key] = value.mean(dim=1, keepdim=True)
```

Solo il primo layer Conv2d ha `shape[1] == 3` (i 3 canali di input). La media sui canali produce un filtro `[out_ch, 1, kH, kW]` che mantiene le stesse capacità di rilevare bordi e texture del modello RGB, adatto all'immagine termica che non ha colore ma ha la stessa struttura di gradienti di intensità.

---

## 7. Flusso dati completo (batch_size=1, immagine 800×600)

```
Input [1, 4, 800, 600]
  │
  ├─ [:, :3] → rgb_backbone → 4 feature map  [1, 512/1024/2048/256, H/l, W/l]
  └─ [:, 3:] → ir_backbone  → 4 feature map  [1, 512/1024/2048/256, H/l, W/l]
                                    │
                     per ogni livello: cat(dim=1) → Conv1×1 + GN + ReLU
                                    │
                              4 feature map fuse  [1, 512/1024/2048/256, H/l, W/l]
                                    │
                         input_proj → tutte a d_model=256
                                    │
                         flatten + concat → [1, ~10000, 256]  (sequenza token)
                                    │
                         Encoder transformer (6 layer) → [1, ~10000, 256]
                                    │
                         Decoder transformer (6 layer, 300 query) → [1, 300, 256]
                                    │
                    ┌───────────────┴────────────────┐
              class head                        bbox head
           [1, 300, 2]                        [1, 300, 4]
        (person / no-object)              (cx, cy, w, h) normalizzati
```

---