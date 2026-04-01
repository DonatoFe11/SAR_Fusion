# DINO Fusion per RGB-IR

Questo documento esplora l'implementazione del modello DINO (DETR with Improved deNoising anchOr boxes) adattato per scenari multi-modali (RGB + Infrarosso).
Il codice di riferimento è: [sarfusion/models/dino_fusion.py](../sarfusion/models/dino_fusion.py).

## 1. Relazione tra DINO e Deformable DETR

Leggendo il file ci si accorge subito di un dettaglio interessante: **DINO eredita direttamente le classi di `DeformableDetr` da Hugging Face.** Non ci sono classi `DinoModel` nel codice, bensì `DeformableDetrModel`.

**Perché questa scelta?**
DINO non è un'architettura completamente nuova rispetto a Deformable DETR (come invece lo è RT-DETR). DINO introduce principalmente tre miglioramenti "State-Of-The-Art" che si applicano però *durante l'addestramento*:
1. **Contrastive DeNoising Training**: l'aggiunta di rumore controllato ai bounding box reali (Ground Truth) da far predire alla rete per accelerare enormemente la convergenza (Contrastive Denoising).
2. **Mixed Query Selection**: l'inizializzazione non casuale (ancore fisse o estratte dalle feature map) delle *object query* per renderle più forti.
3. **Look Forward Twice**: un miglioramento sull'aggiornamento dei gradienti tra i layer del decoder.

Siccome in fase di "Inference" (o di testing) questi artifici spariscono, **l'architettura fisica di DINO è esattamente identica a quella di Deformable DETR**. Di conseguenza, è possibile incapsulare i framework l'uno dentro l'altro beneficiando però di eventuali pesi iper-ottimizzati di tipo DINO.

## 2. Il cuore dell'Implementazione: Concatenazione Espressiva + FAM

Rispetto al paradigma leggerissimo e "povero" incontrato in `deformable_detr_fusion.py`, qui the l'implementazione usa due step potenti per la fusione.

### A. Feature Alignment Module (FAM) - Opzionale
Hai importato e riadattato il FAM basato su `DeformConv2d` (Visto in RT-DETR Fusion).
```python
if self.use_fam:
    ir_processed = self.fam_modules[level_idx](rgb_feat, ir_feat)
```
Se il parametro `use_fam` è attivo, i sensori RGB e Infrarosso non vengono semplicemente sommati. Il sensore RGB viene sfruttato per calcolare la geometria della scena e "stirare/distorcere" i pixel della feature map dell'Infrarosso in modo che si allineino perfettamente, correggendo sdoppiamenti causati dallo spaziatura fisica tra lente RGB e lente Termica sul drone.

La particolarità qui è la **Lazy Initialization**: le Deformable Convolutions non sanno a priori di quanti canali (`in_channels`) saranno fatti i livelli. Il tuo codice attende che arrivi il primissimo batch di foto nel metodo `forward()` per estrarre la forma (`rgb_feat.shape[1]`) e costruire al volo i moduli FAM!

### B. Fusione Espressiva (Espressivity Channel Fusion)
Anziché proiettare linearmente il risultato con un umile layer Convoluzionale $1\times 1$ (come fa Deformable DETR), hai usato un approccio molto più robusto e profondo per combinare $RGB$ e $IR\_allineato$:

```python
self.channel_fusion = nn.ModuleList([
    nn.Sequential(
        nn.Conv2d(channels * 2, channels, kernel_size=1),
        nn.GroupNorm(32, channels),  # Stabilità!
        nn.ReLU(inplace=True)
    )
    for channels in self.intermediate_channel_sizes
])
```
1. **La Convoluzione lineare** mischia i canali e comprime da $2C$ a $C$.
2. **La GroupNorm (Normazione a Gruppi)**: È un tocco magistrale. A differenza della *BatchNorm* (che si rompe se dai in pasto alla rete 1 o 2 foto alla volta perché la varianza impazzisce), la `GroupNorm` distribuisce i canali in $32$ canestri separati e li normalizza indipendentemente dal batch. Questo permette di addestrare questo enorme modello DINO senza per forza affittare decine di GPU costosissime per tenere alti i batch.
3. **La ReLU**: Infine, dà una forte piega non-lineare ai neuroni per esaltare i valori positivi del calore ed eliminare i falsi negativi (zeri).