# RT-DETR RGB-IR: backbone, FAM e trasferimento dei pesi

> **Versione descritta.** Questa nota spiega il percorso RT-DETR v1 standard in [`rtdetr_fusion.py`](../sarfusion/models/rtdetr_fusion.py): P3–P5, `current_dcnv2`, gate disabilitati e dropout/jitter nulli. I frammenti sono estratti didattici semplificati, non una copia integrale delle firme o dei rami correnti. SSJ aggiunge rumore agli offset solo in training quando `spatial_jitter_std > 0`; risultati e motivazione sono in [fam_lazy_init_behavior.md](fam_lazy_init_behavior.md).

> **Risultati aggiornati.** La scelta finale è FAM standard senza SSJ. Le
> singole run storiche sono state sostituite dal confronto a cinque seed in
> [rtdetr_reproducibility.md](rtdetr_reproducibility.md); SSJ non ha mostrato un
> miglioramento medio rispetto al FAM standard.

## Feature Alignment Module (FAM)
Il modulo FAM è progettato per allineare le feature map IR con quelle RGB prima di fonderle. Questo è cruciale perché le due modalità (RGB e IR) possono essere disallineate a causa di differenze nella geometria, prospettiva o distorsioni ottiche.

---
### Costruttore `__init__`

Parto dal costruttore del FAM, che contiene due layer: `Conv2d` per predire offset e mask, `DeformConv2d` per trasformare le feature IR.

---

### `def __init__(self, in_channels, freeze=False, spatial_jitter_std=0.0):`
Inizializza il modulo definendo i pesi che dovrà imparare (o congelare).
- `in_channels`: È il numero di canali delle feature map in input. Ad esempio, se siamo in un livello profondo della backbone, in_channels potrebbe essere 256.
- `freeze`: Se impostato a True, congela i gradienti del layer. Il FAM resta una trasformazione convoluzionale fissa, non un'identità.
- `spatial_jitter_std`: Deviazione standard del rumore SSJ sugli offset. Con il valore predefinito `0.0` SSJ è disattivato.

### `super().__init__()`
Chiama il costruttore della classe padre (`nn.Module`). È standard in PyTorch per far sì che il modulo venga registrato correttamente.

---

### Layer 1: `self.offset_conv = nn.Conv2d(...)`
Questo è un normale layer convoluzionale standard di PyTorch, ma **non serve a estrarre feature visive**. Serve a **"predire come deformare"** la griglia della convoluzione successiva.

Come è fatto internamente:
- **`in_channels * 2`**: È il numero di canali in input. È `*2` perché nel `forward` concateno la map RGB e la map IR lungo l'asse dei canali (e.g. $256 + 256 = 512$).
- **`27`**: È il numero di canali in output. **Perché proprio 27?**
  Questa convoluzione deve produrre i parametri per un kernel $3 \times 3$ deformabile. Un kernel $3 \times 3$ ha 9 celle (o punti).
  - Per ognuno dei 9 punti, la rete deve predire uno spostamento lungo $x$ ($\Delta x$) e uno lungo $y$ ($\Delta y$). Quindi $9 \times 2 = 18$ canali (gli "offsets").
  - Per ognuno dei 9 punti, la rete deve predire anche un "peso modulatore" (la `mask`), compreso tra 0 e 1, che decide quanto quel punto debba essere 'acceso' o 'spento'. Quindi altri $9$ canali.
  - Totale canali in uscita: $18 + 9 = 27$.
- **`kernel_size=3`** e **`padding=1`**: Configurazione classica che mantiene inalterate le dimensioni spaziali di altezza ($H$) e larghezza ($W$) in output. I pesi interni sono un tensore `[27, 2C, 3, 3]`, cioè `[canali_output, canali_input, altezza_kernel, larghezza_kernel]`. Gli ultimi due `3` sono quindi la finestra spaziale $3\times3$ su $H$ e $W$, non una dimensione che scorre lungo i canali. Per produrre ciascuno dei 27 canali di output, la convoluzione applica un filtro $3\times3$ separato a ognuno dei $2C$ canali in input e ne somma i contributi.

**Esempio sul passaggio attraverso la profondità.** Con batch size 1 e $C=20$, la feature concatenata ha forma `[1, 40, 80, 80]`: i canali `0:20` sono RGB e `20:40` sono IR. Per calcolare un singolo valore, ad esempio `out[0, k, y, x]`, il kernel non scorre prima sui 20 canali RGB e poi sui 20 canali IR. Resta centrato nella stessa posizione spaziale `(y,x)`, prende una finestra $3\times3$ da **ciascuno** dei 40 canali e ne calcola una somma pesata. In forma compatta:

$$
out_{k,y,x} = b_k +
\sum_{c=0}^{19}\langle W^{RGB}_{k,c},F^{RGB}_{c,y\pm1,x\pm1}\rangle+
\sum_{c=0}^{19}\langle W^{IR}_{k,c},F^{IR}_{c,y\pm1,x\pm1}\rangle.
$$

Quindi il campo ricettivo del kernel attraversa tutta la profondità dei canali in un unico calcolo, mentre scorre soltanto su altezza e larghezza. Non accoppia automaticamente `RGB[c]` con `IR[c]`: eventuali relazioni fra canali delle due modalità devono essere apprese dai pesi della convoluzione.

---

### Layer 2: `self.deform_conv = DeformConv2d(...)`
Questo **non** è un layer convoluzionale standard, ma un layer speciale di `torchvision.ops`.

Come è fatto internamente:
- A differenza di una Conv2D standard che ha una griglia fissa (es: una griglia $3 \times 3$ quadrata), la Deformable Conv campiona i punti dell'immagine non necessariamente in una griglia regolare, ma **nei punti indicati dagli offsets** calcolati dal layer precedente.
- Ha anch'essa dei pesi apprendibili (come una conv normale), con dimensione tipica `[in_channels, in_channels, 3, 3]`.
- I parametri d'ingresso sono:
  - **`in_channels`**: Canali in input (in questo caso la mappa IR da far passare).
  - **`in_channels`**: Canali in output (restituisce una mappa allineata con gli stessi canali).
  - **`kernel_size=3`** e **`padding=1`**: Di base è un kernel $3 \times 3$ centrato sul pixel, ma come detto sopra, prima di moltiplicare per i pesi, campionerà i valori a posizioni deformate.

---

## Offset iniziali nulli e congelamento
Successivamente i pesi e i bias della `offset_conv` vengono inizializzati a `0`:
```python
nn.init.constant_(self.offset_conv.weight, 0)
nn.init.constant_(self.offset_conv.bias, 0)

if freeze:
    for param in self.parameters():
        param.requires_grad = False
```
**Nel modulo isolato**, queste istruzioni producono offset nulli e mask 0.5; la DCNv2 mantiene pesi propri e non è un’identità.

**Nel detector completo**, il successivo `post_init()` di Hugging Face reinizializza il predittore del FAM standard. L’inizializzazione effettiva storica non ha quindi offset necessariamente nulli né mask necessariamente 0.5. Le varianti identity ripristinano esplicitamente la propria inizializzazione dopo `post_init`; l’ablation [zero-offset](rtdetr_fam_zero_offset_stage_a.md) azzera soltanto le prime 18 righe dopo il trasferimento pretrained.

`freeze=True` disabilita i gradienti di tutti i parametri FAM; non impedisce l’inizializzazione successiva o il caricamento dei pesi. Congelare il modulo non garantisce dunque offset nulli. Anche con pesi congelati, offset dipendenti dall’input e l’eventuale SSJ possono variare fra forward.

---

## Il metodo `forward` della classe FAM
Nel metodo `forward`, la rete prende in input le feature map RGB e IR, le concatena, e poi calcola gli offset e le maschere per deformare la mappa IR.

```python
concat = torch.cat([rgb_feat, ir_feat], dim=1)  # [B, 2C, H, W]
```
La concatenazione non sovrappone né somma i valori RGB e IR: li dispone in due blocchi contigui sull'asse dei canali. Se ogni feature map ha $C$ canali, `concat` contiene prima `RGB[0:C]` e poi `IR[C:2C]`. A ogni posizione spaziale, `offset_conv` riceve comunque la finestra $3\times3$ di tutti i $2C$ canali e usa pesi distinti per il blocco RGB e per il blocco IR. Può quindi apprendere, dalla loss di detection, un campo di offset basato su pattern congiunti delle due modalità; non esegue però un matching o una correlazione esplicita fra esse.

```python
out = self.offset_conv(concat)  # [B, 27, H, W]
```
Il tensore passa attraverso il layer che abbiamo spiegato prima (offset_conv). Il risultato ha 27 canali.

```python
offset = out[:, :18, :, :]  # [B, 18, H, W]
mask = torch.sigmoid(out[:, 18:, :, :])  # [B, 9, H, W]
```
Qui il tensore viene splittato:
1. **`offset`**: I primi 18 canali rappresentano coppie interlacciate ($\Delta y$, $\Delta x$) per i 9 punti del kernel 3x3.
2. **`mask` (Modulation Scalars)**: Gli ultimi 9 canali passano in una funzione `sigmoid` in modo che il risultato sia limitato tra 0 e 1. Questi valori fanno da "moltiplicatori di importanza". Se la rete calcola che un certo offset finisce su un pixel rumoroso o non utile temporaneamente, la mask lo spinge verso lo 0, ignorandolo.

```python
ir_aligned = self.deform_conv(ir_feat, offset, mask)
return ir_aligned
```
Infine, la `DeformConv2d` viene applicata **esclusivamente su `ir_feat`**, ma guidata dagli `offset` e `mask` calcolati guardando *entrambe* le immagini. Questo fissa anche la direzione dell'allineamento: RGB resta nel proprio sistema di coordinate e `ir_aligned` deve produrre, alla posizione $p$, informazione IR utile da sommare a `rgb_feat[p]`. In seguito il codice esegue infatti `rgb_feat + ir_aligned`. Per allineare RGB a IR bisognerebbe invece applicare la deformable convolution a `rgb_feat` e lasciare IR come riferimento.

Non esiste una supervisione diretta del campo di offset: la direzione è imposta dal flusso dei tensori, mentre i valori degli offset vengono appresi dalla loss di detection. L'output è una nuova mappa feature termica (`ir_aligned`), i cui punti di campionamento possono essere "tirati" o "spinti" spazialmente verso il riferimento RGB. Il modulo apprende un allineamento utile al task, senza garantire una corrispondenza perfetta per ogni posizione o la vera trasformazione fisica fra sensori.

---

## Riepilogo:
1. Viene impostata la `offset_conv` che prenderà le feature RGB+IR fuse assieme per calcolare una mappa densa di trasformazioni ($18$ di shift e $9$ di maschera).
2. Viene definita la `deform_conv` che rappresenta l'operazione che userà effettivamente la mappa di trasformazioni appena calcolata (applicandola sull'IR) per ottenere una nuova mappa IR riallineata alle feature RGB.

Dunque, `offset_conv` prende in input la concatenazione delle feature RGB e IR e produce i parametri di deformazione, mentre `deform_conv` prende in input solo la feature map IR e applica la deformazione per riallinearle con le feature RGB usando gli offset appena calcolati.

<br><br>

## Fusion Backbone (`RTDetrFusionBackbone`)
Ho raccolto l’estrazione e la fusione delle feature in questa classe. Gestisce due estrattori separati (uno per l'RGB e uno per l'IR) e implementa la logica di parallelismo e, infine, di fusione delle feature map (con o senza il FAM).

---

### Scelte di implementazione
1. **La Backbone**: In PyTorch/HuggingFace, l'architettura RT-DETR usa un `RTDetrConvEncoder` (che incapsula reti per l'estrazione visiva, come ResNet50 o PPLCNet) come backbone. Noi stiamo istanziando *due copie identiche* di questo encoder: una per RGB e una per IR.
2. **Pesi Indipendenti**: Strutturalmente le due backbone sono cloni (stessi strati e canali), ma non sono la stessa rete in memoria (nessun *weight sharing*). All'inizio dell'addestramento i pesi sono quasi uguali, ma proseguendo evolveranno in modo indipendente: la backbone RGB si specializzerà su texture/colori, quella IR sulle firme termiche.
3. **Pesi Pre-addestrati**: Il modello di base di partenza è `PekingU/rtdetr_r50vd`. È un modello pre-addestrato sul dataset COCO (immagini RGB a 3 canali). Per questo motivo è essenziale l'adattamento da 3 a 1 canale sulla porta infrarossi in fase di inizializzazione.

---

### Costruttore `__init__`

```python
def __init__(self, config: RTDetrConfig, use_fam: bool = False,
             freeze_fam: bool = False, ir_dropout_rate: float = 0.0,
             spatial_jitter_std: float = 0.0):
    super().__init__()
```
Riceve la configurazione standard di RT-DETR e i flag delle varianti sperimentali: `use_fam` abilita l'allineamento, `freeze_fam` congela il FAM, `ir_dropout_rate` abilita lo Spatial Dropout e `spatial_jitter_std` abilita SSJ. Con i valori predefiniti è attiva la fusione base senza FAM, dropout o jitter.

```python
    # RGB backbone (standard)
    rgb_cfg = copy.deepcopy(config)
    rgb_cfg.num_channels = 3
    self.rgb_backbone = RTDetrConvEncoder(rgb_cfg)
```
Crea la backbone per l'RGB. Clona la configurazione di base, forza esplicitamente il numero di canali in input a 3 e istanzia l'encoder standard di RT-DETR.

```python
    # IR backbone (1 channel)
    ir_cfg = copy.deepcopy(config)
    ir_cfg.num_channels = 1
    self.ir_backbone = RTDetrConvEncoder(ir_cfg)

    self._adapt_ir_backbone()
```
Fa esattamente la stessa cosa per l'IR, ma imposta i canali in ingresso a 1. Dopodiché chiama una funzione fondamentale (`_adapt_ir_backbone`) per sistemare i pesi iniziali.

```python
    self.use_fam = use_fam
    self.freeze_fam = freeze_fam
    self.ir_dropout_rate = ir_dropout_rate
    self.spatial_jitter_std = spatial_jitter_std
    if self.use_fam:
        feature_channels = getattr(config, "encoder_in_channels", None)
        self.fam_modules = nn.ModuleList(
            [FeatureAlignmentModule(ch, freeze=self.freeze_fam,
                                    spatial_jitter_std=self.spatial_jitter_std)
             for ch in feature_channels]
        )
    else:
        self.fam_modules = None
```
**Inizializzazione Eager (Immediata)**: A differenza di implementazioni passate che usavano la "lazy initialization" (causando problemi con il tracciamento dei gradienti negando gli update al modulo FAM), qui i moduli vengono istanziati **in modo esplicito (eagerly)**. Leggendo i numeri di canale da `config.encoder_in_channels`, la rete crea da subito tante copie del modulo quanti sono i livelli estratti, risolvendo il problema dell'ottimizzatore. Inoltre, `freeze_fam` permette di congelare esplicitamente i pesi del FAM; la trasformazione risultante resta fissa, non è un allineamento casuale aggiornato dal training.

---

### Metodo `_adapt_ir_backbone`

Questo metodo risolve un problema pratico: vogliamo usare pesi pre-addestrati da un modello standard RGB (che si aspetta 3 canali in ingresso), ma la nostra backbone IR riceve in input 1 solo canale (grayscale/termico). 

```python
def _adapt_ir_backbone(self):
    for module in self.ir_backbone.modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == 3:
            module.weight = nn.Parameter(
                module.weight.mean(dim=1, keepdim=True)
            )
            module.in_channels = 1
```
Il codice cerca la prima convoluzione della rete (quella originariamente configurata per 3 canali di input). Prende i suoi pesi, che hanno dimensione `[out_channels, 3, kernel_size, kernel_size]`, e **fa la media (mean)** lungo la dimensione dei canali di input (`dim=1`). Il risultato diventa di dimensione `[out_channels, 1, kernel_size, kernel_size]`. In questo modo l'energia della convoluzione originaria è preservata e la rete IR ha una buona inizializzazione "warm start" invece di partire da parametri puramente casuali.

---

### Metodo `forward`
Questo metodo definisce il flusso esplorativo dei dati. Può operare in 3 modalità dinamiche controllando il numero di canali dell'input `c`.

```python
def forward(self, pixel_values: torch.FloatTensor, pixel_mask: Optional[torch.Tensor] = None):
    c = pixel_values.shape[1]
```

**Modalità 1 e 2: Singolo Sensore (RGB-only o IR-only)**
```python
    if c == 3:
        return self.rgb_backbone(pixel_values, pixel_mask)
    if c == 1:
        return self.ir_backbone(pixel_values, pixel_mask)
```
Con tre canali viene eseguita solo la backbone RGB; con uno solo la backbone IR. Entrambi i percorsi bypassano FAM e fusione. **Un tensore a quattro canali con una modalità azzerata segue invece il ramo fusion**: esegue entrambe le backbone e l’eventuale FAM. Le condizioni paired masked delle valutazioni non equivalgono quindi a passare direttamente uno o tre canali.

**Modalità 3: Fusione (RGB + IR)**
```python
    if c == 4:
        rgb_feats = self.rgb_backbone(pixel_values[:, :3], pixel_mask)
        ir_feats  = self.ir_backbone(pixel_values[:, 3:], pixel_mask)
```
Se passiamo 4 canali, assume la concatenazione. Smezza il tensore (primi 3 per RGB, ultimo logico per IR) e fa processare le metà in parallelo alle rispettive backbone. Il risultato (`rgb_feats` e `ir_feats`) sono **liste di feature map** a vari livelli di risoluzione.

*Quanti sono i livelli di queste feature map?*
In RT-DETR, grazie al design della backbone (es. ResNet), nel percorso standard vengono estratti **3 livelli di feature map** per gestire il multi-scala:
- **Livello 1 ($C3$)**: Scala spaziale $H/8$, $W/8$ (di solito 512 canali).
- **Livello 2 ($C4$)**: Scala spaziale $H/16$, $W/16$ (di solito 1024 canali).
- **Livello 3 ($C5$)**: Scala spaziale $H/32$, $W/32$ (di solito 2048 canali).
Quindi `rgb_feats` conterrà 3 tensori e il ciclo le attraversa tre volte, usando i moduli FAM già istanziati nel costruttore. Con `use_p2=true` i livelli diventano quattro (P2–P5), come descritto nella [nota P2](rtdetr_fam_p2_stage_a.md).

#### Ramo con FAM (Allineamento e Fusione)
```python
        if self.use_fam:
```
Qui uso l’inizializzazione *eager*. Nel costruttore, la rete ha già creato una `nn.ModuleList` contenente tre FAM (uno per livello di canale: 512, 1024, 2048). Nel `forward` richiamo i moduli già creati per ciascuno dei 3 livelli.

```python
            fused_feats = []
            for idx, ((r_feat, r_mask), (i_feat, _)) in enumerate(zip(rgb_feats, ir_feats)):
                # Allinea IR a RGB tramite FAM
                i_aligned = self.fam_modules[idx](r_feat, i_feat)
                
                # Fusione additiva
                fused_feats.append((r_feat + i_aligned, r_mask))
            return fused_feats
```
Cicla sui 3 livelli. Tramite il modulo FAM specifico di quel livello, trasforma la mappa termica (`i_feat`) usando RGB come guida. Infine applica una **Fusione Additiva** (`r_feat + i_aligned`): somma punto-a-punto i canali dei due tensori, combinando l'informazione ottica con la feature termica dopo l'allineamento appreso.

#### Ramo Base (Senza FAM)
```python
        else:
            fused_feats = []
            for (r_feat, r_mask), (i_feat, _) in zip(rgb_feats, ir_feats):
                fused_feats.append((r_feat + i_feat, r_mask))
            return fused_feats
```
Se `use_fam` è disattivato, esegue comunque la somma additiva per unire le informazioni delle due modalità, ma saltando il passaggio di allineamento deformabile spaziale. In entrambi i rami è inoltre applicato `Dropout2d` alla feature IR quando `ir_dropout_rate > 0`; il frammento sopra mostra il caso con dropout disattivato.

<br><br>

## RT-DETR Model e Object Detection Wrapper

Dopo aver definito la backbone che estrae le feature fuse, dobbiamo calare questa logica nell'architettura finale completa per poter fare object detection. Questo viene fatto sfruttando le classi messe a disposizione dalla libreria `transformers` di HuggingFace.

### Il Modello Intermedio (`RTDetrFusionModel`)
```python
class RTDetrFusionModel(RTDetrModel):
    def __init__(self, config: RTDetrConfig, use_fam: bool = False,
                 freeze_fam: bool = False, ir_dropout_rate: float = 0.0,
                 spatial_jitter_std: float = 0.0):
        super().__init__(config)
        self.backbone = RTDetrFusionBackbone(
            config, use_fam=use_fam, freeze_fam=freeze_fam,
            ir_dropout_rate=ir_dropout_rate,
            spatial_jitter_std=spatial_jitter_std)
        self.post_init()
```
Questa classe è estremamente compatta: eredita dal modello completo RT-DETR (senza le teste di detection). L'unico override che fa è sostituire la `self.backbone` originale (che si aspetterebbe solo un'immagine a 3 canali) con la nostra nuova e fiammante `RTDetrFusionBackbone`. 
Non serve sovrascrivere il metodo `forward()` perché il flusso di dati dopo la backbone (verso l'encoder Transformer e poi il decoder) è assolutamente identico a quello del modello RT-DETR originale.

---

### Object Detection Wrapper (`RTDetrFusionForObjectDetection`)
Questa è la classe principale a cui l'utente fa riferimento quando carica il modello per l'addestramento o per l'inferenza. Eredita da `RTDetrForObjectDetection` (il modello finale che include anche le *detection heads* per capire bounding box e classi).

#### Detection Heads
In architetture moderne come RT-DETR (o i vari YOLO), la rete è divisa concettualmente in due blocchi:
1. **Il Corpo (Backbone + Encoder/Decoder)**: Il suo scopo è capire "cosa c'è nell'immagine", elaborare le feature e trovare le correlazioni spaziali.
2. **Le Teste (Heads)**: Sono gli ultimissimi strati neurali (spesso semplici layer lineari `nn.Linear`). Trasformano le feature del decoder nelle predizioni di classe e box:
   - `class_embed`: produce logits di classificazione, convertiti in score mediante sigmoid nel postprocessing.
   - `bbox_embed`: Un layer che calcola coordinate. "Le coordinate \((x, y, w, h)\) del box sono queste".

#### Costruttore e collegamento delle teste
```python
def __init__(self, config: RTDetrConfig, use_fam: bool = False,
             freeze_fam: bool = False, ir_dropout_rate: float = 0.0,
             spatial_jitter_std: float = 0.0):
    # Trick: inizializziamo come RGB standard
    tmp_cfg = copy.deepcopy(config)
    tmp_cfg.num_channels = 3
    super().__init__(tmp_cfg)
```
Nel costruttore ho inizializzato la classe base con una configurazione RGB a tre canali, così da creare i moduli standard. Ho poi sostituito il corpo con `RTDetrFusionModel`, che gestisce l’input a quattro canali e le due backbone.

```python
    # Salviamo le teste originali
    saved_class_embed = self.class_embed
    saved_bbox_embed = self.bbox_embed

    # Sostituiamo il modello 
    self.model = RTDetrFusionModel(
        config, use_fam=use_fam, freeze_fam=freeze_fam,
        ir_dropout_rate=ir_dropout_rate,
        spatial_jitter_std=spatial_jitter_std)

    # Ripristiniamo le teste nel decoder del nuovo corpo
    self.model.decoder.class_embed = saved_class_embed
    self.model.decoder.bbox_embed = saved_bbox_embed
    self.config = config
    self.use_fam = use_fam
    self.freeze_fam = freeze_fam
    self.ir_dropout_rate = ir_dropout_rate
    self.spatial_jitter_std = spatial_jitter_std
```
Il passaggio avviene in tre fasi:

1. conservo i riferimenti alle teste create dalla classe base;
2. sostituisco `self.model` con `RTDetrFusionModel`;
3. collego gli stessi moduli al decoder del nuovo corpo.

Le assegnazioni non copiano i pesi: `self.class_embed` e i riferimenti nel
decoder puntano agli stessi moduli. In questo modo mantengo il collegamento
tra il wrapper di detection e le teste usate dal decoder.

---

### Trasferimento dei pesi: `from_pretrained`

Il metodo carica `PekingU/rtdetr_r50vd`, costruisce il detector fusion e trasferisce i tensori compatibili di encoder, decoder e teste. Le due backbone ricevono separatamente i pesi della backbone RGB pretrained. Per l’IR, i kernel con tre canali di ingresso vengono mediati su quell’asse (`mean(dim=1, keepdim=True)`). Il FAM non ha pesi COCO corrispondenti.

La gestione della testa dipende da `reuse_pretrained_class_head`:

- `false` (default API): il modello sorgente è già costruito con le label richieste; le dimensioni incompatibili sono gestite da `ignore_mismatched_sizes`, quindi una testa ridimensionata non conserva automaticamente i pesi COCO;
- `true` (protocollo RT-DETR finale): viene caricata la testa COCO originale e `copy_matching_pretrained_label_heads` trasferisce semanticamente le righe corrispondenti alle label richieste, inclusa `person`.

Con `use_p2=true`, il trasferimento rimappa esplicitamente i livelli pretrained P3–P5; non basta un caricamento basato sulla forma dei tensori. Vedi [P2](rtdetr_fam_p2_stage_a.md).

### Varianti opzionali del percorso corrente

`fam_variant` seleziona il FAM standard, bounded, identity, grid-sample o box-guided. La variante box-guided aggiunge una guida supervisionata al solo P3; non vale quindi per essa la descrizione del FAM standard come privo di supervisione diretta degli offset.

I tre meccanismi opzionali di gate richiedono FAM e sono mutuamente esclusivi:

- [reliability gating](rtdetr_fam_reliability_gate_stage_a.md): pesa separatamente RGB e IR allineata prima della somma;
- [RCRA](rtdetr_fam_residual_alignment_stage_a.md): seleziona localmente il residuo fra IR allineata e IR grezza;
- [controllo scalare](rtdetr_fam_scalar_alignment_control_stage_a.md): usa un unico coefficiente apprendibile per livello.

Le formule e gli esperimenti specifici sono nelle rispettive note. Il flusso seguente assume che queste opzioni siano disabilitate.

<br><br>

## Esempio del flusso dei tensori

Per seguire il forward considero il caso standard con FAM attivo e input RGB-IR a 640 pixel.

### 1. Input del Wrapper
Passiamo l'input al nostro wrapper `RTDetrFusionForObjectDetection`:
- **Dimensione iniziale**: Entra un tensore `pixel_values` di dimensione `[Batch, 4, 640, 640]`. I primi 3 canali sono i pixel RGB, l'ultimo canale è quello IR.

Il wrapper passa il tensore a `self.model`, che richiama la `RTDetrFusionBackbone`.

### 2. Esecuzione della `RTDetrFusionBackbone`
Siamo entrati nel metodo `forward` della nostra backbone personalizzata. Siccome la dimensione del canale di input è 4, la condizione `if c == 4:` si attiva.

- **Splitting**: Il tensore a 4 canali viene diviso lungo i canali in due:
  - `rgb_input` = `[Batch, 3, 640, 640]`
  - `ir_input` = `[Batch, 1, 640, 640]`

- **Estrazione Parallela**:
  - `rgb_input` passa dentro `self.rgb_backbone` (l'estrattore standard visivo).
  - `ir_input` passa dentro `self.ir_backbone` (l'estrattore identico nella struttura, ma specializzato a 1 canale).
  
  Ognuna delle due backbone restituisce una *lista* di tre feature map estratte in profondità sequenziale, riducendo le dimensioni spaziali via via:
  - Livello $C3$: RGB size `[Batch, 512, 80, 80]`, IR size `[Batch, 512, 80, 80]`.
  - Livello $C4$: RGB size `[Batch, 1024, 40, 40]`, IR size `[Batch, 1024, 40, 40]`.
  - Livello $C5$: RGB size `[Batch, 2048, 20, 20]`, IR size `[Batch, 2048, 20, 20]`.

### 3. Esecuzione del Modulo FAM e Fusione
Siamo ancora dentro `RTDetrFusionBackbone`, nel ramo in cui il modulo FAM è attivo (`use_fam=True`).
Si avvia un ciclo `for` che scorre "a coppie" i 3 livelli appena estratti.

Prendiamo ad esempio il livello $C4$ (1024 canali, risoluzione 40x40):
- **Allineamento**: Il tensore RGB e quello IR del livello $C4$ vengono concatenati e passati in input al modulo FAM corrispondente a quel livello. La convoluzione stima gli "offset". Il modulo `DeformConv2d` usa questi offset per trasformare solo le feature IR, creando la feature map `ir_aligned`. Questa feature map mantiene le dimensioni `[Batch, 1024, 40, 40]`; l'addestramento determina in quale misura risulti meglio registrata rispetto alla geometria RGB.
- **Fusione additiva**: sommo le feature elemento per elemento, `fused_C4 = r_feat + ir_aligned`. La forma rimane `[Batch, 1024, 40, 40]`.

Questa operazione di *FAM + Fusione additiva* viene eseguita su tutti i livelli ($C3$, $C4$, $C5$). Alla fine del ciclo, la `RTDetrFusionBackbone` raggruppa le tre mappe fuse in una singola *lista finale* pronta da servire e le restituisce in output.

### 4. Dal Transformer alle Teste Finali (Heads)
Le tre mappe fuse entrano a questo punto nel vero e proprio modulo Transformer di RT-DETR (Encoder e poi Decoder). Le forme delle feature restano quelle previste dall’encoder.
Il meccanismo di `self-attention` calcola le relazioni a lungo e corto raggio in tutta l'immagine e condensa le risposte in un piccolo gruppo di `queries` (i potenziali bounding boxes/oggetti predetti).

Questi vettori passano in ultimo per lo snodo di uscita, le **Teste (Heads)**:
- Per ogni query, `self.model.decoder.class_embed` produce logits; il postprocessing li trasforma in score di classe tramite sigmoid.
- Le teste `bbox_embed` producono i delta di raffinamento; il modello restituisce `pred_boxes` in formato `[centro_x, centro_y, larghezza, altezza]` **normalizzato in [0, 1]**, dopo la trasformazione sigmoid. La conversione in coordinate pixel appartiene al postprocessing, non all’uscita grezza del detector.

Durante il training le predizioni vengono confrontate con la ground truth per calcolare la loss. La retropropagazione attraversa decoder, fusione e le due backbone.
