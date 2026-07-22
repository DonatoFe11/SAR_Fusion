# RT-DETR Fusion with Feature Alignment Module (FAM): detailed explanation.

## Feature Alignment Module (FAM)
Il modulo FAM è progettato per allineare le feature map IR con quelle RGB prima di fonderle. Questo è cruciale perché le due modalità (RGB e IR) possono essere disallineate a causa di differenze nella geometria, prospettiva o distorsioni ottiche.

---
### Costruttore `__init__`

Come prima cosa esaminiamo riga per riga la funzione `__init__` e capiamo a cosa servono questi layer `Conv2d` e `DeformConv2d` e "come sono fatti" matematicamente.

---

### `def __init__(self, in_channels, freeze=False):`
Inizializza il modulo definendo i pesi che dovrà imparare (o congelare).
- `in_channels`: È il numero di canali delle feature map in input. Ad esempio, se siamo in un livello profondo della backbone, in_channels potrebbe essere 256.
- `freeze`: Se impostato a True, congela i gradienti del layer, bloccando l'aggiornamento dei pesi e forzando la rete a usare il FAM come una Random Projection fissa (ottimo per regolarizzazione o noise injection).

### `super().__init__()`
Chiama il costruttore della classe padre (`nn.Module`). È standard in PyTorch per far sì che il modulo venga registrato correttamente.

---

### Layer 1: `self.offset_conv = nn.Conv2d(...)`
Questo è un normale layer convoluzionale standard di PyTorch, ma **non serve a estrarre feature visive**. Serve a **"predire come deformare"** la griglia della convoluzione successiva.

Come è fatto internamente:
- **`in_channels * 2`**: È il numero di canali in input. È `*2` perché (come vedrai nel `forward`) concateneremo la map RGB e la map IR lungo l'asse dei canali (e.g. $256 + 256 = 512$).
- **`27`**: È il numero di canali in output. **Perché proprio 27?**
  Questa convoluzione deve produrre i parametri per un kernel $3 \times 3$ deformabile. Un kernel $3 \times 3$ ha 9 celle (o punti).
  - Per ognuno dei 9 punti, la rete deve predire uno spostamento lungo $x$ ($\Delta x$) e uno lungo $y$ ($\Delta y$). Quindi $9 \times 2 = 18$ canali (gli "offsets").
  - Per ognuno dei 9 punti, la rete deve predire anche un "peso modulatore" (la `mask`), compreso tra 0 e 1, che decide quanto quel punto debba essere 'acceso' o 'spento'. Quindi altri $9$ canali.
  - Totale canali in uscita: $18 + 9 = 27$.
- **`kernel_size=3`** e **`padding=1`**: Configurazione classica che mantiene inalterate le dimensioni spaziali di altezza ($H$) e larghezza ($W$) in output. I pesi interni di questo layer saranno un tensore di dimensione `[27, in_channels * 2, 3, 3]`.

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

## L'inizializzazione a zero e il Congelamento (identity mapping)
Successivamente i pesi e i bias della `offset_conv` vengono inizializzati a `0`:
```python
nn.init.constant_(self.offset_conv.weight, 0)
nn.init.constant_(self.offset_conv.bias, 0)

if freeze:
    for param in self.parameters():
        param.requires_grad = False
```
**Perché l'inizializzazione a zero?** Se gli offset iniziali sono 0, la `DeformConv2d` si comporta esattamente come una normale convoluzione 3x3 statica (nessuna deformazione della griglia). Questo è un trucco standard chiamato **identity mapping iniziale**: evita che all'inizio dell'addestramento vengano previsti offset casuali che distruggerebbero le feature pre-addestrate. La rete impara a deformare l'immagine solo progressivamente e dove serve.
**Perché il comando `freeze`?** Se si imposta a `True`, la backpropagation ignorerà questo modulo, i gradienti non si aggiorneranno e l'offset rimarrà sempre 0 (rendendo l'hardware una 3x3 statica con pesi convoluzionali *random* fissi, per l'effetto noise-injection).

---

## Il metodo `forward` della classe FAM
Nel metodo `forward`, la rete prende in input le feature map RGB e IR, le concatena, e poi calcola gli offset e le maschere per deformare la mappa IR.

```python
concat = torch.cat([rgb_feat, ir_feat], dim=1)  # [B, 2C, H, W]
```
Concatenando l'RGB e l'IR lungo l'asse dei canali (`dim=1`), la rete "sovrappone" le due modalità. In questo modo la convoluzione successiva può confrontare le due mappe e dedurre dove gli oggetti termici sono disallineati rispetto agli oggetti visibili.

```python
out = self.offset_conv(concat)  # [B, 27, H, W]
```
Il tensore passa attraverso il layer che abbiamo spiegato prima (offset_conv). Il risultato ha 27 canali.

```python
offset = out[:, :18, :, :]  # [B, 18, H, W]
mask = torch.sigmoid(out[:, 18:, :, :])  # [B, 9, H, W]
```
Qui il tensore viene splittato:
1. **`offset`**: I primi 18 canali rappresentano gli spostamenti ($\Delta x$ e $\Delta y$) per i 9 punti del kernel 3x3.
2. **`mask` (Modulation Scalars)**: Gli ultimi 9 canali passano in una funzione `sigmoid` in modo che il risultato sia limitato tra 0 e 1. Questi valori fanno da "moltiplicatori di importanza". Se la rete calcola che un certo offset finisce su un pixel rumoroso o non utile temporaneamente, la mask lo spinge verso lo 0, ignorandolo.

```python
ir_aligned = self.deform_conv(ir_feat, offset, mask)
return ir_aligned
```
Infine, la `DeformConv2d` viene applicata **esclusivamente su `ir_feat`**, ma guidata dagli `offset` e `mask` calcolati guardando *entrambe* le immagini. L'output è una nuova mappa feature termica (`ir_aligned`), in cui gli elementi sono stati "tirati" o "spinti" spazialmente per allinearsi alla perfezione alla geometria della feature map RGB.

---

## Riepilogo:
1. Viene impostata la `offset_conv` che prenderà le feature RGB+IR fuse assieme per calcolare una mappa densa di trasformazioni ($18$ di shift e $9$ di maschera).
2. Viene definita la `deform_conv` che rappresenta l'operazione che userà effettivamente la mappa di trasformazioni appena calcolata (applicandola sull'IR) per ottenere una nuova mappa IR riallineata alle feature RGB.

Dunque, `offset_conv` prende in input la concatenazione delle feature RGB e IR e produce i parametri di deformazione, mentre `deform_conv` prende in input solo la feature map IR e applica la deformazione per riallinearle con le feature RGB usando gli offset appena calcolati.

<br><br>

## Fusion Backbone (`RTDetrFusionBackbone`)
Questa classe è il vero "motore" dell'estrazione delle feature. Gestisce due estrattori separati (uno per l'RGB e uno per l'IR) e implementa la logica di parallelismo e, infine, di fusione delle feature map (con o senza il FAM).

---

### Considerazioni Architetturali:
1. **La Backbone**: In PyTorch/HuggingFace, l'architettura RT-DETR usa un `RTDetrConvEncoder` (che incapsula reti per l'estrazione visiva, come ResNet50 o PPLCNet) come backbone. Noi stiamo istanziando *due copie identiche* di questo encoder: una per RGB e una per IR.
2. **Pesi Indipendenti**: Strutturalmente le due backbone sono cloni (stessi strati e canali), ma non sono la stessa rete in memoria (nessun *weight sharing*). All'inizio dell'addestramento i pesi sono quasi uguali, ma proseguendo evolveranno in modo indipendente: la backbone RGB si specializzerà su texture/colori, quella IR sulle firme termiche.
3. **Pesi Pre-addestrati**: Il modello di base di partenza è `PekingU/rtdetr_r50vd`. È un modello pre-addestrato sul dataset COCO (immagini RGB a 3 canali). Per questo motivo è essenziale l'adattamento da 3 a 1 canale sulla porta infrarossi in fase di inizializzazione.

---

### Costruttore `__init__`

```python
def __init__(self, config: RTDetrConfig, use_fam: bool = False):
    super().__init__()
```
Riceve in input la configurazione standard di RT-DETR e un flag `use_fam` per decidere se attivare o meno l'allineamento tramite deformazione.

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
    if self.use_fam:
        feature_channels = getattr(config, "encoder_in_channels", None)
        self.fam_modules = nn.ModuleList(
            [FeatureAlignmentModule(ch, freeze=self.freeze_fam) for ch in feature_channels]
        )
    else:
        self.fam_modules = None
```
**Inizializzazione Eager (Immediata)**: A differenza di implementazioni passate che usavano la "lazy initialization" (causando problemi con il tracciamento dei gradienti negando gli update al modulo FAM), qui i moduli vengono istanziati **in modo esplicito (eagerly)**. Leggendo i numeri di canale da `config.encoder_in_channels`, la rete crea da subito tante copie del modulo quanti sono i livelli estratti, risolvendo il problema dell'ottimizzatore. Inoltre, `freeze_fam` permette di "congelare" esplicitamente i pesi del FAM qualora si volesse sfruttare l'allineamento casuale come regolarizzatore formale.

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
Se passiamo un tensore a 3 canali usa solo la parte RGB, se ne passiamo 1 usa solo l'IR (molto flessibile per i test).

**Modalità 3: Fusione (RGB + IR)**
```python
    if c == 4:
        rgb_feats = self.rgb_backbone(pixel_values[:, :3], pixel_mask)
        ir_feats  = self.ir_backbone(pixel_values[:, 3:], pixel_mask)
```
Se passiamo 4 canali, assume la concatenazione. Smezza il tensore (primi 3 per RGB, ultimo logico per IR) e fa processare le metà in parallelo alle rispettive backbone. Il risultato (`rgb_feats` e `ir_feats`) sono **liste di feature map** a vari livelli di risoluzione.

*Quanti sono i livelli di queste feature map?*
In RT-DETR, grazie al design della backbone (es. ResNet), vengono estratti esattamente **3 livelli di feature map** per gestire il multi-scala:
- **Livello 1 ($C3$)**: Scala spaziale $H/8$, $W/8$ (di solito 512 canali).
- **Livello 2 ($C4$)**: Scala spaziale $H/16$, $W/16$ (di solito 1024 canali).
- **Livello 3 ($C5$)**: Scala spaziale $H/32$, $W/32$ (di solito 2048 canali).
Quindi `rgb_feats` conterrà 3 tensori e il ciclo che le attraversa girerà esattamente 3 volte, instanziando 3 moduli FAM differenti!

#### Ramo con FAM (Allineamento e Fusione)
```python
        if self.use_fam:
```
Ecco l'attivazione *eager* all'opera. Nel costruttore, la rete ha già creato una `nn.ModuleList` contenente tre FAM (uno per livello di canale: 512, 1024, 2048). Nel `forward` dobbiamo solo chiamarli ciclando sui 3 livelli in parallelo.

```python
            fused_feats = []
            for idx, ((r_feat, r_mask), (i_feat, _)) in enumerate(zip(rgb_feats, ir_feats)):
                # Allinea IR a RGB tramite FAM
                i_aligned = self.fam_modules[idx](r_feat, i_feat)
                
                # Fusione additiva
                fused_feats.append((r_feat + i_aligned, r_mask))
            return fused_feats
```
Cicla sui 3 livelli. Tramite il modulo FAM specifico di quel livello, allinea la mappa termica (`i_feat`) a quella visiva (`r_feat`). Infine applica una **Fusione Additiva** (`r_feat + i_aligned`): somma punto-a-punto i canali dei due tensori per arricchire l'informazione ottica con il contrasto termico perfettamente allineato.

#### Ramo Base (Senza FAM)
```python
        else:
            fused_feats = []
            for (r_feat, r_mask), (i_feat, _) in zip(rgb_feats, ir_feats):
                fused_feats.append((r_feat + i_feat, r_mask))
            return fused_feats
```
Se `use_fam` è disattivato, esegue comunque la somma additiva per unire le informazioni delle due modalità, ma saltando il passaggio di allineamento deformabile spaziale. Molto utile per eseguire gli Ablation Study.

<br><br>

## RT-DETR Model e Object Detection Wrapper

Dopo aver definito la backbone che estrae le feature fuse, dobbiamo calare questa logica nell'architettura finale completa per poter fare object detection. Questo viene fatto sfruttando le classi messe a disposizione dalla libreria `transformers` di HuggingFace.

### Il Modello Intermedio (`RTDetrFusionModel`)
```python
class RTDetrFusionModel(RTDetrModel):
    def __init__(self, config: RTDetrConfig, use_fam: bool = False):
        super().__init__(config)
        self.backbone = RTDetrFusionBackbone(config, use_fam=use_fam)
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
2. **Le Teste (Heads)**: Sono gli ultimissimi strati neurali (spesso semplici layer lineari `nn.Linear`). Prendono le feature elaborate dal corpo e sputano fuori concretamente i numeri finali che ci servono:
   - `class_embed`: Un layer che calcola le probabilità. "Questo oggetto al 90% è un cane, al 10% è un gatto".
   - `bbox_embed`: Un layer che calcola coordinate. "Le coordinate \((x, y, w, h)\) del box sono queste".

#### Costruttore e Trick delle Teste
```python
def __init__(self, config: RTDetrConfig, use_fam: bool = False, freeze_fam: bool = False):
    # Trick: inizializziamo come RGB standard
    tmp_cfg = copy.deepcopy(config)
    tmp_cfg.num_channels = 3
    super().__init__(tmp_cfg)
```
Poiché il nostro input sarà un tensore a 4 canali (RGB + IR), se passassimo la configurazione `num_channels = 4` al costruttore originale (`super().__init__`), HuggingFace cercherebbe di creare interamente il suo estrattore standard configurato a 4 canali. Per ingannarlo momentaneamente, gli passiamo 3 canali. Questo fa sì che la classe madre (`RTDetrForObjectDetection`) inizializzi correttamente tutti i moduli standard e la sua intera cascata di funzioni, creando un modello standard integro (con tanto di corpo e teste).

```python
    # Salviamo le teste originali
    saved_class_embed = self.class_embed
    saved_bbox_embed = self.bbox_embed

    # Sostituiamo il modello 
    self.model = RTDetrFusionModel(config, use_fam=use_fam, freeze_fam=freeze_fam)

    # Ripristiniamo le teste nel decoder del nuovo corpo
    self.model.decoder.class_embed = saved_class_embed
    self.model.decoder.bbox_embed = saved_bbox_embed
    self.config = config
    self.use_fam = use_fam
    self.freeze_fam = freeze_fam
```
Ecco cosa succede riga per riga in questo passaggio:
1. Poco fa, istanziando il `super().__init__`, la libreria ha creato automaticamente le Teste (gli strati finali `self.class_embed` e `self.bbox_embed`). Prima di toccare qualsiasi cosa, **ne salviamo una copia** in due variabili temporanee (`saved_...`).
2. Poi prendiamo `self.model` (il Corpo originale) e lo spazziamo via, sovrascrivendolo con il nostro `RTDetrFusionModel` (quello che sa gestire i 4 canali e che al suo interno ha la biforcazione in due backbone diverse).
3. Infine, prendiamo le Teste che avevamo messo da parte e **le incolliamo all'interno del decoder del nuovo Corpo** appena creato (`self.model.decoder... = saved...`). 

**Perché le attacchiamo specificamente a `self.model.decoder` e non di nuovo a `self`?**
Perché nell'architettura dei Transformers implementata da HuggingFace, il "flusso logico" dei dati durante l'addestramento va dal `model` (che contiene l'encoder) verso il `model.decoder`. Il decoder è il modulo responsabile di generare le feature finali, e le Teste di predizione sono pensate per essere fisicamente connesse ai layer di uscita del decoder. L'`RTDetrForObjectDetection` base le espone anche fuori su `self.class_embed` come "scorciatoia" per leggerle comodamente, ma il reale blocco neurale che fa la moltiplicazione di matrici sta alla fine del decoder. Andando a ricucire i riferimenti puntando a `self.model.decoder`, ci assicuriamo che quando i tensori viaggiano nel nostro nuovo motore custom, trovino le teste esatte di classificazione al posto giusto senza lanciare eccezioni!

Questo "ponteggio" ci permette di sostituire in tronco l'intero motore di estrazione feature (il Corpo) a metà della rete, ma mantenere esattamente gli stessi strati neurali logici in uscita che l'API di PyTorch si aspetta per il suo calcolo.

---

### Il Cuore del Transfer Learning: `from_pretrained`
La vera "magia" della gestione dei pesi avviene in questo costruttore alternativo (`@classmethod`), che viene invocato quando vogliamo usare pesi preesistenti, come `PekingU/rtdetr_r50vd`.

```python
@classmethod
def from_pretrained(cls, pretrained_model_name, id2label, label2id, ignore_mismatched_sizes=True, use_fam=False):
    # Diciamo alla libreria base di scaricare i pesi RGB normali
    base = RTDetrForObjectDetection.from_pretrained(
        pretrained_model_name, ...
    )
```
Qui diciamo a HuggingFace di scaricare ed inizializzare un modello RT-DETR *puramente visivo* dai server (o in locale).

```python
    # Istanziamo il nostro modello a 4 canali "vuoto"
    config = base.config
    config.num_channels = 4
    instance = cls(config, use_fam=use_fam)
    
    # Carichiamo tutti i pesi comuni (Encoder, Decoder, Teste d'uscita)
    instance.load_state_dict(base.state_dict())
```
Questo passaggio `load_state_dict` carica tutto tranne le backbone all'interno della nostra istanza, permettendo al Transformer Encoder/Decoder di avere già tutta l'intelligenza di visione spaziale ereditata.

Ora viene la gestione delle *due* backbone (finora `base` ne ha solo una RGB da 3 canali):

**1. Backbone RGB:**
```python
    sd = base.state_dict()
    rgb_w = {
        k.replace("model.backbone.", ""): v
        for k, v in sd.items() if "model.backbone" in k
    }
    instance.model.backbone.rgb_backbone.load_state_dict(rgb_w, strict=False)
```
Filtra tutti i pesi di `base` che appartengono alla sua singola backbone. Li inietta in blocco dentro la nostra copia `self.rgb_backbone`. Questa operazione non richiede modifiche.

**2. Backbone IR (L'Adattamento dei Canali di Input):**
```python
    ir_w = copy.deepcopy(rgb_w)
    for k in list(ir_w.keys()):
        if ir_w[k].dim() == 4 and ir_w[k].shape[1] == 3:
            ir_w[k] = ir_w[k].mean(dim=1, keepdim=True)

    instance.model.backbone.ir_backbone.load_state_dict(ir_w, strict=False)

    return instance
```
Questo è il culmine logico del _warm start_ preannunciato. Creiamo un nuovo dizionario virtuale (`ir_w`) per la porta IR, identico a quello visivo. Poi passiamo al setaccio ogni tensore che lo compone: se troviamo un tensore convoluzionale a 4 dimensioni `[out, in, H, W]` in cui i canali `in=3`, applichiamo la **media** sul canale 1. 

Questo trasforma fisicamente e matematicamente il tensore da _"conoscitivo di 3 colori"_ a _"conoscitivo dell'intensità complessiva"_. Infine, questo nuovo set di pesi compressi viene iniettato nel nostro `self.ir_backbone`.

L'oggetto testuale finito restituito da questa funzione è un modello intero RT-DETR che:
- Riceve un sensore a 4 canali in ingresso.
- Processa 3 canali di ottico e 1 canale di termico in vie parallele ma con un livello di intelligenza visiva iniziale altissimo su entrambi i percorsi.
- Li fonde progressivamente ai vari livelli spaziali tramite una deformazione dei tensori.
- Esce con le stesse teste di detection che aveva RT-DETR in origine.

<br><br>

## 🌊 Esempio di Flusso Dati (Data Flow)

Immaginiamo di trovarci in fase di *forward pass* (durante l'addestramento). Vogliamo predire dove si trovano determinati oggetti passando alla rete l'immagine di una telecamera visiva (RGB) e l'immagine della telecamera termica (IR).

### 1. Input del Wrapper
Passiamo l'input al nostro wrapper `RTDetrFusionForObjectDetection`:
- **Dimensione iniziale**: Entra un tensore `pixel_values` di dimensione `[Batch, 4, 640, 640]`. I primi 3 canali sono i pixel RGB, l'ultimo canale è quello IR.

Il Wrapper non "tocca" i pixel in modo intelligente, ma si limita a riversarli dentro al suo `self.model` (il Corpo). Il flusso arriva così alla base del modello, cioè alla `self.backbone` (che nel nostro caso truccato è `RTDetrFusionBackbone`).

### 2. Esecuzione della `RTDetrFusionBackbone`
Siamo entrati nel metodo `forward` della nostra backbone personalizzata. Siccome la dimensione del canale di input è 4, la condizione `if c == 4:` si attiva.

- **Splitting**: Il tensore a 4 canali viene smezzato logitamente in due:
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
- **Allineamento**: Il tensore RGB e quello IR del livello $C4$ vengono fusi e passati in input al modulo FAM corrispondente a quel livello. La convoluzione stima gli "offset". Il modulo `DeformConv2d` prende questi offset e modifica solo le feature IR, riallineandone i pattern geometrici, creando la feature map: `ir_aligned`. Questa feature map mantiene inalterate le dimensioni `[Batch, 1024, 40, 40]`, ma ora è semanticamente speculare e allineata rispetto alla geometria dell'RGB.
- **Fusione Additiva**: Viene eseguita la fusione tra i due mondi tramite una banale ma efficacissima sommma algebrica per elemento: `fused_C4 = r_feat + ir_aligned`. La nuova mappa, densa delle informazioni fiorite da entrambe le bande elettro-magnetiche originarie, resta di dimensioni `[Batch, 1024, 40, 40]`.

Questa operazione di *FAM + Fusione additiva* viene eseguita su tutti i livelli ($C3$, $C4$, $C5$). Alla fine del ciclo, la `RTDetrFusionBackbone` raggruppa le tre mappe fuse in una singola *lista finale* pronta da servire e le restituisce in output.

### 4. Dal Transformer alle Teste Finali (Heads)
Le tre mappe fuse entrano a questo punto nel vero e proprio modulo Transformer di RT-DETR (Encoder e poi Decoder). Il transformer "non si è accorto di nulla", lui vede scorrere normalissime feature map di quelle esatte grandezze previste!
Il meccanismo di `self-attention` calcola le relazioni a lungo e corto raggio in tutta l'immagine e condensa le risposte in un piccolo gruppo di `queries` (i potenziali bounding boxes/oggetti predetti).

Questi vettori passano in ultimo per lo snodo di uscita, le **Teste (Heads)**:
- Per ogni query, lo strato `self.model.decoder.class_embed` analizza i vettori e stima lo score di appartenenza alle classi a disposizione (es: "Probabilità: Persona al 98%").
- Lo strato complementare `self.model.decoder.bbox_embed` restituisce numericamente le coordinate in pixel finali `[centro_x, centro_y, larghezza, altezza]`.

L'intero flusso end-to-end è così concluso! Le coordinate prodotte sono adesso a disposizione del train-loop PyTorch per sfidare la *Ground Truth*, calcolare la loss e retro-propagare l'errore calcolando i gradienti di tutte e due le backbone.
