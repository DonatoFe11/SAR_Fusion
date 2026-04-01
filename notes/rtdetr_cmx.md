# RT-DETR CMX: Cross-Modal Fusion Framework

Questo documento descrive l'implementazione del modello CMX (Cross-Modal Fusion) all'interno del framework RT-DETR. Rispetto all'approccio basato sul partizionamento geometrico (come il Feature Alignment Module), CMX si concentra sull'**interazione semantica bidirezionale** tra i sensori RGB e Infrarosso (IR).

Il file di riferimento per questa implementazione è: [sarfusion/models/rtdetr_cmx.py](../sarfusion/models/rtdetr_cmx.py).

L'architettura si articola in due step fondamentali applicati a ciascun livello della piramide delle feature (P3, P4, P5): **Rettifica** e **Fusione**.

---

## 1. CM-FRM: Cross-Modal Feature Rectification Module

Il primo passaggio è la "rettifica", ovvero usare un sensore per calibrare ("pulire") il rumore presente nell'altro sensore *prima* di effettuare la fusione vera e propria.

Il modulo `CM_FRM` lavora su due livelli:

1. **Rettifica Channel-wise (sui Canali)**:
   - Estrae statistiche globali per ogni canale usando l'Average Pooling e il Max Pooling su entrambe le modalità (RGB e IR).
   - Concatena queste 4 statistiche e le passa in un modulo percettrone (MLP) che genera dei pesi di ricalibrazione.
   - I pesi estratti dall'IR calibrano l'RGB e viceversa ($rgb\_c = rgb \times w\_ir$).

2. **Rettifica Spatial-wise (sullo Spazio)**:
   - Identifica le regioni spazialmente importanti fondendo i tensori lungo l'asse dei canali.
   - Una CNN con kernel $1 \times 1$ apprende coefficienti di attenzione per ogni singolo pixel.

**Equazione di Skip-Connection Finale**:
Il tensore restituito dal modulo ripristina il segnale originale affiancandogli le versioni calibrate:
$$ Output_{rgb} = RGB_{orig} + 0.5 \cdot RGB_{channel\_rect} + 0.5 \cdot RGB_{spatial\_rect} $$
(Lo stesso avviene simmetricamente per l'Infrarosso).

---

## 2. FFM: Feature Fusion Module (Cross-Attention)

Una volta che le feature RGB e IR sono state raffinate e pulite dal CM-FRM, passano al modulo di fusione `FFM`. L'idea alla base di CMX è che le due modalità si "interroghino" a vicenda per estrarre la semantica migliore.

Questo si ottiene tramite il paradigma della **Single-Head / Multi-Head Cross-Attention**:
- **Query (Q)**: Provengono dalle feature RGB.
- **Key (K) e Value (V)**: Provengono dalle feature IR.
L'RGB "cerca" delle informazioni nell'IR e l'attenzione modula i valori di uscita.

### Il trucco contro l'esplosione della memoria
Le architetture Transformer soffrono di una complessità computazionale e di memoria originariamente *quadratica* ($O(N^2)$) rispetto al numero di token. Nel caso della *Computer Vision*, il numero di token $N$ è dato da $Altezza \times Larghezza$ ($H \times W$).
Alla scala piramidale P3 (la più bassa semanticamente ma la più grande spazialmente), una risoluzione tipica $80 \times 80$ genera $6400$ token. Fare un'attenzione $6400 \times 6400$ consumerebbe decine di Gigabyte di memoria VRAM generando errori OOM (Out Of Memory).

Per evitarlo, nell'implementazione è stata posta una **soglia e un fallback lineare**:
```python
if h * w > 1600:
    return self.out_conv(torch.cat([rgb, ir], dim=1))
```
Se l'area è troppo vasta, il modulo salta l'attenzione quadratica e si affida a una tradizionale e leggera Convoluzione $1 \times 1$ applicata alla concatenazione spaziale dei due tensori. In questo modo si mantengono i benefici della Cross-Attention alle alte scale piramidali (P4, P5), disinnescando l'esplosione per le scale ad alta risoluzione in ingresso.

---

## 3. Gestione Sensori Mancanti (Modality Dropout Nativo)

All'interno di `RTDetrCMXBackbone` è stata implementata una logica di robustezza per aggirare ostacoli causati da malfunzionamenti hardware del sensore (es. IR rotto) o strategie di **Modality Dropout** durante l'addestramento.

Se il `forward` riceve un sensore "castrato" dal Dataloader (es. num_canali = 3 invece di 4):
```python
elif num_ch == 3:
    rgb_o = self.rgb_backbone(pixel_values, pixel_mask)
    # Creiamo un "finto" IR di zeri per passare attraverso i moduli CMX
    ir_o = [(torch.zeros_like(f), m) for f, m in rgb_o]
```
Invece di lanciare un'eccezione o bypassare l'intera architettura, il framework crea un "sensore fittizio" riempiendo tensori della dimensione esatta con zeri assoluti (`torch.zeros_like`). 

Dato che i moduli successivi (CM-FRM e FFM) processeranno questo "nulla cosmico":
1. L'architettura non si rompe.
2. I pesi della rete imparano durante l'addestramento ad auto-bilanciarsi e non dipendere ciecamente sempre da entrambi i sensori, rendendo il rilevatore più affidabile e robusto.

---

## 4. Wrapper Configuration

Il modello base, istanziato tramite `RTDetrCMXForObjectDetection`, si occupa poi di recuperare tutte queste feature bilanciate e multi-scala, agganciandole ai normali **Heads (Class & BBox Embed)** del framework standard Hugging Face RT-DETR, preservando le intatte capacità formidabili del decoder temporale.
La media aritmetica sui canali per caricare correttamente i pesi dal ResNet pre-addestrato per l'Infrarosso avviene fluidamente all'interno della sovrascrittura di `.from_pretrained`.

---

## 5. Limiti Architetturali e Miglioramenti Ibridi nel Contesto SAR

Nonostante CMX sia un approccio "State of the Art" nel *Dense Scene Understanding*, applicato su dataset da droni (come WiSARD) per compiti di *Search and Rescue* può riscontrare criticità legate alle particolarità del dominio. Di seguito le analisi dei problemi riscontrati e le soluzioni ibride progettate per i futuri esperimenti.

### Problema 1: Disallineamento e Parallasse (Incompatibilità con CM-FRM)
Sui droni, i sensori RGB e Termici non sono mai perfettamente allineati spazialmente a livello di pixel (parallasse, lenti diverse, leggero ritardo d'acquisizione). Il modulo CM-FRM assume un ambiente **Pixel-Perfect**: per "pulire" l'RGB usa pedissequamente le feature allocate alle medesime coordinate sull'IR. Se l'IR è spostato, si va a incrociare lo sfondo freddo termico con il bersaglio visivo, abbattendo radicalmente l'utilità del segnale (*Interferenza Distruttiva*).

### Problema 2: Perdita di Contesto Spaziale (FFM)
L'attenzione tradizionale utilizzata dal modulo di fusione ($F.scaled\_dot\_product\_attention$) srotola i token e li tratta come un "sacco". Senza la cognizione delle posizioni 2D relative (le feature vengono prelevate dalla CNN pura prima che RT-DETR applichi il Positional Encoding nativamente nel Decoder), si apre la strada al *Ghosting*, in cui l'RGB preleva feature termiche da parti casuali dell'immagine perché semanticamente affini.

### Problema 3: "Schiacciamento" dei target piccoli in P3
Per via della mole $O(N^2)$ richiesta dalla matrix multiplication dell'Attention, l'hardware va in *Out of Memory* sulle feature di primo livello ($P3: 80 \times 80 = 6400$ token). Il bypass imposto dal codice esegue per P3 una semplice concatenazione + convoluzione lineare $1 \times 1$. Purtroppo per il task *Search And Rescue*, i dispersi rientrano spesso come bersagli microscopici, ed è proprio in scala P3 che risiede la probabilità maggiore di trovarli. Eseguire una concatenazione lineare su due mappe disallineate produce la corruzione irrimediabile dei corpi di dimensione limitata.

### La Soluzione: Architettura Ibrida (FAM + CMX)
Per risolvere contemporaneamente questi punti, è possibile implementare una variante Ibrida unendo i pregi del Modulo F.A.M. (Feature Alignment Module descritto per l'architettura originaria):

1. **Pre-Allineamento (FAM)**: Inserendo le *Deformable Convolutions* **prima** della fase CMX (`[RGB, IR] -> FAM -> CM-FRM -> FFM`), la rete userebbe offset spaziali elastici per far coincidere le mappe termiche ai corrispettivi visivi.
2. **Successo nel CM-FRM**: Quando le mappe giungono al Modulo CMX, ora sono state modellate *pixel-perfect*; la calibrazione e soppressione del rumore non distruggeranno i target, ma smorzeranno autenticamente solo gli artefatti come sperato teoricamente.
3. **Fusione P3 Miracolata**: Se le feature sono pre-allineate, il fastidioso vincolo bypass-lineare sul P3 diventa improvvisamente preziosissimo; la convoluzione pura concatenando un IR e un RGB che si ricalcano temporalmente e spazialmente in modo esatto costituirà una fusione robustissima sui target microscopici, arginando così del tutto sia gli effetti di parallasse che la deficienza computazionale.
4. **Positional Encoding Iniettato**: Aggiungendo un *2D Sine Positional Encoding* personalizzato allo stadio *FFM*, la rete acquisirebbe la capacità spaziale locale per smaltire il ghosting.