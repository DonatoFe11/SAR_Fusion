# Deformable DETR Fusion per RGB-IR

> **Stato.** Le tre configurazioni sono già state replicate cinque volte, ma
> precedono il protocollo RT-DETR finale basato su testa `person`, checkpoint
> finale e validation disaccoppiata. Sono evidenza esplorativa utile, non una
> classifica quantitativa direttamente omogenea con la campagna RT-DETR 2026.

Questa nota documenta l'implementazione finale di Deformable DETR per la rilevazione RGB-IR su WiSARD. L'architettura usa due backbone indipendenti, una fusione per concatenazione dei canali non lineare e, nelle varianti dedicate, il Feature Alignment Module (FAM) e lo Stochastic Spatial Jitter (SSJ).

I risultati della famiglia Deformable DETR vanno interpretati come **mediana e intervallo su cinque run**, non come il risultato di una singola esecuzione perchè l'attenzione deformabile multi-scala introduce non determinismo run-to-run anche a seed fissato.

## Contesto storico: Spatial Concatenation scartata

Un primo tentativo concatenava le feature RGB e IR lungo l'altezza della mappa ($H \rightarrow 2H$), nell'ipotesi che la Deformable Attention imparasse autonomamente le corrispondenze tra i sensori. È stato scartato per due motivi, entrambi osservati sperimentalmente:

- il raddoppio della dimensione spaziale ha imposto di ridurre i livelli della piramide da quattro a due per evitare l'OOM;
- la geometria alterata ha reso incompatibili i reference point pre-addestrati dell'attenzione deformabile, con mAP prossima allo zero.

La configurazione finale non modifica quindi le dimensioni spaziali delle feature.

## Architettura finale

### Dual backbone

L'input multimodale ha quattro canali, tre RGB e uno IR. Viene diviso in due flussi:

```text
pixel_values [B, 4, H, W]
  ├── RGB: pixel_values[:, :3]  → backbone ResNet-50 RGB
  └── IR:  pixel_values[:, 3:]  → backbone ResNet-50 IR
```

Le due backbone ResNet estraggono separatamente tre feature map native (C3, C4 e C5). La backbone IR è inizializzata a partire da quella RGB pre-addestrata: nel primo layer, i pesi sui tre canali di ingresso sono mediati,

$$W_{IR} = \frac{1}{3}\sum_{c=1}^{3} W_{RGB,c},$$

così da ottenere filtri a un canale con un warm start coerente con il pretrained RGB. I layer successivi, che non dipendono dal numero di canali dell'immagine, mantengono la stessa struttura.

### Fusione per concatenazione dei canali

Per ciascuno dei tre livelli nativi $i$, le feature RGB e IR sono concatenate lungo i canali senza alterare la griglia $H_i \times W_i$:

$$F^{(i)}_{cat} = \operatorname{Concat}(F^{(i)}_{RGB}, F^{(i)}_{IR}) \in \mathbb{R}^{2C_i \times H_i \times W_i}.$$

La concatenazione passa poi in un blocco di fusione **non lineare** dedicato al livello:

$$F^{(i)}_{fused} = \operatorname{ReLU}\left(\operatorname{GN}_{32}\left(\operatorname{Conv}_{1\times1}(F^{(i)}_{cat})\right)\right) \in \mathbb{R}^{C_i \times H_i \times W_i}.$$

Il ruolo dei componenti è distinto:

- la Conv $1\times1$ riporta i canali da $2C_i$ a $C_i$ e apprende la combinazione tra le modalità;
- la GroupNorm a 32 gruppi stabilizza le attivazioni con batch size 1;
- la ReLU consente interazioni non lineari, non riducibili a un peso fisso RGB/IR.

Le tre feature fuse vengono proiettate separatamente a $d_{model}=256$. Poiché `num_feature_levels=4`, il modello Deformable DETR genera poi un quarto livello a più bassa risoluzione applicando una Conv $3\times3$ con stride 2 alla feature fusa C5; questo livello aggiuntivo non ha una seconda coppia RGB/IR e non passa in un quarto FAM. I quattro livelli complessivi vengono infine appiattiti e concatenati nella sequenza passata all'encoder.

La configurazione del processor `SenseTime/deformable-detr` usa `shortest_edge=800` e `longest_edge=1333`: preserva l'aspect ratio e ridimensiona entro questi due limiti, senza forzare universalmente un input quadrato $800\times800$. Se il vincolo sul lato lungo interviene, anche il lato corto finale è inferiore a 800: per esempio, un'immagine $3840\times2160$ (larghezza $\times$ altezza) diventa $1333\times750$. Il padding, quando necessario, porta i campioni alle dimensioni massime presenti nel batch; nelle configurazioni di questi esperimenti `batch_size=1`, quindi non trasforma un singolo campione non quadrato in un quadrato. Per un input che risulti effettivamente quadrato $800\times800$, i quattro livelli complessivi hanno risoluzione $100\times100$, $50\times50$, $25\times25$ e $13\times13$, per un totale di 13.294 token; per immagini non quadrate il numero di token varia con la forma finale. Il decoder usa 300 query e produce le usuali teste di classificazione e regressione dei bounding box.

## Varianti con FAM e SSJ

### Feature Alignment Module (FAM)

Nella variante FAM, prima della concatenazione dei canali, la feature IR viene allineata a quella RGB a ciascuno dei tre livelli nativi della backbone:

1. `offset_conv` riceve `Concat(RGB, IR)` e produce 27 canali: 18 offset $(dx,dy)$ per i nove punti di un kernel $3\times3$ e 9 mask di modulazione;
2. una DCNv2 applica offset e mask alla sola feature IR, producendo $F'_{IR}$;
3. il blocco di channel fusion riceve `Concat(RGB, IR_aligned)`.

I pesi di `offset_conv` sono inizializzati a zero: il campo di offset parte nullo e la correzione geometrica viene appresa progressivamente. Questo non rende l'intero FAM un'identità: la DCNv2 mantiene i propri pesi convoluzionali.

### Stochastic Spatial Jitter (SSJ)

Lo SSJ è attivo solo nel training e aggiunge rumore gaussiano agli offset predetti, prima della DCNv2. L'intento è evitare che il FAM converga verso un singolo pattern di allineamento, regolarizzandolo rispetto a piccole perturbazioni geometriche.

Su questa famiglia, SSJ non ha riprodotto il beneficio che era apparso nella
singola run RT-DETR preliminare: la sorgente di rumore aggiunta si associa a
una maggiore variabilità osservata. La successiva campagna RT-DETR a cinque
seed non ha a sua volta mostrato un miglioramento medio di SSJ, quindi non è
corretto descrivere oggi SSJ come regolarizzatore generalmente vantaggioso.

## Stabilità sperimentale e risultati

Il backward della Multi-Scale Deformable Attention esegue accumuli su posizioni campionate non intere. L'ordine degli accumuli CUDA può variare; piccole differenze numeriche sono poi amplificate dal matcher di Hungarian, discreto. Per questo ogni configurazione è stata addestrata su cinque seed e riportata come mediana [min--max], a soglia di confidenza 0.01.

| Configurazione | VIS-only | IR-only | RGB-IR |
| --- | --- | --- | --- |
| Deformable DETR | 0.157 [0.131--0.197] | 0.115 [0.086--0.142] | 0.249 [0.199--0.303] |
| + FAM | 0.184 [0.171--0.253] | 0.131 [0.097--0.165] | 0.294 [0.213--0.314] |
| + FAM + SSJ | 0.142 [0.115--0.206] | 0.088 [0.073--0.109] | 0.263 [0.170--0.350] |

FAM mostra la tendenza più promettente, con mediana RGB-IR da 0.249 a 0.294. Gli intervalli si sovrappongono però ampiamente: con cinque run non è possibile considerare le differenze statisticamente consolidate. FAM+SSJ ha la variabilità più elevata e una mediana inferiore a FAM.

Nessuna variante raggiungeva il riferimento RT-DETR storico (`0.357`, singola
run). Il confronto aggiornato non deve però usare quel numero come stima
deterministica: RT-DETR Additive finale ha media `0.3080 ± 0.0677`, mentre FAM
ha `0.3780 ± 0.0440` su cinque seed. Il percorso CUDA nativo di RT-DETR non è
pienamente deterministico; la differenza rispetto a Deformable DETR riguarda
l'architettura, il costo e la distribuzione dei risultati osservati, non
l'assenza assoluta di non-determinismo.

Non è necessario ripetere questa campagna se Deformable DETR resta una linea
esplorativa. Per dichiarare invece una superiorità cross-architettura definitiva
andrebbero riallenate almeno baseline e FAM con testa, checkpoint, split e
protocollo statistico uniformati a RT-DETR.
