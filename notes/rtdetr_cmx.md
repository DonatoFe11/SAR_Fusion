# RT-DETR + CMX: fusione cross-modale e variante ibrida

> **Stato storico/esplorativo.** I risultati CMX derivano da singole run del
> protocollo precedente. Documentano il fallimento di queste specifiche
> implementazioni e hanno motivato il FAM, ma non stimano la variabilità tra
> seed. I confronti con il vecchio RT-DETR `0.357` non sono una classifica
> finale; il riferimento aggiornato è in
> [`rtdetr_reproducibility.md`](rtdetr_reproducibility.md).

Questa nota documenta le due implementazioni CMX sperimentate con RT-DETR:

- **CMX puro**: [`sarfusion/models/rtdetr_cmx.py`](../sarfusion/models/rtdetr_cmx.py);
- **CMX ibrido**: [`sarfusion/models/rtdetr_cmx_hybrid.py`](../sarfusion/models/rtdetr_cmx_hybrid.py), che antepone FAM e aggiunge positional encoding 2D.

Entrambe usano due `RTDetrConvEncoder` ResNet-50vd: una backbone RGB a tre canali e una IR a un canale. I pesi dello stem IR sono inizializzati dalla media dei canali dello stem RGB pre-addestrato. Le feature P3, P4 e P5 (512, 1024 e 2048 canali) sono quindi fuse prima dell'hybrid encoder e del decoder standard di RT-DETR.

## CMX puro

Il modello puro applica a ogni livello della piramide due moduli del framework CMX.

### CM-FRM: rettifica cross-modale

Il *Cross-Modal Feature Rectification Module* calibra RGB e IR lungo due assi:

1. **Canali.** Average pooling e max pooling globali delle due modalità producono quattro vettori di statistiche. Un MLP con sigmoide genera due insiemi di pesi: quelli derivati dalle statistiche IR modulano RGB e viceversa.
2. **Spazio.** Una piccola rete `Conv1×1 → ReLU → Conv1×1 → Sigmoid` riceve la concatenazione RGB–IR e produce due mappe di pesi spaziali, una per modalità.

L'uscita conserva un residuale della feature originale:

$$
F'_{RGB}=F_{RGB}+0.5(F_{RGB}\odot w_{IR}^{c})+0.5(F_{RGB}\odot w_{IR}^{s}),
$$

con la stessa operazione, a ruoli invertiti, per l'IR. La rettifica è utile se le feature corrispondenti sono già registrate (i pixel delle due immagini sono corrispondenti) poichè non corregge di per sé la parallasse fra sensori.

### FFM: cross-attention e bypass P3

Il *Feature Fusion Module* usa RGB come query e IR come key/value. Le proiezioni `1×1` producono quattro teste di cross-attention, l'uscita attenzionale IR viene concatenata alle feature RGB e ricondotta a $C$ canali da una `Conv1×1`.

L'attenzione piena ha costo $O((HW)^2)$. Per $H\times W>1600$ — normalmente P3, il livello più importante per target piccoli — il codice evita l'OOM con un bypass `Conv1×1(Concat(RGB, IR))`, mentre per P4 e P5 usa `scaled_dot_product_attention`.

### Modalità mancanti

In input a quattro canali il modello separa RGB e IR. Con soli tre canali crea feature IR nulle, con il solo canale IR crea feature RGB nulle. I moduli CMX restano quindi nel percorso di inferenza anche in assenza di un sensore. Questo rende l'esecuzione possibile, ma non costituisce da solo una garanzia di robustezza: dipende anche da come il modello è stato addestrato.

### Esito sperimentale

Il CMX puro è stato fine-tuned per 25 epoche su WiSARD e ha ottenuto **0.032 mAP@50** in fusione a soglia 0.01, contro **0.357** del baseline RT-DETR. A soglia 0.10 il CMX puro non ha prodotto rilevazioni utili (0.000). Le prestazioni IR-only (0.081) superiori a quelle in fusione sono coerenti con un'interferenza fra modalità, ma non permettono di isolare una causa unica del collasso.

I fattori plausibili, osservati nel contesto sperimentale, sono il disallineamento RGB–IR, l'assenza di coordinate esplicite nell'attenzione e il sovraccarico di parametri/ottimizzazione su un dataset limitato. Il dato sperimentale supporta il fallimento del CMX puro.

| Modello | Test | Soglia | mAP@50 |
| --- | --- | ---: | ---: |
| CMX puro | $D_{RGB-IR}$ | 0.01 | 0.0320 |
| CMX puro | $D_{IRo}$ | 0.01 | 0.0810 |
| RT-DETR baseline | $D_{RGB-IR}$ | 0.01 | 0.3570 |

## CMX ibrido: FAM → CM-FRM → FFM + positional encoding

Per ogni livello P3–P5 applica la pipeline:

$$
[F_{RGB},F_{IR}]\xrightarrow{FAM}[F_{RGB},\widetilde{F}_{IR}]
\xrightarrow{CM\text{-}FRM}[F'_{RGB},F'_{IR}]
\xrightarrow{FFM+PE}F_{fused}.
$$

### FAM: allineamento dell'IR guidato da RGB

Il *Feature Alignment Module* concatena le due feature e predice, con una convoluzione `3×3`, 18 offset e 9 maschere di modulazione per una `DeformConv2d` $3\times3$. La deformable convolution trasforma la sola feature IR rispetto al riferimento RGB. Gli offset sono limitati da `4·tanh(·)` e la maschera è una sigmoide clampata, scelte che limitano instabilità numeriche in AMP; gli input e gli output passano inoltre da `nan_to_num`.

> **Differenza rispetto agli altri FAM del progetto.** Questo bound è specifico del FAM del CMX ibrido. Nei FAM di RT-DETR e Deformable DETR, l'offset passato alla deformable convolution è l'output grezzo della convoluzione (`offset = out[:, :18]`): non passa da `4·tanh(·)`. Anche la maschera usa `sigmoid`, ma senza il `clamp(1e-4, 1-1e-4)`. Di conseguenza, soltanto qui ogni componente dell'offset è ristretta a circa $(-4,4)$ celle della feature map, negli altri due modelli non esiste un limite esplicito nel codice.

L'inizializzazione a zero della convoluzione che predice offset e maschera produce offset iniziali nulli e maschera circa 0.5. Non rende però il FAM un'identità: i pesi della `DeformConv2d` rimangono convoluzionali e apprendibili. Il suo scopo è rendere più probabile una corrispondenza spaziale utile prima della rettifica CMX, non garantire un allineamento pixel-perfect.

### FFM con coordinate 2D

Nel ramo ibrido un *2D sine positional encoding* con temperatura 10000 viene aggiunto a RGB e IR prima delle proiezioni query e key. `GroupNorm(32)` normalizza query, key e value; la value IR non riceve il positional encoding. Per P4/P5, Q, K e V sono calcolati in `float32` dentro `scaled_dot_product_attention` e poi riportati al dtype originale. In questo modo l'attenzione mantiene informazione sulle coordinate, riducendo il rischio di associare regioni semanticamente simili ma spazialmente errate (*attention ghosting*).

Il bypass P3 resta necessario per evitare il costo quadratico, ma ora riceve feature IR già trasformate dal FAM. È un'ipotesi architetturale più favorevole della concatenazione di mappe palesemente disallineate, non una prova che il bypass sia innocuo per ogni caso.

## Risultati della variante ibrida

L'ibrido recupera una parte consistente del collasso del CMX puro: in fusione passa da **0.0320** a **0.1447 mAP@50** a soglia 0.01 (+352% relativo) e produce risultati non nulli anche a soglia 0.10. Resta però nettamente sotto il baseline RT-DETR (0.3570): FAM e positional encoding migliorano questa specifica integrazione CMX, senza renderla competitiva con la fusione additiva semplice sul dataset valutato.

| Modello | Test | Soglia | mAP@50 |
| --- | --- | ---: | ---: |
| CMX ibrido | $D_{RGB-IR}$ | 0.01 | 0.1447 |
| CMX ibrido | $D_{RGB-IR}$ | 0.10 | 0.0878 |
| CMX ibrido | $D_{IRo}$ | 0.01 | 0.0944 |
| CMX ibrido | $D_{IRo}$ | 0.10 | 0.0543 |
| CMX ibrido | $D_{VISo}$ | 0.01 | 0.1023 |
| CMX ibrido | $D_{VISo}$ | 0.10 | 0.0607 |
| CMX puro | $D_{RGB-IR}$ | 0.01 | 0.0320 |
| RT-DETR baseline | $D_{RGB-IR}$ | 0.01 | 0.3570 |
