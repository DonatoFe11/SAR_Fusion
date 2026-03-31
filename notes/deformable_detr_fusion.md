# Deformable DETR Fusion per RGB-IR

Questo documento esplora l'implementazione del modello Deformable DETR adattato per accogliere immagini multi-modali (RGB + Infrarosso). Il codice di riferimento è: [sarfusion/models/deformable_detr_fusion.py](../sarfusion/models/deformable_detr_fusion.py).

Rispetto all'architettura CMX o a quella basata sul Feature Alignment Module (FAM), questa versione si distingue per essere un'implementazione **"Early/Mid Fusion" molto più semplice, leggera e diretta**, basata esclusivamente sulla concatenazione lineare dei canali.

## 1. La Dual Backbone e il partizionamento modale

Il cuore di questa modifica è la classe `DeformableDetrFusionBackbone`. Quando il tensore delle immagini in ingresso a 4 canali (`[Batch, 4, H, W]`) raggiunge il layer iniziale, viene diviso in due flussi indipendenti:
- **RGB Backbone**: riceve i primi 3 canali (`pixel_values[:, :3]`).
- **IR Backbone**: riceve il quarto canale (`pixel_values[:, 3:]`).

Entrambe le dorsali sono istanze normali di `DeformableDetrConvModel` (tipicamente reti ResNet). Insieme all'estrazione semantica dei pixel, queste dorsali generano anche le **Positional Embeddings** (codifiche posizionali), essenziali per mantenere le informazioni sulle coordinate spaziali ($x, y$) prima di passare il tutto al Transformer.

## 2. Strategia di Fusione (Channel Concatenation & Linear Projection)

Il meccanismo di fusione in questo modello non fa uso dell'Heavy Cross-Attention (come CMX) né delle Deformable Convolutions (come in RT-DETR Fusion). Impiega invece un approccio sequenziale estremamente collaudato e rapido:

1. **Concatenazione (Channel-wise)**: 
   Ad ogni singola scala piramidale (feature level), la feature map RGB (di dimensione $C$) e la feature map IR (di dimensione $C$) vengono "incollate" lungo l'asse dei canali.
   ```python
   fused_feat = torch.cat([rgb_feat, ir_feat], dim=1) # Risultato: [B, 2C, H, W]
   ```
2. **Proiezione Lineare ($1 \times 1$ Convolution)**:
   Il tensore concatenato a doppia capacità ($2C$) viene processato da una conv $1 \times 1$ appresa, chiamata `channel_proj`. 
   Il ruolo di questo layer è comprimere e "mischiare" linearmente le informazioni per ridurla di nuovo alla dimensione attesa dal Transformer ($C$ canali). È qui che la rete impara attivamente quali tratti spaziali/canali dell'IR privilegiare rispetto a quelli dell'RGB.
   ```python
   fused_feat = self.channel_proj[level_idx](fused_feat) # Risultato: [B, C, H, W]
   ```

3. **Positional Embeddings e Mask condivise**:
   Non c'è motivo di fondere le maschere o il position embedding, dato che lo spazio fisico ($H, W$) descritto dai due sensori è identico (le immagini sono registrate/allineate a monte). Il codice sceglie in automatico di propagare quelli calcolati per l'RGB (`rgb_pos`, `rgb_mask`) alla fase finale.

## 3. Gestione e Inizializzazione dei Pesi (`from_pretrained`)

Analogamente alle versioni di RT-DETR modificate, si rende necessario un Override per "ingannare" i controlli di sicurezza di Hugging Face legati al numero di canali.

Nella classe `DeformableDetrFusionForObjectDetection`:
1. Durante lo scoping iniziale (nel costruttore), dichiariamo fittiziamente alla configurazione di avere `3 canali`, permettendo il bootstrap predefinito di `super()`.
2. Assegniamo la nostra nuova dorsale fusa (con 4 canali reali in entrata).
3. Estendiamo il metodo `.from_pretrained()` per pre-caricare i pesi allenati su enormi moli di dati (come COCO). 
   Per la dorsale Infrarosso (che si aspetta solo 1 canale in ingresso), si sfrutta il **"trick della media dei filtri"**: il tensore del primissimo layer della ResNet originaria (shape $64 \times 3 \times 7 \times 7$) viene schiacciato con un'operazione di media lineare `.mean(dim=1, keepdim=True)` ottenendo $64 \times 1 \times 7 \times 7$. Così facendo, le feature di base (come edge detection) già apprese in RGB non vengono perse, ma trasferite all'Infrarosso come forte punto di partenza per l'ottimizzazione.

## 4. Feature Levels (Riduzione Multi-scala per Velocità)

Un dettaglio specifico di questa implementazione risiede nella potatura esplicita delle scale geometriche tramite il parametro `config.num_feature_levels`.

Nei rilevatori di oggetti a piramide (Feature Pyramid Network), la Backbone emette più mappe di risoluzioni diverse:
- **Level 0 (C3)**: Mappa molto grande e ricca di dettagli (risoluzione $H/8 \times W/8$), serve per rilevare oggetti microscopici.
- **Level 1 (C4)**: Mappa di medie dimensioni ($H/16 \times W/16$), per oggetti normali.
- **Level 2 (C5)**: Mappa molto piccola ($H/32 \times W/32$), per comprendere il contesto e gli oggetti enormi.
- **Level 3 (Extra)**: Spesso aggiunta per oggetti mastodontici ($H/64 \times W/64$).

Nel codice `deformable_detr_fusion.py` troviamo queste righe critiche durante l'estrazione:
```python
rgb_features = rgb_features[:self.num_feature_levels]
```
Se il modello Deformable DETR viene impostato con `num_feature_levels = 2` invece dei canonici "4", lo slicing `[:2]` della lista "taglia via e butta" proattivamente le feature a risoluzione altissima o bassissima dipendentemente da come è configurato.

**Qual è lo scopo?**
Il Transformer computa l'attenzione attraversando *tutti* i pixel di queste mappe. La mappa più grande (es. la $C3$) contiene da sola il $75\%$ dei pixel totali dell'intera piramide! Rimuovendola scartando i feature level in eccesso, si ottiene un boost vertiginoso in FPS (Frames Per Second) e un crollo dell'uso di VRAM. Questo taglio di precisione è fondamentale per deploy hardware su edge devices (come le telecamere dei droni SAR), dove c'è un rigido trade-off: sacrificare l'abilità di trovare una scarpa a centinaia di metri di distanza, per poter processare in tempo reale il video stabilizzato alla ricerca di sagome umane (medie dimensioni).