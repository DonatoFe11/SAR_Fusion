# DINO Fusion per RGB-IR

Questa nota documenta l'implementazione **DINO completa** in [dino_fusion.py](../sarfusion/models/dino_fusion.py), da usare per i prossimi training. Essa mantiene backbone, FAM opzionale e channel fusion di [Deformable DETR Fusion](deformable_detr_fusion.md), ma sostituisce l'inizializzazione delle query e il decoder con le tre componenti proprie di DINO:

- **Contrastive DeNoising (CDN)** durante il training;
- **Mixed Query Selection (MQS)**;
- **Look-Forward-Twice (LFT)** con box refinement attivo.

I risultati riportati in fondo sono invece **storici**: sono stati prodotti dalla precedente variante CDN-only e non misurano questa nuova implementazione.

## Base RGB-IR e livelli di feature

La base resta `DeformableDetrFusionModel`:

```text
RGB [B, 3, H, W] ─┐
                  ├─ ResNet C3/C4/C5 → FAM opzionale → channel fusion
IR  [B, 1, H, W] ─┘
                                      ↓
                      quarto livello da Conv 3×3 stride 2 su C5 fusa
                                      ↓
                         encoder Deformable DETR a quattro livelli
```

Le due ResNet producono tre feature map native; FAM e la fusione di canale agiscono su queste tre coppie. Con `num_feature_levels=4`, Deformable DETR genera poi il quarto livello a risoluzione più bassa dalla C5 fusa. La scelta rimane coerente con il checkpoint `SenseTime/deformable-detr`.

## Query DINO: encoder a due stadi e Mixed Query Selection

All'interno di `DINOFusionForObjectDetection.from_pretrained`, il checkpoint Deformable DETR one-stage viene convertito per il nuovo modello impostando:

```python
config.two_stage = True
config.two_stage_num_proposals = config.num_queries  # 300
config.with_box_refine = True
```

L'encoder assegna classe e box a ogni proposta spaziale. Le 300 proposte con score di foreground più alto forniscono i **reference point** iniziali del decoder. MQS separa deliberatamente i due ruoli:

- le coordinate selezionate dall'encoder producono anchor e positional embedding;
- il contenuto delle 300 query proviene da `mixed_query_content`, una embedding appresa indipendente dalle feature dell'encoder.

Questa è la mixed query selection: anchor informativi dall'encoder, contenuto di query appreso. In inference esistono soltanto queste 300 query di matching.

## Look-Forward-Twice (LFT)

Il decoder usa box refinement a ogni layer. DINO distingue il reference point
passato all'attenzione del layer successivo dalla copia usata per calcolare la
predizione successiva:

```python
reference_points = new_reference_points.detach()
intermediate_reference_points.append(new_reference_points)
```

La prima copia evita di retropropagare attraverso tutti i successivi blocchi di
deformable attention; la seconda resta nel grafo e fa sì che la loss della
predizione al layer seguente aggiorni anche la testa box del layer precedente.
Questo collegamento fra loss adiacenti è il Look-Forward-Twice. Poiché
`with_box_refine=True` viene forzato, il refinement viene realmente eseguito.
Le nuove teste di proposal dell'encoder e la settima copia delle teste (una per
l'encoder oltre ai sei layer decoder) sono inizializzate ex novo; i pesi
compatibili del checkpoint vengono comunque riutilizzati.

## Contrastive DeNoising (CDN)

CDN è attivo solo durante il training e se il batch contiene annotazioni. Con $M$ ground truth massime nel batch e $G =$ `num_dn_groups`, vengono aggiunte:

$$N_{DN} = 2 \cdot G \cdot M$$

query di denoising prima delle 300 matching query. Nella configurazione corrente $G=5$.

Per ciascun gruppo vengono create copie positive (rumore spaziale ridotto e label noise) e negative (rumore più ampio e target no-object). Il decoder riceve una maschera additiva quadrata che:

- blocca gli scambi CDN ↔ matching;
- blocca gli scambi tra gruppi CDN diversi;
- consente le interazioni all'interno di un gruppo CDN e fra matching query.

La self-attention del decoder è estesa esplicitamente per consumare questa maschera: non è solo costruita, ma effettivamente applicata. Le query CDN usano come positional embedding la stessa trasformazione degli anchor a quattro coordinate usata dalle query DINO; la loss CDN ricostruisce il box ground truth originale a partire dal reference point perturbato. Comprende classificazione, L1 e GIoU ed è moltiplicata per `cdn_loss_coef`.

Gli slot CDN sono rimossi prima delle predizioni e della valutazione standard. In inference non vengono creati, quindi il costo e l'interfaccia restano quelli delle 300 query di matching.

## Inizializzazione e configurazione

Il file da lanciare è [fusion_dino.yaml](../parameters/DINO/fusion_dino.yaml), rinominato in modo da non confondere le nuove run con quelle storiche:

| Parametro | Valore |
| --- | --- |
| esperimento | `DINO_Full_Fusion_DefDETR_FAM_SSJ_vis_ir` |
| `num_feature_levels` | 4 |
| query di matching | 300 (`config.num_queries`) |
| `num_dn_groups` | 5 |
| `label_noise_prob` | 0.5 |
| `box_noise_scale` | 1.0 |
| `cdn_loss_coef` | 1.0 |
| `two_stage` | `true`, impostato dal codice DINO |
| `with_box_refine` | `true`, impostato dal codice DINO |
| `batch_size` | 1 |
| modal dropout (IR / RGB / fusion) | 0.2 / 0.2 / 0.6 |

Il checkpoint COCO inizializza backbone RGB, backbone IR adattata mediando i tre canali della prima convoluzione, encoder, decoder e pesi compatibili delle teste. I blocchi di channel fusion, FAM, embedding CDN, contenuto MQS e moduli aggiuntivi two-stage sono appresi nei nuovi training.

## Risultati storici: non DINO completo

Le seguenti run erano etichettate nei report come “DINO (CDN + LFT)”, ma usavano il codice precedente: `with_box_refine=false`, nessuna MQS e una maschera CDN che non raggiungeva la self-attention. Vanno quindi lette come risultati di una variante **CDN legacy**, non come risultati del modello definito sopra. Sono mediana [min--max] su cinque seed, mAP@50 a soglia 0.01.

| Configurazione storica | VIS-only | IR-only | RGB-IR |
| --- | --- | --- | --- |
| CDN legacy | 0.177 [0.139--0.217] | 0.116 [0.077--0.146] | 0.251 [0.229--0.302] |
| CDN legacy + FAM | 0.141 [0.117--0.175] | 0.113 [0.096--0.154] | 0.265 [0.209--0.299] |
| CDN legacy + FAM + SSJ | 0.149 [0.075--0.193] | 0.071 [0.062--0.132] | 0.243 [0.105--0.277] |

Non si devono confrontare questi valori come ablation di LFT o MQS: entrambe le componenti mancavano o erano inattive. Le prossime run con `DINO_Full_*` costituiranno il primo risultato sperimentale del DINO completo nel progetto.
