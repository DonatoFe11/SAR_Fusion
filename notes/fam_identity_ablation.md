# Ablation FAM: DCNv2 identity e warp diretto

Questa nota descrive le due varianti aggiunte a RT-DETR per separare l'effetto
dell'allineamento geometrico da quello della convoluzione aggiuntiva del FAM.
Il comportamento storico resta disponibile come `fam_variant:
current_dcnv2` ed è ancora il valore predefinito, così i checkpoint del Modello
B non cambiano significato.

## Varianti

### `identity_dcnv2`

La struttura coincide con il FAM DCNv2 corrente: una convoluzione sulle
feature concatenate RGB--IR predice 18 offset e 9 logit di mask, poi una
`DeformConv2d` 3x3 trasforma l'IR. L'inizializzazione differisce come segue:

- pesi e bias del predittore di offset/mask a zero;
- pesi DCNv2 nulli fuori dal centro del kernel;
- matrice diagonale tra canali al centro;
- valore della diagonale pari a 2;
- bias DCNv2 nullo.

Poiché i logit iniziali della mask sono zero, la mask vale
`sigmoid(0) = 0.5`. La diagonale pari a 2 compensa esattamente questo fattore:
prima del primo aggiornamento il solo campione centrale contribuisce
all'uscita e `FAM(RGB, IR) = IR`.

### `grid_sample`

Una convoluzione 3x3 sulle feature concatenate predice un solo spostamento
`(dx, dy)` per ogni posizione. Gli spostamenti sono espressi in pixel della
feature map e convertiti nelle coordinate normalizzate richieste da
`grid_sample`. Il warp usa:

- `mode="bilinear"`;
- `padding_mode="zeros"`;
- `align_corners=False`;
- griglia base costruita sui centri dei pixel;
- predittore degli offset inizializzato a zero.

Questa variante non contiene una convoluzione che filtra l'IR dopo il
campionamento: il suo contributo appreso è soltanto geometrico.
Poiché `grid_sample` in float32 può introdurre un errore di interpolazione
dell'ordine di qualche `1e-5` su dimensioni dispari anche usando la griglia
corretta, l'output sottrae il campionamento numerico della griglia base:

```text
IR + grid_sample(IR, identity_grid + offset)
   - grid_sample(IR, identity_grid)
```

A offset zero i due campionamenti sono identici e si cancellano esattamente.
Il ramo deformato resta nel grafo, quindi il predittore degli offset continua
a ricevere gradienti e può abbandonare l'identità dopo il primo update.

Entrambe le varianti rifiutano `spatial_jitter_std != 0`, perché lo SSJ
romperebbe l'identità nello stato iniziale e introdurrebbe una seconda
variabile nell'ablation.

## Verifica numerica

La suite `tests/test_rtdetr_fam_identity.py` controlla entrambe le varianti su
dimensioni pari e dispari analoghe ai livelli P3, P4 e P5:

- shape invariata;
- valori finiti;
- errore assoluto massimo inferiore a `1e-5`;
- errore L2 relativo inferiore a `1e-5`;
- identità invariata su forward ripetuti senza update;
- gradienti finiti sull'IR e sui parametri addestrabili;
- possibilità di abbandonare l'identità dopo un `optimizer.step()`;
- rifiuto esplicito dello SSJ.

Nell'ambiente `sarfusion`:

```bash
python -m unittest discover -s tests -p 'test_rtdetr_fam_identity.py' &&
python main.py experiment \
  --parameters parameters/RTDETR/fam_ablation_identity_grid.yaml
```

L'operatore `&&` impedisce l'avvio del training se anche un solo controllo
numerico fallisce.

## Training

La configurazione `parameters/RTDETR/fam_ablation_identity_grid.yaml` genera
due run, una per variante, mantenendo:

- seed 42;
- 10 epoche;
- AdamW con learning rate `2e-5`;
- pretrained `PekingU/rtdetr_r50vd`;
- single class e split WiSARD già usati;
- modal dropout IR/RGB/fusion pari a 20/20/60;
- nessuno spatial dropout e nessuno SSJ;
- selezione del checkpoint tramite mAP.

Dopo aver copiato i due checkpoint selezionati nei percorsi indicati in
`parameters/RTDETR/fam_ablation_eval.yaml`, la valutazione genera le sei run
correttamente accoppiate (tre modalità per ciascuna architettura):

```bash
python main.py experiment \
  --parameters parameters/RTDETR/fam_ablation_eval.yaml
```

## Risultati

| Variante RT-DETR | Fusion | IR-only | VIS-only |
|---|---:|---:|---:|
| Additive Fusion senza FAM | 0.357 | 0.252 | 0.246 |
| FAM DCNv2 corrente, Modello B | 0.396 | 0.263 | 0.262 |
| FAM DCNv2 identity | 0.3775 | 0.2778 | 0.2495 |
| Warp diretto `grid_sample` | 0.3918 | 0.3473 | 0.2554 |

Rispetto all'Additive Fusion, la DCNv2 identity guadagna `+0.0205` in
Fusion, `+0.0258` in IR-only e `+0.0035` in VIS-only. Il warp diretto
guadagna rispettivamente `+0.0348`, `+0.0953` e `+0.0094`.

Il confronto più informativo è tra le tre varianti FAM:

- `grid_sample` supera la DCNv2 identity di `+0.0143` in Fusion, `+0.0695`
  in IR-only e `+0.0059` in VIS-only;
- rispetto alla DCNv2 corrente, `grid_sample` è inferiore di appena `0.0042`
  in Fusion e di `0.0066` in VIS-only, ma superiore di `0.0843` in IR-only;
- la DCNv2 identity è inferiore alla DCNv2 corrente di `0.0185` in Fusion e
  `0.0125` in VIS-only, pur migliorando IR-only di `0.0148`.

Il warp privo di filtraggio convoluzionale recupera quindi quasi interamente
la prestazione Fusion della DCNv2 corrente e produce la migliore robustezza
IR-only. Questo risultato è coerente con l'ipotesi che una parte sostanziale
del beneficio del FAM provenga dall'allineamento geometrico, senza richiedere
la trasformazione convoluzionale aggiuntiva. La differenza residua di `0.0042`
in Fusion è troppo piccola, su una singola run, per attribuire con sicurezza
un vantaggio alla DCNv2 corrente. Analogamente, la prestazione più bassa della
DCNv2 identity mostra che l'inizializzazione modifica la dinamica di
ottimizzazione, ma non dimostra da sola che il filtraggio convoluzionale sia
la causa del vantaggio. Servirebbero seed multipli per conclusioni
statisticamente robuste.
