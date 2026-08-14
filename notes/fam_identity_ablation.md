# Ablation FAM: DCNv2 identity e warp diretto

> **Stato.** Le vecchie ablation a seed 42 sono state sostituite dalla campagna
> finale a cinque seed e checkpoint finale. I risultati confermativi sono
> riportati in questa nota; i dettagli completi sono in
> [`rtdetr_reproducibility.md`](rtdetr_reproducibility.md).

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
python -m unittest discover -s tests -p 'test_rtdetr_fam_identity.py'
```

I training finali sono già completati; il protocollo è riportato nella sezione
seguente.

## Protocollo finale

Le due varianti sono state addestrate separatamente con lo stesso protocollo
usato per Additive e FAM standard:

- seed appaiati `40–44`;
- un nuovo processo per ogni seed;
- 10 epoche fisse;
- AdamW con learning rate `2e-5`;
- checkpoint COCO `PekingU/rtdetr_r50vd` e riuso della riga `person` nelle
  teste single-class;
- Modal Dropout IR/RGB/fusion 20/20/60 soltanto nel training;
- nessuna validation usata per la selezione;
- test del checkpoint finale `latest`;
- valutazioni successive VIS, IR e VIS+IR con Modal Dropout disattivato.

I YAML separati usati per queste campagne e per le valutazioni delle modalità
sono stati rimossi dopo il completamento. Per un'eventuale replica si parte dal
template comune `parameters/RTDETR/rtdetr_protocol.yaml` e si imposta una sola
variante alla volta come indicato nei commenti del file.

## Risultati finali

### VIS+IR per seed

| Seed | FAM corrente | Identity DCNv2 | Grid Sample |
|---:|---:|---:|---:|
| 40 | 0.3783 | 0.4336 | 0.4050 |
| 41 | 0.4335 | 0.2113 | 0.3483 |
| 42 | 0.3129 | 0.2926 | 0.4083 |
| 43 | 0.3964 | 0.2860 | 0.3326 |
| 44 | 0.3690 | 0.4162 | 0.3564 |

| Variante | Media | Mediana | Dev. std. | Min–max | IC 95% media |
|---|---:|---:|---:|---:|---:|
| FAM corrente | 0.3780 | 0.3783 | 0.0440 | 0.3129–0.4335 | 0.3234–0.4326 |
| Identity DCNv2 | 0.3280 | 0.2926 | 0.0943 | 0.2113–0.4336 | 0.2109–0.4451 |
| Grid Sample | 0.3701 | 0.3564 | 0.0345 | 0.3326–0.4083 | 0.3273–0.4129 |

### Medie per modalità

| Variante | VIS | IR | VIS+IR |
|---|---:|---:|---:|
| FAM corrente | 0.2442 | 0.2028 | 0.3780 |
| Identity DCNv2 | 0.2150 | 0.1768 | 0.3281 |
| Grid Sample | **0.2614** | 0.1895 | 0.3701 |

Nel confronto appaiato per seed:

- Identity DCNv2 − FAM ha un delta medio VIS+IR di `−0.0501`, vince in 2/5
  seed e presenta la dispersione più alta;
- Grid Sample − FAM ha un delta medio di `−0.0079`, vince in 2/5 seed e ha un
  IC 95% della differenza pari a `[−0.0979, +0.0820]`;
- Grid Sample supera FAM in media su VIS di `+0.0172`, ma non su IR
  (`−0.0133`) né su VIS+IR (`−0.0079`).

L'inizializzazione identità non costituisce quindi una soluzione alla
variabilità e non va promossa a variante principale. Il warp diretto, privo
del filtraggio DCNv2, è prestazionalmente vicino al FAM corrente e supporta
l'ipotesi che una componente geometrica sia utile; con cinque seed, però, non
è possibile attribuire causalmente tutto il vantaggio del FAM
all'allineamento né dichiarare equivalenti le due implementazioni.

## Risultato preliminare storico

La precedente run a seed 42 aveva prodotto `0.3775` per Identity e `0.3918`
per Grid Sample in VIS+IR, facendo apparire Grid Sample quasi identico al FAM
storico (`0.3960`) e molto più forte in IR-only (`0.3473`). La campagna finale
mostra perché quel confronto non era sufficiente: l'Identity varia da `0.2113`
a `0.4336` e il vantaggio IR-only di Grid Sample non si ripete nella media dei
cinque seed. I numeri storici possono essere citati soltanto come motivazione
dell'ablation finale, non come risultato conclusivo.
