# YOLO26 + FAM — protocollo Stage A

## Scopo e interpretazione

La domanda primaria è: **sullo stesso YOLO26s dual-backbone, il FAM standard
migliora la fusione Additive?**  Questo non è ancora un confronto quantitativo
YOLO26-vs-YOLOv10: il vecchio protocollo YOLOv10 usava dati, orizzonte e regola
di checkpoint differenti.

La validation FHL 0401/0402 è già stata consultata negli esperimenti precedenti.
Va quindi descritta come *development validation interna*, non come nuovo
holdout di conferma. MtErie, Carnation, confirmation set e stress geometrico
sono esclusi da training, tuning e gate Stage A.

## Software, pesi e architettura congelati

- ambiente separato: `sarfusion-yolo26`;
- Ultralytics ufficiale `8.4.138`, tag commit
  `dad7bb4534c95021bc14969ab25d77b77c4efdc3`;
- PyTorch `2.4.0`, torchvision `0.19.0`;
- checkpoint COCO `yolo26s.pt`, SHA256
  `646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b`;
- YOLO26s P3 standard, non P2 (non esistono pesi P2 ufficiali);
- backbone RGB e IR separati, layer 0–10; stem IR inizializzato con la media
  dei tre canali dello stem RGB;
- fusione delle feature ai layer 4, 6 e 10 (P3/P4/P5, 256/256/512 canali),
  seguita da neck e head end-to-end ufficiali;
- loss YOLO26 nativa `E2ELoss`, inclusi Progressive Loss e STAL; nessun riuso
  di `v10DetectLoss`;
- FAM `current_dcnv2`, SSJ=0, consistency e Box-Guided disattivati.

Il controllo e il candidato costruiscono entrambi i tre moduli FAM; nel
controllo vengono soltanto bypassati. A parità di seed devono coincidere gli
hash dei parametri condivisi, del backbone IR e dei FAM prima del primo step.

Riferimenti ufficiali: [modello YOLO26](https://docs.ultralytics.com/models/yolo26),
[training recipe](https://docs.ultralytics.com/guides/yolo26-training-recipe),
[sorgente v8.4.138](https://github.com/ultralytics/ultralytics/tree/v8.4.138).

## Dati congelati

- train: FHL VIS 0405 / IR 0406 + Baker Enterprise VIS 1 / IR 1,
  **3.123 coppie**;
- development-validation: FHL VIS 0401 / IR 0402, **896 coppie**;
- pairing per indice frame, senza sovrapposizione di sequenza;
- sono esclusi perché privi del partner: frame RGB FHL 0405 `943,944` e
  frame RGB FHL 0401 `896`;
- SHA256 su path relativi, dimensioni e byte di RGB, IR e annotazioni:
  `c92b43c2f69b35e61816f8c2c95f8bea76bec2a1eaa112f36f4303ed92bbdbaf`;
- classe singola `person`; label VIS; adattamento IR storico: decode BGR del
  JPEG, selezione del canale B, resize preservando l'aspect ratio all'altezza
  VIS e padding orizzontale simmetrico;
- modal dropout feature-gated `[IR-only, RGB-only, fusion] = [0.2, 0.2, 0.6]`.

## Recipe Stage A

- seed 40; 50 epoche complete; nessun early stopping;
- `imgsz=640`, batch fisico 4, `nbs=16` (accumulo effettivo a 16);
- AdamW esplicito, `lr0=0.001`, `lrf=0.01`, weight decay `0.0005`, warmup 3;
- AMP FP16, seed e ordine dei dati fissati, `compile=false`,
  `channels_last=false`, cache disattivata. Il determinismo algoritmico CUDA è
  disattivato in entrambi i bracci perché il backward di
  `torchvision::DeformConv2d` non dispone di un kernel deterministico; usare
  `warn_only` sarebbe una dichiarazione di determinismo fuorviante;
- HSV, mosaic, mixup, cutmix e copy-paste a zero; translate `0.1`, scale
  `0.5`, horizontal flip `0.5` applicati con la stessa geometria ai 4 canali;
- validation completa a ogni epoca; nessuna valutazione test automatica;
- `best.pt` scelto solo su mAP@50, miglioramento strettamente maggiore di
  `0.001`; in quasi-parità rimane il primo checkpoint;
- replay successivo dei 896 frame e controllo stretto del checkpoint.

Il replay usa una tolleranza assoluta mAP@50 preregistrata di `0,003`. Il
checkpoint Ultralytics serializza l'EMA in FP16, mentre la validation durante
il training usa l'EMA viva; la tolleranza copre questo percorso numerico ed è
comunque inferiore al guadagno minimo di promozione `0,010`. La validazione
finale già eseguita da Ultralytics ricaricando `best.pt` costituisce il replay:
il runner la registra e dichiara il run non integro se supera la tolleranza.
L'audit deve osservare `validation_replay=passed` prima di autorizzare FAM.

Prima del training il runner verifica versione, manifest sorgenti, hash dei
dati e dei pesi, parità RGB-only, inizializzazione accoppiata ed esegue un
forward/backward/update FP16 del candidato FAM a batch 4 includendo una copia
EMA. Un fallimento ferma il comando prima di produrre metriche scientifiche.
La riduzione automatica del batch viene considerata invalidante.

## Ordine e gate preregistrati

1. training Additive seed 40;
2. audit di completamento (50/50 epoche, valori finiti, inventory/trace,
   `best.pt` e `last.pt`, replay del best);
3. solo se il controllo è integro: training FAM seed 40;
4. `delta40 = best mAP50(FAM) - best mAP50(Additive)`;
5. promozione ai seed 41–44 solo se `delta40 >= +0.010000` e tutti i gate di
   integrità/update FAM passano;
6. gate a cinque seed: media dei delta almeno `+0.010000`, almeno 4/5 delta
   positivi e integrità 5/5. Solo allora si considera uno Stage B.

mAP50-95, mAP75 e prestazioni mono-modali sono secondarie e non possono
salvare il fallimento del gate primario. Se il gate seed 40 fallisce, gli altri
seed e lo Stage B vengono chiusi come non applicabili.
