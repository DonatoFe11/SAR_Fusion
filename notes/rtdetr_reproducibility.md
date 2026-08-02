# RT-DETR base: protocollo di riproducibilità

## Evidenze già disponibili

La validation e il test non provengono dalla stessa distribuzione:

| Split | Sequenze | Frame | Frame vuoti | Box/frame | Area mediana box |
|---|---:|---:|---:|---:|---:|
| train | 3 | 4022 | 17.85% | 2.464 | 0.1641% |
| validation | 1 | 273 | 67.40% | 0.542 | 0.0133% |
| test | 3 | 708 | 2.68% | 2.500 | 0.0764% |

La validation contiene una sola sequenza e oggetti circa 12 volte più piccoli
rispetto al train in termini di area mediana. Non va usata per scegliere il
checkpoint degli esperimenti correnti. Il checkpoint finale (`latest`, epoca
fissata a priori) è una scelta più corretta del `best` finché non viene definito
un nuovo protocollo di validation.

Inoltre il checkpoint COCO ha 80 classi, mentre il modello ha una sola classe.
Le sei teste `class_embed` e `denoising_class_embed` vengono inizializzate
casualmente da Transformers. Il seed cambia quindi sia i dati sia una parte
importante dell'inizializzazione del modello.

È ora disponibile `reuse_pretrained_class_head: true`: la riga COCO `person`
viene trasferita nelle sei `class_embed`, in `enc_score_head` e nella embedding
di denoising. Con questa opzione l'intero hash iniziale è identico usando seed
40, 41 o 42; non restano parametri inizializzati casualmente.

## Probe same-seed

Il file `parameters/RTDETR/rtdetr_base_reproducibility_probe.yaml` esegue tre
processi indipendenti con seed 42. Ogni processo allena solo 20 batch e non
esegue validation, test o salvataggio dei checkpoint.

```bash
conda run -n sarfusion python main.py experiment \
  --parameters parameters/RTDETR/rtdetr_base_reproducibility_probe.yaml
```

Ogni run produce `reproducibility_trace.jsonl` nella propria cartella W&B. Il
trace contiene:

- fingerprint del runtime e opzioni deterministiche;
- hash dei pesi iniziali e della testa di detection;
- indici dei sample e decisioni del Modal Dropout;
- hash degli input per i primi batch;
- loss e componenti della loss;
- hash dei pesi dopo i primi optimizer step.

I trace si confrontano con:

```bash
python scripts/compare_reproducibility_traces.py \
  wandb/run-.../files/reproducibility_trace.jsonl \
  wandb/run-.../files/reproducibility_trace.jsonl \
  wandb/run-.../files/reproducibility_trace.jsonl
```

Il probe sostituisce il `grid_sample` interno all'attenzione deformabile con
un'interpolazione bilineare equivalente e deterministica. Il normale training
non cambia; questo percorso viene attivato soltanto con
`reproducibility.deterministic: true`.

### Risultati eseguiti il 2 agosto 2026

Le run strict `19w5h1le`, `n3cdodmv` e `ihdcocrh` coincidono in tutti i 42
eventi del trace: inizializzazione, sample, Modal Dropout, input, loss e pesi
dopo gli optimizer step.

Le run CUDA native `evx7z10m`, `wsedqoj1` e `tj61x1r9` divergono invece al
primo optimizer step. La loss del primo forward è identica (`461.251495`), ma
gli hash dei pesi dopo il backward sono diversi. Già al batch successivo le
loss differiscono e al batch 14 il loro range raggiunge `20.8075`.

Il probe con testa COCO `person` (`btqbha68`, `yuy9g0k9`, `zdxk95k8`) conserva
la piccola divergenza del backward nativo, ma ne riduce fortemente
l'amplificazione:

| Inizializzazione | Loss iniziale | Range medio loss (20 step) | Range massimo |
|---|---:|---:|---:|
| testa casuale | 461.2515 | 4.0763 | 20.8075 |
| testa COCO person | 19.8649 | 0.4393 | 1.6424 |

La testa pretrained riduce il range medio di circa 9.3 volte e il massimo di
circa 12.7 volte, senza il rallentamento della modalità strict.

## Interpretazione e passo successivo

- Se i tre trace coincidono, il non determinismo same-seed è risolto. La
  varianza restante è sensibilità reale al seed.
- Se divergono già su `batch`, la causa è nel sampler, nei worker o nel Modal
  Dropout.
- Se batch e input coincidono ma diverge il primo `optimizer_step`, la causa è
  nel forward/backward o nell'optimizer.
- Se la modalità strict solleva un errore, il messaggio identifica direttamente
  un altro operatore CUDA senza implementazione deterministica.

Dopo un probe coincidente, le sorgenti del seed si possono separare senza altre
modifiche al codice:

- `reproducibility.model_seed`: inizializzazione del modello;
- `reproducibility.data_seed`: shuffle, worker e Modal Dropout;
- `reproducibility.training_seed`: operatori stocastici del modello.

Il confronto architetturale finale dovrebbe usare seed appaiati, checkpoint
`latest` a epoca fissata e riportare media, deviazione standard e intervalli di
confidenza. La validation attuale può restare una diagnostica out-of-domain, ma
non un selettore del checkpoint.

Le prime due run complete CUDA native con testa pretrained hanno confermato
che la mitigazione non basta per rendere stabile il risultato finale:

| Run | mAP@50 test | Loss media finale |
|---|---:|---:|
| `lecqa23d` | 0.4171 | 6.3060 |
| `tkmixj3u` | 0.2934 | 7.2605 |

La differenza di 0.1236 mAP@50 rende inutile una terza replica come tentativo
di validazione della stabilità. I trace hanno stessa loss iniziale
(`19.8649006`) e stessa testa dopo il primo aggiornamento, ma l'hash globale
dei pesi diverge già al primo `optimizer_step`, coerentemente con il backward
CUDA non deterministico dell'attenzione deformabile.

Il passo successivo è `rtdetr_base_deterministic_full.yaml`: una sola run di
dieci epoche con interpolazione deterministica, testa `person` pretrained,
validation a ogni epoca usata soltanto come diagnostica e valutazione del
checkpoint finale `latest`. La validation non seleziona il checkpoint: il
checkpoint viene salvato una sola volta a fine training. La run va ripetuta
identica solo se la sua accuratezza finale è accettabile.
