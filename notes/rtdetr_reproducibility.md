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

### Prima run deterministica completa

La run `d19h9d3h` ha completato dieci epoche in circa 13 ore e 21 minuti:

| Metrica | Valore |
|---|---:|
| test mAP@50 | 0.3844 |
| test mAP@50:75 | 0.1312 |
| loss media train, epoca 10 | 6.4535 |
| validation mAP@50, epoca 10 | 0.000038 |

Il risultato test è compreso tra quelli delle due run CUDA native complete
(`0.2934–0.4171`), quindi il percorso deterministico non mostra una perdita
anomala di accuratezza. La validation quasi nulla conferma invece che lo split
attuale può essere registrato come diagnostica, ma non usato per scegliere il
checkpoint. Il controllo conclusivo consiste nel rilanciare la stessa
configurazione e confrontare trace, metriche finali e hash del checkpoint.

La replica `pso8m75j` ha poi confermato la riproducibilità completa: tutti i
402 eventi dei trace coincidono, tutte le metriche train/validation/test sono
identiche e i due checkpoint finali hanno lo stesso SHA-256
`d05d553c555674bc7a045c83b8fe40ec856382e606d54e4bc30c10475252dfa0`.

## Limite deterministico del FAM DCNv2

Il probe strict FAM `amw689bj` si arresta al primo backward con l'errore
`compute_grad_input does not have a deterministic implementation`.
L'operatore incompatibile è `torchvision.ops.DeformConv2d`, usato dalla
variante `current_dcnv2`. Un equivalente PyTorch basato su sampling bilineare
ha confermato l'equivalenza numerica su CPU, ma sui livelli da 512, 1024 e 2048
canali richiede più di due minuti per il primo batch ed è quindi inadatto a un
training completo. Il fallback sperimentale non è stato mantenuto.

Tre probe FAM CUDA native same-seed (`vaiwp67x`, `5nbzxxma`, `axvu0yll`)
mostrano che DCNv2 amplifica ulteriormente il non determinismo:

| Probe | Range medio loss, 20 step | Range massimo |
|---|---:|---:|
| base nativo, testa pretrained | 0.4393 | 1.6424 |
| FAM DCNv2 nativo, testa pretrained | 1.9942 | 5.8039 |

Anche nel FAM input, inizializzazione e prima loss coincidono, ma gli hash dei
pesi divergono al primo optimizer step. Un singolo full training FAM nativo
non può quindi sostenere un confronto architetturale affidabile. Le alternative
sono un kernel DCNv2 CUDA deterministico efficiente, una diversa architettura
di allineamento oppure un protocollo statistico con repliche native esplicite.

La variante `identity_dcnv2` è stata verificata separatamente con i probe
`35v9lmpk`, `78wb16t0` e `q8yd4qa6`. Mantiene DCNv2, ma inizializza la sua
mappatura esattamente come identità. La loss iniziale torna da `34.9876` a
`19.8649`, uguale al base, e il range medio sui primi 20 step scende da
`1.9942` a `0.8248`. Resta comunque non deterministica e presenta un range
massimo di `6.4395`; va quindi considerata una variante architetturale distinta
e non una soluzione deterministica per `current_dcnv2`.

## Protocollo finale della tesi

La campagna confermativa è separata in sei esperimenti autonomi. Tutti usano
CUDA nativa, cinque seed appaiati (`40–44`), un nuovo processo per ogni seed,
testa COCO `person`, Modal Dropout soltanto sul training, dieci epoche fisse e
test del solo checkpoint finale `latest`. La validation corrente non viene
eseguita durante il training e non seleziona checkpoint.

| N. | File | Configurazione | Progetto W&B | Run |
|---:|---|---|---|---:|
| 1 | `rtdetr_additive.yaml` | Base/Additive | `RTDETR_Protocol` | 5 |
| 2 | `rtdetr_fam.yaml` | FAM `current_dcnv2` | `RTDETR_FAM_Protocol` | 5 |
| 3 | `rtdetr_fam_ir_dropout.yaml` | FAM + IR Dropout 0.4 | `RTDETR_FAM_IR_Dropout_Protocol` | 5 |
| 4 | `rtdetr_fam_ssj.yaml` | FAM + SSJ 0.5 | `RTDETR_FAM_SSJ_Protocol` | 5 |
| 5 | `rtdetr_ablation_identity_dcnv2.yaml` | ablation `identity_dcnv2` | `RTDETR_FAM_Identity_DCNv2_Ablation` | 5 |
| 6 | `rtdetr_ablation_grid_sample.yaml` | ablation `grid_sample` | `RTDETR_FAM_Grid_Sample_Ablation` | 5 |

Ogni file espande quindi esattamente cinque run sequenziali. I seed occupano
le posizioni `start_from_run` da `0` a `4`; `start_from_grid` resta `0` perché
ogni file contiene una sola configurazione. Prima di una ripresa vanno
controllate le run effettivamente concluse su W&B per evitare duplicati.

Il controllo Frozen Random non fa parte del protocollo: congelerebbe una
trasformazione DCNv2 inizializzata casualmente e non un modulo FAM pretrained,
quindi non risponderebbe in modo pulito alla domanda sperimentale della tesi.
I precedenti YAML `rerun`, i vecchi YAML `ablation`, il protocollo combinato e
i file temporanei di probe/smoke sono stati rimossi perché sostituiti da questi
sei file. Sono stati invece conservati i YAML diagnostici del modello base,
che documentano le verifiche di riproducibilità.

Prima della separazione, uno smoke test del protocollo ha completato due batch
per ognuna delle sei configurazioni il 4 agosto 2026. I singoli esperimenti si
avviano con:

```bash
python main.py experiment --parameters parameters/RTDETR/rtdetr_additive.yaml
python main.py experiment --parameters parameters/RTDETR/rtdetr_fam.yaml
python main.py experiment --parameters parameters/RTDETR/rtdetr_fam_ir_dropout.yaml
python main.py experiment --parameters parameters/RTDETR/rtdetr_fam_ssj.yaml
python main.py experiment --parameters parameters/RTDETR/rtdetr_ablation_identity_dcnv2.yaml
python main.py experiment --parameters parameters/RTDETR/rtdetr_ablation_grid_sample.yaml
```

Su una singola GPU gli esperimenti vanno eseguiti uno alla volta.

## Valutazione delle modalità

Dopo i 30 training, ciascuno dei checkpoint finali viene rivalutato con seed
di evaluation fisso `42` sui tre input `vis`, `ir` e `vis_ir`. Il seed di
training `40–44` identifica soltanto il checkpoint sorgente e non viene
riutilizzato come seed della valutazione. Sebbene `vis_ir` sia già testato alla
fine del training, viene ripetuto qui con lo stesso protocollo, batch size e
organizzazione W&B delle modalità singole. La campagna contiene 30 test per
modalità, per un totale di 90 valutazioni.

| Configurazione | File | Progetto W&B | Test |
|---|---|---|---:|
| Base/Additive | `rtdetr_additive_modality_evaluation.yaml` | `RTDETR_Additive_Modality_Evaluation` | 15 |
| FAM `current_dcnv2` | `rtdetr_fam_modality_evaluation.yaml` | `RTDETR_FAM_Modality_Evaluation` | 15 |
| FAM + IR Dropout | `rtdetr_fam_ir_dropout_modality_evaluation.yaml` | `RTDETR_FAM_IR_Dropout_Modality_Evaluation` | 15 |
| FAM + SSJ | `rtdetr_fam_ssj_modality_evaluation.yaml` | `RTDETR_FAM_SSJ_Modality_Evaluation` | 15 |
| ablation `identity_dcnv2` | `rtdetr_ablation_identity_dcnv2_modality_evaluation.yaml` | `RTDETR_FAM_Identity_DCNv2_Modality_Evaluation` | 15 |
| ablation `grid_sample` | `rtdetr_ablation_grid_sample_modality_evaluation.yaml` | `RTDETR_FAM_Grid_Sample_Modality_Evaluation` | 15 |

Ogni file crea cinque riferimenti ai checkpoint e, per ciascuno, una run VIS,
una IR e una VIS+IR. I checkpoint sono individuati localmente tramite progetto
W&B, seed di training e nome `latest`; la risoluzione fallisce se il checkpoint
manca o se esistono più run complete compatibili. Il caricamento deve inoltre
corrispondere al 100% delle chiavi del modello. I test usano
`test_checkpoint: current` perché il checkpoint sorgente viene caricato
durante la costruzione del modello.

I file possono essere avviati soltanto dopo che tutti e cinque i training
della configurazione corrispondente sono terminati:

```bash
python main.py experiment --parameters parameters/RTDETR/rtdetr_additive_modality_evaluation.yaml
python main.py experiment --parameters parameters/RTDETR/rtdetr_fam_modality_evaluation.yaml
python main.py experiment --parameters parameters/RTDETR/rtdetr_fam_ir_dropout_modality_evaluation.yaml
python main.py experiment --parameters parameters/RTDETR/rtdetr_fam_ssj_modality_evaluation.yaml
python main.py experiment --parameters parameters/RTDETR/rtdetr_ablation_identity_dcnv2_modality_evaluation.yaml
python main.py experiment --parameters parameters/RTDETR/rtdetr_ablation_grid_sample_modality_evaluation.yaml
```
