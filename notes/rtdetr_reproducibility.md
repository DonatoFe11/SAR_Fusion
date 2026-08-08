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

Il probe ha eseguito tre processi indipendenti con seed 42. Ogni processo ha
allenato solo 20 batch, senza validation, test o salvataggio dei checkpoint.
La configurazione YAML temporanea è stata rimossa dopo il completamento
dell'indagine; parametri, run W&B e risultati restano documentati qui.

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

Il passo successivo è stato una sola run di dieci epoche con interpolazione
deterministica, testa `person` pretrained, validation a ogni epoca usata
soltanto come diagnostica e valutazione del checkpoint finale `latest`. La
validation non selezionava il checkpoint, salvato una sola volta a fine
training.

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

| N. | Configurazione | Progetto W&B | Run |
|---:|---|---|---:|
| 1 | Base/Additive | `RTDETR_Protocol` | 5 |
| 2 | FAM `current_dcnv2` | `RTDETR_FAM_Protocol` | 5 |
| 3 | FAM + IR Dropout 0.4 | `RTDETR_FAM_IR_Dropout_Protocol` | 5 |
| 4 | FAM + SSJ 0.5 | `RTDETR_FAM_SSJ_Protocol` | 5 |
| 5 | ablation `identity_dcnv2` | `RTDETR_FAM_Identity_DCNv2_Ablation` | 5 |
| 6 | ablation `grid_sample` | `RTDETR_FAM_Grid_Sample_Ablation` | 5 |

Ciascuna campagna ha eseguito esattamente cinque run sequenziali. I seed
occupavano le posizioni `start_from_run` da `0` a `4`; `start_from_grid`
restava `0` perché ogni lancio conteneva una sola configurazione.

Il controllo Frozen Random non fa parte del protocollo: congelerebbe una
trasformazione DCNv2 inizializzata casualmente e non un modulo FAM pretrained,
quindi non risponderebbe in modo pulito alla domanda sperimentale della tesi.
I file separati usati per training, probe, smoke e valutazioni sono stati
rimossi dopo il completamento. Il repository conserva un solo template
generale, `parameters/RTDETR/rtdetr_protocol.yaml`, con il protocollo comune e
la tabella dei parametri da impostare per ricostruire ognuna delle sei
configurazioni. I risultati restano legati ai progetti e alle run W&B elencati.

Prima della separazione, uno smoke test del protocollo ha completato due batch
per ognuna delle sei configurazioni il 4 agosto 2026. Per un eventuale nuovo
esperimento si modifica il template per una sola configurazione e si avvia:

```bash
python main.py experiment --parameters parameters/RTDETR/rtdetr_protocol.yaml
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

| Configurazione | Progetto W&B | Test |
|---|---|---:|
| Base/Additive | `RTDETR_Additive_Modality_Evaluation` | 15 |
| FAM `current_dcnv2` | `RTDETR_FAM_Modality_Evaluation` | 15 |
| FAM + IR Dropout | `RTDETR_FAM_IR_Dropout_Modality_Evaluation` | 15 |
| FAM + SSJ | `RTDETR_FAM_SSJ_Modality_Evaluation` | 15 |
| ablation `identity_dcnv2` | `RTDETR_FAM_Identity_DCNv2_Modality_Evaluation` | 15 |
| ablation `grid_sample` | `RTDETR_FAM_Grid_Sample_Modality_Evaluation` | 15 |

Ogni file crea cinque riferimenti ai checkpoint e, per ciascuno, una run VIS,
una IR e una VIS+IR. I checkpoint sono individuati localmente tramite progetto
W&B, seed di training e nome `latest`; la risoluzione fallisce se il checkpoint
manca o se esistono più run complete compatibili. Il caricamento deve inoltre
corrispondere al 100% delle chiavi del modello. I test usano
`test_checkpoint: current` perché il checkpoint sorgente viene caricato
durante la costruzione del modello.

Le configurazioni di sola valutazione sono state rimosse una volta completate
e verificate tutte le 90 combinazioni; i risultati aggregati e l'identità dei
progetti W&B sono conservati nelle sezioni successive.

## Risultati finali del protocollo

La campagna è stata completata il 7 agosto 2026. Tutti i 30 training hanno
terminato le dieci epoche, ricaricato il proprio checkpoint finale `latest` e
prodotto il test VIS+IR. Anche le 90 valutazioni successive delle modalità
sono complete: per ciascuna delle sei configurazioni sono presenti i cinque
seed e i tre input `vis`, `ir` e `vis_ir`, senza combinazioni mancanti o
duplicate.

Salvo diversa indicazione, la deviazione standard riportata è quella
campionaria (`n-1`) e gli intervalli di confidenza al 95% della media usano la
distribuzione t di Student con quattro gradi di libertà. Con soltanto cinque
seed, intervalli e test di ipotesi vanno considerati esplorativi. I p-value non
sono corretti per confronti multipli e non sostituiscono la dimensione
dell'effetto e la consistenza tra seed.

### Risultati VIS+IR per seed

La metrica primaria è mAP@50 sul test set, valutata sul checkpoint finale
fissato a priori.

| Seed | Additive | FAM | FAM + IR Dropout | FAM + SSJ | Identity DCNv2 | Grid Sample |
|---:|---:|---:|---:|---:|---:|---:|
| 40 | 0.2566 | 0.3783 | 0.4087 | 0.3663 | 0.4336 | 0.4050 |
| 41 | 0.4029 | 0.4335 | 0.3234 | 0.3734 | 0.2113 | 0.3483 |
| 42 | 0.2958 | 0.3129 | 0.4452 | 0.3528 | 0.2926 | 0.4083 |
| 43 | 0.3476 | 0.3964 | 0.3986 | 0.4030 | 0.2860 | 0.3326 |
| 44 | 0.2372 | 0.3690 | 0.3598 | 0.3792 | 0.4162 | 0.3564 |

### Statistiche aggregate VIS+IR

| Configurazione | Media | Mediana | Dev. std. | Min–max | IC 95% media |
|---|---:|---:|---:|---:|---:|
| Additive | 0.3080 | 0.2958 | 0.0677 | 0.2372–0.4029 | 0.2239–0.3921 |
| FAM | 0.3780 | 0.3783 | 0.0440 | 0.3129–0.4335 | 0.3234–0.4326 |
| FAM + IR Dropout | 0.3871 | 0.3986 | 0.0469 | 0.3234–0.4452 | 0.3289–0.4453 |
| FAM + SSJ | 0.3749 | 0.3734 | 0.0185 | 0.3528–0.4030 | 0.3519–0.3980 |
| Identity DCNv2 | 0.3280 | 0.2926 | 0.0943 | 0.2113–0.4336 | 0.2109–0.4451 |
| Grid Sample | 0.3701 | 0.3564 | 0.0345 | 0.3326–0.4083 | 0.3273–0.4129 |

Le altre metriche test sono riportate come media ± deviazione standard. `mAP`
indica la metrica COCO mediata sulle soglie IoU da 0.50 a 0.95.

| Configurazione | mAP | mAP@50 | mAP@75 | mAR@100 | Loss test |
|---|---:|---:|---:|---:|---:|
| Additive | 0.1059 ± 0.0243 | 0.3080 ± 0.0677 | 0.0474 ± 0.0128 | 0.2474 ± 0.0302 | 12.7694 ± 0.7434 |
| FAM | 0.1351 ± 0.0096 | 0.3780 ± 0.0440 | 0.0692 ± 0.0043 | 0.2798 ± 0.0141 | 12.2408 ± 0.5836 |
| FAM + IR Dropout | 0.1360 ± 0.0165 | 0.3871 ± 0.0469 | 0.0698 ± 0.0079 | 0.2764 ± 0.0222 | 11.8911 ± 0.2126 |
| FAM + SSJ | 0.1297 ± 0.0095 | 0.3749 ± 0.0185 | 0.0625 ± 0.0086 | 0.2719 ± 0.0331 | 12.1010 ± 0.4534 |
| Identity DCNv2 | 0.1134 ± 0.0305 | 0.3280 ± 0.0943 | 0.0560 ± 0.0141 | 0.2556 ± 0.0357 | 12.2466 ± 0.6618 |
| Grid Sample | 0.1331 ± 0.0077 | 0.3701 ± 0.0345 | 0.0631 ± 0.0044 | 0.2849 ± 0.0115 | 11.9871 ± 0.2651 |

### Confronti appaiati VIS+IR

FAM viene confrontato con Additive; regolarizzazioni e ablation vengono
confrontate con FAM `current_dcnv2`. `Vittorie` indica in quanti seed la prima
configurazione supera il riferimento. Il test t è appaiato; Wilcoxon è il test
esatto bilaterale. Con cinque coppie, il più piccolo p-value bilaterale
possibile per Wilcoxon è 0.0625.

| Confronto | Δ medio | Δ mediano | Dev. std. Δ | IC 95% Δ medio | Vittorie | p t-test | p Wilcoxon |
|---|---:|---:|---:|---:|---:|---:|---:|
| FAM − Additive | +0.0700 | +0.0488 | 0.0531 | [+0.0040, +0.1359] | 5/5 | 0.0421 | 0.0625 |
| FAM + IR Dropout − FAM | +0.0091 | +0.0023 | 0.0869 | [−0.0988, +0.1171] | 3/5 | 0.8260 | 0.8125 |
| FAM + SSJ − FAM | −0.0031 | +0.0067 | 0.0369 | [−0.0488, +0.0427] | 3/5 | 0.8617 | 1.0000 |
| Identity DCNv2 − FAM | −0.0501 | −0.0203 | 0.1170 | [−0.1953, +0.0952] | 2/5 | 0.3928 | 0.6250 |
| Grid Sample − FAM | −0.0079 | −0.0127 | 0.0724 | [−0.0979, +0.0820] | 2/5 | 0.8192 | 1.0000 |

### Risultati delle modalità per seed

| Configurazione | Seed | VIS | IR | VIS+IR |
|---|---:|---:|---:|---:|
| Additive | 40 | 0.1294 | 0.1167 | 0.2566 |
| Additive | 41 | 0.1889 | 0.1988 | 0.4029 |
| Additive | 42 | 0.1441 | 0.1268 | 0.2960 |
| Additive | 43 | 0.1897 | 0.2150 | 0.3476 |
| Additive | 44 | 0.1193 | 0.1262 | 0.2373 |
| FAM | 40 | 0.2378 | 0.2604 | 0.3780 |
| FAM | 41 | 0.2823 | 0.2558 | 0.4334 |
| FAM | 42 | 0.2064 | 0.1477 | 0.3130 |
| FAM | 43 | 0.2675 | 0.2274 | 0.3964 |
| FAM | 44 | 0.2269 | 0.1229 | 0.3693 |
| FAM + IR Dropout | 40 | 0.2747 | 0.2244 | 0.4087 |
| FAM + IR Dropout | 41 | 0.2350 | 0.2227 | 0.3234 |
| FAM + IR Dropout | 42 | 0.2807 | 0.2083 | 0.4444 |
| FAM + IR Dropout | 43 | 0.2159 | 0.1616 | 0.3988 |
| FAM + IR Dropout | 44 | 0.1775 | 0.2226 | 0.3598 |
| FAM + SSJ | 40 | 0.2177 | 0.2427 | 0.3670 |
| FAM + SSJ | 41 | 0.2300 | 0.1381 | 0.3737 |
| FAM + SSJ | 42 | 0.2134 | 0.1745 | 0.3527 |
| FAM + SSJ | 43 | 0.2515 | 0.1936 | 0.4031 |
| FAM + SSJ | 44 | 0.2434 | 0.1483 | 0.3792 |
| Identity DCNv2 | 40 | 0.2837 | 0.2198 | 0.4339 |
| Identity DCNv2 | 41 | 0.1542 | 0.1739 | 0.2117 |
| Identity DCNv2 | 42 | 0.2183 | 0.1226 | 0.2928 |
| Identity DCNv2 | 43 | 0.1641 | 0.1005 | 0.2861 |
| Identity DCNv2 | 44 | 0.2548 | 0.2674 | 0.4161 |
| Grid Sample | 40 | 0.3011 | 0.2118 | 0.4050 |
| Grid Sample | 41 | 0.2713 | 0.1791 | 0.3479 |
| Grid Sample | 42 | 0.2603 | 0.1790 | 0.4084 |
| Grid Sample | 43 | 0.2369 | 0.1657 | 0.3326 |
| Grid Sample | 44 | 0.2373 | 0.2119 | 0.3566 |

### Statistiche aggregate per modalità

| Configurazione | Input | Media | Mediana | Dev. std. | Min–max | IC 95% media |
|---|---|---:|---:|---:|---:|---:|
| Additive | VIS | 0.1543 | 0.1441 | 0.0331 | 0.1193–0.1897 | 0.1131–0.1954 |
| Additive | IR | 0.1567 | 0.1268 | 0.0464 | 0.1167–0.2150 | 0.0991–0.2143 |
| Additive | VIS+IR | 0.3081 | 0.2960 | 0.0677 | 0.2373–0.4029 | 0.2240–0.3922 |
| FAM | VIS | 0.2442 | 0.2378 | 0.0307 | 0.2064–0.2823 | 0.2061–0.2822 |
| FAM | IR | 0.2028 | 0.2274 | 0.0635 | 0.1229–0.2604 | 0.1240–0.2817 |
| FAM | VIS+IR | 0.3780 | 0.3780 | 0.0439 | 0.3130–0.4334 | 0.3235–0.4325 |
| FAM + IR Dropout | VIS | 0.2368 | 0.2350 | 0.0428 | 0.1775–0.2807 | 0.1837–0.2899 |
| FAM + IR Dropout | IR | 0.2079 | 0.2226 | 0.0267 | 0.1616–0.2244 | 0.1748–0.2411 |
| FAM + IR Dropout | VIS+IR | 0.3870 | 0.3988 | 0.0466 | 0.3234–0.4444 | 0.3291–0.4449 |
| FAM + SSJ | VIS | 0.2312 | 0.2300 | 0.0163 | 0.2134–0.2515 | 0.2109–0.2514 |
| FAM + SSJ | IR | 0.1794 | 0.1745 | 0.0416 | 0.1381–0.2427 | 0.1279–0.2310 |
| FAM + SSJ | VIS+IR | 0.3751 | 0.3737 | 0.0185 | 0.3527–0.4031 | 0.3522–0.3981 |
| Identity DCNv2 | VIS | 0.2150 | 0.2183 | 0.0561 | 0.1542–0.2837 | 0.1453–0.2847 |
| Identity DCNv2 | IR | 0.1768 | 0.1739 | 0.0686 | 0.1005–0.2674 | 0.0917–0.2620 |
| Identity DCNv2 | VIS+IR | 0.3281 | 0.2928 | 0.0942 | 0.2117–0.4339 | 0.2112–0.4451 |
| Grid Sample | VIS | 0.2614 | 0.2603 | 0.0267 | 0.2369–0.3011 | 0.2282–0.2946 |
| Grid Sample | IR | 0.1895 | 0.1791 | 0.0211 | 0.1657–0.2119 | 0.1633–0.2157 |
| Grid Sample | VIS+IR | 0.3701 | 0.3566 | 0.0345 | 0.3326–0.4084 | 0.3272–0.4130 |

Le metriche secondarie delle valutazioni di modalità sono:

| Configurazione | Input | mAP | mAP@50 | mAP@75 | mAR@100 | Loss test |
|---|---|---:|---:|---:|---:|---:|
| Additive | VIS | 0.0461 ± 0.0126 | 0.1543 ± 0.0331 | 0.0163 ± 0.0082 | 0.1873 ± 0.0205 | 14.2692 ± 2.4691 |
| Additive | IR | 0.0584 ± 0.0157 | 0.1567 ± 0.0464 | 0.0357 ± 0.0136 | 0.2547 ± 0.0373 | 17.6092 ± 3.3208 |
| Additive | VIS+IR | 0.1059 ± 0.0244 | 0.3081 ± 0.0677 | 0.0474 ± 0.0129 | 0.2474 ± 0.0302 | 12.8286 ± 0.7528 |
| FAM | VIS | 0.0748 ± 0.0067 | 0.2442 ± 0.0307 | 0.0280 ± 0.0043 | 0.2340 ± 0.0128 | 14.0200 ± 2.2376 |
| FAM | IR | 0.0737 ± 0.0212 | 0.2028 ± 0.0635 | 0.0395 ± 0.0092 | 0.2748 ± 0.0359 | 18.8267 ± 1.6471 |
| FAM | VIS+IR | 0.1351 ± 0.0096 | 0.3780 ± 0.0439 | 0.0692 ± 0.0042 | 0.2800 ± 0.0140 | 12.2847 ± 0.5716 |
| FAM + IR Dropout | VIS | 0.0729 ± 0.0096 | 0.2368 ± 0.0428 | 0.0275 ± 0.0037 | 0.2391 ± 0.0238 | 16.9621 ± 3.3330 |
| FAM + IR Dropout | IR | 0.0759 ± 0.0092 | 0.2079 ± 0.0267 | 0.0421 ± 0.0097 | 0.3051 ± 0.0496 | 18.2355 ± 1.4117 |
| FAM + IR Dropout | VIS+IR | 0.1360 ± 0.0164 | 0.3870 ± 0.0466 | 0.0698 ± 0.0079 | 0.2764 ± 0.0220 | 11.9374 ± 0.2100 |
| FAM + SSJ | VIS | 0.0692 ± 0.0045 | 0.2312 ± 0.0163 | 0.0249 ± 0.0042 | 0.2296 ± 0.0204 | 15.1340 ± 1.9395 |
| FAM + SSJ | IR | 0.0628 ± 0.0104 | 0.1794 ± 0.0416 | 0.0342 ± 0.0050 | 0.2497 ± 0.0138 | 19.5248 ± 2.6536 |
| FAM + SSJ | VIS+IR | 0.1297 ± 0.0095 | 0.3751 ± 0.0185 | 0.0625 ± 0.0087 | 0.2719 ± 0.0329 | 12.1508 ± 0.4518 |
| Identity DCNv2 | VIS | 0.0625 ± 0.0178 | 0.2150 ± 0.0561 | 0.0199 ± 0.0079 | 0.2218 ± 0.0282 | 15.3777 ± 3.8104 |
| Identity DCNv2 | IR | 0.0682 ± 0.0263 | 0.1768 ± 0.0686 | 0.0403 ± 0.0162 | 0.2582 ± 0.0375 | 17.9357 ± 4.3983 |
| Identity DCNv2 | VIS+IR | 0.1135 ± 0.0305 | 0.3281 ± 0.0942 | 0.0559 ± 0.0140 | 0.2557 ± 0.0359 | 12.2988 ± 0.6816 |
| Grid Sample | VIS | 0.0762 ± 0.0074 | 0.2614 ± 0.0267 | 0.0234 ± 0.0035 | 0.2390 ± 0.0171 | 13.8314 ± 2.3030 |
| Grid Sample | IR | 0.0747 ± 0.0069 | 0.1895 ± 0.0211 | 0.0468 ± 0.0046 | 0.2962 ± 0.0398 | 17.8708 ± 1.0868 |
| Grid Sample | VIS+IR | 0.1330 ± 0.0077 | 0.3701 ± 0.0345 | 0.0631 ± 0.0044 | 0.2851 ± 0.0116 | 12.0245 ± 0.2516 |

### Effetto della modalità e della fusione

La tabella seguente riporta il Δ medio mAP@50 appaiato per seed. Tra parentesi
è indicato il numero di seed nei quali la prima configurazione supera il
riferimento.

| Confronto | VIS | IR | VIS+IR |
|---|---:|---:|---:|
| FAM − Additive | +0.0899 (5/5) | +0.0462 (4/5) | +0.0699 (5/5) |
| FAM + IR Dropout − FAM | −0.0074 (2/5) | +0.0051 (2/5) | +0.0090 (3/5) |
| FAM + SSJ − FAM | −0.0130 (2/5) | −0.0234 (2/5) | −0.0029 (3/5) |
| Identity DCNv2 − FAM | −0.0292 (3/5) | −0.0260 (1/5) | −0.0499 (2/5) |
| Grid Sample − FAM | +0.0172 (3/5) | −0.0133 (2/5) | −0.0079 (2/5) |

Per quantificare il valore della fusione, VIS+IR viene confrontato, per ogni
checkpoint, con la migliore tra VIS e IR dello stesso seed.

| Configurazione | Δ medio | Δ mediano | Dev. std. Δ | IC 95% Δ medio | Vittorie | p t-test |
|---|---:|---:|---:|---:|---:|---:|
| Additive | +0.1454 | +0.1325 | 0.0359 | [+0.1008, +0.1900] | 5/5 | 0.0008 |
| FAM | +0.1293 | +0.1289 | 0.0180 | [+0.1069, +0.1517] | 5/5 | 0.0001 |
| FAM + IR Dropout | +0.1412 | +0.1372 | 0.0357 | [+0.0969, +0.1856] | 5/5 | 0.0009 |
| FAM + SSJ | +0.1389 | +0.1393 | 0.0101 | [+0.1264, +0.1515] | 5/5 | <0.0001 |
| Identity DCNv2 | +0.1067 | +0.1220 | 0.0492 | [+0.0456, +0.1677] | 5/5 | 0.0083 |
| Grid Sample | +0.1087 | +0.1038 | 0.0269 | [+0.0753, +0.1421] | 5/5 | 0.0008 |

La rivalutazione VIS+IR riproduce il test eseguito al termine del training: la
massima differenza assoluta tra i due passaggi, su tutti i 30 checkpoint, è
`0.0008` mAP@50. Questo conferma sia la corretta risoluzione dei checkpoint sia
la coerenza del protocollo di evaluation.

### Interpretazione conclusiva

- FAM `current_dcnv2` migliora Additive di `+0.0700` mAP@50 medio, vince in
  tutti i cinque seed e riduce la deviazione standard da `0.0677` a `0.0440`.
  Il test t appaiato fornisce `p=0.0421`, mentre Wilcoxon esatto si ferma a
  `p=0.0625`, il minimo possibile con cinque coppie. Il risultato è coerente e
  promettente, ma va presentato insieme alla dimensione campionaria ridotta e
  al non determinismo CUDA documentato.
- IR Dropout raggiunge la media mAP@50 più alta (`0.3871`), ma il guadagno
  appaiato su FAM è soltanto `+0.0091`, cambia segno tra seed e ha un intervallo
  di confidenza molto ampio. Non emerge quindi evidenza di un beneficio
  affidabile.
- SSJ conserva quasi la stessa media di FAM (`0.3749` contro `0.3780`) e non
  mostra un miglioramento appaiato. Presenta però la dispersione descrittiva
  più bassa dell'intera campagna (`0.0185`), dato interessante ma non
  sufficiente da solo per concludere che riduca statisticamente la varianza.
- `identity_dcnv2` peggiora la media rispetto a FAM e produce la varianza più
  alta (`0.0943`). L'inizializzazione identità non risolve quindi
  l'instabilità finale e non va promossa a variante principale.
- `grid_sample` è vicino a FAM in media (`0.3701` contro `0.3780`), ma non lo
  supera in modo consistente. Può essere descritto come alternativa
  prestazionalmente comparabile nei limiti dei cinque seed, non come
  miglioramento dimostrato.
- VIS+IR supera la migliore modalità singola in tutti i 30 checkpoint. Il
  guadagno medio varia da `+0.1067` a `+0.1454` mAP@50 a seconda della
  configurazione: il vantaggio della fusione multimodale è il risultato più
  consistente dell'intera campagna.
- Il miglioramento FAM su Additive è particolarmente netto usando soltanto VIS
  (`+0.0899`, 5/5 seed); sull'IR isolato è più piccolo e meno stabile
  (`+0.0462`, 4/5). Le regolarizzazioni IR Dropout e SSJ non producono un
  vantaggio unimodale coerente rispetto al FAM standard.

## Integrazione nella tesi e lavoro residuo

Questa campagna sostituisce nelle conclusioni della tesi le precedenti singole
run RT-DETR (`0.357` Additive, `0.396` FAM e `0.438` FAM + SSJ). I numeri
storici possono restare nella ricostruzione dello sviluppo, ma non devono più
determinare il modello migliore.

La formulazione metodologica consigliata è che le analisi preliminari del
progetto hanno rivelato una variabilità capace di invertire l'ordinamento dei
metodi; per questo è stato bloccato un protocollo con seed appaiati, checkpoint
finale e reporting della distribuzione. Non va scritto genericamente "a
differenza di lavori precedenti", perché potrebbe essere interpretato come un
confronto con la letteratura anziché con le fasi precedenti del progetto.

Il test MtErie era già stato consultato durante lo sviluppo di architetture ed
iperparametri. Le metriche di questa sezione costituiscono quindi il benchmark
interno principale e un confronto appaiato a protocollo bloccato, non una stima
da holdout cieco. La limitazione va dichiarata esplicitamente. La sequenza
Carnation esclusa dagli split può essere usata una sola volta come stress test
esterno di Additive e FAM, ma non come validation o come nuova sorgente di
tuning.

La diagnostica di offset/feature sui 15 checkpoint finali FAM, FAM + SSJ e
Grid Sample è stata completata aggregando prima per checkpoint e poi tra seed.
Ha confermato l'attività del FAM a P3/P4, ma ha anche identificato una
degenerazione completa del ramo P5 nel FAM seed 41: offset medi di circa 5.030
pixel e uscita costante nello spazio, nonostante quel checkpoint ottenga la
migliore mAP@50 FAM. Questa anomalia non genera il vantaggio su Additive:
escludendo il seed 41, FAM vince ancora in 4/4 seed con delta medio `+0.0799`.
I dettagli e le cautele sull'interpretazione geometrica sono in
[`verifica_allineamento_FAM.md`](verifica_allineamento_FAM.md).

Restano l'error analysis Additive/FAM e, poiché YOLO deve comparire nelle
conclusioni, il confronto essenziale YOLO Additive contro YOLO + FAM con cinque
seed, feature gating, orizzonte fisso e `last.pt`; il training YOLO è stato
avviato l'8 agosto 2026.

La roadmap completa e la classificazione degli esperimenti storici sono in
[`thesis_experiment_audit.md`](thesis_experiment_audit.md).

## Conservazione locale dei checkpoint

Inventario eseguito il 7 agosto 2026, prima della diagnostica FAM:

- `wandb/`: circa 250 GiB;
- `checkpoints/`: circa 30 GiB;
- prima run del protocollo RT-DETR finale:
  `wandb/run-20260804_105327-wshd26fr` (`RTDETR_Protocol`, seed 40);
- 30 run finali da conservare: cinque checkpoint per ciascuno dei sei progetti
  elencati nella tabella del protocollo;
- le 90 run di sola valutazione delle modalità occupano complessivamente circa
  28 MiB e non sono la causa dell'uso di spazio;
- le 231 directory W&B precedenti al protocollo finale occupano circa
  214.9 GiB;
- nelle 30 run finali i `latest/model.safetensors` occupano circa 11.3 GiB,
  mentre i `latest/optimizer.bin` occupano circa 22.5 GiB.

I checkpoint finali non sono stati caricati su W&B perché il tracker esclude
`*.safetensors`; le relative directory non vanno quindi eliminate. Una volta
concluso un training che non sarà ripreso, `optimizer.bin` non serve né alla
diagnostica né all'inferenza: può essere rimosso mantenendo modello, config,
trace e summary. Le directory precedenti alla run di confine possono essere
eliminate soltanto dopo avere accettato che i checkpoint non copiati in
`checkpoints/` non saranno recuperabili dal server W&B. Per questo la prima
pulizia consigliata conserva inizialmente l'intera cartella `checkpoints/`.

La pulizia locale è stata completata dopo questo inventario. Sono state
eliminate 322 directory W&B non finali e i 30 file dell'optimizer delle run
concluse, mantenendo tutti i 30 checkpoint `latest` del protocollo RT-DETR.
Sono stati inoltre conservati 38 checkpoint storici di Deformable DETR e DINO
in `checkpoints/historical_wandb/`, insieme a un archivio leggero dei metadati
delle run rimosse in `out/wandb_metadata_archive_before_cleanup/`. Al termine,
`wandb/` occupava circa 13 GiB e `checkpoints/` circa 44 GiB.
