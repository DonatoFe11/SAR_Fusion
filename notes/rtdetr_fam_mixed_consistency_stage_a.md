# RT-DETR + FAM: Modal Dropout misto e consistency training

## Stato

**Screen Stage A seed 40 completato e candidato chiuso il 31 agosto 2026. I
tre gate congelati sono falliti; i seed 41--44 e lo Stage B non devono essere
eseguiti.**

Protocollo: `rtdetr_fam_mixed_consistency_stage_a_v1`.

Configurazioni:

- [screen scientifico seed 40](../parameters/RTDETR/rtdetr_fam_mixed_consistency_sequence_validation_seed40.yaml),
  SHA-256 `1d6c22cdcdd344a52c6c4381c1b425c6821262401f60d45301889c38b88305ce`;
- [espansione congelata a cinque seed](../parameters/RTDETR/rtdetr_fam_mixed_consistency_sequence_validation_five_seed.yaml),
  SHA-256 `0a9ba86b40b8af0f6e1dc5155dc35aa7245e249e035c01b0c89376a600c7ddfc`;
- [valutazione e gate seed 40](../parameters/RTDETR/rtdetr_fam_mixed_consistency_probe_evaluation.yaml),
  SHA-256 `3d691b0c3202bf830a16228653aee96256d9896a61b312b8a58fa0170e7d92f0`.

L'espansione a cinque seed resta archiviata per riproducibilità ma non è più
autorizzata, perché il prerequisito seed 40 è fallito.

## Esito dello screen seed 40

La run scientifica `hghkalag` ha completato dieci epoch e 7.810 optimizer
step, senza OOM, NaN o traceback. Il tempo totale W&B è stato `21.499,8 s`,
circa 5 ore e 58 minuti. Il checkpoint `best` è rimasto quello dell'epoch 1,
con validation fusion mAP@50 `0,125263`; SHA-256 del modello:
`052d743c5107c0a905596f45429d8509ea853139682235c6d75cc452c13f141a`.
Il `latest` dell'epoch 10 ha SHA-256
`118c5016da3aaef3e8bea4a4bdddb244305c9e78e9b1229fe018d217978310c2`.

La differenza nei tempi per epoch è intenzionale:

- epoch 1: `11:15`, solo forward/backward supervisionato perché
  `w_epoch = 0`;
- epoch 2: `35:11`, con teacher, student e consistency a peso `0,5`;
- epoch 3--10: circa 34--40 minuti, con consistency a peso `1,0`.

Dal secondo epoch ogni batch passa quindi da un solo percorso
forward/backward a un forward/backward supervisionato, un forward teacher
senza gradiente e un forward/backward student. Il salto di durata non indica
un errore o un blocco. La validation fusion è scesa da `0,125263` all'epoch 1
a `0,065633` all'epoch 2 e `0,036052` all'epoch 3; all'epoch 10 vale
`0,043328`. La selezione dell'epoch 1 segnala che, in questa run, gli
aggiornamenti di consistency non hanno migliorato la metrica primaria.

Il gate ha confrontato i checkpoint `best` del FAM baseline seed 40
(`m7xjslb6`) e del candidato sulle stesse 896 coppie FHL:

| Condizione | FAM baseline | Mixed consistency | Delta | Criterio | Esito |
|---|---:|---:|---:|---:|---|
| fusion / VIS-GT | 0,152148 | 0,125263 | -0,026884 | >= -0,010 | fallito |
| VIS-only / VIS-GT | 0,151107 | 0,091621 | -0,059487 | descrittivo | -- |
| paired masked-IR / VIS-GT | 0,002519 | 0,004348 | +0,001829 | >= +0,030 | fallito |
| IR nativa / IR-GT | 0,263709 | 0,213128 | -0,050581 | >= -0,030 | fallito |

Il verdetto automatico è `close_mixed_consistency_candidate`: tutti e tre i
requisiti obbligatori sono falliti. Non si eseguono i seed 41--44, non si crea
lo Stage B e non si consulta MtErie per questo candidato. Il risultato non
dimostra che ogni possibile consistency training sia inefficace; chiude la
specifica variante predefinita, che non ha ottenuto il miglioramento masked-IR
atteso e ha superato i limiti di regressione fusion e IR nativa.

Il `best` precede l'attivazione della consistency. Il suo piccolo incremento
masked-IR non va quindi attribuito causalmente alla nuova loss: è compatibile
con la variabilità stocastica della singola run. Le epoche effettivamente
consistency-trained hanno invece metriche fusion inferiori e non sono state
selezionate dal criterio congelato.

Risultati versionati:

- [CSV completo](Search_and_Rescue/results/rtdetr_fam_mixed_consistency_probe_evaluation.csv),
  SHA-256 `5cb36e63d7a5758215e9fa4cf431e3ad4e56d7b285259dd00466d7627b608d65`;
- aggregato JSON locale completo, SHA-256
  `1115324e84ab68a781695de4d65eadb7ac272a28886b411a07badae9eb1434bd`;
- hash semantico del protocollo registrato nell'aggregato:
  `afb1aca2f022748e1180e39a37bfa4a116b90f7375a5c78328285be45c6704c5`.

## Motivazione

Il FAM finale beneficia della fusione, ma il precedente audit ha mostrato una
forte differenza fra due condizioni IR-only:

- IR nativa con annotazioni IR: il ramo termico rimane capace di rilevare;
- IR adattata al canvas VIS, RGB mascherato e ground truth VIS: la prestazione
  è quasi nulla.

La sostituzione completa del Modal Dropout nativo con esempi paired-VIS non ha
risolto il problema: sul seed 40 il masked-IR è aumentato soltanto di `0,0003`
mAP@50 e la mAP@50 IR nativa è diminuita di `0,2366`. Quel candidato è chiuso
e non viene riaperto.

La nuova ipotesi è distinta: **conservare integralmente la supervisione nativa
e aggiungere una loss di consistenza paired**, usando la predizione fusion
pulita come riferimento per la stessa coppia con una modalità assente. Il
candidato cambia l'addestramento, non l'architettura FAM né l'inferenza.

## Intervento congelato

### Percorso supervisionato invariato

Ogni campione mantiene il Modal Dropout storico con probabilità in ordine
IR-only/RGB-only/fusion `[0,2; 0,2; 0,6]`:

| Estrazione | Input supervisionato | Ground truth |
|---|---|---|
| IR-only | IR e canvas nativi, RGB a zero | annotazioni IR native |
| RGB-only | RGB nativa, IR a zero | annotazioni VIS |
| fusion | RGB + IR adattata al canvas VIS | annotazioni VIS |

In particolare, `modal_dropout_coordinate_contract: native` è esplicito. La
loss di detection RT-DETR esistente viene calcolata senza modifiche su questo
percorso. La supervisione IR nativa che il probe precedente aveva eliminato
non viene sostituita né ripesata.

### Percorso paired di consistenza

Per la stessa coppia il loader costruisce inoltre:

1. un input teacher fusion pulito nelle coordinate VIS;
2. uno student nelle stesse coordinate, ottenuto mascherando RGB oppure IR con
   probabilità `[0,5; 0,5]`;
3. lo stesso `pixel_mask` per teacher e student.

La modalità student è indipendente dall'estrazione del Modal Dropout
supervisionato. Non vengono introdotti rumore, blur, traslazioni, scale o una
condizione fusion aggiuntiva: la versione 1 isola soltanto l'assenza completa
di una modalità.

Il teacher è il medesimo modello online, temporaneamente in evaluation mode e
senza gradiente (`online_eval_stop_gradient`). Non esiste un secondo modello,
un EMA o un checkpoint teacher esterno. Lo student usa gli stessi pesi in
training mode e riceve il gradiente della consistency loss.

### Matching e loss

Le query RT-DETR possono cambiare ordine sotto il masking. Non vengono quindi
confrontate per indice. Per ogni immagine:

- si calcola la probabilità teacher con sigmoid;
- si conservano al massimo 20 query con confidenza almeno `0,2`;
- si abbina ciascuna predizione teacher a una query student mediante matching
  ungherese con costi classificazione/L1/GIoU `[2; 5; 2]`;
- sulle coppie abbinate si applicano MSE fra probabilità di classe, L1 sulle
  box `cxcywh` normalizzate e `1-GIoU`, pesate `[2; 5; 2]`;
- le query sono pesate per la confidenza teacher.

Se nessuna query supera la soglia, il contributo paired di quell'immagine è
zero; non viene forzata una pseudo-label arbitraria. La loss complessiva è:

```text
L = L_detection_native_mixed
    + w_epoch * (2 L_class + 5 L_bbox + 2 L_giou)
```

Il primo epoch usa soltanto la baseline supervisionata. La consistenza parte
all'epoch 2 con `w_epoch = 0,5` e raggiunge `1,0` dall'epoch 3. Questa attesa
riduce l'influenza del teacher iniziale non adattato e rimane identica per
tutti i seed.

## Stage A invariato

Il candidato usa lo stesso protocollo della baseline FAM definitiva:

- train: FHL 0405/0406 e Baker pair 1, 3.123 coppie;
- validation: intera sequenza FHL 0401/0402, 896 coppie;
- input `640x640`, batch 4 e validation batch 12;
- AdamW, learning rate `2e-5` e dieci epoch completi;
- nessun early stopping;
- `best` selezionato soltanto dalla validation fusion mAP@50, con
  `min_delta = 0,001`;
- `latest` conservato all'epoch 10;
- seed 40--44;
- MtErie, confirmation set e stress geometrico esclusi da selezione e tuning.

Un test automatico rimuove i soli campi di consistency e richiede che ogni
run candidata coincida con la corrispondente run della baseline congelata.
L'aumento del costo computazionale è parte dell'intervento: dopo l'avvio della
consistenza ogni batch usa un forward/backward supervisionato, un forward
teacher senza gradiente e un forward/backward student.

## Regole di promozione congelate

### Screen seed 40

Il seed 40 è contemporaneamente screen scientifico e, se promosso, primo seed
della campagna. Dopo il training vengono valutati il suo `best` e il `best`
FAM seed 40 sullo stesso inventario FHL di 896 coppie già bloccato, nelle
quattro condizioni del probe precedente.

L'espansione ai seed 41--44 richiede **tutti** i seguenti criteri:

- delta fusion/VIS-GT `>= -0,01` mAP@50;
- delta paired masked-IR/VIS-GT `>= +0,03` mAP@50;
- delta IR nativa/IR-GT `>= -0,03` mAP@50.

VIS-only è riportato ma non costituisce un quarto gate. Le soglie sono le
stesse del precedente screen paired-VIS: non vengono rilassate dopo il suo
fallimento. Se un criterio fallisce, il candidato viene chiuso dopo un seed e
non viene valutato su MtErie.

Il runner
[`run_rtdetr_fam_mixed_consistency_probe_evaluation.py`](../scripts/run_rtdetr_fam_mixed_consistency_probe_evaluation.py)
applica automaticamente i tre gate. Riusa l'evaluator già verificato del
precedente probe, ma congela il contratto specifico del nuovo candidato. Il
preflight ha ricostruito esattamente 896 coppie FHL, 3.319 box VIS, 3.122 box
IR e l'inventario SHA-256
`47e2f348ebdc202cb749b1bbf2741fc868d8681eb24ffa4356cf4f584dfa4ec4`;
MtErie non è stato costruito. Pertanto questa non è una nuova validation: è
la valutazione multimodale della medesima sequenza di validation Stage A,
necessaria a decidere se espandere il candidato.

### Stage A a cinque seed

Se lo screen passa, i quattro seed rimanenti vengono eseguiti senza ripetere il
seed 40. L'accesso allo Stage B richiede contemporaneamente:

- guadagno medio candidato--FAM sulla `best` validation fusion mAP@50
  `>= +0,01`;
- guadagno fusion positivo in almeno 4/5 seed;
- guadagno medio paired masked-IR/VIS-GT `>= +0,03`;
- delta medio IR nativa/IR-GT `>= -0,03`.

Si riportano sempre valori per seed, deviazione standard, IC t 95%, differenze
appaiate e numero di vittorie. Gli intervalli con `n=5` sono descrittivi e non
sostituiscono la regola di promozione.

### Eventuale Stage B

Soltanto dopo la promozione Stage A verrà creato il full-data Stage B:

- 4.019 coppie, dieci epoch e checkpoint `latest`;
- seed 40--44 e confronto con il FAM Stage B configuration-matched;
- decisione finale su MtErie con guadagno medio fusion `>= +0,01` e almeno
  4/5 vittorie;
- paired masked-IR e IR nativa mantenuti come requisiti di efficacia/sicurezza;
- confirmation set e stress sintetico usati, eventualmente, soltanto dopo la
  decisione finale come caratterizzazione, non come selector.

## Verifiche pre-training

- intera suite del repository superata: 206 test;
- 15 test mirati per implementazione, freeze, inventario e gate superati;
- configurazioni seed 40 e cinque-seed uguali alla baseline salvo i campi
  dichiarati;
- supervisione IR nativa verificata mentre teacher/student restano paired-VIS;
- masking RGB e IR verificato senza mutare il tensore teacher;
- matching ungherese verificato rispetto a una permutazione delle query;
- gradienti student finiti per classificazione e localizzazione;
- consistency disabilitata automaticamente in validation e test.

Il probe GPU operativo `cqxsizsw` ha eseguito due batch reali di dimensione 4
senza OOM o NaN. Ha abbinato rispettivamente 5 query in 2/4 immagini e 14 query
in 3/4 immagini; la consistency loss totale è stata `3,6512` e `2,6088`. Il
probe forza l'avvio immediato della loss soltanto per attraversare il codice,
non salva checkpoint e non è un risultato della campagna.

## Comandi archiviati

Il seed 40 è stato prodotto con:

```bash
MPLCONFIGDIR=/tmp/rtdetr_mixed_consistency_mpl \
YOLO_CONFIG_DIR=/tmp/rtdetr_mixed_consistency_yolo \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
conda run --no-capture-output -n sarfusion python main.py experiment \
  --parameters parameters/RTDETR/rtdetr_fam_mixed_consistency_sequence_validation_seed40.yaml
```

Il gate completato è stato eseguito con:

```bash
MPLCONFIGDIR=/tmp/rtdetr_mixed_consistency_mpl \
YOLO_CONFIG_DIR=/tmp/rtdetr_mixed_consistency_yolo \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
conda run --no-capture-output -n sarfusion python \
  scripts/run_rtdetr_fam_mixed_consistency_probe_evaluation.py --workers 0
```

Il runner verifica prima configurazioni, hash, completezza dei due run e
checkpoint `best`; poi salva le otto valutazioni raw e il verdetto in
`out/rtdetr_fam_mixed_consistency_probe_evaluation/`. Nessuna soglia viene
scelta o modificata leggendo il risultato. `workers=0` è stato usato soltanto
per il vincolo multiprocessing dell'ambiente di esecuzione dell'evaluator;
ordine, batch, trasformazioni e metriche restano quelli congelati.

Il seguente comando era stato congelato per un'eventuale espansione:

```bash
MPLCONFIGDIR=/tmp/rtdetr_mixed_consistency_mpl \
YOLO_CONFIG_DIR=/tmp/rtdetr_mixed_consistency_yolo \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
conda run --no-capture-output -n sarfusion python main.py experiment \
  --parameters parameters/RTDETR/rtdetr_fam_mixed_consistency_sequence_validation_five_seed.yaml \
  --start-from-run 1
```

Quest'ultimo comando **non deve essere eseguito**, perché lo screen è fallito.
