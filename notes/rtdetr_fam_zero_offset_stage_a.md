# RT-DETR v1 + FAM standard: inizializzazione dei soli offset a zero

## Stato

**Campagna completata il 15 settembre 2026; analisi aggiornata il 16 settembre
2026:** cinque nuovi training da 10 epoche, sui seed 40--44, e cinque replay
del checkpoint best. I cinque controlli FAM Stage A già disponibili sono
stati riutilizzati.

Il candidato con inizializzazione dei soli offset a zero ottiene una media
validation mAP@50 `best` di **0.152080**, contro **0.164563** del FAM standard:
delta medio **-0.012482**, con **1/5** seed positivo. **La regola di promozione
non è superata: nessuno Stage B e nessuna valutazione test.** Il risultato
positivo medio sui checkpoint `latest` è diagnostico e non cambia la decisione.

I risultati sono integrati nella tesi come ablation Stage A a cinque seed,
con `best` primario e `latest` diagnostico, insieme agli altri confronti sullo
stesso controllo FAM. Il test non modifica i risultati precedenti o il
comportamento predefinito del FAM. Il codice preesistente e le campagne
precedenti sono salvati nel commit `7d733aa`.

## Domanda e intervento

Nel FAM standard di RT-DETR, `post_init()` di Hugging Face sovrascrive
l'inizializzazione a zero del predittore. Ho quindi misurato l’effetto di partire
da offset nulli senza cambiare l'inizializzazione degli altri componenti.

Si costruisce completamente il normale RT-DETR v1 + FAM, inclusi il
caricamento del checkpoint `PekingU/rtdetr_r50vd` e il trasferimento della
testa `person`. Solo dopo si azzerano pesi e bias delle righe `0:18` di
`offset_conv` nei tre livelli P3/P4/P5.

- Le righe `18:27`, che predicono le maschere, restano identiche al controllo:
  **le maschere non vengono forzate a 0.5**.
- I pesi e bias DCNv2, i backbone e le teste restano identici al controllo.
- Nessun nuovo parametro, nessun consumo aggiuntivo di numeri casuali.
- Gli offset sono inizialmente nulli, ma rimangono **apprendibili**.
- Non è `identity_dcnv2`: il filtro DCNv2 non viene inizializzato come identità.
- Non è `grid_sample`, né un test con offset congelati durante il training.

La factory `fusion_rtdetr_zero_offset` viene registrata soltanto dal runner
dedicato. I file storici in `sarfusion/` e `main.py` non cambiano, e il manifest
storico `rtdetr_box_guided_training_source_v1` mantiene il proprio hash.
**Non lanciare questo YAML direttamente con `main.py`: usare il comando sotto.**

## Cinque nuovi training, riuso dei cinque controlli

Il controllo è RT-DETR v1 + FAM standard del progetto
`RTDETR_FAM_SequenceVal_Fixed10_Protocol`, non Base senza FAM, RT-DETRv2
o il confronto storico con media test 0.3780.

| Seed | ID controllo FAM standard | ID candidato offset-zero |
|---:|---|---|
| 40 | `m7xjslb6` | [z1h6skvd](../wandb/run-20260915_114932-z1h6skvd/files) |
| 41 | `2kil4xq9` | [b18jmijy](../wandb/run-20260915_135849-b18jmijy/files) |
| 42 | `fu87g1i2` | [gm9fziap](../wandb/run-20260915_162027-gm9fziap/files) |
| 43 | `pz5tzni4` | [u5mfyfuh](../wandb/run-20260915_184156-u5mfyfuh/files) |
| 44 | `398272cv` | [nd6il42x](../wandb/run-20260915_210921-nd6il42x/files) |

Media del controllo: **0.1645626575** sulla validation Stage A.
ID, percorsi, hash delle configurazioni e dello stato iniziale e metriche
best/latest sono congelati nel
[protocollo](../parameters/RTDETR/rtdetr_fam_zero_offset_stage_a_protocol.json).

Il runner verifica le configurazioni effettive delle cinque run archiviate e
la presenza di entrambi i checkpoint. Per ciascun nuovo seed, prima del
training, l'hash del modello standard appena ricostruito deve coincidere con
quello registrato dalla run di controllo, e l'intervento deve conservare tutti
i parametri non interessati e lo stato RNG. Una discrepanza ferma il lavoro
come errore tecnico: non avvia automaticamente cinque nuovi controlli.

**Limite del riuso:** l'uguaglianza di configurazione e inizializzazione non
attesta uno snapshot sorgente, un ambiente o una traiettoria numerica storica
interamente identici. Il controllo resta il riferimento Stage A già usato per
le altre candidate. Gli offset iniziali non sono stati scelti osservando
nuovi risultati; non si confronta la validation con la mAP test 0.3780.

## Protocollo Stage A/B

- Seed 40--44, modello/dati/training appaiati; un processo nuovo per seed.
- Training sulle 3.123 coppie FHL 0405/0406 + Baker 1.
- Validation sulle 896 coppie FHL 0401/0402.
- 10 epoche complete, AdamW `2e-5`, batch 4, Modal Dropout nativo 20/20/60.
- Nessun early stopping e nessun filtro prestazionale sul seed 40.
- Primario: `best` selezionato sulla validation mAP@50 con `min_delta=0.001`.
- Diagnostico: `latest` dell'epoca 10; non sostituisce il primario.
- Nessuna valutazione MtErie/test e nessun training Stage B automatico.

Dopo ogni training, un processo separato ricarica **strettamente** il best
serializzato e ne ripete la sola validation, utilizzando l'evaluatore comune
già presente in `replay_rtdetr_v2_stage_a_validation.py` (la funzione di replay
è compatibile anche con RT-DETR v1). Tolleranza assoluta mAP@50: `0.0002`.
Il valore primario rimane quello live di selezione, non il massimo tra live
e replay. Gli hash dei file best/latest vengono conservati e riverificati.

Solo dopo cinque training completi e cinque replay riusciti vengono calcolati
delta appaiati, media, DS campionaria, mediana, IC t 95% e numero di vittorie.
La regola è **delta medio >= +0.01 e almeno 4/5 delta positivi**.
Un eventuale superamento richiede la preparazione separata dello Stage B:
nuovi training dai pesi pretrained sulle 4.019 coppie, cinque seed candidato e
controllo appaiato, 10 epoche, nessuna validation e checkpoint finale unico.
Non si fa resume dei best Stage A. Un mancato superamento non attiva Stage B.

## Risultati Stage A

Fonte numerica: [decision.json](../out/rtdetr_fam_zero_offset_stage_a/decision.json),
ricalcolato in memoria dai record dei cinque training e confrontato con il
file salvato. Tutti i valori seguenti sono **validation mAP@50**, in scala
0--1; il delta è sempre candidato meno controllo dello stesso seed.
Le tabelle arrotondano a sei decimali, mentre la decisione usa i valori completi.
Le epoche riportate sono numerate da 1 a 10.

### Primario: checkpoint best

Il `best` è quello selezionato durante il training con `min_delta=0.001`,
non un checkpoint scelto successivamente in base al replay o al test.

| Seed | FAM standard best | Offset-zero best | Delta appaiato | Epoca best offset-zero |
|---:|---:|---:|---:|---:|
| 40 | 0.152148 | 0.174178 | +0.022030 | 3 |
| 41 | 0.142360 | 0.133232 | -0.009127 | 1 |
| 42 | 0.165466 | 0.163615 | -0.001852 | 2 |
| 43 | 0.193932 | 0.148416 | -0.045516 | 5 |
| 44 | 0.168908 | 0.140961 | -0.027947 | 4 |
| **Media** | **0.164563** | **0.152080** | **-0.012482** | — |

- FAM standard: media ± DS campionaria **0.164563 ± 0.019554**;
  mediana `0.165466`.
- Offset-zero: **0.152080 ± 0.016686**; mediana `0.148416`.
- Delta appaiato: **-0.012482 ± 0.025718**; mediana `-0.009127`;
  IC t 95% **[-0.044416, +0.019451]**; seed positivi **1/5**.

Il delta medio equivale a circa **-1.25 punti percentuali** di mAP@50, non a
una riduzione relativa dell'1.25%. L'IC usa i cinque delta appaiati, DS
campionaria e distribuzione t con quattro gradi di libertà. Include zero:
la media osservata è inferiore, ma questi dati non dimostrano un peggioramento
universale né una superiorità statisticamente accertata del FAM standard.

### Diagnostico: checkpoint latest, epoca 10

| Seed | FAM standard latest | Offset-zero latest | Delta appaiato |
|---:|---:|---:|---:|
| 40 | 0.039738 | 0.065085 | +0.025346 |
| 41 | 0.103029 | 0.100854 | -0.002176 |
| 42 | 0.094595 | 0.130998 | +0.036403 |
| 43 | 0.078981 | 0.082365 | +0.003384 |
| 44 | 0.103729 | 0.099740 | -0.003990 |
| **Media** | **0.084014** | **0.095808** | **+0.011794** |

- FAM standard: media ± DS campionaria **0.084014 ± 0.026683**;
  mediana `0.094595`.
- Offset-zero: **0.095808 ± 0.024515**; mediana `0.099740`.
- Delta appaiato: **+0.011794 ± 0.018057**; mediana `+0.003384`;
  IC t 95% **[-0.010628, +0.034215]**; seed positivi **3/5**.

L'inizializzazione a zero migliora dunque la media all'ultima epoca, ma
anche questo IC include zero. Tutti i candidati hanno il best prima
dell'epoca 10 e terminano sotto il proprio best. Il calo medio best--latest
è `0.056272` per offset-zero e `0.080548` per il controllo: è un'osservazione
descrittiva, non una dimostrazione di maggiore stabilità o generalizzazione.
Non autorizza a sostituire a posteriori il primario con `latest`; inoltre,
anche su `latest`, i seed positivi sarebbero tre e non i quattro richiesti.

### Decisione e interpretazione

La regola congelata richiede contemporaneamente un delta medio `best`
almeno `+0.01` e almeno quattro delta positivi. L'esito è invece `-0.012482`
e `1/5`: **nessuna promozione allo Stage B**. Non sono stati avviati training
Stage B né valutazioni MtErie/test per questa variante.

Il miglioramento del solo seed 40 non si conferma sugli altri quattro seed;
tutti e cinque sono stati comunque addestrati per l'intero budget, senza
filtro preliminare. Nel protocollo eseguito, azzerare i soli offset iniziali
non fornisce evidenza sufficiente per sostituire il FAM standard. Non è un
confronto FAM contro Base senza FAM e non dimostra che l'inizializzazione a
zero sia sempre sfavorevole in altri protocolli o architetture. Il riferimento
della tesi non cambia; la media test storica `0.3780` non è confrontabile
direttamente con queste medie di validation Stage A.

## Audit della campagna e dell'apprendimento

Verifiche effettuate sui record e sui checkpoint già presenti, senza nuovi
training o forward di valutazione:

- **5/5 training completi:** dieci metriche live per seed; selezione `best`
  ricostruita dalla cronologia, `latest` coincidente con l'epoca 10 e metadati
  coerenti con i riepiloghi W&B.
- **5/5 inizializzazioni valide:** lo stato del modello standard prima
  dell'intervento coincide con l'hash iniziale del controllo appaiato; dopo
  l'intervento il suo hash coincide con la traccia iniziale del candidato.
  I report attestano offset inizialmente nulli, parametri non interessati
  invariati, nessun reset di maschere/DCNv2, identità dei parametri e RNG
  conservati. Queste uguaglianze riguardano l'inizializzazione, non i pesi
  dopo il training.
- **10/10 hash dei checkpoint verificati:** ricalcolato lo SHA-256 dei file
  `best/model.safetensors` e `latest/model.safetensors` di ciascun candidato.
- **5/5 replay già eseguiti e riusciti:** sulle stesse 896 coppie di validation
  (75 batch), la mAP@50 del best ricaricato coincide esattamente con quella
  live in tutti i seed; errore assoluto **0.0**, entro la tolleranza `0.0002`.
  Il caricamento stretto ricostruisce 1.099 tensori di stato da 1.051 tensori
  serializzati e 48 alias esatti dei pesi condivisi: non ignora parametri
  mancanti. Non sono stati rieseguiti i replay dei controlli storici.
- **Configurazione e sorgenti coerenti su tutte le run:** gli hash registrati
  coincidono con i file attuali. Manifest
  `rtdetr_fam_zero_offset_training_source_v1`, 29 file,
  SHA-256 `ef969a5792db4c118a4ce6074d57532a4347269758ac0abbda8cb7673d532a67`.
  SHA-256 dello YAML candidato:
  `2e87b7631eab5734dc00be35ac2298f72c4f28febd9b488fa0a0e2369c2ed15a`.

### I predittori inizialmente uguali sono rimasti uguali?

**No, nei checkpoint ispezionati.** La lettura su CPU delle righe `0:18`
di `offset_conv.weight` e `offset_conv.bias`, nei tre livelli FAM di entrambi
i checkpoint di tutti i seed, verifica **30 predittori** (5 × 2 × 3).
In ciascuno, tutte le 18 righe dei pesi sono non nulle e distinte fra loro;
anche i 18 bias sono non nulli e tutti i valori verificati sono finiti.
Questo vale già per i `best`, non soltanto per i checkpoint finali.

L'inizializzazione a zero non ha quindi bloccato l'aggiornamento dei
predittori né mantenuto identiche le loro righe. Questa è però una verifica
dei **parametri appresi**, non una misura degli offset in pixel: non dimostra
che ogni posizione abbia offset differenti o che corrispondano a un migliore
allineamento VIS--IR. Non sono stati aggiunti test geometrici o valutazioni
delle modalità mancanti. Resta inoltre il limite del riuso dei controlli
storici descritto sopra: configurazione e inizializzazione appaiate non
certificano l'identità completa degli ambienti e delle traiettorie numeriche.

## Lancio da tmux (riferimento di riproducibilità)

La campagna è conclusa: **non occorre rilanciare questi comandi** per ottenere
i risultati sopra. Sono conservati come documentazione operativa.

Dalla radice del repository, nell'ambiente locale con i pesi già in cache:

```bash
set -o pipefail
bash scripts/run_rtdetr_fam_zero_offset_stage_a.sh \
  2>&1 | tee "rtdetr_fam_zero_offset_stage_a_$(date +%Y%m%d_%H%M%S).log"
```

Il launcher seleziona l'ambiente Conda `sarfusion`; non installa dipendenze.
Il nuovo progetto W&B è `RTDETR_FAM_ZeroOffset_StageA_FiveSeed`.

Controllo preventivo senza training:

```bash
bash scripts/run_rtdetr_fam_zero_offset_stage_a.sh --dry-run
CUDA_VISIBLE_DEVICES='' bash scripts/run_rtdetr_fam_zero_offset_stage_a.sh --check-init
```

Il secondo comando costruisce soltanto i modelli su CPU per verificare le
inizializzazioni dei cinque seed; non crea run W&B e non esegue training.

## Artefatti e gestione delle interruzioni

Output in [out/rtdetr_fam_zero_offset_stage_a/](../out/rtdetr_fam_zero_offset_stage_a/):

- `seedXX/initialization.json`: audit dell'intervento;
- `seedXX/validation.json`: tutte le dieci metriche live;
- `seedXX/training_complete.json`: provenienza, best/latest e hash checkpoint;
- `seedXX/replay.json`: caricamento stretto e replay del best;
- `decision.json`: confronto aggregato, generato soltanto a campagna completa.

Log della campagna:
[rtdetr_fam_zero_offset_stage_a_20260915_114919.log](../rtdetr_fam_zero_offset_stage_a_20260915_114919.log).
Le directory W&B dei cinque candidati sono collegate nella tabella degli ID;
i percorsi dei controlli sono nel protocollo congelato.

I checkpoint veri restano nelle rispettive directory W&B. Un lock impedisce
due lanci simultanei dello stesso supervisore. Rilanciare lo stesso comando
salta i seed già completi e riprende un replay non ancora eseguito senza
ripetere il training.

Se un training è interrotto, il comando si ferma e richiede esplicitamente:

```bash
set -o pipefail
bash scripts/run_rtdetr_fam_zero_offset_stage_a.sh --restart-incomplete \
  2>&1 | tee "rtdetr_fam_zero_offset_restart_$(date +%Y%m%d_%H%M%S).log"
```

Questo archivia i record del tentativo incompleto sotto `incomplete/` senza
cancellare le run W&B. Il seed interrotto riparte **da zero dai pesi
pretrained**, non dal best né dall'ultimo stato optimizer. I seed completi non
vengono ripetuti. Un replay fallito resta un errore da diagnosticare, non un
motivo per scegliere o riaddestrare un altro checkpoint.

Durante la campagna si possono modificare i file della tesi: sono esclusi dal
manifest sorgente. Non modificare runner, configurazioni o sorgenti di training
fra i seed; il controllo degli hash blocca una campagna mista.
