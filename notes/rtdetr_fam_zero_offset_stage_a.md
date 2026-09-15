# RT-DETR v1 + FAM standard: inizializzazione dei soli offset a zero

## Stato

Esperimento preparato il 15 settembre 2026 su richiesta dell'autore.
**Training non ancora avviati.** Non modifica la tesi, i risultati precedenti
o il comportamento predefinito del FAM. Il codice preesistente e le campagne
concluse sono salvati nel commit `7d733aa`.

## Domanda e intervento

Nel FAM standard di RT-DETR, `post_init()` di Hugging Face sovrascrive
l'inizializzazione a zero del predittore. Il test misura l'effetto di partire
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

| Seed | ID controllo | Best validation mAP@50 |
|---:|---|---:|
| 40 | `m7xjslb6` | 0.1521475464 |
| 41 | `2kil4xq9` | 0.1423595846 |
| 42 | `fu87g1i2` | 0.1654664278 |
| 43 | `pz5tzni4` | 0.1939318329 |
| 44 | `398272cv` | 0.1689078957 |

Media del controllo: **0.1645626575** sulla validation Stage A.
ID, percorsi, hash delle configurazioni e dello stato iniziale e metriche
best/latest sono congelati in
`parameters/RTDETR/rtdetr_fam_zero_offset_stage_a_protocol.json`.

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

## Lancio da tmux

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

## Risultati e interruzioni

Output in `out/rtdetr_fam_zero_offset_stage_a/`:

- `seedXX/initialization.json`: audit dell'intervento;
- `seedXX/validation.json`: tutte le dieci metriche live;
- `seedXX/training_complete.json`: provenienza, best/latest e hash checkpoint;
- `seedXX/replay.json`: caricamento stretto e replay del best;
- `decision.json`: confronto aggregato, generato soltanto a campagna completa.

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
