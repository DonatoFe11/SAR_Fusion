# RT-DETRv2 Additive–FAM: Stage A a cinque seed e Stage B condizionale

## Stato e motivazione — 9 settembre 2026

**Preparata su richiesta dell'autore; training non avviato.** Sono previsti
dieci training nuovi: Additive e FAM standard per ciascuno dei seed 40–44.
Il vecchio screen resta archiviato in
[`rtdetr_v2_fam_stage_a.md`](rtdetr_v2_fam_stage_a.md).

Il pilot aveva già completato dieci epoche per entrambi i bracci: l'epoca 1
era il checkpoint scelto dalla validation, non un'interruzione del training.
Il fallimento del suo gate `best` resta valido come decisione del protocollo
originario. L'autore ha ora autorizzato una campagna multi-seed indipendentemente
dall'esito di quel filtro.

Questa campagna conserva la strategia precedente: **Stage A sui `best`
selezionati dalla validation**, eventuale **Stage B sui `latest` dopo un nuovo
training full-data**. La prima bozza dei nuovi YAML proponeva `latest` come
primario anche in Stage A: è stata corretta prima del lancio, su richiesta
dell'autore, per mantenere la stessa regola usata nelle campagne precedenti.
Il confronto diagnostico `latest` sarà riportato anche quando il suo segno
differisce da quello dei `best`.

L'espansione a cinque seed è stata decisa dopo aver osservato il pilot ed è
quindi un follow-up sullo stesso benchmark di sviluppo. Le vecchie run seed 40
e i probe da 20 step non entrano nella nuova media. Si stima la variabilità
mediante dieci training nuovi senza un ulteriore filtro al solo seed 40.

## Configurazioni e disegno

- [Additive, cinque seed](../parameters/RTDETR/rtdetr_v2_additive_sequence_validation_five_seed_v2.yaml);
- [FAM, cinque seed](../parameters/RTDETR/rtdetr_v2_fam_sequence_validation_five_seed_v2.yaml).

Entrambi i file usano:

- seed 40, 41, 42, 43, 44, ciascuno in un processo isolato;
- dieci epoche complete, nessun early stopping e nessun filtro prestazionale
  che interrompa la campagna dopo il primo seed;
- train FHL 0405/0406 + Baker 1: 3.123 coppie;
- validation FHL 0401/0402: 896 coppie, valutazione ogni epoca;
- `run_test=false`: nessuna inferenza automatica su MtErie;
- AdamW, LR `2e-5`, batch train 4, batch evaluation 12, resize 640;
- Modal Dropout nativo 20/20/60 soltanto nel training;
- checkpoint `PekingU/rtdetr_v2_r50vd`, revisione
  `282494075698cab9faa1096ae26856890030c817`, processore `use_fast=false`;
- FAM `current_dcnv2` con inizializzazione `historical_hf_post_init`, nessun
  SSJ, IR feature dropout o modifica aggiuntiva;
- caricamento stretto, trace degli input e dei seed, CUDA nativa;
- progetti W&B nuovi `RTDETRv2_Additive_SequenceVal_FiveSeed_V2` e
  `RTDETRv2_FAM_SequenceVal_FiveSeed_V2`.

Si mantiene lo split dello screen v2 per studiare la variabilità dell'effetto
FAM in quelle condizioni. Questa campagna non riproduce il train full-data
4.019 frame e il test MtErie del confronto storico RT-DETR v1: le mAP assolute
dei due protocolli non vanno confrontate come una classifica v1/v2.

Ogni YAML contiene cinque griglie da una sola run: indice 0 = seed 40,
1 = 41, 2 = 42, 3 = 43, 4 = 44. Gli override espliciti mantengono
`seed = data_seed = model_seed = training_seed`. Elencare cinque valori
indipendentemente in tutti e quattro i campi produrrebbe invece un prodotto
cartesiano di 625 run.

Il manifest sorgente mantiene l'identificatore
`rtdetr_v2_fam_training_source_v1`, che descrive quali file includere, ma
congela i **116 file correnti** con SHA-256:

```text
e63ac26308d3a9e7651d216901f093492815dfe3c787c994b63482130a61d9bd
```

Il vecchio digest `d32e9870...` descriveva 110 file prima dell'aggiunta del
port YOLO26. Il nuovo pin permette al controllo dei sorgenti di verificare
l'albero corrente; non richiede modifiche al modello RT-DETRv2. La verifica
del manifest arresta il lancio se i sorgenti cambiano durante la campagna.

## Stage A: checkpoint, analisi e promozione

1. **Primario:** differenze appaiate FAM − Additive della validation mAP@50
   sui rispettivi `best`, selezionati entro dieci epoche complete con
   `watch_metric=map_50`, `min_delta=0.001`.
2. **Secondario obbligatorio:** lo stesso confronto all'epoca 10, checkpoint
   `latest`.
3. **Diagnostica:** curve validation delle dieci epoche; mAP@[.50:.95],
   mAP@75 e recall sui checkpoint finali e sui `best`.

Riportare i cinque valori per braccio, i cinque delta appaiati, media,
deviazione standard campionaria, mediana, intervallo t al 95% dei delta e
numero di vittorie. L'unità statistica è il seed/checkpoint: le dieci epoche
e gli 896 frame non sono repliche indipendenti. Si completa il confronto
di tutti e cinque i seed anche se il primo delta è negativo.

La promozione a Stage B richiede entrambi i criteri del precedente protocollo:

- delta medio appaiato della **best validation mAP@50** almeno `+0.01`;
- delta strettamente positivo in almeno **4/5 seed**.

Si verifica inoltre che tutte le run siano complete, con checkpoint integri,
configurazioni e runtime appaiati. La decisione si prende dopo tutte le dieci
run; un risultato favorevole su `latest` o su una metrica secondaria non
sostituisce il criterio primario. Se il gate fallisce, si documenta l'esito
Stage A e non si avvia Stage B nel presente protocollo.

`save_final_checkpoint_only=true` salva `latest` al termine e conserva anche
`best` quando la validation migliora. `watch_metric` continua a scegliere
quest'ultimo; non ferma il training. `test_checkpoint=best` indica quale
checkpoint caricare per un eventuale test, ma **non calcola il riepilogo
multi-seed**: con `run_test=false` l'analisi userà le metriche validation dei
`best`, quelle dell'epoca 10 e i checkpoint salvati. La comparazione aggregata
va effettuata dopo la chiusura delle dieci run.

Gli script `audit_rtdetr_v2_stage_a_checkpoint.py` e
`replay_rtdetr_v2_stage_a_validation.py` restano legati al pilot seed 40 e al
suo gate `best`. Non sono un aggregatore di questa campagna.

## Stage B: retraining full-data dopo promozione

Se FAM supera lo Stage A, congelare la recipe e preparare due nuove campagne
appaiate RT-DETRv2 Additive/FAM, seed 40–44:

- ripartire dal checkpoint pretrained, senza proseguire i checkpoint Stage A;
- ripristinare tutte le **4.019 coppie train**, includendo FHL 0401/0402;
- eseguire dieci epoche complete con gli stessi iperparametri;
- disattivare validation, early stopping e test automatico;
- salvare e confrontare soltanto il **`latest` all'epoca 10**;
- valutare sul medesimo benchmark interno MtErie di 708 coppie soltanto dopo
  aver completato e verificato tutte le dieci nuove run.

La conferma finale di FAM rispetto ad Additive richiede delta medio mAP@50
almeno `+0.01` e almeno 4/5 delta positivi, come nel precedente Stage B. MtErie
resta un benchmark interno già consultato. Non si confronta la mAP Stage A su
FHL direttamente con la mAP Stage B su MtErie.

I due YAML forniti ora eseguono **solo Stage A**. Lo Stage B è condizionale e
richiederà configurazioni full-data e valutazione dedicate dopo la decisione
multi-seed; il comando qui sotto non lo avvia automaticamente.

## Comando di lancio

Dalla radice del repository, nell'ambiente RT-DETRv2 già installato:

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp/matplotlib-rtdetrv2 YOLO_CONFIG_DIR=/tmp/yolo-rtdetrv2

conda run --no-capture-output -n sarfusion-rtdetrv2 python main.py experiment \
  --parameters parameters/RTDETR/rtdetr_v2_additive_sequence_validation_five_seed_v2.yaml && \
conda run --no-capture-output -n sarfusion-rtdetrv2 python main.py experiment \
  --parameters parameters/RTDETR/rtdetr_v2_fam_sequence_validation_five_seed_v2.yaml
```

Il comando esegue prima i cinque Additive, poi i cinque FAM, in sequenza sulla
GPU. `&&` impedisce di avviare FAM se Additive termina con un errore tecnico;
nessuna soglia mAP impedisce il passaggio. I pesi pretrained devono essere già
presenti nella cache dell'ambiente, come per le run del pilot.

Per ripartire dopo un'interruzione, identificare l'ultima run completata e
impostare `experiment.start_from_grid` all'indice del primo seed da eseguire
in una copia del relativo YAML, lasciando `start_from_run=0`. Con queste
griglie singleton, `--start-from-run` non seleziona il seed globale. Il lancio
riparte da zero per quella run; non effettua il resume dell'optimizer.
Controllare gli artefatti della run interrotta per non includerla insieme a
una ripetizione completa della stessa coppia progetto/seed.
