# Stage A/B a cinque seed, senza screening sul seed 40

## Revisione autorizzata il 10 settembre 2026

L'autore elimina il filtro prestazionale sul solo seed 40 e include mixed
consistency, box-guided alignment e YOLO26.

**Aggiornamento 13 settembre 2026: Stage A completato.** Sono terminati tutti
i **20 nuovi training**, con riuso dei cinque FAM standard Stage A già
disponibili. Nessuna delle tre candidate supera la regola prestazionale
aggregata; **lo Stage B non viene attivato**. Valori per seed, best/latest,
provenienza e limiti degli audit sono nel
[resoconto dei risultati a cinque seed](stage_a_five_seed_v2_results.md).

Questa nota sostituisce le regole di allocazione delle nuove campagne. I pilot,
i relativi risultati, YAML, manifest e runner restano archiviati: non vengono
cancellati o riscritti retroattivamente. I loro divieti di espansione e i loro
audit seed 40 **non sono prerequisiti** della revisione v2. La scelta del
follow-up è successiva all'osservazione dei pilot e va dichiarata nella tesi.

Ogni campagna completa i seed **40, 41, 42, 43, 44**, anche quando il primo o
un altro seed ha metriche basse o delta negativo. Si arresta soltanto per
errori tecnici o di integrità (OOM, NaN, sorgenti/dati non corrispondenti,
checkpoint incompleti, replay incoerente), non per una soglia prestazionale
del singolo seed. Non si fanno aggiustamenti degli iperparametri fra seed.

## Stage A eseguito: 20 training nuovi e 5 controlli riutilizzati

Split comune: 3.123 coppie FHL 0405/0406 + Baker 1 per training;
896 coppie FHL 0401/0402 per validation. Nessun test su MtErie.

| Campagna | Run | Epoche per run | Configurazione / provenienza |
|---|---:|---:|---|
| RT-DETR v1 + FAM standard, controllo comune già addestrato | 5 riutilizzate | 10 | Progetto storico `RTDETR_FAM_SequenceVal_Fixed10_Protocol`; ID nel resoconto |
| RT-DETR v1 + FAM box-guided P3 | 5 | 10 | `parameters/RTDETR/rtdetr_fam_box_guided_stage_a_five_seed_v2.yaml` |
| RT-DETR v1 + FAM mixed consistency | 5 | 10 | `parameters/RTDETR/rtdetr_fam_mixed_consistency_stage_a_five_seed_v2.yaml` |
| YOLO26 Additive | 5 | 50 | `parameters/YOLO26/yolo26s_additive_stage_a_five_seed_v2.yaml` |
| YOLO26 + FAM | 5 | 50 | `parameters/YOLO26/yolo26s_fam_stage_a_five_seed_v2.yaml` |

Il piano iniziale prevedeva 25 nuovi training, inclusi cinque FAM standard.
Su indicazione dell'autore, questi ultimi sono stati sostituiti dalle cinque
run Stage A già complete del 14--15 agosto: non serviva ripetere la baseline.
Sono stati verificati la configurazione comune di modello, training e dati,
lo split e, ricostruendo l'inizializzazione corrente, gli hash iniziali per
tutti i seed. La baseline riutilizzata non è il matched control del pilot
seed 40, né RT-DETRv2, che è un detector diverso.

Le due nuove campagne RT-DETR usano lo stesso snapshot sorgente e l'ambiente
`sarfusion`. Le run storiche non attestano invece lo stesso snapshot sorgente:
l'equivalenza di configurazione e inizializzazione non prova l'identità
completa di codice, ambiente o traiettoria numerica. Questa limitazione del
riuso va mantenuta anche nella tesi. I confronti sono appaiati per seed,
non contro il miglior seed del controllo. Il YAML
`parameters/RTDETR/rtdetr_fam_stage_a_five_seed_v2.yaml` resta disponibile come
configurazione del controllo inizialmente proposto, ma non è stato eseguito
nella campagna dei 20 training.

La configurazione mixed conserva anche il percorso dati aggiuntivo e i suoi
sorteggi per lo student. Il seed appaiato non implica che tutti i successivi
sorteggi Modal Dropout siano identici al controllo: si studia l'intervento
completo di training, non una differenza di loss con tutti i tensori garantiti
identici. Le trace permettono di verificare ordine e realizzazioni effettive.

Ogni YAML RT-DETR ha cinque griglie singleton e mantiene esplicitamente
`seed = data_seed = model_seed = training_seed`, senza prodotto cartesiano.
Il codice RT-DETR non viene modificato. Il manifest sorgente corrente è
`rtdetr_box_guided_training_source_v1`, SHA-256
`3320b24d060aaacec316290c114933db1dce8d71eaaf455f1ccb6448183f76e5`.

YOLO26 usa l'ambiente separato `sarfusion-yolo26`. Conserva la recipe del
repair (in particolare AdamW e `warmup_bias_lr=0.0`), gli stessi pesi e hash
del dataset. Il nuovo runner `scripts/run_yolo26_stage_a_five_seed_v2.py`
accetta tutti i seed e non richiede un audit prestazionale Additive per
autorizzare FAM. Mantiene la verifica dell'ambiente, dei sorgenti e dei pesi,
l'uguaglianza dell'inizializzazione, il preflight GPU FAM, il controllo del
budget di 50 epoche e il replay del checkpoint best sulla sola validation.
I nuovi output sono in `runs/yolo26_stage_a_five_seed_v2/`.

## Selezione e decisione solo dopo tutti i seed

- Training completo senza early stopping.
- Primario: `best` per validation mAP@50, con miglioramento minimo `0.001`.
- Secondario obbligatorio: ultima epoca (`latest` RT-DETR, `last.pt` YOLO26).
- Riportare i cinque valori, delta appaiati, media, deviazione standard
  campionaria, mediana, IC t 95% e numero di vittorie. Cinque seed misurano
  variabilità di training sullo split fissato, non fra acquisizioni nuove.

La regola prestazionale Stage A rimane **delta medio >= +0.01 e almeno 4/5
delta strettamente positivi**, ma non esiste più alcun requisito aggiuntivo
di delta minimo sul seed 40. Non è un test di significatività statistica.

Confronti primari:

1. box-guided meno il FAM standard Stage A riutilizzato;
2. mixed consistency meno il FAM standard Stage A riutilizzato;
3. YOLO26 FAM meno YOLO26 Additive, per isolare il contributo di FAM.

Entrambi i YOLO26 vengono inoltre riportati contro il riferimento RT-DETR
FAM sullo stesso split, esplicitando il diverso budget e la diversa recipe
(50 contro 10 epoche). Il confronto interno YOLO26 non dimostra da solo che
YOLO26 superi il modello di riferimento della tesi.

Per un'eventuale promozione restano necessarie anche le verifiche specifiche
**dopo** i cinque seed:

- mixed: guadagno medio paired masked-IR/VIS-GT almeno `+0.03` e delta medio
  IR nativa/IR-GT almeno `-0.03`, oltre al criterio fusion comune;
- box-guided: audit del meccanismo su tutti i checkpoint candidati e
  controfattuale guida attiva/azzerata, conservando i criteri meccanicistici
  precedenti e distinguendo l'effetto di training da quello di inferenza;
- YOLO26: integrità e caricamento dei checkpoint, replay e tracciamento degli
  input appaiati. Il vecchio requisito di vitalità Additive sul seed 40 non
  si applica; le eventuali prestazioni collassate vanno comunque riportate.

I runner di audit storici che richiedono il pilot seed 40 o i vecchi progetti
non sono aggregatori v2 e non devono essere usati per deciderne la promozione.
Il launcher non decide né avvia lo Stage B. Al 13 settembre l'aggregazione
best/latest e i controlli di integrità sono conclusi, inclusi i replay YOLO26.
Le valutazioni mixed masked-IR/IR nativa e gli audit box-guided del meccanismo
e active-vs-zero non sono stati ripetuti sui nuovi cinque seed. Non vengono
dichiarati superati o falliti sulla base dei vecchi pilot: la mancata
promozione è già determinata dal criterio fusion primario, necessario e
fallito da tutte e tre le candidate.

## Stage B condizionale: non attivato dopo questi risultati

Solo dopo la decisione aggregata e gli audit si preparano i YAML e
l'evaluatore full-data dei candidati promossi:

- nuovo training dai pesi pretrained, **non resume dei best Stage A**;
- tutte le 4.019 coppie train, reintegrando la sequenza di validation;
- cinque seed per candidato e relativo controllo appaiato;
- budget invariato per famiglia: 10 epoche RT-DETR, 50 YOLO26;
- nessuna validation, early stopping o consultazione automatica del test;
- checkpoint dell'ultima epoca (`latest` / `last.pt`) come unico primario;
- valutazione comune sulle 708 coppie MtErie solo dopo il completamento
  di tutte le run previste, con riferimento FAM configuration-matched;
- conferma con delta medio >= +0.01 e almeno 4/5 vittorie nel confronto
  dichiarato; conservare anche le valutazioni multimodali per mixed.

Per YOLO26 si distinguono la conferma del delta FAM/Additive e la comparazione
finale con RT-DETR FAM: sono due domande, non un unico risultato intercambiabile.
MtErie rimane un benchmark interno già consultato, non un nuovo test cieco.
Non si confrontano direttamente le mAP Stage A su FHL con quelle Stage B su
MtErie. **Nessuno Stage B né nuova valutazione MtErie è stato eseguito per
questa campagna.** Un nuovo intervento o una modifica del criterio richiede
un protocollo distinto e dichiarato, non una promozione retroattiva.

## Comando dei 20 training eseguiti (archivio, non da rilanciare)

La campagna è terminata il 13 settembre alle 03:55 CEST. Il log è
`stage_a_20_training_20260910_105454.log`. Il comando utilizzato dalla radice
del repository, con pesi già presenti nella cache, escludeva il nuovo FAM
standard e completava tutti i seed delle quattro campagne:

```bash
set -o pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp/matplotlib-stage-a-five-seed-v2
export YOLO_CONFIG_DIR=/tmp/yolo-stage-a-five-seed-v2
export YOLO_AUTOINSTALL=false CUBLAS_WORKSPACE_CONFIG=:4096:8
{
  conda run --no-capture-output -n sarfusion python main.py experiment \
    --parameters parameters/RTDETR/rtdetr_fam_box_guided_stage_a_five_seed_v2.yaml &&
  conda run --no-capture-output -n sarfusion python main.py experiment \
    --parameters parameters/RTDETR/rtdetr_fam_mixed_consistency_stage_a_five_seed_v2.yaml &&
  bash scripts/run_stage_a_five_seed_v2.sh --yolo26-only
} 2>&1 | tee "stage_a_20_training_$(date +%Y%m%d_%H%M%S).log"
```

Attenzione: il launcher senza opzioni include ancora FAM standard e quindi
il piano iniziale da 25 run; non rappresenta il comando dei 20 training
effettivamente eseguiti. Un errore tecnico ferma la catena; una metrica bassa
non la ferma. Le opzioni disponibili sono `--dry-run`, `--rtdetr-only` e
`--yolo26-only`. Non occorre rilanciare alcun training di questa campagna.

### Indicazioni tecniche di ripresa (archivio)

Per disconnettersi da tmux: `Ctrl+B`, poi `D`. Non rilanciare tutto alla cieca
dopo un'interruzione: identificare prima le run complete. Per RT-DETR una
copia del YAML può impostare `start_from_grid` all'indice del primo seed da
rifare (0--4), lasciando `start_from_run=0`; il training di quella run riparte
dall'inizio. Per YOLO26 il runner accetta `--seed`, ma rifiuta di sovrascrivere
una directory esistente: conservare separatamente una run incompleta prima di
un nuovo lancio. Escludere le run incomplete dalle statistiche.
