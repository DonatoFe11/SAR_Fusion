# Valutazione finale YOLO per modalità

## Stato

Protocollo congelato il 12 agosto 2026, prima della nuova inferenza sui dieci
checkpoint finali. La campagna è completa: 30/30 valutazioni, 708 frame per
valutazione e aggregato marcato `protocol_complete: true`. Non è stato eseguito
training e non sono state selezionate epoche o checkpoint sul test.

## Domanda sperimentale

Il confronto VIS+IR sui `last.pt` aveva mostrato che, dopo 200 epoche, FAM non
migliora in media la baseline YOLO Additive. Questa valutazione completa il
confronto quando manca RGB oppure IR.

Sono valutate due configurazioni:

- YOLOv10 dual-backbone Additive;
- YOLOv10 dual-backbone con FAM standard.

Per entrambe si usano i seed appaiati `40--44` e lo stesso `last.pt` già
prodotto dal protocollo fisso di 200 epoche.

## Condizioni di input

Tutte le condizioni usano le stesse 708 coppie del file `test_vis_ir.txt` e le
stesse annotazioni. Cambia soltanto la maschera `[RGB, IR]` passata al modello:

| Condizione | Maschera | Percorso attivo |
|---|---:|---|
| VIS+IR | `[1, 1]` | entrambe le backbone; FAM attivo nella variante FAM |
| VIS | `[1, 0]` | sola backbone RGB |
| IR | `[0, 1]` | sola backbone IR; FAM bypassato |

Usare gli YAML mono-modali come normali input a tre canali non sarebbe
equivalente: il percorso di compatibilità a tre canali del modello usa la
backbone standard. Il runner utilizza invece il tensore accoppiato a quattro
canali e il feature gating esplicito con cui il modello è stato addestrato. La
modalità esclusa non viene elaborata e non può contribuire alla detection.

## Protocollo bloccato

La specifica machine-readable è
[`../parameters/YOLO/yolov10_final_modality_evaluation.yaml`](../parameters/YOLO/yolov10_final_modality_evaluation.yaml).
Fissa:

- checkpoint `last.pt` dopo 200 epoche;
- seed `40--44`;
- split `test` MtErie;
- batch size 8;
- `max_det=300`;
- tolleranza tecnica VIS+IR `0.002` rispetto ai risultati automatici
  arrotondati già registrati;
- tre condizioni VIS+IR, VIS e IR;
- output separato per ogni checkpoint/modalità;
- aggregazione prima per checkpoint e poi fra i cinque seed.

Il runner
[`../scripts/run_yolo_final_modality_evaluation.py`](../scripts/run_yolo_final_modality_evaluation.py)
verifica inoltre per ogni run:

- seed registrato in `args.yaml`;
- Modal Dropout feature-gated `20/20/60`;
- selettore `test_checkpoint: last`;
- esistenza del checkpoint;
- esattamente 200 righe in `results.csv`;
- hash del checkpoint e del file contenente il test set.

La modalità VIS+IR viene ricalcolata anche dal runner standalone. La soglia
predefinita richiedeva uno scarto massimo `0.002` dal risultato automatico già
registrato; questo è un controllo tecnico del percorso di metrica, non una
nuova selezione del modello.

## Risultati

Tutti i valori seguenti sono mAP@50 calcolati dallo stesso evaluator standalone
per le tre modalità:

| Seed | Additive VIS+IR | FAM VIS+IR | Additive VIS | FAM VIS | Additive IR | FAM IR |
|---:|---:|---:|---:|---:|---:|---:|
| 40 | 0.2321 | 0.1904 | 0.1710 | 0.1547 | 0.0287 | 0.0375 |
| 41 | 0.2751 | 0.2761 | 0.1995 | 0.2440 | 0.0259 | 0.0364 |
| 42 | 0.2506 | 0.1965 | 0.2093 | 0.1779 | 0.0257 | 0.0356 |
| 43 | 0.2174 | 0.2619 | 0.1784 | 0.2281 | 0.0329 | 0.0374 |
| 44 | 0.2738 | 0.1817 | 0.2240 | 0.1599 | 0.0251 | 0.0377 |

| Modalità | Additive, media ± SD | FAM, media ± SD | Delta medio FAM − Additive | Vittorie FAM |
|---|---:|---:|---:|---:|
| VIS+IR | 0.2498 ± 0.0254 | 0.2213 ± 0.0441 | −0.0285 ± 0.0526 | 2/5 |
| VIS | 0.1964 ± 0.0218 | 0.1929 ± 0.0407 | −0.0035 ± 0.0494 | 2/5 |
| IR | 0.0277 ± 0.0032 | 0.0369 ± 0.0009 | +0.0092 ± 0.0030 | 5/5 |

Per VIS+IR il delta appaiato ha IC t al 95% `[-0.0938, +0.0369]`, test t
appaiato `p=0.2930` e Wilcoxon esatto bilaterale `p=0.4375`. Per VIS l'IC è
`[-0.0649, +0.0578]`, con `p=0.8807` e `p=1.0000`. Entrambe le condizioni
restano altamente variabili e non supportano un beneficio YOLO del FAM.

In IR il delta è positivo in tutti i seed, con IC t al 95%
`[+0.0056, +0.0129]`, test t appaiato `p=0.0022` e Wilcoxon esatto
`p=0.0625`. Con soltanto cinque coppie i test restano esplorativi; soprattutto,
la prestazione assoluta FAM è appena `0.0369` mAP@50 e non rappresenta una
robustezza operativamente utile.

In VIS-only e IR-only il FAM è bypassato perché è disponibile una sola
modalità. Le differenze misurano quindi l'effetto che il training con FAM ha
avuto sui pesi della backbone e del neck condiviso, non un beneficio diretto
dell'allineamento durante l'inferenza mono-modale. Il vantaggio IR 5/5 è un
risultato secondario interessante, ma non compensa la perdita VIS+IR.

## Audit del sanity check VIS+IR

L'evaluator standalone ha prodotto valori VIS+IR sistematicamente maggiori
del validator Ultralytics già eseguito al termine del training. Rispetto ai
riferimenti arrotondati bloccati nello YAML, gli scarti assoluti sono compresi
fra `0.001004` e `0.002077`; il massimo supera la tolleranza predefinita di
`0.000077`. Anche usando i valori W&B a piena precisione, il massimo è
`0.002062`.

La tolleranza non è stata aumentata post-hoc e `vis_ir_sanity_passed` rimane
`false`. Il segno comune e l'entità degli scarti sono compatibili con la
differenza sistematica già documentata fra AP Ultralytics e AP torchmetrics;
i checkpoint, le 708 immagini, le 1.770 annotazioni e il ranking appaiato sono
corretti. Per il confronto tra modalità si usano i valori standalone, perché
VIS+IR, VIS e IR sono così calcolati dallo stesso evaluator. La tabella
automatica originale resta il risultato primario del protocollo di training.

Dopo l'esecuzione è stata corretta soltanto la semantica del metadato:
`protocol_complete` indica la presenza delle 30 unità sperimentali, mentre
`vis_ir_sanity_passed` conserva separatamente l'esito negativo del controllo.
Nessun JSON grezzo, checkpoint, risultato o valore di tolleranza è stato
modificato.

## Sequenza operativa

YAML, runner, test e questa nota sono stati versionati prima dell'inferenza. Il
dry run usato è:

```bash
MPLCONFIGDIR=/tmp/matplotlib-yolo-final-eval \
YOLO_CONFIG_DIR=/tmp/yolo-final-eval-config \
/home/donato/miniconda3/envs/sarfusion/bin/python \
  scripts/run_yolo_final_modality_evaluation.py \
  --device cuda \
  --dry-run
```

Lo smoke test ha usato una directory distinta, così non può essere confuso con
i risultati finali:

```bash
MPLCONFIGDIR=/tmp/matplotlib-yolo-final-eval \
YOLO_CONFIG_DIR=/tmp/yolo-final-eval-config \
/home/donato/miniconda3/envs/sarfusion/bin/python \
  scripts/run_yolo_final_modality_evaluation.py \
  --device cuda \
  --configurations additive \
  --seeds 40 \
  --modalities vis_ir vis ir \
  --max-batches 1 \
  --output-dir out/yolo_final_modality_evaluation_smoke
```

La campagna completa è stata eseguita con:

```bash
MPLCONFIGDIR=/tmp/matplotlib-yolo-final-eval \
YOLO_CONFIG_DIR=/tmp/yolo-final-eval-config \
PYTHONUNBUFFERED=1 \
/home/donato/miniconda3/envs/sarfusion/bin/python \
  scripts/run_yolo_final_modality_evaluation.py \
  --device cuda \
  2>&1 | tee yolo_final_modality_evaluation.log
```

Il runner resta resumable: i JSON compatibili già completi vengono saltati.

## Criteri di completamento

I criteri di completamento verificati sono:

- 30 JSON grezzi: due configurazioni per cinque seed per tre modalità;
- `out/yolo_final_modality_evaluation/yolo_final_modality_evaluation.json` con
  `protocol_complete: true`;
- una tabella CSV di 30 righe;
- dieci confronti VIS+IR standalone tutti vicini ai test automatici, con la
  deviazione marginale dalla tolleranza documentata sopra;
- CSV compatto e figura appaiata copiati negli artefatti versionati della tesi.
