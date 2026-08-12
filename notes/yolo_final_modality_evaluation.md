# Valutazione finale YOLO per modalità

## Stato

Protocollo congelato il 12 agosto 2026, prima della nuova inferenza sui dieci
checkpoint finali. Questa campagna non esegue training e non seleziona epoche o
checkpoint sul test.

## Domanda sperimentale

Il confronto VIS+IR sui `last.pt` ha mostrato che, dopo 200 epoche, FAM non
migliora in media la baseline YOLO Additive. La valutazione finale deve ancora
stabilire se il comportamento cambi quando manca RGB oppure IR.

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

La modalità VIS+IR viene ricalcolata anche dal runner standalone. Deve
coincidere entro `0.002` con il risultato automatico già registrato; questo è
un controllo tecnico del percorso di metrica, non una nuova selezione del
modello.

## Sequenza operativa

Prima dell'inferenza completa vanno versionati YAML, runner, test e questa
nota. Poi si esegue il dry run:

```bash
MPLCONFIGDIR=/tmp/matplotlib-yolo-final-eval \
YOLO_CONFIG_DIR=/tmp/yolo-final-eval-config \
/home/donato/miniconda3/envs/sarfusion/bin/python \
  scripts/run_yolo_final_modality_evaluation.py \
  --device cuda \
  --dry-run
```

Lo smoke test opzionale usa una directory distinta, così non può essere
confuso con i risultati finali:

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

La campagna completa si avvia con:

```bash
MPLCONFIGDIR=/tmp/matplotlib-yolo-final-eval \
YOLO_CONFIG_DIR=/tmp/yolo-final-eval-config \
PYTHONUNBUFFERED=1 \
/home/donato/miniconda3/envs/sarfusion/bin/python \
  scripts/run_yolo_final_modality_evaluation.py \
  --device cuda \
  2>&1 | tee yolo_final_modality_evaluation.log
```

In caso di interruzione si rilancia lo stesso comando. I JSON compatibili già
completi vengono saltati; non usare `--force` senza avere prima ispezionato
l'output esistente.

## Criteri di completamento

Al termine devono esistere:

- 30 JSON grezzi: due configurazioni per cinque seed per tre modalità;
- `out/yolo_final_modality_evaluation/yolo_final_modality_evaluation.json` con
  `protocol_complete: true`;
- una tabella CSV di 30 righe;
- dieci confronti VIS+IR standalone coerenti con i test automatici già
  registrati.

Solo dopo questi controlli verranno calcolati i confronti appaiati e aggiornate
le conclusioni sulla trasferibilità e sulla robustezza mono-modale.
