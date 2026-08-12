# Stress test Carnation RT-DETR Additive--FAM

## Stato

Il protocollo `rtdetr_carnation_stress_test_v1` è stato congelato il 12 agosto
2026 **prima di eseguire inferenza o osservare metriche su Carnation**. La
campagna è uno stress test esterno singolo e non modifica il modello principale,
la soglia, i checkpoint o il protocollo MtErie.

Il file sorgente del protocollo è
[`rtdetr_carnation_stress_test.yaml`](../parameters/RTDETR/rtdetr_carnation_stress_test.yaml).
Il runner resumable è
[`run_rtdetr_carnation_stress_test.py`](../scripts/run_rtdetr_carnation_stress_test.py).

## Domanda sperimentale

La coppia Carnation presenta una differenza di scala fra RGB e IR più forte
rispetto alle acquisizioni del benchmark interno. La domanda è quindi:

> il vantaggio medio del FAM standard rispetto ad Additive persiste, si annulla
> o cambia segno quando i dieci checkpoint finali vengono trasferiti senza
> tuning a questa acquisizione con forte mismatch geometrico?

Un risultato negativo descriverà un limite di generalizzazione sotto scale
diverse fra sensori. Non confuterà il confronto appaiato MtErie e non potrà
essere usato per scegliere retroattivamente un'altra variante FAM.

## Dati bloccati

Le sole sequenze ammesse sono:

- `210529_Carnation_Enterprise_VIS_0023`;
- `210529_Carnation_Enterprise_IR_0024`.

Il ramo VIS contiene 739 frame, mentre il ramo IR contiene un frame finale
aggiuntivo (`1045`). Il protocollo usa l'intersezione degli identificatori
numerici, ottenendo gli stessi **739 frame** in VIS+IR, VIS e IR. Nessun frame
viene scelto sulla base delle predizioni.

Prima di ogni inferenza il runner verifica:

- hash dei 739 identificatori comuni;
- inventario di immagini e annotazioni;
- 1.942 box e quattro frame vuoti nel ramo VIS;
- 1.713 box e quattro frame vuoti nel ramo IR;
- esclusione del solo frame IR `1045`.

VIS+IR viene costruito dal loader esistente tramite `adapt_ir2rgb` e valutato
nel sistema di coordinate e con le annotazioni VIS. VIS-only usa le annotazioni
VIS; IR-only usa le annotazioni IR. Nelle modalità singole i canali mancanti
sono azzerati, coerentemente con la valutazione finale RT-DETR. Modal Dropout è
sempre disattivato in inferenza.

## Modelli e protocollo bloccato

Sono ammesse soltanto due configurazioni:

| Configurazione | Progetto checkpoint | FAM |
|---|---|---:|
| Additive | `RTDETR_Protocol` | no |
| FAM standard | `RTDETR_FAM_Protocol` | `current_dcnv2` |

Per entrambe:

- seed appaiati `40--44`;
- checkpoint finale `latest` del protocollo fisso a 10 epoche;
- soglia di confidenza `0.01`;
- seed di evaluation `42`;
- modalità VIS+IR, VIS e IR;
- nessun training, early stopping, tuning o selezione di seed;
- unità sperimentale: checkpoint/seed.

La campagna contiene quindi 2 configurazioni × 5 seed × 3 modalità = **30
valutazioni**. Per ogni modalità verranno riportati i valori per seed, media,
mediana, deviazione standard campionaria, min--max, IC t al 95%, delta appaiati
FAM meno Additive, vittorie su cinque seed e test appaiati esplorativi. La
mAP@50 resta la metrica primaria; mAP COCO, mAP@75, mAR e metriche per
dimensione vengono conservate come secondarie.

## Vincoli interpretativi predefiniti

I risultati Carnation non possono essere usati per:

- selezionare modello, seed, checkpoint o soglia;
- progettare e provare un'altra correzione sullo stesso stress set;
- decidere un nuovo training sulla base della sola metrica Carnation;
- presentare Carnation come campione rappresentativo della distribuzione SAR;
- sostituire MtErie nella conclusione sperimentale principale.

Carnation viene consultata una sola volta dopo il congelamento e il versionamento
di questi vincoli.

## Sequenza operativa

Verifica dell'inventario, senza caricare checkpoint o calcolare metriche:

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
/home/donato/miniconda3/envs/sarfusion/bin/python \
  scripts/run_rtdetr_carnation_stress_test.py \
  --prepare-only
```

Risoluzione dei dieci checkpoint, ancora senza inferenza:

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
/home/donato/miniconda3/envs/sarfusion/bin/python \
  scripts/run_rtdetr_carnation_stress_test.py \
  --dry-run
```

Dopo commit e push del protocollo, smoke tecnico su un solo batch e una coppia
di checkpoint, in una directory separata:

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
MPLCONFIGDIR=/tmp/matplotlib-rtdetr-carnation-smoke \
/home/donato/miniconda3/envs/sarfusion/bin/python \
  scripts/run_rtdetr_carnation_stress_test.py \
  --device cuda \
  --configurations additive fam \
  --seeds 43 \
  --modalities vis_ir \
  --max-batches 1 \
  --output-dir out/rtdetr_carnation_stress_test_smoke
```

Campagna completa resumable:

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
MPLCONFIGDIR=/tmp/matplotlib-rtdetr-carnation \
PYTHONUNBUFFERED=1 \
/home/donato/miniconda3/envs/sarfusion/bin/python \
  scripts/run_rtdetr_carnation_stress_test.py \
  --device cuda \
  2>&1 | tee rtdetr_carnation_stress_test.log
```

Il completamento richiede 30 JSON compatibili, 739 campioni in ciascuno e
`protocol_complete: true` nell'aggregato. Non usare `--force` salvo dopo avere
ispezionato manualmente un output incompatibile.

## Risultati

Non ancora eseguiti. Questa sezione verrà compilata soltanto dopo il
versionamento del protocollo e il completamento delle 30 valutazioni.
