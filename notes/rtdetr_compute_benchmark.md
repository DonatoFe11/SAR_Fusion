# Costo computazionale RT-DETR Additive--FAM

## Stato

Il protocollo `rtdetr_additive_fam_compute_benchmark_v1` è stato congelato il
12 agosto 2026 prima delle misure GPU. Il benchmark confronta soltanto le due
configurazioni finali RT-DETR Additive e FAM `current_dcnv2`; non introduce
training, tuning o selezione di checkpoint.

Configurazione e runner:

- [`rtdetr_additive_fam_compute_benchmark.yaml`](../parameters/RTDETR/rtdetr_additive_fam_compute_benchmark.yaml);
- [`run_rtdetr_compute_benchmark.py`](../scripts/run_rtdetr_compute_benchmark.py).

## Protocollo bloccato

L'architettura non dipende dal seed per parametri o costo. Viene quindi usata
la coppia di checkpoint finali seed 43, già predefinita come coppia tipica per
le figure qualitative, senza selezionarla in base alla velocità:

| Configurazione | Progetto | Checkpoint |
|---|---|---|
| Additive | `RTDETR_Protocol` | seed 43, `latest` |
| FAM | `RTDETR_FAM_Protocol` | seed 43, `latest` |

Le condizioni sono:

- input sintetico fisso `[1, 4, 640, 640]` e pixel mask interamente valida;
- eager inference, `torch.inference_mode()`, tensori FP32 e TF32 disabilitato;
- nessun autocast, `torch.compile`, preprocessing o postprocessing;
- 30 iterazioni di warm-up e 100 misure per trial;
- tre trial per configurazione, con ordine alternato
  Additive--FAM, FAM--Additive, Additive--FAM;
- sincronizzazione CUDA ed eventi CUDA per ogni forward;
- memoria CUDA misurata dopo il caricamento del modello e dell'input, separando
  allocazione di base e picco incrementale del forward;
- hardware, versioni software, parametri, buffer e dimensioni P3/P4/P5 salvati
  nell'output.

L'input sintetico serve solo a fissare forma e carico degli operatori densi:
non vengono calcolate metriche di detection e il contenuto non modifica il
numero di operazioni. La latenza rappresenta un limite superiore di throughput
del solo detector su questa macchina, non il tempo end-to-end di una pipeline
SAR con caricamento, preprocessing, trasferimenti e visualizzazione.

## Conteggio delle operazioni

Non viene presentato un numero FLOPs come esatto. `torch.profiler` attribuisce
FLOPs agli operatori supportati, principalmente convoluzioni e moltiplicazioni
di matrici, ma non copre necessariamente softmax, interpolazioni e tutti gli
operatori custom.

Per il FAM viene inoltre calcolato dalle forme effettivamente osservate il
costo convenzionale di:

- convoluzione `offset_conv`;
- convoluzione DCNv2 trattata come una convoluzione densa con lo stesso kernel.

Si usa la convenzione un MAC = due FLOPs. Il costo analitico DCNv2 non include
campionamento bilineare, applicazione della mask, sigmoid e concatenazione;
costituisce quindi una proxy convenzionale, non una misura completa del lavoro
hardware. Il runner conserva separatamente conteggio profiler, correzione
DCNv2 e totale aggiustato, evitando di nascondere questa limitazione.

## Criteri di completamento

Il benchmark è completo soltanto se:

- entrambi i checkpoint sono risolti univocamente e caricati integralmente;
- parametri non-FAM condivisi e forme P3/P4/P5 sono coerenti;
- sono presenti tre trial e 300 latenze finite per configurazione;
- tutte le misure di memoria sono finite e non negative;
- l'output dichiara hardware, protocol hash e `protocol_complete: true`.

## Risultati

Non ancora eseguiti. Il protocollo deve essere committato e pushato prima della
prima misura GPU.
