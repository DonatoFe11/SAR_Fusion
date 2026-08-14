# Costo computazionale RT-DETR Additive--FAM

## Stato

Il protocollo `rtdetr_additive_fam_compute_benchmark_v1` è stato congelato il
12 agosto 2026 prima delle misure GPU ed è ora **completo**. Il benchmark
confronta soltanto le due configurazioni finali RT-DETR Additive e FAM
`current_dcnv2`; non introduce training, tuning o selezione di checkpoint.

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

Il protocollo è stato eseguito su NVIDIA GeForce RTX 4070 Laptop GPU, con
PyTorch 2.4.0, torchvision 0.19.0 e CUDA 12.1 sotto WSL2. Ogni configurazione
ha 300 forward misurati in tre trial indipendenti. Le deviazioni standard dopo
`±` sono calcolate sulle tre latenze medie di trial, non trattando i 300 forward
come repliche sperimentali indipendenti.

| Configurazione | Parametri | Stato modello | Proxy GFLOPs | Latenza (ms) | Throughput (img/s) | Picco CUDA | Picco incrementale forward |
|---|---:|---:|---:|---:|---:|---:|---:|
| Additive | 66.20 M | 253.44 MiB | 208.37 | 66.91 ± 0.30 | 14.95 | 436.26 MiB | 139.14 MiB |
| FAM | 117.49 M | 449.10 MiB | 304.54 | 78.87 ± 0.37 | 12.68 | 712.18 MiB | 220.65 MiB |

Le medie dei singoli trial sono `66.77`, `67.25` e `66.70` ms per Additive e
`78.49`, `78.88` e `79.24` ms per FAM. Considerando tutti i forward, le mediane
sono rispettivamente `66.29` e `78.82` ms e i percentili 95 `73.18` e `83.45`
ms.

Rispetto ad Additive, FAM introduce:

- 51.290.705 parametri (`+77,5%`) e 195,66 MiB di stato del modello;
- `+96,18` GFLOPs nella proxy (`+46,2%`);
- `+11,96` ms per forward (`+17,9%`) e un throughput teorico del solo detector
  inferiore del `15,2%`;
- `+275,92` MiB di picco CUDA totale e `+81,51` MiB di picco incrementale del
  forward.

La proxy FAM di `304,54` GFLOPs combina `213,95` GFLOPs attribuiti dagli
operatori supportati dal profiler e `90,60` GFLOPs equivalenti per le tre
convoluzioni DCNv2 non conteggiate dal profiler. Il costo convenzionale
complessivo dei tre FAM, includendo anche i predittori degli offset già
osservati dal profiler, è `96,17` GFLOPs. I valori non comprendono tutto il
costo del campionamento bilineare e della modulazione DCNv2: la latenza GPU
misurata resta il dato operativo, mentre i GFLOPs servono come proxy
architetturale riproducibile.

## Controllo dell'isolamento

La prima implementazione del runner ricostruiva i modelli nello stesso processo
Python. Un controllo sui valori di memoria ha mostrato che riferimenti interni
del detector restavano vivi fra le ricostruzioni: il picco cresceva
artificialmente da trial a trial fino a circa 2,4 GiB. Quel risultato è stato
rigettato prima dell'interpretazione e non è riportato come misura.

Il runner è stato corretto senza cambiare il protocollo scientifico: ogni
coppia trial/configurazione viene ora eseguita in un nuovo processo, che termina
prima della misura successiva. Nel risultato valido le allocazioni di base sono
identiche nei tre trial (`297,13` MiB Additive e `491,54` MiB FAM), così come i
picchi. Questa separazione rende indipendente lo stato dell'allocatore CUDA e
conserva l'ordine alternato predefinito.

## Artefatti

Il riepilogo versionato è
[`rtdetr_compute_benchmark.csv`](Search_and_Rescue/results/rtdetr_compute_benchmark.csv).
Il JSON completo locale è
`out/rtdetr_additive_fam_compute_benchmark.json`, dichiara
`protocol_complete: true` e ha SHA-256:

```text
cb942c8876f763d17b14bfc40a0e3371efd10c907266ab8cd9b2c41ca5902cbb
```

Il risultato mostra un trade-off netto: nel benchmark interno FAM migliora la
detection rispetto ad Additive, ma non è un miglioramento gratuito. La tesi
deve quindi riportare insieme beneficio di accuratezza, crescita del modello e
costo di inferenza; il termine *real-time* può essere usato solo riferendosi
alla macchina e allo scope qui dichiarati, non a una pipeline SAR end-to-end.
