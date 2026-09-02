# RT-DETRv2 + FAM: protocollo Stage A

## Stato

**Stage A seed 40 completato; gate di allocazione fallito e ramo chiuso.** Il
controllo RT-DETRv2 dual-backbone con fusione additiva ottiene `0,181978`
best validation mAP@50, mentre lo stesso detector con FAM standard
`current_dcnv2` ottiene `0,158466`. Il delta preregistrato è `-0,023512`, a
fronte della soglia `+0,010000`; seed 41--44 e Stage B non sono autorizzati.
Tutti i controlli d'integrità, incluso il replay indipendente esatto dei due
checkpoint `best`, sono superati. La baseline RGB standard resta soltanto una
verifica della parità del port e non è il controllo scientifico della FAM.

Il probe tecnico `v1` (W&B `aki9lfef`, 2 settembre 2026) si è arrestato al
primo batch Additive prima di produrre metriche: la sottoclasse custom veniva
instradata da Transformers verso la loss generica per object detection invece
che verso la loss RT-DETRv2. Il contratto è stato reso esplicito e coperto da
un test forward/loss/backward; quel tentativo resta escluso e non viene
sovrascritto. Il probe `v2` ha poi completato entrambi i bracci (W&B Additive
`qp9bh8e1`, FAM `96qphw09`), ma viene conservato come pre-freeze e sostituito
da `v3` dopo l'hardening dell'inizializzazione e del checkpoint. Il probe `v3`
ha completato i 20 step e la validation intera in entrambi i bracci (W&B
Additive `i9h5ytq1`, best mAP@50 `0,1201647`; FAM `fti59iq9`, best mAP@50
`0,1305213`). Queste metriche sono diagnostiche, sono state osservate soltanto
dopo avere congelato il gate scientifico e non entrano nella campagna né
autorizzano altri seed.

File operativi:

- `requirements-rtdetrv2.txt`: overlay dell'ambiente isolato;
- `parameters/RTDETR/rtdetr_v2_fam_runtime_probe.yaml`: due probe tecnici
  Additive/FAM da 20 step, esclusi dalla campagna;
- `parameters/RTDETR/rtdetr_v2_additive_sequence_validation_seed40.yaml`:
  controllo appaiato;
- `parameters/RTDETR/rtdetr_v2_fam_sequence_validation_seed40.yaml`:
  candidata seed 40;
- `scripts/audit_rtdetr_v2_stage_a_checkpoint.py`: audit stretto dei
  checkpoint e dell'aggiornamento FAM;
- `scripts/replay_rtdetr_v2_stage_a_validation.py`: replay validation
  indipendente e decisione automatica del gate;
- `notes/Search_and_Rescue/results/rtdetr_v2_fam_stage_a_validation.csv`:
  risultato canonico sintetico.

## Modifica valutata

Il detector e il checkpoint diventano RT-DETRv2-R50 ufficiali. Restano
invariati split WiSARD, resize a 640, dual backbone RGB/IR, Modal Dropout,
fusione additiva, optimizer, learning rate, numero di epoche, metrica di
selezione e regola del checkpoint. Nel solo braccio candidato, tre FAM DCNv2
allineano le feature IR P3/P4/P5 prima della somma.

L'inizializzazione FAM è congelata come `historical_hf_post_init`. Il modulo
grezzo azzera il predittore degli offset, ma il modello RT-DETR v1 effettivamente
usato negli esperimenti principali applica in seguito `post_init()` di Hugging
Face e ottiene pesi casuali dipendenti dal seed. Lo Stage A replica
esplicitamente quel comportamento storico: passare allo zero-init sarebbe una
seconda modifica sperimentale e richiederebbe un'ablation separata.

Il processore immagine lento del checkpoint è selezionato esplicitamente con
`use_fast=false`, oltre al pin della versione, per evitare che il cambio di
default annunciato da Transformers alteri in futuro il preprocessing.

Il checkpoint è `PekingU/rtdetr_v2_r50vd`, bloccato alla revisione Hugging Face
`282494075698cab9faa1096ae26856890030c817`; il file `model.safetensors` di
quella revisione dichiara SHA-256
`3331d977dbc0c7a6cdae9ec0b0b6ad156eb6720d65b7cf0fa710dcc541d88d71`.
Il decoder mantiene `decoder_method=default`: il sampling discreto sarebbe
un'altra ablation.

RT-DETRv2 mantiene il framework RT-DETR e modifica la deformable attention del
decoder; aggiunge inoltre una recipe di training con dynamic augmentation e
iperparametri scale-adaptive. Questo Stage A usa intenzionalmente la recipe
WiSARD congelata per isolare detector/checkpoint e FAM. Non va quindi descritto
come replica integrale della recipe COCO RT-DETRv2. Fonti primarie:
[paper](https://arxiv.org/abs/2407.17140),
[repository ufficiale](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch),
[documentazione Transformers 4.51.3](https://huggingface.co/docs/transformers/v4.51.3/en/model_doc/rt_detr_v2),
[checkpoint R50](https://huggingface.co/PekingU/rtdetr_v2_r50vd).

## Gate 0: integrazione

**Esito: superato il 2 settembre 2026.** L'audit finale ha verificato anche
l'hash del checkpoint ufficiale, l'identità dei 1.093 tensori condivisi fra i
bracci, gradienti finiti per tutte e tre le FAM e un round-trip stretto di
1.105 tensori. Nel trace dei probe `v3`, indici dei campioni, Modal Dropout,
input e stato RNG coincidono fra controllo e candidata nei batch confrontati.

Prima delle run scientifiche devono passare tutti i punti seguenti:

1. ambiente isolato con le versioni esatte dell'overlay;
2. caricamento del checkpoint v2 alla revisione congelata e trasferimento
   stretto backbone base -> RGB e stem medio -> IR;
3. forward standard RGB e forward fusion a quattro canali con output finiti;
4. loss e backward finiti, tre FAM eager visibili all'optimizer e aggiornati;
5. stato iniziale bit-identico per tutti i tensori condivisi fra controllo e
   candidata e stato RNG globale identico dopo la costruzione;
6. salvataggio e ricaricamento stretti del checkpoint custom;
7. completamento dei due probe da 20 step e della validation intera senza
   OOM/NaN. Le metriche dei probe non entrano in alcuna tabella.

La modalità `deterministic=true` non è autorizzata: la patch deterministica
storica copre soltanto RT-DETR v1 e il backward CUDA di `DeformConv2d` non
offre il medesimo contratto. Le run restano appaiate tramite seed e trace, senza
definirle bit-deterministiche.

Il manifest RT-DETRv2 include `main.py`, l'ambiente storico di base, l'overlay
v2 e tutti i file Python sotto `sarfusion/`, non soltanto i moduli
apparentemente attivi. Il
fingerprint registra anche le versioni delle librerie di preprocessing,
metriche e tracking. Il restore è `strict=True`; gli unici nomi ricostruibili
sono alias esatti dello stesso tensore che Safetensors omette intenzionalmente.
Qualunque altra chiave mancante o inattesa resta un errore bloccante.
Il manifest congelato per i due bracci seed 40 contiene 110 file e ha SHA-256
`d32e9870cbd847990f8e65e17b747f41d6294f3b1d749bcb9fecbb838f3d477f`.

## Stage A seed 40

Entrambi i bracci usano:

- train: FHL 0405/0406 + Baker 1, 3.123 frame appaiati;
- validation: intero FHL 0401/0402, 896 frame, già usato come validation
  interna negli esperimenti precedenti;
- nessuna valutazione o selezione su MtErie/test, confirmation set, Carnation
  o stress sintetico;
- seed/data seed/model seed/training seed 40;
- 10 epoche esatte, validation ogni epoca, nessun early stopping;
- AdamW, LR `2e-5`, batch train 4, batch evaluation 12;
- Modal Dropout `[0,2; 0,2; 0,6]` nel contratto di coordinate nativo;
- `best` selezionato su validation mAP@50 con `min_delta=0,001`; `latest`
  salvato soltanto alla fine; metrica primaria = valore del `best`.

Il controllo Additive è stato completato per primo; la candidata è partita
soltanto dopo la sua chiusura regolare. Entrambi provengono dallo stesso
manifest sorgente e dallo stesso fingerprint runtime.

## Gate di allocazione seed 40

Il gate è congelato prima di osservare le metriche:

```text
delta40 = best_mAP50(RT-DETRv2 + FAM)
        - best_mAP50(RT-DETRv2 Additive)

passa se e solo se delta40 >= +0,010000
```

Servono inoltre 10 epoche complete in entrambi i bracci, checkpoint `best`
univoci e ricaricabili strettamente, replay indipendente dei 896 frame entro
`0,0002` dalla summary, valori finiti e prova che i pesi FAM siano cambiati.
mAP@[.50:.95], mAP@75, precision e recall sono descrittivi e non possono
salvare un fallimento della metrica primaria. Il seed 40 è un filtro di
allocazione del compute, non una conclusione di efficacia.

Se uno dei requisiti fallisce, seed 41--44 e Stage B restano chiusi; non si
cambia soglia, checkpoint o iperparametro post-hoc.

### Esito osservato

Le run scientifiche sono W&B `dfp1gm92` per Additive e `vxrin9p2` per FAM;
entrambe hanno exit code 0, 10/10 epoche e 10 validation complete. Il checkpoint
`best` è l'epoca 1 in entrambi i bracci.

| Braccio | best epoca | mAP@[.50:.95] | mAP@50 | mAP@75 | mAR@100 |
|---|---:|---:|---:|---:|---:|
| RT-DETRv2 Additive | 1 | 0,082368 | 0,181978 | 0,060587 | 0,192136 |
| RT-DETRv2 + FAM | 1 | 0,082284 | 0,158466 | 0,075040 | 0,185508 |
| Delta FAM − Additive | — | -0,000084 | **-0,023512** | +0,014453 | -0,006628 |

Il margine rispetto al gate è `-0,033512`. Il miglior mAP@75 descrittivo della
candidata non può compensare il fallimento della metrica primaria. Anche il
valore dell'epoca finale non va sostituito post-hoc al `best`: all'epoca 10
Additive vale `0,072617` mAP@50 e FAM `0,126312`, ma il protocollo aveva
congelato selezione e confronto sui rispettivi checkpoint migliori.

Il replay indipendente del 2 settembre 2026 ha attraversato nuovamente tutti i
896 frame, in 75 batch per braccio, riproducendo entrambi i best mAP@50 con
errore assoluto `0,0`, quindi ben entro la tolleranza `0,0002`. Il restore è
stretto: Safetensors serializza 1.045 tensori Additive e 1.057 FAM; per entrambi
sono stati ricostruiti 48 alias esatti delle teste condivise, ottenendo stati
completi rispettivamente di 1.093 e 1.105 tensori. Gli hash SHA-256 dei
checkpoint `best` sono:

- Additive: `ddec36a41ed160dae81db60fafb14b9e09c627d7560f98d0d0ffa25155c23021`;
- FAM: `82cc1c298fc8dba0c622c81741456fbd59166c650313270a0cd061596f862560`.

Tutti i 12 tensori FAM risultano modificati rispetto all'inizializzazione; la
massima variazione assoluta è `0,004959823`. Le trace contengono 200 batch per
braccio con indici, modalità e RNG appaiati; i 50 batch per cui era congelato
anche l'hash degli input coincidono esattamente. Il manifest comune contiene
110 file e conserva SHA-256
`d32e9870cbd847990f8e65e17b747f41d6294f3b1d749bcb9fecbb838f3d477f`.
Gli errori transitori nel log interno riguardano upload e monitoraggio GPU di
W&B; non sono errori del training e gli artefatti locali finali sono integri.

**Decisione:** l'integrità sperimentale passa, il gate di efficacia fallisce.
RT-DETRv2 + FAM standard viene archiviato come esperimento negativo seed 40;
non si eseguono seed 41--44 né Stage B e non si valuta il test set per tentare
di recuperare il risultato. Lo `status=failed` del report indica quindi il
fallimento scientifico del gate, non un errore tecnico. Un singolo seed usato
come filtro di allocazione non dimostra un'inferiorità universale della FAM;
dimostra soltanto che questa configurazione non soddisfa il criterio concordato
per meritare ulteriore compute. Poiché `deterministic=false`, seed e trace
documentano un confronto appaiato ma non autorizzano a definirlo bitwise
deterministico.

## Espansione condizionale: non applicabile

Il piano preregistrato prevedeva configurazioni fresche per seed 41--44 solo
dopo il passaggio documentato del gate. L'eventuale promozione Stage A avrebbe
richiesto:

```text
media dei cinque delta appaiati >= +0,010000
e almeno 4/5 delta strettamente positivi
e integrità/replay 5/5
```

Poiché il seed 40 non ha superato il filtro, questa espansione non è stata
eseguita. Non esistono quindi stime multi-seed, test inferenziali o Stage B per
questa variante; MtErie è rimasto escluso.

## Comandi archiviati

L'esperimento è stato eseguito nell'ambiente separato seguente, lavorando
offline dopo il download iniziale:

```bash
conda create --name sarfusion-rtdetrv2 --clone sarfusion
conda run -n sarfusion-rtdetrv2 python -m pip install \
  -r requirements-rtdetrv2.txt
```

Eseguire prima i test e il probe. Solo dopo Gate 0:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
MPLCONFIGDIR=/tmp/matplotlib-rtdetrv2 \
YOLO_CONFIG_DIR=/tmp/yolo-rtdetrv2 PYTHONUNBUFFERED=1 \
conda run --no-capture-output -n sarfusion-rtdetrv2 python main.py experiment \
  --parameters parameters/RTDETR/rtdetr_v2_additive_sequence_validation_seed40.yaml

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
MPLCONFIGDIR=/tmp/matplotlib-rtdetrv2 \
YOLO_CONFIG_DIR=/tmp/yolo-rtdetrv2 PYTHONUNBUFFERED=1 \
conda run --no-capture-output -n sarfusion-rtdetrv2 python main.py experiment \
  --parameters parameters/RTDETR/rtdetr_v2_fam_sequence_validation_seed40.yaml
```
