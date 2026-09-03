# YOLO26 dual-backbone — esito Stage A

## Stato finale

La linea YOLO26 è **completata e chiusa allo Stage A**. Il controllo Additive
seed 40 ha completato sia il pilot originario sia l'unico repair consentito,
ma non ha raggiunto una baseline sufficientemente vitale. Il candidato FAM,
i seed 41--44 e uno Stage B non sono quindi autorizzati.

Questo esito non dimostra che FAM sia inefficace con YOLO26: il FAM non è stato
addestrato perché è fallito il prerequisito del controllo. Dimostra invece che
il port YOLO26 dual-backbone, con la recipe e lo split congelati, non fornisce
un controllo scientificamente utile per misurare il delta FAM--Additive.

## Risultati

| Run | Best epoch | Best mAP@50 | Best mAP@50--95 | Epoch 50 mAP@50 | Esito |
| --- | ---: | ---: | ---: | ---: | --- |
| Pilot Additive v1 | 2 | 0,06472 | 0,01907 | 0,00001 | integro, prestazioni collassate |
| Repair Additive v1 | 1 | 0,04353 | 0,02145 | 0,00511 | integrità passata, vitalità fallita |

Per il repair, il gate considerava soltanto le epoche 4--50 e richiedeva
almeno una `mAP@50 >= 0,10`. Il massimo eleggibile è `0,01535` all'epoca 13;
46 epoche su 50 restano sotto `0,01` e nessuna raggiunge la soglia.

## Integrità del repair

- 50/50 epoche e 50 record di selezione checkpoint;
- 3.123 coppie train e 896 validation, hash contenuto dataset
  `c92b43c2f69b35e61816f8c2c95f8bea76bec2a1eaa112f36f4303ed92bbdbaf`;
- `warmup_bias_lr=0.0`; massimo learning rate di fine epoca del gruppo bias
  `0,00095999`, uguale agli altri gruppi entro `1e-12`;
- restore stretto di `best.pt` e `last.pt`;
- replay del best `0,0434207214` contro `0,04353` live, errore assoluto
  `0,000109279`, inferiore alla tolleranza `0,003`;
- nessuna valutazione sul test set;
- stato audit finale: `control_integrity_passed_vitality_failed`.

## Interpretazione

Il pilot aveva rivelato un problema reale della recipe: l'uso esplicito di
AdamW aveva lasciato `warmup_bias_lr=0.1`, con learning rate iniziale dei bias
di due ordini di grandezza superiore agli altri gruppi. Il repair ha isolato e
corretto soltanto questo parametro. L'audit conferma che la correzione è stata
applicata, ma la prestazione non è migliorata; l'interazione del warmup non era
quindi una spiegazione sufficiente del collasso.

Il comportamento resta compatibile con più fattori non isolati:

- batch fisico 4, due backbone con BatchNorm e modal dropout: ciascun ramo può
  osservare pochissimi campioni effettivi per step;
- somma delle feature RGB e IR prima del neck, che modifica la distribuzione
  vista dai pesi COCO iniziali quando entrambe le modalità sono presenti;
- forte difficoltà degli oggetti molto piccoli nella development-validation e
  differenza di distribuzione rispetto alle sequenze train;
- recipe di fine-tuning molto distante per batch e scala di ottimizzazione
  dalla recipe ufficiale YOLO26.

Questi punti sono ipotesi diagnostiche, non cause dimostrate. Separarli
richiederebbe un'altra campagna di ablation sul controllo; il protocollo aveva
però limitato esplicitamente il recupero a un solo repair per evitare tuning
post-hoc sulla stessa validation. Non è inoltre corretto confrontare
direttamente questi valori con YOLOv10: split, orizzonte, batch e regola di
checkpoint delle vecchie run non erano equivalenti.

## Decisione

Non eseguire YOLO26 + FAM, altri seed o una terza taratura. Conservare pilot e
repair come esperimenti negativi detector-level e mantenere RT-DETR + FAM
standard come risultato principale della tesi. L'esito supporta una
conclusione circoscritta alla configurazione valutata, non l'affermazione che
le prestazioni del task siano in assoluto non migliorabili.

I dati di audit versionati sono in
[`Search_and_Rescue/results/yolo26_additive_seed40_stage_a_repair_v1.json`](Search_and_Rescue/results/yolo26_additive_seed40_stage_a_repair_v1.json).

