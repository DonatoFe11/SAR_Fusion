# YOLO26 Stage A — repair v1 del warmup AdamW

## Perché esiste questo amendment

Il controllo Additive originario ha completato 50/50 epoche e ha superato
tutti i controlli di integrità. Il best è però un picco all'epoca 2
(`mAP50=0,06472`), mentre all'epoca 50 la mAP50 è `0,00001`.

La configurazione specificava esplicitamente `optimizer=AdamW`. In
Ultralytics 8.4.138 questo evita il ramo `optimizer=auto`, che per AdamW forza
`warmup_bias_lr=0.0`. È quindi rimasto il valore `0.1`: il CSV registra per il
gruppo bias LR `0,0670423` all'epoca 1 e `0,0340291` all'epoca 2, contro
`0,000332907` e `0,000653048` per gli altri gruppi. L'integrità del loading
COCO, dei dati 4-canali, della loss, dei checkpoint e del replay è stata
verificata; questa interazione della recipe è la causa correggibile più
diretta, pur non costituendo da sola una prova causale del collasso.

Il run originario resta archiviato come pilot negativo e non viene
sovrascritto. Il suo audit non può autorizzare il FAM repair.

## Unica modifica

Additive repair e FAM repair sono congelati prima del nuovo controllo e
differiscono dal protocollo originario soltanto per:

```yaml
warmup_bias_lr: 0.0
```

Restano invariati seed 40, dataset e relativo hash, pesi, batch fisico 4,
`nbs=16`, AdamW, `lr0=0.001`, `lrf=0.01`, warmup di 3 epoche, 50 epoche,
modal dropout feature-gated 20/20/60, augmentazioni, selezione di `best.pt`,
tolleranza replay e assenza di valutazione test. In particolare non si cambia
anche il batch, perché il pilot non isola causalmente l'effetto delle
statistiche BatchNorm.

## Gate preregistrati dopo il pilot

L'audit repair richiede tutti i gate di integrità precedenti e inoltre:

- LR del gruppo bias finito, mai superiore a `0,001` e uguale agli altri due
  gruppi nelle righe per epoca del CSV;
- almeno una epoca tra 4 e 50 con `mAP50 >= 0,10`.

Se il gate di vitalità fallisce, non si modifica nuovamente la recipe e non si
esegue FAM. Se passa, il candidato FAM usa la configurazione già congelata e
deve migliorare il best Additive di almeno `+0,010000` per accedere ai seed
41--44. Il confronto repair è un amendment di sviluppo informato dal pilot e
non una conferma indipendente, poiché la development-validation è già stata
osservata.
