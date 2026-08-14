# Evoluzione del Feature Alignment Module (FAM): Da Bug Architetturale a Regolarizzazione Stocastica

> **Stato storico.** I Modelli A–E riportati qui sono singole run della fase di
> sviluppo. Servono a ricostruire l'origine delle ipotesi, ma non stabiliscono
> quale configurazione sia migliore. La campagna finale a cinque seed ha
> selezionato il FAM standard senza SSJ; vedi la sezione conclusiva e
> [`rtdetr_reproducibility.md`](rtdetr_reproducibility.md).

## Introduzione
Durante gli esperimenti con l'architettura RT-DETR per la fusione RGB-IR (dataset `vis_ir`), è emerso un comportamento anomalo. Un'implementazione del Feature Alignment Module (FAM) contenente un "bug" architetturale ha superato ampiamente le prestazioni della sua controparte formalmente corretta, raggiungendo un eccezionale `0.4286` di mAP@50. 

Per indagare questa anomalia e trasformarla in un'ipotesi metodologica verificabile, è stato condotto uno studio su più architetture. Di seguito vengono spiegati i test effettuati, dal rilevamento del bug formale fino all'introduzione di un regolarizzatore geometrico. Tutti i valori riportati fanno riferimento alla threshold `0.01`.

| Modalità In Input | Modello A "Lazy Bug" (`fusion_rtdetr_fam`) | Modello B "Corretto" (`use_fam=True, freeze=False`) | Modello C "Eager Freeze" (`use_fam=True, freeze=True`) | Modello D "Eager + 40% Dropout" | Modello E "Eager + SSJ" |
|-------------------|--------------------------------------------|------------------------------------------------------|--------------------------------------------------------|---------------------------------|-------------------------|
| **IR (Unimodale)**| 0.0054                                     | 0.2629                                               | 0.2250                                                 | 0.2248                          | 0.1818                  |
| **VIS (Unimodale)**| 0.3000                                     | 0.2620                                               | 0.2517                                                 | 0.2716                          | 0.2697                  |
| **VIS_IR (Fusione)**| 0.4286                                     | 0.3960                                               | 0.3857                                                 | 0.3936                          | **0.4381**              |

---

## Il Funzionamento del Feature Alignment Module (FAM)
Per comprendere le dinamiche del bug e delle successive soluzioni, è fondamentale riepilogare il processo logico-matematico dell'allineamento a livello di tensori (Feature-Level Alignment, **non** Image-Level Alignment), che accomuna tutte le architetture testate:
1. **Estrazione Indipendente**: L'immagine RGB grezza entra nel suo backbone (ResNet) e diventa una *Feature Map* astratta. Parallelamente, l'immagine IR entra nel suo backbone, diventando una *Feature Map* infrarossa.
2. **Predizione degli offset (Concatenazione)**: Il FAM riceve insieme le due feature, concatenate lungo i canali (`torch.cat([rgb_feat, ir_feat], dim=1)`). La convoluzione `offset_conv` è addestrata a predire un campo di offset e una maschera a partire da questa informazione congiunta; non misura direttamente il disallineamento dei sensori.
3. **Deformazione (Warping)**: Gli offset predetti vengono applicati alla sola *Feature Map* IR tramite `deform_conv`, con RGB come guida indiretta. L'obiettivo è rendere le feature più compatibili per la fusione, non garantire una registrazione perfetta posizione per posizione.
4. **Fusione Additiva**: Solo alla fine di questo allineamento le due astrazioni (la *Feature Map* RGB originale intonsa e la *Feature Map* IR deformata/allineata) vengono matematicamente sommate e fuse.

---

## Fase 1: Il Modello Originale e il Collasso IR (Modello A)
Il modello pioniere (`rtdetr_fusion_fam.py`) ha registrato una mAP@50 di fusione elevata, ma `0.0054` in IR-only. La causa tecnica verificabile è la **lazy initialization** del FAM; il meccanismo preciso con cui il resto della rete ha reagito a quel difetto resta un'interpretazione dei risultati.

I layer di questo modello venivano istanziati dinamicamente durante il primo `.forward()`. Poiché l'ottimizzatore (`AdamW`) viene creato prima di quel forward, i parametri FAM non entravano nei suoi param-group e quindi non venivano aggiornati. Le conseguenze tecniche osservabili sono:
1. I parametri della rete pilota (`offset_conv`) erano inizializzati a `0` e bloccati. Generavano costantemente un offset nullo e una maschera di modulazione pari a `0.5` (la Sigmoide di 0).
2. I parametri della `deform_conv` sfruttavano l'inizializzazione di default: **una matrice di pesi fissa ma puramente casuale**.
3. Il FAM applicava quindi alle feature IR una trasformazione convoluzionale inizializzata casualmente, a offset nulli e con maschera circa $0.5$. Non è un'identità; senza aggiornamenti dei suoi pesi può degradare o rendere poco utilizzabile l'informazione termica.
4. La fusione con il ramo RGB è *additiva*. Il valore IR-only molto basso è compatibile con il fatto che l'ottimizzazione abbia ridotto il contributo utile del ramo termico per minimizzare la loss di fusione. Non sono state però misurate né dimostrate una cancellazione letterale dei pesi della ResNet IR né l'azzeramento esatto delle sue feature.

L'alta mAP di fusione del Modello A (`0.4286`) insieme alla sua bassa mAP IR-only suggerisce che il modello abbia fatto molto meno affidamento sul segnale termico rispetto alle varianti eager. È una lettura coerente con i risultati, ma non dimostra che il ramo RGB abbia sostenuto letteralmente il 100% dell'apprendimento.

---

## Fase 2: Il Modello "Corretto" e la Modality Interference (Modello B)
Per standardizzare il codice (`rtdetr_fusion.py`), il FAM è stato riportato ai canoni di PyTorch usando la **Eager Initialization**: dichiarare ogni componente all'interno del costruttore `__init__`.
Questo ha permesso ad AdamW di aggiornare l'intero FAM. Nella run riportata, la mAP IR-only è salita a **0.2629**.

La mAP di fusione è scesa a `0.3960`, mentre VIS-only è `0.2620`. Una possibile interpretazione è una maggiore dipendenza dal ramo termico o una *modality interference*; i soli risultati finali non isolano però la causa della differenza.

---

## Fase 3: La Verifica del Rumore Fisso (Modello C)
Cercare di riprodurre i risultati del "Bug" quantificando l'impatto di un FAM disattivato ho prodotto il **Modello C** (`use_fam=True, freeze=True`, ovvero `requires_grad=False`).
A differenza del Modello A (dove l'ottimizzatore ignorava l'esistenza del layer a causa del bug *lazy*), qui l'istanza è dichiarata correttamente nell'inizializzazione Eager. 

Proiettando questa meccanica sul processo a 4 step del FAM:
- Durante la **Trasformazione (Step 3)**, la `deform_conv` congelata applica una trasformazione convoluzionale costante e deterministica rispetto al suo stato iniziale. Con `offset_conv` inizializzata a zero non introduce inizialmente uno spostamento geometrico, ma non è comunque un'identità perché conserva pesi convoluzionali propri.
- Poiché la rete è istanziata e collegata correttamente, la funzione `autograd` di PyTorch riesce a fluire passivamente *attraverso* il FAM congelato, propagando i gradienti a ritroso fino all'IR backbone sottostante. 

L'IR backbone resta invece nell'ottimizzatore e riceve gradienti attraverso il FAM congelato. La risalita della mAP IR-only a `0.2250` è compatibile con un adattamento delle feature IR alla trasformazione fissa, ma non ne costituisce una dimostrazione diretta.

La fusione scende a `0.3857`. Il confronto con il Modello A motiva l'ipotesi che una perturbazione variabile, anziché una trasformazione fissa, possa regolarizzare meglio la fusione; non permette di attribuire il divario a una sola causa.

---

## Fase 4: L'introduzione dello Spatial Dropout (Modello D)
Sulla base dei risultati precedenti, si è tentato di "danneggiare" attivamente la mappa termica prima della fusione additiva (Step 4) pur mantenendo l'intero FAM in addestramento attivo (`use_fam=True, freeze=False`).

In questa variante si è inserito uno `Spatial Dropout` al 40%. Proiettando questa meccanica sul processo a 4 step, il Modello D si comporta esattamente come il regolare Modello B fino allo Step 3 compreso:
- Il FAM completa l'**Estrazione (Step 1)**, la **Predizione degli offset (Step 2)** e la **Trasformazione (Step 3)** con parametri allenabili (`requires_grad=True`).
- **Intervento del Dropout**: Subito prima della **Fusione Additiva (Step 4)**, il layer di Dropout agisce **esclusivamente sulla Feature Map IR de-formata**. Azzera (porta a zero) randomicamente il 40% dei canali termici, lasciando la *Feature Map* RGB totalmente intatta.
- Quando avviene la **Fusione Additiva (Step 4)**, per il 40% dei canali mascherati l'operazione matematica diventerà banalmente `RGB + 0 = RGB`.

Nella run riportata, VIS-only sale a `0.2716` e la fusione migliora rispetto al Modello C (`0.3936` vs `0.3857`), ma resta sotto il Modello A (`0.4286`) e al modello B (`0.3960`). I dati sono compatibili con una maggiore pressione sul ramo RGB ma non la dimostrano in modo causale.

Una lettura qualitativa è che `Dropout2d` rimuova interi canali e quindi anche informazione semantica, mentre una perturbazione degli offset agisce sulla geometria di campionamento. Il bug lazy non ha però generato *jitter* geometrico: gli offset erano fissi a zero; ciò che restava fisso e non addestrato era la trasformazione convoluzionale del FAM.

---

## Fase 5: L'ipotesi Stochastic Spatial Jitter (SSJ)
Per verificare se una perturbazione geometrica potesse regolarizzare il FAM è
stato introdotto lo **Stochastic Spatial Jitter (SSJ)**.

**Cos'è lo Spatial Jitter?**
Il termine *Jitter* indica un "tremolio" o un'instabilità fluttuante. Qui consiste nell'iniettare rumore stocastico negli offset, cioè nelle coordinate di campionamento della `DeformConv2d`. A differenza di `Dropout2d`, non azzera direttamente canali della feature IR; cambia però indirettamente i valori in uscita perché la convoluzione campiona posizioni diverse. Durante il training rende quindi meno affidabile la geometria con cui l'informazione IR arriva alla fusione.

L'approccio implementato in `rtdetr_fusion.py` inserisce in questo processo il nostro regolarizzatore SSJ in 5 passaggi:
1. L'architettura è regolarmente istanziata ed allenabile in tutte le sue parti (Eager).
2. In fase di training multimodale, il FAM predice gli offset ($\Delta x, \Delta y$) a partire dalle feature RGB e IR (Fase 2 del FAM).
3. **Iniezione del Jitter**: agli offset predetti viene sommato rumore Gaussiano bianco ($\mathcal{N}(0, \sigma^2)$).
4. La mappa di offset rumorosa è usata dalla `deform_conv` per campionare e trasformare solo le **feature map termiche** prima della fusione additiva. Il rumore viene ricampionato a ogni forward di training.
5. In fase di inferenza l'iniezione stocastica è disattivata: il FAM usa gli offset predetti senza rumore aggiuntivo.

### Perché perturbare un offset che il FAM ha appena appreso?

SSJ non equivale a rinunciare al FAM o a tornare al disallineamento originario dei sensori. Senza FAM l'errore RGB--IR è sistematico e può essere ampio. Con FAM, l'offset appreso lo corregge in media; durante il training SSJ aggiunge soltanto una perturbazione locale, a media zero:

$$
\Delta_{\mathrm{train}} = \Delta_{\mathrm{FAM}} + \varepsilon,
\qquad \varepsilon \sim \mathcal{N}(0, \sigma^2).
$$

Per esempio, se il FAM ha appreso uno spostamento di $-3$ celle di feature per correggere una parallasse di $+3$, SSJ espone il modello a campionamenti vicini a $-3$, non al disallineamento originario di $+3$. In inferenza $\varepsilon$ viene rimosso e rimane il solo offset appreso.

In questa maniera viene introdotta un'incertezza **spaziale** e mutevole
(wobbling) alle feature IR, senza azzerarne direttamente i canali.
L'interpretazione perseguita è che il modello non possa affidarsi in modo
rigido a una corrispondenza RGB--IR perfetta durante l'addestramento. Il
vantaggio apparso nella singola run preliminare non si è però ripetuto nella
media RT-DETR a cinque seed; anche nelle esperienze Deformable DETR e DINO SSJ
non ha mostrato un beneficio affidabile.

### Risultato preliminare del Modello E

Nella singola campagna storica, il Modello E ha ottenuto la migliore mAP@50 di
fusione fra le cinque run confrontate:
- **VIS**: `0.2697`
- **IR**: `0.1818`
- **VIS_IR**: `0.4381`

Il Modello E supera il Modello A in fusione (`0.4381` vs `0.4286`) e migliora nettamente rispetto alla mAP IR-only del Modello A (`0.1818` vs `0.0054`).

## Rivalutazione finale a cinque seed

La stessa conclusione non si conserva nel protocollo finale. Con seed appaiati
`40–44`, checkpoint finale a dieci epoche, testa `person` pretrained e Modal
Dropout soltanto nel training, i risultati VIS+IR sono:

| Configurazione | Media | Mediana | Dev. std. | Min–max |
|---|---:|---:|---:|---:|
| FAM standard | 0.3780 | 0.3783 | 0.0440 | 0.3129–0.4335 |
| FAM + IR Dropout | 0.3871 | 0.3986 | 0.0469 | 0.3234–0.4452 |
| FAM + SSJ | 0.3749 | 0.3734 | 0.0185 | 0.3528–0.4030 |

SSJ − FAM ha un delta medio appaiato di `−0.0031`, cambia segno tra seed e ha
un IC 95% pari a `[−0.0488, +0.0427]`. SSJ mostra una deviazione standard
descrittiva più bassa, ma con cinque run non è possibile concludere che riduca
la varianza della popolazione; soprattutto, non produce un miglioramento medio
di accuratezza.

Anche IR Dropout non mostra un guadagno affidabile: la media è leggermente più
alta, ma il delta appaiato su FAM è `+0.0091`, con segno non consistente.

La configurazione principale della tesi è quindi **FAM standard, senza SSJ e
senza IR Dropout**. Lazy, Frozen e Spatial Dropout restano passaggi esplorativi
che hanno motivato controlli più puliti; non vanno presentati come una classifica
finale né ripetuti soltanto per inseguire i valori delle singole run.
