# Validazione del Feature Alignment Module (FAM)

## Obiettivo

L'obiettivo della validazione era stabilire se il **Feature Alignment Module** (FAM) apprenda effettivamente una correzione geometrica tra le feature RGB e IR. I risultati in mAP, pur utili, costituiscono infatti un'evidenza solo indiretta: un miglioramento della metrica non dimostra da solo che il meccanismo responsabile sia l'allineamento spaziale.

L'analisi è stata svolta sui modelli RT-DETR già addestrati. Si è considerato sia il modello con solo FAM sia quello addestrato con FAM e Spatial Jitter (SSJ), così da distinguere il comportamento del modulo da quello indotto dal regime di training.

## Metodo di verifica

È stato sviluppato lo script `fam_alignment_check.py`, una diagnostica riutilizzabile per modelli fusion che contengono la classe `FeatureAlignmentModule`.

### Caricamento coerente con gli esperimenti

Per RT-DETR, lo script ricostruisce prima l'architettura tramite `build_model()` usando i parametri della run selezionata nel file YAML; successivamente carica nel modello vuoto i pesi appresi dal checkpoint `.safetensors`.

Il campione viene caricato attraverso la stessa pipeline dati usata per training/evaluation. La pipeline Hugging Face applica il preprocessing definito dal relativo `AutoProcessor`, che può includere resize, riscalamento e normalizzazione tramite media e deviazione standard.

### Cattura delle feature e delle variabili interne del FAM

Per ciascun livello della piramide, durante l'inferenza lo script registra due forward hook:

- un hook sul `FeatureAlignmentModule` completo, che salva le feature RGB e IR in ingresso e la feature IR in uscita dal FAM (`FAM(IR)`);
- un hook su `offset_conv`, layer interno del FAM, che salva offset e mask usati dalla deformable convolution.

Il primo hook permette quindi di confrontare `RGB`, `IR pre-FAM` e `IR post-FAM`; il secondo permette di osservare *come* il FAM effettua la correzione. L'output di `offset_conv` contiene 18 canali di offset e 9 canali di mask: i primi corrispondono alle coordinate `(dx, dy)` dei nove punti di un kernel deformabile 3×3, mentre le mask, applicate dopo sigmoid, ne modulano il peso.

Gli hook sono registrati cercando il nome della classe `FeatureAlignmentModule`, anziché un percorso fisso nell'albero dei moduli. A ogni FAM trovato viene assegnato un indice nell'ordine di registrazione (`level 0`, `level 1`, ...). Nell'implementazione RT-DETR questo ordine corrisponde ai livelli a stride 8, 16 e 32.

Lo script riusa le funzioni del progetto per ricostruire la configurazione di grid search, costruire il modello e caricare i dati. In questo modo checkpoint, configurazione e preprocessing corrispondono a quelli degli esperimenti originali.

### Confronto visivo delle feature

Le feature interne hanno forma `(C, H, W)`: in ogni posizione spaziale contengono un vettore di `C` valori, spesso centinaia di canali, e non sono quindi visualizzabili direttamente come immagini RGB. La PCA riduce questi vettori a tre componenti principali, usate come pseudo-canali R, G e B, preservando le principali variazioni nelle feature.

Le tre feature RGB, IR e FAM(IR) vengono visualizzate con PCA a tre componenti, insieme agli overlay RGB+IR prima e dopo il FAM. Per rendere i colori confrontabili, la PCA viene fittata su una **base condivisa** ottenuta combinando le tre mappe dello stesso livello (ad esempio per il livello 0 / P3: RGB P3 + IR P3 + FAM(IR) P3 → una prima PCA condivisa). Prima del fit, ogni feature viene standardizzata tramite z-score globale. Questa scelta evita che differenze puramente di scala alterino la visualizzazione e garantisce che colori simili nei pannelli corrispondano alle stesse direzioni nello spazio delle feature, perchè ad esempio FAM(IR) esce da una deformable convolution senza una ReLU finale, quindi può avere valori negativi e una distribuzione diversa dalle feature RGB/IR della backbone.

Le figure risultanti non mostrano immagini RGB o IR originali: mostrano pseudo-colori di feature astratte e a risoluzione ridotta. Gli overlay sono alpha blend al 50% tra le due immagini PCA e costituiscono quindi un aiuto qualitativo per osservare strutture o bordi potenzialmente disallineati, non una misura diretta dell'allineamento né una sovrapposizione fotografica.

### Misura quantitativa degli offset

L'ampiezza degli offset viene riportata in pixel dell'immagine originale, non soltanto in pixel della feature map. La conversione usa la stride effettiva di ciascun livello (8, 16 e 32): è necessaria perché lo stesso offset nella feature map rappresenta uno spostamento fisico diverso a risoluzioni diverse.

Su più campioni, lo script aggrega tutti i valori assoluti degli offset e calcola media, mediana, 90° percentile e massimo. Visualizza inoltre un quiver plot dello spostamento medio, pesato con la mask, sui nove punti campionati dalla deformable convolution.

Il quiver plot è un grafico a frecce: ogni freccia sintetizza, in una posizione della feature map, direzione e ampiezza della correzione locale. Non mostra tutti i nove offset individuali; visualizza la loro media pesata dalle rispettive mask. Le frecce sono espresse nelle coordinate della feature map e possono apparire corte anche quando lo spostamento convertito nei pixel dell'immagine originale è significativo.

### Come leggere le statistiche stampate dallo script

Per ogni feature `rgb`, `ir` e `fam(ir)`, lo script stampa statistiche calcolate sul tensore `(C, H, W)`:

- `mean`: media di tutte le attivazioni, cioè di tutti i canali e di tutte le posizioni spaziali;
- `std_globale`: deviazione standard calcolata su tutti i valori insieme. Include sia differenze tra canali sia differenze tra posizioni e serve soprattutto a verificare che l'output del FAM non abbia una scala di attivazione anomala, collassata o instabile;
- `std_spaziale`: per ogni canale calcola la deviazione standard sulle sole posizioni `(H, W)`, poi media il risultato sui canali. Misura quindi quanta variazione spaziale contiene mediamente un canale della feature;
- `min` e `max`: estremi delle attivazioni osservate.

`std_globale` e `std_spaziale` rispondono a domande diverse. Ad esempio, due canali che siano costanti nello spazio ma abbiano valori costanti diversi possono avere `std_spaziale = 0` e tuttavia `std_globale > 0`. Per discutere lo smoothing è quindi più informativa `std_spaziale`: una sua riduzione da `IR` a `FAM(IR)` indica che l'output varia meno tra posizioni della feature map. Il confronto va comunque interpretato con cautela, perché l'output della deformable convolution può avere valori sia positivi sia negativi, mentre le feature di backbone sono spesso non negative; queste statistiche descrivono distribuzione e scala, non accuratezza semantica delle feature.

Per gli offset, `mean_abs` e `max_abs` sono rispettivamente media e massimo dei valori assoluti delle componenti `dx` e `dy`, su tutti i nove punti del kernel e su tutte le celle della feature map. Non sono la lunghezza media dei vettori 2D. Il valore viene poi moltiplicato per la stride del livello per ottenere pixel dell'immagine di input.

L'indicatore di uniformità è definito come:

```text
uniformity_ratio = offset_spatial_std / offset_magnitude
```

`offset_spatial_std` misura quanto ciascuno dei 18 campi di offset cambia passando da una cella della feature map a un'altra; `offset_magnitude` è la sua ampiezza media assoluta. Un rapporto vicino a zero indica un campo quasi costante tra celle, mentre un rapporto vicino a uno indica variazioni spaziali dell'ordine dell'ampiezza media. In una singola cella non esiste un solo offset: il kernel deformabile 3×3 usa nove vettori `(dx, dy)`. Il quiver plot ne mostra una sola freccia perché calcola una sintesi pesata dalle mask.

## Protocollo sperimentale

Il confronto principale ha usato dieci campioni distribuiti lungo il test set MtErie e la configurazione comune `parameters/RTDETR/fusion_rtdetr.yaml`.

| Modello | Checkpoint | Condizione di training |
|---|---|---|
| B | `fusion_rtdetr_fam_correct.safetensors` | FAM senza Spatial Jitter |
| E | `spatial_jitter.safetensors` | FAM con Spatial Jitter |

Gli indici test usati per entrambi i modelli sono: `35, 106, 177, 247, 318, 389, 460, 531, 601, 672`.

L'analisi è stata poi estesa a due sessioni di volo differenti presenti nel train split: FHL 210924 e Baker 220109. L'uso di `--split train` seleziona solo il sottoinsieme da analizzare: il modello resta in `eval()` e nessun peso viene aggiornato. Per ciascuna sessione sono stati scelti dieci campioni, identici per B ed E:

| Sessione | Indici dataset | Modello B | Modello E |
|---|---|---|---|
| FHL 210924 | 90, 269, 448, 627, 806, 990, 1132, 1367, 1603, 1745 | `fusion_rtdetr_fam_correct.safetensors` | `spatial_jitter.safetensors` |
| Baker 220109 | 1948, 2166, 2384, 2602, 2929, 3038, 3256, 3474, 3692, 3910 | `fusion_rtdetr_fam_correct.safetensors` | `spatial_jitter.safetensors` |

Questo controllo non misura la generalizzazione, poiché le due sessioni fanno parte del train split. Serve invece a verificare se il campo di offset cambia al variare della sessione e delle condizioni geometriche di acquisizione.

Passando, ad esempio, `--sample-idx 0 1 2`, lo script analizza individualmente i tre campioni e salva una figura per ogni combinazione campione/livello; con tre FAM vengono quindi prodotte nove figure. Al termine, non media le immagini né le feature map: raccoglie tutti i valori assoluti degli offset e ne calcola statistiche aggregate separate per livello.

## Risultati quantitativi

Offset del FAM in pixel dell'immagine originale, aggregati sui dieci campioni test.

| Livello | Stride | Modello B: mean / median / p90 / max | Modello E: mean / median / p90 / max | max/mean B | max/mean E |
|---|---:|---|---|---:|---:|
| 0 | 8 | 2.74 / 2.50 / 5.17 / 23.07 px | 4.01 / 3.08 / 9.13 / 30.51 px | 8.4× | 7.6× |
| 1 | 16 | 2.66 / 2.02 / 5.69 / 59.23 px | 4.18 / 3.23 / 9.27 / 56.57 px | 22.3× | 13.5× |
| 2 | 32 | 3.95 / 3.10 / 7.87 / 94.08 px | 4.61 / 3.79 / 8.47 / 204.54 px | 23.8× | 44.4× |

Il FAM produce quindi offset non trascurabili: l'ampiezza tipica è dell'ordine di pochi pixel dell'immagine. Le code estreme sono molto più alte, soprattutto a P5 nel Modello E, ma il p90 resta sotto 10 px in entrambi i modelli e a tutti i livelli.

### Uniformità spaziale della correzione

Oltre all'ampiezza, è stato misurato quanto l'offset vari tra celle di output diverse rispetto alla propria ampiezza media (`uniformity_ratio`: deviazione standard spaziale dell'offset, normalizzata sulla sua ampiezza media). Un valore vicino a 0 indica uno spostamento pressoché costante su tutta la mappa; un valore vicino a 1 indica una correzione che varia tanto quanto la propria ampiezza, quindi più dipendente dalla posizione.

| Livello | Stride | Modello B | Modello E |
|---|---:|---:|---:|
| 0 | 8 | 0.606 | 0.456 |
| 1 | 16 | 0.961 | 0.608 |
| 2 | 32 | 1.156 | 1.168 |

Sul test set di MtErie il Modello E ha un rapporto più basso a P3 e P4, mentre a P5 il valore medio è comparabile con B e varia molto tra singoli campioni. Il controllo cross-sessione riportato sotto conferma lo stesso quadro; la SSJ non va quindi descritta come una regolarizzazione verso maggiore uniformità a ogni livello e in ogni condizione.

### Controllo cross-sessione

Le quattro run cross-sessione confermano che gli offset mantengono lo stesso ordine di grandezza, ma non sono identici tra FHL e Baker.

| Modello / sessione | Livello 0: mean / median / p90 / max | Livello 1: mean / median / p90 / max | Livello 2: mean / median / p90 / max |
|---|---|---|---|
| B — FHL | 2.57 / 2.24 / 4.89 / 22.65 px | 2.63 / 1.97 / 5.66 / 58.96 px | 3.81 / 2.92 / 7.64 / 95.19 px |
| B — Baker | 2.33 / 1.88 / 4.68 / 22.31 px | 2.67 / 2.01 / 5.78 / 60.33 px | 4.03 / 3.17 / 7.95 / 94.06 px |
| E — FHL | 3.75 / 2.89 / 8.45 / 29.79 px | 4.11 / 3.09 / 9.26 / 53.37 px | 4.60 / 3.78 / 8.56 / 204.78 px |
| E — Baker | 3.30 / 2.65 / 6.97 / 28.55 px | 4.02 / 2.90 / 9.27 / 53.75 px | 4.68 / 3.68 / 9.02 / 216.27 px |

Gli offset tipici cambiano moderatamente tra sessioni: nel Modello B la differenza nelle medie va da circa 2% a 11%, nel Modello E da circa 2% a 12%. Le code estreme, soprattutto a livello 2 del Modello E, non sono rappresentative del comportamento tipico: i p90 restano nell'intervallo 8.6-9.0 px, mentre i massimi isolati raggiungono 205-216 px.

La media dei `uniformity_ratio` dei dieci campioni per sessione è:

| Modello / sessione | Livello 0 | Livello 1 | Livello 2 |
|---|---:|---:|---:|
| B — FHL | 0.659 | 0.968 | 1.207 |
| B — Baker | 0.729 | 0.978 | 1.205 |
| E — FHL | 0.491 | 0.625 | 1.180 |
| E — Baker | 0.549 | 0.671 | 1.211 |

Nei livelli 0 e 1 la SSJ porta a campi più uniformi nello spazio anche in entrambe le sessioni. Al livello 2, invece, il rapporto medio è pressoché uguale tra B ed E e varia sensibilmente da campione a campione.

## Osservazioni e interpretazione

### Il FAM apprende una correzione geometrica reale

Il campo di offset non è nullo né casuale e la sua ampiezza è compatibile con un disallineamento RGB–IR dovuto alla geometria e alla calibrazione delle due camere. Questo fornisce un'evidenza diretta del funzionamento interno del modulo, complementare ai risultati di detection.

### Effetto dello Spatial Jitter

Il Modello E presenta offset medi e mediani più alti del Modello B a tutti i livelli, sia sul test set di MtErie sia nel controllo FHL/Baker. Lo Spatial Jitter non regolarizza quindi il FAM verso correzioni più piccole.

Sul test set di MtErie, E riduce il rapporto `max/mean` a P3 e P4 e presenta `uniformity_ratio` più basso di B negli stessi livelli. A P5, invece, E presenta occasionalmente massimi molto elevati e non riduce né `max/mean` né l'indicatore di uniformità rispetto a B. Il controllo cross-sessione conferma questo comportamento dipendente dal livello.

La caratterizzazione più appropriata è:

> Lo Spatial Jitter aumenta l'ampiezza tipica degli offset. Nei livelli più fini (P3 e P4) rende il campo mediamente più uniforme nello spazio; l'effetto non risulta invece stabile a P5.

### Natura della correzione appresa

Il FAM predice gli offset a ogni forward a partire dalle feature della coppia RGB-IR. I dieci campioni distribuiti del test set MtErie, provenienti dalla stessa sessione e da condizioni di acquisizione comparabili, producono offset dello stesso ordine di grandezza; questo controllo, limitato a una sola sessione, non dimostra che il campo sia fisso.

Il controllo FHL/Baker mostra invece variazioni moderate e ripetibili nelle ampiezze tipiche degli offset. L'evidenza è quindi compatibile con una correzione **dipendente dall'input**, la cui ampiezza riflette la geometria della singola acquisizione: quota, orientamento della piattaforma, prospettiva e altre condizioni che modificano la relazione RGB-IR. Una componente legata alla geometria fissa delle camere può comunque contribuire alla stabilità osservata all'interno della stessa sessione.

I risultati non permettono di attribuire la dipendenza osservata al solo contenuto semantico della scena. Per dimostrare direttamente quanto cambino i campi cella per cella tra immagini servirebbe confrontare i tensori di offset, ad esempio con MAE o correlazione tra campioni.

### Aspetto delle feature allineate

Le feature FAM(IR) appaiono generalmente più lisce e a minore frequenza rispetto alle feature IR iniziali. Le statistiche delle attivazioni, insieme alla standardizzazione applicata prima della PCA, non suggeriscono che il fenomeno sia un semplice artefatto di scala. L'interpretazione più plausibile è uno smoothing introdotto dal campionamento bilineare e dalla combinazione modulata dei nove punti della deformable convolution.

Le bande nere verticali osservabili nella PCA dell'IR al primo livello sono invece attribuibili al letterboxing di `adapt_ir2rgb`, usato per compensare il diverso aspect ratio tra IR e RGB; non sono un effetto del FAM.

## Conclusione

La validazione supporta che il FAM sia operativo e che predice una correzione geometrica RGB-IR di entità plausibile. Gli offset sono molto stabili tra immagini della stessa sessione, ma variano moderatamente fra sessioni di acquisizione differenti. Il comportamento è compatibile con un allineamento condizionato dalla geometria della singola acquisizione.

Lo Spatial Jitter modifica in modo misurabile questo comportamento: aumenta l'ampiezza tipica degli offset e, nei livelli P3/P4, li rende più uniformi nello spazio. L'effetto di uniformità non è invece conclusivo nel livello P5.

## Limiti e attività residue

- La verifica attuale riguarda feature e offset interni. Un confronto qualitativo sui bounding box predetti, sovrapposti alle immagini RGB e IR, è ora tecnicamente possibile ma non è ancora stato eseguito.
- Le conclusioni sulla dipendenza dall'input si basano su statistiche aggregate. Un confronto diretto dei tensori di offset tra campioni (ad esempio MAE e correlazione per livello) renderebbe quantitativa la variazione del campo cella per cella.
