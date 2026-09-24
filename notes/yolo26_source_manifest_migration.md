# Manifest YOLO26 dopo la rinomina della tesi

Dopo aver completato la tesi, ho rinominato `notes/Search_and_Rescue` in
`notes/Thesis`. Ho aggiornato i percorsi nei sorgenti; di conseguenza sono
cambiati anche alcuni hash. I risultati degli esperimenti sono gli stessi.

## Manifest degli esperimenti conclusi

Questi file restano identici alle versioni registrate nei risultati della tesi:

- `parameters/YOLO26/stage_a_source_manifest.json`: esperimento pilota;
- `parameters/YOLO26/stage_a_repair_v1_source_manifest.json`: esperimento repair v1.

Gli hash dei manifest sono verificati contro i campi `source_manifest_sha256`
dei rispettivi risultati in `notes/Thesis/results/`. I percorsi e gli hash
interni descrivono il checkout storico, recuperabile dai commit indicati nei
risultati (`git_commit` per il pilota e `commit` per il repair). Non ci si aspetta
che quei sorgenti coincidano con il codice successivo al refactoring.

## Revisioni operative

Le quattro configurazioni Stage A pilota/repair puntano ora a:

- `parameters/YOLO26/stage_a_source_manifest_thesis_paths_v1.json`;
- `parameters/YOLO26/stage_a_repair_v1_source_manifest_thesis_paths_v1.json`.

Ogni revisione registra percorso e hash del manifest storico di origine e
include quel file fra quelli verificati. La revisione repair registra inoltre
percorso e hash della revisione operativa del pilota.

Ho rigenerato queste revisioni anche dopo aver aggiornato le note
`yolo26_fam_stage_a.md` e `yolo26_stage_a_repair_v1.md`, incluse nei manifest.
Descrivono il checkout aggiornato; non corrispondono a nuovi esperimenti. Gli audit di controllo storici conservano gli hash originali;
i controlli che richiedono un'identità esatta dei sorgenti continuano a rifiutare
il loro riutilizzo per autorizzare un'esecuzione con una revisione diversa.

Per rigenerare le revisioni operative, eseguire dalla radice del repository:

```bash
python scripts/freeze_yolo26_source_manifest.py
python scripts/freeze_yolo26_stage_a_repair_manifest.py
```

L'ordine è necessario perché la seconda revisione include l'hash della prima.
I due generatori scrivono esclusivamente i file con suffisso `thesis_paths_v1`;
i manifest degli esperimenti conclusi vengono soltanto letti.
