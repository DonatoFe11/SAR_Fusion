"""Descriptive recall--FPPI reports for the frozen historical RT-DETR contrast.

This module does not run models or choose deployment confidence thresholds.
The independent repetitions summarized here are training seeds, not frames.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np

from scripts.rtdetr_froc_metrics import recall_at_budgets


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SEEDS = (40, 41, 42, 43, 44)
EXPECTED_COUNTS = {
    "mterie": (708, 1770, 19),
    "carnation_0025_0026": (1313, 5238, 100),
    "fhl_0407_0408": (1035, 2022, 239),
}
CONFIGURATIONS = ("historical_additive", "historical_fam")
ACQUISITION_LABELS = {
    "mterie": "MtErie",
    "carnation_0025_0026": "Carnation 0025/0026",
    "fhl_0407_0408": "FHL 0407/0408",
}
CONFIGURATION_LABELS = {
    "historical_additive": "RT-DETR Additive",
    "historical_fam": "RT-DETR + FAM",
}
STEM = "rtdetr_recall_fppi"
PROVENANCE_FIELDS = (
    "source_sha256", "implementation_sha256", "inventory_sha256", "content_sha256",
    "torch_version", "transformers_version", "batch_size", "collection_threshold",
    "native_queries", "max_batches",
)


def _write_csv(path, rows, columns):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _summary(values):
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return {
        "n": int(array.size),
        "mean": float(array.mean()) if array.size else None,
        "sd": float(array.std(ddof=1)) if array.size > 1 else None,
    }


def _validated_records(records, protocol, publish):
    expected_keys = {
        (acquisition, configuration, seed)
        for acquisition in EXPECTED_COUNTS
        for configuration in CONFIGURATIONS
        for seed in EXPECTED_SEEDS
    }
    indexed = {}
    for record in records:
        key = (record["acquisition"], record["configuration"], int(record["seed"]))
        if key not in expected_keys:
            raise ValueError(f"Unexpected recall--FPPI job: {key}")
        if key in indexed:
            raise ValueError(f"Duplicate recall--FPPI job: {key}")
        counts = tuple(int(record[field]) for field in ("n_images", "n_gt", "n_empty"))
        if counts != EXPECTED_COUNTS[key[0]]:
            raise ValueError(f"Unexpected sample counts for {key}: {counts}")
        fppi = np.asarray(record["curve"]["fppi"], dtype=np.float64)
        recall = np.asarray(record["curve"]["recall"], dtype=np.float64)
        if fppi.ndim != 1 or fppi.size == 0 or recall.shape != fppi.shape:
            raise ValueError(f"Invalid recall--FPPI curve shape for {key}")
        if not np.all(np.isfinite(fppi)) or not np.all(np.isfinite(recall)):
            raise ValueError(f"Nonfinite recall--FPPI values for {key}")
        if np.any(fppi < 0) or np.any(np.diff(fppi) < 0):
            raise ValueError(f"FPPI must be nonnegative and nondecreasing for {key}")
        if np.any(recall < 0) or np.any(recall > 1) or np.any(np.diff(recall) < -1e-12):
            raise ValueError(f"Recall must be in [0,1] and nondecreasing for {key}")
        if not record.get("checkpoint_sha256"):
            raise ValueError(f"Missing checkpoint hash for {key}")
        indexed[key] = record
    complete = set(indexed) == expected_keys
    if publish:
        if not complete:
            raise ValueError(f"Publication requires all 30 unique jobs; found {len(indexed)}")
        if (
            set(protocol["acquisitions"]) != set(EXPECTED_COUNTS)
            or set(protocol["configurations"]) != set(CONFIGURATIONS)
            or tuple(protocol["seeds"]) != EXPECTED_SEEDS
            or float(protocol["iou_threshold"]) != 0.5
        ):
            raise ValueError("Publication protocol differs from the frozen historical contrast")
    return indexed, complete


def _sample(curve, budgets):
    """Sample empirical attainable recall, never extending collected support."""
    budgets = np.asarray(budgets, dtype=np.float64)
    values = np.asarray(recall_at_budgets(curve, budgets), dtype=np.float64)
    values = values.copy()
    values[budgets > float(np.max(curve["fppi"]))] = np.nan
    return values


def _plot(grid_rows, output_dir, complete):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"historical_additive": "#4477AA", "historical_fam": "#CC6677"}
    # Keep labels readable when the three panels occupy a 15.5 cm thesis column.
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 3.6), sharey=True)
    for axis, acquisition in zip(axes, EXPECTED_COUNTS):
        for configuration in CONFIGURATIONS:
            rows = [
                row for row in grid_rows
                if row["acquisition"] == acquisition
                and row["configuration"] == configuration
                and row["recall_mean"] is not None
            ]
            if not rows:
                continue
            x = np.asarray([row["fppi_budget"] for row in rows])
            mean = np.asarray([row["recall_mean"] for row in rows])
            sd = np.asarray([row["recall_sd"] or 0.0 for row in rows])
            axis.step(x, mean, where="post", color=colors[configuration],
                      label=CONFIGURATION_LABELS[configuration])
            axis.fill_between(x, np.maximum(0, mean - sd), np.minimum(1, mean + sd),
                              step="post", color=colors[configuration], alpha=0.17)
        axis.set_xscale("log")
        axis.set_xlim(0.01, 10.0)
        axis.set_ylim(0, 1)
        axis.set_title(ACQUISITION_LABELS[acquisition], fontsize=11)
        axis.set_xlabel("False positives per image")
        axis.grid(True, which="both", alpha=0.20)
    axes[0].set_ylabel("Recall (IoU ≥ 0.50)")
    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        for axis in axes[1:]:
            handles, labels = axis.get_legend_handles_labels()
            if handles:
                break
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
                   bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("Historical latest checkpoints · " +
                 ("five seeds, mean ± sample SD" if complete else "PARTIAL results"),
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.095, 1, 0.94))
    paths = []
    for extension in ("pdf", "png"):
        path = output_dir / f"{STEM}.{extension}"
        # Embed TrueType outlines for reliable inclusion through XeLaTeX.
        with matplotlib.rc_context({"pdf.fonttype": 42}):
            fig.savefig(path, dpi=220, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def _format_stat(mean, sd):
    if mean is None:
        return "non disponibile"
    if sd is None:
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {sd:.4f}"


def _markdown(report, summary_rows):
    lines = [
        "# RT-DETR storico: recall rispetto ai falsi positivi per immagine",
        "", "## Stato e obiettivo", "",
        f"Analisi completata: 30/30 unità. Protocollo `{report['protocol_id']}`.",
        "Si confrontano Additive e FAM storici in fusion VIS+IR sui checkpoint",
        "`latest` dei cinque seed 40–44, senza nuovo training. Non sono inclusi",
        "i checkpoint FAM current-code Stage B, RCRA o altre varianti.",
        "", "Questa è una caratterizzazione post-hoc dei checkpoint esistenti:",
        "non seleziona un nuovo modello, un checkpoint o una soglia operativa",
        "da applicare in deployment. Le acquisizioni sono già state valutate",
        "nelle campagne precedenti e non vengono presentate come nuovi holdout ciechi.",
        "", "## Protocollo", "",
        "| Acquisizione | Frame paired | Box VIS | Frame vuoti |",
        "|---|---:|---:|---:|",
    ]
    for acquisition, counts in EXPECTED_COUNTS.items():
        lines.append(f"| {ACQUISITION_LABELS[acquisition]} | {counts[0]} | {counts[1]} | {counts[2]} |")
    lines.extend([
        "", "Si riutilizzano gli inventari e la ground truth VIS congelati:",
        "per MtErie l'associazione paired storica (sorted zip); per Carnation",
        "0025/0026 e FHL 0407/0408 l'intersezione degli ID numerici comuni.",
        "Anche i frame senza persone rientrano nel denominatore FPPI.",
        "", "Un solo forward per immagine e checkpoint raccoglie tutte le predizioni",
        "native del modello (soglia di raccolta 0, nessuna nuova NMS). Le curve",
        "si ricavano offline variando la confidenza: una predizione è conservata",
        "se `score >= soglia`; score uguali entrano insieme. È incluso il punto",
        "che rifiuta tutte le predizioni. Il JSON conserva gli hash dei checkpoint",
        "e i percorsi delle cache delle predizioni.",
        "", "Il matching è uno-a-uno a IoU ≥ 0.50, in ordine decrescente di confidenza,",
        "con scelta della GT non ancora assegnata a IoU maggiore. Le predizioni",
        "duplicate sono falsi positivi. Questo criterio mantiene coerenti i",
        "prefissi della curva; differisce dal precedente error analysis che",
        "ricalcolava un matching greedy globale per IoU a ogni soglia. Piccole",
        "differenze rispetto ai vecchi punti non indicano un cambiamento del modello.",
        "", "`Recall = TP / box GT`; `FPPI = FP / numero totale di frame`.",
        "A ogni budget FPPI si prende la massima recall empiricamente raggiungibile",
        "senza superarlo, senza interpolazione o frazionamento di score uguali.",
        "Ogni seed viene campionato sulla stessa griglia logaritmica 0.01–10",
        "FPPI; solo dopo si calcolano media e deviazione standard campionaria.",
        "Non si media a una medesima confidenza, che può corrispondere a budget",
        "di falsi positivi differenti nei diversi checkpoint. Le curve si fermano",
        "all'endpoint minimo raccolto tra i seed e le due configurazioni del pannello:",
        "non si estrapolano punti oltre il supporto osservato.",
        "", "Le bande descrivono la variabilità tra cinque training seed, non sono",
        "intervalli di confidenza sui frame né prove di generalizzazione SAR.",
        "Le tre acquisizioni sono riportate separatamente; i frame non vengono",
        "trattati come repliche statistiche indipendenti.",
        "Gli eventuali budget oltre l'endpoint di un checkpoint sono marcati",
        "non disponibili: il CSV conserva esplicitamente il numero di seed",
        "disponibili per ogni statistica, senza imputazione.",
        "", "## Risultati ai budget prefissati", "",
        "I budget 0.1, 0.5 e 1 FP/immagine sono punti descrittivi prefissati.",
        "Ogni delta è appaiato sullo stesso seed; `vittorie` conta solo delta",
        "strettamente positivi. Un vantaggio a un budget non implica una dominanza",
        "su tutta la curva o su tutte le acquisizioni.",
        "", "| Acquisizione | FPPI ≤ | Recall Additive (media ± SD) | Recall FAM (media ± SD) | Δ FAM−Additive (media ± SD) | Vittorie |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in summary_rows:
        lines.append(
            f"| {ACQUISITION_LABELS[row['acquisition']]} | {row['fppi_budget']:g} | "
            f"{_format_stat(row['additive_mean'], row['additive_sd'])} | "
            f"{_format_stat(row['fam_mean'], row['fam_sd'])} | "
            f"{_format_stat(row['paired_delta_mean'], row['paired_delta_sd'])} | "
            f"{row['positive_seed_count']}/{row['paired_seed_count']} |"
        )
    lines.extend([
        "", "## Endpoint e tracciabilità", "",
        "| Acquisizione | Endpoint minimo FPPI (10 checkpoint) |",
        "|---|---:|",
    ])
    for acquisition, endpoint in report["common_endpoint_fppi"].items():
        lines.append(f"| {ACQUISITION_LABELS[acquisition]} | {endpoint:.4f} |")
    lines.extend([
        "", "I CSV conservano i risultati per seed ai budget, i delta appaiati,",
        "le statistiche riassuntive e la griglia comune aggregata. Il JSON",
        "conserva protocollo, checkpoint, percorsi delle cache ed endpoint,",
        "hash degli inventari e dell'implementazione e versioni software.",
        "Il CSV per seed riporta anche FPPI effettivamente raggiunta, soglia,",
        "TP e FP del punto empirico: la soglia descrive quel checkpoint su",
        "quell'acquisizione e non è una soglia ottimizzata per il deployment.",
        "", f"- [Curve PDF](Search_and_Rescue/images/{STEM}.pdf)",
        f"- [Curve PNG](Search_and_Rescue/images/{STEM}.png)",
        f"- [Riepilogo CSV](Search_and_Rescue/results/{STEM}_summary.csv)",
        f"- [Budget per seed](Search_and_Rescue/results/{STEM}_budgets.csv)",
        f"- [Delta appaiati](Search_and_Rescue/results/{STEM}_paired.csv)",
        f"- [Griglia aggregata](Search_and_Rescue/results/{STEM}_grid.csv)",
        f"- [Metadati JSON](Search_and_Rescue/results/{STEM}.json)",
        "", "## Riproduzione", "",
        "Dalla root del repository, nell'ambiente `sarfusion`:",
        "", "```bash", "python scripts/run_rtdetr_recall_fppi.py --device cuda", "```", "",
        "Il comando riutilizza le cache compatibili già completate e riparte",
        "in sicurezza dalle unità mancanti; cache con provenienza incompatibile",
        "vengono rifiutate. Non è necessario un nuovo training.",
        "Per rigenerare soltanto i riepiloghi da tutte le cache già presenti:",
        "", "```bash", "python scripts/run_rtdetr_recall_fppi.py --summarize-only", "```", "",
        "Configurazione: [`rtdetr_recall_fppi.yaml`](../parameters/RTDETR/rtdetr_recall_fppi.yaml).",
        "Le predizioni dense sono conservate in `out/rtdetr_recall_fppi/predictions/`;",
        "le curve esatte e gli altri risultati sono in `out/rtdetr_recall_fppi/`.",
        "", "## Integrazione nella tesi", "",
        "La sottosezione `sec:recall-fppi` del capitolo sperimentale include la figura",
        "PDF e tutti i nove confronti ai budget prefissati. Il testo chiarisce matching,",
        "media sui seed, popolazioni separate, carattere post-hoc e l'eccezione di",
        "Carnation seed 43 a 0.1 FPPI. Abstract, discussione e limiti riprendono il",
        "risultato a parità di budget; l'appendice e il README degli artefatti indicano",
        "le fonti numeriche e il comando di rigenerazione. Non vengono modificate le",
        "decisioni sulle architetture né selezionate soglie operative sui set di test.", "",
    ])
    return "\n".join(lines)


def write_report(records, protocol, output_dir, publish=False):
    """Write numerical summaries and figures; publish only a complete campaign.

    ``records`` must contain one curve for every acquired checkpoint/acquisition
    job. A partial report is allowed for diagnostics but cannot be published to
    thesis assets. The returned JSON-compatible dictionary includes output paths.
    """
    indexed, complete = _validated_records(records, protocol, publish)
    budgets = np.asarray(protocol["fppi_budgets"], dtype=np.float64)
    grid_spec = protocol["fppi_grid"]
    if budgets.ndim != 1 or not budgets.size or not np.all(np.isfinite(budgets)) or np.any(budgets <= 0):
        raise ValueError("FPPI budgets must be finite and positive")
    if not (0 < float(grid_spec["min"]) < float(grid_spec["max"])) or int(grid_spec["points"]) < 2:
        raise ValueError("Invalid FPPI grid")
    if publish and (budgets.tolist() != [0.1, 0.5, 1.0] or
                    float(grid_spec["min"]) != 0.01 or float(grid_spec["max"]) != 10.0 or
                    int(grid_spec["points"]) != 301):
        raise ValueError("Publication requires the predefined FPPI budgets and grid")
    grid = np.geomspace(float(grid_spec["min"]), float(grid_spec["max"]), int(grid_spec["points"]))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    budget_rows, paired_rows, summary_rows, grid_rows, jobs = [], [], [], [], []
    sampled = {}
    common_endpoints = {}
    for key, record in sorted(indexed.items()):
        acquisition, configuration, seed = key
        sampled[key] = _sample(record["curve"], budgets)
        endpoint = float(np.max(record["curve"]["fppi"]))
        common_endpoints[acquisition] = min(common_endpoints.get(acquisition, float("inf")), endpoint)
        job = {
            "acquisition": acquisition, "configuration": configuration, "seed": seed,
            "n_images": int(record["n_images"]), "n_gt": int(record["n_gt"]),
            "n_empty": int(record["n_empty"]), "checkpoint_sha256": record["checkpoint_sha256"],
            "cache_path": str(record.get("cache_path", "")), "max_fppi": endpoint,
            "max_recall": float(np.max(record["curve"]["recall"])),
            "curve_points": int(len(record["curve"]["fppi"])),
        }
        job.update({field: record[field] for field in PROVENANCE_FIELDS if field in record})
        jobs.append(job)
        for budget, recall in zip(budgets, sampled[key]):
            supported = bool(np.isfinite(recall))
            point_index = int(np.searchsorted(record["curve"]["fppi"], budget, side="right") - 1)
            threshold = float(record["curve"]["threshold"][point_index])
            reject_all = bool(np.isposinf(threshold))
            budget_rows.append({
                "acquisition": acquisition, "configuration": configuration, "seed": seed,
                "fppi_budget": float(budget), "recall": float(recall) if supported else None,
                "within_collected_support": supported,
                "attained_fppi": float(record["curve"]["fppi"][point_index]) if supported else None,
                "threshold": threshold if supported and not reject_all else None,
                "reject_all": reject_all if supported else None,
                "tp": int(record["curve"]["tp"][point_index]) if supported else None,
                "fp": int(record["curve"]["fp"][point_index]) if supported else None,
                "checkpoint_sha256": record["checkpoint_sha256"],
            })
    for acquisition in EXPECTED_COUNTS:
        for configuration in CONFIGURATIONS:
            available = [indexed[(acquisition, configuration, seed)] for seed in EXPECTED_SEEDS
                         if (acquisition, configuration, seed) in indexed]
            if not available:
                continue
            sampled_grid = np.stack([_sample(record["curve"], grid) for record in available])
            for column, budget in enumerate(grid):
                # Mask the whole panel at the smallest endpoint; never change its seed set.
                if budget > common_endpoints[acquisition]:
                    continue
                stats = _summary(sampled_grid[:, column])
                grid_rows.append({
                    "acquisition": acquisition, "configuration": configuration,
                    "fppi_budget": float(budget), "n_seeds": stats["n"],
                    "recall_mean": stats["mean"], "recall_sd": stats["sd"],
                })
        for budget_index, budget in enumerate(budgets):
            by_configuration = {}
            for configuration in CONFIGURATIONS:
                by_configuration[configuration] = {
                    seed: sampled[(acquisition, configuration, seed)][budget_index]
                    for seed in EXPECTED_SEEDS
                    if (acquisition, configuration, seed) in sampled
                    and np.isfinite(sampled[(acquisition, configuration, seed)][budget_index])
                }
            additive = by_configuration["historical_additive"]
            fam = by_configuration["historical_fam"]
            deltas = []
            for seed in sorted(set(additive) & set(fam)):
                delta = float(fam[seed] - additive[seed])
                deltas.append(delta)
                paired_rows.append({
                    "acquisition": acquisition, "seed": seed, "fppi_budget": float(budget),
                    "additive_recall": float(additive[seed]), "fam_recall": float(fam[seed]),
                    "paired_delta": delta,
                })
            stats_additive, stats_fam, stats_delta = map(_summary, [list(additive.values()), list(fam.values()), deltas])
            summary_rows.append({
                "acquisition": acquisition, "fppi_budget": float(budget),
                "additive_n": stats_additive["n"], "additive_mean": stats_additive["mean"],
                "additive_sd": stats_additive["sd"], "fam_n": stats_fam["n"],
                "fam_mean": stats_fam["mean"], "fam_sd": stats_fam["sd"],
                "paired_seed_count": stats_delta["n"], "paired_delta_mean": stats_delta["mean"],
                "paired_delta_sd": stats_delta["sd"],
                "positive_seed_count": sum(delta > 0 for delta in deltas),
                "zero_seed_count": sum(delta == 0 for delta in deltas),
                "negative_seed_count": sum(delta < 0 for delta in deltas),
            })
    csv_payloads = {
        "budgets": (budget_rows, ["acquisition", "configuration", "seed", "fppi_budget", "recall", "within_collected_support", "attained_fppi", "threshold", "reject_all", "tp", "fp", "checkpoint_sha256"]),
        "paired": (paired_rows, ["acquisition", "seed", "fppi_budget", "additive_recall", "fam_recall", "paired_delta"]),
        "summary": (summary_rows, ["acquisition", "fppi_budget", "additive_n", "additive_mean", "additive_sd", "fam_n", "fam_mean", "fam_sd", "paired_seed_count", "paired_delta_mean", "paired_delta_sd", "positive_seed_count", "zero_seed_count", "negative_seed_count"]),
        "grid": (grid_rows, ["acquisition", "configuration", "fppi_budget", "n_seeds", "recall_mean", "recall_sd"]),
    }
    csv_paths = []
    for suffix, (rows, columns) in csv_payloads.items():
        path = output_dir / f"{STEM}_{suffix}.csv"
        _write_csv(path, rows, columns)
        csv_paths.append(path)
    figure_paths = _plot(grid_rows, output_dir, complete)
    report = {
        "protocol_id": protocol["id"], "protocol": protocol,
        "complete": complete, "n_jobs": len(indexed), "expected_jobs": 30,
        "common_endpoint_fppi": common_endpoints,
        "summary": summary_rows, "jobs": jobs,
        "interpretation": {
            "posthoc_characterization": True, "model_selection_allowed": False,
            "deployment_threshold_selection_allowed": False,
            "dispersion": "sample SD across training seeds, not frame confidence interval",
            "budget_sampling": "maximum attainable recall at or below FPPI budget; no interpolation",
            "curve_support": "each panel stops at the minimum collected endpoint across its checkpoints",
        },
        "files": [str(path) for path in csv_paths + figure_paths],
    }
    json_path = output_dir / f"{STEM}.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    if publish:
        results_dir = REPO_ROOT / "notes" / "Search_and_Rescue" / "results"
        images_dir = REPO_ROOT / "notes" / "Search_and_Rescue" / "images"
        results_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)
        for path in csv_paths + [json_path]:
            destination = results_dir / path.name
            if path.resolve() != destination.resolve():
                shutil.copy2(path, destination)
        for path in figure_paths:
            destination = images_dir / path.name
            if path.resolve() != destination.resolve():
                shutil.copy2(path, destination)
        (REPO_ROOT / "notes" / f"{STEM}.md").write_text(_markdown(report, summary_rows), encoding="utf-8")
    return report
