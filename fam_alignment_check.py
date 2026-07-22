"""
fam_alignment_check.py
-----------------------
Verifica dell'allineamento del FAM (Feature Alignment Module) tramite
visualizzazione PCA delle feature map, stile DINOv2/DINOv3.

Supporta due famiglie di modelli:
  --model-type hf   (default) RT-DETR / Deformable DETR / DINO fusion,
                     caricati da .safetensors + config yaml grid-search HF.
  --model-type yolo  YOLOv10FusionFAM, caricato da checkpoint .pt ultralytics
                     (l'istanza di modello e' pickled per intero nel
                     checkpoint, quindi non serve build_model()+load_state_dict()).

Per un modello fusion con use_fam=True, cattura via forward hook le feature
RGB/IR immediatamente prima e dopo ciascun FeatureAlignmentModule della
backbone, e le proietta a colori RGB con PCA (3 componenti principali ->
canali R,G,B), con isolamento opzionale del foreground tramite la prima
componente (lo stesso trucco usato nelle visualizzazioni DINOv2/v3).

Per ogni livello della piramide di feature produce una figura con:
  (a) PCA(feature RGB)
  (b) PCA(feature IR)              -- pre-FAM
  (c) PCA(feature FAM(IR))         -- post-FAM, l'input reale del decoder/neck
  (d) overlay RGB+IR               -- blend alpha, mostra il disallineamento
  (e) overlay RGB+FAM(IR)          -- blend alpha, post-allineamento
  (f) campo di offset del FAM      -- bonus diagnostico, quiver plot

L'hook e' registrato per NOME DI CLASSE (FeatureAlignmentModule), non per
path fisso: e' quindi automaticamente compatibile con qualunque architettura
che riusi quella classe (rtdetr_fusion.py, deformable_detr_fusion.py,
yolo_fusion_fam.py), senza modifiche.

Uso (HF, comportamento originale invariato):
    python fam_alignment_check.py \
        --config /path/to/fusion_rtdetr.yaml \
        --checkpoint /path/to/tracking_dir/<run>/best/model.safetensors \
        --dataset-root /path/assoluto/a/dataset/WiSARD \
        --sample-idx 0 \
        --split val \
        --out-dir ./fam_alignment_vis

Uso (YOLO):
    python fam_alignment_check.py \
        --model-type yolo \
        --config parameters/YOLO/30.yolov10-fam.yaml \
        --run-index 2 \
        --checkpoint SarYOLO/YOLOv10-FAM-Grid2/weights/best.pt \
        --data-yaml wisards_vis_ir.yaml \
        --sample-idx 0 1 2 \
        --split val \
        --out-dir ./fam_alignment_vis_yolo

Dipendenze extra rispetto all'ambiente sarfusion: scikit-learn, matplotlib
    pip install scikit-learn matplotlib --break-system-packages
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from safetensors import safe_open
from sklearn.decomposition import PCA

from sarfusion.models import build_model
from sarfusion.data import get_dataloaders
from sarfusion.utils.utils import load_yaml
from sarfusion.utils.grid import make_grid

# --- YOLO-specific imports (solo per --model-type yolo) ---
from ultralytics.data.utils import check_det_dataset
from ultralytics.cfg import cfg2dict, IterableSimpleNamespace
from ultralytics.utils import colorstr

from sarfusion.data.wisard import WiSARDYOLODataset
from sarfusion.experiment.yolo import WISARD_DEFAULT_CFG


# ---------------------------------------------------------------------------
# 0. Ricostruzione della config di un run dal formato "grid search"
# ---------------------------------------------------------------------------

def load_run_config(config_path, run_index=0):
    """
    Riusa sarfusion.utils.grid.make_grid (la stessa funzione usata da
    Experimenter.calculate_runs) per trasformare la sezione "parameters"
    del yaml, dove ogni valore terminale e' wrappato in una lista (asse di
    grid search), in una config piatta con valori scalari/dict/list "veri".

    Formato HF: "parameters" annidato in {model, dataset, dataloader, ...}.
    """
    raw = load_yaml(config_path)
    parameters = raw.get("parameters", raw)  # fallback se e' gia' in formato flat
    grid = make_grid(parameters)
    if run_index >= len(grid):
        raise ValueError(
            f"Il config produce {len(grid)} combinazioni di grid search, "
            f"run_index={run_index} non e' valido."
        )
    if len(grid) > 1:
        print(
            f"ATTENZIONE: il config produce {len(grid)} run diversi (vero grid "
            f"search). Sto usando la combinazione run_index={run_index}. "
            "Usa --run-index per selezionarne un'altra."
        )
    return grid[run_index]


def load_yolo_run_config(config_path, run_index=0):
    """
    Come load_run_config(), ma per file tipo 30.yolov10-fam.yaml, dove
    "parameters" e' gia' flat (task, model, data, epochs, batch, ...) e non
    annidato in {model, dataset, dataloader}. make_grid si riusa identico:
    e' la stessa funzione usata da Experimenter per i grid YOLO.
    """
    raw = load_yaml(config_path)
    parameters = raw.get("parameters", raw)
    grid = make_grid(parameters)
    if run_index >= len(grid):
        raise ValueError(
            f"Il config produce {len(grid)} combinazioni di grid search, "
            f"run_index={run_index} non e' valido."
        )
    if len(grid) > 1:
        print(
            f"ATTENZIONE: il config produce {len(grid)} run diversi (vero grid "
            f"search). Sto usando la combinazione run_index={run_index}. "
            "Usa --run-index per selezionarne un'altra."
        )
    return grid[run_index]


# ---------------------------------------------------------------------------
# 1. Caricamento modello + checkpoint
# ---------------------------------------------------------------------------

def load_fusion_model(model_params, checkpoint_path, device):
    """Caricamento modelli HF (RT-DETR / Deformable DETR / DINO fusion) da
    .safetensors, con recupero degli alias condivisi (class_embed/bbox_embed
    aliasati sia top-level sia dentro model.decoder)."""
    model = build_model(model_params)
    model.eval().to(device)

    with safe_open(checkpoint_path, framework="pt") as f:
        raw_weights = {k: f.get_tensor(k) for k in f.keys()}

    # I checkpoint sono salvati a partire da un WrapperModule (run.py),
    # le cui chiavi sono prefissate con "model." rispetto al modello nudo
    # che costruiamo qui con build_model(). Rimuoviamo il prefisso.
    weights = {
        (k[len("model."):] if k.startswith("model.") else k): v
        for k, v in raw_weights.items()
    }

    missing, unexpected = model.load_state_dict(weights, strict=False)

    # RTDetrFusionForObjectDetection tiene class_embed/bbox_embed sia come
    # attributi top-level (usati dal forward() di HF) sia aliasati dentro
    # model.decoder (letteralmente lo stesso oggetto Python, vedi
    # rtdetr_fusion.py: saved_class_embed = self.class_embed; poi
    # self.model.decoder.class_embed = saved_class_embed). safetensors non
    # puo' salvare due chiavi che condividono la stessa memoria, quindi nel
    # checkpoint sopravvive un solo path e l'altro risulta "missing" qui pur
    # essendo esattamente lo stesso peso allenato. Fallback: per ogni
    # parametro "missing", troviamo (per identita' di oggetto, non per nome
    # indovinato) tutti gli alias nel nostro modello, e vediamo se il
    # checkpoint ha il valore sotto uno di quei nomi alternativi.
    if missing:
        all_named_params = list(model.named_parameters(remove_duplicate=False))
        id_to_names = {}
        for name, param in all_named_params:
            id_to_names.setdefault(id(param), []).append(name)
        param_by_name = dict(all_named_params)

        recovered = []
        for key in list(missing):
            param = param_by_name.get(key)
            if param is None:
                continue
            for alias in id_to_names.get(id(param), []):
                if alias != key and alias in weights:
                    with torch.no_grad():
                        param.data.copy_(weights[alias])
                    recovered.append(key)
                    break
        if recovered:
            print(f"  Recuperati {len(recovered)}/{len(missing)} pesi 'missing' da alias condivisi (stesso tensore, path diverso nell'albero dei moduli)")
            missing = [k for k in missing if k not in recovered]
    print(f"[load_state_dict] missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  missing (prime 10):", list(missing)[:10])
    if unexpected:
        print("  unexpected (prime 10):", list(unexpected)[:10])
    if missing or unexpected:
        print(
            "  ATTENZIONE: lo state_dict non combacia perfettamente. "
            "Verifica il prefisso delle chiavi o l'architettura costruita "
            "da build_model() rispetto a quella salvata nel checkpoint."
        )
    else:
        print("  OK: tutte le chiavi combaciano.")

    return model


def load_yolo_model(checkpoint_path, device):
    """
    Carica un modello YOLOv10FusionFAM da un checkpoint .pt di ultralytics.
    A differenza dei modelli HF (safetensors + state_dict), il checkpoint
    ultralytics contiene l'istanza pickled del modello intero sotto la
    chiave "model" (vedi BaseTrainer.save_model), quindi non serve
    build_model() + load_state_dict(): i fam_modules sono gia' popolati
    con i pesi allenati, cosi' come tutto il resto dell'architettura.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model = model.float().eval().to(device)
    print(f"[load_yolo_model] tipo modello: {type(model).__name__}, use_fam={getattr(model, 'use_fam', 'N/A')}")
    return model


# ---------------------------------------------------------------------------
# 2. Hook su ogni istanza di FeatureAlignmentModule
# ---------------------------------------------------------------------------

class FAMCapture:
    """Cattura input (rgb_feat, ir_feat) e output (ir_aligned) di ogni
    FeatureAlignmentModule incontrato nel modello, individuato per nome
    di classe (indipendente dal path esatto nell'architettura, quindi
    funziona identico su RT-DETR, Deformable DETR/DINO e YOLOv10FusionFAM)."""

    def __init__(self, model):
        self.records = {}  # level_idx -> dict
        self.hooks = []
        self._register(model)

    def _register(self, model):
        level = 0
        last_name = None
        for name, module in model.named_modules():
            if type(module).__name__ == "FeatureAlignmentModule":
                self.hooks.append(
                    module.register_forward_hook(self._make_fam_hook(level, name))
                )
                if hasattr(module, "offset_conv"):
                    self.hooks.append(
                        module.offset_conv.register_forward_hook(
                            self._make_offset_hook(level)
                        )
                    )
                last_name = name
                level += 1
        if level == 0:
            raise RuntimeError(
                "Nessun FeatureAlignmentModule trovato nel modello. Controlla "
                "che use_fam=True nel config e che la classe si chiami "
                "esattamente 'FeatureAlignmentModule'."
            )
        print(f"Registrati hook su {level} istanze di FeatureAlignmentModule (es. '{last_name}')")

    def _make_fam_hook(self, level_idx, module_name):
        def hook(module, inputs, output):
            if len(inputs) < 2:
                raise RuntimeError(
                    f"FeatureAlignmentModule '{module_name}' chiamato con "
                    f"{len(inputs)} argomenti posizionali, ne servono 2 "
                    "(rgb_feat, ir_feat)."
                )
            rgb_feat, ir_feat = inputs[0], inputs[1]
            rec = self.records.setdefault(level_idx, {})
            rec["module_name"] = module_name
            rec["rgb"] = rgb_feat.detach().cpu()
            rec["ir"] = ir_feat.detach().cpu()
            rec["ir_aligned"] = output.detach().cpu()
        return hook

    def _make_offset_hook(self, level_idx):
        def hook(module, inputs, output):
            out = output.detach().cpu()
            rec = self.records.setdefault(level_idx, {})
            rec["offset"] = out[:, :18]
            rec["mask"] = torch.sigmoid(out[:, 18:])
        return hook

    def remove(self):
        for h in self.hooks:
            h.remove()


# ---------------------------------------------------------------------------
# 3. PCA -> RGB (stile DINOv2/DINOv3)
# ---------------------------------------------------------------------------

def fit_pca_projector(feats, isolate_foreground=True, fg_percentile=50):
    """
    feats: lista di tensor (C, H, W) che condividono lo stesso C (es. rgb,
    ir, ir_aligned dello stesso livello). Fitta UNA base PCA condivisa
    (foreground-split + proiezione a 3 componenti) sui pixel messi in comune
    di tutte le feature map, cosi' i colori risultanti sono direttamente
    confrontabili tra i pannelli.

    IMPORTANTE: se si fitta una PCA indipendente per ogni feature map, le
    basi risultanti sono arbitrarie nel segno/rotazione delle componenti:
    "blu" in una mappa e "blu" in un'altra non corrispondono necessariamente
    alla stessa struttura. Con una base condivisa, lo stesso colore in due
    pannelli indica davvero la stessa direzione nello spazio delle feature.

    Ritorna una funzione project(feat) -> uint8 (H, W, 3).
    """
    flats = [f.permute(1, 2, 0).reshape(-1, f.shape[0]).numpy().astype(np.float64) for f in feats]
    pooled = np.concatenate(flats, axis=0)

    fg_pca1, fg_thresh, fg_invert = None, None, False
    if isolate_foreground and pooled.shape[0] > 3:
        fg_pca1 = PCA(n_components=1)
        comp1_pooled = fg_pca1.fit_transform(pooled).squeeze(-1)
        fg_thresh = np.percentile(comp1_pooled, fg_percentile)
        fg_invert = (comp1_pooled > fg_thresh).mean() > 0.5

    def fg_mask_of(flat):
        if fg_pca1 is None:
            return np.ones(flat.shape[0], dtype=bool)
        comp1 = fg_pca1.transform(flat).squeeze(-1)
        mask = comp1 > fg_thresh
        if fg_invert:
            mask = ~mask
        return mask if mask.sum() >= 3 else np.ones(flat.shape[0], dtype=bool)

    pooled_fg_mask = fg_mask_of(pooled)
    n_comp = min(3, int(pooled_fg_mask.sum()))
    pca3 = PCA(n_components=n_comp)
    pca3.fit(pooled[pooled_fg_mask])

    proj_pooled = pca3.transform(pooled[pooled_fg_mask])
    norm_bounds = [tuple(np.percentile(proj_pooled[:, c], [1, 99])) for c in range(n_comp)]

    def project(feat):
        C, H, W = feat.shape
        flat = feat.permute(1, 2, 0).reshape(-1, C).numpy().astype(np.float64)
        mask = fg_mask_of(flat)
        proj = pca3.transform(flat)
        if proj.shape[1] < 3:
            proj = np.pad(proj, ((0, 0), (0, 3 - proj.shape[1])))
        norm = np.zeros_like(proj)
        for c in range(min(3, len(norm_bounds))):
            lo, hi = norm_bounds[c]
            norm[:, c] = np.clip((proj[:, c] - lo) / max(hi - lo, 1e-6), 0, 1)
        img = norm.reshape(H, W, 3)
        if isolate_foreground:
            img = img * mask.reshape(H, W, 1)
        return (img * 255).astype(np.uint8)

    return project


def standardize(feat):
    """
    Z-score globale (una sola media/std per l'intera feature map), per
    rimuovere differenze di scala pure tra RGB/IR/FAM(IR) prima del fit PCA
    condiviso (deform_conv non ha una BatchNorm/GroupNorm a valle, quindi il
    suo output puo' avere una scala di attivazione diversa da rgb_feat/
    ir_feat), mantenendo pero' le proporzioni relative tra i canali.
    """
    mean = feat.mean()
    std = feat.std().clamp_min(1e-6)
    return (feat - mean) / std


def overlay(img_a, img_b, alpha=0.5):
    """Blend alpha per confrontare direttamente due PCA-map spazialmente."""
    return (alpha * img_a.astype(np.float32) + (1 - alpha) * img_b.astype(np.float32)).astype(np.uint8)


# ---------------------------------------------------------------------------
# 4. Campo di offset del FAM (bonus diagnostico)
# ---------------------------------------------------------------------------

def plot_offset_field(ax, offset, mask, stride=4):
    """
    offset: (18, H, W) - 9 punti kernel x (dx, dy); mask: (9, H, W).
    Visualizza lo spostamento medio (pesato dalla mask di modulazione) sui
    9 punti campionati per cella: da un'idea dello spostamento "netto"
    imparato dal FAM in quella posizione.
    """
    _, H, W = offset.shape
    off = offset.reshape(9, 2, H, W)
    m = mask.reshape(9, 1, H, W)
    denom = m.sum(0) + 1e-6
    mean_dx = (off[:, 0:1] * m).sum(0) / denom
    mean_dy = (off[:, 1:2] * m).sum(0) / denom
    mean_dx, mean_dy = mean_dx[0].numpy(), mean_dy[0].numpy()

    ys, xs = np.mgrid[0:H:stride, 0:W:stride]
    ax.quiver(
        xs, ys,
        mean_dx[::stride, ::stride], -mean_dy[::stride, ::stride],
        color="red", angles="xy", scale_units="xy", scale=0.3, width=0.003,
    )
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_title("Campo di offset FAM\n(media pesata sui 9 punti)", fontsize=9)


def offset_spatial_uniformity(offset):
    """
    offset: (18, H, W) - 9 punti kernel x (dx, dy), in pixel di feature map.

    Misura se l'offset predetto e' quasi COSTANTE su tutte le celle di
    output (uno shift/bias uniforme, non content-adaptive) oppure se varia
    genuinamente in funzione della posizione.

    NOTA STORICA: una prima versione di questa diagnostica (rimossa)
    confrontava la varianza della posizione ASSOLUTA campionata (griglia +
    offset) con la varianza di una griglia senza offset. Si e' rivelata
    matematicamente cieca al fenomeno cercato: quando l'offset ha ampiezza
    piccola (qui: 1-2 px di feature map) rispetto all'estensione della
    griglia su cui varia la posizione base (es. 0..79 a P3), il suo
    contributo alla varianza totale e' trascurabile sia che l'offset sia
    perfettamente costante sia che vari normalmente -- il ratio risultava
    sempre ~1.0 in entrambi gli scenari, senza discriminare nulla.

    Questa versione guarda invece la deviazione standard SPAZIALE
    dell'offset stesso (quanto cambia tra celle di output diverse),
    normalizzata sulla sua ampiezza media: un rapporto vicino a 0 indica un
    offset quasi-costante (stesso vettore ovunque, uno shift globale);
    vicino a 1 indica un offset che varia tanto quanto la sua stessa
    ampiezza media (genuinamente dipendente dalla posizione di output).
    """
    off = offset.reshape(9, 2, -1)  # (9, 2, H*W)
    spatial_std = off.std(dim=2).mean().item()   # quanto l'offset varia tra celle diverse
    magnitude = off.abs().mean().item()           # ampiezza media dell'offset
    uniformity_ratio = spatial_std / max(magnitude, 1e-8)
    return {
        "offset_spatial_std": spatial_std,
        "offset_magnitude": magnitude,
        "uniformity_ratio": uniformity_ratio,  # ~0 = shift uniforme/costante, ~1 = varia quanto la propria ampiezza
    }


# ---------------------------------------------------------------------------
# 5. Campione RGB-IR sincronizzato (stessa pipeline di training/eval)
# ---------------------------------------------------------------------------

def load_sample(dataset_params, dataloader_params, sample_idx, split, device):
    """Campione per modelli HF (RT-DETR/DefDETR/DINO), via get_dataloaders
    (pipeline WiSARDDataset + AutoProcessor HF)."""
    (_train_l, _val_l, _test_l), (train_set, val_set, test_set), _collate, denormalize = get_dataloaders(
        dict(dataset_params), dict(dataloader_params), return_datasets=True
    )
    dataset = {"train": train_set, "val": val_set, "test": test_set}[split]
    sample = dataset[sample_idx]
    pixel_values = sample.pixel_values.unsqueeze(0).to(device)  # (1, 4, H, W) RGB+IR
    return pixel_values, denormalize


def load_yolo_sample(data_yaml, run_config, sample_idx, split, device):
    """
    Campione per YOLOv10FusionFAM, via WiSARDYOLODataset, costruito con la
    stessa identica funzione usata in produzione (build_yolo_dataset in
    sarfusion/experiment/yolo.py) per evitare scostamenti di preprocessing
    rispetto a training/eval reali.
    """
    data_dict = check_det_dataset(data_yaml)
    img_path = data_dict[split]

    cfg_dict = cfg2dict(WISARD_DEFAULT_CFG)
    for k in ["imgsz", "rect", "cache", "single_cls", "task", "classes", "fraction"]:
        if k in run_config:
            cfg_dict[k] = run_config[k]
    cfg = IterableSimpleNamespace(**cfg_dict)

    dataset = WiSARDYOLODataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=1,
        augment=False,
        hyp=cfg,
        rect=False,
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=32,
        pad=0.5,
        prefix=colorstr(f"{split}: "),
        task=cfg.task,
        classes=cfg.classes,
        data=data_dict,
        fraction=1.0,
        augment_vis_ir=False,  # forzato: vogliamo sempre la coppia RGB+IR completa e deterministica
    )

    sample = dataset[sample_idx]
    img = sample["img"]  # CHW, float32 gia' normalizzato in [0,1] da BaseDataset.__getitem__
    if img.dtype == torch.uint8:
        # Difesa: se in futuro il comportamento di ultralytics cambiasse, non
        # falliamo silenziosamente con valori scalati male.
        img = img.float() / 255.0
    pixel_values = img.unsqueeze(0).to(device)  # (1, 4, H, W)
    return pixel_values, None


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", choices=["hf", "yolo"], default="hf",
                         help="hf: RT-DETR/DefDETR/DINO (safetensors); yolo: YOLOv10FusionFAM (.pt ultralytics)")
    parser.add_argument("--config", required=True, help="yaml del run da ispezionare")
    parser.add_argument("--run-index", type=int, default=0, help="indice della combinazione da usare se il config produce piu' di un run (grid search vera)")
    parser.add_argument("--checkpoint", required=True, help="[hf] path a best/model.safetensors | [yolo] path a weights/best.pt")
    parser.add_argument("--dataset-root", default=None, help="[solo hf] override del path assoluto al dataset WiSARD (il campo 'root' nel yaml e' relativo)")
    parser.add_argument("--data-yaml", default=None, help="[solo yolo] path al file dataset yaml (es. wisards_vis_ir.yaml)")
    parser.add_argument("--sample-idx", type=int, nargs="+", default=[0], help="uno o piu' indici di campione; se piu' di uno, oltre alle figure per-campione stampa un riepilogo aggregato degli offset (in pixel immagine) su tutti i campioni")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--levels", type=int, nargs="+", default=None, help="livelli piramide da visualizzare (default: tutti)")
    parser.add_argument("--out-dir", default="./fam_alignment_vis")
    parser.add_argument("--no-fg-isolation", action="store_true", help="disattiva l'isolamento del foreground nella PCA")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    if args.model_type == "yolo":
        if not args.data_yaml:
            raise ValueError("--data-yaml e' obbligatorio con --model-type yolo")
        run_config = load_yolo_run_config(args.config, run_index=args.run_index)
        print(f"Modello YOLO | model params: {run_config.get('model')}")
        model = load_yolo_model(args.checkpoint, device)
    else:
        run_config = load_run_config(args.config, run_index=args.run_index)
        model_params = run_config["model"]
        dataset_params = run_config["dataset"]
        dataloader_params = run_config["dataloader"]
        if args.dataset_root:
            dataset_params["root"] = args.dataset_root
        print(f"Modello: {model_params['name']} | params: {model_params['params']}")
        print(f"Dataset root: {dataset_params['root']} | folders: {dataset_params['folders']}")
        model = load_fusion_model(model_params, args.checkpoint, device)

    capture = FAMCapture(model)

    isolate_fg = not args.no_fg_isolation
    offset_px_by_level = {}  # level -> lista di np.array piatti, uno per campione, in pixel immagine

    for sample_idx in args.sample_idx:
        capture.records.clear()

        if args.model_type == "yolo":
            pixel_values, _ = load_yolo_sample(args.data_yaml, run_config, sample_idx, args.split, device)
        else:
            pixel_values, _ = load_sample(dataset_params, dataloader_params, sample_idx, args.split, device)

        with torch.no_grad():
            if args.model_type == "yolo":
                model(pixel_values)
            else:
                model(pixel_values=pixel_values)

        print(f"--- Campione {sample_idx} | shape input: {tuple(pixel_values.shape)} ---")

        levels = args.levels or sorted(capture.records.keys())

        for level in levels:
            rec = capture.records.get(level)
            if rec is None or "rgb" not in rec:
                print(f"Livello {level}: nessun dato catturato, salto.")
                continue

            rgb_feat = rec["rgb"][0]
            ir_feat = rec["ir"][0]
            ir_aligned_feat = rec["ir_aligned"][0]

            def _stats(name, t):
                global_std = t.std()
                spatial_std = t.std(dim=(1, 2)).mean()  # std nello spazio (H,W), media sui canali
                print(f"  [stats campione {sample_idx} livello {level}] {name:8s}: mean={t.mean():+.4f} std_globale={global_std:.4f} std_spaziale={spatial_std:.4f} min={t.min():+.4f} max={t.max():+.4f}")

            _stats("rgb", rgb_feat)
            _stats("ir", ir_feat)
            _stats("fam(ir)", ir_aligned_feat)
            if "offset" in rec:
                off = rec["offset"][0]
                stride = pixel_values.shape[-1] / rgb_feat.shape[-1]
                off_px = off.abs().numpy() * stride
                offset_px_by_level.setdefault(level, []).append(off_px.reshape(-1))
                print(
                    f"  [stats campione {sample_idx} livello {level}] offset  : mean_abs={off.abs().mean():.4f} max_abs={off.abs().max():.4f} "
                    f"(feature-px) | stride~{stride:.0f} -> mean~{off_px.mean():.2f}px max~{off_px.max():.2f}px (pixel immagine originale)"
                )
                pos_var = offset_spatial_uniformity(off)
                print(
                    f"  [stats campione {sample_idx} livello {level}] uniformity: "
                    f"offset_spatial_std={pos_var['offset_spatial_std']:.4f} "
                    f"offset_magnitude={pos_var['offset_magnitude']:.4f} "
                    f"uniformity_ratio={pos_var['uniformity_ratio']:.3f} "
                    "[ratio~0 = offset quasi costante/shift uniforme su tutta la cella, ratio~1 = varia genuinamente con la posizione]"
                )

            rgb_n = standardize(rgb_feat)
            ir_n = standardize(ir_feat)
            ir_aligned_n = standardize(ir_aligned_feat)

            pca_projector = fit_pca_projector(
                [rgb_n, ir_n, ir_aligned_n], isolate_foreground=isolate_fg
            )
            pca_rgb = pca_projector(rgb_n)
            pca_ir = pca_projector(ir_n)
            pca_ir_aligned = pca_projector(ir_aligned_n)

            overlay_pre = overlay(pca_rgb, pca_ir)
            overlay_post = overlay(pca_rgb, pca_ir_aligned)

            has_offset = "offset" in rec
            n_cols = 6 if has_offset else 5
            fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4.2))

            panels = [
                (pca_rgb, "PCA(RGB)"),
                (pca_ir, "PCA(IR) - pre FAM"),
                (pca_ir_aligned, "PCA(FAM(IR)) - post FAM"),
                (overlay_pre, "Overlay RGB+IR (pre)"),
                (overlay_post, "Overlay RGB+FAM(IR)\n(= input decoder/neck, post)"),
            ]
            for ax, (img, title) in zip(axes, panels):
                ax.imshow(img)
                ax.set_title(title, fontsize=9)
                ax.axis("off")

            if has_offset:
                plot_offset_field(axes[-1], rec["offset"][0], rec["mask"][0])

            fig.suptitle(f"FAM alignment check [{args.model_type}] - campione {sample_idx} - livello {level} - {rec.get('module_name', '')}")
            fig.tight_layout()
            out_path = Path(args.out_dir) / f"fam_sample{sample_idx}_level_{level}.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"Salvato: {out_path}")

    capture.remove()

    if len(args.sample_idx) > 1 and offset_px_by_level:
        print("\n=== Riepilogo offset aggregato su tutti i campioni (pixel immagine originale) ===")
        for level in sorted(offset_px_by_level.keys()):
            pooled = np.concatenate(offset_px_by_level[level])
            mean_v = pooled.mean()
            median_v = np.median(pooled)
            p90_v = np.percentile(pooled, 90)
            max_v = pooled.max()
            print(
                f"  livello {level}: mean={mean_v:.2f}px median={median_v:.2f}px "
                f"p90={p90_v:.2f}px max={max_v:.2f}px (n_campioni={len(args.sample_idx)}, "
                f"n_offset_totali={pooled.size})"
            )


if __name__ == "__main__":
    main()