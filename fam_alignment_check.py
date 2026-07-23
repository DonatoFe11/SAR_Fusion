"""
fam_alignment_check.py
-----------------------
Validates FAM (Feature Alignment Module) alignment through PCA
visualization of feature maps, in the style of DINOv2/DINOv3.

Supports two model families:
  --model-type hf    (default) RT-DETR / Deformable DETR / DINO fusion,
                      loaded from .safetensors + an HF grid-search YAML config.
  --model-type yolo  YOLOv10FusionFAM, loaded from an Ultralytics .pt checkpoint
                      (the complete model instance is pickled in the checkpoint,
                      so build_model()+load_state_dict() is not needed).

For a fusion model with use_fam=True, forward hooks capture RGB/IR features
immediately before and after every FeatureAlignmentModule in the backbone.
The features are projected to RGB with PCA (three principal components ->
R, G, B channels), with optional foreground isolation based on the first
component (the same technique used in DINOv2/v3 visualizations).

For each feature-pyramid level, the script produces a figure containing:
  (a) PCA(feature RGB)
  (b) PCA(feature IR)              -- pre-FAM
  (c) PCA(feature FAM(IR))         -- post-FAM, the actual decoder/neck input
  (d) RGB+IR overlay                -- alpha blend, shows pre-alignment differences
  (e) RGB+FAM(IR) overlay           -- alpha blend after alignment
  (f) FAM offset field              -- additional diagnostic quiver plot

Hooks are registered by CLASS NAME (FeatureAlignmentModule), rather than by a
fixed path. The script is therefore compatible, without modification, with any
architecture that reuses this class (rtdetr_fusion.py,
deformable_detr_fusion.py, yolo_fusion_fam.py).

Usage (HF):
    python fam_alignment_check.py \
        --config /path/to/fusion_rtdetr.yaml \
        --checkpoint /path/to/tracking_dir/<run>/best/model.safetensors \
        --dataset-root /path/assoluto/a/dataset/WiSARD \
        --sample-idx 0 \
        --split val \
        --out-dir ./fam_alignment_vis

Usage (YOLO):
    python fam_alignment_check.py \
        --model-type yolo \
        --config parameters/YOLO/30.yolov10-fam.yaml \
        --run-index 2 \
        --checkpoint SarYOLO/YOLOv10-FAM-Grid2/weights/best.pt \
        --data-yaml wisards_vis_ir.yaml \
        --sample-idx 0 1 2 \
        --split val \
        --out-dir ./fam_alignment_vis_yolo

Additional dependencies beyond the sarfusion environment: scikit-learn, matplotlib
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

# --- YOLO-specific imports (used only with --model-type yolo) ---
from ultralytics.data.utils import check_det_dataset
from ultralytics.cfg import cfg2dict, IterableSimpleNamespace
from ultralytics.utils import colorstr

from sarfusion.data.wisard import WiSARDYOLODataset
from sarfusion.experiment.yolo import WISARD_DEFAULT_CFG


# ---------------------------------------------------------------------------
# 0. Reconstructing a run configuration from the grid-search format
# ---------------------------------------------------------------------------

def load_run_config(config_path, run_index=0):
    """
    Reuses sarfusion.utils.grid.make_grid (the same function used by
    Experimenter.calculate_runs) to turn the YAML "parameters" section,
    where every terminal value is wrapped in a list as a grid-search axis,
    into a flat configuration with actual scalar/dict/list values.

    HF format: "parameters" is nested under {model, dataset, dataloader, ...}.
    """
    raw = load_yaml(config_path)
    parameters = raw.get("parameters", raw)  # fallback when the file is already flat
    grid = make_grid(parameters)
    if run_index >= len(grid):
        raise ValueError(
            f"The config produces {len(grid)} grid-search combinations; "
            f"run_index={run_index} is invalid."
        )
    if len(grid) > 1:
        print(
            f"WARNING: the config produces {len(grid)} distinct runs (a real grid "
            f"search). Using combination run_index={run_index}. "
            "Use --run-index to select another one."
        )
    return grid[run_index]


def load_yolo_run_config(config_path, run_index=0):
    """
    Like load_run_config(), but for files such as 30.yolov10-fam.yaml, where
    "parameters" is already flat (task, model, data, epochs, batch, ...) rather
    than nested under {model, dataset, dataloader}. make_grid is reused unchanged:
    it is the same function used by Experimenter for YOLO grids.
    """
    raw = load_yaml(config_path)
    parameters = raw.get("parameters", raw)
    grid = make_grid(parameters)
    if run_index >= len(grid):
        raise ValueError(
            f"The config produces {len(grid)} grid-search combinations; "
            f"run_index={run_index} is invalid."
        )
    if len(grid) > 1:
        print(
            f"WARNING: the config produces {len(grid)} distinct runs (a real grid "
            f"search). Using combination run_index={run_index}. "
            "Use --run-index to select another one."
        )
    return grid[run_index]


# ---------------------------------------------------------------------------
# 1. Model and checkpoint loading
# ---------------------------------------------------------------------------

def load_fusion_model(model_params, checkpoint_path, device):
    """Load HF fusion models (RT-DETR / Deformable DETR / DINO) from
    .safetensors, recovering shared aliases (class_embed/bbox_embed are
    aliased both at top level and inside model.decoder)."""
    model = build_model(model_params)
    model.eval().to(device)

    with safe_open(checkpoint_path, framework="pt") as f:
        raw_weights = {k: f.get_tensor(k) for k in f.keys()}

    # Checkpoints are saved from a WrapperModule (run.py), whose keys carry a
    # "model." prefix relative to the bare model built here with build_model().
    # Remove that prefix.
    weights = {
        (k[len("model."):] if k.startswith("model.") else k): v
        for k, v in raw_weights.items()
    }

    missing, unexpected = model.load_state_dict(weights, strict=False)

    # RTDetrFusionForObjectDetection keeps class_embed/bbox_embed both as
    # top-level attributes (used by the HF forward()) and aliases inside
    # model.decoder (the exact same Python object; see rtdetr_fusion.py).
    # safetensors cannot save two keys sharing the same memory. Consequently,
    # only one path is retained in a checkpoint and the other may appear as
    # missing even though it refers to the trained tensor. For every missing
    # parameter, find all aliases by object identity, then restore it if the
    # checkpoint contains one of the alternative names.
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
            print(f"  Recovered {len(recovered)}/{len(missing)} missing weights from shared aliases (same tensor, different module-tree path)")
            missing = [k for k in missing if k not in recovered]
    print(f"[load_state_dict] missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  missing (first 10):", list(missing)[:10])
    if unexpected:
        print("  unexpected (first 10):", list(unexpected)[:10])
    if missing or unexpected:
        print(
            "  WARNING: the state_dict does not match exactly. "
            "Check the key prefix and whether the architecture built by "
            "build_model() matches the one saved in the checkpoint."
        )
    else:
        print("  OK: all keys match.")

    return model


def load_yolo_model(checkpoint_path, device):
    """
    Load a YOLOv10FusionFAM model from an Ultralytics .pt checkpoint.
    Unlike HF models (safetensors + state_dict), an Ultralytics checkpoint
    contains the complete pickled model instance under the "model" key (see
    BaseTrainer.save_model). Therefore build_model() + load_state_dict() is
    not needed: FAM modules and the rest of the architecture already contain
    their trained weights.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model = model.float().eval().to(device)
    print(f"[load_yolo_model] model type: {type(model).__name__}, use_fam={getattr(model, 'use_fam', 'N/A')}")
    return model


# ---------------------------------------------------------------------------
# 2. Hooks on every FeatureAlignmentModule instance
# ---------------------------------------------------------------------------

class FAMCapture:
    """Capture inputs (rgb_feat, ir_feat) and output (ir_aligned) of every
    FeatureAlignmentModule in the model. Modules are identified by class name,
    making this independent of their architecture path and applicable to
    RT-DETR, Deformable DETR/DINO, and YOLOv10FusionFAM."""

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
                "No FeatureAlignmentModule found in the model. Check that "
                "use_fam=True in the config and that the class is named "
                "exactly 'FeatureAlignmentModule'."
            )
        print(f"Registered hooks on {level} FeatureAlignmentModule instances (e.g. '{last_name}')")

    def _make_fam_hook(self, level_idx, module_name):
        def hook(module, inputs, output):
            if len(inputs) < 2:
                raise RuntimeError(
                    f"FeatureAlignmentModule '{module_name}' was called with "
                    f"{len(inputs)} positional arguments; 2 are required "
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
# 3. PCA -> RGB (DINOv2/DINOv3 style)
# ---------------------------------------------------------------------------

def fit_pca_projector(feats, isolate_foreground=True, fg_percentile=50):
    """
    feats: list of (C, H, W) tensors with the same C (e.g. rgb, ir, and
    ir_aligned at one level). Fit ONE shared PCA basis (foreground split +
    three-component projection) on the pooled pixels of all feature maps so
    that colors are directly comparable across panels.

    IMPORTANT: independently fitting PCA for each feature map gives bases
    with arbitrary component sign and rotation. Therefore, "blue" in one map
    need not represent the same structure as "blue" in another. With a shared
    basis, the same color in two panels denotes the same feature-space direction.

    Returns a project(feat) -> uint8 (H, W, 3) function.
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
    Global z-score (one mean/std for the entire feature map) removes pure
    scale differences between RGB/IR/FAM(IR) before fitting shared PCA.
    deform_conv has no downstream BatchNorm/GroupNorm, so its output may use
    a different activation scale from rgb_feat/ir_feat. This normalization
    preserves relative channel proportions within each feature map.
    """
    mean = feat.mean()
    std = feat.std().clamp_min(1e-6)
    return (feat - mean) / std


def overlay(img_a, img_b, alpha=0.5):
    """Alpha-blend two PCA maps for a direct spatial comparison."""
    return (alpha * img_a.astype(np.float32) + (1 - alpha) * img_b.astype(np.float32)).astype(np.uint8)


# ---------------------------------------------------------------------------
# 4. FAM offset field (additional diagnostic)
# ---------------------------------------------------------------------------

def plot_offset_field(ax, offset, mask, stride=4):
    """
    offset: (18, H, W) - 9 kernel points x (dx, dy); mask: (9, H, W).
    Visualizes the mask-weighted mean displacement across the nine sampling
    points per cell, providing an intuitive view of the net shift learned by
    the FAM at each position.
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
    ax.set_title("FAM offset field\n(mask-weighted mean over 9 points)", fontsize=9)


def offset_spatial_uniformity(offset):
    """
    offset: (18, H, W) - 9 kernel points x (dx, dy), in feature-map pixels.

    Measures whether the predicted offset is nearly CONSTANT over all output
    cells (a uniform shift/bias rather than a content-adaptive correction), or
    genuinely varies with position.

    It computes the SPATIAL standard deviation of the offset itself (how much
    it changes among output cells), normalized by its mean magnitude. A ratio
    near 0 indicates a nearly constant offset (the same vector everywhere, a
    global shift); a ratio near 1 indicates variation comparable to the mean
    offset magnitude and hence a genuinely position-dependent correction.
    """
    off = offset.reshape(9, 2, -1)  # (9, 2, H*W)
    spatial_std = off.std(dim=2).mean().item()   # variation across output cells
    magnitude = off.abs().mean().item()          # mean offset magnitude
    uniformity_ratio = spatial_std / max(magnitude, 1e-8)
    return {
        "offset_spatial_std": spatial_std,
        "offset_magnitude": magnitude,
        "uniformity_ratio": uniformity_ratio,  # ~0 = uniform shift, ~1 = varies as much as its magnitude
    }


# ---------------------------------------------------------------------------
# 5. Synchronized RGB-IR sample (same training/evaluation pipeline)
# ---------------------------------------------------------------------------

def load_sample(dataset_params, dataloader_params, sample_idx, split, device):
    """Load an HF-model sample (RT-DETR/DefDETR/DINO) through get_dataloaders
    (WiSARDDataset + HF AutoProcessor pipeline)."""
    (_train_l, _val_l, _test_l), (train_set, val_set, test_set), _collate, denormalize = get_dataloaders(
        dict(dataset_params), dict(dataloader_params), return_datasets=True
    )
    dataset = {"train": train_set, "val": val_set, "test": test_set}[split]
    sample = dataset[sample_idx]
    pixel_values = sample.pixel_values.unsqueeze(0).to(device)  # (1, 4, H, W) RGB+IR
    return pixel_values, denormalize


def load_yolo_sample(data_yaml, run_config, sample_idx, split, device):
    """
    Load a YOLOv10FusionFAM sample through WiSARDYOLODataset, using the same
    construction procedure as production (build_yolo_dataset in
    sarfusion/experiment/yolo.py) to avoid preprocessing discrepancies from
    actual training/evaluation.
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
        augment_vis_ir=False,  # always use a complete, deterministic RGB+IR pair
    )

    sample = dataset[sample_idx]
    img = sample["img"]  # CHW float32, already normalized to [0, 1] by BaseDataset.__getitem__
    if img.dtype == torch.uint8:
        # Defensive check: if Ultralytics changes this behavior in the future,
        # do not silently continue with incorrectly scaled values.
        img = img.float() / 255.0
    pixel_values = img.unsqueeze(0).to(device)  # (1, 4, H, W)
    return pixel_values, None


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", choices=["hf", "yolo"], default="hf",
                         help="hf: RT-DETR/DefDETR/DINO (.safetensors); yolo: YOLOv10FusionFAM (Ultralytics .pt)")
    parser.add_argument("--config", required=True, help="run YAML configuration to inspect")
    parser.add_argument("--run-index", type=int, default=0, help="combination index when the config produces multiple grid-search runs")
    parser.add_argument("--checkpoint", required=True, help="[hf] path to best/model.safetensors | [yolo] path to weights/best.pt")
    parser.add_argument("--dataset-root", default=None, help="[hf only] absolute WiSARD dataset-root override (the YAML root field is relative)")
    parser.add_argument("--data-yaml", default=None, help="[yolo only] path to the dataset YAML (e.g. wisards_vis_ir.yaml)")
    parser.add_argument("--sample-idx", type=int, nargs="+", default=[0], help="one or more sample indices; with multiple samples, prints pooled offset statistics in image pixels in addition to per-sample figures")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val", help="dataset split to inspect")
    parser.add_argument("--levels", type=int, nargs="+", default=None, help="feature-pyramid levels to visualize (default: all)")
    parser.add_argument("--out-dir", default="./fam_alignment_vis")
    parser.add_argument("--no-fg-isolation", action="store_true", help="disable foreground isolation in PCA")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    if args.model_type == "yolo":
        if not args.data_yaml:
            raise ValueError("--data-yaml is required with --model-type yolo")
        run_config = load_yolo_run_config(args.config, run_index=args.run_index)
        print(f"YOLO model | model params: {run_config.get('model')}")
        model = load_yolo_model(args.checkpoint, device)
    else:
        run_config = load_run_config(args.config, run_index=args.run_index)
        model_params = run_config["model"]
        dataset_params = run_config["dataset"]
        dataloader_params = run_config["dataloader"]
        if args.dataset_root:
            dataset_params["root"] = args.dataset_root
        print(f"Model: {model_params['name']} | params: {model_params['params']}")
        print(f"Dataset root: {dataset_params['root']} | folders: {dataset_params['folders']}")
        model = load_fusion_model(model_params, args.checkpoint, device)

    capture = FAMCapture(model)

    isolate_fg = not args.no_fg_isolation
    offset_px_by_level = {}  # level -> flat np.array list, one per sample, in image pixels

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

        print(f"--- Sample {sample_idx} | input shape: {tuple(pixel_values.shape)} ---")

        levels = args.levels or sorted(capture.records.keys())

        for level in levels:
            rec = capture.records.get(level)
            if rec is None or "rgb" not in rec:
                print(f"Level {level}: no data captured; skipping.")
                continue

            rgb_feat = rec["rgb"][0]
            ir_feat = rec["ir"][0]
            ir_aligned_feat = rec["ir_aligned"][0]

            def _stats(name, t):
                global_std = t.std()
                spatial_std = t.std(dim=(1, 2)).mean()  # std across channels, then mean over channels
                print(f"  [sample {sample_idx}, level {level} stats] {name:8s}: mean={t.mean():+.4f} global_std={global_std:.4f} spatial_std={spatial_std:.4f} min={t.min():+.4f} max={t.max():+.4f}")

            _stats("rgb", rgb_feat)
            _stats("ir", ir_feat)
            _stats("fam(ir)", ir_aligned_feat)
            if "offset" in rec:
                off = rec["offset"][0]
                stride = pixel_values.shape[-1] / rgb_feat.shape[-1]
                off_px = off.abs().numpy() * stride
                offset_px_by_level.setdefault(level, []).append(off_px.reshape(-1))
                print(
                    f"  [sample {sample_idx}, level {level} stats] offset  : mean_abs={off.abs().mean():.4f} max_abs={off.abs().max():.4f} "
                    f"(feature-map px) | stride~{stride:.0f} -> mean~{off_px.mean():.2f}px max~{off_px.max():.2f}px (original-image pixels)"
                )
                pos_var = offset_spatial_uniformity(off)
                print(
                    f"  [sample {sample_idx}, level {level} stats] uniformity: "
                    f"offset_spatial_std={pos_var['offset_spatial_std']:.4f} "
                    f"offset_magnitude={pos_var['offset_magnitude']:.4f} "
                    f"uniformity_ratio={pos_var['uniformity_ratio']:.3f} "
                    "[ratio~0 = nearly constant uniform shift, ratio~1 = genuinely varies with position]"
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
                (pca_ir, "PCA(IR) - pre-FAM"),
                (pca_ir_aligned, "PCA(FAM(IR)) - post-FAM"),
                (overlay_pre, "RGB+IR overlay (pre-FAM)"),
                (overlay_post, "RGB+FAM(IR) overlay\n(= decoder/neck input, post-FAM)"),
            ]
            for ax, (img, title) in zip(axes, panels):
                ax.imshow(img)
                ax.set_title(title, fontsize=9)
                ax.axis("off")

            if has_offset:
                plot_offset_field(axes[-1], rec["offset"][0], rec["mask"][0])

            fig.suptitle(f"FAM alignment check [{args.model_type}] - sample {sample_idx} - level {level} - {rec.get('module_name', '')}")
            fig.tight_layout()
            out_path = Path(args.out_dir) / f"fam_sample{sample_idx}_level_{level}.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"Saved: {out_path}")

    capture.remove()

    if len(args.sample_idx) > 1 and offset_px_by_level:
        print("\n=== Offset summary pooled across all samples (original-image pixels) ===")
        for level in sorted(offset_px_by_level.keys()):
            pooled = np.concatenate(offset_px_by_level[level])
            mean_v = pooled.mean()
            median_v = np.median(pooled)
            p90_v = np.percentile(pooled, 90)
            max_v = pooled.max()
            print(
                f"  level {level}: mean={mean_v:.2f}px median={median_v:.2f}px "
                f"p90={p90_v:.2f}px max={max_v:.2f}px (n_samples={len(args.sample_idx)}, "
                f"n_total_offsets={pooled.size})"
            )


if __name__ == "__main__":
    main()
