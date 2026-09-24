"""Evaluate YOLOv10FusionFAM checkpoints in VIS, IR and VIS+IR modes.

PaddedWiSARDYOLODataset keeps four-channel inputs for single-modality
samples. An explicit mask selects the available backbone; FAM applies
to samples with both modalities present.

Boxes and predictions are mapped back to the original image coordinates
using the Ultralytics validator's preprocessing. Metrics are computed
with torchmetrics.MeanAveragePrecision.

Example:
    python eval_yolo_modalities.py --config parameters/YOLO/30.yolov10-fam.yaml
        --run-index 0 --checkpoint SarYOLO/YOLOv10-FAM-Grid8/weights/best.pt
        --data-yaml wisards_vis.yaml --modality auto --split test --batch-size 8
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ultralytics.utils import ops, colorstr
from ultralytics.cfg import cfg2dict, IterableSimpleNamespace
from ultralytics.data.utils import check_det_dataset
from torchmetrics.detection import MeanAveragePrecision

from sarfusion.data.wisard import WiSARDYOLODataset
from sarfusion.experiment.yolo import WISARD_DEFAULT_CFG
from sarfusion.utils.utils import load_yaml
from sarfusion.utils.grid import make_grid


# ---------------------------------------------------------------------------
# 0. Run configuration (flat format, as in fam_alignment_check.py)
# ---------------------------------------------------------------------------

def load_yolo_run_config(config_path, run_index=0):
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
            f"ATTENZIONE: il config produce {len(grid)} run diversi. "
            f"Sto usando run_index={run_index}."
        )
    return grid[run_index]


# ---------------------------------------------------------------------------
# 1. Model loading (pickled instance, as in fam_alignment_check.py)
# ---------------------------------------------------------------------------

def load_yolo_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model = model.float().eval().to(device)
    head_nc = getattr(getattr(model, "full_model", [None])[-1], "nc", "N/A") if hasattr(model, "full_model") else "N/A"
    print(f"[load_yolo_model] tipo modello: {type(model).__name__}, use_fam={getattr(model, 'use_fam', 'N/A')}, "
          f"model.nc (bookkeeping, puo' essere stale)={getattr(model, 'nc', 'N/A')}, head.nc (reale)={head_nc}")
    return model


# ---------------------------------------------------------------------------
# 2. Four-channel padding for single-modality inputs (VIS-only, IR-only)
# ---------------------------------------------------------------------------

class PaddedWiSARDYOLODataset(WiSARDYOLODataset):
    """Evaluation dataset with zero padding for the missing modality.

    VIS-only and IR-only samples use four channels, matching the RGB_ITEM
    and IR_ITEM handling in the Hugging Face WiSARDDataset pipeline.
    Paired VIS+IR samples already have four channels and remain unchanged.
    """

    def load_image(self, i, rect_mode=True):
        im, hw0, hw = super().load_image(i, rect_mode=rect_mode)
        c = im.shape[2] if im.ndim == 3 else 1
        if c == 3:  # VIS-only: pad the IR channel with zeros
            pad = np.zeros((*im.shape[:2], 1), dtype=im.dtype)
            im = np.concatenate([im, pad], axis=2)
        elif c == 1:  # IR-only: pad the RGB channels with zeros
            if im.ndim == 2:
                im = im[:, :, None]
            pad = np.zeros((*im.shape[:2], 3), dtype=im.dtype)
            im = np.concatenate([pad, im], axis=2)
        return im, hw0, hw


def build_dataloader(data_yaml, run_config, split, batch_size, workers):
    data_dict = check_det_dataset(data_yaml)
    img_path = data_dict[split]

    cfg_dict = cfg2dict(WISARD_DEFAULT_CFG)
    for k in ["imgsz", "cache", "single_cls", "task", "classes", "fraction"]:
        if k in run_config:
            cfg_dict[k] = run_config[k]
    cfg = IterableSimpleNamespace(**cfg_dict)

    dataset = PaddedWiSARDYOLODataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch_size,
        augment=False,
        hyp=cfg,
        rect=True,  # Match WisardTrainer.final_eval(): the test loader uses
                    # mode="val" in yolo.py, which sets rect=True for test evaluation too.
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=32,
        pad=0.5,  # Match build_yolo_dataset when mode != "train".
        prefix=colorstr(f"{split}: "),
        task=cfg.task,
        classes=cfg.classes,
        data=data_dict,
        fraction=1.0,
        augment_vis_ir=False,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=WiSARDYOLODataset.collate_fn,
    )
    return loader, dataset


# ---------------------------------------------------------------------------
# 3. NMS-free v10 postprocessing, following
#    ultralytics.models.yolov10.val.YOLOv10DetectionValidator.postprocess
# ---------------------------------------------------------------------------

def postprocess_v10(preds, max_det, nc):
    """Apply the YOLOv10 postprocessing used by WisardValidator.

    Box selection follows ultralytics/models/yolov10/val.py in the project
    environment, matching the evaluation used for the W&B metrics.
    """
    if isinstance(preds, dict):
        preds = preds["one2one"]
    if isinstance(preds, (list, tuple)):
        preds = preds[0]

    # Acknowledgement: Thanks to sanha9999 in #190 and #181! (original Ultralytics comment)
    if preds.shape[-1] == 6:
        return preds
    preds = preds.transpose(-1, -2)
    boxes, scores, labels = ops.v10postprocess(preds, max_det, nc)
    bboxes = ops.xywh2xyxy(boxes)
    return torch.cat([bboxes, scores.unsqueeze(-1), labels.unsqueeze(-1)], dim=-1)


# ---------------------------------------------------------------------------
# 4. Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(model, loader, device, modality, max_det=300):
    # Read nc from the detection head: the trainer can overwrite model.nc
    # with the dataset value even when the detector was built for a single class.
    nc = model.full_model[-1].nc
    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)

    n_images = 0
    n_gt_boxes = 0
    n_pred_boxes = 0

    with torch.no_grad():
        for batch in loader:
            img = batch["img"].to(device)
            img = img.float() / 255.0 if img.dtype == torch.uint8 else img.float()

            masks = {
                "fusion": (1.0, 1.0),
                "rgb": (1.0, 0.0),
                "ir": (0.0, 1.0),
            }
            modality_mask = img.new_tensor(masks[modality]).expand(img.shape[0], -1)
            raw_preds = model(img, modality_mask=modality_mask)
            preds = postprocess_v10(raw_preds, max_det=max_det, nc=nc)  # (B, max_det, 6) in inference-image coordinates

            imgsz = img.shape[2:]  # (H, W) of the letterboxed inference tensor
            batch_size = img.shape[0]

            for si in range(batch_size):
                n_images += 1

                # --- Targets: follow _prepare_batch ---
                idx = batch["batch_idx"] == si
                cls = batch["cls"][idx].squeeze(-1)
                bbox = batch["bboxes"][idx]
                ori_shape = batch["ori_shape"][si]
                ratio_pad = batch["ratio_pad"][si]

                if len(cls):
                    bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=bbox.device)[[1, 0, 1, 0]]
                    ops.scale_boxes(imgsz, bbox, ori_shape, ratio_pad=ratio_pad)
                    n_gt_boxes += len(cls)

                target = {
                    "boxes": bbox.to(device).float() if len(cls) else torch.zeros((0, 4), device=device),
                    "labels": cls.to(device).long() if len(cls) else torch.zeros((0,), dtype=torch.long, device=device),
                }

                # --- Predictions: follow _prepare_pred ---
                predn = preds[si].clone()
                ops.scale_boxes(imgsz, predn[:, :4], ori_shape, ratio_pad=ratio_pad)
                n_pred_boxes += predn.shape[0]

                pred = {
                    "boxes": predn[:, :4].to(device).float(),
                    "scores": predn[:, 4].to(device).float(),
                    "labels": predn[:, 5].to(device).long(),
                }

                metric.update(preds=[pred], target=[target])

    print(f"Immagini valutate: {n_images} | GT box totali: {n_gt_boxes} | pred box totali (pre-metrica): {n_pred_boxes}")
    results = metric.compute()
    return results


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="es. parameters/YOLO/30.yolov10-fam.yaml")
    parser.add_argument("--run-index", type=int, default=0)
    parser.add_argument("--checkpoint", required=True, help="path a weights/best.pt")
    parser.add_argument("--data-yaml", required=True, help="wisards_vis.yaml | wisards_ir.yaml | wisards_vis_ir.yaml")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument(
        "--modality",
        choices=["auto", "fusion", "rgb", "ir"],
        default="auto",
        help="Percorso feature da usare; auto lo deduce dal nome del data YAML.",
    )
    parser.add_argument("--out", default=None, help="path opzionale per salvare i risultati in JSON")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_config = load_yolo_run_config(args.config, run_index=args.run_index)
    print(f"Config run: {run_config.get('model')}")

    model = load_yolo_model(args.checkpoint, device)
    loader, dataset = build_dataloader(args.data_yaml, run_config, args.split, args.batch_size, args.workers)
    modality = args.modality
    if modality == "auto":
        data_name = Path(args.data_yaml).stem.lower()
        if "vis_ir" in data_name:
            modality = "fusion"
        elif data_name.endswith("_ir") or data_name == "ir":
            modality = "ir"
        else:
            modality = "rgb"
    print(
        f"Dataset: {args.data_yaml} | modalità: {modality} | "
        f"split: {args.split} | n. immagini: {len(dataset)}"
    )

    results = evaluate(
        model,
        loader,
        device,
        modality=modality,
        max_det=args.max_det,
    )

    print("\n=== Risultati ===")
    for k in ["map", "map_50", "map_75", "map_small", "map_medium", "map_large", "mar_1", "mar_10", "mar_100"]:
        if k in results:
            v = results[k]
            v = v.item() if torch.is_tensor(v) and v.dim() == 0 else v
            print(f"  {k:12s}: {v}")
    if "map_per_class" in results:
        print(f"  map_per_class: {results['map_per_class']}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {
            k: (v.tolist() if torch.is_tensor(v) else v) for k, v in results.items()
        }
        with out_path.open("w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\nSalvato: {out_path}")


if __name__ == "__main__":
    main()
