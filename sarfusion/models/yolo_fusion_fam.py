"""
YOLOv10-Fusion-FAM: Dual RGB+IR backbone with FeatureAlignmentModule.

Architecture:
    Input (4ch: RGB[0:3] + IR[3])
           ↓                    ↓
    RGB Backbone            IR Backbone
    (YOLOv10-s, 3ch)        (YOLOv10-s, 1ch, adapted)
           ↓                    ↓
      feat[4]  (P3)          feat[4]  (P3)
      feat[6]  (P4)          feat[6]  (P4)
      feat[10] (P5)          feat[10] (P5)
           │         FAM          │
           └──→ ir_aligned ←──────┘
                    ↓
        Fused = rgb + ir_aligned  (additive fusion)
                    ↓
             YOLO Neck (FPN+PAN)
                    ↓
             v10Detect Head (one2one + one2many)

Compatible with ultralytics trainer: forward() accepts batch dicts
and delegates to loss() for training, predict() for inference.
"""

import torch
import torch.nn as nn
from copy import deepcopy

from ultralytics.utils.loss import v10DetectLoss
from ultralytics.utils import IterableSimpleNamespace
from sarfusion.models.utils import yaml_model_load
from sarfusion.models.parse import parse_model
from sarfusion.models.rtdetr_fusion import FeatureAlignmentModule


class YOLOv10FusionFAM(nn.Module):
    """
    YOLOv10 with dual RGB+IR backbone, FAM alignment, and additive fusion.

    Args:
        cfg_path: Path to YOLOv10 YAML config (e.g., 'cfg/yolov10-s.yaml')
        nc: Number of classes (overrides YAML default)
        use_fam: Enable FeatureAlignmentModule for IR→RGB alignment
        freeze_fam: Freeze FAM weights (for regularization experiments)
        ir_dropout_rate: Spatial dropout rate on IR features (0.0 = disabled)
        spatial_jitter_std: Gaussian noise std on FAM offsets during training (0.0 = disabled)
    """

    def __init__(
        self,
        cfg_path: str = "cfg/yolov10-s.yaml",
        nc: int = 1,
        use_fam: bool = True,
        freeze_fam: bool = False,
        ir_dropout_rate: float = 0.0,
        spatial_jitter_std: float = 0.0,
        pretrained: bool = True,
    ):
        super().__init__()

        # --- Load and configure YAML ---
        cfg = yaml_model_load(cfg_path)
        if nc is not None:
            cfg["nc"] = nc
        self.yaml = cfg
        self.nc = cfg["nc"]
        self.names = {i: f"{i}" for i in range(self.nc)}
        self.inplace = cfg.get("inplace", True)
        self.args = IterableSimpleNamespace(
            box=7.5, cls=0.5, dfl=1.5, label_smoothing=0.0,
        )
        self.task = "detect"

        # --- Build RGB model (ch=3) ---
        rgb_cfg = deepcopy(cfg)
        self.full_model, self.save = parse_model(rgb_cfg, ch=3, verbose=False)
        self.model = self.full_model  # compatibility alias for loss functions

        # --- Build IR backbone (first 11 layers, ch=1) ---
        ir_cfg = deepcopy(cfg)
        ir_model, _ = parse_model(ir_cfg, ch=1, verbose=False)
        self.ir_layers = nn.ModuleList(ir_model[:11])
        self._adapt_ir_first_conv()

        # --- Architecture split points ---
        self.num_backbone_layers = 11
        self.neck_start = 11
        self.feat_indices = [4, 6, 10]  # P3, P4, P5 connection points for neck

        # --- Discover channel counts at fusion points ---
        dummy_rgb = torch.zeros(1, 3, 256, 256)
        dummy_ir = torch.zeros(1, 1, 256, 256)
        with torch.no_grad():
            rgb_out = self._run_backbone(dummy_rgb)
            ir_out = self._run_ir_backbone(dummy_ir)
        self.feat_channels = [rgb_out[idx].shape[1] for idx in self.feat_indices]

        # --- FAM modules ---
        self.use_fam = use_fam
        if use_fam:
            self.fam_modules = nn.ModuleList([
                FeatureAlignmentModule(
                    ch,
                    freeze=freeze_fam,
                    spatial_jitter_std=spatial_jitter_std,
                )
                for ch in self.feat_channels
            ])
        else:
            self.fam_modules = nn.ModuleList([
                nn.Identity() for _ in self.feat_channels
            ])

        # --- IR dropout ---
        self.ir_dropout_rate = ir_dropout_rate
        if ir_dropout_rate > 0.0:
            self.ir_dropout = nn.Dropout2d(ir_dropout_rate)
        else:
            self.ir_dropout = None

        # --- Load COCO pretrained weights ---
        if pretrained:
            self.load_coco_pretrained()

        # --- Compute stride using full model forward ---
        self._compute_stride()

        # --- Loss criterion (set lazily by trainer) ---
        self._criterion = None

    # ------------------------------------------------------------------
    #  Initialization helpers
    # ------------------------------------------------------------------

    def _adapt_ir_first_conv(self):
        """
        Shrink IR backbone's first Conv from 3ch to 1ch in-place
        so the state-dict shape matches the RGB→IR weights that
        load_coco_pretrained() copies over.
        parse_model(ir_cfg, ch=1) already creates a 1ch Conv when
        building from scratch, but the pretrained weight remapping
        needs the module to report in_channels=1 *before* the copy.
        """
        for layer in self.ir_layers:
            for module in layer.modules():
                if isinstance(module, nn.Conv2d) and module.in_channels == 3:
                    module.in_channels = 1

    def _compute_stride(self):
        """Compute and set detection stride using standard YOLO forward."""
        s = 256
        head = self.full_model[-1]
        head.inplace = self.inplace
        with torch.no_grad():
            y = []
            x = torch.zeros(1, 3, s, s)
            for m in self.full_model:
                if m.f != -1:
                    if isinstance(m.f, int):
                        x = y[m.f]
                    else:
                        x = [x if j == -1 else y[j] for j in m.f]
                x = m(x)
                y.append(x if m.i in self.save else None)
            one2many = x["one2many"]
        head.stride = torch.tensor([s / feat.shape[-2] for feat in one2many])
        self.stride = head.stride
        head.bias_init()

    # ------------------------------------------------------------------
    #  Forward logic
    # ------------------------------------------------------------------

    def _run_backbone(self, x: torch.Tensor) -> list:
        """
        Run the RGB backbone (self.full_model layers 0..10).
        Returns list of outputs indexed by layer position.
        """
        y = []
        for i in range(self.num_backbone_layers):
            m = self.full_model[i]
            if m.f != -1:
                if isinstance(m.f, int):
                    x = y[m.f]
                else:
                    x = [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        return y

    def _run_ir_backbone(self, x: torch.Tensor) -> list:
        """
        Run the IR backbone (self.ir_layers).
        Returns list of outputs indexed by layer position.
        """
        y = []
        for i, m in enumerate(self.ir_layers):
            if m.f != -1:
                if isinstance(m.f, int):
                    x = y[m.f]
                else:
                    x = [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        return y

    def _forward_single(self, x: torch.Tensor, _ir) -> torch.Tensor:
        """Single-modality forward (3ch) — used during AutoBackend warmup."""
        return self._run_standard_forward(x)

    def _run_standard_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run standard YOLO forward through the full model."""
        y = []
        for m in self.full_model:
            if m.f != -1:
                if isinstance(m.f, int):
                    x = y[m.f]
                else:
                    x = [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        return x

    def forward(self, x, *args, **kwargs):
        """Forward pass. Accepts 4ch tensor [B,4,H,W] or batch dict from trainer.
        Extra kwargs (augment, etc.) from ultralytics validator are ignored."""
        if isinstance(x, dict):
            return self.loss(x)
        return self.predict(x)

    def predict(self, x: torch.Tensor):
        """
        Forward pass with dual backbone, FAM, and additive fusion.

        Args:
            x: [B, C, H, W] — C=4 (RGB ch0-2 + IR ch3) for fusion,
                C=3 for warmup/standard single-modality inference

        Returns:
            YOLOv10 detection output (dict with one2one + one2many during
            training, postprocessed tensor during inference)
        """
        if x.shape[1] == 3:
            return self._forward_single(x, None)

        rgb = x[:, :3].contiguous()
        ir = x[:, 3:4].contiguous()

        rgb_feats = self._run_backbone(rgb)
        ir_feats = self._run_ir_backbone(ir)

        fused = {}
        for i, lvl_idx in enumerate(self.feat_indices):
            r = rgb_feats[lvl_idx]
            i_f = ir_feats[lvl_idx]
            if self.use_fam:
                i_f = self.fam_modules[i](r, i_f)
            if self.ir_dropout is not None and self.training:
                i_f = self.ir_dropout(i_f)
            fused[lvl_idx] = r + i_f

        total = len(self.full_model)
        y = [None] * total
        for idx in range(self.num_backbone_layers):
            y[idx] = fused.get(idx, rgb_feats[idx])

        prev = y[10]
        for i in range(self.neck_start, total):
            m = self.full_model[i]
            if m.f != -1:
                if isinstance(m.f, int):
                    prev = y[m.f]
                else:
                    prev = [prev if j == -1 else y[j] for j in m.f]
            prev = m(prev)
            y[m.i] = prev if m.i in self.save else None

        return prev

    # ------------------------------------------------------------------
    #  Pretrained weight loading
    # ------------------------------------------------------------------

    def load_coco_pretrained(self):
        """
        Load COCO-pretrained weights from jameslahm/yolov10s into the model.

        - RGB backbone (full_model layers 0-10): direct copy
        - Neck+head (full_model layers 11+): direct copy
        - IR backbone (ir_layers): copy RGB backbone weights, adapting
          the first Conv from 3ch to 1ch via channel-wise mean
        - FAM modules: left at zero initialization (identity mapping)
        """
        from sarfusion.models.yolov10 import YOLOv10WiSARD

        pretrained = YOLOv10WiSARD.from_pretrained("jameslahm/yolov10s")
        ckpt = pretrained.model.state_dict()
        our = self.state_dict()

        remap = {}

        # RGB backbone and full neck+head: model.k → full_model.k
        for ckpt_k, v in ckpt.items():
            if ckpt_k.startswith("model."):
                our_k = "full_model." + ckpt_k[len("model."):]
                if our_k in our and v.shape == our[our_k].shape:
                    remap[our_k] = v

        # IR backbone: same weights as RGB backbone, adapt first conv
        for ckpt_k, v in ckpt.items():
            if not ckpt_k.startswith("model."):
                continue
            ir_k = "ir_layers." + ckpt_k[len("model."):]
            if ir_k not in our:
                continue
            # First layer conv: adapt 3ch → 1ch
            if ir_k == "ir_layers.0.conv.weight" and v.dim() == 4 and v.shape[1] == 3:
                v = v.mean(dim=1, keepdim=True)
            if v.shape == our[ir_k].shape:
                remap[ir_k] = v

        missing, unexpected = [], []
        for k in our:
            if k in remap:
                continue
            if k.startswith("fam_modules."):
                continue  # FAM left at zero init — expected
            if k.startswith("full_model.0.bn.num_batches_tracked"):
                continue  # num_batches_tracked is non-persistent, expected missing
            if k.startswith("ir_layers.") and "num_batches_tracked" in k:
                continue
            missing.append(k)

        for k in remap:
            if k not in our:
                unexpected.append(k)

        self.load_state_dict(remap, strict=False)
        matched = len(remap)
        print(
            f"✅ COCO pretrained loaded: {matched} keys, "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )
        if missing:
            print(f"   First 5 missing: {missing[:5]}")

    # ------------------------------------------------------------------
    #  Trainer compatibility
    # ------------------------------------------------------------------

    def init_criterion(self):
        """Initialize v10DetectLoss."""
        return v10DetectLoss(self)

    def loss(self, batch, preds=None):
        """Compute loss from batch dict. Compatible with ultralytics trainer."""
        if preds is None:
            preds = self.predict(batch["img"])
        if not hasattr(self, "criterion") or self.criterion is None:
            self.criterion = self.init_criterion()
        return self.criterion(preds, batch)
