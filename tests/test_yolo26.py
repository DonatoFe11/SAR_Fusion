from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

ultralytics = pytest.importorskip("ultralytics")
if ultralytics.__version__ != "8.4.138":
    pytest.skip("YOLO26 tests require the isolated 8.4.138 environment", allow_module_level=True)

from ultralytics.cfg import get_cfg
from torchvision.transforms import functional as tvf

from sarfusion.yolo26.data import (
    PairedWiSARDYOLODataset,
    RGBTFormat,
    discover_split_records,
)
from sarfusion.yolo26.protocol import build_fusion_model, load_pretrained_model


REPOSITORY = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def pretrained():
    path = REPOSITORY / "yolo26s.pt"
    if not path.is_file():
        pytest.skip("official yolo26s.pt is not present")
    return load_pretrained_model(path)[0]


def _flatten(value):
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        return sum((_flatten(value[key]) for key in sorted(value)), [])
    if isinstance(value, (tuple, list)):
        return sum((_flatten(item) for item in value), [])
    return []


def test_rgbt_format_channel_order():
    image = np.zeros((3, 4, 4), dtype=np.uint8)
    image[:] = (10, 20, 30, 40)
    output = RGBTFormat(bgr=0.0)._format_img(image)
    assert output[:, 0, 0].tolist() == [30, 20, 10, 40]


def test_historical_ir_adaptation_parity():
    generator = np.random.default_rng(40)
    rgb = generator.integers(0, 256, size=(96, 192, 3), dtype=np.uint8)
    ir_bgr = generator.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)

    # Literal equivalent of the historical adapt_ir2rgb() call followed by
    # ``im_ir[:1]`` in the legacy WiSARD loader.
    legacy = torch.tensor(ir_bgr).permute(2, 0, 1)
    resized_width = int(64 * (96 / 48))
    legacy = tvf.resize(legacy, (96, resized_width))
    padding = (192 - resized_width) // 2
    legacy = tvf.pad(legacy, (padding, 0, padding, 0))[:1]

    current = PairedWiSARDYOLODataset.adapt_ir_to_rgb_canvas(
        rgb,
        ir_bgr[:, :, 0],
    )
    assert torch.equal(torch.from_numpy(current), legacy.squeeze(0))


def test_stage_a_pair_inventory_counts():
    split = yaml.safe_load(
        (REPOSITORY / "parameters/YOLO26/stage_a_split.yaml").read_text()
    )
    root = REPOSITORY / split["root"]
    if not root.is_dir():
        pytest.skip("WiSARD dataset is not present")
    train, train_dropped = discover_split_records(root, "train", split["train"]["pairs"])
    val, val_dropped = discover_split_records(root, "val", split["val"]["pairs"])
    assert len(train) == 3123
    assert len(val) == 896
    assert train_dropped[0]["rgb_only_frame_indices"] == [943, 944]
    assert val_dropped[0]["rgb_only_frame_indices"] == [896]
    assert not ({record.rgb for record in train} & {record.rgb for record in val})


def test_matched_initialization_and_rgb_only_parity(pretrained):
    control = build_fusion_model(
        pretrained, seed=40, use_fam=False, deterministic=True
    )
    candidate = build_fusion_model(
        pretrained, seed=40, use_fam=True, deterministic=True
    )
    for scope in ("shared", "ir", "fam"):
        assert control.state_sha256(scope) == candidate.state_sha256(scope)
    assert [module.deform_conv.in_channels for module in candidate.fam_modules] == [
        256,
        256,
        512,
    ]

    control.eval()
    rgb = torch.rand(2, 3, 128, 128)
    rgbt = torch.cat((rgb, torch.zeros(2, 1, 128, 128)), 1)
    with torch.no_grad():
        standard = _flatten(control.predict(rgb))
        wrapped = _flatten(
            control.predict(rgbt, modality_mask=torch.tensor(((1, 0), (1, 0))))
        )
    assert len(standard) == len(wrapped)
    assert max(float((a - b).abs().max()) for a, b in zip(standard, wrapped)) == 0.0


def test_native_yolo26_loss_updates_every_fam(pretrained):
    model = build_fusion_model(pretrained, seed=40, use_fam=True, deterministic=True)
    config = yaml.safe_load(
        (
            REPOSITORY
            / "parameters/YOLO26/yolo26s_fam_seed40_stage_a.yaml"
        ).read_text()
    )
    model.args = get_cfg(overrides=config["training"])
    model.train()
    batch = {
        "img": torch.rand(2, 4, 128, 128),
        "modality_mask": torch.ones(2, 2),
        "batch_idx": torch.tensor((0, 1)),
        "cls": torch.zeros(2, 1),
        "bboxes": torch.tensor(((0.5, 0.5, 0.2, 0.2), (0.5, 0.5, 0.2, 0.2))),
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = [next(module.parameters()).detach().clone() for module in model.fam_modules]
    loss, items = model(batch)
    assert set(items) == {"box_loss", "cls_loss", "l1_loss"}
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    for module in model.fam_modules:
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        assert gradients
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
    optimizer.step()
    assert all(
        not torch.equal(next(module.parameters()).detach(), old)
        for module, old in zip(model.fam_modules, before)
    )
