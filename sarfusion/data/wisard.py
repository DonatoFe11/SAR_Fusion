import glob
import hashlib
from itertools import repeat
import math
import os
from pathlib import Path
import random
from PIL import Image, ImageOps
import cv2
import numpy as np
from torch.utils.data import Dataset
from multiprocessing.pool import ThreadPool
import torch
import torchvision.transforms.functional as tvF

from ultralytics.utils import LOCAL_RANK, NUM_THREADS, TQDM, colorstr, is_dir_writeable
from ultralytics.utils.ops import segments2boxes
from ultralytics.data.augment import (
    Compose,
    LetterBox,
)
from ultralytics.data.base import BaseDataset
from ultralytics.data.utils import (
    HELP_URL,
    LOGGER,
    get_hash,
    img2label_paths,
    verify_image,
    verify_image_label,
    exif_size,
)
from ultralytics.data.dataset import (
    DATASET_CACHE_VERSION,
    load_dataset_cache_file,
    save_dataset_cache_file,
)
from ultralytics.utils import LOCAL_RANK, NUM_THREADS, TQDM, colorstr, is_dir_writeable

from sarfusion.data.transforms import wisard_transforms
from sarfusion.data.utils import (
    collate_images,
    dict_collate_fn,
    load_annotations,
    process_image_annotation_folders,
    yolo_to_coco_annotations,
)
from sarfusion.data.transforms import Format
from sarfusion.utils.dataloaders import HELP_URL, IMG_FORMATS
from sarfusion.utils.general import xywhn2xyxy, xyxy2xywhn
from sarfusion.utils.transforms import letterbox
from sarfusion.utils.structures import DataDict
from sarfusion.utils.transforms import ResizePadKeepRatio
from sarfusion.data.yolo import YOLODataset


NO_LABELS = [
    "210924_FHL_Enterprise_VIS_0564",
    "210924_FHL_Enterprise_VIS_0566",
    "210812_Hannegan_Enterprise_VIS_0055",
    "210812_Hannegan_Enterprise_IR_0056",
    "210924_FHL_Enterprise_VIS_0409",
    "210924_FHL_Enterprise_IR_0410",
    "210812_Hannegan_Enterprise_VIS_0053",
    "210812_Hannegan_Enterprise_IR_0054",
    "210924_FHL_Enterprise_VIS_0403",
    "210529_Carnation_Enterprise_IR_0026",
    "210924_FHL_Enterprise_IR_0408",
    "210924_FHL_Enterprise_IR_0127",
]

ONLY_VIS_LABELS = [
    "210529_Carnation_Enterprise_IR_0026",
    "210924_FHL_Enterprise_IR_0127",
    "210924_FHL_Enterprise_IR_0408",
]

ONLY_IR_LABELS = [
    "210924_FHL_Enterprise_VIS_0564",
    "210924_FHL_Enterprise_VIS_0566",
]

VIS_ONLY = [
    "200321_Baker_Phantom_VIS",
    "200402_Carnation_Inspire_VIS",
    "200402_Karen_Inspire_VIS",
    "200426_SkookumCreek_Mavic_Mini_VIS_0006",
    "200426_SkookumCreek_Mavic_Mini_VIS_0007",
    "200426_SkookumCreek_Mavic_Mini_VIS_0008",
    "200505_Bellingham_Mavic_Mini_VIS_0015",
    "200505_Bellingham_Mavic_Mini_VIS_0016",
    "200528_Everson_Mavic_Mini_VIS_0021",
    "200528_Everson_Mavic_Mini_VIS_0028",
    "200528_Everson_Mavic_Mini_VIS_0029",
    "200528_Everson_Mavic_Mini_VIS_0031",
    "200614_SuddenValley_Phantom_VIS_0005",
    "200614_SuddenValley_Phantom_VIS_0006",
    "200614_SuddenValley_Phantom_VIS_0012",
    "200614_SuddenValley_Phantom_VIS_0013",
    "200614_SuddenValley_Phantom_VIS_0014",
    "200717_Mission_FLIR_VIS",
    "210327_Airfield_FLIR_VIS_1",
    "210327_Airfield_FLIR_VIS_2",
    "210327_Airfield_FLIR_VIS_3",
    "210327_Airfield_FLIR_VIS_4",
]

IR_ONLY = [
    "200704_Baker_FLIR_IR_1",
    "200704_Baker_FLIR_IR_2",
    "200910_Carnation_FLIR_IR_1",
    "200910_Carnation_FLIR_IR_2",
    "200910_Carnation_FLIR_IR_3",
    "200910_Carnation_FLIR_IR_4",
    "200910_Carnation_FLIR_IR_5",
    "200910_Carnation_FLIR_IR_6",
    "200910_Carnation_FLIR_IR_7",
    "200929_Karen_FLIR_IR_1",
    "200929_Karen_FLIR_IR_2",
    "200929_Karen_FLIR_IR_3",
    "200929_Karen_FLIR_IR_4",
    "200929_Karen_FLIR_IR_5",
    "200929_Karen_FLIR_IR_6",
    "220109_Baker_Enterprise_IR_2",
    "210327_Airfield_FLIR_IR_1",
    "210327_Airfield_FLIR_IR_2",
    "210327_Airfield_FLIR_IR_3",
    "210327_Airfield_FLIR_IR_4",
    "210327_Airfield_FLIR_IR_5",
    "210327_Airfield_FLIR_IR_6",
    "210327_Airfield_FLIR_IR_7",
    "210327_Airfield_FLIR_IR_8",
]

TOTAL_VIS_IR = [
    (
        "210417_MtErie_Enterprise_VIS_0003",
        "210417_MtErie_Enterprise_IR_0004",
    ),  # Synced (test)
    (
        "210417_MtErie_Enterprise_VIS_0005",
        "210417_MtErie_Enterprise_IR_0006",
    ),  # Synced (test)
    (
        "210417_MtErie_Enterprise_VIS_0007",
        "210417_MtErie_Enterprise_IR_0008",
    ),  # Synced (test)
    (
        "210529_Carnation_Enterprise_VIS_0023",
        "210529_Carnation_Enterprise_IR_0024",
    ),  # RGB is zoomed in (val)
    (
        "210529_Carnation_Enterprise_VIS_0025",
        "210529_Carnation_Enterprise_IR_0026",
    ),  # Synced (val)
    ("210812_Hannegan_Enterprise_VIS_0053", "210812_Hannegan_Enterprise_IR_0054"),
    ("210812_Hannegan_Enterprise_VIS_0055", "210812_Hannegan_Enterprise_IR_0056"),
    (
        "210924_FHL_Enterprise_VIS_0126",
        "210924_FHL_Enterprise_IR_0127",
    ),  # Synced (train)
    (
        "210924_FHL_Enterprise_VIS_0134",
        "210924_FHL_Enterprise_IR_0135",
    ),  # Synced (train)
    (
        "210924_FHL_Enterprise_VIS_0401",
        "210924_FHL_Enterprise_IR_0402",
    ),  # Synced (train)
    ("210924_FHL_Enterprise_VIS_0403", "210924_FHL_Enterprise_IR_0404"),
    (
        "210924_FHL_Enterprise_VIS_0405",
        "210924_FHL_Enterprise_IR_0406",
    ),  # Synced (train)
    (
        "210924_FHL_Enterprise_VIS_0407",
        "210924_FHL_Enterprise_IR_0408",
    ),  # Synced (train)
    ("210924_FHL_Enterprise_VIS_0409", "210924_FHL_Enterprise_IR_0410"),
    ("210924_FHL_Enterprise_VIS_0564", "210924_FHL_Enterprise_IR_0565"),
    ("210924_FHL_Enterprise_VIS_0566", "210924_FHL_Enterprise_IR_0567"),
    ("220109_Baker_Enterprise_VIS_1", "220109_Baker_Enterprise_IR_1"),  # Synced (train)
]
VIS = VIS_ONLY + [f[0] for f in TOTAL_VIS_IR]
IR = IR_ONLY + [f[1] for f in TOTAL_VIS_IR]

MISSING_ANNOTATIONS = [
    "210924_FHL_Enterprise_VIS_0126/labels/DJI_0126_00000211.txt",
    "210924_FHL_Enterprise_VIS_0126/labels/DJI_0126_00000212.txt",
]

TRAIN_VIS = [1, 2, 3, 4, 5, 8, 9, 10, 11, 12]
VAL_VIS = [0, 6, 7, 13, 14]
TEST_VIS = [15, 16, 17, 18, 19, 20, 21]

TRAIN_VIS_IR = [7, 9, 10, 11, 12, 13, 14, 15, 16]
VAL_VIS_IR = [
    # 3, RGB is zoomed in
    4, 
    5,
    6,
    8,
]
TEST_VIS_IR = [0, 1, 2]

# Remove NO_LABELS from VIS_IR
LABELED_VIS_IR = [
    (f[0], f[1]) for f in TOTAL_VIS_IR if f[0] not in NO_LABELS and f[1] not in NO_LABELS
]
TRAIN_VIS_IR = [
    LABELED_VIS_IR.index(TOTAL_VIS_IR[f])
    for f in TRAIN_VIS_IR
    if TOTAL_VIS_IR[f] in LABELED_VIS_IR
]
VAL_VIS_IR = [
    LABELED_VIS_IR.index(TOTAL_VIS_IR[f])
    for f in VAL_VIS_IR
    if TOTAL_VIS_IR[f] in LABELED_VIS_IR
]
TEST_VIS_IR = [
    LABELED_VIS_IR.index(TOTAL_VIS_IR[f])
    for f in TEST_VIS_IR
    if TOTAL_VIS_IR[f] in LABELED_VIS_IR
]
VIS_IR = LABELED_VIS_IR

TRAIN_IR = [9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23]
VAL_IR = [2, 3, 4, 5, 6, 7, 8]
TEST_IR = [0, 1, 15]

TRAIN_FOLDERS = (
    [VIS_ONLY[i] for i in TRAIN_VIS]
    + [IR_ONLY[i] for i in TRAIN_IR]
    + [VIS_IR[i] for i in TRAIN_VIS_IR]
    + [VIS_IR[i][0] for i in TRAIN_VIS_IR]
    + [VIS_IR[i][1] for i in TRAIN_VIS_IR]
)
VAL_FOLDERS = (
    [VIS[i] for i in VAL_VIS]
    + [IR[i] for i in VAL_IR]
    + [VIS_IR[i] for i in VAL_VIS_IR]
    + [VIS_IR[i][0] for i in VAL_VIS_IR]
    + [VIS_IR[i][1] for i in VAL_VIS_IR]
)
TEST_FOLDERS = (
    [VIS[i] for i in TEST_VIS]
    + [IR[i] for i in TEST_IR]
    + [VIS_IR[i] for i in TEST_VIS_IR]
    + [VIS_IR[i][0] for i in TEST_VIS_IR]
    + [VIS_IR[i][1] for i in TEST_VIS_IR]
)

RGB_ITEM = 0
IR_ITEM = 1
MULTI_MODALITY_ITEM = 2


def adapt_ir2rgb(rgb, ir):
    if isinstance(rgb, np.ndarray):
        rgb = torch.from_numpy(rgb)
        ir = torch.from_numpy(ir)
    elif isinstance(rgb, Image.Image):
        rgb = tvF.pil_to_tensor(rgb)
        ir = tvF.pil_to_tensor(ir)
    elif isinstance(rgb, torch.Tensor):
        pass
    else:
        raise NotImplementedError(f"Unknown type {type(rgb)}")
    
    # ir = ir[0:1]  # All channels are the same
    # calculate h, w displacement
    rgb_h, rgb_w = rgb.shape[1:]
    ir_h, ir_w = ir.shape[1:]

    new_ir_h = rgb_h
    new_ir_w = int(ir_w * (rgb_h / ir_h))

    new_ir = tvF.resize(ir, (new_ir_h, new_ir_w))
    w_pad = (rgb_w - new_ir_w) // 2
    new_ir = tvF.pad(new_ir, (w_pad, 0, w_pad, 0))

    return rgb, new_ir


def get_wisard_folders(folders):
    if folders == "vis":
        folders = VIS
    elif folders == "ir":
        folders = IR
    elif folders == "vis_all_ir_sync":
        folders = VIS_ONLY + VIS_IR
    elif folders == "vis_sync_ir_all":
        folders = IR_ONLY + VIS_IR
    elif folders == "vis_ir":
        folders = VIS_IR
    elif folders == "all":
        folders = VIS_ONLY + IR_ONLY + VIS_IR
    elif folders == "vis_only":
        folders = VIS_ONLY
    elif folders == "ir_only":
        folders = IR_ONLY
    return folders


def build_wisard_items(root, folders):
    folders = get_wisard_folders(folders)
    rgb_datasets = list(filter(lambda x: x in VIS, folders))
    ir_datasets = list(filter(lambda x: x in IR, folders))
    multi_modality_datasets = list(filter(lambda x: isinstance(x, tuple), folders))
    rgb_items = [
        list(zip(*process_image_annotation_folders(os.path.join(root, folder))))
        for folder in rgb_datasets
    ]
    rgb_items = [(RGB_ITEM, item) for dataset in rgb_items for item in dataset]
    ir_items = [
        list(zip(*process_image_annotation_folders(os.path.join(root, folder))))
        for folder in ir_datasets
    ]
    ir_items = [(IR_ITEM, item) for dataset in ir_items for item in dataset]
    # multi_modality_rgb_items = [
    #     list(zip(*process_image_annotation_folders(os.path.join(root, folder[0]))))
    #     for folder in multi_modality_datasets
    # ]
    multi_modality_rgb_items = []
    for folder in multi_modality_datasets:
        image_paths, annotations_paths = process_image_annotation_folders(
            os.path.join(root, folder[0])
        )
        multi_modality_rgb_items.append(list(zip(*(image_paths, annotations_paths))))

    # multi_modality_ir_items = [
    #     list(zip(*process_image_annotation_folders(os.path.join(root, folder[1]))))
    #     for folder in multi_modality_datasets
    # ]
    multi_modality_ir_items = []
    for folder in multi_modality_datasets:
        image_paths, annotations_paths = process_image_annotation_folders(
            os.path.join(root, folder[1])
        )
        if len(annotations_paths) == 0:
            annotations_paths = [""] * len(image_paths)
        multi_modality_ir_items.append(list(zip(*(image_paths, annotations_paths))))
    # multi_modality_items = [
    #     (MULTI_MODALITY_ITEM, (rgb_item, ir_item))
    #     for rgb_dataset, ir_dataset in zip(
    #         multi_modality_rgb_items, multi_modality_ir_items
    #     )
    #     for rgb_item, ir_item in zip(rgb_dataset, ir_dataset)
    # ]
    multi_modality_items = []
    for rgb_dataset, ir_dataset in zip(
        multi_modality_rgb_items, multi_modality_ir_items
    ):
        for rgb_item, ir_item in zip(rgb_dataset, ir_dataset):
            multi_modality_items.append((MULTI_MODALITY_ITEM, (rgb_item, ir_item)))

    return rgb_items + ir_items + multi_modality_items


class WiSARDDataset(Dataset):

    id2class = {
        0: "running",
        1: "walking",
        2: "laying_down",
        3: "not_defined",
        4: "seated",
        5: "stands",
    }

    def __init__(
        self,
        root,
        folders,
        transform=None,
        ir_transform=None,
        return_path=False,
        augment=False,
        image_size=640,
        single_class=False,
        modal_dropout=False,
        modal_dropout_probs=None
    ):
        self.items = build_wisard_items(root, folders)
        if not self.items:
            raise ValueError("No items found in dataset")
        self.transform = transform
        self.ir_transform = ir_transform or transform
        self.image_size = image_size
        self.augment = augment
        self.return_path = return_path
        self.single_class = single_class
        self.modal_dropout = modal_dropout
        self.modal_dropout_probs = modal_dropout_probs or [0.2, 0.2, 0.6]  # IR, RGB, Fusion
        if single_class:
            self.id2class = {0: "person"}

    def __len__(self):
        return len(self.items)

    def _load_rgb(self, img_path):
        img = Image.open(img_path).convert("RGB")
        return img

    def _load_ir(self, img_path):
        img = Image.open(img_path).convert("RGB")
        return img
    
    def _load_annotations(self, annotation_path, img, img_id):
        targets = load_annotations(annotation_path)
        if self.single_class:
            for target in targets:
                target[0] = 0
        targets = yolo_to_coco_annotations(bboxes=targets, image_id=img_id, img_width=img.size[0], img_height=img.size[1])
        return targets

    def __getitem__(self, idx):
        item_type, item = self.items[idx]

        if item_type == RGB_ITEM:
            img_path, annotation_path = item
            img = self._load_rgb(img_path)
            targets = self._load_annotations(annotation_path, img, idx)
            inputs = self.transform(img, annotations=targets, return_tensors="pt")
            img = inputs['pixel_values'][0]
            targets = inputs['labels'][0]
            img_path_vis = img_path
        elif item_type == IR_ITEM:
            img_path, annotation_path = item
            img = self._load_ir(img_path)
            targets = self._load_annotations(annotation_path, img, idx)
            inputs = self.transform(img, annotations=targets, return_tensors="pt")
            targets = inputs['labels'][0]
            img = inputs['pixel_values'][0][0:1]
            img_path_vis = img_path
        else: # item_type == FUSION_ITEM
            (img_path_vis, annotation_path), (img_path_ir, annotation_path_ir) = item
            img_vis = self._load_rgb(img_path_vis)
            img_ir = self._load_ir(img_path_ir)

            targets_vis_raw = self._load_annotations(annotation_path, img_vis, idx)
            targets_ir_raw = self._load_annotations(annotation_path_ir, img_ir, idx)


            if self.modal_dropout:
                mode = random.choices(['ir', 'rgb', 'fusion'], weights=self.modal_dropout_probs, k=1)[0]
            else:
                mode = 'fusion'  # default behavior

            if mode == 'fusion':
                # adjust IR to match RGB size for fusion
                img_vis_processed, img_ir_processed = adapt_ir2rgb(img_vis, img_ir)

                inputs_vis = self.transform(img_vis_processed, annotations=targets_vis_raw, return_tensors="pt")
                img_vis_tensor = inputs_vis['pixel_values'][0]  # [3, H, W]
                targets_vis = inputs_vis['labels'][0]

                inputs_ir = self.transform(img_ir_processed, annotations=targets_ir_raw, return_tensors="pt")
                img_ir_tensor = inputs_ir['pixel_values'][0][0:1]  # [1, H, W]
                targets_ir = inputs_ir['labels'][0]

                img = torch.cat([img_vis_tensor, img_ir_tensor], dim=0)  # [4, H, W]
                targets = targets_vis

            elif mode == 'rgb':
                inputs_vis = self.transform(img_vis, annotations=targets_vis_raw, return_tensors="pt")
                img = inputs_vis['pixel_values'][0]  # [3, H, W]
                targets = inputs_vis['labels'][0]

            else:  # mode == 'ir'
                # print("DEBUG: Modal Dropout -> SOLO IR")
                img_path_vis = img_path_ir
                inputs_ir = self.transform(img_ir, annotations=targets_ir_raw, return_tensors="pt")
                img = inputs_ir['pixel_values'][0][0:1]  # [1, H, W]
                targets = inputs_ir['labels'][0]

        data_dict = DataDict(pixel_values=img, labels=dict(targets), dims=targets["orig_size"])
        if self.return_path:
            data_dict.path = img_path_vis

        return data_dict
    
    def collate_fn(self, batch):
        pixel_values = [item["pixel_values"] for item in batch]
        encoding = self.transform.pad(pixel_values, return_tensors="pt", input_data_format="channels_first") 
        labels = [item["labels"] for item in batch]
        dims = [item["dims"] for item in batch]

        batch = {}
        batch["pixel_values"] = encoding["pixel_values"]
        batch["pixel_mask"] = encoding["pixel_mask"]
        batch["labels"] = labels
        batch["dims"] = dims

        return batch

def img2label_paths(img_paths):
    """Define label paths as a function of image paths."""
    sa, sb = (
        f"{os.sep}images{os.sep}",
        f"{os.sep}labels{os.sep}",
    )  # /images/, /labels/ substrings
    labels_paths = []
    for img_path in img_paths:
        if isinstance(img_path, tuple):
            img_path = img_path[0]
        labels_paths.append(img_path.replace(sa, sb).rsplit(".", 1)[0] + ".txt")
    return labels_paths


def get_hash(paths):
    """Returns a single hash value of a list of paths (files or dirs)."""
    size = sum(
        (
            os.path.getsize(p)
            if isinstance(p, str)
            else os.path.getsize(p[0]) + os.path.getsize(p[1])
        )
        for p in paths
        if (
            isinstance(p, str)
            and os.path.exists(p)
            or (os.path.exists(p[0]) and os.path.exists(p[1]))
        )
    )  # sizes
    h = hashlib.sha256(str(size).encode())  # hash sizes
    hash_paths = []
    for p in paths:
        if isinstance(p, str):
            hash_paths.append(p)
        else:
            hash_paths.append(p[0])
            hash_paths.append(p[1])
    h.update("".join(hash_paths).encode())  # hash paths
    return h.hexdigest()  # return hash


def verify_image_label(args):
    """Verify one image-label pair."""
    im_files, lb_file, prefix, keypoint, num_cls, nkpt, ndim = args
    # Number (missing, found, empty, corrupt), message, segments, keypoints
    nm, nf, ne, nc, msg, segments, keypoints = 0, 0, 0, 0, "", [], None
    try:
        # Verify images
        if isinstance(im_files, str):
            im_files = (im_files,)
        for im_file in im_files:
            im = Image.open(im_file)
            im.verify()  # PIL verify
            shape = exif_size(im)  # image size
            shape = (shape[1], shape[0])  # hw
            assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
            assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}"
            if im.format.lower() in ("jpg", "jpeg"):
                with open(im_file, "rb") as f:
                    f.seek(-2, 2)
                    if f.read() != b"\xff\xd9":  # corrupt JPEG
                        ImageOps.exif_transpose(Image.open(im_file)).save(
                            im_file, "JPEG", subsampling=0, quality=100
                        )
                        msg = f"{prefix}WARNING ⚠️ {im_file}: corrupt JPEG restored and saved"
        if len(im_files) == 1:
            im_files = im_files[0]

        # Verify labels
        if os.path.isfile(lb_file):
            nf = 1  # label found
            with open(lb_file) as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                if any(len(x) > 6 for x in lb) and (not keypoint):  # is segment
                    classes = np.array([x[0] for x in lb], dtype=np.float32)
                    segments = [
                        np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb
                    ]  # (cls, xy1...)
                    lb = np.concatenate(
                        (classes.reshape(-1, 1), segments2boxes(segments)), 1
                    )  # (cls, xywh)
                lb = np.array(lb, dtype=np.float32)
            nl = len(lb)
            if nl:
                if keypoint:
                    assert lb.shape[1] == (
                        5 + nkpt * ndim
                    ), f"labels require {(5 + nkpt * ndim)} columns each"
                    points = lb[:, 5:].reshape(-1, ndim)[:, :2]
                else:
                    assert (
                        lb.shape[1] == 5
                    ), f"labels require 5 columns, {lb.shape[1]} columns detected"
                    points = lb[:, 1:]
                assert (
                    points.max() <= 1
                ), f"non-normalized or out of bounds coordinates {points[points > 1]}"
                assert lb.min() >= 0, f"negative label values {lb[lb < 0]}"

                # All labels
                max_cls = lb[:, 0].max()  # max label count
                assert max_cls <= num_cls, (
                    f"Label class {int(max_cls)} exceeds dataset class count {num_cls}. "
                    f"Possible class labels are 0-{num_cls - 1}"
                )
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:  # duplicate row check
                    lb = lb[i]  # remove duplicates
                    if segments:
                        segments = [segments[x] for x in i]
                    msg = f"{prefix}WARNING ⚠️ {im_files}: {nl - len(i)} duplicate labels removed"
            else:
                ne = 1  # label empty
                lb = np.zeros(
                    (0, (5 + nkpt * ndim) if keypoint else 5), dtype=np.float32
                )
        else:
            nm = 1  # label missing
            lb = np.zeros((0, (5 + nkpt * ndim) if keypoints else 5), dtype=np.float32)
        if keypoint:
            keypoints = lb[:, 5:].reshape(-1, nkpt, ndim)
            if ndim == 2:
                kpt_mask = np.where(
                    (keypoints[..., 0] < 0) | (keypoints[..., 1] < 0), 0.0, 1.0
                ).astype(np.float32)
                keypoints = np.concatenate(
                    [keypoints, kpt_mask[..., None]], axis=-1
                )  # (nl, nkpt, 3)
        lb = lb[:, :5]
        return im_files, lb, shape, segments, keypoints, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f"{prefix}WARNING ⚠️ {im_files}: ignoring corrupt image/label: {e}"
        return [None, None, None, None, None, nm, nf, ne, nc, msg]


class WiSARDYOLODataset(YOLODataset):
    def __init__(self, *args, **kwargs):
        self.augment_vis_ir = kwargs.pop("augment_vis_ir", False)
        super().__init__(*args, **kwargs)

    def get_labels(self):
        """Returns dictionary of labels for YOLO training."""
        self.label_files = img2label_paths(self.im_files)
        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")
        try:
            cache, exists = (
                load_dataset_cache_file(cache_path),
                True,
            )  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash(
                self.label_files + self.im_files
            )  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop(
            "results"
        )  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in (-1, 0):
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            TQDM(None, desc=self.prefix + d, total=n, initial=n)  # display results
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))  # display warnings

        # Read cache
        [cache.pop(k) for k in ("hash", "version", "msgs")]  # remove items
        labels = cache["labels"]
        if not labels:
            LOGGER.warning(
                f"WARNING ⚠️ No images found in {cache_path}, training may not work correctly. {HELP_URL}"
            )
        self.im_files = [lb["im_file"] for lb in labels]  # update im_files

        # Check if the dataset is all boxes or all segments
        lengths = (
            (len(lb["cls"]), len(lb["bboxes"]), len(lb["segments"])) for lb in labels
        )
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            LOGGER.warning(
                f"WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = {len_segments}, "
                f"len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. "
                "To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset."
            )
            for lb in labels:
                lb["segments"] = []
        if len_cls == 0:
            LOGGER.warning(
                f"WARNING ⚠️ No labels found in {cache_path}, training may not work correctly. {HELP_URL}"
            )
        return labels

    def get_img_files(self, img_path):
        """Read image files."""
        try:
            f = []  # image files
            for p in img_path if isinstance(img_path, list) else [img_path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / "**" / "*.*"), recursive=True)
                    # F = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [
                            tuple(x.split(",")) if "," in x else x for x in t
                        ]  # Image or image couple (VIS and IR)
                        # f += [x.replace("./", parent) if x.startswith("./") else x for x in t]  # local to global path
                        # F += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise FileNotFoundError(f"{self.prefix}{p} does not exist")
            im_files = []
            for x in f:
                if isinstance(x, tuple):
                    if (
                        x[0].split(".")[-1].lower() in IMG_FORMATS
                        and x[1].split(".")[-1].lower() in IMG_FORMATS
                    ):
                        im_files.append(
                            (x[0].replace("/", os.sep), x[1].replace("/", os.sep))
                        )
                    else:
                        LOGGER.warning(
                            f"WARNING ⚠️ Skipping image pair {x} with different formats"
                        )
                else:
                    if x.split(".")[-1].lower() in IMG_FORMATS:
                        im_files.append(x.replace("/", os.sep))
                    else:
                        LOGGER.warning(
                            f"WARNING ⚠️ Skipping image {x} with unsupported format"
                        )
            im_files = sorted(
                im_files, key=lambda x: x[0] if isinstance(x, tuple) else x
            )
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert im_files, f"{self.prefix}No images found in {img_path}"
        except Exception as e:
            raise FileNotFoundError(
                f"{self.prefix}Error loading data from {img_path}\n{HELP_URL}"
            ) from e
        if self.fraction < 1:
            # im_files = im_files[: round(len(im_files) * self.fraction)]
            num_elements_to_select = round(len(im_files) * self.fraction)
            im_files = random.sample(im_files, num_elements_to_select)
        return im_files

    def cache_labels(self, path=Path("./labels.cache")):
        """
        Cache dataset labels, check images and read shapes.

        Args:
            path (Path): Path where to save the cache file. Default is Path('./labels.cache').

        Returns:
            (dict): labels.
        """
        x = {"labels": []}
        nm, nf, ne, nc, msgs = (
            0,
            0,
            0,
            0,
            [],
        )  # number missing, found, empty, corrupt, messages
        desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
        total = len(self.im_files)
        nkpt, ndim = self.data.get("kpt_shape", (0, 0))
        if self.use_keypoints and (nkpt <= 0 or ndim not in (2, 3)):
            raise ValueError(
                "'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'"
            )
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(
                func=verify_image_label,
                iterable=zip(
                    self.im_files,
                    self.label_files,
                    repeat(self.prefix),
                    repeat(self.use_keypoints),
                    repeat(len(self.data["names"])),
                    repeat(nkpt),
                    repeat(ndim),
                ),
            )
            pbar = TQDM(results, desc=desc, total=total)
            for (
                im_file,
                lb,
                shape,
                segments,
                keypoint,
                nm_f,
                nf_f,
                ne_f,
                nc_f,
                msg,
            ) in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x["labels"].append(
                        dict(
                            im_file=im_file,
                            shape=shape,
                            cls=lb[:, 0:1],  # n, 1
                            bboxes=lb[:, 1:],  # n, 4
                            segments=segments,
                            keypoints=keypoint,
                            normalized=True,
                            bbox_format="xywh",
                        )
                    )
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            pbar.close()

        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(
                f"{self.prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}"
            )
        x["hash"] = get_hash(self.label_files + self.im_files)
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        x["msgs"] = msgs  # warnings
        save_dataset_cache_file(self.prefix, path, x)
        return x

    def load_image(self, i, rect_mode=True):
        """Loads 1 image from dataset index 'i', returns (im, resized hw)."""
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
        if self.augment_vis_ir and isinstance(f, tuple):
            im = None
            fn = None
            choice = np.random.choice([0, 1, 2], p=[0.25, 0.25, 0.5])
            if choice in [0, 1]:
                f = f[choice]

        if im is None:  # not cached in RAM
            if isinstance(im, Path) and fn.exists():  # load npy
                try:
                    im = np.load(fn)
                except Exception as e:
                    LOGGER.warning(
                        f"{self.prefix}WARNING ⚠️ Removing corrupt *.npy image file {fn} due to: {e}"
                    )
                    Path(fn).unlink(missing_ok=True)
                    im = cv2.imread(f)  # BGR
            else:  # read image
                if isinstance(f, (Path, str)):
                    im = cv2.imread(f)  # BGR
                    if "IR" in f.split(os.sep)[-3].split("_"):  # is IR image
                        im = im[:, :, :1]  # only use first channel
                else:
                    im_vis = torch.tensor(cv2.imread(f[0])).permute(2, 0, 1)  # BGR
                    im_ir = torch.tensor(cv2.imread(f[1])).permute(2, 0, 1)  # IR
                    im = adapt_ir2rgb(im_vis, im_ir).permute(1, 2, 0).numpy()
            if im is None:
                raise FileNotFoundError(f"Image Not Found {f}")

            h0, w0 = im.shape[:2]  # orig hw
            if rect_mode:  # resize long side to imgsz while maintaining aspect ratio
                r = self.imgsz / max(h0, w0)  # ratio
                if r != 1:  # if sizes are not equal
                    w, h = (
                        min(math.ceil(w0 * r), self.imgsz),
                        min(math.ceil(h0 * r), self.imgsz),
                    )
                    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            elif not (
                h0 == w0 == self.imgsz
            ):  # resize by stretching image to square imgsz
                im = cv2.resize(
                    im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR
                )

            # Add to buffer if training with augmentations
            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = (
                    im,
                    (h0, w0),
                    im.shape[:2],
                )  # im, hw_original, hw_resized
                self.buffer.append(i)
                if len(self.buffer) >= self.max_buffer_length:
                    j = self.buffer.pop(0)
                    self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None

            return im, (h0, w0), im.shape[:2]

        return self.ims[i], self.im_hw0[i], self.im_hw[i]

    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches."""
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k == "img":
                value = collate_images(value)
            if k in ["masks", "keypoints", "bboxes", "cls", "segments", "obb"]:
                value = torch.cat(value, 0)
            new_batch[k] = value
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i  # add target image index for build_targets()
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        return new_batch

    def build_transforms(self, hyp=None):
        """Builds and appends transforms to the list."""
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = wisard_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose(
                [LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)]
            )
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=hyp.bgr if self.augment else 0.0,  # only affect training.
            )
        )
        return transforms


def generate_wisard_filelist(root, folders, filename):
    items = build_wisard_items(root, folders)
    images = []
    for item in items:
        if isinstance(item[1][0], str):
            images.append(item[1][0])
        else:
            images.append(item[1][0][0] + "," + item[1][1][0])
    with open(os.path.join(root, filename), "w") as f:
        for item in images:
            f.write(f"{item}\n")
    return images
