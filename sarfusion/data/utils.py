from copy import deepcopy
import os

import torch
import torchvision.transforms as T
from transformers import AutoProcessor

from sarfusion.utils.augmentations import denormalize
from sarfusion.utils.structures import DataDict


def dict_collate_fn(batch):
    d = {}
    keys = batch[0].keys()
    for key in keys:
        if isinstance(batch[0][key], torch.Tensor):
            d[key] = torch.stack([sample[key] for sample in batch])
        elif isinstance(batch[0][key], int) or isinstance(batch[0][key], float):
            d[key] = torch.stack([torch.tensor(sample[key]) for sample in batch])
        elif isinstance(batch[0][key], list):
            d[key] = [sample[key] for sample in batch]
        else:
            raise ValueError(f"Unsupported type {type(batch[0][key])}")

    return d


def get_collate_fn(dataset):
    if hasattr(dataset, "collate_fn"):
        return dataset.collate_fn
    return dict_collate_fn


def build_preprocessor(params):
    params = deepcopy(params)
    preprocessor_params = params.pop("preprocessor")
    if "path" in preprocessor_params:
        path = preprocessor_params["path"]
        auto_processor = AutoProcessor.from_pretrained(path, **preprocessor_params)
        return auto_processor, lambda x: x
    return T.Compose(
        [
            T.ToTensor(),
            T.Normalize(
                mean=preprocessor_params["mean"], std=preprocessor_params["std"]
            ),
        ]
    ), lambda x: denormalize(x, preprocessor_params["mean"], preprocessor_params["std"])


def is_annotation_valid(annotation):
    bbox = annotation[1:]
    return all([0 <= x <= 1 for x in bbox])


def load_annotations(annotation_path):
    with open(annotation_path, "r") as file:
        annotations = file.readlines()

    # Parse annotations
    targets = []
    for annotation in annotations:
        annotation = annotation.strip().split()
        class_label = int(annotation[0])
        x_center, y_center, width, height = map(float, annotation[1:])
        targets.append([class_label, x_center, y_center, width, height])
    return targets


def process_image_annotation_folders(root):
    image_path = os.path.join(root, "images")
    annotation_path = os.path.join(root, "labels")
    annotations = os.listdir(annotation_path)
    images = os.listdir(image_path)
    annotation_paths = sorted(
        [os.path.join(annotation_path, ann) for ann in annotations]
    )
    image_paths = sorted([os.path.join(image_path, img) for img in images])
    if len(annotation_paths) != len(image_paths):
        if len(annotation_paths) == 0:
            print(f"WARNING: No annotations found in {annotation_path}")
        else:
            raise ValueError(
                f"Number of annotations ({len(annotation_paths)}) does not match number of images ({len(image_paths)})"
            )
    for i in range(len(annotation_paths)):
    #     assert (
    #         annotation_paths[i].split("/")[-1].split(".")[0]
    #         == image_paths[i].split("/")[-1].split(".")[0]
    #     ), f"Image and annotation names do not match: {annotation_paths[i]} != {image_paths[i]}"
        pass
    return image_paths, annotation_paths


def collate_images(images):
    IGNORE_VALUE = 114
    images = list(images)
    depths = [image.shape[0] for image in images]
    if len(set(depths)) > 1:
        for i, image in enumerate(images):
            if image.shape[0] == 4:  # RGBT
                continue
            elif image.shape[0] == 3:  # RGB
                padding = torch.full((1, *image.shape[1:]), IGNORE_VALUE)
                images[i] = torch.cat([image, padding], 0)
            elif image.shape[0] == 1:  # Thermal
                padding = torch.full((3, *image.shape[1:]), IGNORE_VALUE)
                images[i] = torch.cat([padding, image], 0)
            else:
                raise ValueError(f"Unsupported image shape {image.shape}")
    return torch.stack(images, 0)


def yolo_to_coco_annotations(bboxes, image_id, img_width, img_height):
    annotations = []
    annotation_id = 0  # Can be used to give unique IDs to annotations

    for bbox in bboxes:
        class_id, x_center, y_center, width, height = bbox

        # Convert YOLO bbox (center_x, center_y, width, height) to COCO bbox (x_min, y_min, width, height)
        x_min = (x_center - width / 2) * img_width
        y_min = (y_center - height / 2) * img_height
        box_width = width * img_width
        box_height = height * img_height

        # COCO format annotation
        annotation = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": class_id,
            "bbox": [x_min, y_min, box_width, box_height],
            "area": box_width * box_height,
            "iscrowd": 0
        }

        annotations.append(annotation)
        annotation_id += 1

    # Return the COCO-formatted annotations as a list of dicts
    return {"image_id": image_id, "annotations": annotations}