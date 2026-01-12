import os
import torch


from sarfusion.data.sard import PoseClassificationDataset
from sarfusion.data.utils import dict_collate_fn, get_collate_fn
from sarfusion.data.utils import build_preprocessor
from sarfusion.data.wisard import (
    WiSARDDataset,
    TRAIN_FOLDERS,
    VAL_FOLDERS,
    TEST_FOLDERS,
    get_wisard_folders,
)


DATASET_REGISTRY = {"sard_pose": PoseClassificationDataset, "wisard": WiSARDDataset}


def get_wisard_phase_folders(folders, phase):
    phase_folders = {"train": TRAIN_FOLDERS, "val": VAL_FOLDERS, "test": TEST_FOLDERS}[
        phase
    ]
    return [folder for folder in get_wisard_folders(folders) if folder in phase_folders]


def get_train_val_test_params(name, dataset_params):
    if name == "sard_pose":
        train_dataset_params = {
            **dataset_params,
            "root": os.path.join(dataset_params["root"], "train"),
        }
        val_dataset_params = {
            **dataset_params,
            "root": os.path.join(dataset_params["root"], "valid"),
        }
        test_dataset_params = {
            **dataset_params,
            "root": os.path.join(dataset_params["root"], "test"),
        }
    elif name == "wisard":
        train_folders = dataset_params.get("train_folders", dataset_params["folders"])
        train_folders = get_wisard_phase_folders(train_folders, "train")
        print(f"Using as train folders: \n {train_folders}")

        val_folders = dataset_params.get("val_folders", dataset_params["folders"])
        val_folders = get_wisard_phase_folders(val_folders, "val")
        print(f"Using as val folders: \n {val_folders}")

        test_folders = dataset_params.get("test_folders", dataset_params["folders"])
        test_folders = get_wisard_phase_folders(test_folders, "test")
        print(f"Using as test folders: \n {test_folders}")

        train_dataset_params = {
            **dataset_params,
            "folders": train_folders,
        }
        val_dataset_params = {**dataset_params, "folders": val_folders}
        test_dataset_params = {**dataset_params, "folders": test_folders}
    else:
        raise ValueError(f"Unknown dataset name: {name}")
    return train_dataset_params, val_dataset_params, test_dataset_params


def get_dataloaders(dataset_params, dataloader_params, return_datasets=False):
    dataset_params = dataset_params.copy()

    transforms, denormalize = build_preprocessor(dataset_params)

    name = dataset_params.pop("name")
    dataclass = DATASET_REGISTRY[name]
    train_dataset_params, val_dataset_params, test_dataset_params = (
        get_train_val_test_params(name, dataset_params)
    )

    train_dataset_params.pop("preprocessor", None)
    val_dataset_params.pop("preprocessor", None)
    test_dataset_params.pop("preprocessor", None)
    train_set = dataclass(
        transform=transforms,
        **train_dataset_params,
    )
    train_loader = torch.utils.data.DataLoader(
        train_set,
        collate_fn=get_collate_fn(train_set),
        shuffle=True,
        **dataloader_params,
    )
    val_set = dataclass(
        transform=transforms,
        **val_dataset_params,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        collate_fn=get_collate_fn(val_set),
        **dataloader_params,
    )
    test_set = dataclass(
        transform=transforms,
        **test_dataset_params,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        collate_fn=get_collate_fn(test_set),
        **dataloader_params,
    )
    if return_datasets:
        return (
            (train_loader, val_loader, test_loader),
            (
                train_set,
                val_set,
                test_set,
            ),
            (
                get_collate_fn(train_set),
                get_collate_fn(val_set),
                get_collate_fn(test_set),
            ),
            denormalize,
        )

    return (train_loader, val_loader, test_loader), denormalize
