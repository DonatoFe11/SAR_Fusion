import os
import torch
import random
import numpy as np


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
from sarfusion.data.temporal_split import (
    load_temporal_split_manifest,
    manifest_folder_pairs,
    select_temporal_split_items,
)


DATASET_REGISTRY = {"sard_pose": PoseClassificationDataset, "wisard": WiSARDDataset}

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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
        temporal_manifest_path = dataset_params.get("temporal_split_manifest")
        temporal_inventory = None
        if temporal_manifest_path:
            temporal_manifest, temporal_inventory = load_temporal_split_manifest(
                temporal_manifest_path,
                dataset_params["root"],
                verify=True,
            )
            train_folders = manifest_folder_pairs(temporal_manifest)
            val_folders = list(train_folders)
        else:
            if "train_folders" in dataset_params:
                train_folders = get_wisard_folders(dataset_params["train_folders"])
            else:
                train_folders = get_wisard_phase_folders(
                    dataset_params["folders"], "train"
                )
            if "val_folders" in dataset_params:
                val_folders = get_wisard_folders(dataset_params["val_folders"])
            else:
                val_folders = get_wisard_phase_folders(
                    dataset_params["folders"], "val"
                )
        print(f"Using as train folders: \n {train_folders}")
        print(f"Using as val folders: \n {val_folders}")

        if "test_folders" in dataset_params:
            test_folders = get_wisard_folders(dataset_params["test_folders"])
        else:
            test_folders = get_wisard_phase_folders(
                dataset_params["folders"], "test"
            )
        print(f"Using as test folders: \n {test_folders}")

        # Phase-specific selectors configure the split; they are not
        # WiSARDDataset constructor arguments.
        shared_dataset_params = {
            key: value
            for key, value in dataset_params.items()
            if key not in {"train_folders", "val_folders", "test_folders"}
        }

        train_dataset_params = {
            **shared_dataset_params,
            "folders": train_folders,
            "temporal_split_phase": "train" if temporal_inventory else None,
            "temporal_split_inventory": temporal_inventory,
        }
        # Modal dropout is a training augmentation. Validation and test must
        # always evaluate the complete modality input selected by their
        # folders, otherwise checkpoint selection and final metrics depend on
        # randomly masked samples.
        val_dataset_params = {
            **shared_dataset_params,
            "folders": val_folders,
            "modal_dropout": False,
            "temporal_split_phase": "val" if temporal_inventory else None,
            "temporal_split_inventory": temporal_inventory,
        }
        test_dataset_params = {
            **shared_dataset_params,
            "folders": test_folders,
            "modal_dropout": False,
            "test_all_tiles": True if dataset_params.get("use_tiling", False) else False,
            "temporal_split_phase": None,
            "temporal_split_inventory": None,
        }
    else:
        raise ValueError(f"Unknown dataset name: {name}")
    return train_dataset_params, val_dataset_params, test_dataset_params


def get_dataloaders(
    dataset_params,
    dataloader_params,
    return_datasets=False,
    seed=None,
):
    dataset_params = dataset_params.copy()
    dataloader_params = dataloader_params.copy()

    # Training and evaluation have different memory profiles: validation and
    # test do not retain gradients, so they can normally use a larger batch.
    # Keep the historical behaviour when no dedicated size is configured.
    evaluation_batch_size = dataloader_params.pop(
        "evaluation_batch_size", dataloader_params["batch_size"]
    )
    evaluation_dataloader_params = {
        **dataloader_params,
        "batch_size": evaluation_batch_size,
    }

    transforms, denormalize = build_preprocessor(dataset_params)

    name = dataset_params.pop("name")
    dataclass = DATASET_REGISTRY[name]
    train_dataset_params, val_dataset_params, test_dataset_params = (
        get_train_val_test_params(name, dataset_params)
    )

    train_dataset_params.pop("preprocessor", None)
    val_dataset_params.pop("preprocessor", None)
    test_dataset_params.pop("preprocessor", None)

    train_temporal_phase = train_dataset_params.pop("temporal_split_phase", None)
    train_temporal_inventory = train_dataset_params.pop(
        "temporal_split_inventory", None
    )
    train_dataset_params.pop("temporal_split_manifest", None)
    train_set = dataclass(
        transform=transforms,
        **train_dataset_params,
    )
    if train_temporal_phase:
        train_set.items = select_temporal_split_items(
            train_set.items,
            train_dataset_params["root"],
            train_temporal_inventory,
            train_temporal_phase,
        )
    # Keep sampler/worker RNGs separate from the model RNG.  Using the
    # experiment seed explicitly also makes the data stream independent of
    # random numbers consumed while constructing other objects.
    seed = int(torch.initial_seed() if seed is None else seed)
    generator_train = torch.Generator().manual_seed(seed)
    generator_val = torch.Generator().manual_seed(seed + 1)
    generator_test = torch.Generator().manual_seed(seed + 2)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        collate_fn=get_collate_fn(train_set),
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=generator_train,
        **dataloader_params,
    )
    val_temporal_phase = val_dataset_params.pop("temporal_split_phase", None)
    val_temporal_inventory = val_dataset_params.pop("temporal_split_inventory", None)
    val_dataset_params.pop("temporal_split_manifest", None)
    val_set = dataclass(
        transform=transforms,
        **val_dataset_params,
    )
    if val_temporal_phase:
        val_set.items = select_temporal_split_items(
            val_set.items,
            val_dataset_params["root"],
            val_temporal_inventory,
            val_temporal_phase,
        )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        collate_fn=get_collate_fn(val_set),
        worker_init_fn=seed_worker,
        generator=generator_val,
        **evaluation_dataloader_params,
    )
    test_dataset_params.pop("temporal_split_phase", None)
    test_dataset_params.pop("temporal_split_inventory", None)
    test_dataset_params.pop("temporal_split_manifest", None)
    test_set = dataclass(
        transform=transforms,
        **test_dataset_params,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        collate_fn=get_collate_fn(test_set),
        worker_init_fn=seed_worker,
        generator=generator_test,
        **evaluation_dataloader_params,
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
