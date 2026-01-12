from pathlib import Path
import re
import torch

from ultralytics.utils import LOGGER, yaml_load
from ultralytics.utils.checks import check_yaml
from ultralytics.nn.tasks import guess_model_scale


from safetensors import safe_open
from safetensors.torch import save_file


def torch_dict_load(file_path):
    if (
        file_path.endswith(".pth")
        or file_path.endswith(".pt")
        or file_path.endswith(".bin")
    ):
        return torch.load(file_path)
    if file_path.endswith(".safetensors"):
        with safe_open(file_path, framework="pt") as f:
            d = {}
            for k in f.keys():
                d[k] = f.get_tensor(k)
        return d
    raise ValueError("File extension not supported")


def torch_dict_save(data, file_path):
    if (
        file_path.endswith(".pth")
        or file_path.endswith(".pt")
        or file_path.endswith(".bin")
    ):
        torch.save(data, file_path)
    elif file_path.endswith(".safetensors"):
        save_file(data, file_path)
    else:
        raise ValueError("File extension not supported")


def fusion_pretraining_load(model, weights):
    incompatible_keys = model.load_state_dict(weights, strict=False)
    expected_unexpected_keys = ["model.model.0.conv.weight"]
    # expected_missing_keys_with_emb = [
    #     "model.model.0.optional_rgb.conv.weight",
    #     "model.model.0.optional_rgb.conv.bias",
    #     "model.model.0.optional_rgb.bn.weight",
    #     "model.model.0.optional_rgb.bn.bias",
    #     "model.model.0.optional_rgb.bn.running_mean",
    #     "model.model.0.optional_rgb.bn.running_var",
    #     "model.model.0.optional_rgb.embedding.weight",
    #     "model.model.0.optional_ir.conv.weight",
    #     "model.model.0.optional_ir.conv.bias",
    #     "model.model.0.optional_ir.bn.weight",
    #     "model.model.0.optional_ir.bn.bias",
    #     "model.model.0.optional_ir.bn.running_mean",
    #     "model.model.0.optional_ir.bn.running_var",
    #     "model.model.0.optional_ir.embedding.weight",
    #     "model.model.0.conv.conv.weight",
    #     "model.model.0.conv.bn.weight",
    #     "model.model.0.conv.bn.bias",
    #     "model.model.0.conv.bn.running_mean",
    #     "model.model.0.conv.bn.running_var",
    # ]
    # expected_missing_keys_without_emb = [
    #     item for item in expected_missing_keys_with_emb if "embedding" not in item
    # ]
    assert (keys.startswith("model.model.0") for keys in incompatible_keys.unexpected_keys)
    assert (keys.startswith("model.model.0") for keys in incompatible_keys.missing_keys)


def nc_safe_load(model, weights, nc):
    try:
        result = model.load_state_dict(weights, strict=False)
    except RuntimeError as e:
        error_msg = str(e)
        pattern = r"size mismatch for ([\w.]+): copying a param with shape torch.Size\((\[.*?\])\) from checkpoint, the shape in current model is torch.Size\((\[.*?\])\)"
        matches = re.findall(pattern, error_msg)
        for m in matches:
            print(f"Detected mismatch in {m[0]}: {m[1]} vs {m[2]}")
        checkpoint_shapes = [x for m in matches for x in eval(m[1])]
        model_shapes = [x for m in matches for x in eval(m[2])]
        diffs = [
            diff for diff in zip(checkpoint_shapes, model_shapes) if diff[0] != diff[1]
        ]
        for diff in diffs:
            assert (
                diff[1] == nc
                or diff[0] == 80  # 80 is the default number of classes of COCO
            ), "Detected a mismatch which is not due to the number of classes"
        print(f"Loading model with {nc} classes")


def yaml_model_load(path):
    """Load a YOLOv8 model from a YAML file."""
    import re

    path = Path(path)
    if path.stem in (f"yolov{d}{x}6" for x in "nsmlx" for d in (5, 8)):
        new_stem = re.sub(r"(\d+)([nslmx])6(.+)?$", r"\1\2-p6\3", path.stem)
        LOGGER.warning(
            f"WARNING ⚠️ Ultralytics YOLO P6 models now use -p6 suffix. Renaming {path.stem} to {new_stem}."
        )
        path = path.with_name(new_stem + path.suffix)

    if "v10" not in str(path) and "yolo" in str(path):
        unified_path = re.sub(
            r"(\d+)([nsblmx])(.+)?$", r"\1\3", str(path)
        )  # i.e. yolov8x.yaml -> yolov8.yaml
    else:
        unified_path = path
    yaml_file = check_yaml(unified_path, hard=False) or check_yaml(path)
    d = yaml_load(yaml_file)  # model dict
    d["scale"] = guess_model_scale(path)
    d["yaml_file"] = str(path)
    return d
