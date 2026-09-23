from pathlib import Path

import torch
import yaml


def _exact_tensor_alias_identity(tensor):
    """Identify exact tensor aliases without grouping unrelated views."""
    if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
        return None
    return (
        tensor.device.type,
        tensor.device.index,
        tensor.untyped_storage().data_ptr(),
        tensor.storage_offset(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
    )


def complete_shared_state_dict_aliases(model, state_dict):
    """Restore only exact tied-tensor aliases omitted by Safetensors.

    Accelerator intentionally serializes one key for parameters shared by
    several module paths.  A subsequent strict ``load_state_dict`` needs all
    aliases.  This function reconstructs a missing key only when the current
    model proves that it is an exact storage/offset/shape/stride alias of a key
    present in the checkpoint; every unrelated mismatch remains strict.
    """
    completed = dict(state_dict)
    alias_groups = {}
    for name, tensor in model.state_dict().items():
        identity = _exact_tensor_alias_identity(tensor)
        if identity is not None:
            alias_groups.setdefault(identity, []).append(name)

    restored = {}
    for names in alias_groups.values():
        available = [name for name in names if name in completed]
        if not available:
            continue
        source_name = available[0]
        for name in names:
            if name not in completed:
                completed[name] = completed[source_name]
                restored[name] = source_name
    return completed, restored


def _wandb_config_value(config, key):
    """Read a value from W&B's local ``files/config.yaml`` format."""
    value = config.get(key)
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def resolve_local_wandb_checkpoint(
    project,
    seed,
    checkpoint="latest",
    wandb_root="wandb",
):
    """Resolve one completed local W&B checkpoint by project and seed.

    Evaluation configs use this instead of timestamped ``wandb/run-*`` paths,
    which are not known until a training run starts. Ambiguous matches fail
    explicitly so a thesis evaluation can never silently select a rerun.
    """
    if checkpoint not in {"best", "latest"}:
        raise ValueError(
            "checkpoint must be 'best' or 'latest', got "
            f"{checkpoint!r}"
        )

    root = Path(wandb_root)
    candidates = []
    for config_path in sorted(root.glob("*run-*/files/config.yaml")):
        try:
            with config_path.open(encoding="utf-8") as config_file:
                config = yaml.safe_load(config_file) or {}
        except (OSError, yaml.YAMLError):
            continue

        experiment = _wandb_config_value(config, "experiment") or {}
        run_project = experiment.get("name") if isinstance(experiment, dict) else None
        run_seed = _wandb_config_value(config, "seed")
        try:
            seed_matches = int(run_seed) == int(seed)
        except (TypeError, ValueError):
            seed_matches = False

        checkpoint_path = config_path.parent / checkpoint / "model.safetensors"
        if run_project == project and seed_matches and checkpoint_path.is_file():
            candidates.append(checkpoint_path)

    description = f"project={project!r}, seed={int(seed)}, checkpoint={checkpoint!r}"
    if not candidates:
        raise FileNotFoundError(
            f"No completed local W&B checkpoint found for {description} under "
            f"{root.resolve()}"
        )
    if len(candidates) > 1:
        paths = "\n".join(f"- {path}" for path in candidates)
        raise RuntimeError(
            f"Multiple local W&B checkpoints found for {description}; use an "
            f"explicit pretrained_path:\n{paths}"
        )
    return str(candidates[0])
