from pathlib import Path

import yaml


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
