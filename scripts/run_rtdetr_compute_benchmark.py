#!/usr/bin/env python3
"""Benchmark the frozen RT-DETR Additive/FAM inference implementations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import torchvision


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fam_alignment_check import load_fusion_model, load_run_config  # noqa: E402
from sarfusion.models.checkpoints import resolve_local_wandb_checkpoint  # noqa: E402
from sarfusion.utils.utils import load_yaml  # noqa: E402


DEFAULT_PROTOCOL = "parameters/RTDETR/rtdetr_additive_fam_compute_benchmark.yaml"
PROTOCOL_ID = "rtdetr_additive_fam_compute_benchmark_v1"
CONFIGURATION_NAMES = ("additive", "fam")
FAM_CLASS_NAMES = {
    "FeatureAlignmentModule",
    "BoundedFeatureAlignmentModule",
    "IdentityInitializedFeatureAlignmentModule",
    "GridSampleFeatureAlignmentModule",
}


def stable_json_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_protocol(path):
    payload = load_yaml(path)
    protocol = payload.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("Compute benchmark YAML must contain a protocol mapping")
    if protocol.get("id") != PROTOCOL_ID:
        raise ValueError("Unexpected compute benchmark protocol id")
    if protocol.get("status") != "frozen_before_measurement":
        raise ValueError("Compute protocol must remain frozen_before_measurement")
    if protocol.get("checkpoint") != "latest":
        raise ValueError("Compute benchmark must use latest checkpoints")
    if int(protocol.get("checkpoint_seed", -1)) != 43:
        raise ValueError("Compute benchmark checkpoint seed must remain 43")
    if protocol.get("fam_variant") != "current_dcnv2":
        raise ValueError("Compute benchmark must use current_dcnv2")
    if set(protocol.get("configurations", {})) != set(CONFIGURATION_NAMES):
        raise ValueError("Compute benchmark must contain Additive and FAM")
    input_config = protocol.get("input", {})
    expected_input = {"batch_size": 1, "channels": 4, "height": 640, "width": 640}
    actual_input = {key: int(input_config.get(key, -1)) for key in expected_input}
    if actual_input != expected_input:
        raise ValueError(f"Unexpected frozen input shape: {actual_input}")
    execution = protocol.get("execution", {})
    expected_execution = {
        "tensor_dtype": "float32",
        "autocast": False,
        "tf32": False,
        "inference_mode": True,
        "torch_compile": False,
        "scope": "detector_network_forward_only",
        "preprocessing_included": False,
        "postprocessing_included": False,
        "warmup_iterations": 30,
        "measured_iterations_per_trial": 100,
        "trials": 3,
    }
    for key, expected in expected_execution.items():
        if execution.get(key) != expected:
            raise ValueError(
                f"Frozen execution setting {key}={execution.get(key)!r}, expected {expected!r}"
            )
    expected_order = [
        ["additive", "fam"],
        ["fam", "additive"],
        ["additive", "fam"],
    ]
    if execution.get("trial_order") != expected_order:
        raise ValueError("Unexpected frozen trial order")
    return protocol


def percentile(values, probability):
    values = sorted(float(value) for value in values)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    position = probability * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def summarize_values(values):
    values = [float(value) for value in values]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("Benchmark summaries require finite values")
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "p05": percentile(values, 0.05),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


def unique_tensors(named_tensors):
    seen = set()
    for name, tensor in named_tensors:
        identity = id(tensor)
        if identity in seen:
            continue
        seen.add(identity)
        yield name, tensor


def parameter_summary(model):
    parameters = list(unique_tensors(model.named_parameters(remove_duplicate=False)))
    buffers = list(unique_tensors(model.named_buffers(remove_duplicate=False)))
    fam_parameter_ids = {
        id(parameter)
        for module in model.modules()
        if type(module).__name__ in FAM_CLASS_NAMES
        for parameter in module.parameters(recurse=True)
    }
    total_parameters = sum(parameter.numel() for _name, parameter in parameters)
    trainable_parameters = sum(
        parameter.numel() for _name, parameter in parameters if parameter.requires_grad
    )
    fam_parameters = sum(
        parameter.numel()
        for _name, parameter in parameters
        if id(parameter) in fam_parameter_ids
    )
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for _name, parameter in parameters
    )
    buffer_elements = sum(buffer.numel() for _name, buffer in buffers)
    buffer_bytes = sum(
        buffer.numel() * buffer.element_size() for _name, buffer in buffers
    )
    return {
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "non_fam_parameters": total_parameters - fam_parameters,
        "fam_parameters": fam_parameters,
        "parameter_bytes": parameter_bytes,
        "buffer_elements": buffer_elements,
        "buffer_bytes": buffer_bytes,
        "state_bytes": parameter_bytes + buffer_bytes,
    }


def find_fam_modules(model):
    return [module for module in model.modules() if type(module).__name__ in FAM_CLASS_NAMES]


def capture_fam_shapes(model, detector, pixel_values, pixel_mask):
    modules = find_fam_modules(model)
    records = []
    hooks = []
    for level, module in enumerate(modules):
        def capture(_module, inputs, _level=level):
            if len(inputs) < 2:
                raise RuntimeError("FAM shape capture requires RGB and IR inputs")
            records.append(
                {
                    "level": _level,
                    "rgb_shape": list(inputs[0].shape),
                    "ir_shape": list(inputs[1].shape),
                }
            )

        hooks.append(module.register_forward_pre_hook(capture))
    try:
        with torch.inference_mode():
            output = detector(pixel_values=pixel_values, pixel_mask=pixel_mask)
        del output
        if pixel_values.is_cuda:
            torch.cuda.synchronize(pixel_values.device)
    finally:
        for hook in hooks:
            hook.remove()
    records.sort(key=lambda row: row["level"])
    if len(records) != len(modules):
        raise RuntimeError(
            f"Captured {len(records)} FAM calls for {len(modules)} modules"
        )
    return modules, records


def _conv_macs(module, shape):
    batch_size, _channels, height, width = [int(value) for value in shape]
    kernel_height, kernel_width = module.kernel_size
    weight_input_channels = int(module.weight.shape[1])
    return (
        batch_size
        * height
        * width
        * int(module.out_channels)
        * weight_input_channels
        * kernel_height
        * kernel_width
    )


def fam_conventional_cost(modules, shape_records):
    if len(modules) != len(shape_records):
        raise ValueError("FAM modules and shape records differ")
    rows = []
    for module, shape_record in zip(modules, shape_records):
        if not hasattr(module, "offset_conv") or not hasattr(module, "deform_conv"):
            raise ValueError("Current DCNv2 FAM must expose offset_conv and deform_conv")
        offset_macs = _conv_macs(module.offset_conv, shape_record["rgb_shape"])
        deform_macs = _conv_macs(module.deform_conv, shape_record["ir_shape"])
        rows.append(
            {
                **shape_record,
                "offset_conv_macs": offset_macs,
                "deform_conv_macs": deform_macs,
                "total_macs": offset_macs + deform_macs,
                "conventional_flops_two_per_mac": 2 * (offset_macs + deform_macs),
                "deform_conv_flops_two_per_mac": 2 * deform_macs,
            }
        )
    return {
        "levels": rows,
        "offset_conv_macs": sum(row["offset_conv_macs"] for row in rows),
        "deform_conv_macs": sum(row["deform_conv_macs"] for row in rows),
        "total_macs": sum(row["total_macs"] for row in rows),
        "conventional_flops_two_per_mac": sum(
            row["conventional_flops_two_per_mac"] for row in rows
        ),
        "deform_conv_flops_two_per_mac": sum(
            row["deform_conv_flops_two_per_mac"] for row in rows
        ),
        "excluded_operations": [
            "DCNv2 bilinear sampling",
            "modulation-mask multiplication",
            "sigmoid",
            "concatenation",
            "bias additions",
        ],
    }


def profile_supported_flops(detector, pixel_values, pixel_mask):
    from torch.profiler import ProfilerActivity, profile

    activities = [ProfilerActivity.CPU]
    if pixel_values.is_cuda:
        activities.append(ProfilerActivity.CUDA)
        torch.cuda.synchronize(pixel_values.device)
    with torch.inference_mode(), profile(
        activities=activities,
        record_shapes=False,
        profile_memory=False,
        with_flops=True,
    ) as profiler:
        output = detector(pixel_values=pixel_values, pixel_mask=pixel_mask)
        if pixel_values.is_cuda:
            torch.cuda.synchronize(pixel_values.device)
    del output
    rows = []
    for event in profiler.key_averages():
        flops = int(event.flops or 0)
        if flops or "deform_conv" in event.key:
            rows.append({"operator": event.key, "calls": int(event.count), "flops": flops})
    rows.sort(key=lambda row: (-row["flops"], row["operator"]))
    deform_profiled_flops = sum(
        row["flops"] for row in rows if "deform_conv" in row["operator"]
    )
    return {
        "supported_operator_flops": sum(row["flops"] for row in rows),
        "deform_conv_profiled_flops": deform_profiled_flops,
        "operator_rows": rows,
    }


def warmup(detector, pixel_values, pixel_mask, iterations):
    with torch.inference_mode():
        for _ in range(iterations):
            output = detector(pixel_values=pixel_values, pixel_mask=pixel_mask)
        del output
    torch.cuda.synchronize(pixel_values.device)


def measure_peak_memory(detector, pixel_values, pixel_mask):
    torch.cuda.synchronize(pixel_values.device)
    baseline_allocated = torch.cuda.memory_allocated(pixel_values.device)
    baseline_reserved = torch.cuda.memory_reserved(pixel_values.device)
    torch.cuda.reset_peak_memory_stats(pixel_values.device)
    with torch.inference_mode():
        output = detector(pixel_values=pixel_values, pixel_mask=pixel_mask)
    torch.cuda.synchronize(pixel_values.device)
    peak_allocated = torch.cuda.max_memory_allocated(pixel_values.device)
    peak_reserved = torch.cuda.max_memory_reserved(pixel_values.device)
    del output
    torch.cuda.synchronize(pixel_values.device)
    return {
        "baseline_allocated_bytes": baseline_allocated,
        "peak_allocated_bytes": peak_allocated,
        "incremental_peak_allocated_bytes": peak_allocated - baseline_allocated,
        "baseline_reserved_bytes": baseline_reserved,
        "peak_reserved_bytes": peak_reserved,
    }


def measure_latency(detector, pixel_values, pixel_mask, iterations):
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    with torch.inference_mode():
        for start, end in zip(starts, ends):
            start.record()
            output = detector(pixel_values=pixel_values, pixel_mask=pixel_mask)
            end.record()
        del output
    torch.cuda.synchronize(pixel_values.device)
    values = [float(start.elapsed_time(end)) for start, end in zip(starts, ends)]
    if not all(math.isfinite(value) and value > 0 for value in values):
        raise RuntimeError("CUDA latency measurements must be finite and positive")
    return values


def build_inputs(protocol, device):
    settings = protocol["input"]
    generator = torch.Generator(device=device).manual_seed(int(settings["generator_seed"]))
    pixel_values = torch.randn(
        int(settings["batch_size"]),
        int(settings["channels"]),
        int(settings["height"]),
        int(settings["width"]),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    pixel_mask = torch.ones(
        int(settings["batch_size"]),
        int(settings["height"]),
        int(settings["width"]),
        device=device,
        dtype=torch.bool,
    )
    return pixel_values, pixel_mask


def environment_summary(device):
    properties = torch.cuda.get_device_properties(device)
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_name": properties.name,
        "gpu_total_memory_bytes": properties.total_memory,
        "gpu_compute_capability": [properties.major, properties.minor],
        "device_index": device.index if device.index is not None else torch.cuda.current_device(),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def resolve_checkpoints(protocol):
    seed = int(protocol["checkpoint_seed"])
    result = {}
    for configuration, settings in protocol["configurations"].items():
        path = Path(
            resolve_local_wandb_checkpoint(
                settings["project"],
                seed,
                checkpoint=protocol["checkpoint"],
                wandb_root=REPO_ROOT / "wandb",
            )
        ).resolve()
        result[configuration] = {
            "path": path,
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
    return result


def build_model_for_configuration(protocol, configuration, checkpoint, device):
    seed = int(protocol["checkpoint_seed"])
    current_config = load_run_config(
        REPO_ROOT / protocol["training_config"], run_index=seed - 40
    )
    if int(current_config["seed"]) != seed:
        raise RuntimeError("Training grid run_index does not map to checkpoint seed")
    expected_use_fam = bool(
        protocol["configurations"][configuration]["expected_use_fam"]
    )
    model_params = current_config["model"]
    model_params["params"].update(
        {
            "use_fam": expected_use_fam,
            "freeze_fam": False,
            "fam_variant": protocol["fam_variant"],
            "ir_dropout_rate": 0.0,
            "spatial_jitter_std": 0.0,
        }
    )
    model = load_fusion_model(model_params, checkpoint, device)
    if bool(getattr(model, "use_fam", None)) != expected_use_fam:
        raise RuntimeError(f"Unexpected use_fam for {configuration}")
    detector = model.model
    detector.eval()
    return model, detector


def aggregate_results(protocol, trial_rows, static_rows, profiler_rows, fam_cost):
    configurations = {}
    for configuration in CONFIGURATION_NAMES:
        selected = [row for row in trial_rows if row["configuration"] == configuration]
        pooled_latencies = [value for row in selected for value in row["latency_ms"]]
        trial_means = [row["latency_summary_ms"]["mean"] for row in selected]
        incremental_peaks = [
            row["memory"]["incremental_peak_allocated_bytes"] for row in selected
        ]
        peak_totals = [row["memory"]["peak_allocated_bytes"] for row in selected]
        profile = profiler_rows[configuration]
        dcn_correction = 0
        if configuration == "fam" and profile["deform_conv_profiled_flops"] == 0:
            dcn_correction = fam_cost["deform_conv_flops_two_per_mac"]
        adjusted_flops = profile["supported_operator_flops"] + dcn_correction
        configurations[configuration] = {
            **static_rows[configuration],
            "trial_count": len(selected),
            "measured_forward_count": len(pooled_latencies),
            "latency_ms": summarize_values(pooled_latencies),
            "trial_mean_latency_ms": summarize_values(trial_means),
            "throughput_images_per_second_from_mean_latency": 1000.0
            / statistics.fmean(pooled_latencies),
            "incremental_peak_allocated_bytes": summarize_values(incremental_peaks),
            "peak_allocated_bytes": summarize_values(peak_totals),
            "profiler": profile,
            "unprofiled_dcnv2_convolution_equivalent_flops_added": dcn_correction,
            "adjusted_supported_operator_flops": adjusted_flops,
        }

    additive = configurations["additive"]
    fam = configurations["fam"]
    comparison = {
        "parameter_delta": fam["parameters"]["total_parameters"]
        - additive["parameters"]["total_parameters"],
        "parameter_ratio": fam["parameters"]["total_parameters"]
        / additive["parameters"]["total_parameters"],
        "state_bytes_delta": fam["parameters"]["state_bytes"]
        - additive["parameters"]["state_bytes"],
        "mean_latency_delta_ms": fam["latency_ms"]["mean"]
        - additive["latency_ms"]["mean"],
        "mean_latency_ratio": fam["latency_ms"]["mean"]
        / additive["latency_ms"]["mean"],
        "mean_latency_percent_increase": 100.0
        * (fam["latency_ms"]["mean"] / additive["latency_ms"]["mean"] - 1.0),
        "median_latency_delta_ms": fam["latency_ms"]["median"]
        - additive["latency_ms"]["median"],
        "incremental_peak_memory_delta_bytes": fam["incremental_peak_allocated_bytes"][
            "median"
        ]
        - additive["incremental_peak_allocated_bytes"]["median"],
        "peak_allocated_memory_delta_bytes": fam["peak_allocated_bytes"]["median"]
        - additive["peak_allocated_bytes"]["median"],
        "adjusted_supported_operator_flops_delta": fam[
            "adjusted_supported_operator_flops"
        ]
        - additive["adjusted_supported_operator_flops"],
        "adjusted_supported_operator_flops_ratio": fam[
            "adjusted_supported_operator_flops"
        ]
        / additive["adjusted_supported_operator_flops"],
    }
    expected_trials = int(protocol["execution"]["trials"])
    expected_measurements = expected_trials * int(
        protocol["execution"]["measured_iterations_per_trial"]
    )
    complete = all(
        configurations[name]["trial_count"] == expected_trials
        and configurations[name]["measured_forward_count"] == expected_measurements
        for name in CONFIGURATION_NAMES
    )
    if fam["parameters"]["non_fam_parameters"] != additive["parameters"][
        "total_parameters"
    ]:
        raise RuntimeError("FAM and Additive non-FAM parameter counts differ")
    return configurations, comparison, complete


def write_payload(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, sort_keys=True)
        output_file.write("\n")


def configure_cuda(protocol, requested_device):
    device = torch.device(requested_device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The frozen compute benchmark requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = bool(protocol["execution"]["cudnn_benchmark"])
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(int(protocol["input"]["generator_seed"]))
    torch.cuda.manual_seed_all(int(protocol["input"]["generator_seed"]))
    return device


def run_isolated_worker(
    protocol,
    checkpoints,
    configuration,
    trial,
    order_position,
    warmup_iterations,
    measured_iterations,
    requested_device,
    include_profile,
    output_path,
):
    """Measure one configuration in a fresh process-owned CUDA allocator."""
    device = configure_cuda(protocol, requested_device)
    print(
        f"[worker load] trial={trial} position={order_position} "
        f"configuration={configuration}"
    )
    model, detector = build_model_for_configuration(
        protocol,
        configuration,
        checkpoints[configuration]["path"],
        device,
    )
    pixel_values, pixel_mask = build_inputs(protocol, device)
    parameters = parameter_summary(model)
    modules, shape_records = capture_fam_shapes(
        model, detector, pixel_values, pixel_mask
    )
    fam_cost = None
    if configuration == "additive" and modules:
        raise RuntimeError("Additive unexpectedly contains FAM modules")
    if configuration == "fam":
        if len(modules) != 3:
            raise RuntimeError(f"Expected three FAM modules, found {len(modules)}")
        fam_cost = fam_conventional_cost(modules, shape_records)

    static = {
        "checkpoint": str(checkpoints[configuration]["path"]),
        "checkpoint_sha256": checkpoints[configuration]["sha256"],
        "checkpoint_size_bytes": checkpoints[configuration]["size_bytes"],
        "parameters": parameters,
        "fam_feature_shapes": shape_records,
    }
    warmup(detector, pixel_values, pixel_mask, warmup_iterations)
    memory = measure_peak_memory(detector, pixel_values, pixel_mask)
    latencies = measure_latency(detector, pixel_values, pixel_mask, measured_iterations)
    trial_row = {
        "trial": trial,
        "order_position": order_position,
        "configuration": configuration,
        "warmup_iterations": warmup_iterations,
        "measured_iterations": measured_iterations,
        "latency_ms": latencies,
        "latency_summary_ms": summarize_values(latencies),
        "memory": memory,
    }
    profiler = (
        profile_supported_flops(detector, pixel_values, pixel_mask)
        if include_profile
        else None
    )
    payload = {
        "configuration": configuration,
        "environment": environment_summary(device),
        "static": static,
        "fam_conventional_cost": fam_cost,
        "trial_result": trial_row,
        "profiler": profiler,
    }
    write_payload(output_path, payload)
    print(
        f"[worker done] trial={trial} configuration={configuration} "
        f"mean={trial_row['latency_summary_ms']['mean']:.3f} ms "
        f"peak={memory['peak_allocated_bytes'] / 2**20:.1f} MiB"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--worker-configuration", choices=CONFIGURATION_NAMES, help=argparse.SUPPRESS
    )
    parser.add_argument("--worker-trial", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-order-position", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-warmup", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-measured", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-profile", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", help=argparse.SUPPRESS)
    args = parser.parse_args()

    protocol_path = Path(args.protocol)
    if not protocol_path.is_absolute():
        protocol_path = REPO_ROOT / protocol_path
    protocol = load_protocol(protocol_path.resolve())
    protocol_hash = stable_json_hash(protocol)
    checkpoints = resolve_checkpoints(protocol)
    for configuration, checkpoint in checkpoints.items():
        print(
            f"configuration={configuration} checkpoint={checkpoint['path']} "
            f"sha256={checkpoint['sha256']}"
        )
    if args.prepare_only:
        print("Prepare-only OK: frozen protocol and both checkpoints are valid")
        return

    requested_device = args.device or protocol["execution"]["device"]
    if args.worker_configuration is not None:
        required = {
            "--worker-trial": args.worker_trial,
            "--worker-order-position": args.worker_order_position,
            "--worker-warmup": args.worker_warmup,
            "--worker-measured": args.worker_measured,
            "--worker-output": args.worker_output,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            parser.error(f"Missing isolated-worker arguments: {missing}")
        run_isolated_worker(
            protocol,
            checkpoints,
            args.worker_configuration,
            args.worker_trial,
            args.worker_order_position,
            args.worker_warmup,
            args.worker_measured,
            requested_device,
            args.worker_profile,
            Path(args.worker_output),
        )
        return

    output_path = Path(args.output or protocol["output"])
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    if args.smoke and args.output is None:
        output_path = output_path.with_name(output_path.stem + "_smoke.json")
    if output_path.exists() and not args.force:
        raise RuntimeError(f"Output already exists: {output_path}. Inspect before --force")

    execution = protocol["execution"]
    trial_order = execution["trial_order"]
    warmup_iterations = int(execution["warmup_iterations"])
    measured_iterations = int(execution["measured_iterations_per_trial"])
    if args.smoke:
        trial_order = [trial_order[0]]
        warmup_iterations = 2
        measured_iterations = 5

    output_path.parent.mkdir(parents=True, exist_ok=True)
    environment = None
    static_rows = {}
    profiler_rows = {}
    fam_cost = None
    trial_rows = []
    profiled = set()
    with tempfile.TemporaryDirectory(
        prefix="rtdetr-compute-workers-", dir=output_path.parent
    ) as temporary_directory:
        temporary_root = Path(temporary_directory)
        for trial_index, order in enumerate(trial_order, start=1):
            for order_position, configuration in enumerate(order, start=1):
                worker_output = temporary_root / (
                    f"trial_{trial_index}_{order_position}_{configuration}.json"
                )
                include_profile = configuration not in profiled
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--protocol",
                    str(protocol_path.resolve()),
                    "--device",
                    requested_device,
                    "--worker-configuration",
                    configuration,
                    "--worker-trial",
                    str(trial_index),
                    "--worker-order-position",
                    str(order_position),
                    "--worker-warmup",
                    str(warmup_iterations),
                    "--worker-measured",
                    str(measured_iterations),
                    "--worker-output",
                    str(worker_output),
                ]
                if include_profile:
                    command.append("--worker-profile")
                subprocess.run(command, cwd=REPO_ROOT, check=True)
                with worker_output.open(encoding="utf-8") as input_file:
                    worker = json.load(input_file)
                if worker["configuration"] != configuration:
                    raise RuntimeError("Isolated worker returned the wrong configuration")
                if environment is None:
                    environment = worker["environment"]
                elif worker["environment"] != environment:
                    raise RuntimeError("CUDA/software environment changed between workers")
                current_static = worker["static"]
                if configuration not in static_rows:
                    static_rows[configuration] = current_static
                elif current_static != static_rows[configuration]:
                    raise RuntimeError(f"Static model metadata changed for {configuration}")
                current_fam_cost = worker["fam_conventional_cost"]
                if current_fam_cost is not None:
                    if fam_cost is None:
                        fam_cost = current_fam_cost
                    elif current_fam_cost != fam_cost:
                        raise RuntimeError("FAM shapes/cost changed between workers")
                if worker["profiler"] is not None:
                    profiler_rows[configuration] = worker["profiler"]
                    profiled.add(configuration)
                trial_rows.append(worker["trial_result"])

    if environment is None or fam_cost is None:
        raise RuntimeError("Isolated benchmark workers did not return required metadata")
    if set(profiler_rows) != set(CONFIGURATION_NAMES):
        raise RuntimeError("Missing supported-operator profiles")

    configurations, comparison, complete = aggregate_results(
        protocol, trial_rows, static_rows, profiler_rows, fam_cost
    )
    payload = {
        "schema_version": 1,
        "protocol_id": protocol["id"],
        "protocol_sha256": protocol_hash,
        "protocol_complete": complete and not args.smoke,
        "smoke": args.smoke,
        "environment": environment,
        "input": protocol["input"],
        "execution": {
            **protocol["execution"],
            "actual_warmup_iterations": warmup_iterations,
            "actual_measured_iterations_per_trial": measured_iterations,
            "actual_trial_order": trial_order,
        },
        "complexity_method": protocol["complexity"],
        "fam_conventional_cost": fam_cost,
        "trial_results": trial_rows,
        "configurations": configurations,
        "comparison_fam_minus_additive": comparison,
    }
    write_payload(output_path, payload)
    print(f"Saved benchmark: {output_path}")
    print(f"protocol_complete={payload['protocol_complete']}")


if __name__ == "__main__":
    main()
