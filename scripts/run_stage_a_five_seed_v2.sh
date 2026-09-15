#!/usr/bin/env bash
# Complete Stage A only. No seed-specific performance or vitality screen.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

scope=all
dry_run=false
for argument in "$@"; do
  case "$argument" in
    --dry-run) dry_run=true ;;
    --rtdetr-only) scope=rtdetr ;;
    --yolo26-only) scope=yolo26 ;;
    *) echo "Usage: bash $0 [--dry-run] [--rtdetr-only|--yolo26-only]" >&2; exit 2 ;;
  esac
done

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp/matplotlib-stage-a-five-seed-v2
export YOLO_CONFIG_DIR=/tmp/yolo-stage-a-five-seed-v2
export YOLO_AUTOINSTALL=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8

run_command() {
  printf 'Running: '
  printf '%q ' "$@"
  printf '\n'
  if [[ "$dry_run" == false ]]; then
    "$@"
  fi
}

if [[ "$scope" != yolo26 ]]; then
  for config in \
    parameters/RTDETR/rtdetr_fam_stage_a_five_seed_v2.yaml \
    parameters/RTDETR/rtdetr_fam_box_guided_stage_a_five_seed_v2.yaml \
    parameters/RTDETR/rtdetr_fam_mixed_consistency_stage_a_five_seed_v2.yaml; do
    run_command conda run --no-capture-output -n sarfusion python main.py experiment \
      --parameters "$config"
  done
fi

if [[ "$scope" != rtdetr ]]; then
  for arm in additive fam; do
    for seed in 40 41 42 43 44; do
      run_command conda run --no-capture-output -n sarfusion-yolo26 python \
        scripts/run_yolo26_stage_a_five_seed_v2.py \
        --config "parameters/YOLO26/yolo26s_${arm}_stage_a_five_seed_v2.yaml" \
        --seed "$seed"
    done
  done
fi

echo 'Stage A commands finished. Stage B requires the aggregate five-seed decision; it is not launched here.'
