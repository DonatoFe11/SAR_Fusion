#!/usr/bin/env bash
# Five new RT-DETR v1 FAM runs; no baseline retraining or automatic Stage B.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp/matplotlib-rtdetr-zero-offset
export YOLO_CONFIG_DIR=/tmp/yolo-rtdetr-zero-offset
export YOLO_AUTOINSTALL=false CUBLAS_WORKSPACE_CONFIG=:4096:8
exec conda run --no-capture-output -n sarfusion python \
  scripts/run_rtdetr_fam_zero_offset_stage_a.py "$@"
