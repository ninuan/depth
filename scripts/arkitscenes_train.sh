#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -lt 4 ]]; then
  echo "Usage: bash scripts/arkitscenes_train.sh baseline|depth DATASET_DIR WEIGHTS OUTPUT_DIR [Hydra overrides...]" >&2
  exit 2
fi
variant=$1
dataset_dir=$2
weights=$3
run_dir=$4
shift 4
case "$variant" in
  baseline) depth_weight=0.0 ;;
  depth) depth_weight=0.1 ;;
  *) echo "Variant must be baseline or depth" >&2; exit 2 ;;
esac
[[ -f "$dataset_dir/manifest.json" && -f "$weights" ]] || { echo "Missing dataset manifest or weights" >&2; exit 2; }
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
exec python -m src.main +experiment=arkitscenes \
  "dataset.roots=[\"$dataset_dir\"]" "checkpointing.pretrained_model=$weights" \
  "output_dir=$run_dir" "train.depth_loss_weight=$depth_weight" "$@"
