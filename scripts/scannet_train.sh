#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -lt 4 ]]; then
  echo "Usage: bash scripts/scannet_train.sh baseline|depth DATASET_DIR WEIGHTS OUTPUT_DIR [Hydra overrides...]" >&2
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
[[ -f "$dataset_dir/manifest.json" ]] || { echo "Convert/check the v2 dataset first" >&2; exit 2; }
[[ -f "$weights" ]] || { echo "Missing weights: $weights" >&2; exit 2; }
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
exec python -m src.main +experiment=scannet \
  "dataset.roots=[\"$dataset_dir\"]" \
  "checkpointing.pretrained_model=$weights" \
  "output_dir=$run_dir" "train.depth_loss_weight=$depth_weight" "$@"
