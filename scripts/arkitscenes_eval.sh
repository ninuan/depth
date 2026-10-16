#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -lt 3 ]]; then
  echo "Usage: bash scripts/arkitscenes_eval.sh DATASET_DIR CHECKPOINT OUTPUT_DIR [Hydra overrides...]" >&2
  exit 2
fi
dataset_dir=$1
weights=$2
run_dir=$3
shift 3
[[ -f "$dataset_dir/manifest.json" && -f "$weights" ]] || { echo "Missing dataset manifest or checkpoint" >&2; exit 2; }
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
exec python -m src.main +experiment=arkitscenes mode=test \
  "dataset.roots=[\"$dataset_dir\"]" "checkpointing.pretrained_model=$weights" \
  "output_dir=$run_dir" "test.output_path=$run_dir" dataset.test_split=test \
  test.compute_scores=true "$@"
