#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -lt 4 ]]; then
  echo 'Usage: bash scripts/mvsplat_train.sh scannet|arkitscenes DATASET_DIR WEIGHTS OUTPUT_DIR [Hydra overrides...]' >&2
  exit 2
fi
dataset=$1; dataset_dir=$2; weights=$3; run_dir=$4
shift 4
case "$dataset" in scannet|arkitscenes) ;; *) echo 'Unknown dataset' >&2; exit 2 ;; esac
[[ -f "$dataset_dir/manifest.json" && -f "$weights" ]] || { echo 'Missing manifest or weights' >&2; exit 2; }
resume=false
for arg in "$@"; do [[ "$arg" != 'checkpointing.resume=true' ]] || resume=true; done
if [[ -f "$run_dir/run_config.yaml" && "$resume" != true ]]; then
  echo "Existing run: $run_dir. Use checkpointing.resume=true with last.ckpt, or a new output directory." >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
exec python -m src.main "+experiment=mvsplat_$dataset" \
  "dataset.roots=[\"$dataset_dir\"]" "checkpointing.pretrained_model=$weights" \
  "output_dir=$run_dir" seed=111123 data_loader.train.seed=1234 \
  data_loader.train.batch_size=1 trainer.max_steps=30000 "$@"
