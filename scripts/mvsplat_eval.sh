#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -lt 4 ]]; then
  echo 'Usage: bash scripts/mvsplat_eval.sh scannet|arkitscenes DATASET_DIR LAST_CHECKPOINT OUTPUT_DIR [Hydra overrides...]' >&2
  exit 2
fi
dataset=$1; dataset_dir=$2; weights=$3; run_dir=$4
shift 4
case "$dataset" in scannet|arkitscenes) ;; *) echo 'Unknown dataset' >&2; exit 2 ;; esac
[[ -f "$dataset_dir/manifest.json" && -f "$weights" ]] || { echo 'Missing manifest or checkpoint' >&2; exit 2; }
# Compare the same training budget, never silently use an interrupted checkpoint.
python - "$weights" <<'PY'
import sys, torch
w = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
if w.get('global_step') != 30000:
    raise SystemExit(f"Expected step 30000; got {w.get('global_step')}")
print('Verified checkpoint: step 30000')
PY
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
exec python -m src.main "+experiment=mvsplat_$dataset" mode=test \
  "dataset.roots=[\"$dataset_dir\"]" "checkpointing.pretrained_model=$weights" \
  "output_dir=$run_dir" "test.output_path=$run_dir" \
  dataset.test_split=test test.compute_scores=true "$@"
