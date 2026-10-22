#!/usr/bin/env bash
# Export the first eight fixed validation scenes, with the same views for all methods.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -ne 5 ]]; then
  echo 'Usage: bash scripts/arkitscenes_compare_visuals.sh DATASET BASELINE_CKPT DEPTH_CKPT MVSPLAT_CKPT OUTPUT_ROOT' >&2
  exit 2
fi
dataset=$1; baseline=$2; depth=$3; mvsplat=$4; output=$5
[[ -f "$dataset/manifest.json" ]] || { echo 'Missing dataset manifest' >&2; exit 2; }
python - "$baseline" "$depth" "$mvsplat" <<'PY'
import sys, torch
for path in sys.argv[1:]:
    state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    step = state.get('global_step')
    if step != 30000:
        raise SystemExit(f'Expected step 30000: {path}; got {step}')
    print(f'Checked step 30000: {path}')
    del state
PY
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
common=(dataset.test_split=val dataset.test_len=8 seed=111123
        test.save_image=true test.save_gt_image=true test.save_input_images=true
        test.save_depth=true test.save_depth_npy=true)
bash scripts/arkitscenes_eval.sh "$dataset" "$baseline" "$output/baseline" \
  train.depth_loss_weight=0.0 "${common[@]}"
bash scripts/arkitscenes_eval.sh "$dataset" "$depth" "$output/depth" "${common[@]}"
bash scripts/mvsplat_eval.sh arkitscenes "$dataset" "$mvsplat" "$output/mvsplat" "${common[@]}"
