#!/usr/bin/env bash
# Re-evaluate existing DepthSplat checkpoints; no training and no checkpoint selection.
set -euo pipefail
cd "$(dirname "$0")/.."
dataset_dir=${1:-datasets/scannet_processed_v2}
output_root=${2:-outputs/scannet_fixed30000}
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1

# Check all six files before spending GPU time. Never substitute a best checkpoint.
python - "$dataset_dir" <<'PY'
import json
import sys
from pathlib import Path
import torch
from omegaconf import OmegaConf

dataset = Path(sys.argv[1])
if not (dataset / 'manifest.json').is_file():
    raise SystemExit(f'Missing dataset manifest: {dataset}')
count = len(json.loads((dataset / 'test/index.json').read_text()))
if count != 159:
    raise SystemExit(f'Expected the fixed 159-scene test set; got {count}')
for seed in (111123, 222123, 333123):
    suffix = '' if seed == 111123 else f'_seed{seed}'
    for variant, weight in (('rgb', 0.0), ('depth', 0.1)):
        train_dir = Path(f'outputs/scannet_{variant}{suffix}')
        cfg = OmegaConf.load(train_dir / 'run_config.yaml')
        if cfg.mode != 'train' or cfg.seed != seed or cfg.train.depth_loss_weight != weight:
            raise SystemExit(f'Training metadata mismatch: {train_dir}')
        checkpoint = train_dir / 'checkpoints/last.ckpt'
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        step = state.get('global_step')
        del state
        if step != 30000:
            raise SystemExit(f'Expected step 30000: {checkpoint}; got {step}')
        print(f'Checked step 30000: {checkpoint}', flush=True)
PY

for seed in 111123 222123 333123; do
  suffix="_seed${seed}"
  if [[ "$seed" == 111123 ]]; then suffix=""; fi
  for variant in rgb depth; do
    weight=0.0
    label=baseline
    if [[ "$variant" == depth ]]; then weight=0.1; label=depth; fi
    checkpoint="outputs/scannet_${variant}${suffix}/checkpoints/last.ckpt"
    run_dir="$output_root/${label}_seed${seed}"
    echo "Evaluating $label, training seed $seed, step 30000"
    python -m src.main +experiment=scannet mode=test \
      "dataset.roots=[\"$dataset_dir\"]" dataset.test_split=test dataset.test_len=-1 \
      "checkpointing.load=$checkpoint" checkpointing.pretrained_model=null \
      checkpointing.resume=false checkpointing.no_strict_load=false \
      "output_dir=$run_dir" "test.output_path=$run_dir" \
      seed=111123 data_loader.test.seed=2345 data_loader.test.batch_size=1 \
      "train.depth_loss_weight=$weight" test.compute_scores=true test.save_image=false
  done
done

python -m src.scripts.audit_rgbd_results --dataset scannet \
  --root "$output_root" --output "$output_root/checkpoint_audit.json"
echo "Finished six evaluations. Report: $output_root/checkpoint_audit.json"
