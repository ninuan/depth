#!/usr/bin/env bash
# Predeclare the first eight validation scenes; never select by method performance.
set -euo pipefail
cd "$(dirname "$0")/.."
dataset=${1:-datasets/scannet_processed_v2}
output=${2:-outputs/scannet_three_method_visuals_seed111123}
baseline=outputs/scannet_rgb/checkpoints/last.ckpt
depth=outputs/scannet_depth/checkpoints/last.ckpt
mvsplat=outputs/mvsplat_scannet_seed111123/checkpoints/last.ckpt
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1

python - "$dataset" "$output" "$baseline" "$depth" "$mvsplat" <<'PY'
import hashlib, json, sys
from pathlib import Path
import torch
from omegaconf import OmegaConf

dataset, output = map(Path, sys.argv[1:3])
index = json.loads((dataset / 'val/index.json').read_text())
if len(index) != 48:
    raise SystemExit(f'Expected the fixed 48-scene validation set, got {len(index)}')
if output.exists():
    raise SystemExit(f'Output already exists; use a new output path to preserve exports: {output}')
records = {}
for method, raw, weight in zip(('baseline', 'depth', 'mvsplat'), sys.argv[3:], (0.0, 0.1, 0.0)):
    path = Path(raw)
    cfg = OmegaConf.load(path.parent.parent / 'run_config.yaml')
    if (cfg.mode != 'train' or cfg.seed != 111123 or cfg.dataset.name != 'scannet'
            or cfg.train.depth_loss_weight != weight):
        raise SystemExit(f'Training metadata mismatch: {path}')
    if method == 'depth' and cfg.train.depth_intermediate_weight != 0.5:
        raise SystemExit(f'Expected Multi-stage depth supervision: {path}')
    expected_encoder = 'mvsplat' if method == 'mvsplat' else 'depthsplat'
    if cfg.model.encoder.name != expected_encoder:
        raise SystemExit(f'Encoder mismatch: {path}')
    state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    step = state.get('global_step')
    del state
    if step != 30000:
        raise SystemExit(f'Expected step 30000: {path}; got {step}')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    records[method] = {'checkpoint': str(path), 'step': step, 'sha256': digest.hexdigest()}
    print(f'Checked step 30000: {path}')
plan = {'dataset': 'scannet', 'split': 'val', 'seed': 111123,
        'selection_rule': 'First eight scene IDs in sorted validation index, before inference',
        'scenes': sorted(index)[:8], 'checkpoints': records}
output.mkdir(parents=True)
(output / 'selection_plan.json').write_text(json.dumps(plan, indent=2) + '\n')
print(json.dumps(plan, indent=2))
PY

common=(mode=test "dataset.roots=[\"$dataset\"]"
        dataset.test_split=val dataset.test_len=8 seed=111123
        checkpointing.load=null checkpointing.resume=false checkpointing.no_strict_load=false
        data_loader.test.seed=2345 data_loader.test.batch_size=1
        test.compute_scores=true test.save_image=true test.save_gt_image=true
        test.save_input_images=true test.save_depth=true test.save_depth_npy=true)
for method in baseline depth; do
  weight=0.0
  checkpoint=$baseline
  if [[ "$method" == depth ]]; then weight=0.1; checkpoint=$depth; fi
  python -m src.main +experiment=scannet "${common[@]}" \
    "checkpointing.pretrained_model=$checkpoint" "train.depth_loss_weight=$weight" \
    "output_dir=$output/$method" "test.output_path=$output/$method"
done
bash scripts/mvsplat_eval.sh scannet "$dataset" "$mvsplat" "$output/mvsplat" "${common[@]}"
