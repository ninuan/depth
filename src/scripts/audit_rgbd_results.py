"""Inspect existing RGB-D test configs and referenced checkpoints without a GPU.

This reports the checkpoint file currently present on disk. It cannot prove
which bytes produced an old score if that checkpoint was subsequently replaced.
"""
import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('outputs'))
    parser.add_argument('--dataset', choices=['scannet', 'arkitscenes'], default='scannet')
    parser.add_argument('--output', type=Path, default=Path('outputs/rgbd_checkpoint_audit.json'))
    args = parser.parse_args()
    records, cache = [], {}
    for config_path in sorted(args.root.rglob('run_config.yaml')):
        cfg = OmegaConf.load(config_path)
        if cfg.get('mode') != 'test' or cfg.dataset.name != args.dataset:
            continue
        score_path = config_path.parent / 'metrics/scores_all_avg.json'
        if not score_path.is_file():
            continue
        scores = json.loads(score_path.read_text())
        raw_path = cfg.checkpointing.get('pretrained_model')
        record = dict(run=str(config_path.parent), encoder=cfg.model.encoder.name,
                      split=cfg.dataset.get('test_split', 'test'),
                      test_len=cfg.dataset.get('test_len', -1),
                      checkpoint=raw_path, scores=scores)
        per_scene = score_path.parent / 'scores_psnr_all.json'
        record['scene_count'] = len(json.loads(per_scene.read_text())) if per_scene.is_file() else None
        if raw_path:
            path = Path(raw_path)
            if raw_path not in cache:
                try:
                    state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
                    cache[raw_path] = dict(global_step=state.get('global_step'))
                    del state
                except Exception as exc:
                    cache[raw_path] = dict(global_step=None, checkpoint_error=str(exc))
            record.update(cache[raw_path])
            record['filename'] = path.name
            train_cfg_path = path.parent.parent / 'run_config.yaml'
            if train_cfg_path.is_file():
                train_cfg = OmegaConf.load(train_cfg_path)
                if train_cfg.get('mode') == 'train':
                    record['training_seed'] = train_cfg.seed
                    record['training_depth_weight'] = train_cfg.train.depth_loss_weight
        records.append(record)
        print(f"\n{record['run']}")
        print(f"  encoder={record['encoder']} seed={record.get('training_seed', 'unknown')} "
              f"depth_weight={record.get('training_depth_weight', 'unknown')}")
        print(f"  split={record['split']} scenes={record['scene_count']} "
              f"step={record.get('global_step')} checkpoint={record.get('filename', 'unknown')}")
        print('  ' + '  '.join(f'{k}={scores[k]:.6f}' for k in
                             ('psnr', 'ssim', 'lpips', 'depth_abs_rel', 'depth_rmse', 'depth_a1')
                             if k in scores))
        if record.get('checkpoint_error'):
            print('  CHECKPOINT ERROR: ' + record['checkpoint_error'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2) + '\n')
    print(f'\nSaved {len(records)} result records to {args.output}')


if __name__ == '__main__':
    main()
