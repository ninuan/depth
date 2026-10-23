"""Evaluate the frozen ScanNet ablations at step 30000; no training or selection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

import numpy as np
from omegaconf import OmegaConf
import torch

SEEDS = (111123, 222123, 333123)
METRICS = ('psnr', 'ssim', 'lpips', 'depth_abs_rel', 'depth_rmse', 'depth_a1')


def jobs():
    result = []
    for seed in SEEDS:
        suffix = '' if seed == 111123 else f'_seed{seed}'
        result.append((f'multistage_seed{seed}', f'outputs/scannet_depth{suffix}', seed, 0.1, 0.5))
        result.append((f'finalonly_seed{seed}', f'outputs/scannet_depth_finalonly_seed{seed}', seed, 0.1, 0.0))
    for label, weight in (('w005', 0.05), ('w020', 0.2)):
        result.append((f'{label}_seed111123', f'outputs/scannet_depth_{label}_seed111123', 111123, weight, 0.5))
    return result


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def protocol(cfg):
    """Shared training settings; loader RNG is checked within each seed block."""
    fields = ('model', 'optimizer', 'loss', 'data_loader.train',
              'dataset.image_shape', 'dataset.view_sampler', 'dataset.near',
              'dataset.far', 'trainer.max_steps', 'train.depth_loss_mode',
              'train.intermediate_loss_weight', 'train.l1_loss', 'train.depth_mode',
              'train.forward_depth_only', 'train.train_ignore_large_loss')
    value = {key: OmegaConf.to_container(OmegaConf.create({'v': OmegaConf.select(cfg, key)}),
                                       resolve=True)['v'] for key in fields}
    value['data_loader.train'].pop('seed', None)
    return value


def config_differences(reference, current, prefix=''):
    """Show actual leaf values instead of just the enclosing config section."""
    if isinstance(reference, dict) and isinstance(current, dict):
        result = {}
        for key in sorted(reference.keys() | current.keys()):
            path = f'{prefix}.{key}' if prefix else key
            result.update(config_differences(reference.get(key, '<missing>'),
                                             current.get(key, '<missing>'), path))
        return result
    return {} if reference == current else {prefix: {'reference': reference, 'current': current}}


def evaluation_config(cfg, checkpoint, dataset, output):
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg.mode = 'test'
    cfg.seed = 111123  # Evaluation RNG is shared; training seeds identify checkpoints.
    cfg.dataset.roots = [str(dataset)]
    cfg.dataset.test_split = 'val'
    cfg.dataset.test_len = -1
    cfg.dataset.overfit_to_scene = None
    cfg.output_dir = str(output)
    cfg.test.output_path = str(output)
    cfg.test.compute_scores = True
    for key in list(cfg.test):
        if key.startswith('save_'):
            cfg.test[key] = False
    cfg.checkpointing.load = str(checkpoint)
    cfg.checkpointing.resume = False
    cfg.checkpointing.no_strict_load = False
    for key in list(cfg.checkpointing):
        if key.startswith('pretrained_'):
            cfg.checkpointing[key] = None
    cfg.data_loader.test.batch_size = 1
    cfg.data_loader.test.seed = 2345
    cfg.wandb.mode = 'disabled'
    return cfg


def read_scores(run, count=48):
    average = json.loads((run / 'metrics/scores_all_avg.json').read_text())
    result = {}
    for key in METRICS:
        values = np.asarray(json.loads((run / f'metrics/scores_{key}_all.json').read_text()), dtype=float)
        if values.shape != (count,) or not np.isfinite(values).all():
            raise ValueError(f'Expected {count} finite per-scene scores: {run}/{key}')
        if not np.isclose(values.mean(), average[key], atol=1e-6, rtol=1e-6):
            raise ValueError(f'Average disagrees with per-scene scores: {run}/{key}')
        result[key] = float(average[key])
    return result


def summarize(records):
    by_name = {r['name']: r for r in records}
    stages, deltas = {}, {}
    for group in ('multistage', 'finalonly'):
        stages[group] = {}
        for key in METRICS:
            values = [by_name[f'{group}_seed{s}']['scores'][key] for s in SEEDS]
            stages[group][key] = {'mean': statistics.mean(values), 'sample_sd': statistics.stdev(values)}
    for key in METRICS:
        values = [by_name[f'finalonly_seed{s}']['scores'][key] -
                  by_name[f'multistage_seed{s}']['scores'][key] for s in SEEDS]
        deltas[key] = dict(zip(map(str, SEEDS), values))
    weights = {label: by_name[name]['scores'] for label, name in
               (('0.05', 'w005_seed111123'), ('0.1', 'multistage_seed111123'), ('0.2', 'w020_seed111123'))}
    return {'split': 'val', 'scene_count': 48, 'step': 30000,
            'main_method': {'depth_weight': 0.1, 'depth_intermediate_weight': 0.5},
            'stage_three_seed_statistics': stages, 'finalonly_minus_multistage': deltas,
            'weight_sensitivity_seed111123': weights, 'runs': records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('datasets/scannet_processed_v2'))
    parser.add_argument('--output', type=Path, default=Path('outputs/scannet_ablation_fixed30000'))
    parser.add_argument('--check-only', action='store_true', help='Validate all checkpoints/configs without inference')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    os.chdir(repo)
    dataset, output = args.dataset.resolve(), args.output.resolve()
    manifest = json.loads((dataset / 'manifest.json').read_text())
    if manifest.get('format_version') != 2:
        raise ValueError('Expected the existing v2 processed dataset')
    index = json.loads((dataset / 'val/index.json').read_text())
    if len(index) != 48:
        raise ValueError(f'Expected the fixed 48-scene validation set, got {len(index)}')
    # Detect changed source/config/data/checkpoint before reusing completed evaluations.
    code = hashlib.sha256()
    for path in sorted((repo / 'src').rglob('*.py')):
        code.update(str(path.relative_to(repo)).encode())
        code.update(path.read_bytes())
    common = {'source_sha256': code.hexdigest(), 'manifest_sha256': sha256(dataset / 'manifest.json'),
              'val_index_sha256': sha256(dataset / 'val/index.json')}
    prepared, reference, loader_seeds = [], None, {}
    for name, directory, seed, weight, intermediate in jobs():
        train_dir = repo / directory
        cfg = OmegaConf.load(train_dir / 'run_config.yaml')
        if (cfg.mode != 'train' or cfg.dataset.name != 'scannet' or cfg.seed != seed or
                cfg.train.depth_loss_weight != weight or cfg.train.depth_intermediate_weight != intermediate):
            raise ValueError(f'Training config mismatch: {directory}; expected seed={seed}, lambda={weight}, alpha={intermediate}')
        signature = protocol(cfg)
        if reference is None:
            reference = signature
        elif signature != reference:
            changed = config_differences(reference, signature)
            raise ValueError(f'Other training settings differ in {directory}: '
                             f'{json.dumps(changed)}; inspect before comparing')
        loader = OmegaConf.to_container(cfg.data_loader.train, resolve=True)
        if seed in loader_seeds and loader['seed'] != loader_seeds[seed]:
            raise ValueError(f'Within-seed loader RNG mismatch in {directory}: training seed={seed}, '
                             f'data_loader.train.seed reference={loader_seeds[seed]}, current={loader["seed"]}')
        loader_seeds[seed] = loader['seed']
        checkpoint = train_dir / 'checkpoints/last.ckpt'
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        step = state.get('global_step')
        del state
        if step != 30000:
            raise ValueError(f'Expected step 30000, got {step}: {checkpoint}')
        run = output / name
        eval_cfg = evaluation_config(cfg, checkpoint, dataset, run)
        config_text = OmegaConf.to_yaml(eval_cfg, resolve=True)
        fingerprint = dict(common, checkpoint_sha256=sha256(checkpoint),
                           eval_config_sha256=hashlib.sha256(config_text.encode()).hexdigest())
        record = dict(name=name, checkpoint=str(checkpoint), global_step=step, training_seed=seed,
                      training_data_loader=loader, depth_weight=weight,
                      depth_intermediate_weight=intermediate, fingerprint=fingerprint)
        receipt = run / 'completed.json'
        complete = False
        if receipt.is_file():
            old = json.loads(receipt.read_text())
            if old.get('fingerprint') != fingerprint:
                raise ValueError(f'Completed run has different provenance: {run}; use a new --output directory')
            record['scores'] = read_scores(run)
            complete = True
        prepared.append((run, config_text, record, complete))
        print(f'Checked: {name}, step={step}, lambda={weight}, alpha={intermediate}, '
              f'train_loader_seed={loader["seed"]}', flush=True)
    if args.check_only:
        print('All eight checkpoints/configs passed; no GPU inference was run.')
        return
    environment = dict(os.environ)
    environment.setdefault('CUDA_VISIBLE_DEVICES', '0')
    environment.setdefault('OMP_NUM_THREADS', '4')
    environment['PYTHONUNBUFFERED'] = '1'
    configs = output / 'evaluation_configs'
    configs.mkdir(parents=True, exist_ok=True)
    records = []
    for run, config_text, record, complete in prepared:
        if complete:
            print(f'Skipping verified completed evaluation: {record["name"]}', flush=True)
        else:
            name = record['name']
            (configs / f'{name}.yaml').write_text(config_text)
            print(f'Evaluating {name} on all 48 validation scenes', flush=True)
            subprocess.run([sys.executable, '-m', 'src.main', '--config-path', str(configs),
                            '--config-name', name], cwd=repo, env=environment, check=True)
            record['scores'] = read_scores(run)
            (run / 'completed.json').write_text(json.dumps(record, indent=2) + '\n')
        records.append(record)
    summary = summarize(records)
    target = output / 'summary.json'
    target.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'runs'}, indent=2))
    print(f'Saved: {target}')


if __name__ == '__main__':
    main()
