"""Audit and plot aligned baseline/depth/mvsplat exports; CPU only, no inference.

Uses raw metric Z arrays with a shared 0.1--10 m scale and shared masks.
RGB errors come from exported 8-bit PNGs; RGB scores come from evaluation JSON.
All exported scenes are retained. Input-view depth is distinct from target RGB.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from omegaconf import OmegaConf
from PIL import Image


METHODS = ('mvsplat', 'baseline', 'depth')
LABELS = ('MVSplat', 'DepthSplat RGB', 'DepthSplat + depth')
METRICS = ('psnr', 'ssim', 'lpips', 'depth_abs_rel', 'depth_rmse', 'depth_a1')


def rgb(path):
    return np.asarray(Image.open(path).convert('RGB'))


def check_equal(a, b, message):
    if not np.array_equal(a, b, equal_nan=True):
        raise ValueError(message)


def load_exports(root):
    configs = {m: OmegaConf.load(root / m / 'run_config.yaml') for m in METHODS}
    reference = configs['baseline']
    fields = ('dataset', 'seed', 'data_loader.test')
    for m, cfg in configs.items():
        for field in fields:
            if OmegaConf.select(cfg, field) != OmegaConf.select(reference, field):
                raise ValueError(f'Protocol mismatch: {m}/{field}')
        if (root / m / 'dataset_manifest_0.json').read_bytes() != (
                root / 'baseline/dataset_manifest_0.json').read_bytes():
            raise ValueError(f'Dataset manifest mismatch: {m}')
    scenes = sorted(p.name for p in (root / 'baseline/images').iterdir() if p.is_dir())
    scores = {}
    for m in METHODS:
        if scenes != sorted(p.name for p in (root / m / 'images').iterdir() if p.is_dir()):
            raise ValueError(f'Scene mismatch: {m}')
        scores[m] = {k: json.loads((root / m / f'metrics/scores_{k}_all.json').read_text())
                     for k in METRICS}
        if any(len(v) != len(scenes) for v in scores[m].values()):
            raise ValueError(f'Score count mismatch: {m}')
    records, data = [], {}
    for index, scene in enumerate(scenes):
        base = root / 'baseline/images' / scene
        targets = sorted(p.name.removesuffix('_gt.png') for p in (base / 'color').glob('*_gt.png'))
        views = sorted(p.name.removesuffix('_gt.npy') for p in (base / 'depth').glob('*_gt.npy'))
        if len(targets) != 1 or len(views) != 2:
            raise ValueError(f'Expected two inputs and one target: {scene}')
        target = targets[0]
        gt_rgb = rgb(base / f'color/{target}_gt.png')
        gt = np.stack([np.load(base / f'depth/{v}_gt.npy') for v in views])
        mask = np.stack([np.load(base / f'depth/{v}_mask.npy') for v in views]).astype(bool)
        if not mask.any() or not np.isfinite(gt[mask]).all() or (gt[mask] <= 0).any():
            raise ValueError(f'Invalid GT/mask: {scene}')
        item = dict(gt_rgb=gt_rgb, gt=gt, mask=mask, views=views, target=target)
        record = dict(scene=scene, context=views, target=target, valid_fraction=float(mask.mean()), methods={})
        for m in METHODS:
            directory = root / m / 'images' / scene
            if sorted(p.name for p in (directory / 'color').glob('*.png')) != sorted(
                    p.name for p in (base / 'color').glob('*.png')):
                raise ValueError(f'View file mismatch: {m}/{scene}')
            check_equal(rgb(directory / f'color/{target}_gt.png'), gt_rgb, f'RGB GT: {m}/{scene}')
            for v in views:
                check_equal(rgb(directory / f'color/input_{v}.png'), rgb(base / f'color/input_{v}.png'),
                            f'Input RGB: {m}/{scene}/{v}')
                for suffix in ('gt', 'mask'):
                    check_equal(np.load(directory / f'depth/{v}_{suffix}.npy'),
                                np.load(base / f'depth/{v}_{suffix}.npy'), f'Depth {suffix}: {m}/{scene}/{v}')
            pred = np.stack([np.load(directory / f'depth/{v}.npy') for v in views])
            if pred.shape != gt.shape or not np.isfinite(pred).all() or (pred <= 0).any():
                raise ValueError(f'Invalid prediction: {m}/{scene}')
            p, g = pred[mask], gt[mask]
            computed = dict(depth_abs_rel=float(np.mean(np.abs(p-g)/g)),
                            depth_rmse=float(np.sqrt(np.mean((p-g)**2))),
                            depth_a1=float(np.mean(np.maximum(p/g, g/p) < 1.25)))
            reported = {k: scores[m][k][index] for k in METRICS}
            for k, value in computed.items():
                if not np.isclose(value, reported[k], atol=2e-6, rtol=2e-5):
                    raise ValueError(f'Raw/exported score or scene ordering mismatch: {m}/{scene}/{k}')
            record['methods'][m] = reported
            item[m] = dict(rgb=rgb(directory / f'color/{target}.png'), depth=pred)
        data[scene] = item
        records.append(record)
    return configs, records, data


def show(ax, array, title, cmap=None, limits=None, mask=None):
    if mask is not None:
        array = np.ma.masked_where(~mask, array)
    kwargs = {}
    if cmap:
        palette = plt.get_cmap(cmap).copy()
        palette.set_bad('#bbbbbb')
        kwargs = dict(cmap=palette, vmin=limits[0], vmax=limits[1])
    ax.imshow(array, interpolation='nearest', **kwargs)
    ax.set_title(title, fontsize=9)
    ax.set_axis_off()


def add_bar(fig, cmap, limits, rect, label):
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(*limits), cmap=cmap),
                      cax=fig.add_axes(rect), orientation='horizontal', extend='max')
    bar.set_label(label, fontsize=9)
    bar.ax.tick_params(labelsize=8)


def paper_panel(root, output, records, data, dataset_label, split, seed):
    """Use the first two fixed scenes, regardless of which method wins."""
    selected = records[:2]
    fig, axes = plt.subplots(3 * len(selected), 4, figsize=(8.2, 8.6), squeeze=False)
    for i, record in enumerate(selected):
        scene, item = record['scene'], data[record['scene']]
        row = i * 3
        show(axes[row, 0], item['gt_rgb'], f'{scene} | target RGB GT')
        show(axes[row+1, 0], item['gt'][0], 'Input-view depth GT', 'viridis', (.1, 10), item['mask'][0])
        input_image = rgb(root / 'baseline/images' / scene / f"color/input_{item['views'][0]}.png")
        show(axes[row+2, 0], input_image, 'Input RGB | error reference')
        for col, (m, label) in enumerate(zip(METHODS, LABELS), 1):
            show(axes[row, col], item[m]['rgb'], label)
            show(axes[row+1, col], item[m]['depth'][0], 'Predicted depth', 'viridis', (.1, 10), item['mask'][0])
            show(axes[row+2, col], np.abs(item[m]['depth'][0]-item['gt'][0]),
                 'Absolute depth error', 'magma', (0, 1), item['mask'][0])
    for ax in axes.flat:
        ax.title.set_fontsize(7)
    fig.suptitle(f'{dataset_label} {split} | seed {seed} | first two fixed scenes', fontsize=11)
    fig.subplots_adjust(top=.94, bottom=.10, left=.012, right=.988, hspace=.28, wspace=.03)
    add_bar(fig, 'viridis', (.1, 10), [.12, .055, .30, .012], 'Camera-Z depth (m)')
    add_bar(fig, 'magma', (0, 1), [.58, .055, .30, .012], 'Depth error (m); >1 saturated')
    fig.text(.5, .012, 'Gray: invalid sensor depth. Target RGB and input depth use different views. No scale alignment.',
             ha='center', fontsize=7)
    fig.savefig(output / 'paper_panel_first2.png', dpi=300)
    fig.savefig(output / 'paper_panel_first2.pdf')
    plt.close(fig)


def plot(root, output):
    configs, records, data = load_exports(root)
    seed = configs['baseline'].seed
    dataset_name = configs['baseline'].dataset.name
    dataset_label = {'scannet': 'ScanNet', 'arkitscenes': 'ARKitScenes'}.get(dataset_name, dataset_name)
    selection_path = root / 'selection_plan.json'
    selection = json.loads(selection_path.read_text()) if selection_path.is_file() else None
    if selection is not None:
        if (selection['scenes'] != [r['scene'] for r in records] or
                selection['dataset'] != dataset_name or selection['seed'] != seed or
                selection['split'] != configs['baseline'].dataset.test_split):
            raise ValueError('Exported scenes/protocol do not match predeclared selection plan')
    output.mkdir(parents=True, exist_ok=True)
    n = len(records)
    paper_panel(root, output, records, data, dataset_label, configs['baseline'].dataset.test_split, seed)
    for kind in ('rgb', 'depth_context0', 'depth_context1'):
        fig, axes = plt.subplots(n, 4, figsize=(11, 2.15*n), squeeze=False)
        for row, record in enumerate(records):
            item = data[record['scene']]
            if kind == 'rgb':
                show(axes[row, 0], item['gt_rgb'], f"{record['scene']} | target {item['target']} GT")
                for col, (m, label) in enumerate(zip(METHODS, LABELS), 1):
                    show(axes[row, col], item[m]['rgb'], f"{label} | PSNR {record['methods'][m]['psnr']:.2f}")
            else:
                v = int(kind[-1])
                show(axes[row, 0], item['gt'][v], f"{record['scene']} | input {item['views'][v]} GT",
                     'viridis', (0.1, 10), item['mask'][v])
                for col, (m, label) in enumerate(zip(METHODS, LABELS), 1):
                    show(axes[row, col], item[m]['depth'][v], label, 'viridis', (0.1, 10), item['mask'][v])
        fig.suptitle(f"{dataset_label} {configs['baseline'].dataset.test_split} | seed {seed} | {n} fixed scenes", fontsize=13)
        fig.subplots_adjust(top=.95, bottom=.05, left=.01, right=.99, hspace=.25, wspace=.02)
        if kind != 'rgb':
            add_bar(fig, 'viridis', (.1, 10), [.32, .022, .36, .008], 'Camera-Z depth (m); gray = invalid GT')
        fig.savefig(output / f'{kind}_overview.png', dpi=170)
        plt.close(fig)
    for record in records:
        scene, item = record['scene'], data[record['scene']]
        fig, axes = plt.subplots(6, 4, figsize=(12, 13.5))
        show(axes[0, 0], item['gt_rgb'], f"Target {item['target']} GT")
        show(axes[1, 0], item['gt_rgb'], 'Target GT | RGB error reference')
        for col, (m, label) in enumerate(zip(METHODS, LABELS), 1):
            show(axes[0, col], item[m]['rgb'], f"{label} | PSNR {record['methods'][m]['psnr']:.2f}")
            error = np.abs(item[m]['rgb'].astype(float)-item['gt_rgb'].astype(float)).mean(-1)/255
            show(axes[1, col], error, 'RGB mean absolute error', 'magma', (0, .25))
        for v in range(2):
            row = 2+v*2
            show(axes[row, 0], item['gt'][v], f"Input {item['views'][v]} depth GT",
                 'viridis', (.1, 10), item['mask'][v])
            input_img = rgb(root / 'baseline/images' / scene / f"color/input_{item['views'][v]}.png")
            show(axes[row+1, 0], input_img, f"Input {item['views'][v]} RGB | depth error reference")
            for col, (m, label) in enumerate(zip(METHODS, LABELS), 1):
                show(axes[row, col], item[m]['depth'][v], label, 'viridis', (.1, 10), item['mask'][v])
                show(axes[row+1, col], np.abs(item[m]['depth'][v]-item['gt'][v]),
                     'Absolute depth error', 'magma', (0, 1), item['mask'][v])
        fig.suptitle(f"{dataset_label} {configs['baseline'].dataset.test_split} | {scene} | seed {seed}\n"
                     'Target-view RGB and input-view metric Z depth; no depth scale alignment', fontsize=13)
        fig.subplots_adjust(top=.925, bottom=.08, left=.01, right=.99, hspace=.25, wspace=.03)
        add_bar(fig, 'viridis', (.1, 10), [.05, .04, .25, .009], 'Depth (m); gray = invalid GT')
        add_bar(fig, 'magma', (0, 1), [.38, .04, .25, .009], 'Depth absolute error (m); >1 saturated')
        add_bar(fig, 'magma', (0, .25), [.71, .04, .25, .009], 'RGB error [0,1]; >0.25 saturated')
        fig.savefig(output / f'scene_{scene}.png', dpi=170)
        plt.close(fig)
    summary = dict(dataset=dataset_name, split=configs['baseline'].dataset.test_split, seed=configs['baseline'].seed,
                   scene_count=n, input_view_count=2*n, alignment_checks='passed',
                   raw_depth_metrics_match=True,
                   selection_plan=selection,
                   checkpoints={m: str(configs[m].checkpointing.pretrained_model) for m in METHODS},
                   means={m: {k: float(np.mean([r['methods'][m][k] for r in records])) for k in METRICS}
                          for m in METHODS},
                   depth_vs_baseline_improved={k: sum(
                       (r['methods']['depth'][k] > r['methods']['baseline'][k]) if k in ('psnr', 'ssim', 'depth_a1')
                       else (r['methods']['depth'][k] < r['methods']['baseline'][k]) for r in records)
                       for k in METRICS}, scenes=records)
    (output / 'comparison_audit.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'scenes'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    plot(args.root, args.output)
