"""CPU data preflight, optionally a real CUDA forward/backward/optimizer smoke test.

Run from repository root: python -m src.scripts.check_scannet --dataset_dir PATH
"""
import argparse
import json
from pathlib import Path

from dacite import from_dict, Config
from hydra import compose, initialize_config_dir
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import torch
from torch.utils.data import DataLoader

from src.dataset import get_dataset
from src.dataset.dataset_scannet import DatasetScanNetCfg
from src.model.depth_supervision import masked_depth_loss, valid_depth_mask


def check_cuda_projection():
    from src.model.decoder.cuda_splatting import render_cuda
    device = torch.device('cuda')
    k = torch.tensor([[[.8, 0., .6], [0., 1., .4], [0., 0., 1.]]], device=device)
    mean = torch.tensor([[[0., 0., 2.]]], device=device, requires_grad=True)
    image = render_cuda(
        torch.eye(4, device=device)[None], k, torch.tensor([.1], device=device),
        torch.tensor([10.], device=device), (64, 64), torch.zeros(1, 3, device=device),
        mean, torch.eye(3, device=device)[None, None] * .002,
        torch.ones(1, 1, 3, 1, device=device), torch.ones(1, 1, device=device) * .8,
        use_sh=False,
    )
    weight = image[0].mean(0)
    yy, xx = torch.meshgrid(torch.arange(64, device=device), torch.arange(64, device=device), indexing='ij')
    centroid = torch.stack([(xx*weight).sum(), (yy*weight).sum()]) / weight.sum()
    expected = torch.tensor([.6*64-.5, .4*64-.5], device=device)
    if not torch.allclose(centroid, expected, atol=.3):
        raise RuntimeError(f"CUDA projection mismatch: {centroid.tolist()} vs {expected.tolist()}")
    (image.square().sum()).backward()
    if mean.grad is None or not torch.isfinite(mean.grad).all():
        raise RuntimeError('Rasterizer backward produced invalid gradients')
    return {'centroid': centroid.tolist(), 'expected': expected.tolist()}


def gpu_smoke(cfg_dict, batch):
    if not torch.cuda.is_available():
        raise RuntimeError('--gpu requires an NVIDIA CUDA GPU and the rasterizer extension')
    from src.config import load_typed_root_config
    from src.global_cfg import set_cfg
    from src.loss import get_losses
    from src.model.encoder import get_encoder
    from src.model.decoder import get_decoder
    from src.model.model_wrapper import ModelWrapper
    from src.model.depth_supervision import supervised_depth_loss

    projection = check_cuda_projection()
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    torch.manual_seed(cfg.seed)
    encoder, _ = get_encoder(cfg.model.encoder)
    model = ModelWrapper(cfg.optimizer, cfg.test, cfg.train, encoder, None,
                         get_decoder(cfg.model.decoder, cfg.dataset), get_losses(cfg.loss), None).cuda()
    weights = torch.load(cfg.checkpointing.pretrained_model, map_location='cpu', weights_only=False)
    model.load_state_dict(weights.get('state_dict', weights), strict=True)
    captured = {}
    model.log = lambda name, value, **kwargs: captured.update({name: float(value.detach()) if torch.is_tensor(value) else float(value)})
    batch = model.transfer_batch_to_device(batch, torch.device('cuda'), 0)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optimizer.lr)
    loss = model.training_step(batch, 0)
    if not torch.isfinite(loss):
        raise RuntimeError('Nonfinite training loss')
    loss.backward()
    params = list(model.encoder.depth_predictor.parameters())
    grads = [p.grad for p in params if p.grad is not None]
    if not grads or any(not torch.isfinite(g).all() for g in grads):
        raise RuntimeError('Missing/nonfinite depth network gradients')
    changed = next((p for p in params if p.grad is not None and p.grad.abs().max() > 0), None)
    if changed is None:
        raise RuntimeError('Depth network received zero gradients')
    before = changed.detach().clone()
    torch.nn.utils.clip_grad_norm_(model.parameters(), .5, error_if_nonfinite=True)
    optimizer.step()
    if torch.equal(before, changed.detach()):
        raise RuntimeError('Optimizer failed to update depth network')
    optimizer.zero_grad(set_to_none=True)
    del loss, before
    # Also prove the direct depth loss reaches the real predictor, independently of RGB.
    result = model.encoder(batch['context'], 0)
    direct, _ = supervised_depth_loss(result['depths'], batch['context'], cfg.train.depth_loss_mode,
                                      cfg.train.depth_intermediate_weight)
    direct.backward()
    if not any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max() > 0 for p in params):
        raise RuntimeError('Direct GT depth supervision is disconnected')
    direct_value = float(direct.detach())
    model.zero_grad(set_to_none=True)
    del result, direct
    model.eval()
    with torch.no_grad():
        prediction = model.encoder(batch['context'], 0)
    if prediction['gaussians'].means.shape[0] != batch['context']['image'].shape[0]:
        raise RuntimeError('Evaluation still contains intermediate Gaussian batches')
    return dict(projection=projection, training_metrics=captured,
                direct_depth_loss=direct_value,
                peak_memory_gb=torch.cuda.max_memory_allocated()/1024**3)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset_dir', type=Path, required=True)
    p.add_argument('--output_dir', type=Path, default=Path('outputs/scannet_check'))
    p.add_argument('--max_scenes', type=int, default=8, help='Per split; 0 checks all scenes')
    p.add_argument('--gpu', action='store_true')
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--dataset', choices=['scannet', 'arkitscenes'], default='scannet')
    p.add_argument('--splits', nargs='+', choices=['train', 'val', 'test'], default=['train', 'val', 'test'])
    args = p.parse_args()
    if args.max_scenes < 0:
        p.error('--max_scenes must be nonnegative')
    torch.manual_seed(1234)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[2] / 'config'), version_base=None):
        cfg_dict = compose(config_name='main', overrides=[f'+experiment={args.dataset}'])
    cfg_dict.dataset.roots = [str(args.dataset_dir.resolve())]
    cfg_dict.dataset.augment = False
    if args.checkpoint:
        cfg_dict.checkpointing.pretrained_model = str(args.checkpoint)
    cfg = from_dict(DatasetScanNetCfg, OmegaConf.to_container(cfg_dict.dataset),
                    config=Config(type_hooks={Path: Path}))
    manifest = json.loads((args.dataset_dir / 'manifest.json').read_text())
    if args.dataset == 'arkitscenes':
        locations = [{manifest['physical_locations'][n] for n in manifest['retained_splits'][s]}
                     for s in ('train', 'val', 'test')]
    else:
        locations = [{n.split('_')[0] for n in manifest['retained_splits'][s]} for s in ('train', 'val', 'test')]
    if any(locations[i] & locations[j] for i in range(3) for j in range(i+1, 3)):
        raise ValueError('Physical-location leakage between splits')
    report = {'scene_counts': manifest['scene_counts'], 'splits': {}}
    first_batch = None
    for stage in args.splits:
        dataset = get_dataset(cfg, stage, None)
        count = min(len(dataset), args.max_scenes) if args.max_scenes else len(dataset)
        fractions = []
        for i in range(count):
            sample = dataset[i]
            for view in ('context', 'target'):
                v = sample[view]
                if not torch.isfinite(v['intrinsics']).all() or not torch.isfinite(v['extrinsics']).all():
                    raise ValueError(f'Invalid camera: {sample["scene"]}')
                fractions.append(v['depth_mask'].float().mean().item())
            if stage != 'train':
                repeated = dataset[i]
                if not torch.equal(sample['context']['index'], repeated['context']['index']):
                    raise ValueError('Evaluation sampling is not deterministic')
        batch = next(iter(DataLoader(dataset, batch_size=1, num_workers=0)))
        # Labels matching themselves must yield zero; checks the actual mask/range contract.
        identity = masked_depth_loss(batch['context']['depth'].clamp_min(cfg.near), batch['context'])
        if identity != 0 or not valid_depth_mask(batch['context']).any():
            raise ValueError('Invalid depth/mask/range contract')
        report['splits'][stage] = dict(checked_scenes=count, valid_fraction=float(np.mean(fractions)),
                                      example_scene=batch['scene'][0],
                                      context=batch['context']['index'].tolist(), target=batch['target']['index'].tolist())
        if first_batch is None:
            first_batch = batch
        rgb = (batch['context']['image'][0, 0].permute(1, 2, 0).numpy()*255).astype(np.uint8)
        d = batch['context']['depth'][0, 0].numpy()
        edge = (np.abs(np.diff(d, axis=0, prepend=d[:1])) + np.abs(np.diff(d, axis=1, prepend=d[:, :1]))) > .08
        overlay = rgb.copy(); overlay[edge & (d > 0)] = [255, 32, 32]
        Image.fromarray(np.concatenate([rgb, overlay], axis=1)).save(args.output_dir / f'{stage}_alignment.png')
    if args.gpu:
        report['gpu'] = gpu_smoke(cfg_dict, first_batch)
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print('PASS: ' + ('CPU data + CUDA training/backward' if args.gpu else 'CPU data checks (CUDA not tested)'))


if __name__ == '__main__':
    main()
