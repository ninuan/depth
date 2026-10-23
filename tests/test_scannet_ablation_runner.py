import json
from pathlib import Path
import sys

from hydra import compose, initialize_config_dir
import pytest
from omegaconf import OmegaConf
import torch

from src.scripts import scannet_ablation_fixed30000 as runner


@pytest.fixture
def training_cfg():
    repo = Path(__file__).resolve().parents[1]
    with initialize_config_dir(version_base=None, config_dir=str(repo / 'config')):
        return compose(config_name='main', overrides=['+experiment=scannet', 'data_loader.train.batch_size=1'])


def test_evaluation_config_preserves_model_and_loss(training_cfg, tmp_path):
    cfg = runner.evaluation_config(training_cfg, tmp_path / 'last.ckpt', tmp_path / 'data', tmp_path / 'run')
    assert cfg.model == training_cfg.model
    assert cfg.loss == training_cfg.loss
    assert cfg.train == training_cfg.train
    assert cfg.mode == 'test' and training_cfg.mode == 'train'
    assert cfg.checkpointing.load.endswith('last.ckpt')
    assert cfg.checkpointing.pretrained_model is None
    assert cfg.dataset.test_split == 'val' and cfg.dataset.test_len == -1
    OmegaConf.save(cfg, tmp_path / 'evaluation.yaml')
    with initialize_config_dir(version_base=None, config_dir=str(tmp_path)):
        reloaded = compose(config_name='evaluation')
        assert reloaded == cfg


def scores_at(run, value):
    metrics = run / 'metrics'
    metrics.mkdir(parents=True, exist_ok=True)
    (metrics / 'scores_all_avg.json').write_text(json.dumps({k: value for k in runner.METRICS}))
    for key in runner.METRICS:
        (metrics / f'scores_{key}_all.json').write_text(json.dumps([value] * 48))


def test_incomplete_scores_rejected(tmp_path):
    scores_at(tmp_path, 0.5)
    (tmp_path / 'metrics/scores_lpips_all.json').write_text('[0.5]')
    with pytest.raises(ValueError, match='48 finite'):
        runner.read_scores(tmp_path)


def prepare_repo(tmp_path, monkeypatch, cfg):
    monkeypatch.setattr(runner, '__file__', str(tmp_path / 'src/scripts/runner.py'))
    monkeypatch.chdir(tmp_path)
    dataset = tmp_path / 'datasets/scannet_processed_v2'
    (dataset / 'val').mkdir(parents=True)
    (dataset / 'manifest.json').write_text('{"format_version": 2}')
    (dataset / 'val/index.json').write_text(json.dumps({str(i): f'{i}.torch' for i in range(48)}))
    for _, directory, seed, weight, alpha in runner.jobs():
        path = tmp_path / directory
        (path / 'checkpoints').mkdir(parents=True)
        current = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        current.seed = seed
        # Actual cloud runs use a different loader RNG per training-seed block.
        current.data_loader.train.seed = 1234 if seed == 111123 else seed
        current.train.depth_loss_weight = weight
        current.train.depth_intermediate_weight = alpha
        OmegaConf.save(current, path / 'run_config.yaml')
        torch.save({'global_step': 30000, 'state_dict': {}}, path / 'checkpoints/last.ckpt')
    monkeypatch.setattr(sys, 'argv', ['ablation'])


def test_all_checked_before_inference_and_completed_runs_reused(tmp_path, monkeypatch, training_cfg):
    prepare_repo(tmp_path, monkeypatch, training_cfg)
    calls = []

    def fake_inference(command, **kwargs):
        cfg = OmegaConf.load(Path(command[command.index('--config-path') + 1]) / f'{command[-1]}.yaml')
        assert cfg.mode == 'test' and cfg.dataset.test_split == 'val'
        calls.append(command[-1])
        scores_at(Path(cfg.output_dir), 0.5)

    monkeypatch.setattr(runner.subprocess, 'run', fake_inference)
    # The last item fails preflight, so even the first item must not launch.
    checkpoint = tmp_path / runner.jobs()[-1][1] / 'checkpoints/last.ckpt'
    torch.save({'global_step': 21000}, checkpoint)
    with pytest.raises(ValueError, match='Expected step 30000'):
        runner.main()
    assert not calls
    torch.save({'global_step': 30000}, checkpoint)
    runner.main()
    assert len(calls) == 8
    runner.main()
    assert len(calls) == 8
    summary_path = tmp_path / 'outputs/scannet_ablation_fixed30000/summary.json'
    summary = json.loads(summary_path.read_text())
    assert len(summary['runs']) == 8
    assert len(summary['stage_three_seed_statistics']) == 2
    assert len(summary['weight_sensitivity_seed111123']) == 3
    loaders = {r['training_seed']: r['training_data_loader']['seed'] for r in summary['runs']}
    assert loaders == {111123: 1234, 222123: 222123, 333123: 333123}
    # Changed checkpoint provenance must not silently reuse cached scores.
    torch.save({'global_step': 30000, 'changed': True}, checkpoint)
    with pytest.raises(ValueError, match='different provenance'):
        runner.main()
    assert len(calls) == 8


@pytest.mark.parametrize('directory,field,value,message', [
    ('outputs/scannet_depth_finalonly_seed222123', 'seed', 1234, 'Within-seed loader RNG mismatch'),
    ('outputs/scannet_depth_seed222123', 'batch_size', 2, 'data_loader.train.batch_size'),
    ('outputs/scannet_depth_seed222123', 'num_workers', 0, 'data_loader.train.num_workers'),
])
def test_real_mismatches_still_block_inference(tmp_path, monkeypatch, training_cfg,
                                              directory, field, value, message):
    prepare_repo(tmp_path, monkeypatch, training_cfg)
    path = tmp_path / directory / 'run_config.yaml'
    cfg = OmegaConf.load(path)
    cfg.data_loader.train[field] = value
    OmegaConf.save(cfg, path)

    def unexpected_inference(*args, **kwargs):
        pytest.fail('Preflight must reject mismatched controls before GPU inference')

    monkeypatch.setattr(runner.subprocess, 'run', unexpected_inference)
    with pytest.raises(ValueError, match=message):
        runner.main()


def test_summary_uses_sample_sd_and_paired_direction():
    records = []
    for name, _, seed, _, _ in runner.jobs():
        value = runner.SEEDS.index(seed) + (0.25 if name.startswith('finalonly') else 0)
        records.append({'name': name, 'scores': dict.fromkeys(runner.METRICS, value)})
    summary = runner.summarize(records)
    assert summary['stage_three_seed_statistics']['multistage']['psnr'] == {'mean': 1, 'sample_sd': 1}
    assert all(v == 0.25 for v in summary['finalonly_minus_multistage']['psnr'].values())
