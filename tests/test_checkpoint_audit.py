from pathlib import Path
import json
import sys

from omegaconf import OmegaConf
import pytest
import torch

from src.scripts.audit_rgbd_results import main, resolve_checkpoint


@pytest.mark.parametrize('settings,expected', [
    ({'load': 'best.ckpt', 'pretrained_model': None}, ('best.ckpt', 'load')),
    ({'load': 'best.ckpt', 'pretrained_model': 'init.pth'}, ('best.ckpt', 'load')),
    ({'load': None, 'pretrained_model': 'last.ckpt'}, ('last.ckpt', 'pretrained_model')),
    ({'resume': True, 'load': 'best.ckpt', 'pretrained_model': 'init.pth'},
     ('outputs/run/checkpoints/last.ckpt', 'resume')),
    ({'load': None, 'pretrained_model': None}, (None, 'unresolved')),
])
def test_checkpoint_precedence(settings, expected):
    cfg = OmegaConf.create({'checkpointing': settings, 'output_dir': 'outputs/run'})
    assert resolve_checkpoint(cfg, Path('outputs/test')) == expected


def test_resume_falls_back_to_saved_run_directory():
    cfg = OmegaConf.create({'checkpointing': {'resume': True}})
    assert resolve_checkpoint(cfg, Path('outputs/test')) == (
        'outputs/test/checkpoints/last.ckpt', 'resume')


def test_audit_reads_old_load_field_and_training_metadata(tmp_path, monkeypatch):
    train = tmp_path / 'training'
    checkpoint = train / 'checkpoints' / 'best.ckpt'
    checkpoint.parent.mkdir(parents=True)
    torch.save({'global_step': 19000, 'state_dict': {}}, checkpoint)
    OmegaConf.save(OmegaConf.create({'mode': 'train', 'seed': 222123,
                                   'train': {'depth_loss_weight': 0.1}}), train / 'run_config.yaml')
    run = tmp_path / 'test'
    (run / 'metrics').mkdir(parents=True)
    OmegaConf.save(OmegaConf.create({
        'mode': 'test', 'dataset': {'name': 'scannet', 'test_split': 'test'},
        'model': {'encoder': {'name': 'depthsplat'}},
        'checkpointing': {'load': str(checkpoint), 'pretrained_model': None},
    }), run / 'run_config.yaml')
    (run / 'metrics/scores_all_avg.json').write_text('{"psnr": 19.0}')
    (run / 'metrics/scores_psnr_all.json').write_text('[19.0]')
    output = tmp_path / 'audit.json'
    monkeypatch.setattr(sys, 'argv', ['audit', '--root', str(tmp_path), '--output', str(output)])
    main()
    record, = json.loads(output.read_text())
    assert record['global_step'] == 19000
    assert record['training_seed'] == 222123
    assert record['training_depth_weight'] == 0.1
    assert record['checkpoint_source'] == 'load'
    assert record['scene_count'] == 1
