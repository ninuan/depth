"""Actual ModelWrapper integration with tiny differentiable CPU encoder/renderer.

The CUDA extension is unavailable on CPU; only its import is stubbed. No native
rasterization or DepthSplat network accuracy is claimed by these tests.
"""
import importlib.util
from pathlib import Path
import sys
import types
import tempfile
import unittest
from unittest.mock import patch

import torch
from hydra import compose, initialize_config_dir
from torch import nn
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, LightningDataModule
from pytorch_lightning.callbacks import ModelCheckpoint

if importlib.util.find_spec('diff_gaussian_rasterization') is None:
    module = types.ModuleType('diff_gaussian_rasterization')
    module.GaussianRasterizationSettings = object
    module.GaussianRasterizer = object
    sys.modules[module.__name__] = module

from src.config import load_typed_root_config
from src.model.model_wrapper import ModelWrapper
from src.model.types import Gaussians
from src.model.decoder.decoder import DecoderOutput, Decoder
from src.model.encoder.encoder import Encoder
from src.loss.loss_mse import LossMse, LossMseCfg, LossMseCfgWrapper
from src.global_cfg import set_cfg
from src.misc.LocalLogger import LocalLogger


class TinyDataModule(LightningDataModule):
    def __init__(self, sample, dataset_name='scannet'):
        super().__init__()
        self.dataset_cfg = types.SimpleNamespace(name=dataset_name)
        self.samples = [sample] * 4

    def train_dataloader(self):
        return DataLoader(self.samples, batch_size=1)

    def val_dataloader(self):
        return DataLoader(self.samples[:1], batch_size=1)


class TinyEncoder(Encoder):
    def __init__(self):
        super().__init__(None)
        self.log_depth = nn.Parameter(torch.tensor(0.5))

    def forward(self, context, *args, **kwargs):
        b, v, _, h, w = context['image'].shape
        pred = self.log_depth.exp().expand(b, v, h, w)
        depths = torch.cat([pred * 1.1, pred]) if self.training else pred
        n = depths.shape[0]
        return dict(depths=depths, gaussians=Gaussians(
            means=self.log_depth.expand(n, 1, 3),
            covariances=torch.eye(3).expand(n, 1, 3, 3),
            harmonics=torch.ones(n, 1, 3, 1), opacities=torch.ones(n, 1)))


class TinyDecoder(Decoder):
    def __init__(self, dataset_cfg):
        super().__init__(None, dataset_cfg)

    def forward(self, g, extrinsics, intrinsics, near, far, shape, **kwargs):
        b, v = extrinsics.shape[:2]
        return DecoderOutput(color=g.means[:, 0, 0].sigmoid().view(b, 1, 1, 1, 1)
                             .expand(b, v, 3, *shape), depth=None)


class TrainingIntegrationTests(unittest.TestCase):
    def setUp(self):
        with initialize_config_dir(config_dir=str(Path('config').resolve()), version_base=None):
            self.raw = compose(config_name='main', overrides=[f'+experiment={getattr(self, "experiment", "scannet")}'])
        set_cfg(self.raw)
        self.cfg = load_typed_root_config(self.raw)
        self.model = ModelWrapper(self.cfg.optimizer, self.cfg.test, self.cfg.train,
                                 TinyEncoder(), None, TinyDecoder(self.cfg.dataset),
                                 [LossMse(LossMseCfgWrapper(LossMseCfg(1.)))], None)
        self.model.log = lambda *a, **kw: None
        def views(v):
            return dict(image=torch.zeros(2, v, 3, 16, 16),
                        depth=torch.ones(2, v, 16, 16),
                        depth_mask=torch.ones(2, v, 16, 16, dtype=torch.bool),
                        near=torch.full((2, v), .1), far=torch.full((2, v), 10.),
                        intrinsics=torch.eye(3).expand(2, v, 3, 3),
                        extrinsics=torch.eye(4).expand(2, v, 4, 4),
                        index=torch.arange(v).expand(2, v))
        self.batch = dict(context=views(2), target=views(1), scene=['a', 'b'])

    def test_actual_training_step_adds_depth_and_backpropagates(self):
        self.model.train()
        supervised = self.model.training_step(self.batch, 0)
        supervised.backward()
        supervised_grad = self.model.encoder.log_depth.grad.item()
        self.model.zero_grad()
        self.cfg.train.depth_loss_weight = 0
        baseline = self.model.training_step(self.batch, 0)
        baseline.backward()
        self.assertGreater(supervised.item(), baseline.item())
        self.assertGreater(supervised_grad, self.model.encoder.log_depth.grad.item())
        # Exact added final/coarse loss with weights 1 and .5.
        expected = .1 * (.5 + .5*(.5 + torch.log(torch.tensor(1.1)).item())) / 1.5
        self.assertAlmostEqual((supervised-baseline).item(), expected, places=6)

    def test_missing_depth_fails_loudly(self):
        del self.batch['context']['depth']
        with self.assertRaisesRegex(ValueError, 'RGB-D labels'):
            self.model.training_step(self.batch, 0)

    def test_hydra_configuration(self):
        self.assertEqual(self.cfg.dataset.view_sampler.name, 'scannet')
        self.assertEqual(self.cfg.train.eval_model_every_n_val, 0)
        self.assertTrue(self.cfg.model.encoder.return_depth)

    def test_lightning_fit_validation_checkpoint_and_resume(self):
        del self.model.log  # Use real Lightning logging and the production training/validation steps.
        sample = {stage: {k: v[0] for k, v in self.batch[stage].items()}
                  for stage in ('context', 'target')}
        sample['scene'] = 'cpu-fixture'
        dm = TinyDataModule(sample, self.cfg.dataset.name)
        with tempfile.TemporaryDirectory() as directory, patch(
                'src.model.model_wrapper.compute_lpips',
                side_effect=lambda gt, pred: (gt-pred).square().mean((1, 2, 3))):
            checkpoint = ModelCheckpoint(directory, monitor='val/psnr', mode='max',
                                         save_top_k=2, save_last=True, save_on_train_epoch_end=False)
            trainer = Trainer(accelerator='cpu', devices=1, max_steps=2,
                              check_val_every_n_epoch=None, val_check_interval=1,
                              num_sanity_val_steps=1, enable_progress_bar=False,
                              logger=LocalLogger(Path(directory)/'logs'), callbacks=[checkpoint],
                              log_every_n_steps=1)
            trainer.fit(self.model, datamodule=dm)
            last = Path(directory)/'last.ckpt'
            self.assertTrue(last.is_file())
            self.assertTrue(Path(checkpoint.best_model_path).is_file())
            self.assertIn('val/depth_abs_rel', trainer.callback_metrics)
            self.assertIn('val/psnr', trainer.callback_metrics)
            # Same scheduler budget as a real interrupted run; advance exactly one more step.
            state = torch.load(last, weights_only=False)
            self.assertEqual(state['global_step'], 2)
            resumed = Trainer(accelerator='cpu', devices=1, max_steps=3,
                              check_val_every_n_epoch=None, val_check_interval=1,
                              num_sanity_val_steps=0, enable_progress_bar=False,
                              logger=LocalLogger(Path(directory)/'resume'),
                              enable_checkpointing=False, log_every_n_steps=1)
            resumed.fit(self.model, datamodule=dm, ckpt_path=str(last))
            self.assertEqual(resumed.global_step, 3)
            self.assertTrue(list((Path(directory)/'logs').rglob('metrics.csv')))


class ARKitTrainingIntegrationTests(TrainingIntegrationTests):
    experiment = 'arkitscenes'


if __name__ == '__main__':
    unittest.main()
