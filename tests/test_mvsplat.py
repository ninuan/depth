"""MVSplat camera conventions, strict weights, label independence and gradients.

CPU encoder tests exercise the actual network; CUDA rendering is covered by
check_scannet --encoder mvsplat --gpu on the experiment machine.
"""
import copy
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from dacite import from_dict
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch

from src.misc.mvsplat_checkpoint import load_mvsplat_weights
from src.model.encoder.encoder_mvsplat import EncoderMVSplat, EncoderMVSplatCfg
from src.model.encoder.mvsplat.common.gaussian_adapter import GaussianAdapterCfg
from src.model.encoder.mvsplat.costvolume import depth_predictor_multiview as cv


def context(batch=1, h=64, w=64):
    result = dict(
        image=torch.rand(batch, 2, 3, h, w),
        intrinsics=torch.tensor([[.8, 0, .5], [0, 1., .5], [0, 0, 1.]])
            .expand(batch, 2, 3, 3).clone(),
        extrinsics=torch.eye(4).expand(batch, 2, 4, 4).clone(),
        near=torch.full((batch, 2), .1), far=torch.full((batch, 2), 10.),
    )
    result['extrinsics'][:, 1, 0, 3] = .1
    result['intrinsics'][:, 1, 0, 0] = .9
    return result


class MVSplatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        torch.manual_seed(42)
        cls.model = EncoderMVSplat(EncoderMVSplatCfg('mvsplat', GaussianAdapterCfg(.5, 15., 4)))

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_config_matches_existing_data_and_rgb_loss(self):
        for dataset in ('scannet', 'arkitscenes'):
            with initialize_config_dir(config_dir=str(Path('config').resolve()), version_base=None):
                base = compose(config_name='main', overrides=[f'+experiment={dataset}'])
                cfg = compose(config_name='main', overrides=[f'+experiment=mvsplat_{dataset}'])
            self.assertEqual(OmegaConf.to_container(base.dataset), OmegaConf.to_container(cfg.dataset))
            self.assertEqual(OmegaConf.to_container(base.loss), OmegaConf.to_container(cfg.loss))
            self.assertEqual(cfg.trainer.max_steps, 30000)
            self.assertEqual(cfg.train.depth_loss_weight, 0)
            self.assertEqual(cfg.data_loader.train.batch_size, 1)
            typed = from_dict(EncoderMVSplatCfg, OmegaConf.to_container(cfg.model.encoder))
            self.assertFalse(typed.supervise_intermediate_depth)

    def test_warp_uses_source_camera_not_reference_camera(self):
        # Source feature is the x coordinate: a +1 source principal point must
        # move reference x=2 to source x=3 for identity pose.
        feature = torch.arange(7.).view(1, 1, 1, 7).expand(1, 1, 5, 7)
        k = torch.eye(3)[None]
        source = k.clone(); source[:, 0, 2] = 1
        pose = torch.eye(4)[None]
        depth = torch.ones(1, 2, 5, 7) * 2
        same = cv.warp_with_pose_depth_candidates(feature, k, pose, depth)
        explicit = cv.warp_with_pose_depth_candidates(feature, k, pose, depth, source_intrinsics=k)
        torch.testing.assert_close(same, explicit, rtol=0, atol=0)
        shifted = cv.warp_with_pose_depth_candidates(feature, k, pose, depth, source_intrinsics=source)
        torch.testing.assert_close(shifted[0, 0, :, 2, 2], torch.tensor([3., 3.]))
        torch.testing.assert_close(same[0, 0, :, 2, 2], torch.tensor([2., 2.]))

    def test_real_network_batch_view_order_depth_and_no_label_input(self):
        self.model.eval()
        c = context(batch=2)
        c['intrinsics'][1, :, 0, 0] += .2
        with patch.object(cv, 'warp_with_pose_depth_candidates', wraps=cv.warp_with_pose_depth_candidates) as warp:
            with torch.no_grad():
                result = self.model(c)
        args = warp.call_args
        ref = args.args[1]
        src = args.kwargs['source_intrinsics']
        torch.testing.assert_close(src, ref[[2, 3, 0, 1]])
        # Each Gaussian's camera-Z equals the depth exposed to the shared metrics.
        means = result['gaussians'].means.reshape(2, 2, 64, 64, 3)
        origin = c['extrinsics'][:, :, None, None, :3, 3]
        forward = c['extrinsics'][:, :, None, None, :3, 2]
        torch.testing.assert_close(result['depths'], ((means-origin)*forward).sum(-1))
        labels = copy.deepcopy(c)
        labels['depth'] = torch.rand(2, 2, 64, 64) * 100
        labels['depth_mask'] = torch.zeros(2, 2, 64, 64, dtype=torch.bool)
        dump = {}
        with torch.no_grad():
            repeated = self.model(labels, visualization_dump=dump)
        torch.testing.assert_close(result['depths'], repeated['depths'], rtol=0, atol=0)
        torch.testing.assert_close(result['gaussians'].means, repeated['gaussians'].means, rtol=0, atol=0)
        self.assertEqual(dump['depth'].shape, (2, 2, 64, 64, 1, 1))
        torch.testing.assert_close(dump['depth'][..., 0, 0], result['depths'], rtol=0, atol=0)

    def test_preserves_official_unit_rays(self):
        from src.model.encoder.mvsplat.common.projection import get_world_rays
        xy = torch.tensor([[1., 0.]])
        _, ray = get_world_rays(xy, torch.eye(4), torch.eye(3))
        torch.testing.assert_close(ray.norm(dim=-1), torch.ones(1))
        self.assertAlmostEqual((2*ray)[0, 2].item(), 2**.5, places=6)

    def test_real_network_gaussian_backward_and_strict_loading(self):
        self.model.train()
        self.model.zero_grad(set_to_none=True)
        result = self.model(context())
        loss = result['gaussians'].means.square().mean() + result['gaussians'].harmonics.square().mean()
        loss.backward()
        for name in ('backbone', 'depth_predictor'):
            grads = [p.grad for p in getattr(self.model, name).parameters() if p.grad is not None]
            self.assertTrue(grads)
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
            self.assertTrue(any(g.abs().max() > 0 for g in grads))
        self.model.zero_grad(set_to_none=True)
        state = {f'encoder.{k}': v for k, v in self.model.state_dict().items()}
        load_mvsplat_weights(self.model, {'state_dict': state})
        state.pop(next(iter(state)))
        with self.assertRaises(RuntimeError):
            load_mvsplat_weights(self.model, {'state_dict': state})

    @unittest.skipUnless(os.environ.get('MVSPLAT_OFFICIAL_CHECKPOINT'), 'Optional downloaded official weights')
    def test_official_checkpoint_compatibility(self):
        weights = torch.load(os.environ['MVSPLAT_OFFICIAL_CHECKPOINT'], map_location='cpu', weights_only=False)
        load_mvsplat_weights(self.model, weights)
        self.assertEqual(len(self.model.state_dict()), 471)
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), 11955734)


if __name__ == '__main__':
    unittest.main()
