"""CPU regression checks for the ScanNet geometry/data/supervision contract."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from src.scripts.convert_scannet import (align_registered_rgb, scene_example,
                                         split_scenes, depth_overlap)
from src.dataset.dataset_scannet import DatasetScanNet, DatasetScanNetCfg
from src.dataset.view_sampler import get_view_sampler
from src.dataset.view_sampler.view_sampler_scannet import ViewSamplerScanNetCfg
from src.dataset.shims.crop_shim import apply_crop_shim_to_views
from src.dataset.shims.augmentation_shim import reflect_views
from src.dataset.shims.patch_shim import apply_patch_shim_to_views
from src.geometry.pinhole import projection_from_intrinsics
from src.model.depth_supervision import supervised_depth_loss, masked_depth_loss, depth_metrics


def make_scene(root, name="scene0000_00", frames=5):
    scene = root / name
    for folder in ("color", "depth", "pose"):
        (scene / folder).mkdir(parents=True)
    k = np.eye(4)
    k[0, 0] = k[1, 1] = 55
    k[0, 2], k[1, 2] = 31.0, 23.0
    for kind in ("color", "depth"):
        np.savetxt(scene / f"intrinsics_{kind}.txt", k)
    for i in range(frames):
        depth = np.full((48, 64), 2000, np.uint16)
        depth[0:4, 0:4] = 0
        rgb = np.full((48, 64, 3), 100 + i, np.uint8)
        pose = np.eye(4); pose[0, 3] = i * .04
        Image.fromarray(depth).save(scene / "depth" / f"{i*100:06}.png")
        Image.fromarray(rgb).save(scene / "color" / f"{i*100:06}.jpg")
        np.savetxt(scene / "pose" / f"{i*100:06}.txt", pose)
    return scene


def views(depth):
    return dict(depth=depth, depth_mask=torch.isfinite(depth) & (depth > 0),
                near=torch.full(depth.shape[:2], .1), far=torch.full(depth.shape[:2], 10.))


class DepthSupervisionTests(unittest.TestCase):
    def test_mask_before_log_and_multistage_gradient(self):
        gt = torch.ones(2, 2, 4, 4)
        gt[:, :, 0, 0] = float("nan")
        gt[:, :, 0, 1] = 0
        gt[:, :, 0, 2] = 20  # outside far
        p = torch.full((4, 2, 4, 4), 2., requires_grad=True)
        loss, final = supervised_depth_loss(p, views(gt), intermediate_weight=.5)
        self.assertAlmostEqual(loss.item(), np.log(2), places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(p.grad).all())
        self.assertTrue((p.grad[:, :, 1:, :] != 0).all())
        self.assertTrue((p.grad[:, :, 0, :3] == 0).all())
        self.assertTrue(torch.allclose(p.grad[:2], p.grad[2:] * .5))

    def test_empty_mask_and_invalid_predictions(self):
        p = torch.ones(1, 2, 4, 4, requires_grad=True)
        loss = masked_depth_loss(p, views(torch.zeros_like(p)))
        loss.backward()
        self.assertEqual(loss.item(), 0)
        self.assertEqual(p.grad.abs().sum().item(), 0)
        with self.assertRaises(ValueError):
            masked_depth_loss(torch.full_like(p, float("nan")), views(torch.ones_like(p)))

    def test_metrics_and_modes(self):
        p = torch.full((1, 2, 4, 4), 2.)
        v = views(torch.ones_like(p))
        self.assertAlmostEqual(masked_depth_loss(p, v, "l1").item(), 1.)
        self.assertAlmostEqual(masked_depth_loss(p, v, "inverse_l1").item(), .5)
        m = depth_metrics(p, v)
        self.assertEqual(m["abs_rel"].item(), 1.)
        self.assertEqual(m["rmse"].item(), 1.)
        self.assertEqual(m["a1"].item(), 0.)


class GeometryTests(unittest.TestCase):
    def test_registered_remap_preserves_identity(self):
        rgb = np.arange(20 * 30 * 3, dtype=np.uint8).reshape(20, 30, 3)
        k = np.array([[25., 0, 14.1], [0, 25., 9.3], [0, 0, 1.]])
        out, _ = align_registered_rgb(rgb, k, k, (20, 30))
        np.testing.assert_array_equal(out, rgb)

    def test_crop_flip_and_patch_principal_point(self):
        k = torch.tensor([[[1., 0, .53], [0, 1., .48], [0, 0, 1.]]])
        images = torch.rand(1, 3, 48, 64)
        depth = torch.ones(1, 48, 64)
        depth[:, :, :12] = 0
        v = dict(image=images, intrinsics=k, extrinsics=torch.eye(4)[None],
                 depth=depth, depth_mask=depth > 0)
        c = apply_crop_shim_to_views(v, (32, 32))
        self.assertEqual(c['depth'].shape, (1, 32, 32))
        self.assertTrue(torch.equal(c['depth_mask'], c['depth'] > 0))
        self.assertAlmostEqual(c['intrinsics'][0, 0, 2].item(), (.53*43-5)/32, places=6)
        f = reflect_views(c)
        self.assertAlmostEqual(f['intrinsics'][0, 0, 2].item(), 1-c['intrinsics'][0, 0, 2].item())
        self.assertTrue(torch.equal(reflect_views(f)['image'], c['image']))
        bv = {key: value.unsqueeze(0) for key, value in v.items()}
        patch = apply_patch_shim_to_views(bv, 20)
        self.assertAlmostEqual(patch['intrinsics'][0, 0, 0, 2].item(), (.53*64-2)/60, places=6)

    def test_offcenter_projection_matches_pinhole(self):
        k = torch.tensor([[[.9, 0., .53], [0., 1.2, .48], [0., 0., 1.]]])
        p = torch.tensor([.1, -.2, 2., 1.])
        matrix = projection_from_intrinsics(k, torch.tensor([.1]), torch.tensor([10.]))
        clip = matrix[0] @ p
        projected = (clip[:2] / clip[3] + 1) / 2
        expected = k[0] @ p[:3]; expected = expected[:2] / expected[2]
        self.assertTrue(torch.allclose(projected, expected, atol=1e-7))

    def test_overlap_rejects_inconsistent_surface(self):
        k = np.array([[50., 0, 32], [0, 50., 24], [0, 0, 1]])
        depth = np.full((48, 64), 2.)
        self.assertEqual(depth_overlap(depth, depth, k, np.eye(4), np.eye(4)), 1.)
        self.assertEqual(depth_overlap(depth, depth * 2, k, np.eye(4), np.eye(4)), 0.)


class PipelineTests(unittest.TestCase):
    def test_grouped_split(self):
        names = [f"scene{i:04}_{j:02}" for i in range(20) for j in range(3)]
        splits = split_scenes(names, 1234, .1, .1)
        spaces = {s: {n.split('_')[0] for n in ns} for s, ns in splits.items()}
        self.assertFalse(spaces['train'] & spaces['val'])
        self.assertFalse(spaces['train'] & spaces['test'])
        self.assertFalse(spaces['test'] & spaces['val'])
        self.assertEqual(splits, split_scenes(names, 1234, .1, .1))
        with self.assertRaises(ValueError):
            split_scenes(names, 1, .1, .1, {'scene0000_00'}, {'scene0000_01'})

    def test_conversion_dataset_collation_and_fixed_eval(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            item = scene_example(make_scene(root))
            self.assertTrue(item['valid_triplets'])
            self.assertEqual(len(item['images']), 5)
            self.assertAlmostEqual(item['cameras'][0, 2].item(), 31.5 / 64)
            packed_w2c = torch.eye(4); packed_w2c[:3] = item['cameras'][2, 6:].reshape(3, 4)
            self.assertAlmostEqual(torch.linalg.inv(packed_w2c)[0, 3].item(), .08, places=6)
            (root / 'manifest.json').write_text(json.dumps({'format_version': 2}))
            for stage in ('train', 'val', 'test'):
                (root / stage).mkdir()
                torch.save([item], root / stage / '0.torch')
                (root / stage / 'index.json').write_text(json.dumps({item['key']: '0.torch'}))
            cfg = DatasetScanNetCfg(name='scannet', roots=[root], image_shape=[32, 32],
                    background_color=[0., 0., 0.], cameras_are_circular=False,
                    overfit_to_scene=None, view_sampler=ViewSamplerScanNetCfg('scannet'), augment=False)
            for stage in ('train', 'val', 'test'):
                sampler = get_view_sampler(cfg.view_sampler, stage, False, False, None)
                dataset = DatasetScanNet(cfg, stage, sampler)
                sample = dataset[0]
                self.assertFalse(set(sample['context']['index'].tolist()) & set(sample['target']['index'].tolist()))
                if stage != 'train':
                    self.assertTrue(torch.equal(sample['target']['index'], dataset[0]['target']['index']))
                batch = next(iter(DataLoader(dataset, batch_size=1)))
                self.assertEqual(batch['context']['depth'].shape, (1, 2, 32, 32))
                self.assertEqual(batch['context']['depth'].max().item(), 2.)
                loss = masked_depth_loss(batch['context']['depth'].clamp_min(.1), batch['context'])
                self.assertAlmostEqual(loss.item(), 0.)

    def test_bad_pose_is_reported_per_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = make_scene(Path(directory))
            np.savetxt(scene / 'pose/000100.txt', np.full((4, 4), np.inf))
            item = scene_example(scene)
            self.assertEqual(len(item['images']), 4)
            self.assertEqual(item['rejected_frames'][0]['frame'], '000100')


if __name__ == '__main__':
    unittest.main()
