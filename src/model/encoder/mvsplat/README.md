# Vendored MVSplat encoder

Source: https://github.com/donydchen/mvsplat

Pinned commit: `01f9a28edb5eb68416e7e63b01f8d90c3bdfbf01`.
`UPSTREAM.json` records hashes of the original files. `LICENSE` is the upstream
MIT license; embedded third-party notices are retained.

We vendor the default RE10K model, not an ablation: 128 depth candidates,
cost U-Net 128 channels / [1,1,1] / attention [4], refinement U-Net 32 channels /
[1,1,1,1,1] / attention [16], one Gaussian per pixel, SH degree 4.
The official `re10k.ckpt` has 471 encoder state entries and 11,955,734 parameters.
No backbone or monocular network is downloaded during encoder construction.

Changes from upstream:

- Package import paths; unused upstream training, dataset and visualization
  infrastructure is not included. The outer adapter is `../encoder_mvsplat.py`.
- Cost-volume warping takes separate reference/source K, preserving upstream
  arithmetic when both K are equal. This supports ARKitScenes per-frame K.
- Our dataset stores half-pixel-center normalized K; a copy is converted to
  integer-pixel K for CNN feature warping, as in this repository's DepthSplat
  adapter. Gaussian rays and rendering retain the stored half-pixel-center K.
- Original sigmoid scale mapping, unit-length rays, quaternion normalization,
  opacity mapping (identity under the official default), SH masking and rotation
  are retained. `common/projection.py` preserves upstream ray geometry; the shared
  DepthSplat ray helper has different depth semantics and must not be substituted.
- The returned depth map is the camera-Z coordinate of each context Gaussian,
  indexed by its originating pixel (including its learned subpixel offset).
  This converts upstream ray distance to the sensor depth convention without
  altering Gaussian means or introducing scale alignment. It is not a rendered
  target-view depth map. Depth is evaluated only, never supervised for MVSplat.
- The existing renderer handles off-center principal points for every method.
  The existing dataset, augmentation, fixed triplets, losses and metrics are shared.

This is a protocol-adapted MVSplat fine-tuning baseline, not a reproduction of
the original RE10K benchmark numbers. See `MVSPLAT.md` at the repository root.
