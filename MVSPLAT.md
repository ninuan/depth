# MVSplat 同协议外部对比

本轮只接入 **MVSplat**。先完成种子 `111123` 的 ScanNet、ARKitScenes 两组独立训练及测试，确认协议后再补 `222123`、`333123`。保留所有种子结果，不根据测试集表现决定是否报告。

三种子结果汇总、ScanNet checkpoint 核对和三方法可视化命令见 [MVSPLAT_RESULTS.md](MVSPLAT_RESULTS.md)。

## 对比设置

| 项目 | MVSplat | 现有 DepthSplat Baseline / Depth |
|---|---|---|
| 数据 | 现有已处理数据，不重新筛选 | 同一份数据 |
| 输入 / 目标视角 | 2 / 1 | 2 / 1 |
| ScanNet 分辨率 | 240 × 320 | 240 × 320 |
| ARKitScenes 分辨率 | 192 × 256 | 192 × 256 |
| 位姿 / 深度范围 | 米制，0.1–10 m | 相同 |
| 首轮训练预算 | 单 GPU，batch 1，30000 步 | 相同 |
| 训练种子 / loader 种子 | 111123 / 1234 | 相同 |
| RGB 损失 | MSE + 0.05 × VGG-LPIPS | 相同损失类型与权重 |
| GT 深度训练损失 | 0 | Baseline: 0；Depth: 0.1 |
| 初始化 | 官方 MVSplat RE10K | 官方 DepthSplat RE10K |
| 测试 checkpoint | 固定第 30000 步 last.ckpt | 全部使用固定第 30000 步 |

MVSplat 保留官方网络和高斯参数化。使用当前协议的 AdamW、weight decay 0.01、OneCycleLR，峰值 lr 2e-5，梯度裁剪 0.5；MVSplat 没有 DepthSplat 的单目预训练模块，使用一个学习率组。不同网络的初始化、参数量和原生中间输出不同，应在论文中写为“统一下游协议微调的外部方法比较”，不描述为完全相同预训练或对原论文数字的复现。这里的 lr 是首轮共享预算设置，尚未进行 MVSplat 专门的验证集调参。

GT 深度仅用于已有数据预处理的视角筛选，以及验证/测试指标；MVSplat 网络输入只使用图像、相机和 near/far，训练不加深度损失。外部对比与现有 Baseline/Depth 消融承担不同作用：后者用于判断加入深度监督的增益。

## 1. 同步代码与权重

在已完成其他实验的 uv 环境中执行，不必重装 PyTorch/CUDA：

```bash
cd /root/autodl-tmp/depthsplat
git pull --ff-only origin main
# 如果已激活可用的 .venv，保留当前环境。
which python
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
uv pip install --python "$(which python)" 'gdown>=5.2,<6'

export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
python -m src.scripts.prepare_mvsplat --download
```

下载的是[官方权重目录](https://drive.google.com/drive/folders/14_E_5R6ojOWnLSrSVLVEMHnTiKsfddjU)中的默认 `re10k.ckpt`，保存为 `pretrained/mvsplat-re10k.ckpt`，约 48 MB。脚本检查 SHA-256 和全部 471 个 encoder 条目。错误/消融权重会直接报错。

若云平台连接 Google Drive 中断，可在本地下载后上传到该路径，再执行：

```bash
python -m src.scripts.prepare_mvsplat
```

MVSplat 构造时不下载额外 backbone。VGG-LPIPS 继续使用原实验的 PyTorch 缓存。首次使用新机器时，先初始化一次，让网络下载问题在正式训练前暴露：

```bash
python -c 'from lpips import LPIPS; LPIPS(net="vgg"); print("VGG-LPIPS ready")'
```

## 2. GPU 预检

先检查一小批真实数据、CUDA 相机投影、MVSplat 权重、实际 RGB 损失反向传播及参数更新。预检输出目录独立于正式训练目录，预检更新不会保存到训练权重。

```bash
python -m src.scripts.check_scannet \
  --encoder mvsplat --dataset scannet \
  --dataset_dir datasets/scannet_processed_v2 \
  --checkpoint pretrained/mvsplat-re10k.ckpt \
  --output_dir outputs/mvsplat_scannet_check --max_scenes 2 --gpu

python -m src.scripts.check_scannet \
  --encoder mvsplat --dataset arkitscenes \
  --dataset_dir datasets/arkitscenes_processed_v1 \
  --checkpoint pretrained/mvsplat-re10k.ckpt \
  --output_dir outputs/mvsplat_arkit_check --max_scenes 2 --gpu
```

两次都出现 `PASS: CPU data + CUDA training/backward` 后开始正式训练。`direct_depth_loss: null` 对 RGB-only MVSplat 是正常结果。GPU 显存峰值在对应 `report.json` 中；先确认显存，再跑长实验。

## 3. 首轮两组训练（前台运行）

两组分别从官方 RE10K 初始化，不把 ScanNet 微调模型作为 ARKitScenes 独立训练的起点。

```bash
bash <<'BASH'
set -euo pipefail
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

bash scripts/mvsplat_train.sh scannet \
  datasets/scannet_processed_v2 \
  pretrained/mvsplat-re10k.ckpt \
  outputs/mvsplat_scannet_seed111123

bash scripts/mvsplat_train.sh arkitscenes \
  datasets/arkitscenes_processed_v1 \
  pretrained/mvsplat-re10k.ckpt \
  outputs/mvsplat_arkit_seed111123
BASH
```

默认已设 `seed=111123 data_loader.train.seed=1234 data_loader.train.batch_size=1 trainer.max_steps=30000`。训练每 20 步打印日志，每 1000 步验证并保存 checkpoint；不需要 nohup。

中断后，在原命令最后追加 `checkpointing.resume=true`，例如：

```bash
bash scripts/mvsplat_train.sh scannet \
  datasets/scannet_processed_v2 \
  pretrained/mvsplat-re10k.ckpt \
  outputs/mvsplat_scannet_seed111123 \
  checkpointing.resume=true
```

恢复读取该目录的 `checkpoints/last.ckpt` 并恢复优化器和 scheduler。保持 `trainer.max_steps=30000`。若第一次保存前就失败、没有 last.ckpt，先保留失败目录作为记录，再换新输出目录从头开始。中途恢复沿用现有数据加载器机制，不承诺与一次不中断训练逐位一致。

## 4. 固定第 30000 步测试

脚本会检查 checkpoint 的 `global_step == 30000`，再使用已有固定测试三元组：

```bash
bash scripts/mvsplat_eval.sh scannet \
  datasets/scannet_processed_v2 \
  outputs/mvsplat_scannet_seed111123/checkpoints/last.ckpt \
  outputs/mvsplat_scannet_test_seed111123

bash scripts/mvsplat_eval.sh arkitscenes \
  datasets/arkitscenes_processed_v1 \
  outputs/mvsplat_arkit_seed111123/checkpoints/last.ckpt \
  outputs/mvsplat_arkit_test_seed111123
```

汇总在 `outputs/<测试目录>/metrics/scores_all_avg.json`。同时保留 `run_config.yaml`、复制的 `dataset_manifest_0.json` 和逐扫描指标。

如果早期 ScanNet 主表使用了验证集最佳权重，补测 Baseline / Depth 的第 30000 步即可，无需重新训练：

```bash
bash <<'BASH'
set -euo pipefail
for variant in rgb depth; do
  python - "$variant" <<'PY'
import sys, torch
p = f'outputs/scannet_{sys.argv[1]}/checkpoints/last.ckpt'
assert torch.load(p, map_location='cpu', weights_only=False)['global_step'] == 30000, p
PY
  python -m src.main +experiment=scannet mode=test \
    dataset.roots='[datasets/scannet_processed_v2]' \
    "checkpointing.pretrained_model=outputs/scannet_${variant}/checkpoints/last.ckpt" \
    "output_dir=outputs/scannet_${variant}_test_step30000_seed111123" \
    "test.output_path=outputs/scannet_${variant}_test_step30000_seed111123" \
    dataset.test_split=test test.compute_scores=true
done
BASH
```

上述路径对应最早一组 ScanNet 目录；若云端已改名，使用实际路径。ARKitScenes 已有固定 30000 步结果可以复用。不要将单种子 MVSplat 与三种子均值直接作配对差值：首轮先取三个方法相同种子，之后补齐三个种子再汇总。

## 5. 深度定义与后续复用

RGB 指标仍评价目标视角。深度指标评价输入视角的相机 Z，使用相同有效深度 mask 和米制范围，不做 median/scale 对齐。官方 MVSplat 的高斯沿单位射线放置，其原始标量是射线距离。本接入保留原始高斯位置，把各高斯在所属输入相机下的 Z 坐标作为像素深度评价（保留原有亚像素偏移）；不把原始射线距离直接和传感器 Z 比较，也不把它称为目标视角渲染深度。

先完成以上两组。需要补种子时，在训练命令尾部追加 `seed=222123` 或 `seed=333123` 并使用对应新目录，保持 loader seed 1234 和其他设置一致。最终主表应明确每个方法是否使用额外 GT 深度训练监督。

现有 ScanNet → ARKitScenes 的零微调比较也可复用同一个 ScanNet MVSplat checkpoint，无需额外训练：

```bash
bash scripts/mvsplat_eval.sh arkitscenes \
  datasets/arkitscenes_test_v1 \
  outputs/mvsplat_scannet_seed111123/checkpoints/last.ckpt \
  outputs/mvsplat_scannet_to_arkit_seed111123
```

## 验证记录与来源

- 官方源码：[donydchen/mvsplat](https://github.com/donydchen/mvsplat)，固定 commit `01f9a28edb5eb68416e7e63b01f8d90c3bdfbf01`。
- 官方 RE10K 权重 SHA-256：`83d0d9eaa753fa4a1f925288dc1f90b8c3297fad0ab0f6ed1f11a1c5946da25a`。
- 本地已验证严格权重匹配、两种实际输入尺寸的 CPU 网络前向、小尺寸网络反向传播、不同内参投影、批次/视角顺序、输入 GT 标签不影响预测，以及原有 RGB-D 回归测试（31 项通过）。相同内参时，代价体预测器与官方源码逐位一致。
- 本地没有 NVIDIA GPU；真实 CUDA 渲染训练需执行第 2 节预检。
- 内部实现改动、许可证和原文件哈希见 `src/model/encoder/mvsplat/README.md` 与 `UPSTREAM.json`。
