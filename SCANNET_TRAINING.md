# ScanNet 真实深度监督：云 GPU 实验指南

这版实现使用 **RGB＋相机参数预测深度和高斯**，真实深度仅用于训练损失和评估。
默认联合优化 RGB MSE、LPIPS、masked log-depth；不包含渲染深度损失或新的可靠性网络。

## 1. 环境与权重

以下命令面向 Linux x86_64、Ubuntu 22.04、NVIDIA GPU、CUDA 12.1 **devel** 环境，从上传后的本项目根目录执行。需要能运行 `nvcc`，因为高斯渲染扩展要编译。请上传本次修改后的完整代码，而不是只重新克隆上游仓库。

这里选择 Python 3.10、PyTorch 2.4.0、torchvision 0.19.0、xformers 0.0.27.post2。与上游 README 的 cu124 不同，这份 uv 流程使用 cu121，让 PyTorch 和固定版本 xformers 使用同一个 CUDA 安装包源。xformers 的 cu124 索引没有 0.0.27.post2；不要直接升级 xformers，它会改变 PyTorch 依赖。本流程尚需在实际云 GPU 上完成下文检查。

先检查 GPU、CUDA 编译器和 C++ 编译器：

```bash
nvidia-smi
nvcc --version
g++ --version
git --version
```

`nvcc` 应显示 12.1；`nvidia-smi` 中的 CUDA Version 是驱动支持的版本上限，可以高于 12.1。uv 不会安装 NVIDIA 驱动或完整 CUDA Toolkit。缺少 `nvcc` 时应先选用包含 CUDA 12.1 开发工具的云镜像。

安装 uv 并创建独立环境：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.10
uv venv --python 3.10 .venv
source .venv/bin/activate
```

分阶段安装，让高斯渲染扩展编译时能够导入已安装的 torch：

```bash
uv pip install 'setuptools<81' wheel packaging ninja numpy==1.24.4
uv pip install torch==2.4.0 torchvision==0.19.0 \
  --index-url https://download.pytorch.org/whl/cu121
uv pip install xformers==0.0.27.post2 --no-deps \
  --index-url https://download.pytorch.org/whl/cu121
export MAX_JOBS=4
uv pip install --no-build-isolation -r requirements.txt \
  torch==2.4.0 torchvision==0.19.0 'setuptools<81'
uv pip check
python -m xformers.info
python - <<'PY'
import torch
import torchvision
from diff_gaussian_rasterization import _C
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("CUDA runtime:", torch.version.cuda)
assert torch.cuda.is_available(), "PyTorch 无法访问 GPU"
print("GPU:", torch.cuda.get_device_name(0))
x = torch.randn(64, 64, device="cuda")
print("CUDA computation:", (x @ x).mean().item())
PY
uv pip freeze > requirements-cloud.lock.txt
```

确认 torch 为 `2.4.0+cu121`、torchvision 为 `0.19.0+cu121`。`xformers.info` 的部分算子不可用是正常的，但不能出现整个 C++/CUDA 扩展无法加载的警告。`requirements-cloud.lock.txt` 是实际安装版本记录。每次重新登录后，在项目根目录执行 `source .venv/bin/activate`。

下载预训练权重：

```bash
mkdir -p pretrained
curl -fL --retry 3 \
  https://huggingface.co/haofeixu/depthsplat/resolve/main/depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth \
  -o pretrained/depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth
```

安装参考：[uv 安装](https://docs.astral.sh/uv/getting-started/installation/)、[uv 构建隔离](https://docs.astral.sh/uv/concepts/projects/config/#build-isolation)、[PyTorch 历史版本](https://docs.pytorch.org/get-started/previous-versions/)、[xformers cu121 安装包](https://download.pytorch.org/whl/cu121/xformers/)。

所有对照组均用这个官方 **small GS** checkpoint，加载为 strict。
首次构建模型还会通过 `torch.hub` 获取 DINOv2 代码/权重，LPIPS 会获取 VGG 权重；请允许云主机联网，或者提前准备 Torch Hub 缓存。这里只加载完整 GS 权重，不再额外加载 UniMatch 或深度模型权重。

## 2. 重新准备数据（旧 `.torch` 不能混用）

支持本项目现有的官方 `scannet_frames_25k` 导出结构：

```text
scene0000_00/
  color/000000.jpg
  depth/000000.png
  pose/000000.txt
  intrinsics_color.txt
  intrinsics_depth.txt
```

```bash
bash scripts/scannet_prepare.sh \
  datasets/scannet/tasks/scannet_frames_25k \
  datasets/scannet_processed_v2
```

输出目录必须为空；脚本不会覆盖或清理旧数据。该脚本下载 ScanNet 官方 train/val 列表，然后：

- `train`：官方训练地点的 90%。
- `val`：官方训练地点中另留的 10%，用于调参、选择 checkpoint。
- `test`：官方验证列表中满足采样条件的扫描，用于最终报告。

这里的 `test` 是本实验的 holdout，**不是 ScanNet 隐藏测试集**。划分按 `sceneXXXX` 物理地点进行，同一地点的重复扫描不得跨集合。随机种子、实际名单、原始 split 的 SHA256 和所有过滤原因保存到 `manifest.json`。不指定官方列表直接运行转换器时，会使用按地点分组的随机划分，仅适合调试。

25k 已经约每 100 帧采样，因此默认 `frame_stride=1`。转换器保留两张 context 和一张不同的中间 target，最大 context 间隔默认 4 个保留帧，基线至少 2 cm。每个 context 与 target 的**双向深度重投影重叠率**至少为 0.3；深度相容阈值为 `max(5 cm, 5% depth)`。没有有效组合的扫描会在报告中标注排除。训练在有效组合中随机采样，val/test 每个扫描固定使用得分最高的一组；固定组合同时保存为 `evaluation_index.json`。

这是一个自定义的、有共视条件的稀疏视角评估协议，不能把结果直接当作原论文 ScanNet 基准结果。各个消融组必须复用同一份数据、manifest 和评估组合。过滤阈值应在正式测试前固定，并报告实际保留数量。

RGB 根据 `K_color @ inverse(K_depth)` 映射到深度图的像素网格，深度 PNG 保持传感器 camera-Z 单位，读取时除以 1000 转米。**此映射要求两者已注册到相同光学坐标系**，适用于本次官方 25k 数据；不适用于带未知 RGB-D 外参的原始传感器导出。若目录中有不一致的 color/depth 外参，转换会拒绝处理。

代码统一 OpenCV 与模型的像素中心约定，同步处理裁剪／翻转的相机主点，并让渲染器使用完整的非中心主点投影。默认训练尺寸为 `240×320`，保留原始 4:3 视野。

只转换少量场景调试时，可直接使用分组随机划分：

```bash
python -m src.scripts.convert_scannet \
  --input_dir datasets/scannet/tasks/scannet_frames_25k \
  --output_dir datasets/scannet_smoke_v2 --max_scenes 24
```

## 3. 先做 CPU 检查，再做真实 GPU 检查

```bash
python -m unittest discover -s tests -v
python -m src.scripts.check_scannet \
  --dataset_dir datasets/scannet_processed_v2 \
  --output_dir outputs/scannet_check
```

检查输出 `report.json` 和各 split 的 `*_alignment.png`。默认每个 split 抽查 8 个扫描；加 `--max_scenes 0` 可检查全部。CPU 检查验证加载、深度范围、有效 mask、固定视角和集合隔离，不代表 CUDA 已通过。

云 GPU 上执行：

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.scripts.check_scannet \
  --dataset_dir datasets/scannet_processed_v2 \
  --output_dir outputs/scannet_gpu_check \
  --gpu \
  --checkpoint pretrained/depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth
```

GPU 检查运行真实模型和 CUDA rasterizer：检查非中心主点投影、联合损失反向传播、深度分支非零有限梯度、参数更新、直接深度损失到网络的梯度，以及 eval 只返回最终高斯。任何检查失败会直接报错，不会继续长训练。

## 4. 冒烟训练与两组正式实验

先跑 10 步，包含真实验证、日志与 checkpoint 流程：

```bash
bash scripts/scannet_train.sh depth \
  datasets/scannet_processed_v2 \
  pretrained/depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth \
  outputs/scannet_smoke \
  trainer.max_steps=10 trainer.val_check_interval=10 \
  data_loader.train.batch_size=1 data_loader.train.num_workers=0
```

默认启动一张 GPU。正式训练默认 30,000 步、batch size 2、每 1,000 步验证；这是起跑配置，显存占用和收敛情况需实际观察。

```bash
# 对照：同样在 ScanNet 微调，但没有真实深度训练损失
bash scripts/scannet_train.sh baseline \
  datasets/scannet_processed_v2 \
  pretrained/depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth \
  outputs/scannet_rgb

# 方法：相同初始化、样本、预算，增加真实深度监督
bash scripts/scannet_train.sh depth \
  datasets/scannet_processed_v2 \
  pretrained/depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth \
  outputs/scannet_depth
```

`baseline` 仍读取深度做相同的数据过滤和评估，但不把深度加入训练损失。两组都经过本版几何修正，避免把数据／相机修复带来的收益算作新损失的贡献。

显存不足时先加 `data_loader.train.batch_size=1`；仍不足再同时调整两组分辨率，例如 `'dataset.image_shape=[192,256]'`。默认保留中间深度预测对应的 RGB 和深度监督，显存占用会高于只渲染最终预测。

## 5. 损失、指标、日志与恢复

总损失：`L_rgb_final + L_rgb_intermediate + lambda_depth * L_depth`。

`L_depth` 在有效范围 `[near, far]` 内计算；默认是 `abs(log(pred)-log(gt))`。
中间预测按 `depth_intermediate_weight` 的幂递减加权，并除以权重总和，使模型阶段数变化时深度项的量级相对稳定。

可在训练脚本末尾追加：

```text
train.depth_loss_weight=0.05
train.depth_loss_mode=l1                 # 或 log_l1 / inverse_l1
train.depth_intermediate_weight=0.0     # 仅直接监督最终预测；中间 RGB 损失仍在
seed=123
```

没有有效深度的训练 batch、缺失标签、非有限预测都会报错；不会悄悄退化成纯 RGB 训练。

每次运行输出：

- `logs/version_*/metrics.csv`：RGB 与深度损失、验证 PSNR/SSIM/LPIPS、深度 AbsRel/RMSE/a1。
- `logs/version_*/comparison/`：验证 RGB 对比图。
- `checkpoints/last.ckpt`：最近一次验证保存的训练状态，可恢复优化器和学习率调度器。
- `checkpoints/step*-psnr*.ckpt`：验证 PSNR 最好的两个 checkpoint。
- `.hydra/` 或 Hydra 日志目录：配置记录；另外训练入口保存 `run_config.yaml` 到运行输出目录。

深度指标在 context 的有效传感器深度上计算，RMSE 单位为米；RGB 指标只统计未输入的 target。最终汇总采用每扫描指标的平均值。日志不需要 wandb 账号；可另行启用 wandb。

中断恢复使用**相同输出目录、原训练总步数和配置**，追加 `checkpointing.resume=true`：

```bash
bash scripts/scannet_train.sh depth \
  datasets/scannet_processed_v2 \
  pretrained/depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth \
  outputs/scannet_depth checkpointing.resume=true
```

它从 `last.ckpt` 恢复，不重新套用预训练权重。不要把一个 10 步 smoke run 的 OneCycle 学习率调度当作 30,000 步实验继续；正式实验使用新输出目录重新初始化。

## 6. 最终测试

用验证集选好的 checkpoint，在独立 holdout 上一次性评估。将下面路径替换为实际的最佳文件名：

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=scannet mode=test \
  dataset.roots='[datasets/scannet_processed_v2]' \
  checkpointing.pretrained_model=null \
  checkpointing.load=outputs/scannet_depth/checkpoints/SELECTED_CHECKPOINT.ckpt \
  output_dir=outputs/scannet_depth_test \
  test.save_image=true
```

输出 `metrics/scores_all_avg.json` 及逐扫描列表。正式测试强制一张 GPU，避免分布式 sampler 补齐样本导致重复统计。多 GPU 训练由 map-style Dataset 和 Lightning DistributedSampler 切分；多 GPU 验证可能补齐末尾样本，严格复核模型选择指标时也建议单 GPU 评估 `dataset.test_split=val`。

不要仅凭训练 loss 下降判断有效；至少对比同预算 baseline 的深度指标与新视角 RGB 指标。此实现提供可运行的监督基线，不预设性能提升或论文创新性。

## 已在本地验证的范围

在无 CUDA 的 Mac 上完成 CPU 回归测试、实际 `ModelWrapper.training_step` 的梯度对照、Lightning 小模型训练／验证／checkpoint／恢复，以及 24 个真实扫描的转换试验（保留 16 个，8 个因共视条件排除）。实际 DepthSplat 网络和 CUDA 光栅化训练需要上述云端 `--gpu` 检查；本地测试的微型可微编码器／渲染器不能替代它。

参考：[ScanNet 格式与单位](https://kaldir.vc.in.tum.de/scannet_benchmark/documentation)、[ScanNet 地点命名](https://github.com/ScanNet/ScanNet#data-organization)、[DepthSplat 官方仓库](https://github.com/cvg/depthsplat)。
