# ARKitScenes 第二数据集实验

主方法固定深度损失权重 0.1、中间监督权重 0.5。先做 ScanNet 模型的跨数据集测试，再独立在 ARKitScenes 上训练 Baseline / Depth。

## 固定协议

- 使用官方 **RAW** 的 `lowres_wide`、`lowres_depth`、`confidence`、`lowres_wide_intrinsics`、`lowres_wide.traj`。不混用 3DOD 版本或 highres_depth。
- RGB / 深度原生分辨率 192×256；深度为设备测得的毫米相机 Z 深度，转换成米。这不是 FARO 激光参考深度，也不是模型伪深度。
- 只监督/评价置信度为 2 且处于 [0.1, 10] 米的深度像素；RGB 评价使用整张目标图。
- 固定清单 `assets/arkitscenes_subset_v1.json` 包含 200 个训练地点、30 个验证地点、50 个测试地点，每个地点一个扫描。选择按种子 1234 的 SHA-256 排序，不依据模型结果。
- train/val 从官方 Training 中按 visit_id 分开；test 来自官方 Validation。缺失 visit_id、跨官方划分的地点、官方注明缺轨迹的扫描均排除。清单保存元数据摘要和排除列表。
- 这是**本项目固定的 ARKitScenes 子集与新视角合成协议**，不是官方 NVS 排行榜，也不是完整数据集成绩。过滤后的实际扫描数以 manifest 为准；必须报告拒绝扫描，不能根据模型表现替换。
- 同步 RGB/深度/置信度按相同时间戳配对；轨迹最近时间差不超过 5 ms，内参不超过 1 ms，不插值位姿。
- 每隔至少 1 秒选一帧，保留前 200 个满足时间匹配条件的候选；再进行数据质量检查。每帧保存自身内参。
- 三元组按时间顺序选两张 context 和一张中间 target。最小输入基线 2 cm，最大保留帧间隔 4，最大实际时间跨度 4.5 秒；双向真实深度重叠率至少 0.3。训练从合格三元组随机采样，验证/测试每个扫描固定一个最高重叠三元组。
- **GT 深度用于离线筛选三元组以及监督/评价，模型推理不输入 GT 深度。** 此采样协议倾向高重叠场景，不能宣称覆盖所有基线难度。

## 1. 云端准备

进入项目、激活已经能运行 ScanNet 的 Python 环境，然后确认当前代码包含本文件。以下命令在项目根目录执行，不需要安装新的训练依赖：

```bash
cd /root/autodl-tmp/depthsplat
python -c 'import sys, torch, hydra, cv2; print(sys.executable); print(torch.__version__, torch.cuda.is_available())'
command -v curl
df -h /root/autodl-tmp
```

优先使用仓库已冻结的清单，不要重复生成。只有主动建立另一份协议时才执行：

```bash
python -m src.scripts.arkitscenes_data plan \
  --output datasets/arkitscenes/custom_plan.json \
  --train_visits 200 --val_visits 30 --test_visits 50 --seed 1234
```

该命令只下载元数据并生成清单，不下载 RGB-D；已存在的清单不会被覆盖。

## 2. 先下载测试部分

```bash
python -m src.scripts.arkitscenes_data download \
  --plan assets/arkitscenes_subset_v1.json \
  --output_dir datasets/arkitscenes_raw \
  --splits test
```

下载不需要 GPU，建议在云平台的无 GPU 模式完成。大小依扫描长度变化，不应使用单个样本的大小估算全体。脚本有重试、断点续传、压缩包 CRC 检查和完成标记；网络失败后重跑同一命令。成功解压后默认删除下载的 zip，使用 `--keep_zip` 可保留。损坏的完整 zip 会报错，需要移走对应坏包再重试；不会静默换一个扫描。

## 3. 转换并检查

```bash
bash scripts/arkitscenes_prepare.sh \
  datasets/arkitscenes_raw \
  assets/arkitscenes_subset_v1.json \
  datasets/arkitscenes_test_v1 \
  --splits test
```

如果转换中断，只增加 `--resume` 重跑；已完成扫描会复用，清单、划分、预处理参数必须相同。下载缺文件会先报错，不会作为“坏场景”默默丢弃。

检查：

- `datasets/arkitscenes_test_v1/manifest.json`：实际数量、拒绝扫描及原因。
- `datasets/arkitscenes_test_v1/checks/report.json`：数据读入与标签检查。
- `datasets/arkitscenes_test_v1/checks/test_alignment.png`：RGB 与深度边缘叠加样例。

这些是 CPU 检查，不代表 CUDA 渲染或模型效果已通过验证。正式运行前还应检查多个场景的重投影，不能只依赖一张边缘叠加图。

## 4. 不微调，先测现有 ScanNet 模型

首轮对照采用两个模型的固定第 30000 步；这里的路径对应已有第一种子目录，若云端改过目录，请替换。不要把一个模型的最佳 checkpoint 与另一个的 last 混用。

```bash
python - <<'PY'
import torch
for p in ('outputs/scannet_rgb/checkpoints/last.ckpt',
          'outputs/scannet_depth/checkpoints/last.ckpt'):
    c = torch.load(p, map_location='cpu', weights_only=False)
    print(p, 'global_step=', c.get('global_step'))
    assert c.get('global_step') == 30000, '请选择两组固定第 30000 步的 checkpoint'
PY
```

先用一个扫描确认前向与评价流程可运行。它只验证程序，不作为效果筛选依据：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/arkitscenes_eval.sh \
  datasets/arkitscenes_test_v1 \
  outputs/scannet_depth/checkpoints/last.ckpt \
  outputs/arkit_smoke \
  dataset.test_len=1
```

通过后测试全部合格扫描：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/arkitscenes_eval.sh \
  datasets/arkitscenes_test_v1 \
  outputs/scannet_rgb/checkpoints/last.ckpt \
  outputs/scannet_rgb_to_arkit_seed111123

CUDA_VISIBLE_DEVICES=0 bash scripts/arkitscenes_eval.sh \
  datasets/arkitscenes_test_v1 \
  outputs/scannet_depth/checkpoints/last.ckpt \
  outputs/scannet_depth_to_arkit_seed111123
```

汇总文件位于各输出目录的 `metrics/scores_all_avg.json`；逐扫描文件也在 `metrics/`。报告 PSNR、SSIM、LPIPS、depth_abs_rel、depth_rmse、depth_a1。RGB 评价目标视角，深度评价输入视角。后续再测试另两个已有训练种子，不需要重新训练 ScanNet。

## 5. 第二数据集独立训练

跨数据集测试流程确认正确后，可继续下载训练和验证扫描：

```bash
python -m src.scripts.arkitscenes_data download \
  --plan assets/arkitscenes_subset_v1.json \
  --output_dir datasets/arkitscenes_raw \
  --splits train val

bash scripts/arkitscenes_prepare.sh \
  datasets/arkitscenes_raw \
  assets/arkitscenes_subset_v1.json \
  datasets/arkitscenes_processed_v1 \
  --splits train val test
```

使用新输出目录生成全划分数据；测试部分的预处理参数和场景清单保持一致。两组从同一个原始 RE10K 预训练权重开始，不能分别从 ScanNet Baseline / Depth 初始化后将结果称为独立训练：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/arkitscenes_train.sh baseline \
  datasets/arkitscenes_processed_v1 \
  pretrained/depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth \
  outputs/arkit_rgb_seed111123 \
  seed=111123 data_loader.train.seed=1234

CUDA_VISIBLE_DEVICES=0 bash scripts/arkitscenes_train.sh depth \
  datasets/arkitscenes_processed_v1 \
  pretrained/depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth \
  outputs/arkit_depth_seed111123 \
  seed=111123 data_loader.train.seed=1234
```

默认单卡、batch size 1、30000 步，沿用 ScanNet 优化器和多阶段监督。支持同目录加 `checkpointing.resume=true` 从 last.ckpt 恢复。各组使用独立输出目录。正式测试前冻结 checkpoint 选择规则，并对 Baseline / Depth 使用一致规则；不要根据外部测试结果调损失权重或挑 checkpoint。

## 数据来源与复现

- [官方 RAW 格式](https://github.com/apple/ARKitScenes/blob/main/raw/README.md)
- [官方下载说明](https://github.com/apple/ARKitScenes/blob/main/DATA.md)
- [官方位姿与投影实现](https://github.com/apple/ARKitScenes/blob/main/threedod/benchmark_scripts/utils/tenFpsDataLoader.py)
- [官方元数据](https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/metadata.csv)

训练结果会保存 run_config.yaml 与数据 manifest。论文使用时应引用 ARKitScenes，并明确这是自定义固定子集、设备深度评价、2-context/1-target 协议，不能与不同设置的官方结果直接横向比较。
