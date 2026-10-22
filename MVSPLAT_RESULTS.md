# 三种方法的实验汇总

来源：用户在本对话提供的实验输出，整理日期 2026-10-22。所有 ± 为三个训练种子的样本标准差（n=3），不是置信区间。种子为 111123、222123、333123。

## ARKitScenes 独立训练与测试

按已报告的固定第 30000 步结果整理。PSNR、SSIM、a1 越大越好；LPIPS、AbsRel、RMSE 越小越好。

| 指标 | MVSplat | DepthSplat Baseline | DepthSplat + 深度监督 |
|---|---|---|---|
| PSNR | 21.169932 ± 0.019458 | 22.617413 ± 0.276339 | 22.855547 ± 0.050849 |
| SSIM | 0.746054 ± 0.000250 | 0.764688 ± 0.006211 | 0.775309 ± 0.001259 |
| LPIPS | 0.225401 ± 0.000684 | 0.206704 ± 0.009666 | 0.195494 ± 0.000269 |
| Depth AbsRel | 0.344412 ± 0.008418 | 0.200479 ± 0.018378 | 0.087920 ± 0.001880 |
| Depth RMSE (m) | 0.763327 ± 0.000911 | 0.380968 ± 0.067398 | 0.141020 ± 0.002479 |
| Depth a1 | 0.733838 ± 0.000352 | 0.754785 ± 0.017378 | 0.914657 ± 0.006553 |

## ScanNet 训练后直接测试 ARKitScenes

不在 ARKitScenes 微调，使用固定第 30000 步的 ScanNet 权重。

| 指标 | MVSplat | DepthSplat Baseline | DepthSplat + 深度监督 |
|---|---|---|---|
| PSNR | 20.518682 ± 0.095188 | 21.340275 ± 0.350083 | 21.509612 ± 0.034276 |
| SSIM | 0.719400 ± 0.003729 | 0.741459 ± 0.001795 | 0.745810 ± 0.001474 |
| LPIPS | 0.270649 ± 0.001661 | 0.246897 ± 0.005715 | 0.240886 ± 0.001562 |
| Depth AbsRel | 0.526534 ± 0.017135 | 0.286755 ± 0.010477 | 0.242090 ± 0.005615 |
| Depth RMSE (m) | 0.849324 ± 0.010797 | 0.332675 ± 0.018808 | 0.259416 ± 0.003537 |
| Depth a1 | 0.604877 ± 0.011324 | 0.666900 ± 0.007544 | 0.696498 ± 0.007728 |

## ScanNet 独立训练与测试：待核对后合表

用户尚未确认早期 Baseline / Depth 汇总采用 last 还是验证集最佳权重，因此先保留 MVSplat 的固定 30000 步结果，不将旧结果直接混入同协议主表。

| 指标 | MVSplat |
|---|---|
| PSNR | 15.808009 ± 0.042020 |
| SSIM | 0.593482 ± 0.001489 |
| LPIPS | 0.436982 ± 0.000311 |
| Depth AbsRel | 0.217139 ± 0.003554 |
| Depth RMSE (m) | 0.830529 ± 0.013011 |
| Depth a1 | 0.744931 ± 0.002172 |

在云端仓库根目录执行：

```bash
python -m src.scripts.audit_rgbd_results --dataset scannet
```

脚本读取已有测试配置、结果以及当前路径下的 checkpoint 元数据，不运行 GPU 推理。报告保存为 `outputs/rgbd_checkpoint_audit.json`。训练种子来自对应训练目录的配置；缺失时明确显示 unknown。若权重已经替换、旧配置缺失或文件删除，这个检查不证明旧结果来源，需要补测固定权重。

## 结论范围

当前数据支持：在本项目的固定子集和下游微调设置中，Depth 版本的上述两组平均指标均优于 MVSplat 和 Baseline。深度监督本身的增益主要由同结构的 Baseline / Depth 对照支持；MVSplat 是外部竞争力参照，模型结构、预训练经历和额外训练标签并不相同。

三个种子的小标准差不等于跨场景泛化已充分验证，也不等于统计显著。跨数据集 PSNR 的 Depth−Baseline 配对差值此前仅 1/3 种子为正，论文应保留这个限制。MVSplat 使用共享训练预算，尚未做独立验证集超参数搜索，不把这些数值描述为其最优性能或原论文复现。

## 接下来：固定样例可视化

先固定使用种子 111123 和 ARKitScenes 验证集按扫描 ID 排序的前 8 个场景，不根据哪个方法获胜筛选。三个方法使用相同输入、目标和 GT。

```bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0
bash scripts/arkitscenes_compare_visuals.sh \
  datasets/arkitscenes_processed_v1 \
  outputs/arkit_baseline_seed111123/checkpoints/last.ckpt \
  outputs/arkit_depth_seed111123/checkpoints/last.ckpt \
  outputs/mvsplat_arkit_seed111123/checkpoints/last.ckpt \
  outputs/arkit_three_method_visuals_seed111123
```

Baseline 路径对应最初用户提供的训练命令；如果实际目录命名为 `arkit_rgb_seed111123`，仅替换该路径。所有权重先检查训练步数，输出独立于正式完整测试结果。

导出包括 RGB 预测、RGB GT、输入图，以及输入视角的预测 Z 深度、传感器 GT 和有效 mask。深度 `.npy` 单位为米。MVSplat 深度导出与评价使用相同的相机 Z 换算；这一补充不改变模型参数、训练或数值评价。原有彩色深度 PNG 使用各图自动色标，正式并排图应从 `.npy` 重绘统一色标，误差图使用相同 mask 和范围。

打包结果：

```bash
python - <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
root = Path('outputs/arkit_three_method_visuals_seed111123')
with ZipFile(root.with_suffix('.zip'), 'w', ZIP_DEFLATED) as z:
    for p in sorted(root.rglob('*')):
        if p.is_file():
            z.write(p, arcname=str(p.relative_to(root)))
print(root.with_suffix('.zip'))
PY
```

这些是验证集诊断图；论文中标注真实划分，不能标成完整测试集图。后续正式测试可视化也应提前固定场景清单并保留失败样例。
