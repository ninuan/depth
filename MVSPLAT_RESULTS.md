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

2026-10-22 收到的审计报告确认三个 MVSplat ScanNet 测试 checkpoint 均为 30000 步，但旧 DepthSplat 记录为 `checkpoint=null`。发现旧审计脚本只检查 `pretrained_model`，已修正为与 RGB-D 测试入口一致的 `resume > load > pretrained_model` 优先级，并保存 `checkpoint_config` 供排查。需重跑审计后再判断旧结果的步数，当前不据此要求重训。

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

## 固定验证样例的实测检查

已检查用户返回的全部 8 个场景、16 个输入视角。三方法的配置、数据 manifest、输入 RGB、目标 RGB GT、深度 GT 和有效 mask 对齐。原始 `.npy` 重算的三项深度指标与日志一致。

| 指标 | MVSplat | DepthSplat RGB | DepthSplat + depth | Depth 相对 RGB 改善场景数 |
|---|---:|---:|---:|---:|
| PSNR | 23.093630 | 24.103386 | 24.141729 | 3/8 |
| SSIM | 0.820270 | 0.840990 | 0.845011 | 5/8 |
| LPIPS | 0.185904 | 0.153459 | 0.143886 | 6/8 |
| Depth AbsRel | 0.210724 | 0.116091 | 0.055639 | 8/8 |
| Depth RMSE (m) | 0.843171 | 0.347110 | 0.132918 | 8/8 |
| Depth a1 | 0.856443 | 0.879966 | 0.976563 | 8/8 |

以上仅为种子 111123 的固定验证子集。场景 40958733 的 RGB 和深度同时改善；42446608 的深度改善但 PSNR 和 LPIPS 退化，保留作为局限案例。这些输入视角深度图不直接验证目标视角深度或三维重建完整性。

在包含原始导出文件的机器上可复现全部对比图（CPU，无需推理）：

```bash
python -m src.scripts.plot_rgbd_comparison \
  outputs/arkit_three_method_visuals_seed111123 \
  outputs/arkit_qualitative_review_seed111123
```

输出全部场景，不筛选获胜样例；深度色标固定为 0.1–10 米，深度误差固定为 0–1 米，RGB 平均绝对误差固定为 0–0.25，超界饱和仅用于显示。数值统计无色标截断或尺度对齐。深度无效区域显示灰色。RGB 误差从 8-bit PNG 计算，图中文字 PSNR 使用评估日志。
