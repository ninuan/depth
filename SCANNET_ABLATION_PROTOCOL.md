# ScanNet 固定 30000 步消融协议

2026-10-23 用户明确确定：主实验和消融均使用实际 `global_step=30000` 的 checkpoint，后续不切换回验证集最优 checkpoint。主方法保持深度权重 λ=0.1、中间深度权重 α=0.5（Multi-stage）。本轮结果用于报告消融，不重新选择主方法参数。

本轮八组评估已完成，结果与结论见 [消融结果](SCANNET_ABLATION_RESULTS.md)。下列命令保留用于复现，已完成的实验无需重复运行。

## 本轮任务

| 比较 | 训练种子 | 评估划分 | 新评估数量 |
|---|---|---|---:|
| 有无深度监督 | 111123、222123、333123 | 已有固定测试集 159 扫描 | 0，复用已完成主表 |
| Multi-stage vs Final-only，λ=0.1 | 111123、222123、333123 | 固定验证集 48 扫描 | 6 |
| λ=0.05 / 0.1 / 0.2，α=0.5 | 111123 | 同一验证集 48 扫描 | 2，复用上述 λ=0.1 组 |

共评估 8 个已有 `last.ckpt`，本轮无需新训练。验证集消融与测试集主表分别标明划分，不混算均值。已有的少量固定步数验证结果在本轮独立目录中统一复评，旧结果保留。

Final-only 对应训练时 `train.depth_intermediate_weight=0.0`；Multi-stage 为 `0.5`。两者都保留中间 RGB 损失、相同网络结构以及对最终深度的监督。评估时更改损失权重不会把已训练的 Multi-stage 模型变成 Final-only，因此必须读取各自训练目录的权重与配置。

## 执行

在云端已激活的环境内运行：

```bash
cd /root/autodl-tmp/depthsplat
git pull --ff-only origin main
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
python -m src.scripts.scannet_ablation_fixed30000
```

默认数据目录为 `datasets/scannet_processed_v2`，输出为 `outputs/scannet_ablation_fixed30000`。可用 `--dataset` 和 `--output` 指定路径。仅检查文件、配置和步数时追加 `--check-only`。

脚本先检查全部 8 组训练配置、真实训练步数、验证集扫描数，以及网络、优化器、RGB 损失和训练加载器等关键控制变量。任何 checkpoint 缺失、步数不对或控制变量不一致时，在 GPU 推理前报错，保留信息供排查，不自动改用其他 checkpoint。

训练加载器的随机种子按训练种子配对核对，不要求不同训练种子的随机数一致。云端原始配置为：训练种子 111123 的加载器 seed=1234，222123 的 seed=222123，333123 的 seed=333123；各自的 Multi-stage / Final-only 配对一致，λ 敏感性实验均为 seed=1234。batch size、worker 数等设置仍在全部实验间核对。原始训练加载器配置保存到结果的 `training_data_loader`，不改写历史训练配置。

实际评估以各自保存的完整训练配置为基础，切到 `mode=test`、`dataset.test_split=val`，关闭可视化导出，保持原网络和损失配置。评估随机种子固定 111123，测试加载器种子 2345；训练种子由 checkpoint 决定。

前台串行评估，实时显示进度。每组完成后检查六项指标都有 48 个有限值，且逐场景均值与汇总一致，再保存完成记录。网络中断后重复同一命令即可：跳过已完成且源码、配置、checkpoint、数据 manifest/index 指纹未变化的组；中断的那一组从头评估。若指纹发生变化则报错，防止混用新旧结果。

## 输出与分析

最终发送：`outputs/scannet_ablation_fixed30000/summary.json`。

文件包含每组 checkpoint 路径、训练种子、实际步数、λ、α、文件指纹和六项得分，以及：

- 监督位置表：三个训练种子的均值 ± 样本标准差（ddof=1）。
- 每个种子的 `Final-only − Multi-stage` 配对差值。
- 权重敏感性表：种子 111123 的三组数值；该表仅为单种子敏感性分析，不标三种子标准差或跨种子稳定性。

本轮冻结配置，不因某项指标胜负更改主方法。若 Multi-stage 没有一致优势，按实际数据写明其增益有限或指标之间存在取舍。三个训练种子不自动构成统计显著性证据。

预期训练目录：

```text
outputs/scannet_depth
outputs/scannet_depth_seed222123
outputs/scannet_depth_seed333123
outputs/scannet_depth_finalonly_seed111123
outputs/scannet_depth_finalonly_seed222123
outputs/scannet_depth_finalonly_seed333123
outputs/scannet_depth_w005_seed111123
outputs/scannet_depth_w020_seed111123
```
