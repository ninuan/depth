# 固定验证样例：三方法定性比较

完成日期：2026-10-23。两套图均为各数据集独立训练后的验证样例，非跨数据集推理。MVSplat、DepthSplat RGB Baseline、DepthSplat + GT depth 使用相同输入与目标；训练种子 111123，固定第 30000 步。主方法保持 λ=0.1、Multi-stage α=0.5。

## 样例选择与核验

每个数据集按验证扫描 ID 排序取前 8 个扫描，每扫描一个固定双输入/单目标组合，保留全部输出。精简图按原顺序显示前两个扫描，未按结果筛选。ScanNet 的 8 个扫描来自 5 个物理地点；精简图的两个扫描来自同一个地点，应按扫描而非独立房间解读。

ScanNet 在推理前写入 selection_plan.json（已嵌入 comparison_audit.json），记录扫描清单、checkpoint 步数和 SHA256；ARKitScenes 沿用此前固定排序导出，其旧导出包没有单独的 selection_plan.json。两套图的三方法输入 RGB、目标 RGB GT、深度 GT、有效 mask 和评估配置均通过对齐核验，原始深度数组重算指标与日志一致。

## 固定子集结果

下表仅是单种子、每数据集 8 个验证扫描的均值；改善数指 Depth 相对同结构 RGB Baseline 的逐扫描改善数。它们用于解释图像，不替代完整测试集三种子主表，也不作为统计显著性的证据。

### ScanNet

扫描：scene0024_00, scene0024_01, scene0024_02, scene0039_00, scene0042_00, scene0042_02, scene0066_00, scene0140_00。

| 指标 | MVSplat | RGB Baseline | + GT depth | 改善扫描数 |
|---|---:|---:|---:|---:|
| psnr | 14.493825 | 16.394203 | 16.715203 | 7/8 |
| ssim | 0.481334 | 0.530138 | 0.538832 | 5/8 |
| lpips | 0.506057 | 0.468771 | 0.462257 | 6/8 |
| depth_abs_rel | 0.195652 | 0.110845 | 0.076932 | 8/8 |
| depth_rmse | 0.766106 | 0.413659 | 0.238540 | 8/8 |
| depth_a1 | 0.763825 | 0.882858 | 0.961329 | 8/8 |

逐扫描配对差值（Depth − Baseline）：

| 扫描 | PSNR | LPIPS | AbsRel |
|---|---:|---:|---:|
| scene0024_00 | +0.523969 | -0.009800 | -0.032348 |
| scene0024_01 | +0.309183 | -0.011631 | -0.022883 |
| scene0024_02 | -0.025857 | -0.004863 | -0.052960 |
| scene0039_00 | +0.592094 | +0.003207 | -0.029781 |
| scene0042_00 | +0.093050 | -0.005530 | -0.011195 |
| scene0042_02 | +0.119791 | +0.003751 | -0.016603 |
| scene0066_00 | +0.498390 | -0.023802 | -0.052785 |
| scene0140_00 | +0.457375 | -0.003440 | -0.052746 |

### ARKitScenes

扫描：40958733, 40958767, 41098086, 41125932, 41126980, 41159825, 42444499, 42446608。

| 指标 | MVSplat | RGB Baseline | + GT depth | 改善扫描数 |
|---|---:|---:|---:|---:|
| psnr | 23.093630 | 24.103386 | 24.141729 | 3/8 |
| ssim | 0.820270 | 0.840990 | 0.845011 | 5/8 |
| lpips | 0.185904 | 0.153459 | 0.143886 | 6/8 |
| depth_abs_rel | 0.210724 | 0.116091 | 0.055639 | 8/8 |
| depth_rmse | 0.843171 | 0.347110 | 0.132918 | 8/8 |
| depth_a1 | 0.856443 | 0.879966 | 0.976563 | 8/8 |

逐扫描配对差值（Depth − Baseline）：

| 扫描 | PSNR | LPIPS | AbsRel |
|---|---:|---:|---:|
| 40958733 | +1.663502 | -0.056315 | -0.073073 |
| 40958767 | -0.286736 | -0.006496 | -0.046509 |
| 41098086 | -0.343344 | -0.003349 | -0.039881 |
| 41125932 | -0.491766 | -0.005339 | -0.198835 |
| 41126980 | +0.559990 | +0.003479 | -0.025009 |
| 41159825 | +0.271311 | -0.007102 | -0.018973 |
| 42444499 | -0.450645 | -0.003997 | -0.033244 |
| 42446608 | -0.615568 | +0.002539 | -0.048091 |

## 观察与局限

两数据集的 AbsRel、RMSE、a1 均在全部 8 个固定扫描上改善，RGB 指标的改善则随扫描和指标变化。ScanNet 子集 PSNR 均值提高 0.320999 dB，但正式完整测试集三种子均值仅提高 0.014543 dB；论文的总体表述仍应为 ScanNet PSNR 基本持平。ARKitScenes 子集 PSNR 只有 3/8 扫描改善。深度准确度提升不保证每个目标视角的 RGB 渲染质量同步提升。

保留的局限样例：

- ScanNet scene0024_02：AbsRel 由 0.126595 降至 0.073635，PSNR 由 11.370596 降至 11.344739。图中目标 GT 较暗，预测 RGB 明显偏亮，窗口、家具区域仍有模糊与重影。仅凭图像尚未确定亮度差异的具体成因。
- ScanNet scene0039_00 / scene0042_02：深度指标改善，但 LPIPS 分别增加 0.003207 / 0.003751。
- ARKitScenes 42446608：AbsRel 降低 0.048091，但 PSNR 降低 0.615568 dB、LPIPS 增加 0.002539；局部边缘及衣物区域仍有渲染误差。

本组深度为输入相机视角的 Z 深度，RGB 为目标视角。二者来自不同视角，图像不直接验证目标视角深度或三维重建完整性。MVSplat 对照反映本项目共享训练预算下的结果；监督本身的增益以同结构 Baseline/Depth 对照为依据。

## 显示约定

- 深度：统一 0.1–10 m；绝对深度误差：统一 0–1 m。超界值仅在显示时饱和，数值统计不截断，不做尺度对齐。
- GT 无效深度区显示灰色；三方法使用相同 mask。
- 全扫描诊断图中的 RGB 误差为 8-bit PNG 的通道平均绝对误差，色标 0–0.25；PSNR/SSIM/LPIPS 数字取自评估日志。
- 精简图每个扫描三行依次为目标 RGB、输入视角 0 的深度、对应深度绝对误差；参考列第三行显示输入 RGB。完整图包含两个输入视角。

## 文件与图注

每个数据集目录包含精简 PNG（300 dpi）及 PDF、全部 8 扫描的 RGB 总览、两个输入视角的深度总览、8 张逐扫描诊断图、comparison_audit.json。总计 24 张 PNG、2 张 PDF、2 份 JSON。精简图用于排版预览；全部样例及失败样例随包保留。

英文图注（将 DATASET 换为 ScanNet 或 ARKitScenes）：

> Qualitative comparison on fixed DATASET validation scans. All methods use the same two input views and target view, training seed 111123, and checkpoints at step 30,000, with training performed on the respective dataset. The displayed scans are the first two of eight scans selected in sorted validation-ID order, without selection by performance. For each scan, rows show target-view RGB, camera-Z depth at the first input view, and absolute depth error; the reference column shows target RGB GT, sensor depth GT, and input RGB, respectively. Depth and error use shared ranges of 0.1–10 m and 0–1 m; out-of-range values are saturated for display only, and invalid sensor-depth pixels are gray. No scale alignment is applied. Improved input-view depth does not imply improved RGB rendering for every target view. All eight scans are included in the accompanying diagnostic figures.

ScanNet 图注补充：The two displayed scans are repeat captures of the same physical location.

## 复现与来源

在云端已有环境中导出 ScanNet（已有结果勿重复执行；再次导出需使用新输出目录）：

```bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0
bash scripts/scannet_compare_visuals.sh
```

使用含原始导出文件的机器进行 CPU 绘图：

```bash
export OMP_NUM_THREADS=1
python -m src.scripts.plot_rgbd_comparison outputs/scannet_three_method_visuals_seed111123 outputs/paper_qualitative/scannet
python -m src.scripts.plot_rgbd_comparison outputs/arkit_three_method_visuals_seed111123 outputs/paper_qualitative/arkitscenes
```

ScanNet 原始导出保留在云端 outputs/scannet_three_method_visuals_seed111123；对应 ZIP 的 SHA256：`5bc1902a8b8062d2940f77c98fbd8866e799ff37f830fa5ad11ca633f7ec009b`。下载后校验一致。ARKitScenes 原始来源为用户提供的 arkit_three_method_visuals_seed111123.zip。展示包包含绘图结果及审计 JSON，原始 RGB/深度数组保留在原导出包。

主实验见 MVSPLAT_RESULTS.md，固定步数消融见 SCANNET_ABLATION_RESULTS.md。后续可据此撰写实验与局限章节；该组可视化不改变已有主表、损失权重或 checkpoint 选择规则。
