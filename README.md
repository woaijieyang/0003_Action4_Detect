# DJI Action4 双机检测与八角点定位

本项目从原始真实视频出发，利用固定尺寸 DJI Action4 模型生成合成数据，分别训练 B0 完整机身检测器和 A0 八角点/几何模型，输出逐帧标注与 demo 对照视频。

**当前状态：本轮 A0/B0 训练、eval/test、完整真实视频、独立模型封存和文件整理已完成；真实画面精度仍未通过任务验收。** 当前执行登记为 r067，已结束运行不会自动重启。

B0 摄影增强80轮候选 `configs/eval/action4_photometric_best_v001.json` 使用best3220：开发双机178/184、历史test196/200；完整视频中2773个可判定demo包络帧的双机参考一致性为2268/2773（81.79%），77帧不可评分。demo是参考可视化，不能把这个数当认证真实召回率或把合成成绩当真实达到90%。

A0 在线RGB裁剪训练已完成80轮/15120step/606次异步eval，best14000与B0组合的入口为 `configs/eval/action4_a0_online_rgb_best_v001.json`。完整RGB开发八点全部正确171/378、历史test201/400，低于保留PnP基线221/378、253/400；但真实视频双机投影包络一致性由273/2773升至1356/2773，减少巨大漂移，仍有角点方向与检测错误。包络一致性不能证明角点ID正确。因此两种A0研究选择分开保留，无全面达标模型。

2850帧同帧对照视频在服务器 `outputs/action4_a0_online_rgb_validation_v001/video/comparison.mp4`，逐帧预测、模型身份、视觉审查及失败说明同目录上级可查。源码发布不包含模型、数据和视频资产。

## 当前入口

- [阶段计划](docs/stages/2026-10-19-1-plan-Action4角点复现增补.md)、[阶段结论](docs/stages/2026-10-19-1-conclusion-Action4角点复现增补.md)。
- [代码与数据流](docs/ARCHITECTURE.md)、[模型接口及继续训练](docs/action4_model_interfaces.md)。
- [结果表](results/results.md)、[发布记录](README_WORKFLOW.md)。
- 当前执行身份由 `configs/execution/action4_active.json` 指向；不要按文件时间选择旧控制器。

## 目录职责

| 目录 | 内容 |
|---|---|
| `src/data` | 合成、角点/掩码数据、划分校验、视频抽帧和数据接口 |
| `src/model` | 独立 A0/B0、几何模块和自包含权重包 |
| `src/model/training` | 损失、训练、异步评估队列和恢复 |
| `src/eval` | 框指标、八点指标、实际视频与可视化 |
| `configs` | 训练、评估、采样与执行配置 |
| `data/train,eval,test` | 1600/200/200 张已发布数据；test 全双机 |
| `models` | CURRENT_RESEARCH登记独立A0/B0及候选；models/A0为合成基线，candidates/a0_online_rgb_v001为真实视频候选，candidates/b0_photometric_v001为当前B0 |
| `outputs` | 必要的运行证据、预测、视频与仍被消费的缓存 |
| `archive/legacy` | 历史方案与失败证据，按原相对路径归档 |
| `docs`、`results` | 方案、结论与可核对成绩 |

模型权重、数据、Blender 场景、视频、运行缓存和历史归档不进入普通 Git；服务器保留这些资产。仓库克隆只得到源码与配置，必须自行配置合法数据和权重，不能把缺失资产解释为可直接复现全部成绩。

## 运行约定

服务器项目根是 `/data/zy/vln/a-test`，以下命令均从该目录执行。项目输入输出必须留在项目内。现有运行时为 `.venv/bin/python`；OpenCV 补充依赖在 `runtime/surface_deps`，FFmpeg 在 `runtime/ffmpeg/bin`。

```bash
export PYTHONPATH=runtime/surface_deps:src
export TMPDIR="$PWD/runtime"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# 先按最新授权选择有空闲资源的物理 GPU；GPU0 禁止使用。
.venv/bin/python -m eval.action4_staged_eval \
  --selection configs/eval/action4_photometric_best_v001.json \
  --split eval --output outputs/new_development_evaluation
```

完整视频入口是 `eval.action4_staged_video_parallel`。它检查 2850 帧身份、顺序、时间戳、模型身份和合并后的帧数/时长；这些只证明视频工程完整性。当前真实相机内参仍为估计值，无跨帧训练或平滑。

## 权重语义

A0、B0 各自独立保存，A0 包含所需冻结 RGB backbone 和任务头，B0 包含检测器及质量/框修正头。继续改进以 best 权重初始化新实验；没有 best 时刻优化器状态的历史包只支持新优化器热启动，不能声称精确续接最后一次训练。示例配置 `action4_a0_continue_best_v001.json` / `action4_b0_continue_best_v001.json` 尚未启动，不能当作已完成的新实验。

## 许可

保留仓库 Apache-2.0 [LICENSE](LICENSE)。复用 GDR-Net 模块的许可证、来源及改动说明在 `src/model/heads/GDRNET_LICENSE.txt` 和相邻 provenance 文件中。第三方数据、模型与素材的授权不由本仓库代码许可证替代。
