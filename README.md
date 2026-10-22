# DJI Action4 双机检测与八角点定位

本项目从原始真实视频出发，利用固定尺寸 DJI Action4 模型生成合成数据，分别训练 B0 完整机身检测器和 A0 八角点/几何模型，输出逐帧标注与 demo 对照视频。

**当前状态：训练、合成集测评和完整视频链路已有实际结果，但真实画面验收未通过。** B0 开发集双机检出 177/184，历史 test 为 194/200；最终权重在 6 张真实参考帧中仅 2 张双机与 demo 参考框同时达到 IoU≥0.5。不得将合成集成绩表述为真实场景达到 90%。demo 是参考可视化，不是经过认证的真实 GT。

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
| `models/A0/`、`models/B0/` | 实际以 `models/A0/best.pt`、`models/B0/best.pt` 两个独立包为准 |
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
  --selection configs/eval/action4_release_candidate_v001.json \
  --split eval --output outputs/new_development_evaluation
```

完整视频入口是 `eval.action4_staged_video_parallel`。它检查 2850 帧身份、顺序、时间戳、模型身份和合并后的帧数/时长；这些只证明视频工程完整性。当前真实相机内参仍为估计值，无跨帧训练或平滑。

## 权重语义

A0、B0 各自独立保存，A0 包含所需冻结 RGB backbone 和任务头，B0 包含检测器及质量/框修正头。继续改进以 best 权重初始化新实验；没有 best 时刻优化器状态的历史包只支持新优化器热启动，不能声称精确续接最后一次训练。示例配置 `action4_a0_continue_best_v001.json` / `action4_b0_continue_best_v001.json` 尚未启动，不能当作已完成的新实验。

## 许可

保留仓库 Apache-2.0 [LICENSE](LICENSE)。复用 GDR-Net 模块的许可证、来源及改动说明在 `src/model/heads/GDRNET_LICENSE.txt` 和相邻 provenance 文件中。第三方数据、模型与素材的授权不由本仓库代码许可证替代。
