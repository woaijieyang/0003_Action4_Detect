# Action 4 训练、测评与视频执行

这里已经实现真实的训练循环和状态恢复。当前是否有合格的模型，必须查看对应运行的 `completion.json`、D0 gate、eval 原始结果及完整视频复核；代码存在和工程测试通过不代表训练完成。CPU 上用 `--component-smoke` 运行的检查会把 `component_diagnostic` 写入 checkpoint，不能用来冻结正式 test，视频会带组件检查水印。

## 输入与入口

所有命令在项目根、已经安装本项目的隔离环境运行。未安装时可显式设 `PYTHONPATH=src`。共用配置是 `configs/model/action4_train.json`，配置优先级为显式 CLI、显式 `--local-config`、共享配置；不会自动读取本地配置。正式运行默认要求 CUDA，GPU wheel 必须与实际驱动适配。默认导入不下载权重，不加载模型，也不启动设备。

`initialization.backbone_weights` 指向 `models/pretrained/resnet50-11ad3fa6.pth`，SHA-256 固定为 `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`。它是 Torchvision ResNet50 `IMAGENET1K_V2` 的分类预训练权重；模型只移除 `fc.weight/fc.bias`，严格校验其余主干参数。D0 和 A0 分别从相同种子的新主干/新头开始；B0 只加载 A0 模型参数，C1 只加载 B0/C1 模型参数，然后建立新优化器和新阶段调度。来源和校验见 `configs/model/pretrained.lock.json`。

正式 Dataset 仅接受 `status=training_accepted` 的清单。训练、eval 和 test 的 `sample_id/group_id/background_source_id` 必须互不跨集合，S1 只能追加 train。代码统一使用 `data.action4_dataset.is_ignored`，`ignore=true`、`unobservable=true` 或 `status=unobservable` 的目标不进入有效角点监督。数据准入不能靠改状态字符串替代几何和外观验收。

正式 B0/C1 在开始长训练前还要求配置中的开发视频缓存完整、帧号连续有序；应先运行数据入口建立该缓存。自动视频钩子只属于 B0/C1。组件训练默认不在轮末自动处理完整视频，若要检查视频 IO，单独运行视频入口并显式允许组件权重；这类视频始终带组件检查水印。

| 入口 | 实际作用 |
|---|---|
| `scripts/action4_train.py` | D0、A0、B0、C1；离线初始化、DDP、AMP、累积、完整 epoch 恢复、C1 回放 |
| `scripts/action4_eval.py` | 固定 eval/test、独立 GT-ROI 诊断、有限解码网格搜索、写入正式 test 冻结选择 |
| `scripts/action4_video.py` | 完整原始 PNG 清单、主模型 RGB 推理、浮点八角点、红色线框、精确 PTS MP4 |
| `scripts/action4_feedback.py` | 自动疑似事件、真实原图截图、已复核事件到有界 S1 生成请求 |

## D0 → A0 → B0

下面的 `8` 是一个服务器运行配置例子；只能在确有八张已授权 GPU 时使用。其他卡数也支持，但 `world_size × micro_batch` 必须整除阶段的 `global_batch`。默认 ROI 阶段全局 batch 为 32，B0/C1 为完整图像 batch 8，`micro_batch=1`，累积次数由代码计算。

```bash
torchrun --standalone --nproc_per_node=8 scripts/action4_train.py --config configs/model/action4_train.json --stage D0 --run-id D0_v001
torchrun --standalone --nproc_per_node=8 scripts/action4_train.py --config configs/model/action4_train.json --stage A0 --run-id A0_v001 --d0-gate outputs/action4/D0_v001/d0_gate.json
torchrun --standalone --nproc_per_node=8 scripts/action4_train.py --config configs/model/action4_train.json --stage B0 --run-id B0_v001 --d0-gate outputs/action4/D0_v001/d0_gate.json --init-weights models/action4/A0_v001/best.pt
```

这些命令有顺序依赖，上一阶段验收失败时不能继续。D0 固定取 32 个不同的有效 ROI，关闭 ROI 扰动，最多 500 次更新；先实际核查所用 ROI 的仿射往返、坐标渐变图通过共享 RGB 重采样后的位置误差，以及实际网络的空目标 loss。前两项容差都是 0.001 原图像素。训练后要求理想热图解码 P95≤1 热图像素、模型预测 P95≤2 热图像素、相对全零热图的 P95 至少降低 80%、有限非零主干梯度和实际参数变化。只有正式数据上的非组件运行能通过 D0 gate；达到条件可提前结束，失败保留报告。

A0 遍历正式 train 的有效 ROI，每全局 32 个 ROI 更新一次，共 10 epoch。B0 每全局 8 张完整图更新，共计划最多 60 epoch。A0 前 500 次更新丢弃主干梯度，AdamW 不更新主干参数、动量和 weight decay；DDP 从建立时就登记全部主干参数，避免中途改变 `requires_grad` 导致不同步。B0 前五轮用扰动 GT 完整框做 ROI，第六轮起以 50% 概率尝试与 GT 一一匹配且 IoU≥0.5 的预测框，失败回退 GT 并记录数量。完整联合 forward 位于一个 DDP 调用内；离散选择的检测结果 detach，ROI 分支仍更新共享主干与角点头。

主干 BN 的运行统计始终冻结，仿射参数按阶段更新。AdamW 主干/新层初始 LR 分别为 `1e-5/1e-4`，bias 和一维归一化参数不做衰减，其余 `weight_decay=1e-4`。A0/B0 500 次 warmup 后按事先定义的完整阶段预算 cosine 衰减至 1%；D0 用其独立预算。每次累积窗口按实际完整图像数归一化，DDP 尾批次只补零权重项，不重复计入真样本。AMP 在所有 micro-batch 累积后 unscale，再裁剪梯度范数至 1；出现非有限值时各 rank 一致停止更新，调度不前进。

每轮 rank0 独立遍历完整 eval，不使用有 padding 的分布式 eval sampler，因此每个 sample_id 只计一次；其他 rank 等待。本实现没有声称对 eval 做了多卡加速。ROI 诊断和全图自动定位分别保存。模型选择按 `F1@5%`、`F1@2%`、成功匹配角点平均 EPE 的顺序，保留 top3；坏角点仍同时计 FP/FN，不从误差表中消失。权重只保留 top3、最新完整 epoch 的 `last.pt`、B0 `joint_init.pt` 及第20轮必要快照，使用硬链接减少重复存储；每轮指标、原始预测和日志保留。只退休本次运行此前 top3 索引登记的非保留权重，不触及其他运行和输入。

## 中断恢复与 C1 继续训练

```bash
torchrun --standalone --nproc_per_node=8 scripts/action4_train.py --config configs/model/action4_train.json --stage B0 --run-id B0_v001 --d0-gate outputs/action4/D0_v001/d0_gate.json --resume models/action4/B0_v001/last.pt
```

`resume` 只能使用同一 run 当前最新、完整 epoch 边界的 `last.pt`。模型、数据清单哈希、配置、源码、world size、micro/global batch 必须一致；恢复 model、scheduler、optimizer、GradScaler、各 rank Python/NumPy/Torch/CUDA RNG、ROI RNG 和下一轮 sampler。按 PyTorch 要求先初始化并恢复 scheduler，之后加载 optimizer。新尝试写独立日志目录，保留此前的部分日志；旧 checkpoint 不能覆盖已经前进的运行。完成预算后不能继续沿用衰减到末端的 scheduler 冒充新训练。

C1 从选定 B0 权重开始新 run，S0 原池保持不变，每虚拟 epoch 轮换抽 6,000 张 S0，加上 2,000 张独立 S1，再打乱成 8,000 个槽；两个虚拟 epoch 可覆盖完整 8,000 张 S0。S1 不够 2,000 张或 S0 不够 6,000 张会明确拒绝默认配额。沿用固定 eval/test。主干/新层 LR 为 `3e-6/3e-5`，新建 200 次 warmup 和最多20轮调度；不是恢复 B0 的 optimizer。

```bash
torchrun --standalone --nproc_per_node=8 scripts/action4_train.py --config configs/model/action4_train.json --stage C1 --run-id C1_v001 --d0-gate outputs/action4/D0_v001/d0_gate.json --init-weights models/action4/B0_v001/best.pt
```

C1 首先保存 B0 起点在同一 eval 上的 `initial_eval/`，后续保存各场景召回变化。默认在第5、10、15轮完整边界写好 checkpoint、指标和完整视频后，返回 `paused_at_epoch_boundary_for_video_review`。实际查看原失败段和普通段后，确认改善才用同一 C1 run 的最新 `last.pt` 执行 `--resume`。这种暂停用于质量复核，代码不会把自动低置信度事件当成错误真值，也不会自动断言视频改善；若不改善，应保留该版本并修正原因。第20轮完成后仍需要最终视频复核。

## eval、调参和 test

```bash
python scripts/action4_eval.py --config configs/model/action4_train.json --checkpoint models/action4/B0_v001/best.pt --split eval --run-id B0_eval_v001
python scripts/action4_eval.py --config configs/model/action4_train.json --checkpoint models/action4/B0_v001/best.pt --split eval --gt-roi --run-id B0_roi_diagnostic_v001
python scripts/action4_eval.py --config configs/model/action4_train.json --checkpoint models/action4/B0_v001/best.pt --split eval --tune-grid configs/eval/action4_decode_grid.json --run-id B0_decode_tuning_v001
```

默认网格显式列出 score `[0.2,0.3,0.4]` 和严格去重 IoU `[0.95,0.98]`，一共六个组合；不会无限搜索。每个组合运行完整 RGB pipeline，只使用 eval，保存其配置、完整原始预测和全部指标；输出 `selected_config.json`。输入尺度、ROI 规则、角点编号和 test 不在该搜索范围。最多12个唯一组合的上限是本实现的预算界限；需要扩展时明确制定新协议。

```bash
python scripts/action4_eval.py --config outputs/action4/B0_decode_tuning_v001/selected_config.json --checkpoint models/action4/B0_v001/best.pt --freeze-selection outputs/action4/final_selection_v001.json
python scripts/action4_eval.py --config outputs/action4/B0_decode_tuning_v001/selected_config.json --checkpoint models/action4/B0_v001/best.pt --split test --selection outputs/action4/final_selection_v001.json --run-id final_test_v001
```

最后两条只能在完整视频问题已经复核、模型与规则决定冻结后运行。冻结文件绑定 checkpoint、网络/预处理/解码配置、源码及 test manifest；正式 test 会核对全部身份，拒绝 D0/A0、组件 checkpoint、未接受数据和 GT-ROI 输入。test 暴露问题后保留本报告，后续修复属于复查；不能悄悄改 test 再宣称独立成绩。EPE 仅描述成功匹配角点，需与 TP/FP/FN、错角、漏检、ignore、空图和困难分组一起解释。

## 完整视频和失败反馈

```bash
python scripts/action4_video.py --config outputs/action4/B0_decode_tuning_v001/selected_config.json --checkpoint models/action4/B0_v001/best.pt --frame-manifest data/action4_real/cache_v001/frames_manifest.json --run-id B0_full_video_v001
python scripts/action4_feedback.py --run-dir outputs/action4/B0_full_video_v001 --output outputs/action4/B0_feedback_candidates_v001
python scripts/action4_feedback.py --run-dir outputs/action4/B0_full_video_v001 --reviewed-events outputs/action4/B0_reviewed_events_v001.jsonl --output outputs/action4/B0_feedback_reviewed_v001
```

视频只接受完整且哈希一致的原始 PNG 缓存。每帧一次 RGB 推理，保存 `frame_id/PTS/time_base`、原图/模型/配置哈希、分数、原图浮点八角点和不确定标记；没有 GT 框输入、旧帧补框或隐藏时序平滑。红线连接固定 `0–1–2–3–0、4–5–6–7–4、0–4、1–5、2–6、3–7`，只在绘制时裁切越界线，原始数值不 clamp。输出原尺寸 `outputs/action4/<video_scope>/drawn_frames/frame_%08d.png`、`predictions_raw.jsonl`、`full_red_raw.mp4` 与 `.verification.json`；DataAdapter 核对本视频的 2850 帧、`2997/100` 精确帧率和逐帧显示时间戳。`review_coverage.csv` 初始每帧是 `pending_review`，不会伪造已经完整观看。

候选事件来自低置信度、数量变化和角点运动跳变，相邻事件合并并保存原图截图；真实运动、进出画也是可能解释。只有带 `confirmed_failure`、复核者、原图证据和现象的事件，才可生成 S1 请求。全局坐标/时间错误先输出接口修复请求，不用加数据遮盖。每事件默认最多占 2,000 张预算的10%；事件不足时保留未分配配额。

`synthesis_requests.jsonl` 与 `DataAdapter` 的 `plan-s1` 共用一份合同：每行有唯一请求/来源事件、准确 quota、`split=train`，recipe 使用八个已实现场景家族之一；`view.wrist_delta_xyz_deg` 是相对该家族锚姿态的三个 ±8° 以内偏移，`scale` 为 `[0.9,1.1]` 的相对摄影机距离因子。只接受 `occluder=scene_default` 和 `occluder_location=linked_geometry`，其遮挡通过腕部连接几何、家族选择和小旋转实现。扰动字段固定为 `rotation_jitter_deg∈[0,8]`、三个 `position_jitter_m∈[0,0.004]`、`light_intensity_gain∈[0.9,1.1]`、`exposure_delta_ev∈[-0.15,0.15]`，同时保存原视频参考帧。超出这些已实现能力会报错；渲染器重新计算几何真值，仍须验收，失败候选不能自动进入训练。

## 验证与适用范围

训练器工程检查用小型可训练控制网络验证优化器、恢复、DDP、计分和视频 IO，不把它当作 ResNet 成绩；正式 ResNet 的前向、损失、主干梯度和参数更新由独立模型测试及组件实验核查。两进程 Gloo 测试需要本机回环 socket 权限，隔离环境禁止 socket 时应保留失败并通过正常权限流程运行，不能删除或跳过该验证。测试入口为 `python -m unittest discover -s tests -v`；每次使用实际日志说明哪些检查运行、哪些因环境受阻。

一手依据：[PyTorch 2.8 AMP 示例](https://docs.pytorch.org/docs/2.8/notes/amp_examples.html)、[DDP](https://docs.pytorch.org/docs/2.8/generated/torch.nn.parallel.DistributedDataParallel.html)、[Optimizer.load_state_dict](https://docs.pytorch.org/docs/2.8/generated/torch.optim.Optimizer.load_state_dict.html)。项目的匹配、回放、图像权重和视频验收规则属于本任务的明确协议，不是这些文档承诺的模型效果。
