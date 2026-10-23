# Action4 模型与数据接口说明

本文保留历史接口并在末尾记录r071当前入口；顶部旧配置不是当前启动配置。本文说明**现有实现**的调用合同、坐标、产物及基本检验，不另立执行计划。任务顺序、数据准入、训练预算及停止条件以 `docs/stages/2026-10-19-1-plan-Action4角点复现增补.md` 及其后续用户修订为准。本接口检查没有启动正式训练、GPU 作业或队列。

本次核对配置为 `configs/model/action4_pilot2000_v002.json`，其 SHA-256 为 `9d328f71d0ffd8ef7e66180bfcb25919fe059e4903f518ebedd2ad8d08d6f20c`。配置指向未来接受后的 2,000 张数据（train/eval/test=1,600/200/200）；**配置路径存在不代表数据已经生成或接受**。任何后续改动须重新核对当前配置与证据，不用本文中的历史哈希冒充当前状态。

## 1. 模块边界与实际入口

| 接口 | 现有实现 | 输入 → 输出 |
|---|---|---|
| 视频与样本数据合同 | `data.action4_adapter.DataAdapter` | 原视频/清单 → 原生帧、精确时间轴；JSONL → 校验后的样本 |
| 数据与统一采样 | `data.action4_dataset` | RGB CHW、完整机身框 → ImageNet 归一化 full/ROI 张量及原图到画布矩阵 |
| 模型 | `model.action4_network.Action4Net` | 全图/ROI → 检测 dense maps、固定八通道角点 logits |
| 监督与损失 | `data.action4_dataset`、`model.action4_losses` | 几何 GT、实例对应关系 → center/offset/ltrb/角点热图及逐图归一化损失 |
| 原图解码 | `model.action4_decode`、`eval.action4_runner` | 检测输出 → 虚拟 ROI → 八点原图像素；真实推理仅输入 RGB |
| 训练 | `model.training.action4_trainer.train` | 配置、数据、D0 gate、初始化/恢复权重 → 权重、逐轮评估与视频事件 |
| 固定评估 | `eval.action4_runner.run_evaluation` | 冻结权重/配置、eval 或已锁定 test → 原始预测与指标 |
| 视频 | `eval.action4_video.run_video` | 合格权重、完整原帧清单 → 原始逐帧预测、原尺寸叠图、MP4、待审核清单 |
| 资源与视频事件 | `model.training.action4_dispatch.Dispatcher`、`action4_video_queue.VideoQueue` | 绑定身份的作业/不可变 checkpoint → 受控执行、收据与审核状态 |

`Action4Net` 不读取标签、不导入评估模块。`predict_images(model, images, config, device)` 没有参考框或关键点参数；标签仅由评估器在预测完成后用于计分。`evaluate_gt_rois` 是明确的 GT-ROI 诊断，不能替代全图成绩。当前没有额外 HTTP 服务或新模型协议，使用已有 Python 接口和薄 CLI。

## 2. 数据输入合同

`Action4Dataset(root, manifest, *, allow_diagnostic=False, split=None)` 返回 `{'image': uint8 RGB[3,H,W], 'record': 原样本记录}`；`collate_fn` 保留样本列表，允许原生尺寸不同。模型输入前必须通过公共采样函数，不能将 uint8 直接输入网络。

JSONL 每行遵循 `action4.v1`：

- 样本必需 `sample_id/image/width/height/group_id/background_source_id/scene_family/instances`；`image` 为相对项目根的 PNG 路径，解析后仍在项目内，实际尺寸必须一致。
- `instances` 数量为 0、1 或 2；每个实例 `instance_id` 在本帧唯一、`object_id=1`，`corners_px` 为有限 `[8,2]`，`bbox_full_xyxy` 必须等于这八点的 min/max 包络，`corner_visible` 为八个布尔值。
- 不可忽略实例需要真实可见 mask 得到的 `bbox_visible_xyxy`。`ignore/unobservable/status=unobservable` 实例不会成为角点监督 ROI；它们仍保留记录，不能直接改写成无目标背景。
- 训练准入需要 `status=training_accepted` 与 `rgb_sha256`，加载时重新核对 RGB 字节哈希。`allow_diagnostic=True` 仅用于独立组件检查，不可用它把候选数据转成接受数据。
- train/eval/test 的组及背景来源隔离由训练器检查；不能先随机分相邻帧再将相同来源当独立验证。
- 本体标签对应完整 DJI Action4，腕带、安装底座、线缆及手臂不属于机身 mask。遮挡情况下保留完整八点投影；可见框不能替代完整框。

原视频帧位于 `data/action4_real/cache_v001/frames/`，完整清单为 `data/action4_real/cache_v001/frames_manifest.json`。demo 原帧独立保存在 `data/action4_reference/demo_standard_cache_v001/`；它是参考效果，不能混入目标数据或充当模型 GT。

## 3. 张量与坐标合同

| 对象 | 形状/语义 |
|---|---|
| 原始 RGB | uint8 `[3,H,W]` 或有限 float `[3,H,W]`，float 范围 `[0,1]`，RGB 顺序 |
| `prepare_full_images(images,size=(1024,768),device=...)` | `[B,3,768,1024]`，同时返回每图 `original→full` 的 `3×3` 矩阵 |
| `prepare_rois(images,boxes,image_indices,roi_size=256,expansion=1.25,device=...)` | `[N,3,256,256]`，返回每 ROI `original→ROI` 矩阵 |
| `center_logits` | `[B,1,192,256]`；未 sigmoid 的 logits |
| `offset` | `[B,2,192,256]`；检测中心相对整数格点的热图坐标偏移 |
| `ltrb` | `[B,4,192,256]`；softplus 后非负距离，按 full 的 `[W,H,W,H]` 归一化 |
| `corner_logits` | `[N,8,64,64]`；八个固定编号通道，未 sigmoid |
| corner target / valid | `[N,8,64,64]` / bool `[N,8]`；valid 表示监督域有效，不等于角点物理可见 |
| 解码角点 | `[N,8,2]`，每点 `(x,y)`，映射完成后为原图像素中心坐标 |

所有 `size` 参数按 **(W,H)**，张量空间维度按 **(H,W)**。原生目标视频为 3248×2464，但 full 网络输入为 1024×768 的 letterbox，不能把原生比例直接当成 full 画布比例。

像素中心 0 对应第一个像素，整幅图边界为 `[-0.5,W-0.5] × [-0.5,H-0.5]`。采样使用 `align_corners=False`；缩放映射有 half-pixel 平移。完整 ROI 框先取最长边×1.25 的正方形，**不裁到原图范围**，越界部分以黑色 RGB 补边，然后按 ImageNet mean `[.485,.456,.406]`、std `[.229,.224,.225]` 归一化。不得在归一化后直接补零并称黑边。

对于 64×64 角点热图到 256×256 ROI，`x_roi=4*x_hm+1.5`，y 同理。`heatmap_to_original(affines,256,(64,64))` 返回 `inverse(original→ROI) @ heatmap→ROI`；禁止再次添加 0.5 或重复做 [-1,1] 转换。

八通道顺序继承现有资产规范坐标系与 manifest 投影顺序，不按屏幕左右、上下或可见性重新排序。绘制边固定为 `(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)`。这是完整长方体的 12 条边，不是屏幕四角。`draw_cuboids` 只裁绘制线段，不能裁改或补齐原始预测坐标。

检测中心监督取可见 bbox 中心，回归目标是完整八点包络；两个有效实例落同一热图格点会报错，不能后者覆盖前者。角点 `make_corner_heatmaps` 对 ROI 热图域内的隐藏角点继续监督；域外角点保留几何记录、valid=false，不能夹到边缘造标签。

## 4. 模型调用与解码

当前结构是一个共享 ResNet-50 主干和共享三层上采样 decoder；检测头为 1/2/4 通道，角点头为 8 通道。不是为每台相机单建网络。`freeze_bn=True` 固定主干 BN 运行统计，仿射参数仍可训练，阶段冻结由训练器负责。

`Action4Net(weights=None, **config['network'])` 不下载；显式 `load_backbone_weights(path, sha256=...)` 严格加载本地完整 torchvision ResNet50 权重，只去掉 `fc.weight/fc.bias`。不要把 Action4 完整 checkpoint 传给该函数。`weights='IMAGENET1K_V2'` 会显式触发 torchvision 的权重缓存/下载路线，本项目已有本地验证权重，不需要采用该路线。

可用调用：

```python
# 以下是已存在函数的调用模板；变量来自上文真实数据准备，不执行训练。
full, original_to_full = prepare_full_images(images, size=(1024, 768), device=device)
rois, original_to_roi = prepare_rois(images, full_boxes, image_indices,
                                    roi_size=256, expansion=1.25, device=device)
dense_and_corners = model(full, roi_images=rois)
roi_only = model(None, roi_images=rois)['corner_logits']
raw_instances = predict_images(model, images, config, device)
```

空 ROI 返回 `[0,8,64,64]`，不制造占位实例。`model(full, roi_builder=callback)` 支持在一次 DDP forward 中构建 ROI；callback 接收 detached dense maps，返回归一化 ROI 和 metadata，网络原样返回 metadata。它与显式 `roi_images` 互斥。

`decode_detections` 对中心 logits sigmoid 后做局部 NMS/top-k，按阈值与近重合规则取最多两个候选；空结果合法。`decode_corners` 每通道取最高 20 个像素位置的均值，保持编号，不是 soft-argmax。常量热图没有可靠定位含义，解码出坐标本身不能证明目标识别成功。

`predict_images` 返回每张图的实例列表，每实例包括 `score/center_px/bbox_full_xyxy/instance_id/object_id/corners_px/corner_peak_scores/uncertainty_flags`。其中 `instance_id` 是本帧枚举，**不是跨帧跟踪 ID**；当前视频 `association=None/smoothing=None`。`low_corner_peak_heuristic` 只是启发式不确定性标记，不能作为已校准置信度或人工审核结果。

## 5. 训练、权重、恢复和准入

`train(root,config,*,stage,run_id,init_weights=None,resume=None,d0_gate=None,allow_diagnostic=False,component_smoke=False,max_updates=None)` 是唯一现有训练入口。

| 阶段 | 初始化和合同 | 先导配置预算 |
|---|---|---|
| D0 | 从声明的 ImageNet 主干初始化，固定首 32 个有效 ROI；诊断权重不用于 A0 初始化 | 至多 500 更新，global32/micro1，corner bias0.1、新模块 LR1e-3 |
| A0 | 同一声明的 ImageNet 主干重新初始化，要求当前数据/方法对应的成功 D0 gate | 2 epochs，global32/micro1，warmup20/freeze20；实际有效 ROI 数必须支持该预算 |
| B0 | `--init-weights` 使用 A0 task 权重，重新创建 optimizer/scheduler | 5 epochs，global8/micro1，warmup100，第2轮起混合预测 ROI，概率0.5 |
| C1 | B0/C1 权重 + 显式 S1 + S0 回放，重新创建 optimizer/scheduler | 当前配置保留原 6000 S0+2000 S1 协议；2,000先导不能直接满足，不自动调用或降配 |

D0 gate `action4.d0_gate.v2` 同时绑定实际源文件、训练 manifest、固定 ROI/图像身份、初始化权重与方法配置，要求几何检查、主干真实梯度/参数变化、理想热图 P95≤1px、学习后 P95≤2px、相对零热图改善≥80%。组件/诊断记录不能签发正式通过；哈希变化后不能手工更新旧 gate 让它过关。

权重保存在 `models/action4/RUN_ID/`，运行结果保存在 `outputs/action4/RUN_ID/`。checkpoint schema 为 `action4.checkpoint.v1`，包含 model、optimizer、scheduler、scaler、config、identity、stage、run_id、claim_scope、epoch_boundary、next_epoch、global_step、逐 rank RNG/ROI RNG、sampler 和 top_checkpoints。

- `--init-weights` 是跨阶段**仅加载模型参数**；B0 需要 A0，C1 需要 B0/C1。它不会继承旧 optimizer 和学习率进度。
- `--resume` 是同 run 的完整状态恢复，只接受最新已提交 `last.pt` 和完整 epoch 边界；run/data/config/source/topology 身份一致，恢复 optimizer/scheduler/scaler/RNG。不能用较旧 best.pt 覆盖较新训练。
- 两参数互斥；D0/A0 不接收旧 task 权重作为初始化。
- `--max-updates` 只允许显式 `--component-smoke`，不能把截短运行称作完整训练。`--allow-diagnostic` 也不能替代数据接受。
- GPU 数改变不改变 global batch；必须满足 `global_batch % (world_size*micro_batch)==0`。只使用物理卡4–7，资源选择与多进程准入由主控制器和 canonical Dispatcher 管理。

## 6. 真实推理、视频和排队审核

`run_video` 校验 complete 原生帧清单、0..2849 顺序、每帧 PNG 哈希/尺寸、PTS 与 duration。正常视频需要 accepted-data B0/C1 的完整 epoch checkpoint；组件权重需要 `--allow-component` 且会带诊断水印，不能写 `final_inference`。

每个视频运行的实际产物：

- `outputs/action4/RUN_ID/predictions_raw.jsonl`：逐帧 frame_id/PTS/time_base/source_image/hash、checkpoint/config hash 和实例预测；零检测帧也保留。
- `drawn_frames/frame_%08d.png`：原尺寸红色 cuboid；与原始帧分开。
- `review_coverage.csv`：初始全部 `pending_review`，必须实际审核后记录 reviewer/notes，不能编码成功就批量标通过。
- `full_red_raw.mp4` 及编码 verification、`run_manifest.json`、`completion.json`。`complete_io_pending_accuracy_review` 表示 IO 完成，角点精度待审。

先导 `video.execution=queued`：B0/C1 必须是 canonical Dispatcher 活跃 train 作业的真实子进程。**不能直接在 shell 启动 B0 然后期待它自行管理资源**。完整 epoch 到期后持久化视频意图和不可变 checkpoint，Dispatcher 负责视频资源准入；末轮 B0 也必须出完整视频，训练末轮完成不等于视频审核完成。C1 中间审核条件及失败一次受控重试由 `VideoQueue` 验证，不能手动改 ledger/队列 JSON。

生成视频时原始帧、模型预测、绘制图片三者身份分别记录；预测不是 GT。`annotation_stage=checks` 与 `final_inference` 分开标注用途，二者运行结果均存 outputs；真实训练标注、原始数据、三维模型资产才存 data。

## 7. 已核对的命令接口

所有命令先使用以下前置；bootstrap 默认隐藏 GPU，生产 GPU 分配只能由已准入 Dispatcher 子进程设置。

```bash
cd /data/zy/vln/a-test &&
test "$(pwd -P)" = /data/zy/vln/a-test &&
source runtime/bootstrap/project_env.sh &&
export PYTHONPATH=src OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
```

以下为**参数模板**，`RUN_ID/CHECKPOINT.pt/JOB_SPEC.json` 等须替换为主控制器核对过的真实路径；它们不是已完成训练的声明，不要在数据接受前执行。B0 的训练 argv 写入已有 Dispatcher spec 后由其启动；具体 spec 应基于当前实际 profile，不复制过期资源预估。

```bash
# D0/A0/B0 训练入口的参数形式（生产交由 Dispatcher 启动）
python scripts/action4_train.py --config configs/model/action4_pilot2000_v002.json --stage D0 --run-id PILOT_D0_RUN
python scripts/action4_train.py --config configs/model/action4_pilot2000_v002.json --stage A0 --run-id PILOT_A0_RUN --d0-gate outputs/action4/PILOT_D0_RUN/d0_gate.json
python scripts/action4_train.py --config configs/model/action4_pilot2000_v002.json --stage B0 --run-id PILOT_B0_RUN --d0-gate outputs/action4/PILOT_D0_RUN/d0_gate.json --init-weights models/action4/PILOT_A0_RUN/best.pt
# 经核对的 spec 入队接口；不在文档检查中执行
python scripts/action4_dispatch.py enqueue --spec configs/local/dispatch_specs/JOB_SPEC.json
# eval 全图推理；加 --gt-roi 会切换到独立诊断，不能作为全图成绩
python scripts/action4_eval.py --config configs/model/action4_pilot2000_v002.json --checkpoint models/action4/TRAIN_RUN/CHECKPOINT.pt --split eval --run-id EVAL_RUN
# 合格完整帧视频；生产视频由队列管理
python scripts/action4_video.py --config configs/model/action4_pilot2000_v002.json --checkpoint models/action4/TRAIN_RUN/CHECKPOINT.pt --frame-manifest data/action4_real/cache_v001/frames_manifest.json --run-id VIDEO_RUN
# 已有完整叠图重编码；不会预测角点
python scripts/action4_data.py encode --manifest data/action4_real/cache_v001/frames_manifest.json --frames-dir outputs/action4/VIDEO_RUN/drawn_frames --output outputs/action4/VIDEO_RUN/full_red_reencoded.mp4 --fps 2997/100 --crf 16 --threads 4
```

正式 test 先通过 `action4_eval.py --freeze-selection outputs/.../selection.json` 冻结选定 checkpoint/config/test manifest/source，再使用 `--split test --selection ... --run-id ...`；两次都需 `--config` 和 `--checkpoint`。不得用 test 来选择阈值，不能把 GT-ROI 或诊断 checkpoint 导出为正式 test。

新运行输出必须为空，不覆盖旧目录。编码接口缺帧、多帧、损坏、错尺寸、时间轴不符或已有输出会拒绝；详细 Python API 与边界见 `docs/action4_video_io.md`。路径必须经真实路径解析留在项目内，outputs-only 限制不能靠符号链接绕过。

## 8. 本次基本检验与证据界限

证据目录 `outputs/action4_model_interface_check_v001/`：

1. `check_interfaces.py` / `real_network_integration.json` / `.log`：真实 Action4Net（ResNet50/decoder256/head64），加载 SHA 匹配的 ImageNet 权重，在现有诊断样本 `pilot_v003_000000` 上完成实际 full 1024×768 + 2×ROI256 前向、联合损失、真实 RGB-only 两段推理、空 ROI 路径。未 mock 模型、未执行 optimizer、未写 checkpoint、未使用 GPU。
2. 实际原图↔ROI/热图仿射往返最大差 `5.6843e-14px`；由理想 dense target 解码回完整原图 bbox 最大差 `1.39649e-5px`。这是接口几何验证，理想输入不是模型预测。
3. 新 task heads 在默认阈值0.3 下检测0实例；为确实走到预测 ROI 分支，仅在内存中的测试配置将阈值设0，得到2实例及有限八点。该强制分支检查**没有选择或修改训练配置，也不构成精度证据**。
4. 正式加载现有 unaccepted 数据被拒，D0 component checkpoint 被正常视频准入拒绝。检查期间相关 `src` 文件哈希未变。
5. `regression.log`：28项既有测试全部通过、无 skip，范围为 dataset、training_eval、video_io。数据采样与真实 FFmpeg 编码实际执行；训练器恢复/DDP/完整视频控制流部分使用明确的 `TinyControlNet` 夹具，不能宣称本次已完成 ResNet 训练或 GPU DDP 验证。
6. `cli_help.json`：上述 train/eval/video/encode/dispatch enqueue 的真实 `--help` 输出和退出码；这证明参数存在，不证明模板中的未来输入已存在。

重跑本次检查（会重写同名检查报告，先由控制器选择新证据目录保留历史）应复用保存的脚本并更换输出目录。针对性回归命令：

```bash
PYTHONPATH=src:tests python -B -m unittest test_action4_dataset test_action4_training_eval test_action4_video_io -v
```

本次完成的是接口合同核对和基本 CPU 集成检验。没有将诊断图片认定为训练集，没有签发 D0 gate，没有完成2,000张先导训练，也没有生成合格的最终2,850帧角点视频。后续所有训练/效果/存储结论仍需要当前计划规定的真实运行与审核证据。


## 9. 固定评估输入身份（工作流审计增补）

固定 eval/test 清单的字节哈希必须与 checkpoint.identity.data_sha256 中对应 split 一致；该要求同时适用于 eval、decode 调参、test 选择冻结和正式 test，并在创建运行输出、加载 Dataset/模型之前执行。allow_diagnostic 不允许替换固定 split。同字节的项目内合法副本可用；缺失历史身份时明确拒绝，不得手工补写旧 checkpoint。冻结前允许 decode-only 调参，冻结后的 test 仍按已冻结推理合同核验。正式 test 要求 epoch_boundary is True 且 global_step 为严格正整数，布尔值、浮点值及 NaN 均不合格。

证据：outputs/action4_eval_test_dataflow_spec_review_v001/spec_review.json 与 outputs/action4_eval_test_dataflow_quality_review_v001/quality_review.json。此为输入和控制流防护，不构成模型精度或完整GPU训练证据。


## 10. 完整恢复的防护与实测范围

精确resume要求checkpoint内部resume_integrity v1，绑定每epoch更新预算、累计跳过数、设备与AMP；同时校验epoch/global_step、AdamW状态与scheduler/LR、sampler及逐rank CPU/CUDA RNG。当前提交的缺失别名可恢复；历史别名必须满足保存的SHA及来源，缺失时不能用较新权重替代。B0 epoch20 mandatory snapshots仅能从对应提交重放。部分epoch诊断保留last完整提交依赖的别名。rank0先完成修复并广播成功/错误，其他rank随后复核，二次验证错误同步到全部rank。待审核视频依旧在构建模型前阻止恢复训练。训练正常支持自行创建distributed group；外部预建world>1 group明确拒绝并保留，不宣称支持借用。

主控联合90项CPU/Gloo检查通过，证据outputs/action4_workflow_controller_check_v001/report.json。真实GPU6完整ResNet已执行A0组件初始化、B0连续两epoch与首epoch中断后恢复；指定model/AdamW/scheduler/scaler/RNG/ROI RNG/sampler/计数器等字段及更新日志（排除运行秒数）完全一致，证据outputs/action4_cuda_resume_check_v001/gpu/report.json。比较排除了整个top_checkpoints选择/排序元数据，不声称本GPU检查证明top3排序。相关别名防护由独立CPU/Gloo反例覆盖。

此为单卡、两次B0更新的组件续训证明，使用独立历史诊断图，不是正式数据训练、收敛、实机角点效果或多卡NCCL完整恢复证明。源码改变后历史证据只能作为对应版本记录。


## 2026-10-22 独立组件实际训练接口

正式源码职责：
- `src/model/action4_components.py`：旧权重无损拆分、严格角色加载、独立模型基础接口。
- `src/model/action4_component_experiments.py`：本轮冻结的GIoU/质量评分、A0八点关系候选及损失；仍处于开发对照，不代表已接受效果。
- `src/model/training/action4_component_train.py`：独立优化器、step/epoch恢复、随机状态、训练队列发布和背压。
- `src/model/training/action4_component_dispatch.py`：显式调度和受控评估执行器迁移；导入不启动进程。
- `src/data/action4_component_data.py`：一次性验证凭据、逐次文件身份检查、ROI数据与标签适配。
- `src/eval/action4_component_eval.py`：冻结评分函数及原队列执行器。
- `src/eval/action4_component_eval_cached.py`：当前缓存执行器，复用原评分函数；每实验独立worker保证覆盖。
- `src/eval/action4_component_pipeline.py`：RGB-only B0质量评分→完整框→A0自身ROI配置→修正八点映射回原图，GT不进入推理。
- `src/eval/action4_component_audit.py`：固定权重原始热图/修正点诊断。
- `scripts/action4_component_*.py`：薄入口；`tests/check_action4_component_*.py`：验证；配置在`configs/model/component_trials_v001`与`configs/eval/component_trials_v001`。

本轮运行状态、异步队列、完整恢复last.pt、待测快照、最佳开发权重best.pt统一在`outputs/action4_independent_trials_v001/<experiment>/`。这是有身份的实验产物目录；效果接受后才导出到`models/action4/`成为交付权重。源数据与标注不放outputs。首次启动和验证留在outputs的脚本是冻结历史快照，SOURCE_LOCATIONS.json指定后续正式入口，不继续从历史快照开发。

恢复必须使用`model.training.action4_component_train --config <原配置> --resume <该实验last.pt>`；不得用best.pt恢复optimizer，不得跨role/variant/config/source恢复。每个实验持有training.lock，重复启动被拒绝。恢复定位到epoch/cursor，重建相同顺序，恢复所有RNG与optimizer/scheduler；精度为FP32，无AMP scaler。

GPU1旧eval PID已受控退出，新PID以EVAL_REPLACEMENT.json为准，旧controller的-15记录是有意迁移而非训练失败。8个未完成旧领取归档后重新进入队列，不删已完成评分；CachedComponentData缓存RGB但每次仍检查源文件身份。30分钟监控读取实际PID和评估覆盖。

复用原则：Torchvision ResNet50已有直接复用；GIoU公式已与安装环境Torchvision官方loss及梯度对照。下一轮算子整理优先直接调用官方API，禁止热改当前冻结实验。quality候选为连续IoU监督的项目适配，不是完整GFL/DFL；不可将论文COCO结果当作Action4效果。原论文/官方来源与差异应随配置记录。


补充：换方法/增强模块时使用selected best权重做新实验初始化；`last.pt`只用于同一实验恢复。独立入口`src/model/training/action4_best_initialization.py`及`scripts/action4_prepare_component_finetune.py`已验证，不要将此区分解释为换方法后从last继续。A0辅助接口在`src/model/action4_a0_enhancement.py`，目前输出保留base logits并另外返回direction/foreground，尚无正式fusion或部署投票。其CPU构造异常值测试尚失败，原A0部署保持不变。


辅助输出规范：`auxiliary_vote_evidence`保留`base_corners_hm`与独立`auxiliary_corners_hm`，`numerically_valid`只表示几何求解不退化，不代表预测准确率；`support_fraction`也不等于标定可靠度。正式替换开关保持False。新分支尚待全量实际训练与可见/隐藏/完整框配对评估。B0质量排序对照模块仅生成分析报告，不热改正式decoder。


### 冻结式辅助分支实际训练 v002
核心训练`src/model/training/action4_auxiliary_train.py`，评估`src/eval/action4_auxiliary_eval.py`，调度`src/model/training/action4_auxiliary_dispatch.py`。配置`configs/model/action4_a0_additive_v002.json`，当前运行`outputs/action4_a0_additive_v001/train_run_v002`；v001为零更新失败历史，不恢复/重复启动。base_best.pt为初始化时固定硬链接，head快照独立保存；last记录head、optimizer、scheduler、RNG及epoch/cursor，base不更新。当前已实际推进optimizer step，不能将构造投票测试说成学到的精度。
GPU1当前原slot0-6七个cached评估加v002一个aux评估；原slot7有意退休。PID同时查原EVAL_REPLACEMENT和v002/DISPATCH。主六训练加aux训练共72CPU预算。


## 2026-10-23 best 自包含包与继续训练

当前包由 `model.action4_release_bundle` 导出；A0 和 B0 严格独立，A0 包含冻结 backbone。`load_bundle(path, role)` 校验角色/config identity 并 strict load；`warm_start(path, role)` 返回模型、A0 冻结 base、新 AdamW、包元数据。已用真实数据验证参数更新与保存/加载一致，未把检查更新当作新的科学训练结果。

完成的主训练器新增可选 `best_bundle` 配置分支，原始初始化分支保留。`configs/model/action4_a0_continue_best_v001.json` 与 `action4_b0_continue_best_v001.json` 使用封存 best、新实验目录和新优化器；分别每25/20步发布异步eval。它们只是后续入口，未启动额外80轮。精确故障恢复仍要求同一实验完整训练状态，不能将 last 的优化器嫁接到 best 参数。

运行新实验时，为训练与相应 eval 进程设置相同的 ACTION4_ATTEMPT_ID，先创建独占输出目录，使用新配置启动 `eval.action4_gdr_pose_eval_v002` / `eval.action4_online_b0_eval` 与对应训练器；B0训练需2 rank torchrun，A0单卡。原历史调度器带有专属历史run/preflight约束，不作为新实验通用入口。每次运行前核算实际GPU显存与CPU≤72，GPU0禁止。完整训练调度接口的历史依据和退出/恢复语义见上述源码；没有启动新实验的权限判断仍按用户当次请求。


## 在线RGB闭合后的模型入口（2026-10-23）

- 合成基线：configs/eval/action4_photometric_best_v001.json。
- 真实视频候选：configs/eval/action4_a0_online_rgb_best_v001.json。
- load_pipeline读取独立A0、B0包；RGB→B0完整框→2.25扩展ROI→冻结backbone/表面预测→ConvPnP位姿→canonical8投影→逆裁剪变换。GT框/位姿/可见标记不进入实际推理。
- A0训练入口model.training.action4_online_pose_dispatch，已有80轮运行已终态，不得直接再次启动相同输出。新实验复制配置并修改输出/父best/相应来源身份；默认不得拼接旧last优化器。
- eval.action4_staged_eval用于完整预测ROI开发/test；eval.action4_staged_video_parallel用于完整同帧demo视频。旧候选只作证据，不能因目录名含best就覆盖当前选择。
- 缓存开发173/378与完整RGB171/378数值不等价；实际RGB指标与视频独立报告。参数是否可加载、数值位姿有效、完整机身框正确、固定ID八点正确是四个不同验收层次。


## r069：部署一致的异步角点评估

新worker为eval.action4_image_grouped_pose_eval，队列仍采用原子snapshot请求、claim、done与best链接协议，兼容每25个optimizer step产生的pose快照。输入缓存按原图保留GPU裁剪/冻结网络得到的71通道maps、K、canonical；计分使用独立逆仿射、GT角点/visibility和匹配表。保持各图batch，不合并成32个ROI，FP/漏检进入分母。configs/eval/action4_exact_pose_replay_v001.json是封存step14000的真实队列回放配置，不能当作新训练配置；运行已终态。

新数据集或冻结backbone/surface/B0变化时必须重建缓存并绑定新identity。若surface/backbone参与训练更新，应对每个checkpoint执行当前RGB网络，不使用冻结maps加速。旧model.training.action4_online_pose_dispatch为历史运行入口，不会自动切换新worker；第二阶段编排须显式配置并核对，避免暗改已完成实验。

不加入“红色八点必须位于绿色检测框内”的硬规则。B0框提供ROI，A0保持固定CAD几何预测，框和角点的误差分别报告。


## 在线裁剪下的表面监督接口（隔离候选，未接正式训练）

src/data/action4_online_surface_targets.py复用prepare_pose_rgb的RGB裁剪、K变换与八点几何目标；读取已准入surface_v002的native_pixels与xyz_object_m，按本次ROI仿射得到连续heatmap坐标。物体坐标XYZ、region身份不随裁剪改变；visible/amodal mask从原掩码按新ROI重新采样，不搬用旧GT-ROI掩码。

src/model/action4_sparse_surface_losses.py在这些连续坐标上双线性读取预测XYZ/region logits，计算已有XYZ L1、region CE和mask BCE。采样实现支持严格确定性CUDA梯度；位置与标签不参与模型输入、不将不同物体表面的GT XYZ直接插值为新真值。整数网格下与原dense surface loss及梯度一致。

训练调用顺序应是prepare_online_surface_batch → model(RGB_features,K,canonical) → sparse_surface_losses(output.surface,targets) + geometry_losses(output,geometry)。geometry中的真实R/t、角点与visibility只进入损失。若surface要参与学习，不能继续用OnlinePoseRGB中冻结surface的设置；目前没有暗改该正式模型或启动新训练。

本接口当前绑定已准入的data/{split}/surface_v002_candidate及对应review，只在既有4个真实合成目标上完成实际投影/反向检查。第二阶段的新数据必须先生成并验收其表面sidecar、显式绑定新版本入口；不能把本轮检查当成新数据准入。验证见outputs/action4_online_surface_contract_v001/REPORT.json、REVIEW.json。若surface或backbone更新，必须用当前checkpoint重做RGB评估，不使用静态maps。


## r071：当前独立角色编排与动态RGB评估

当前调用链以action4_stage2_pair → action4_stage2_dispatch → action4_stage2_train/eval → action4_stage2_finalize为准；历史专属调度器保留作重放，不能混入新运行。
A0使用JointSurfaceRGB，联合表面预测与几何损失；独立B0使用原完整框训练损失。数据入口action4_stage2_data明确role/split、表面sidecar版本和manifest，在线surface_targets保证ROI仿射后标签同步。
所有新角色配置必须绑定parent_bundle/parent_best_step、独占output、validation_receipt、准入receipt、实际训练/评估数量和source_files。工程模式允许少量真实训练单位；full_training要求完整新数据准入、surface标签身份与至少80轮，不可拿integration gate代替。
A0每25step、B0每20step异步评估；初始和最终快照也评估。评估结束即删除已消费非best快照，选优临时别名按锁收敛，终态best_only_retention移除恢复last及冗余别名。后续改进从best权重初始化新优化器；同run故障恢复才用匹配模型/优化器/RNG完整状态。
本轮已验证DDP真实更新/恢复/全200开发eval/独立封装/16帧视频，模型和优化器恢复与连续运行逐值相同。工程短试验best仍为父权重，重复工程模型已退役并保留记录。正式视频入口校验80轮完成回执，工程模型不可冒充正式达标输出。


### 新geometry sidecar的显式衔接

若生成器将每实例NPZ保存为data/phrase2/{split}/geometry/{sample_id}_i{instance_id}.npz，训练配置surface_source.label_roots必须显式指向这些目录；同时提供admission（candidate_training_allowed须通过真实审核）、regions（冻结64区域中心）、manifest_sha256（train/eval/test与validation_receipt一致）。NPZ字段xyz_object_m、native_pixels、K、object_to_capture按当前在线监督接口读取；不将深度/XYZ/GT mask作为模型推理输入。
候选渲染结果尚未准入时，可用eval.action4_candidate_record_eval做独立train预览诊断，不借此生成正式训练gate。--gt-roi-baseline是显式oracle控制，仅供定因，不可混入B0框成绩或实际部署结果。


### 下一轮父best防误加载（2026-10-23T13:16:33.622885+08:00）

configs/model/action4_best_registry_v001.json固定A0onlineRGB14000与B0photometric3220的绝对项目内位置、文件SHA、大小、角色、架构、包config_id和来源training_config_id。full_training构造默认调用model.action4_best_registry；显式parent_registry可用于后续经过选择的新版本，不能将旧PnP13350或B0 2200直接混入。
configs/eval/action4_next_training_best_v001.json为绑定同一登记的固定模型推理入口。先核对字节身份再加载；实际CPU两角色构造及联合推理加载通过，七类错误身份反例拒绝。旧已完成训练的source_snapshot和配置不重写，不冒称按新guard重新训练。
