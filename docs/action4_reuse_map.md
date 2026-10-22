# Action 4 源码复用与独立实现映射

更新时间：2026-10-18，R3 实现交付。本文是 [上游锁定文件](../configs/data/upstream.lock.json) 与 [预训练权重锁定文件](../configs/model/pretrained.lock.json) 的阅读索引；锁定文件是版本权威，修改版本时同步更新本表。基础仓库为 `main@d88d61eede18382b788fa452054a19450bc5e19a`。R2 当时只完成 DataAdapter、原生 Blender 和固定32张 pilot；下面更新为 R3 已存在的 Dataset、网络、训练、测评和视频接口，不改变 R2 的历史执行范围，也不把实现完成等同于实验通过。

## 1. 上游身份、拉取地址与许可证原文

五个仓库的 HEAD 已与锁定文件逐项核对。上游的已跟踪源码没有本项目修改；未跟踪项由审计报告单列，不能仅由 HEAD 相等推断整个目录干净。R3 已实现 `src/data/action4_sources.py::fetch_upstreams/fetch_weights` 及薄入口 `scripts/action4_sources.py`。首次拉取到新临时目录并检出完整 SHA；已有副本先检查 origin 和已跟踪改动，再按锁定版本 fetch/checkout。已跟踪改动会使操作失败并保留工作树，未跟踪文件保留且计数；Git 本身会拒绝覆盖有冲突的未跟踪文件。不会执行 reset、clean 或浮动分支的直接 pull。`upstream/` 实际代码按仓库规则忽略，代码提交保留无凭据锁定文件和本索引。

| 来源 | 首次拉取 URL → 项目内目录 | 锁定 commit | 本地许可证原文 |
|---|---|---|---|
| HCCEPose | `https://github.com/WangYuLin-SEU/HCCEPose.git` → `upstream/HCCEPose` | `f75cc4d5db2fc298fc7b853f8c75273757872382` | [upstream/HCCEPose/LICENSE](../upstream/HCCEPose/LICENSE)，文件标题为 GNU LGPL v2.1；不据此把所有内嵌第三方包都改标为此许可 |
| BoxDreamer | `https://github.com/zju3dv/BoxDreamer.git` → `upstream/BoxDreamer` | `106f3c16c746770cbad80720ce4a557eb31f30f5` | [upstream/BoxDreamer/LICENSE](../upstream/BoxDreamer/LICENSE)，**Project Registration License (PRL) v1.0**，不是 MIT/Apache |
| CenterNet | `https://github.com/xingyizhou/CenterNet.git` → `upstream/CenterNet` | `4c50fd3a46bdf63dbf2082c5cbb3458d39579e6c` | [upstream/CenterNet/LICENSE](../upstream/CenterNet/LICENSE)，MIT |
| BlenderProc | `https://github.com/DLR-RM/BlenderProc.git` → `upstream/BlenderProc` | `76bd8b82ee19489843578ff045298642197bf883` | [upstream/BlenderProc/LICENSE](../upstream/BlenderProc/LICENSE)，文件标题为 GNU GPL v3 |
| bop_toolkit | `https://github.com/thodan/bop_toolkit.git` → `upstream/bop_toolkit` | `cea62d651c7e395b2e1962b9749e4e89693c6ac4` | [upstream/bop_toolkit/LICENSE](../upstream/bop_toolkit/LICENSE)，MIT |

HCCEPose 的 `blenderproc.zip` SHA-256 为 `f434deb17674770655b1b6b62eb413b3a03f4537b2b4686bc79d9349683633d7`，`bop_toolkit.zip` 为 `9e1db8ec86ddf7c821201eeb3421e3c5c85c20f8dfc8f24d2d03029f431212d6`。本轮仅打开归档作源码对照，没有把这些包加入当前渲染进程。已定位的内嵌许可成员是 `blenderproc.zip!blenderproc/bop_toolkit_lib/LICENSE` 与 `bop_toolkit.zip!bop_toolkit/LICENSE`，两者文件标题均为 MIT；前一个归档没有 BlenderProc 根 LICENSE，不能用其中 BOP 的许可证代替整个归档的许可来源。当前不据这些归档执行额外安装或源码分发。

在项目根、已激活的项目环境中执行以下入口。缺少的源/权重只在显式 fetch 时下载；已有内容可以只审计。报告必须使用新路径。

```bash
python scripts/action4_sources.py upstream --output outputs/action4_upstream_fetch_v001.json
python scripts/action4_sources.py weights --output outputs/action4_weights_fetch_v001.json
python scripts/action4_sources.py upstream --audit-only --output outputs/action4_upstream_audit_v001.json
python scripts/action4_sources.py weights --audit-only --output outputs/action4_weights_audit_v001.json
```

真正加载的初始模型权重只有官方 [ResNet50 IMAGENET1K_V2](https://download.pytorch.org/models/resnet50-11ad3fa6.pth)，目标为 `models/pretrained/resnet50-11ad3fa6.pth`，102,540,417 bytes，SHA-256 为 `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`。`fetch_weights` 同时检查大小和完整哈希；失败下载保留为临时文件，已有不匹配权重不会覆盖。`Action4Net.load_backbone_weights` 再次核验，严格加载空间主干，只移除分类器的 `fc.weight/fc.bias`。这是 ImageNet 分类预训练，不是已学会 Action4 的 checkpoint。当前没有加载 HCCEPose、BoxDreamer、COCO 人体关键点或 CenterNet 整体模型权重。

## 2. 复用状态定义

- **直接运行依赖**：实际调用已安装库或发行二进制，例如 Torchvision ResNet50、Torch 张量与优化器、原生 Blender `bpy`、FFmpeg/FFprobe。依赖与版本另行锁定。
- **现有独立实现，数学对照**：本项目代码已存在；所列上游用于核对公式、字段或边界，没有导入或复制其完整模块。源码中的来源声明保留，不能把相同数学公式写成已经运行上游系统。
- **仅作对照 / 当前不启用**：检查来源以理解假设或不适用行为；未运行其功能。完整 BOP 导出等未实现内容仍单独标明。

R3 当前可运行数据实现包括 [DataAdapter](../src/data/action4_adapter.py)、[Dataset/共享采样](../src/data/action4_dataset.py)、[原生基础后端](../src/data/action4_render.py) 和 [可扩展场景生成器](../src/data/action4_synthesis.py)。网络、损失、解码、D0/A0/B0/C1 训练、eval/test、调参、视频和反馈均已有实现；准确入口见 [训练执行说明](../src/model/training/README.md)。这些模块不把 HCCEPose、BoxDreamer 或 BlenderProc 的源码树加入运行导入路径。HCCEPose 增强标注继续暂停。

## 3. 逐项源码与适配映射

下表的源码路径均相对于对应上游目录；目标路径相对于本项目根。表中目标函数均已存在，另有明确标注的未来 BOP 互操作内容除外。

| 来源文件 / 函数 | 本项目目标 | 精确适配与验证边界 | 当前状态 |
|---|---|---|---|
| HCCEPose `s1_p1_obj_rename_center.py` 的 `__main__` 中 AABB 中心平移；`modify_ply_texture_filename` | `src/data/action4_render.py::prepare_asset` | 当前从只读二进制 STL 解析数值，只平移原 AABB 中心、保留轴向与毫米尺度，记录源三角面编号和退化阈值，导出新 STL/NPZ 与 source_to_object。原 PLY/贴图重命名函数没有调用；不硬编码演示路径、不覆盖输入。 | 上游仅作对照；几何准备为现有独立实现 |
| HCCEPose `s1_p3_obj_infos.py` 的 `__main__`：min/max、边长、直径、对称信息汇总 | `src/data/action4_render.py::prepare_asset`；`src/data/action4_adapter.py::fixed_corners` | 当前记录 bounds/dimensions、固定 0–7 角点和资产哈希。尚未生成该上游完整的 `models_info.json` 或测算其直径/对称性；不得凭机身近似矩形添加旋转对称先验。正式 BOP 互操作如后续启用，再单独补齐其字段。 | 上游仅作对照；当前 AABB/角点为独立实现 |
| HCCEPose `s2_p1_gen_pbr_data.py` 的场景主流程、`sample_pose_func`；`s2_p1_gen_pbr_data.sh` | R2 `src/data/action4_render.py::render_pilot`；R3 `src/data/action4_synthesis.py::create_plan/render_shard/_render_one` | 原生 bpy 独立构造 0/1/2 台腕部相机、连接结构及八类有原帧依据的场景；保存 K、物体/相机变换和 mask。没有运行上游默认 K、随机掉落或无约束 SO(3) 采样。旧 `DataAdapter.render_pilot` 仍只支持32张；任意计划数量、S0配额和分片由 R3 新接口承担，不能用修改旧 count 代替。 | 仅作渲染流程对照；两代独立后端均保留 |
| HCCEPose `blenderproc.zip!blenderproc/python/writer/BopWriterUtility.py::_build_trimesh_objects/write_bop/calc_gt_masks/calc_gt_info` | R2 `src/data/action4_render.py::_mask_nodes/render_pilot`；R3 `src/data/action4_synthesis.py::_render_one/validate_shard`；未来 BOP 导出另行适配 | 已识别同 category 网格缓存、局部网格/对象 scale 与深度容差假设。当前原生后端独立渲染实例 mask 和扩展无遮挡 mask，并分别记录遮挡/出画；未调用此 writer，也不声称 native JSONL 已是完整 BOP。若后续启用，核对局部毫米＋对象 scale=0.001、烘焙几何、实例多材质和接触遮挡容差，不直接喂当前世界米制矩阵。 | 仅作 BOP 边界对照，当前不启用 |
| BlenderProc `blenderproc/python/loader/BlendLoader.py::load_blend` | `src/data/action4_render.py::_core/diagnose`；`src/data/action4_synthesis.py::build_assets/_anatomical_hand` | `_core` 从规范 NPZ 与证据贴图创建本体；R3 通过原生 `bpy.data.libraries.load` 加载已登记的 CC0 通用手资产，保留来源与派生记录，没有调用 BlenderProc loader。通用手不是原片被摄者扫描；保存 `.blend` 也不证明外观验收通过。 | 仅作对照；实际使用原生 bpy |
| BlenderProc `blenderproc/python/object/ObjectPoseSampler.py::sample_poses` | `src/data/action4_synthesis.py::_wrist_rig/_collision_pairs/_render_one` | R3 已实现锚姿态、小扰动和已求值三角面 BVH 表面交叉检查，未调用此上游 sampler。该检查不能证明软组织受力、完整包含式穿模或衣料平衡；手部端面、前臂接缝、衣物与接触仍须逐图审阅。 | 仅作采样边界对照；独立检查已实现，真实性未通过 |
| BlenderProc `blenderproc/python/writer/BopWriterUtility.py::write_bop` 与 `_BopWriterUtility.get_frame_gt/get_frame_camera/calc_gt_masks/calc_gt_info` | 当前数据层 mask/几何核查；未来 BOP 导出适配 | 用于对照官方与 HCCE 打包版的区别，不能用官方快照覆盖作者归档而仍宣称同一实现。当前不导入两套 writer，不混用其单位参数或依赖。 | 仅作对照 |
| bop_toolkit `bop_toolkit_lib/misc.py::project_pts/transform_pts_Rt/calc_2d_bbox` | `src/data/action4_adapter.py::project_corners/validate_sample` | 当前独立 NumPy 针孔投影要求角点和 t 同单位、OpenCV +Z 朝前、K 有效、R 为 proper rotation、深度为正；完整框取八角投影 min/max，不裁成图内框。公式是对照依据，未导入整个 BOP 环境，当前也不使用其位姿指标充当本任务的八角点指标。 | 现有独立实现，BOP 数学仅作对照 |
| BoxDreamer `src/datasets/utils/base/camera_utils.py::reproj_pytorch` | `DataAdapter.project_corners`；`src/data/action4_dataset.py::prepare_full_images/prepare_rois` | 3D 针孔投影仍由 NumPy 独立实现并核查单位/刚体/正深度；Dataset 的 Torch RGB 双线性采样使用同一 DataAdapter 二维像素中心矩阵，并非调用此上游投影函数。全图整数 resize 与虚拟 ROI 都有实际 RGB 坐标渐变采样检查。 | 现有独立实现，投影数学对照 |
| BoxDreamer `src/datasets/utils/base/bbox_utils.py::make_bbox_features` | `src/data/action4_dataset.py::make_corner_heatmaps/build_corner_targets` | 独立实现距离指数数学 `exp(-||u-p||/a)`，`a=max((||p-mean(p)||/10)^2,1)`，每 ROI/每角点只按自身 H/W 最大值归一化。坐标已经处于热图域；64×64 有效中心域为 `[0,63]²`，无支持的越界点不 clamp。保留隐藏虚拟角点监督，不带参考 RGB/参考角点，不采用全批次 amax 或重复 `[-1,1]` 变换。 | 现有独立实现，数学对照 |
| BoxDreamer `src/models/utils/box_utils.py::recover_bb8_corners` | `src/model/action4_decode.py::decode_corners` | 每固定通道 top20 像素位置均值，20 不是实例数；可输入每 ROI 热图到原图矩阵。无权值加权、通道重排或 GT 修正，离散解码不作为可微坐标监督。已运行实际标签理想热图解码与边缘偏差检查。 | 现有独立实现，数学对照 |
| BoxDreamer `src/models/modules/encoder/resnet.py::ResNetWrapper.get_activation/predict` | `src/model/action4_network.py::Action4Net` | 未复用该 detach/no_grad 封装。实际直接调用 Torchvision ResNet50，三个共享反卷积层和四个任务头由本项目构造；全图/ROI 共用参数，主干梯度和参数变化已有实际检查。 | 上游仅作边界对照；网络已实现 |
| Torchvision 0.23 `torchvision.models.resnet50` / `ResNet50_Weights.IMAGENET1K_V2` | `Action4Net.__init__/load_backbone_weights` | 真正运行的第三方主干及离线分类预训练。移除分类池化/FC，空间特征参与优化；不自动加入分类中心裁剪，不把 V2 解释为本项目第二阶段模型。 | 直接运行依赖；权重身份已锁定 |
| CenterNet `src/lib/models/losses.py::_neg_loss` | `src/model/action4_losses.py::focal_center_loss/detection_losses` | 独立实现中心 focal：logits/FP32 稳定计算，每图按 max(有效锚点数,1) 归一化再平均；ignore 画内负项屏蔽，有效正项优先，空图保留背景监督。offset/ltrb 和图像级归一化遵循本项目合同；未复制上游旧训练器。 | 现有独立实现，数学对照 |
| CenterNet `src/lib/models/decode.py::_nms/_topk/ctdet_decode`；`src/lib/models/utils.py::_gather_feat/_transpose_and_gather_feat` | `src/model/action4_decode.py::decode_detections`；`src/model/action4_losses.py::gather_at_indices` | 独立实现3×3峰值和 gather；标准 center+wh 改为可见中心＋offset＋完整框归一化 l/t/r/b。默认最多20候选，再按分数及同时满足 IoU/中心距离的严格重复规则保留0/1/2台，不按 IoU 单一条件删除近邻。像素中心坐标由共享矩阵还原。 | 现有独立实现，数学对照 |
| CenterNet `src/lib/utils/image.py::gaussian_radius/gaussian2D/draw_umich_gaussian` | `src/data/action4_dataset.py::_gaussian_radius/build_detection_targets` | 独立实现对应重叠半径方程与 Gaussian；有效整数锚点标签峰精确为1，两有效实例同格直接拒绝。忽略框先裁成合法画内切片；真实 resize/pad 后再降至热图，不使用另一套 crop/affine，也不加入无证据镜像/强颜色增强。 | 现有独立实现，数学对照 |

## 4. 实际调用、数据准入与当前实验状态

FFmpeg/FFprobe 通过 DataAdapter 的 `probe/extract/encode_video` 显式子进程调用，接口依据 [FFmpeg](https://ffmpeg.org/ffmpeg.html) 与 [FFprobe](https://ffmpeg.org/ffprobe.html) 文档；这属于调用发行二进制，不是五个源码仓库之一。当前回封装合同覆盖零起点、无音轨、严格 CFR；本片为 `2997/100` FPS、`1/11988` time_base、每步400。R3 `src/eval/action4_video.py::run_video` 已实现模型 RGB 推理、原始浮点角点、固定红线与该回封装调用，并通过小型真实 FFmpeg 工程检查；正式模型的整段原片结果尚未生成。

`Action4Dataset` 已在构建时验证 `training_accepted` 行必需的 `rgb_sha256`：严格小写64位十六进制，逐文件计算完整 SHA-256；任何诊断行只要提供哈希，也必须匹配。该检查与字段/几何校验分开，防止同路径 RGB 被替换后仍沿用旧清单。构建后输入仍须只读；代码没有声称每次 `__getitem__` 都重新哈希。训练、eval 与 test 使用同一 Dataset/采样函数，计分单独读取标签；视频预测只接收 RGB。

当前继续修正用 `configs/data/action4_synthesis_r3_v006.json`；原尺寸候选用 `configs/data/action4_synthesis_r3_native_v002.json`。两者仍是待验收配方。`create_plan` 冻结源码/资产/参数，`render_shard` 执行独立分片，`assemble` 要求计划全部样本及绑定原 RGB 哈希的真实审阅。当前每个进程读取同一 plan 中的 `render.device/gpu_ids`，没有按 shard 自动分配独占 GPU；CUDA/OPTIX 后端或 ID 不可用会明确失败，不回落 CPU。不要把进程数量写成已经完成的多卡吞吐验证。S1 通过 `create_s1_plan` 直接消费 `action4_feedback.py` 的有界 `synthesis_requests.jsonl`，只生成 train，由渲染器重新计算真值；自动候选不是确认错误或伪标签。

两组实际 CPU 组件实验各使用4个固定 ROI、64次更新，分别保留 [第一组结果](../outputs/action4/D0_cpu_fixed4_r3_v001/completion.json) 与 [学习率对照结果](../outputs/action4/D0_cpu_fixed4_lr1e3_r3_v001/completion.json)。它们验证梯度和更新确实发生，但 D0 学习门槛未通过，不能用 loss 下降替代角点收敛。R3 场景真实性同样未通过，正式 S0、D0/A0/B0/C1、正式 test 和最终模型整段视频均未执行。工程控制网络的训练/恢复/DDP/视频测试只证明对应接口行为，不能写入正式模型成绩。

每处今后实际复制的源码须保留原版权和 LICENSE，记录 commit、原函数、目标函数与改动；不能凭重命名把复制写成独立实现。当前数学来源声明见模型/数据模块头部和 `src/model/THIRD_PARTY_NOTICES.md`，BoxDreamer 的 PRL 原文仍单独保留，不套用 MIT/Apache。当前没有替用户提交注册或外部表单，没有安装 BoxDreamer 全重建环境、CenterNet 旧 DCNv2，也没有混入 HCCE 归档或官方 BlenderProc 的运行时导入路径。
