# Action 4 R3 数据合成实现与验收说明

本说明记录已实现、已实际运行的渲染部分。**本轮没有生成合格的正式 S0/S1 训练集。** 新版候选的投影、mask 和若干遮挡机制已通过工程校验，但通用手部姿态、腕部接缝、皮肤响应及衣物几何仍未通过原视频对照复核，因此没有扩大到 128 张，更没有启动 10,000 张正式生产。所有目录相对于仓库根目录；命令中的新输出目录不得覆盖已有版本。模型、训练、评测与视频命令由主执行文档及对应模块说明定义。

## 1. 已交付文件与真实运行范围

| 文件或目录 | 内容与状态 |
|---|---|
| `src/data/action4_synthesis.py` | 配方规划、S1 请求消费、Blender 场景、设备选择、shard 渲染、独立验证与准入；只有显式渲染命令才初始化 Blender |
| `scripts/action4_synthesis.py` | 薄 CLI；普通 Python 负责计划和验证，Blender 子进程负责图像与几何 |
| `configs/data/action4_synthesis_r3_v006.json` | 最新两张局部验证使用的零抖动锚点配方；812×616、32 RGB samples，仍是被拒收的候选 |
| `configs/data/action4_synthesis_r3_native_v002.json` | 最新原尺寸候选配方：3248×2464、64 RGB samples、1 mask sample、显式 CPU；尚未实际渲染原尺寸数据 |
| `configs/data/action4_synthesis_acceptance.template.json` | 默认 `pending_review`、六项标准均为 false、空复核列表的待填模板；不能直接批准训练 |
| `outputs/synthesis_s0_native_v003/plan10000.json` | 最终源码冻结后的真实 10,000 条计划，**仅计划，未渲染**；旧 v002 计划保留 |
| `data/DJI_Action4_construct/r3_v001/` | 同一规范机身 STL/NPZ、可编辑本体 `.blend`、源图贴图与来源清单 |
| `data/action4_scene_assets/hand_r3_v001/` | 选定的通用 CC0 手部资源及独立许可证/来源元数据 |
| `data/action4_synth/r3_v005_preview8/`、`r3_v006_preview8/` | 两轮各八个家族的真实候选、RGB、可见/扩大画布 amodal mask、场景工程、逐样本 JSONL 与源码快照；全部未批准训练 |
| `data/action4_synth/r3_v008_near_double/`、`r3_v008_wall_transition/` | 最后两张局部几何修正候选；各 2 目标、16 角点；未通过外观验收 |
| `outputs/synthesis_realism_qa_r3/` | v005、v006、v008 逐图拒收原因、图像哈希、实际 mask 数值和变更对应关系；包含 v007 穿模拒收日志索引 |
| `outputs/feedback_synthesis_schema_bridge_r3/` | 真实调用 `reviewed_requests → JSONL → create_s1_plan` 的纯 schema fixture；无渲染、无真实错误事件、无新增训练数据 |

`r3_v006_preview8` 共 14 个目标、112 个角点，独立投影最大绝对差约 `5.70e-5 px`。该轮左前臂场景实际遮挡达到 `0.634`；衣摆被解析几何抬起，但衣物仍是分离面片。最后的 v008 近双机场景后方目标总画内遮挡为 `0.593814`，墙前场景消除了画面中的悬空前臂端部；其截断仅约 `0.001169`，不能视作充分的边界截断覆盖。几何数值可从对应 `manifest.jsonl` 和 PNG 重新计算，不能据此推导真实视频检测正确率。

## 2. STL、Blender 工程与通用手资源

R3 不给机身做无证据的尺寸拉伸或拓扑美化。继续使用 v001 有记录的修复：原 STL 51,511 个三角面，删除 8 个极小退化面后为 51,503 面，并统一中心坐标；规范输出 SHA256 为 `a78598a95c504ffe61b0e91255e5d4818c6ca0a676102c812ce0d7349bad0795`。R3 新目录中的 STL 是该规范几何的逐字节副本，清单明确 `r3_geometry_change=none`。进一步修复必须先有新的测量、明确的坏面/法线/连接证据及前后比较，不能把“另存文件”写成新做了实体细节修复，也不能为匹配照片任意改变已固定的虚拟框角点。

本体工程为 `data/DJI_Action4_construct/r3_v001/Osmo_Action_4_appearance_r3_v001.blend`，已在独立 Blender 进程重新打开检查：26,556 个顶点、51,503 面，机身边界约为 `±[0.03525027, 0.01621189, 0.02227611] m`；源图屏幕和顶面标识贴图已打包，工程中保留资产清单文本。表面材质和反射参数是可编辑先验，没有实测 BRDF。贴图源像素、透视整形及角点定义沿用已有资产证据；这是已知观测外观的复用，不代表获得了无视角偏差的完整产品扫描。桌面、手、腕带、球形连接座、线缆及衣物留在场景 `.blend`，不并入机身 STL。

通用手来自 Blender 官方 Human Base Meshes v1.0.0。官方索引为 <https://www.blender.org/download/demo-files/>，开发文档为 <https://developer.blender.org/docs/features/asset_system/asset_bundles/human_base_meshes/>，官方原包为 <https://download.blender.org/demo/asset-bundles/human-base-meshes/human-base-meshes-bundle-v1.0.0.zip>。本次官方下载端点受限，实际从公开镜像 <https://mirrors.ocf.berkeley.edu/blender/demo/asset-bundles/human-base-meshes/human-base-meshes-bundle-v1.0.0.zip> 下载 34,586,814 bytes，SHA256 为 `46a912c0524072ac3b78c35d5d2471df7b8df102394a050ca8cd7184e3393648`。直接打开原 `.blend` 核实选定对象 `Hand  - Realistic` 的 author 为 Dan Ulrich、license 为 CC0；包内 README 与选定对象元数据一并保留。没有使用包中无关的 Rain Rig 或把其许可混用于本手资源。下载清单在 `data/action4_scene_assets/blender_human_base_v1_0_0/download_manifest.json`；选定资源清单在 `data/action4_scene_assets/hand_r3_v001/asset_manifest.json`。仅 1,095,476 bytes 的派生手工程 SHA256 为 `3ae5a5f75cbb29602ef89e0a93a75a4c6ceb9b55c046ef8683b24c8e4bef4529`。资源名中的 Realistic 是原资产名称，不是本轮图像已达到真实性标准的证明。

以下是已执行过的资产构建命令；原输出已经存在，重跑时必须指定新的版本目录：

```bash
runtime/blender_portable/blender-3.6.23-linux-x64/blender \
  --background --threads 2 --python-exit-code 1 \
  --python scripts/action4_synthesis.py -- build-assets \
  --source-asset-dir data/DJI_Action4_construct/v001 \
  --hand-bundle data/action4_scene_assets/blender_human_base_v1_0_0/human_base_meshes_bundle.blend \
  --output-asset-dir data/DJI_Action4_construct/r3_v001 \
  --output-hand-dir data/action4_scene_assets/hand_r3_v001
```

## 3. 八个场景家族、实际参数与改进依据

原片为 3248×2464、2,850 帧，SHA256 为 `016628eca2887220080d823f65d4f9c520ded71be363dbbbee95e175df7bd3e8`。使用的原图参考位于 `outputs/video_analysis_r2/`，参考帧编号不是合成 GT。所有腕部坐标、Euler XYZ 角、合成相机内参和光源参数都是观察照片后建立的设计先验；没有声称它们是原片的标定值或恢复出的真实 6D 位姿。本片已用于场景开发，不能再作为未接触过的盲测视频。

| 家族 | 参考帧 | 合成机制 | 权重先验 | 当前复核重点 |
|---|---|---|---:|---|
| `wall_rear` | 0、105 | 灰墙前双腕、背屏与顶面、短球形安装座、左右连线 | 10 | 闭合/放松手指、腕部接缝、机身与肤色亮度 |
| `table_fastening` | 540、960 | 格纹桌布、牛仔衣、双手靠近衣摆与纽扣 | 22 | 实际扣合接触；衣物不能是分离平板 |
| `table_turn` | 1080、1590、1770 | 同一腕部连接几何转到侧面/前面 | 20 | 角点语义保持固定，不按屏幕位置重排 |
| `near_double` | 1200、1290 | 双腕靠拢并有真实前后深度；两机独立实例 | 16 | 必须由另一机身造成可归因遮挡，不能用总遮挡率冒充 |
| `self_forearm_occlusion` | 1980 | 前臂与腕部连接，沿真实三维视线遮挡机身 | 16 | 用深度 mask 实测遮挡，v006 左目标为 63.4% |
| `cloth_lift` | 2550 | 衣摆向手指接触区域抬起；目前为解析形变 | 4 | 必须补连续衣物、实际接触与褶皱；当前仍拒收 |
| `wall_transition_truncation` | 2820、2849 | 双腕移到画面底边并转动，保留图外角点 | 7 | 量化截断；v008 仅 0.117%，不足以说明强截断已覆盖 |
| `hard_negative` | 1980、2550 | 移除相机，保留腕带、白色模块和黑色安装座 | 5 | 零机身标签、真实腕部干扰；外观仍需整体通过 |

权重为设计先验，未从原片逐帧统计出该频率。原尺寸方案对两只腕部采用相对于各自锚点的 `±8°` Euler 扰动和每轴 `±0.004 m` 位移；光强为基准的 `[0.9,1.1]`，曝光为基准 `±0.15 EV`。桌面基础相机位置 `(0,-0.33,0.90) m`、朝向 `(0,0.13,0.06) m`，面积光源 18 W、尺寸 1.1 m、曝光 −0.9 EV；灰墙基础相机 `(0,0.365,0.12) m`、朝向 `(0,0,0.01) m`，面积光源 8 W、尺寸 1 m、曝光 +0.2 EV。镜头 26 mm、传感器宽 32 mm。全部精确位置、旋转和参数均由 JSON 实际读取；改变文件后必须重建计划，不可让旧计划静默吃新设置。这里的瓦数是合成场景参数，不是原现场照度测量。

手与前臂、腕带、连接座和线缆共享腕部父级，转动时一同移动；线缆从机侧连接头沿前臂走线，不使用随机遮挡矩形。手资源先烘焙原 multires，再做通用弯曲。v005/v006 的切平指端被追到“仅移动中心线、未旋转截面”的形变缺陷；改为中心线弧长与截面一起弯曲后，curl=3.9 在实际 BVH 检查中出现手/机身交叉，记录失败，随后在 2.6–3.9 七个候选中取有余量的 3.4 做两张局部验证。该数值控制解析曲率，不是实测关节角，仍不能替代真实的手指关节约束。

近双机还进行了 27 个候选的便宜几何搜索：沿采集视线改变第二腕部前后距离 65/85/105 mm，屏幕横向 18/27/36 mm，纵向 −12/0/+12 mm，保持全连接 rig；先拒绝目标/手/前臂表面交叉、负深度和出画候选，再用 24×24 光线采样分别计数由另一机身挡住的射线。原搜索选中约 46.1% 的“另一机身遮挡”候选；修改手部后重新通过实际 RGB/mask 的几何关卡。搜索记录在 `outputs/synthesis_r3_v007/near_double_search.json`，曲率检查在同目录 `hand_curl_search.json`。光线搜索是粗筛，最终总遮挡仍从 PNG mask 重算；不能把二者混为一个指标。对实际 v008 PNG 再将后方机身画内 amodal 与前方机身的可见深度-ID mask 相交，得到 `1110/2425=45.7732%` 的另一机身遮挡，后方可见 985 px，其他几何/边界遮挡 330 px；报告为 `outputs/synthesis_r3_v008/near_double_occlusion_attribution.json`。该归因来自同一三维场景的真实 mask，不是从两张 RGB 随意估计。

当前衣物是有厚度、解析褶皱和缝线曲线的分离面片，没有连续缝合的衣身/袖子拓扑，也没有求解张力、摩擦或布料动力学。BVH 只检查目标与手、前臂、衣物、另一目标的三角面相交，不检测完整包含，不验证手与手之间的所有接触，也不证明人体软组织压力或布料力学平衡。后续应先依据参考衣物构建连续衣身、袖口、口袋盖、门襟和衣摆，再做受约束的接触/褶皱；不得把当前程序化面片改名为“真实牛仔衣”后绕过审核。

## 4. 数据与几何合同

每个 shard 保存 `launch.json`、`blender.log`、`synthesis_source_snapshot.py`、`manifest.jsonl`、`geometry_validation.json`，以及 `rgb/`、`masks/`、`scenes/`。前八个计划索引的场景会保存 `.blend`，内嵌配方和源样本参数，便于检查机身/腕部/线缆的真实连接。JSONL 保存 seed、group/split、背景源 ID、机身与采集相机矩阵、固定顺序八角点、像素框、可见性、资产版本、原 STL SHA、代码 SHA 和执行后的反馈控制。最终代码还保存 RGB 与两类 mask 的 SHA，并记录计划目标数和实际可见目标数。

坐标采用整数像素中心，Blender 投影转为 `x=uW−0.5`、`y=(1−v)H−0.5`，合成 K 的主点为 `((W−1)/2,(H−1)/2)`。`camera_to_world_m` 与 `object_to_world_m` 是米单位的 Blender 矩阵；独立验证使用 `diag(1,−1,−1,1) × inverse(camera_to_world) × object_to_world` 转成 OpenCV，再把平移转毫米后调用 `DataAdapter.project_corners`，与 Blender 自身投影比较，固定最大绝对差容差为 0.001 px。

可见 mask 来自完整 RGB 场景深度和实例 ID。完整 amodal 使用 3W×3H 画布、三倍传感器宽度，只保留该目标及目标子部件；中心 W×H 裁片是 in-frame amodal，扩大画布原点为 `[-W,-H]`。每次扩大渲染后用 `finally` 立即恢复原分辨率和传感器宽度，避免一个目标完全出画后污染下一目标投影。分别保存 `occlusion_inframe=1−A_visible/A_inframe_amodal` 与 `truncation=1−A_inframe_amodal/A_full_amodal`；隐藏的虚拟框角点仍是几何标签。完全出画目标有独立排除记录，不能静默凑少实际目标数。

低支持规则固定为可见比例小于 10%，或映射到 1024×768 full input 后的可见面积小于 16 px；面积使用 `DataAdapter.letterbox_affine` 的实际 `sx×sy`，不能仅用宽度缩放平方近似。边界抗锯齿允许的非嵌套误差为 `max(4,0.005×A_inframe_amodal)` 像素。正式 assemble 要求实际实例数与每条计划的配额一致；遇到穿模、出画或错误记录，保留拒收日志，修正后建立新版本，不静默换成无关家族填数。

## 5. 计划、分片、CPU/GPU 与执行顺序

首先进行小规模可视化复核；以下命令可重现当前未批准的候选机制，示例新路径在本轮没有执行：

```bash
python scripts/action4_synthesis.py plan \
  --recipe configs/data/action4_synthesis_r3_v006.json \
  --output outputs/synthesis_preview_review_v001/plan8.json \
  --count 8 --seed 20261019

python scripts/action4_synthesis.py render-shard \
  --plan outputs/synthesis_preview_review_v001/plan8.json \
  --output data/action4_synth/preview_review_v001/shard000 \
  --blender runtime/blender_portable/blender-3.6.23-linux-x64/blender \
  --shard-index 0 --num-shards 1 --limit 8 --threads 2

python scripts/action4_synthesis.py validate \
  --output data/action4_synth/preview_review_v001/shard000
```

`count<128` 明确标为 `preview_coverage_exception`，优先覆盖八个家族，不能冒充正式配额。正式计划先划分每四条相关视图所在的来源组，再分配 split 与 0/1/2 实例数。已经冻结的 `outputs/synthesis_s0_native_v003/plan10000.json` 有精确配额：

| split | 0 目标图 | 1 目标图 | 2 目标图 | 图像总数 |
|---|---:|---:|---:|---:|
| train | 400 | 1,200 | 6,400 | 8,000 |
| eval | 50 | 150 | 800 | 1,000 |
| test | 50 | 150 | 800 | 1,000 |

困难负例只消费上述 500 个零目标槽。八个家族在该具体计划中的数量为 `1002/2193/2028/1610/1559/399/709/500`，顺序与前表相同；这才是可审计的实际计划数量。固定模板/机身资产跨 split 复用，本轮不声称得到未知人体或未知背景资产的独立泛化测试；相关四视图和源背景组不跨 split。

当前机器实际枚举到 CPU，未发现 CUDA GPU。`render.device` 只接受 `CPU`、`CUDA`、`OPTIX`；GPU 配方必须另外给实际 Blender `gpu_ids` 列表，找不到后端或 ID 就失败，不静默改用 CPU。服务器可用以下只读命令枚举 CUDA 设备，再在一个新的本机配方中写入真实 ID、重新冻结计划：

```bash
runtime/blender_portable/blender-3.6.23-linux-x64/blender \
  --background --python-expr 'import bpy,json; p=bpy.context.preferences.addons["cycles"].preferences; p.compute_device_type="CUDA"; p.get_devices(); print(json.dumps([{ "id":d.id,"name":d.name,"type":d.type } for d in p.devices]))'
```

一个 Blender worker 会使用该计划 `gpu_ids` 中选中的设备集合；当前没有每个 shard 的独立 `--gpu-id` 覆盖。多个 CPU worker 可以使用同一计划的 `index % num_shards` 子集和独立输出目录；GPU 多 worker 的独立设备调度没有实现和实测，不能写成已有能力。计划、配方、资产、代码变动都会使原冻结身份失效。旧候选保留其源码快照，不覆盖旧输出来掩盖历史。

只有先修复外观并完成原尺寸候选复核后，才在新的本机配方、新计划、新 shard 目录中运行正式生产。示例采用一个 worker；以下命令本轮未执行，`configs/local/action4_synthesis_s0_reviewed.json` 是后续真实复核后建立的配置，不随包伪造：

```bash
python scripts/action4_synthesis.py plan \
  --recipe configs/local/action4_synthesis_s0_reviewed.json \
  --output outputs/synthesis_s0_reviewed_v001/plan10000.json \
  --count 10000 --seed 20261019

python scripts/action4_synthesis.py render-shard \
  --plan outputs/synthesis_s0_reviewed_v001/plan10000.json \
  --output data/action4_synth/s0_reviewed_v001/shard000 \
  --blender runtime/blender_portable/blender-3.6.23-linux-x64/blender \
  --shard-index 0 --num-shards 1 --threads 2

python scripts/action4_synthesis.py assemble \
  --plan outputs/synthesis_s0_reviewed_v001/plan10000.json \
  --shards data/action4_synth/s0_reviewed_v001/shard000 \
  --output data/action4_synth/s0_v001 \
  --acceptance outputs/action4_s0_review/acceptance.json
```

`--limit` 只取该 shard 的前 N 条；部分产物不能用于 assemble 完整计划，当前也不支持向原 shard 追加。失败 shard 保留日志，修正资产/配方需新版本；同一冻结版本的重试使用新目录，并在 assemble 中只选完整成功的那一个。若用独立 128 条预审计划检查原尺寸外观，后续生产计划仍有自己的 SHA，不能把旧计划的验收 JSON 直接用于新计划。最后 assemble 明确要求所有计划样本恰好出现一次，不重复、不缺失。

## 6. 精确准入格式与失败关闭行为

验收 JSON 先从 `configs/data/action4_synthesis_acceptance.template.json` 复制到新的复核目录。真实复核完成后才填写：顶层 `decision="accepted"` 和实际计划 SHA256；`criteria` 中 `anatomy_and_pose`、`mount_and_cable_attachment`、`cloth_contact`、`lighting_and_material`、`native_roi_detail`、`collision_and_masks` 六项必须各有证据且均为 true；`reviewed_samples` 至少含 128 个不同 sample_id，每项有如下字段：

```json
{
  "sample_id": "真实计划中的样本ID",
  "rgb_sha256": "实际复核PNG的SHA256",
  "reviewer": "实际复核者或复核流程身份",
  "observations": ["对手部、衣物接触、光照、原尺寸ROI和遮挡的实际观察"],
  "decision": "pending_review"
}
```

例子故意处于待复核状态，不能复制作为合格证据。正式 S0/S1 的计划分辨率和**每条实际记录**必须精确为 3248×2464，DataAdapter 同时验证真实 PNG 尺寸；不能只改 `native_resolution_pending=false` 就批准 812×616 图。复核集合必须覆盖该计划实际出现的所有场景家族；S1 可只出现有真实错误依据的家族，不要求编造其他家族事件。assemble 还逐项检查 shard 的 launch 计划 SHA、代码、资产绑定、图像和新 mask 字节身份、来源分组、每条实例数与配额、完整样本集合。已有旧候选缺少最终 mask SHA 字段且外观失败，不满足正式准入。

没有提供验收或没有通过时，数据状态只能是 `render_candidate_not_training_accepted`，`dataset_manifest.json.training_accepted=false`。这些文件便于开发接口，不是训练效果结果。正式训练 Dataset 默认拒绝它们，不应加“临时跳过外观检查”入口来完成名义上的实验。

## 7. 模型错误反馈到 S1 的可执行合同

`eval.action4_feedback.reviewed_requests` 只把真实复核为 `confirmed_failure` 的事件转成 `synthesis_requests.jsonl`。每行顶层包含 `request_id/source_event_id/quota/split/source_frame_start/source_frame_end`，`status` 必须为 `reviewed_request_pending_render`，`split` 必须为 train，`ground_truth` 必须为 `recompute_from_renderer_geometry`，`prediction_coordinates_used_as_training_labels` 必须为 false。生成器按每个事件的精确 quota 建立条目，保留事件与组关联；总量超过 `--max-count` 会拒绝，欠额不会自动重复单一事件凑成 2,000 张。

`recipe` 只接受以下七项，任何未实现字段都会拒绝：`scene_family` 为前述家族；`view={"wrist_delta_xyz_deg":[dx,dy,dz]}` 的每轴范围为 [−8,8]；`scale` 为 [0.9,1.1]，实际将采集相机到瞄准点的距离除以该值；`occluder="scene_default"` 和 `occluder_location="linked_geometry"` 只使用该家族既有连接遮挡；`allowed_perturbations` 必须恰有 `rotation_jitter_deg∈[0,8]`、`position_jitter_m` 三轴各在 [0,0.004]、`light_intensity_gain∈[0.9,1.1]`、`exposure_delta_ev∈[−0.15,0.15]`；`reference_frame_ids` 必须是 0–2849 内且落在真实复核事件区间的原帧号。光强/曝光按给定参数执行，view 是相对于该家族锚点的腕部旋转，三个扰动边界被实际消费，不是只写入 JSON 的备注。

```bash
python scripts/action4_synthesis.py plan-s1 \
  --recipe configs/local/action4_synthesis_s0_reviewed.json \
  --requests outputs/action4/B0_feedback_reviewed_v001/synthesis_requests.jsonl \
  --output outputs/synthesis_s1_v001/plan.json \
  --seed 20261020 --max-count 2000
```

S1 继续使用相同 `render-shard → validate → assemble --acceptance` 合同，在新的目录输出全部 train 条目，eval/test 不增样本。当前 S1 请求的实例数由家族决定：`hard_negative` 为 0，其余家族为 2；S0 有精确的一目标配额，但此 S1 合同尚未提供独立 `target_count` 控制。模型预测坐标不进入合成标签，角点与 mask 从新场景几何重新计算。需要单机请求、新遮挡对象或其他自由度时，先实现明确 schema、绑定与验证；当前入口拒绝任意黑色矩形、未实现的 cloth 位置或不受控旋转。

本轮只做过两种明确标识的工程检查：`outputs/synthesis_s1_contract_r3/` 与 `data/action4_synth/s1_contract_r3/` 是一张墙前图的参数执行 fixture，验证了旋转、零抖动、尺度、光强与曝光，不是实际模型错误；`outputs/feedback_synthesis_schema_bridge_r3/` 则直接跨模块调用生成精确 2 条计划，**没有渲染**。两者均非真实 S1 训练数据，不能拿来声称 C1 改善了视频。

## 8. 验证与下一项可执行改进

`tests/test_action4_synthesis.py` 的五个合同测试已通过：10,000 条精确 split/实例配额与来源组隔离、S1 精确 quota 与七项控制保留、拒收未实现/越界控制、拒收伪标签/未复核/非 train/区间外请求、拒收仅改布尔标志或混入小尺寸记录的正式准入。最后一项是明确的 assembly 边界 fixture，仅隔离前置几何依赖，不制造合格图像。真实 Blender 验证与单元测试分别记录；不存在“单元测试通过，所以图像真实”的结论。

下一轮先解决 QA 中已有明确证据的问题：连续牛仔衣和袖口/门襟几何、真实接触的手指关节姿态、连续腕部过渡、原尺寸材质和局部目标清晰度；对近双机继续区分另一机身遮挡与自身遮挡，对边界场景采到有实测意义的截断，再进行原图/渲染并排复核。所有改动保留来源、旧图、前后 mask 和碰撞记录。只有真实外观与覆盖通过后才扩大 128 张、正式 S0 和模型训练；本次在这一门槛前停止生产，并如实交付现有代码、资产、候选与失败证据。
