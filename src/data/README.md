# data：数据处理代码

存放原始数据读取、校验、清洗、格式转换、划分、预处理、数据集封装和派生数据生成。原始数据与派生文件位于根 data 或已授权资产位置，不放在这个包中。

## 输入、输出与依赖

- 外部共享原始数据只读复用；写入派生结果前检查真实路径和符号链接，禁止覆盖输入。
- 在项目自有 data 下按需区分 raw、processed、manifests；目录只是约定，不因名称自动获得外部写入权限。
- 保存来源、处理配置、代码身份、划分规则与种子。派生文件需要能够追溯其输入，不修改旧数据来迁就新结果。
- 数据加载层返回约定的数据结构，不导入 model 或 eval 业务；测评计分规则归 eval。
- 训练、验证、测试划分及重叠检查须显式记录，避免把已经参与选择或调参的样本描述成独立测试。
- 测评答案可由计分器读取，不能混入供模型推理使用的输入。

添加工具前阅读 [运行与实验规范](../../docs/EXPERIMENTS.md)，不为演示功能自动下载数据。

## Action 4：当前模块与执行范围

R2 已完成 DataAdapter、原生 Blender 基础后端和固定32张 pilot；R3 已补齐 Dataset、共享 RGB 采样、可扩展场景计划/分片/准入以及锁定源与权重获取。模型、训练、eval/test、视频绘制和反馈也已有实现，分别位于 `src/model/`、`src/model/training/`、`src/eval/`，不属于数据模块的运行依赖。代码存在不代表实验通过：两组实际4 ROI、各64更新的 CPU 诊断都未通过 D0 学习门槛，R3 图像真实性仍未验收，正式 S0、正式阶段训练、test 和最终模型整段视频没有执行。

| 模块 | 已存在的主要入口 | 职责 |
|---|---|---|
| `action4_adapter.py` | `DataAdapter.probe/extract/read_samples/encode_video`、像素中心矩阵与投影/分组 | 原始媒体、几何和数据 IO |
| `action4_dataset.py` | `Action4Dataset`、`prepare_full_images/prepare_rois`、`build_detection_targets/build_corner_targets`、`make_corner_heatmaps/is_ignored` | 清单准入、真实 RGB 采样、训练目标；推理复用采样 |
| `action4_render.py` | `prepare_asset/extract_reference_textures/diagnose/render_pilot` | 原 STL 的可追溯规范化、基础本体外观、旧固定32张试产 |
| `action4_synthesis.py` | `build_assets/create_plan/create_s1_plan/render_shard/validate_shard/assemble` | R3 场景/原生 Blender 渲染、mask、独立分片与训练准入 |
| `action4_sources.py` | `fetch_upstreams/fetch_weights` | 显式获取或审计锁定源码/离线预训练权重 |

完整训练与视频命令见 [训练执行说明](../model/training/README.md)；逐项来源、许可证与独立数学边界见 [复用映射](../../docs/action4_reuse_map.md)。

## DataAdapter 媒体与基础数据接口

正式实现是 `data.action4_adapter.DataAdapter`，可用 `from data import DataAdapter` 导入。该模块只负责数据、几何与视频 IO；导入时不下载、不启动进程、不加载模型。NumPy 按需用于几何处理，Pillow 用于实际解码并验证 PNG 完整性，FFmpeg/FFprobe 只在显式视频调用中运行。安装到项目隔离环境时使用 `python -m pip install -e '.[data]'`；FFmpeg/FFprobe 是需自行提供的外部命令，不由脚本安装。下列命令从仓库根执行；未安装 editable 包时可临时使用 `PYTHONPATH=src python scripts/action4_data.py ...`。输出目录或文件必须是项目根内的新路径；已有结果和原始输入不会被覆盖。

| 实际入口 | 行为、输出与限制 |
|---|---|
| `python scripts/action4_data.py probe --source /ABS/head_left_rgb_raw.mp4 --output outputs/probe_run001.json` | 解码检查全部帧，记录 SHA256、原整数 PTS、time_base、duration、真实帧数、尺寸、音轨和观察到的 CFR；不生成图片。 |
| `python scripts/action4_data.py profile --source /ABS/head_left_rgb_raw.mp4 --output data/action4_real/profile_run001 --samples 16` | 按解码帧号均匀抽取含首末帧的 PNG，写 `frames_manifest.json`；这是时间与抽样信息，语义、遮挡和光照结论需另行实际审阅。 |
| `python scripts/action4_data.py extract --source /ABS/head_left_rgb_raw.mp4 --output data/action4_real/selected_run001 --frames 0,427,2849` | 抽取明确帧号，保留 `frames/frame_00000000.png` 等源编号，保持原始编码尺寸，不自动旋转或缩放。 |
| `python scripts/action4_data.py extract --source /ABS/head_left_rgb_raw.mp4 --output data/action4_real/cache_v001 --all` | 显式生成完整 PNG 缓存；该路径对应当前训练/视频配置，会消耗大量空间，不是 profile 的默认操作。 |
| `python scripts/action4_data.py validate --manifest data/action4_synth/pilot_run001/manifest.jsonl` | 校验原生 PNG 路径/尺寸、0–2 实例、固定八角点、全框及遮挡字段；未做训练效果验证。 |
| `python scripts/action4_data.py split --manifest data/action4_synth/s0/manifest.jsonl --output data/action4_synth/s0/splits_run001 --seed 20261018 --ratios 0.8,0.1,0.1` | 按 group 和 background 来源的连通分组划分，写 train/eval/test JSONL、实际数量、种子及清单哈希；比例是目标，连通分组不可拆开凑数。 |
| `python scripts/action4_data.py encode --manifest data/action4_real/cache_v001/frames_manifest.json --frames-dir outputs/action4/B0_fullvideo_v001/drawn_frames --output outputs/action4/B0_fullvideo_v001/full_red_reencoded.mp4 --fps 2997/100` | 读取视频运行器已经绘制的完整连续 PNG，另存新 MP4，编码后再次解码核对 PTS、duration、尺寸和数量，写 `.mp4.verification.json`；本接口仅实现零起始 PTS、无音轨的严格 CFR。 |
| `python scripts/action4_data.py render-pilot --asset-dir data/DJI_Action4_construct/v001 --output data/action4_synth/pilot_run001 --blender /ABS/blender --count 32 --seed 20261018 --threads 2` | 显式调用现有 `action4_render.py` 后端，以独立 Blender 进程生成新 pilot；记录来源与代码哈希、命令、stdout/stderr，验证样本编号/数量及 PNG 数据合同。数据合同通过仍需视觉审阅，不自动接受为训练集。 |

Python API 对应 `DataAdapter(root).probe(...)`、`.extract(..., frame_ids=[...])` / `.extract(..., all_frames=True)`、`.profile(..., samples=16)`、`.validate_sample(record)`、`.read_samples(manifest)`、`.grouped_split(records, seed=..., ratios=(.8,.1,.1))`、`.validate_splits(records)`、`.write_split(...)` 和 `.encode_video(...)`。读取外部原始文件是允许的；输出路径解析符号链接后必须仍位于项目根。编码要求 `frame_00000000.png` 到末帧的集合与清单完全一致；缺帧、多帧、错误尺寸、变帧率、音轨、非零起始 PTS 或请求改变 FPS 均显式失败。不得把抽样清单伪装为完整视频输入。音轨复用和 VFR 编码尚未实现；绘框和 RGB 模型推理在 `eval.action4_video`，Dataset 在本包的独立模块，训练循环在 `model.training`。

`.render_pilot(asset_dir, output_dir, blender="blender", count=32, seed=20261018, threads=2)` 保留 R2 的真实后端委托：资产应已包含 `asset_manifest.json` 和 `mesh_v001.npz`；不自动准备或修改资产，不在宿主 Python 导入 `bpy`。它只接受新输出目录，固定传递 `--start 0 --stop 32`，并使用 Blender 的 `--python-exit-code 1`，再检查实际清单而不只看进程返回值。该历史配方仍恰为32张，Python API 和 CLI 均拒绝其他 count；扩大数量使用下面已经实现的 R3 `action4_synthesis.py plan/render-shard`，不改旧接口冒称正式生产。两条路径都不向已有失败目录断点追加，中断日志和已有文件保留。

### 几何及样本合同

所有二维点采用像素中心坐标（首像素中心为 0），`crop_affine` 的边界也在该坐标系表达，所以整图边界为 `[-0.5,-0.5,width-0.5,height-0.5]`；虚拟 ROI 可以越界，渲染/插值实现须使用同一约定并在图外补边。`letterbox_affine((W,H),(outW,outH))` 返回矩阵和真正整数 resize/padding，考虑 `(scale-1)/2` 中心偏移；`transform_points(points,M,inverse=True)` 做逆变换。`fixed_corners(min_xyz,max_xyz)` 顺序固定为下层 `000,100,110,010`、上层 `001,101,111,011`；`project_corners(corners,K,T_cam_obj)` 使用 OpenCV +Z 朝前的已去畸变针孔模型，返回投影与深度，所有 3D 长度和位移使用同一单位。它不会推断拍摄相机内参、真实视频的 6D 姿态或畸变参数。

每行样本使用 `schema_version="action4.v1"`、字符串 `sample_id/image/group_id/background_source_id/scene_family`、整数 `width/height` 和 `instances` 列表。`image` 必须是相对项目根的 PNG 路径。每个实例包含唯一 `instance_id`、`object_id=1`、固定序号的 `corners_px[8][2]`、其 min/max 形成的 `bbox_full_xyxy`、八个布尔 `corner_visible`；越界或不可见角点仍保留。八角点是虚拟包围盒顶点，`corner_visible` 只记录渲染器的可见性定义，不能作为删除隐藏角点监督的理由。可选 `bbox_visible_xyxy` 为掩码对应的像素边界：`[xmin-0.5,ymin-0.5,xmax+0.5,ymax+0.5]`；目标完全不可见时可省略或设 null。可选 `occlusion_inframe/truncation` 限于 `[0,1]`，定义应由渲染元数据明确；可添加渲染 K、T、mask 路径和来源证据。当前验证只校验字段和几何一致性，不据此确认真实感或误差。

当 in-frame amodal 面积为零，`occlusion_inframe` 的分母不存在，不能用 0 假装无遮挡；可以省略该字段，或同时写 `occlusion_inframe=null` 与 `occlusion_inframe_valid=false`。该状态不影响保留固定八角点和真正定义良好的 truncation。

同一 `group_id` 或 `background_source_id` 的样本及其传递关联只能在同一 split。同一个原视频不同帧不应伪装成彼此独立的背景来源；分割器在连通合并后不足三个独立分组时失败，不能宣称由单段视频背景形成独立盲测。已有 `split` 的样本拒绝重新划分，以保护冻结的 eval/test。新增训练版本应继续引用原来的 eval/test 清单，而不是合并后再次调用 split。

## Dataset 准入与共享 RGB 采样

Dataset 需要 Torch 训练依赖，在项目环境安装 `python -m pip install -e '.[data,train]'`；实际 CPU/CUDA wheel 应先按运行设备选择，本模块不会安装设备驱动或下载模型。`action4_dataset` 在模块导入时不导入 Torch，不依赖 model/eval；实例化和调用采样时才加载需要的依赖。

```python
from data.action4_dataset import Action4Dataset, prepare_full_images

dataset = Action4Dataset(
    root=".",
    manifest="data/action4_synth/s0_v001/train.jsonl",
    split="train",
)
items = dataset.collate_fn([dataset[0]])
batch, orig_to_full = prepare_full_images([item["image"] for item in items])
```

`Action4Dataset(root, manifest, allow_diagnostic=False, split=None)` 首先调用 DataAdapter 校验路径、实际 PNG、字段和几何，再验证 split 与训练准入。正式样本必须为 `status=training_accepted`，每行必须有严格小写64位十六进制 `rgb_sha256`；构建时读取实际 RGB 文件全部字节计算 SHA-256 并比对。诊断行可不提供哈希，但任何已提供的哈希都必须格式合法且匹配，即使 `allow_diagnostic=True` 也不会跳过。非 ignored 实例必须有实际 `bbox_visible_xyxy`。修改同路径 RGB 会使构建失败，不能靠清单路径或 PNG 尺寸相同维持原运行身份；构建后仍应保持数据只读，此实现不在每次 `__getitem__` 重新哈希。`action4_data.py validate` 只调用 DataAdapter，不等于完成这一层正式 Dataset 准入，也不等于外观审阅通过。

`__getitem__` 返回 `{"image": uint8 RGB CHW tensor, "record": dict}`，不会先缩放或把标签送进模型。`collate_fn` 返回 item 列表，以保留不同原始尺寸。统一 `is_ignored(instance)` 解释 `ignore=true`、`unobservable=true` 或 `status=unobservable`；训练与测评使用同一定义。`corner_visible=false` 是物理可见性记录，不是忽略整个实例或取消隐藏虚拟角点监督的理由。

| 函数 | 输入、输出与固定约定 |
|---|---|
| `prepare_full_images(images, size=(1024,768), device="cpu")` | 图像列表→归一化 `[B,3,H,W]` 与逐图 `orig_to_full`；严格使用 DataAdapter 的整数 resize/pad、`F.interpolate(align_corners=False)`，先补黑 RGB 再做 ImageNet mean/std |
| `prepare_rois(images, boxes, image_indices, roi_size=256, expansion=1.25, device="cpu")` | 原图域完整框→长边×1.25的虚拟正方形 ROI；不先裁掉图外区域，使用 `grid_sample(align_corners=False,padding_mode="zeros")`，返回 `[N,3,256,256]` 和逐 ROI `orig_to_roi`；N=0合法 |
| `build_detection_targets(records, orig_to_full, full_size=(1024,768), stride=4, device="cpu")` | 输出 `center/indices/instance_mask/offset/ltrb/negative_mask`；可见框中心用于检测锚点，完整八角投影包络用于 l/t/r/b。每距离除对应全图宽/高，offset允许负值；两有效实例同格拒绝，ignore负项按合法画内切片屏蔽且有效正项优先 |
| `make_corner_heatmaps(corners_hm, heatmap_size=(64,64))` | 输入已经处于热图像素中心域的 `[N,8,2]`→`[N,8,H,W]` 与 `[N,8] valid`；距离指数热图，每角独立 H/W 最大值归一化，不重复标准化坐标 |
| `build_corner_targets(records, image_indices, instance_indices, orig_to_roi, roi_size=256, heatmap_size=64, device="cpu")` | 用同一 ROI 矩阵和 half-pixel 降采样产生热图输入；拒绝 ignored ROI，图外原始几何标签保留。64×64离散中心域为`[0,63]²`，不能把无支持点 clamp 成边缘真值 |

训练和推理调用同一 RGB 采样函数；推理侧 `predict_images` 只接收图像。完整图到热图满足 `x_hm=(x_full+0.5)/4−0.5`。虚拟 ROI 图外补黑与隐藏角点是两个不同概念：隐藏但处于支持域的虚拟角点仍训练；无离散热图支持的点标为无效并保留原始几何记录。实际 RGB 坐标渐变检查与理想热图解码用于验证这些接口，不产生实拍角点精度结论。

## R3 场景生成、分片与准入

`action4_synthesis.py` 使用独立原生 Blender 进程，不导入 HCCEPose 或 BlenderProc。当前继续修正配方为 `configs/data/action4_synthesis_r3_v006.json`，原尺寸候选为 `configs/data/action4_synthesis_r3_native_v002.json`（3248×2464、64 samples，现文件选择 CPU）。二者都不是已经接受的正式 S0。`r3_v001` 等历史候选和旧32张 pilot 保留，用于追溯已观察到的问题；不能把换名或删旧图当成质量改善。

八个已实现场景家族为 `wall_rear/table_fastening/table_turn/near_double/self_forearm_occlusion/cloth_lift/wall_transition_truncation/hard_negative`。规范本体、手腕安装结构和遮挡者一起做锚姿态与小扰动；环境、光照与曝光按家族记录。R3 已使用登记来源的通用手资产、连接腕部/线缆几何、衣物构形、可见/无遮挡/扩展画布 mask、正深度投影和三角面 BVH 表面交叉检查。它们仍不能证明被摄者手部形状、真实受力、布料动力学、完整包含式穿模或全部遮挡归因正确；现有真实性拒绝状态保留。

以下命令在项目根、已激活的环境中运行；`ACTION4_BLENDER` 应指向服务器实际存在的 Blender 可执行文件。所有输出使用新目录。该示例仅计划128张并执行其中8张作候选审阅，不是正式生产或完整计划验收：

```bash
python scripts/action4_synthesis.py plan \
  --recipe configs/data/action4_synthesis_r3_v006.json \
  --output data/action4_synth/plans/review_next_v001.json \
  --count 128 --seed 20261019

python scripts/action4_synthesis.py render-shard \
  --plan data/action4_synth/plans/review_next_v001.json \
  --output data/action4_synth/review_next_v001_shard0 \
  --blender "$ACTION4_BLENDER" --shard-index 0 --num-shards 1 \
  --limit 8 --threads 2

python scripts/action4_synthesis.py validate \
  --output data/action4_synth/review_next_v001_shard0
```

`create_plan(recipe_path, output_path, count=10000, seed=20261019)` 冻结代码/资产哈希、场景参数、分组、split、实例数量和样本ID；正式10,000张的 train/eval/test 为8,000/1,000/1,000，0/1/2台配额为5%/15%/80%。少量预览的覆盖例外会写入 plan。`render_shard(..., shard_index, num_shards, limit, threads)` 按相同 plan 的索引分片，记录启动/执行身份、日志、RGB哈希、变换和 mask。`validate_shard` 检查实际样本、PNG、几何、mask及来源。有限 `--limit` 不代表完整 plan；不能用8张结果 assemble 128张计划。

原尺寸候选把上述 recipe 换为 `action4_synthesis_r3_native_v002.json` 并使用新 plan/输出；`native_resolution_pending=false` 只说明渲染配方用了原尺寸，仍要求 `native_resolution_review_status` 对应的真实审阅。修改一个布尔值不能代替验收。需要 GPU 时，应先依据实际 Blender 设备枚举在新配方中设置 `render.device=CUDA` 或 `OPTIX` 以及准确 `render.gpu_ids`，再创建 plan；设备不可用会失败，不静默回退。每个 shard 当前读取同一冻结配方中的设备列表，**没有按 shard 自动分配独占 GPU**，多进程可能竞争同一设备；当前也没有多卡渲染吞吐实测。操作系统进程可见性/调度如有额外设置，需单独记录并核对实际执行设备，不能宣称由该 CLI 自动完成。

`assemble(plan_path, shard_dirs, output_dir, acceptance_path=None)` 要求所有计划样本恰好一次、同一 plan/代码身份，固定已有 split，不重分 eval/test。完整候选可在不传 acceptance 时合并，但不会得到 `training_accepted`。正式验收文件必须包含 `decision="accepted"`、当前 `plan_sha256` 和 `reviewed_samples`；每个审阅项要有 `sample_id/reviewer/observations/decision/rgb_sha256`，覆盖计划中全部场景家族，数量达到配方的 `minimum_reviewed_candidates`。`criteria` 的六项 `anatomy_and_pose/mount_and_cable_attachment/cloth_contact/lighting_and_material/native_roi_detail/collision_and_masks` 必须全部真实通过；所有目标还需有 mask 内容哈希。缺失、陈旧或失败的验收会拒绝，不以修改状态串替代图像审阅。

验收后固定 S0 输出为 `data/action4_synth/s0_v001/{train,eval,test}.jsonl`。下面是接口形状，所列正式 plan、完整 shard 和真实验收文件需先产生，当前没有伪造这些输入：

```bash
python scripts/action4_synthesis.py assemble \
  --plan data/action4_synth/plans/s0_v001.json \
  --shards data/action4_synth/s0_v001_shard0 \
  --output data/action4_synth/s0_v001 \
  --acceptance outputs/action4_s0_review/acceptance.json
```

### S1 反馈与生成接口

`eval.action4_feedback` 先产生待复核事件和原图证据，只有真实确认、带 reviewer/evidence/observation 的事件才生成 `synthesis_requests.jsonl`。数据层 `create_s1_plan(recipe_path, requests_path, output_path, seed=20261020, max_count=2000)` 直接消费该文件，每行 `request_id/source_event_id/quota/split=train/recipe` 精确进入计划，参考帧和全部控制参数保留。预测角点从不作为训练标签，GT由渲染几何重新生成；接口/时间错误先修代码，不能用合成数据掩盖。

recipe 严格使用七项：`scene_family`、`view.wrist_delta_xyz_deg`（三个±8°以内的锚偏移）、`scale∈[0.9,1.1]`、`occluder="scene_default"`、`occluder_location="linked_geometry"`、`allowed_perturbations`、`reference_frame_ids`。扰动仅支持旋转幅度0–8°、位置各轴0–0.004m、光强0.9–1.1和曝光±0.15EV；光强/曝光替换对应基准扰动，不再次叠加随机增益。参考帧须落在已复核事件区间。任意2D黑块、未实现的遮挡者位置或额外字段会被拒绝。

```bash
python scripts/action4_synthesis.py plan-s1 \
  --recipe configs/local/action4_synthesis_s0_accepted.json \
  --requests outputs/action4/B0_feedback_reviewed_v001/synthesis_requests.jsonl \
  --output data/action4_synth/plans/s1_v001.json --seed 20261020
```

该正式 recipe 是后续验收形成的本机文件，当前不提供虚假 accepted 配方。随后同样 render/validate/assemble，S1 只能增加 train；固定 S0 eval/test 不动。默认 C1 要求6,000个不同S0与2,000个不同S1；确认事件不足造成未分配配额时保留欠额，不复制样本伪装独立数据。已经执行的跨模块 schema 检查只证明反馈请求能变成精确计划，没有真实模型失败或新增训练数据声明。

## 锁定来源与离线权重

`scripts/action4_sources.py upstream` / `weights` 分别调用 `fetch_upstreams` / `fetch_weights`；可加 `--audit-only` 禁止下载和检出，并用 `--output` 写新 JSON 审计文件。锁定文件为 `configs/data/upstream.lock.json` 和 `configs/model/pretrained.lock.json`。已有上游 origin/已跟踪改动/commit 不符时按 helper 的显式规则处理，不 reset 用户工作，未跟踪文件保留；官方预训练检查完整大小和 SHA-256，已有不匹配文件不覆盖。来源列表、精确 URL/commit、许可证、实际运行依赖与独立数学对照见 [复用映射](../../docs/action4_reuse_map.md)。模块导入不会发网络请求。

## 已执行的工程验证与结果边界

R2 的 `test_action4_adapter.py` 曾执行8项 CPU 检查，覆盖像素中心往返、固定角点/非法深度、路径/软链、背景传递关联、隐藏/越界角点、截断 PNG、同路径变更的原视频、真实 FFmpeg 7帧 `2997/100` 往返，以及 mock 后端启动合同。mock 只验证委托与失败处理，不声称真实 Blender 渲染；这些历史结果不等于 R3 正式数据或模型成绩。

R3 新增 `tests/test_action4_dataset.py` 的真实 RGB 采样、像素中心、忽略框边界、目标热图和正式 RGB 哈希检查；`test_action4_training_eval.py` 核查训练状态/恢复、最大匹配、test门禁、C1回放、调参与实际小视频 IO，并实际执行双进程 Gloo 控制网络检查。用 `PYTHONPATH=src python -m unittest discover -s tests -v` 重跑；Gloo 需要本机回环 socket 权限，缺权限应记录并通过正常权限流程执行，不能删测试或假报通过。以具体运行日志为准，不由源码推断所有环境都已通过。

实际正式宽度模型的两个64更新、4固定ROI CPU诊断分别保存在 `outputs/action4/D0_cpu_fixed4_r3_v001/` 与 `outputs/action4/D0_cpu_fixed4_lr1e3_r3_v001/`，学习门槛均失败；原始日志、热图分析与权重身份保留。当前数据准入与收敛问题都需要继续处理，不能把缺少GPU说成唯一障碍。只有正式图像验收、D0通过、后续训练/eval/test和完整原片逐帧核查实际完成后，才可登记相应训练或视频交付成绩。
