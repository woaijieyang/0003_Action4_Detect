# Action4角点复现增补：执行结论（进行中）

对应计划：[技术文档](2026-10-19-1-plan-Action4角点复现增补.md)。记录时区 UTC+08:00。

**当前摘要（r069）：第一阶段数据、训练、评估、视频、封存与发布已完成；真实精度未通过。最新本轮修复已实现部署一致的分组缓存异步评估，372匹配目标逐点误差与完整RGB完全一致；不采用真实效果退步的迭代PnP，不采用用户已撤回的红框硬包含约束。第二阶段尚未启动。**

下文按日期保留历史证据；早期“S01/未完成/当前运行”等描述只代表当时，不能作为新的启动授权或当前状态。当前入口见[执行索引](../action4_execution_index.md)和action4_active.json。

## 2026-10-19T16:14:27.237185+08:00 S01：恢复

- 实际服务器 xinglong-a6000；基线 d88d61eede18382b788fa452054a19450bc5e19a，应用前工作区干净。
- 两ZIP加已有同哈希原片恢复863项资产；原补丁应用后57文件逐项哈希通过。日志见 `outputs/action4_server_bootstrap_v001/artifact_precheck.json`、`artifact_restore.json`。
- R4、demo、要求图保存于 `data/DJI_Action4_construct/foundation_r4/` 和 `data/reference/`。原视频与原STL原位保留。
- 正式计划与审核稿差异仅为证据链接重定位和本次批准状态说明，回执保存两版SHA。
- 正在建立项目内Python/CUDA/Blender/FFmpeg环境。原38项工程测试尚未复跑；任何历史通过均不代替本次结果。

## 未完成的验收

S01环境与测试、S02数据覆盖、S03 R4接入及U01–U07、S04正式S0、S05 D0/A0、S06 B0、S07完整视频、条件S08及S09最终验收均未完成。不得从恢复成功推断模型效果。

## 2026-10-19T16:49:31.850936+08:00 环境、增量实现与早期诊断

- S01：项目内Python3.12.3/torch2.8.0+cu128/torchvision0.23.0+cu128已安装，依赖检查通过，完整解析版本见 `outputs/action4_server_bootstrap_v001/environment_freeze.txt`。Blender3.6.23及FFmpeg/FFprobe8.1构建已按计划SHA核验。FFmpeg服务器直连慢，采用本地下载同哈希工具包传输的既定替代路线。
- 原始恢复基线38项测试全部通过（`baseline_tests.log`），包含真实FFmpeg逐帧时间轴及CPU/Gloo控制检查；不冒称CUDA/NCCL训练验证。
- GPU4–7分别验证UUID、单设备映射和小张量运算（`cuda_identity_check.json`）。首次UUID断言因PyTorch字符串缺少GPU-前缀失败，检查实际返回后规范化前缀；不曾将不符身份设备用于训练。
- U01已修补输入/配置/资产/图像路径边界，新增6个反例。第一次新增测试发现synthesis先strict解析再验证边界，已修正检查顺序；原失败日志保留，修正后44项全部通过。
- U02：现有R4通过独立重开，新增loader直接加载其机身和附着贴图；核对26556顶点、51503三角面、固定角点、4张packed图片哈希。实际双机渲染不含原review灯/背景。此项证明接入和几何，完整场景真实性仍待改进。
- U03：S0左右单机按split均分、S1明确target_presence，旧入口兼容；新增3项检查及原合成检查共8项通过。尚未把单机真实渲染检查当完成。
- U05：显式单worker物理UUID和entry-start/count已实现；分片并集、互斥参数、禁用未授权GPU、清除DDP环境等新增4项检查。整套工程检查达到51项通过（`u02_u03_u05_tests.log`）。
- R4场景在GPU5 CUDA与GPU6 OptiX各渲染同种子预览1张，均通过独立PNG/mask/投影检查；最大投影绝对差3.15084e-5原图像素。图像已查看：机身贴图正确，前臂接缝/手指形态仍明显人工；未获得正式数据准入。候选数据位于 `data/action4_synth/server_r4_preview_cuda_v001`、`server_r4_preview_optix_v001`。尚未据一次异卡预览确定最终最快后端。

### 早期32 ROI GPU诊断（失败结果保留）

保留原CPU helper，抽取其逻辑到 `src/model/training/action4_diagnostic.py` 和薄入口 `scripts/action4_diagnostic.py`。首次选择原始manifest缺可见bbox，数据检查在GPU训练前拒绝；改用历史协议实际使用的 `pilot_v003/reviewed_manifest.jsonl`，未伪造bbox。2更新profile完成，之后同协议在GPU4执行500更新，scope始终component_diagnostic，无正式D0 gate。

- run：`gpu_diag32_v001`；来源/固定32 ROI/热图每50更新/全部更新日志与权重保留。
- loss从0.08041496降至0.02875796；最终P95=53.72755热图像素，平均EPE=30.96646，理想热图P95=0.237683，相对零图P95改善16.94%。
- 主干有非零有限梯度且参数实际变化；已跟踪的7个训练相关源文件运行中均未改变。CUDA峰值分配约729MiB，缓存约760MiB，nvidia-smi该进程观测约1110MiB；单更新约0.93秒。
- 角点解码出现八点重合。loss下降不能代表学习通过，禁止放宽门槛。下一步对GT热图、末层特征、梯度与解码进行定因诊断；确认原因后按原计划做有界单变量修订，未进入正式10k生产/A0/B0。

### 用户追加的图片位置要求已落实

权威图片仅在服务器 `data/action4_annotations/`，分 `checks/`、`formal_training/`、`final_inference/`。现有3张本轮图片全部归checks，迁移哈希回执为 `migration_20261019_v002.json`；原运行目录通过annotation_locations.json定位。正式训练和最终推理目录当前没有合格图片。本地3张查看副本确认服务器哈希一致后删除，后续不创建本地镜像。

视频新增annotation_stage（默认checks），拒绝诊断权重写final_inference，同时补齐B0/C1、完整epoch、正global_step资格检查。原3帧时序检查同时核对集中目录与旧目录不存在；相关11项测试通过（`annotation_stage_tests.log`）。此项不代表真实2850帧视频已经生成。

### 当前未完成项

U02多视角/完整腕部场景质量、U03单目标实际渲染、U04 gate版本绑定/连续零检出、U05同卡并发实测、U06统一调度/异步视频、U07对应剩余检查；学习失败定因和有界改进；128原尺寸审核与10k S0、正式D0/A0/B0、必要C1、最终完整视频及全片复核均未完成。目标保持完整，不将工程恢复或组件诊断作为最终交付。

## 2026-10-19T17:40:18.794955+08:00 标签核查、学习修正、完整抽帧及并行复审

### 用户最新目录职责（替代此前临时图片布局）

data仅存原始数据、标注/合成数据集与Blender资产；运行结果全部outputs。已将36张审核/预测图及后完成的bias01诊断图按SHA迁回各自outputs/visualizations，原data/action4_annotations已移除；旧布局说明和回执保存在outputs的superseded_image_layout，历史记录不伪改。两个父任务早期CUDA/OptiX预览的10个运行日志/检查文件亦已迁outputs，见parent_render_log_migration.json。所有权威文件在服务器，未新增本地图片镜像。

### 真值与预测严格区分

用户截图是初始/训练后预测对照，红线未包住机身，确实错误，不是训练真值。旧图缺少GT栏，容易误解；新诊断图分原图、绿色合成GT、初始预测、训练后预测，分8ROI分页及4ROI概览，不再生成难读的32行长条。

实际核查 `outputs/action4_label_audit_v001/report.json`：32图55相机实例，独立3D到2D投影最大差6.0273e-5px；容许1px栅格边界后，可见mask和完整amodal mask在八角凸包外均0像素。原图/GT/失败预测对照在同目录visualizations/gt_vs_failed_prediction_v002.png。该检查确认合成标签覆盖其机身，不能替代真实视频标注精度或合成外观准入。

### 角点学习问题定因与单变量序列

审计 `outputs/action4_corner_collapse_audit_v001/audit.json`：初始角点末层全部权重梯度为正，原500更新后94.92%权重为负，预测峰全部位于隐藏特征全零区。保留原失败权重/曲线。

1. `gpu_diag32_bias01_v001`：仅角点头初始概率0.5改0.1，其余32ROI/seed/loss/LR/decoder/500预算相同。P95降到8.90067，平均1.249，仍失败。
2. `gpu_diag32_bias01_lr1e3_v001`：在上一配置上仅新模块LR1e-4改1e-3，backbone仍1e-5，从同ImageNet初始化重跑500更新。最终P95=0.234770热图px，平均=0.125535，相对零图改善99.637%；主干有有限非零梯度且实际改变，跟踪源码运行中未变。概览为 `outputs/action4/gpu_diag32_bias01_lr1e3_v001/visualizations/roi_gt_prediction_overview.png`，另有4页完整32ROI对照。

这是固定诊断ROI的可学习性通过，不是正式D0 gate、A0/B0或真实视频成功。正式配置将明确采用验证后的角点bias0.1及D0新模块LR1e-3，原预算/损失/解码/主干保持，其余阶段LR不自动提升；正式D0仍须在accepted S0上重新执行。

### GPU工程检查

`outputs/action4_gpu_full_amp_global8_v001/report.json`：实际ResNet50/decoder256、1024x768全图、8个真实诊断样本（含0目标）、micro1累积global8，原FP16 AMP/65536缩放、clip1，一次完整optimizer更新全部梯度有限且主干改变。峰值分配约1.47GiB、预留约1.60GiB。早先global1的独立探测出现中心头梯度溢出，缩放8192等可避免；复核发现global1不等于计划global8，故没有据此修改正式AMP策略，原失败证据保留。NCCL2/4rank和正式训练仍待验证。

### 两视频完整抽帧已验证

- 原片：data/action4_real/cache_v001/frames，2850张3248x2464 PNG，约9.136GiB。
- demo：data/action4_reference/demo_standard_cache_v001/frames，2850张950x720 PNG，约1.153GiB。
- 两者均2997/100 FPS、timebase1/11988、PTS0..1139600步长400；图片frame_00000000.png..frame_00002849.png，无遗漏/额外文件。
- 所有PNG完整解码与SHA记录；独立FFmpeg再次解码源视频和PNG序列，逐帧RGB buffer哈希全部一致；源视频SHA保持不变。报告outputs/action4_extract_raw_v001/verification.json及action4_extract_demo_v001/verification.json，汇总video_extraction_summary.json。

### 用户授权并行子智能体与双层复核

U04由独立实现者交付；独立spec-review PASS（32tests+9trainer过期gate反例），随后quality-review PASS（19tests+27边界反例），文件SHA一致；报告outputs/action4_u04_spec_review_v001.json和action4_u04_quality_review_v001.json。这是工程准入/反馈检查，不是正式数据/模型成绩。

GPU调度器已退回补齐原子2/4卡训练组支持，当前实现待独立spec/quality及实际GPU验证。Blender已完成有来源照片与官方CC0丹宁PBR纹理、腕部连续表面增量、原生前后对照和实际重开；spec PASS，quality仍进行中，不准入128/10k。视频IO子智能体正在复用原encode接口补齐图片转视频验证，之后独立复审。最终整合与真实视频验收仍由主智能体负责。


### 图片转视频接口交付与主控复核

核验时间：2026-10-19T18:54:52.064386+08:00。抽帧分支已复用 DataAdapter.encode_video 完成图片转视频接口，CLI 为 scripts/action4_data.py encode，使用说明 docs/action4_video_io.md。本次主控核对 adapter/CLI/tests SHA 与独立 quality v002 回执一致，实际执行 encode --help 并重新执行 test_action4_video_io.py，11/11 通过、零跳过。spec v001 PASS，quality v002 PASS；此前 FFprobe 未传递线程预算的问题已修正，旧失败报告保留。

接口使用完整帧清单和实际图片序列，保留原生尺寸、精确有理帧率、逐帧 PTS/duration/timebase。编码后逐帧解码检查帧数、时序、尺寸、无音轨和输入哈希未变；缺帧、多帧、损坏、尺寸错误、路径越界或覆盖现有文件均拒绝。CLI 限制生成 MP4 和 verification.json 在实际项目 outputs/ 内，模型叠图同样存 outputs/，原始帧保留 data/。线程预算 1–8，默认4，同时传给编码和完整解码核验。

证据：outputs/action4_video_io_spec_review_v001.json、outputs/action4_video_io_quality_review_v002.json。验证是实际 FFmpeg 七帧接口样例及反例测试，不是2850帧模型预测验收，也不证明角点准确。完整原片与demo抽帧已另行验证；完整预测视频等待合格模型和全部绘制帧。


## 工程验证进展与原生渲染瓶颈（2026-10-19T18:57:39.834082+08:00）

实际NCCL 2rank/4rank完整更新已完成，证据outputs/action4_server_bootstrap_v001/distributed_runtime_decision.json。原P2P路径超时、禁用P2P后的AMP65536溢出均保留；项目进程NCCL_P2P_DISABLE=1、NCCL_SHM_DISABLE=1和AMP8192完成有限梯度及全rank参数一致核验，未修改全局配置。不据单步速度选择正式卡数。

统一dispatcher经spec v002/quality v002通过后，实际同GPU4双诊断进程重叠19.1629秒，两项各8updates成功，峰值各1110MiB且输出独立；回执outputs/action4_server_bootstrap_v001/dispatcher_gpu_overlap_verification.json。此为调度与资源验证，不能代表正式模型精度。源码继续增量后profile需重测。

Blender v008已通过spec及quality，来源照片纹理、CC0丹宁PBR、连续衣物、腕部衔接和局部接触保留。v009原生64samples两后端各3张只作性能检查，尚未128接受。

可选amodal crop实现的两组完整流水线证据outputs/action4_amodal_optimization_v001/phase2/paired_pipeline_comparison.json：8组visible/amodal mask零像素差，GT/面积/状态精确一致；81.59/83.66秒降至42.65/43.33秒，RSS约7GiB降至3.8GiB。RGB跨模式有742/573个通道值差1，同模式重复722个差1，不能称RGB bit-identical。当前独立spec复核进行中，随后quality；未接受为正式数据。

隔离RGB去噪证据outputs/action4_rgb_denoise_preflight_v001/comparison_report.json：CUDA renderer保持，仅OIDN改OptiX，4次完整RGB33.51–34.69秒降至11.69–13.38秒，16张visible mask一致；RGB MAE0.196–0.326/255，父任务与实现者查看原生屏幕/镜头边缘及丹宁细节，未见明显退化，范围仅受控样本。采样GPU峰值由2236升至5134MiB。权威去噪器接口尚未接入，组合流水线与并发资源待测。

视频队列spec v002已通过；quality发现失败首profile无法显式重试的问题，现已补一次有证据的人工控制retry，独立quality复审进行中。未知/存活进程不释放和不重启，旧失败及审核保留。完整128/10k数据接受、正式D0/A0/B0、必要C1、全2850帧合格角点视频与全片复核仍未完成。


### 掩膜与视频队列复审关闭（2026-10-19T19:01:58.607637+08:00）

Amodal spec/quality均PASS，分别见outputs/action4_amodal_spec_review_v001/spec_review.json和outputs/action4_amodal_quality_review_v001/quality_review.json。独立重算8对原生mask零差、核验16文件SHA；100组随机padding/PNG往返及19,603整数边界通过。视频queue质量v002 PASS，60回归和19独立反例通过；已验证显式一次retry与崩溃重放，范围为CPU/mock/Gloo工程控制。主控新执行test_action4_amodal和test_action4_video_queue共40tests全部通过，当前代码身份匹配复审。开始单写者新增显式RGB去噪配置，并在GPU6做同源完整组合配对；尚未准入128/10k。


### 组合渲染复审、真实调度测量及组装遗漏（2026-10-19T19:13:11.842447+08:00）

Denoiser spec/quality均PASS，主控重新执行denoiser/amodal/synthesis/Blender扩展36tests全部通过，亲自查看两场景全图、四目标原生raw/GT以及丹宁/皮肤/线缆细节，未见明显去噪退化。证据outputs/action4_denoiser_{spec,quality}_review_v001/与outputs/action4_render_combined_v001/。

新10k native plan v001误保留隔离试验GPU6 allowlist；首次profile另遗漏显式--gpu-uuid，第二次补GPU4后仍被allowlist拒绝。两次均在Blender前失败，零训练数据；失败queue/log/plan原位保留。新v002恢复GPU4–7、全10000 entries/seed/split与v009精确一致，前128有八family和双107/零7/仅左7/仅右7。native_resolution_pending=false仅表示本配方已使用原生尺寸，显式128审核、六criteria及hash准入仍必须满足，未接受数据。

profile_render_native64_combined_v003真实succeeded/exit0，8图完整生成，显存采样峰5134MiB；精确时间与身份见outputs/action4_server_bootstrap_v001/render_profile_v003_historical.json。freeze独立检查发现NATIVE-FREEZE-001：render将launch.json迁到outputs，assemble仍读data，导致FileNotFoundError。当前profile正常结束后派单写者作最小路径修正及回归，历史profile不改hash复用，修复后必须新freeze/profile。此缺陷在正式10k生成及训练前发现，报告outputs/action4_native_freeze_review_v001.json。


### 组装修复关闭与新冻结（2026-10-19T19:21:45.848320+08:00）

NATIVE-FREEZE-001实现仅修正assemble的运行凭证读取：canonical outputs优先，缺失时兼容legacy，双份冲突、损坏、悬空均拒绝，不改变plan/source/完整ID/128与六criteria门槛。独立spec及quality均PASS，报告outputs/action4_assembly_receipt_{spec,quality}_review_v001/；主控另跑原synthesis8项与assembly_receipts9项共17项全部通过。生产synthesis SHA822dd85ef3a1a435d9374508aa5e2a6439150936f0647043c86085bdad2b1e3e。

新plan data/action4_synth/server_s0_native_v003/plan.json，SHA cc89c91b0f70435ff86cc1591b5fbe193be45256e0c9db27ecce02e8ff7a3d6f；所有entries与v002完全一致。canonical profile_render_native64_combined_v004实际已启动，后续同样8样本比较1/2/4同卡总耗时，所有输出独立，不修改失败与旧profile。

审核页工具 runtime/bootstrap/action4_native_review_pages_v001.py 已在旧profile8图14实例实跑验证，raw原生裁剪14/14与输入像素完全一致，整图明确缩放，GT为绿色并明确非预测，原RGB/manifest/连接规则SHA绑定，所有review状态pending。索引outputs/action4_native_profile_review_v001/index.json；这不是128正式准入。


### 并发验证完成及用户2000先导调整（2026-10-19T19:37:37.887083+08:00）

当前source822dd85的同8样本1/2/4进程均成功，耗时153.45/99.80/77.02秒，按逐档至少10%规则选择4进程/卡候选。56对可见/amodal mask逐像素比较一致，实例GT和状态一致；每进程仍保守预约8192MiB+每卡8192余量。证据outputs/action4_server_bootstrap_v001/render_concurrency_selection_v001.json。所有GPU任务已经结束。真实性改进改变配置/资产后需重新profile，不直接冒用旧profile。

主控实际重看原帧0/540/1200/1980/2550/2820，确认原片腕部相机常下挂、手臂更宽、转腕可见面变化大，桌布/视角及折叠衣形与当前通用合成场景仍有差距；已派独立image-backed差距审计。先落实高影响增量再2000，不把前述工程检查当真实感通过。

无损blend原型已CPU重开验证，原件SHA保持，11类语义指纹一致，两个文件17.69→6.35MB、39.48→27.69MB。普通另存会清理无引用重复图像，保留版在副本内加存储保留标记，差异证据保留。outputs/action4_blend_compression_v001/handoff.json；不改生产src。2000现有未压缩也满足预算；10000即使此压缩示例仍超过当前保守空间要求，最终依据2000真实全流程核算。


### Pilot配置与首张真实性增量复核

configs/model/action4_pilot2000_v001.json已创建，SHA2c70c547901dba2f01b221b4b93925ffe9762122c7ff0ce73b078e297bdae360；独立spec16项及quality均PASS，原默认配置未变。数据三manifest不存在、有效ROI和D0 gate待实测，未启动训练。队列第5轮仅提交一次完整视频，须实际完成和复核，不能将排队当结果。

训练前空间基线见outputs/action4_server_bootstrap_v001/storage_baseline_before_pilot2000_v001.json：六类资产/运行目录全局唯一inode实占32,608,571,392bytes，未跟随symlink，硬链接去重；属于含历史记录的非原子基线，不是pilot增量。

首张新table预览已实际生成，主控查看后确认相机下挂方向较旧版改善，但capture roll与原f540方向相反、手距和目标画面位置仍需调、C形手及大块衣摆接触仍不合适，已退回有针对性修正。证据outputs/action4_realism_increment_v001/controller_first_preview_review.json。该图未接受；真实感审计与实例几何通过分开记录。


### 本体材质闭合与场景手势负例（2026-10-19T12:03:58.127891+00:00）

RG02侧面浅色DJI标识增量已由独立spec及quality复核PASS：outputs/action4_body_material_{spec,quality}_review_v001/。新data/DJI_Action4_construct/server_r4_realism_v002/DJI_Action4_Editable_Foundation_R4_realism_v002.blend，SHA c43de54aa76bc1127f30e54dd17a1b07957da96a91bf51ea0c650fdbfc69197c。仅264既有字形面赋浅色材质slot5；CPU重开验证几何/UV/transform/canonical8/五张packed图像与原slot1实例2屏幕替换行为不变；原f1980参考crop逐像素对应。两组非空mask面积1324814/958343且旧新0差；初次同名输出覆盖的空mask失败保留且未计入证据。主控亲见侧面及后屏斜角新旧原生渲染，浅色logo改善，其他RG02项目不冒称闭合。

场景v004小预览已用新asset。主控实际查看wall/wall_turn，手臂展开和相机挂载方向较旧版改善，但指尖存在明显拉扯/尖刺，明确拒绝该手部版本；实现者亦已识别，正在采用长度归一化/稳定形态，必要时回退不稳定精细手势。128/2000尚未生成接受，训练未开始，不以几何投影通过代替真实感。


### 近场遮挡独立审查（2026-10-19T12:14:14.983500+00:00）

outputs/action4_occlusion_gap_audit_v001/report.md实际检查8个原视频锚帧及v001–v007预览，25实例面积/hash/visible嵌套一致，未发现将腕带/手臂/支架整体并入机身标签的证据。原f1980是强前臂遮挡，f540/960有近腕部及插头/线缆影响，f1200/1290靠近不等于全是双机互遮，f2550衣物主要背景和抓下摆，f2820另含出画。v006折叠场景相机出画过多不匹配参考；v007恢复双实例零截断，但底部黑带/白色三角缝与手布接触仍需修，未接受数据。检查未要求虚构桌沿或随机杂物遮挡。

同步训练计时实现已交付并通过独立spec（9tests含实际CPU/Gloo+23独立检查）；质量复审进行中，未实际GPU训练。配置默认off；新pilot配置启用须另版本冻结，不能沿用旧config/profile身份。


### 同步计时复核闭合与v008遮挡检查（2026-10-19T12:24:21.251811+00:00）

同步计时独立spec/quality均PASS，主控重新运行timing/training_eval/video_queue共48tests全部通过（CPU/Gloo，非GPU吞吐）。独立pilot配置v002仅比v001新增runtime.synchronized_update_timing=true，SHA9d328f71d0ffd8ef7e66180bfcb25919fe059e4903f518ebedd2ad8d08d6f20c；加载/queue校验通过，预算未改，原v001留档。真实30更新2/4卡吞吐仍待已接受数据和A0初始化。

主控实际查看v008八类预览：wall指尖尖刺消失，table_fastening倾斜方向修正，body浅色logo实际消费，fold不再帐篷或大面积出画。未接受为native数据：near_double实例1真实遮挡92.47%且ignored，应保留为极端负例而调整代表基准以对应原片可见范围；self_forearm实例1遮挡59.74%，但前臂从左上横入与原f1980下左进入关系不符；fold手指与布角接触仍需改善。已把这几个高影响问题反馈单写者，未放宽ignore/碰撞门槛，未删除实例凑配额。


### 原生候选与工作流检查进展（2026-10-19T21:59:24.480311+08:00）

场景增量最终renderer SHA549c2d4887211d1d5d0c49d19a8887d68661e1797ba86c6e23d0fba09db42549、foundation SHAbcfb7915d4fa28c9cbbce656c047670fac675435196f46558aef67605fb60bf8已通过独立增量spec/quality；修复袖口拉扯及前臂方向，保留通用手部和有限姿态局限。新2000候选plan SHA2958b5fc39688cd1e30f2bbdc4c82886481e905bcb9dba9d3eab38c8c874e87d，首8张为同计划固定128的indices0..7。outputs/action4_pilot2000_review_v001/reviewer_first8.json逐张接受8张，完整128/2000尚未接受。sample2两处1px mask边界在既定容差内、1px膨胀后outside0；sample6仅0.936%出画，不证明强截断覆盖。训练未开始。

模型核心已有真实GPU6 full1024x768/ROI256/global8一次更新、梯度有限、主干改变及严格网络state reload输出0差；证据outputs/action4_model_core_check_v001/gpu/report.json。这不是optimizer/AMP/RNG完整恢复或效果证明。

工作流审计发现并修复：历史top权重内容替换、步数与optimizer/scheduler不一致、epoch20提交后快照发布中断、partial覆盖破坏保留别名；独立spec又发现CUDA RNG缺失及非零rank在repair前读别名。v002实现75项CPU/Gloo回归通过，目前独立复审中，不称CUDA实测完成。证据outputs/action4_checkpoint_workflow_fix_v002/。一个复用rendezvous端口导致的检查停滞及限定PID停止记录保留；独立端口版本通过，未跳过断言。

eval/test对应split身份及严格epoch/global_step检查已完成独立spec/quality，runner SHA794935b9dcaa46ebcf7251e156b5f8b0e855c8c4c292d76ff810dcf5e38c7e06；合并trainer回归及真实CUDA完整恢复待执行。所有src冻结前不复用旧dispatcher profile；渲染计划本身仍绑定未变renderer资产，首8候选不因模型guard修复而作废。


### 工作流复核闭合与资源并发调整（2026-10-19T22:11:11.779306+08:00）

checkpoint独立spec v002/quality v002均PASS，主控90项联合CPU/Gloo回归通过。随后canonical cuda_full_trainer_resume_v001在GPU6完成真实全ResNet A0初始化/B0连续与epoch边界续训对照，epoch1/epoch2及下一更新数值状态精确一致；峰值2476MiB、16个资源样本。明确未比较整个top_checkpoints选择元数据，不称多卡恢复/正式训练/精度通过。证据outputs/action4_cuda_resume_check_v001/gpu/report.json。

用户指出CPU/GPU利用率低，实测先前仅GPU4单渲染，GPU5/6空闲，确实调度不足。新源码完整8图profile_pilot2000_realism_v003完成，峰值5136MiB，建议6420MiB，继续每worker保守8192MiB预约。固定128已有首8，GPU5/7各4张完成后，剩112张分16任务派发到4–7号卡，各4workers、每worker2CPU，共32渲染线程额度。刚启动采样四卡100%利用率、显存约12.4–13.2GiB，属于瞬时采样，不冒称持续100%或最优并发。coverage receipt逐项验证0..127无漏重，resource_bench副本排除。尚未整体接受128数据；新增结果继续独立目视/几何复核，未放宽标签或图像质量门槛。


### 固定128准入与剩余1872派发（2026-10-19T22:30:09.332701+08:00）

固定128逐图已完成：first8审核8、next8审核8、root补16、review032_079审核48、review080_127审核48，合计128且没有重漏。root实际查看自身16图/27实例及17、37、90、103跨审阅抽查；太大的传输图没有计为已看，重新分半传输后才记录。完整性审计225实例1800角点、578RGB/mask文件，最大独立投影差0.0003652902px，17边界非嵌套像素全部在原1px容差内；mask超投影凸包容差像素0。9实例实际出画，最大11.144%，遮挡最高87.727%。合成皮肤平滑、手姿重复、屏幕重复和布料/捏扣形变近似仍是先导域差距，不宣称精确动作复刻。

主控签收 `outputs/action4_pilot2000_review_v001/acceptance128.json`，SHA072f1d6d868fb3ca0d8276036f9f3ca849976b6af94dcd67df1d7a430d326d09，绑定同plan2958...及128条真实RGB/reviewer/observations。此为扩展先导生成的审核门槛，未把未生成的1872张或全数据集标为training_accepted。最后2000精确assemble、几何检查、分组配额及准入仍必须执行。

GPU4同8图新并发对照：1worker162.949s、2workers107.591s、4workers81.907s，两个提升分别33.97%和23.87%，均满足既定10%收益选择；28实例跨组对照的角点/状态/mask哈希一致。新证据 `render_concurrency_v003.json`，选择4workers不再仅依赖历史场景速度。

剩1872按4逻辑shard各468、每shard7×64+20拆32作业，已通过现有normal dispatcher准入并全部入队；索引128..1999精确一次、无benchmark混入，物理GPU4–7每卡最多4workers、总32CPU线程额度。当前运行监控会持续采样，任务自动按已声明队列接续，未开始训练。派发时可用磁盘667615023104bytes，剩余保守估计125627793408bytes，满足100GiB保留+1.5倍剩余估计。回执 `production_dispatch_receipt.json`。


### 生产碰撞失败与下游入口预检（2026-10-19T22:53:45.458258+08:00）

生产job pilot2000_prod_s2_e288n64_v001因Blender ValueError退出1；不是OOM。最后完成1230，实际失败1234（seed149496605、near_double、train、双机），原碰撞函数CPU复现797对交叉。该分片20条prefix仍为未完成候选，不进入正式assemble；失败日志完整保留。其余任务继续，队列补入下一任务后恢复16个实际live子进程，没有超显存预约。collisionaudit位于outputs/action4_pilot2000_collision_audit_v001，预检及修复尚在执行，未宣称问题闭合。

训练启动审计证明B0 component30的profile不能直接用于formal B0（视频CPU工作负载字段不同）。保留2/4卡30更新测量选择后，以新完整5epoch正式B0 profile=True作为实际训练，最终queuedvideo在trainer退出释放资源后执行，无循环等待；不伪造profile身份。另发现eval/test专属CLI参数被dispatcher拒绝，修复暂存outputs/action4_eval_dispatch_fix_v001，22CPU测试及独立spec通过，尚未应用生产。所有src在渲染期间保持冻结。

存储新核算outputs/action4_pilot2000_storage_v001：当前代码仅首8保存blend，2000数据中心估计约17.163GB；含本轮模型/视频/历史基线及余量总项目预算约81.292GB。此为运行中估计而非最终实占，生产失败及后续修复将继续计入，不删除旧产物来降低统计。


## 用户否定当前场景后撤回准入并停止生产（2026-10-19T23:18:38.001507+08:00）

用户明确指出DJI相对手腕固定方向、双机间距、手臂/手/物体比例及光照环境有根本偏差；旧128“先导视觉可接受”的判断不充分，不能继续作为批量/训练依据。已将acceptance128.json decision改为revoked，并在outputs/action4_realism_reassessment_v001/acceptance128_prior_decision.json保留原决定。没有启动正式训练，没有删除已有候选图。

已通过canonicalqueue的lock/save取消待启动生产作业，按已登记PID/start/命令核验后仅终止本批活动children；另一个正在launch的child稍后出现，补充核验并停止。最终32生产任务3 succeeded/29 failed（包含实际几何失败、用户纠正后主动终止及启动前取消，不能都称算法失败），live_children/live_workers/queued均为空；回执production_stop_final.json。CPU原计划几何扫描亦已停止，保留315条部分结果，不作为完整2000预检。所有暂停仅针对失效的数据生产分支，项目整体goal继续推进全局架构纠正。

新的只读审计已确认程序前臂中心线长度随family从396mm变537mm，而camera_head局部rotation由wall0°变table约-105/-110°，与固定人体和佩戴关系不符。当前开始并行核查rig相对变换、解剖/物体比例、环境与观察成像。官方尺寸通过DJI规格页面核验；后续测量报告会明确图像投影量、模型量与真实世界未知量，不将单目估计冒充实测。旧checkpoint/eval工程检查仍保留对应版本有效性，暂存seedrepair和evaldispatcher补丁均未应用，不再靠seed修补继续失效场景。


### 全局重审的有证据结论（2026-10-19T23:56:16.651868+08:00）

三路审计均实际查看原视频并读取现有模型。rig_relation/report.md确认同侧局部camera安装0→105/110°等变化，且105°变化连带机身中心移动约(0,27.015,−61.271)mm；也排除了“相机被负scale镜像”的错误归因，已查body det≈+1且scale1。anatomy_scale/audit.md确认薄腕截面、缺少掌指关节、前臂中心线随family变长，并找到现有连续人体资产可作统一比例先验，未宣称已绑定骨骼或完成拟合。

environment_capture/environment_capture_analysis.md的人工2D观测显示：f2550原视频两机可见宽度约10.1/9.9%W，旧合成6.3/5.9%；f1980平均中心y原约68.6%H，旧合成53.4%；桌面投影占比亦明显偏大。观察框有显式不确定度，遮挡框不能当全机身GT。显示域区域统计：墙面f0中位数123.9/124.2接近，但f1980桌布144.0/124.4偏暗、格线P05 66.0/36.5偏黑，皮肤73.2/121.4反偏亮。不能统一提亮，也不能把显示域量换算成实测灯瓦数；先对齐几何/视角再调材料/照明。两个污染patch已剔除并保留原记录。

当前只完成全局架构和观测/先验准备。真实多帧拟合、新rig、新场景预览及重新验收尚未完成；没有恢复生产，没有开始正式训练。


### 已纳入用户要求的受约束参数模块（2026-10-20T00:27:37.983621+08:00）

已在同一canonical plan增补校准后可复用的参数范围/场景覆盖模块：固定量、可控量与结果量分离；距离同时核对图像占比；腕姿/双机关系/遮挡联合约束；区域光照外观校准；原场景为主、少量轻微扩展；分组隔离、分层覆盖与真实验证。先导默认90%参考支持/10%轻微扩展是明确的保守规划选择，未宣称最优或强泛化已验证，不追加2000数量或训练预算。距离/角度/亮度等具体数值上下限待基础模型拟合后冻结，未填造数值。

计划定义了拟新增action4_scene_sampling模块及load/propose/preflight/render_measure/validate/audit职责，尚未实现该采样模块或范围配置。当前G1工作只建立统一米制rig初始化和不可变契约，不提前执行增强。global_scene_architecture_v001.json因G1已绑定SHA保持原字节，新要求首先写入plan，避免在验证中悄然改来源。


### G1初始化复核完成与参数模块状态同步（2026-10-20T01:01:05.060882+08:00）

有效资产为global_rig_init_v001/validated_build_v002，SHA084bd1e5afdda619b785eb7a35213782acd3dc5a918d718f840646eef1017890；独立spec和quality均仅通过G1初始化工程范围。质量复核检查15093个顶点权重、单侧运动隔离、固定安装关系及16次非法输入拒绝。此处不签真实比例、纹理、全姿态碰撞或训练数据准入。G1B共享环境与G2a联合投影拟合继续进行，结果待冻结后独立复核。

本次回读确认受约束参数模块要求已写入同一计划，计划SHA5081aa869365ec06d51b750e0e8b03f0ba29b2a817bcef34642c469d00547cfc，与更新回执一致。执行状态中旧的active_render_plan、首8接受、seed修复和待生成1872等过时主线已改为历史状态，修改前完整状态保留在outputs/action4_realism_reassessment_v001/status_before_parameter_module_sync.json。当前无已批准的新生产范围，采样模块尚未实现；不得将计划登记当作模型或生成器已完成。


### 评估调度入口修复应用与生产回归（2026-10-20T01:06:41.204564+08:00）

在canonical队列实际状态全部终态后，应用此前独立spec/quality已通过的精确补丁；src/model/training/action4_dispatch.py由87936adb297e208f22262273c243df68fb0b0c489712a0bb879879822d99556b变为d5dc60a58ef35fa655f50388505dff61d62ebe0943fc4519a9f693d1075da11b。补丁让eval、阈值调参、冻结选择、test的真实CLI选项与输入身份进入调度检查，同时区分各自资源配置；未修改runner的checkpoint/split准入。

实际生产模块21项新合同检查通过；另1项仅重放旧缺陷的测试在生产模式按预设跳过。102项受影响dispatcher、queued-video、training-eval和eval-lineage CPU/Gloo回归通过，无失败。证据outputs/action4_eval_dispatch_apply_v001/application.json及两份日志，原源码和应用时队列快照保留。历史profile/权重来源未回填；后续新作业须建立当前来源和fresh profile。这不是正式模型测评或GPU效果验证。


### G2a拟合未成立与G1B独立质量发现（2026-10-20T01:15:05.437694+08:00）

G2a实际执行4个有界CPU初值，每个最多200nfev；start0/1耗尽预算，start2数值收敛但主控实际查看f540_R原生对照，完整盒明显覆盖手腕/背景，拒绝用于校准或生成。start0与2的左右固定安装候选差72.89°/147.68°和35.51/65.54mm，表明多解。严格只用brace/axis求留出姿态时，1980R机身边RMS84.23px、2820L为134.46px，不能冻结K/A为真值。报告和各次失败均保留在rig_projection_fit_v001；未增加盲目搜索预算。

另确认v001实现把原packet的“可见rear面”增强解释为“该track每条边都是rear四棱”，超出了原观测语义；局部单向边距离也允许整盒错误的退化。后续须先补具体可见外轮廓/边身份和可核实CAD特征，不能虚构隐藏点或把可见mask当完整amodal。

G1B已通过初始化SPEC，但独立QUALITY以真实Blender反例判FAIL：静态快照漏shader mute、TexCoord.object、灯spread和新增未登记光源；0.5mm焦距被钳为1mm但K合同未察觉；无效look在后段失败留下新对象/world/camera。默认26mm资产的既有几何SPEC结论并未因此推翻，但可复用API质量未过。单写者修复并另建v003，旧v002保留；修复后重走SPEC再QUALITY。证据global_environment_quality_review_v001/quality_review.json。当前仍无获准训练数据或新批量渲染。

这21项新生产合同反例已正式归入tests/test_action4_eval_dispatch.py，直接导入生产dispatcher，避免未来回归依赖历史outputs中的暂存模块；独立执行21/21通过且无skip。旧缺陷重放仅保留历史证据，不删除旧日志。


### 连续人体外观整合缺口核查（2026-10-20T01:30:16.755602+08:00）

主控实际CPU Blender重开有效G1：global_source_anatomy有UVMap但material_count=0；相机R4的body_black_surface及分实例rear_screen纹理仍已打包。因此当前初始化人体不能直接作为完成真实纹理的场景交付。已有skin_forearm.png为原frame0的160×160真实裁片，哈希及逐RGB像素与manifest区域一致，主控已看图；它包含拍摄光照/压缩，不是测得反照率或微观皮肤扫描。后续在统一衍生场景中复用现有_skin和apply_photo_material，连同腕带纹理再冻结合同，先几何/观察成像后外观拟合，不靠贴图掩盖比例错误。证据appearance_integration_audit_v001。

主控另实际查看f1980全帧并对照代码：G1当前只有灰腕带与初始黑支撑，原_wrist_rig中的绿边、白色腕上模块、球头/快拆、插头/线缆尚未整合。它们是后续统一rig必须复用/拟合的真实近场部件，不可漏掉，也不可并入Action4本体八角点或机身标签。旧部件尺寸/位置仍为先验，须重定基并对照原视频，不能恢复旧family相机乱转。

已将当前投影拟合实际使用的SciPy1.18.1声明为pyproject calibration可选依赖；该extra要求Python>=3.12，保留核心原requires-python>=3.10不变。本次只声明已安装依赖，没有重新安装或升级numpy/torch；实际版本和修改前pyproject保存在body_projection_fit_v001/dependency_declaration。


### 实际CAD机身对齐改善与指骨控制初始化（2026-10-20T01:50:17.920594+08:00）

body-only按3个固定K、双机、2feature分支执行12starts，实际8–71nfev，均在150预算内。f/W=.6 outer_support下，frame0左机人工边端点RMS3.3436/max5.5225px，右机RMS1.9641/max2.9500px；实际51503三角形投影的已知可见漏检1.8319%/0.2743%，确定背景excess0/0.4373%。主控实际看两张CAD原生图，整机投影回到正确区域；右机顶部/镜头凸起仍有不确定和偏差。仅是frame0人工校准观察下的几何诊断，不是训练角点精度。跨K左机depth0.3052→0.5491m、右0.2544→0.4389m，未确定K或A。独立SPEC重算12最终位姿及全部三角mask精确一致。

主控QUALITY提出BPQ1：fit入口只凭forward review布尔而未绑定图像/特征/当前输入身份；需最小来源绑定修复，历史12次拟合与旧源码身份保留，数学不变且不重跑无必要优化。证据body_projection_fit_v001/controller_visual_review.json、body_projection_spec_review_v001和body_projection_quality_review_v001。

指骨模块已形成原7+新增28骨的衍生G1资产build_v007，原控制网格、7骨、相机/腕带矩阵保持；实现者10CPU与实际重开检查通过，独立SPEC执行中。主控实际查看食指靠近与开放抓握形状的网格投影，仅证明可控形变基础。源末端/指甲已有336交叉对；温和候选未新增但不能称零自交。指尖距73.8mm尚未接触，真实捏合/抓衣必须继续完成。较强姿态仅被rest-normal保守筛查拒绝，不能称真实翻面或生产关节范围。采样范围null、generationfalse。


### 初始化复核闭合与剩余环境缺口（2026-10-20T01:58:51.712825+08:00）

finger初始化独立SPEC和QUALITY均通过限定工程范围，主控核对源/资产SHA及两张实际姿态投影后登记controller_review.json；生产范围与真实接触仍未通过。继续有界thumb/index真实指腹接触诊断：每手最多4starts×150nfev，指腹按源末节掌侧三角排除指甲和根部，要求实际求值表面最近点见证/无新增穿插，不能把骨端近距离当接触。较宽角度仅作为显式工程搜索边界，不改变旧25°初始化接口或宣称真人测量。

G1B v003独立SPEC通过，但QUALITY仍捕获world.use_nodes和桌面visible_camera切换未改变snapshot的残留贡献开关缺口。此前mute/spread修复有效，K/事务回滚未发现新问题；已派原实现者局部修复并另建v004，保留v003。没有因此恢复渲染/训练。


### 审核来源绑定及共享环境初始化修复闭合（2026-10-20T02:10:33.224594+08:00）

BPQ1新forward review绑定5来源/14产物/4CAD图，13绑定+10数学测试及7个独立stale反例通过；主控实际验证当前bound review并比对17非编排函数AST未改，QUALITY闭合。旧12次拟合身份不回填、不重跑。主控首次使用错误层级的forward路径导致只读验证FileNotFound，已记录并改为fix目录内实际路径，未触发求解。

G1B v004显式纳入World/对象的当前Cycles贡献开关和集合/viewlayer层级；独立SPEC通过原两个失败反例及代表层级检查，实际保存重开和27mesh表面等价成立。主控核对当前source/asset/contract哈希、局部diff及独立证据后QUALITY闭合；既有Q2/Q3逻辑未改，未重复无关整套测试。四纽扣只有索引排序不同，不冒称逐bytesmesh相同。此处仅关闭初始化不变量缺陷，仍未拟合真实环境或补齐皮肤/附件。


### 整合初始化、第二视角与手根局部修复的实际进展（2026-10-20T02:47:42.841664+08:00）

整合v004包含真实来源皮肤/腕带纹理和16附件mesh/4根，源finger35骨及目标core保留；GIS1错误把XYZ枚举当canonical序的风险已经修复，原R3 manifest顺序和mm→m转换实际校验，独立SPEC和主控QUALITY通过初始化范围。未生成训练标签，不存在据此证明的旧标签损坏。真实姿态/安装/环境仍未拟合完成。

f540_R经6starts（总65nfev、377含差分调用）完成第二视角诊断，独立重算51503三角union与原生边误差一致。主控看了lowK两variant，rear_facing顶部覆盖更接近观察；其各K正区FN0.037–0.055%，但背景仍未知，未观测-X保持null，不能称完整轮廓准确或K校准。所有K保留作为下一共享A诊断输入。

手根A局部权重过渡的同角对照，thumb最小伸长在t=.6由0.069改善到0.576，原thumb/palm新增27交叉消除；骨位及distal关系不改，因此原同角pad位置不改。原大屈曲整体仍不合格。分解8状态无新增穿插但最近pad24.76mm，未假称pinch；当前新确定性可行搜索已启动，尚未得到最终接触接受。


### 首次真实GPU整合预览暴露尺寸错误（2026-10-20T03:29:31.242201+08:00）

GRAQ1将原nonnested检查移到ai0排除前后，独立8CPU及边界反例通过，主控QUALITY后canonical提交global_scene_native_preview_profile_v002。实际GPU4完成3原生RGB及mask，exit0，耗时103.401秒，峰值4522MiB。主控实际看3张图全部灰帧；wrist visible面积0而amodal76803/77099像素，明确视觉拒绝，不能因工程success准入。

CPU firsthit审计69/69射线击中同一金属纽扣，3观察相机均在它内部。G1B原衣服重定位在helper返回后未更新depsgraph，最后sphere属性scale已设(.007,.007,.0016)但matrix_world仍单位scale，bake后成为直径2m巨球；整个衣片也被group-min支撑逻辑抬高约1m。该遗漏表明静态状态一致不等于物件尺度合理。原错误图/资产/执行profile保留，性能仅代表错误遮挡场景，不能复用为修复后场景。已授权原实现者先update并冻结所有source matrices再rebase，补6纽扣真实尺寸、独立主衣片贴桌及wristinspection目标firsthit检查，另建G1B/整合v005并freshprofile，禁止通过调灯或删除物体掩盖。

安装方面，v002纠正有向前臂初始化后残差显著下降但远侧仍系统超差。随后0优化width/centre耦合审计证明近侧锚定下w34mm、centre远移3mm可使540远侧RMS降至6–7px，同时严格保持近侧/C/源前臂世界位置；K1.1组合在所测12点全部18px内，仍只是未决定生产参数的工程候选。

可行接触搜索232计数候选、221actual评估后间距降至左9.044/右9.296mm，未达0.5–2mm。随后index根带其他指根排斥场的局部权重修复，8新+2旧实际对照通过，左原3个新增穿插消除、最小伸长0.2026→0.3163；同角指垫位置不变，仍未完成接触，运输方向margin仍小。不得将这些局部诊断当作生产手势。


### 参数模块计划细化与修复后实际预览（2026-10-20）

在原计划参数模块节内细化四项审核意见：hard约束/批准范围/可行覆盖配额分层；机身canonical8中心距离、未裁剪投影框与四类目标计数；有界确定性attempt、失败状态恢复及accepted manifest原子登记；动作父样本/轨迹分组和共享CAD基线例外。独立复核四项闭合，模块仍待全局校准验收后实现，未编造任何生产距离/角度阈值。原文备份和差异回执在parameter_plan_review_v001。

灰帧根因修复经独立SPEC与主控QUALITY后，canonical job global_scene_native_preview_profile_v003实际GPU4完成，89.328秒/峰5012MiB。主控查看三张全图缩略并按文件重算可见/完整mask：左76803/76803px、右76780/77099px，nonnested均0；源恢复报告通过。巨大纽扣导致灰帧的问题已解决，人体中立下垂姿态、皮肤条带外观及未校准桌布/衣物/光照仍不满足真实场景要求；不是原生细节验收，不准训练或量产。左安装诊断六候选已完成，只有f0L证据且径向分支歧义保留，等待独立审核和全局FK约束，不选生产A。

首次状态同步误用不存在的global_environment字段而KeyError；发生在状态文件写入之前，已按实际global_environment_initialization字段修正，不涉及代码、资产或运行结果变更。


### 共用人体前向与真实皮肤消融结果（2026-10-20，继续执行中）

新增action4_full_rig_fit及薄入口，真实35骨、3小姿态、6安装组合的共同E前向已完成。12单元测试、独立18记录重算、fresh Blender hand-only与root源码/overview审查通过限定工程范围；骨最大矩阵差1.247e-6、候选相机4.402e-7，hand-only相机/brace严格不动。neutral右机偏差0.679–1.019m并均在画外，左机零差来自锚定而非拟合。当前source实体band28mm与分析参考34mm分别记录；未提供的真实配合接口保持未知。证据full_rig_forward_v001、full_rig_forward_spec_review_v001、full_rig_forward_quality_review_v001。

对image轨迹与source左右映射另行只读核对：proper掌面朝前/手指向上时，source左/右拇指分别位于图像左/右，与f0实际一致；当前direct保留，未因命名/直觉交换。原片f0定性可见证据在hand_mount_visible_evidence_v001；540掌面混合保持unknown，未编造3D法向或隐藏关节。

手根index局部修复通过独立实际两态复演与root有限QUALITY。v002各67/80计数状态将9.044/9.296mm降至7.630/7.886mm；仅沿反复改善方向的后续20状态上限验证将间隙进一步降至3.462/3.740mm，于原thumb opposition75°界停止。后者全部开销左14/20、右9/20，包含首次numpy.bool_元数据写入失败的左5次，未重置预算。root从12个actual mesh缓存独立重算最近点见证的重心坐标/法向/距离/polygon及允许角度方向一致；未独立重新搜索全表面最小值或全部交叉。两次均未达到原0.5–2mm组件目标，未完成衣物接触。证据hand_contact_feasible_search_v002、hand_contact_direction_check_v001/controller_witness_check.json。

皮肤新窄诊断经独立SPEC/rootQUALITY后，旧prepared spec因并行full-rig源码变化被正确拒绝；保留旧规格，另建v002冻结当前源并由canonical GPU5执行。实际两张3248×2464 RGB64/OPTIX对照61.471秒、峰3236MiB，原shader与状态恢复。root实际查看全幅和600×920原生crop：只将photo颜色输入改成源patch线性平均色后，周期横条带消失；两图都有明显多边形面片明暗。来源照片边界跳变为内部邻差约48/115倍，不能直接周期平铺。常量只作消融，不是最终皮肤。证据skin_appearance_audit_v001/comparison_gpu_v001/controller_visual_review.json。

进一步实际重开确认从原human bundle到integrated四级，控制15043面和求值60268面全部flat，0smooth、无custom normals/sharp edges；求值每面loop normals严格相同，前臂相邻法向中位3.70°/P95 11.05°。这为残余面片着色提供直接原因，仍待最小smooth衍生修复和实际图像验证。UV存在但跨4个U tile，不能直接假定单贴图对应。以上均未恢复数据量产或训练；新2000接受数仍0。


### 平滑与周期化皮肤实际修复、手表面约束就绪（2026-10-20，整体任务未完成）

在原整合流程上新增显式body shading合同，衍生v006控制15043/求值60268面全smooth；source/G1/finger与原geometry/骨/权重/CAD8不改。一face改flat能够被新检查拒绝，独立SPEC与rootQUALITY闭合。曾对外部合同临时追加附注造成短暂分叉，已逐字节恢复contract及preview到前快照，并重开核实内嵌/外部仅正常saved_asset差异，记录保留。求值UV最大历史差9.799e-5仍未解释，控制UV/求值几何相同；当前皮肤Object映射不消费UV，未伪称UV逐值不变。

主控canonical GPU4实际smooth-only三视角完成92.113秒、峰3252MiB。实际看三图及原生600×920前臂crop，多边形面片感消失，raw photo条带仍在。左右visible/fullamodal四mask与v005文件SHA逐字节相同，角点/可见性/相机/排除记录一致。首次root比较误在普通Python调用Blender-only helper而缺bpy，写入前失败；改用更强PNG原bytes/像素比较，没有改数据或放松检查。

来源照片数值候选采用Moisan P+S线性RGB单次分解，来源/论文/作者代码/残差均保留。原接缝/内部梯度RMS比x14.64、y38.46降至float约0.692/0.537；无色域裁剪，均值误差1.77e-14、重建误差2.78e-17。PNG8量化增加高频约39%，因此仅作预览；最终待审输入为32bitZIP Linear EXR，新Blender进程RGB逐位等于float32参考。独立SPEC与root数值QUALITY通过；不称albedo恢复或生成真实微观肤纹。

新v2 optional derived_skin引用和build_v007保留原PNG/photo manifest与v1 raw路径，只变活动skin图像输入为已审EXR并显式Linear，原helper未改、腕带保持sRGB。来源装配、geometry、35骨/权重、smooth、光照与48点投影保持；错sRGB、假Linear PNG、断链接、Alpha冒Color等实际反例拒绝。独立SPEC/rootQUALITY后canonical GPU4三视角88.461秒、峰3252MiB；root原生三版本对照确认面片和明显周期条带均已消除。四mask与v006逐字节相同，8点/框/面积/相机/排除记录一致。对照图在global_render_adapter_v001/periodic_skin_native_v007/skin_native_comparison.png。仅局部外观修复通过，不签全场景/训练准入。

新增人工可见手轮廓v002已逐张原生叠图复核；不填隐藏腕关节、不把骨尾当物理指尖。真实finger源9腕X/Z姿态导出18表面缓存，在K1.1 L1/R0/R1共27前向保留thumb子网格投影与完整手空气轮廓的区别。11例缺thumb-air为null，独立6缓存actualBlender/3例native mask重算一致；root有限QUALITY通过诊断输入。中立L1/R0/R1 thumb投影RMS123.29/345.14/84.63px（单向近似，不是正式训练分数）仍显示偏差；腕/手指姿态未知，不能唯一归A或比例。

只读hand变换链审计对实际Blender24记录最大native误差0.001114px，纯HAND子树表面约0.29µm；PALM来源、parent、米制及Ty±3mm未发现实现问题，实际表面重表达误差2.22e-16m。不能靠E修正与E抵消的capture手形，应在固定尺寸下继续安装与腕/指姿态联合相容性。解析双臂模块6初值精确上臂长度闭合，但仍超原25°且未做大姿态蒙皮验证/未优化；同心异半径及近反平行数值边界已修复并经独立复验、root有限QUALITY，六历史初值精确保持。

新2000训练接受数仍0，未启动训练。下一关键工作是可见手表面/brace/body证据下的联合相容性校准，随后实际支架与衣物接触、多帧环境验证，再开放有界采样；不再把只贴合局部腕带或只满足长度闭合当整体正确。


### 参数模块补充稿合入（2026-10-20T10:15:17.274686+08:00）

SSH连接恢复并确认xinglong-a6000及授权根。将已经独立SPEC复核、主控一致性检查通过的参数发布与覆盖验收补充稿，合入现有计划的参数模块节作为第8小节；原1–7节和历史计划保留。稿件SHA167e249711e1de05dda02eff2c322d20e9c564faaa10e9da4829a0c8f4f26cca。新增候选锁定→预览验证→生产冻结、必需覆盖全部达标再发布整批训练清单，明确固定尺寸/安装关系、未知非增强、2D遮挡与3D穿插区别、逐项验收及失败恢复。审核与原文快照在outputs/action4_realism_reassessment_v001/parameter_plan_operational_review_v002。模块实现和真实生产范围仍待校准，不因文档合入恢复v015或训练。


### 固定安装手姿态诊断结果及补充观测复核（2026-10-20T10:19:02.420590+08:00）

固定K1.1/R0候选的实际手姿态实验完成52状态、77提案、6次接受更新，按预定10/5/2度扫描结束，未耗尽100状态预算，不代表收敛或全局不可行。两曲线MSE下降33.50%，但thumb RMS仍470.153px，outer从78.009恶化到105.980px。独立结果复算通过；主控本轮实际查看同尺度起终全手图，整手仍明显高于实拍并侵入空白墙面，判定真实姿态失败，不接受K/A/尺寸或生产数据。controller_quality在hand_pose_compatibility_result_review_v001，原失败完整保留。

hand_mount_visible_evidence_v003的左右源/叠图与两张原生墙面crop已主控逐张实际查看，source及全部交接hash匹配，两墙裁片逐像素等于原frame0。6条非拇指短开放弧贴合可见边，两空墙区域无手；另立controller_review.json允许限定诊断，未修改冻结观测。仍不指定手指身份/隐藏轮廓，15px为工程估计，非正式训练标注。

独立hand_surface_semantics_recheck_v001对0/15/30/60/90的报告倾向斜视掌/指腹，但没有排他证明纯正掌面或source左右映射。此前direct映射对照仅在掌面朝前假设下成立，不能升级为实拍已确认；不因此自动交换左右或镜像。待完整可见手形和多帧约束共同确定。当前仍没有通过的新2000数据或pilot训练。


径向安装相位调查已独立SPEC核验246身份、代数/brace残差、24旧曲线指标和3例实际三角mask；四个物理代表完整皮肤体积/外域通道复核。root实际查看210度state045：thumb73.579px但outer210.268px，非拇指整体仍远高于实拍，未接受A。六相位通过brace条件不代表完整物理/手形通过；物理仅30/210度×state2/45四例。survey hand_cases内physical_check为旧NOT_YET_CHECKED，真实四例终态在physical_checks与physical/results.json，后续消费需明确绑定这两个权威字段，不能据旧字段误读。审核在mount_radial_phase_spec_review_v001，原输出未改。


### 用户收敛优先级（2026-10-20T10:28:13.691990+08:00）

用户明确手指不重要，聚焦DJI与腕部固定安装关系、移动和大小比例。已停止扩展精细指缘/掌缘拟合，取消尚未执行的7个HAND局部X候选；未运行新姿态或GPU。已完成的新辅助指标两代表仅诊断；独立SPEC发现近深度tie绘制顺序反例，故不通过，不进入后续拟合。现转已存在机身/腕带/前臂观测下的跨帧固定安装验证，保持固定尺寸和统一rig，基本手姿不再是精细拟合门槛。


用户追加八角点可收集定位与训练链同步核验，已将并行工作与验收写入同一plan。当前启动的是front CAD forward、actual腕部FK适配准备及既有8点数据流CPU审计；未宣称新位姿/标签已接受或已开始训练。原v015接受撤回仍有效。


### 腕部主线与八点接口实际进展（2026-10-20T11:01:56.378848+08:00）

已独立审核f960R前壳/镜头真实CAD前向：c11外92/内94、c13玻璃24边来源真实，六初值8点齐次投影差≤4.55e−13px。root实际看primary原生raw/CAD/8点三联，方向基本合理但轮廓仍偏；三rim无法在18px人工精度下唯一辨识，保留假设，未接受标定。下一有限拟合准备采用9×100总900nfev而非扩大总预算。

实际腕部neutral适配完成1状态，35骨与FK误差约1.13e−6、骨长误差4.19e−8m；两实例×两表达链32角点完整导出且全部出画仍保留。源A实际core与候选A数学投影严格区分，无匹配RGB不发布标签。独立SPEC/root限定通过；五姿态路径未运行，正在准备endpoint的形变和相对baseline碰撞检查。

角点数据链审计CPU重开原v007资产，3视图×2物理实例重新投影，原保留两实例差<0.0005px；root实际看两原生8点叠图，覆盖完整目标core而非屏幕局部。actual loader保持隐藏八点监督，完整框误换可见框反例被拒绝；这些是渲染标签/接口证据，不是真实视频预测精度。发现动态pose快照缺失、excluded丢8点和结构validator无法识别同集合点序置换，现正补metadata及有身份约束的逐点重投影校验；未改诊断为训练accepted。

刚性解码审计证实现8独立热图无SE3约束；CPU人工高峰反例可自交或重合，不是实际ckpt性能。按用户最新优先级，raw8精确定位/锚定为主，rigid后处理仅默认关闭的可选有界改进，不新增pose head/预算、不阻塞先导；改善须实际raw对照，几何失败回退必须明示。


八角点导出补全已完成独立SPEC及主控QUALITY：新purehelper/global_render元数据与v2配置绑定自身/加载helper路径及SHA，13纯测试+9原入口测试+15独立反例通过；独立freshCPU Blender核6实例48角点，2kept/4excluded旧字段保持，formal loader继续拒绝诊断数据。初次跨NumPy精确浮点比较失败已定位为≤2.22e−16m/4.55e−13px派生误差并留档；仅派生数字有限容差，原身份/pointorder/源矩阵/K/旧字段/type仍精确。当前工程接口通过，不代表新2000场景可训练。详见canonical8_export_completion_v001/controller_quality.json。


### 模块化转向及f960实际结果（2026-10-20T11:27:44.497282+08:00）

用户明确独立部件建模后统一装配，且不影响遮挡环境的手指/衣物操作不必细化，已将该规则置于plan首部。当前fullhuman endpoint只保留诊断：1actual已完成，FK/腕连接/机身尺寸正确，但435新皮肤polygon交叉与部分形变失败，候选右core有370surface pairs；root缓存定位其相关皮肤全为静止torso_root权重，进一步确认位于头前侧，未简单归成腕部安装错。不再扩全身优化/权重修复。

f960R的9body起点实际完成，session87632终态exit0，144solver nfev/930含FD残差调用；9组rim最大误差≤4.857px、两可信机身边actual union最大≤6px、已标确定positive漏覆盖0，前镜头朝向符号合理。root实际看K.8125内rim的body/8点与独立brace叠图：机身贴合改善，但9组旧共享A外推腕带全失败，最大点误差65.94–140.98px（门槛18），因此不能接受装配。全部K/rim保留，未把优化成功当精准标定或GT。独立结果审计进行中，准备三帧固定C下的共享A重估。


### 模块实际构建、冻结与组合接口（2026-10-20T12:10:57.109728+08:00）

左右source-derived腕前臂静态库build_v003已独立SPEC及rootQUALITY通过，每侧8663V/17280tri/8640原poly，唯一44edge肘口，保留源几何/材质坐标。v001临时集合贡献物检查失败、v002 JSON tuple/list比较失败完整留档；最终v003不覆盖旧资产。root实际双库导入发现材质Image ID重名，基于完全相同的冻结graph共享同一已核材质后双库验证通过，原字节不变。

实际34mm右带体独立几何正确；首轮SPEC发现parentInverse缩放和活动weave断链漏检，作者仅修验证器并另建v002。原同反例及独立窄改动反例均被拒绝，新旧几何与材质状态一致；两侧rim与解析点差1.815e−9m。初FAIL保留，最终独立修复SPEC+rootQUALITY通过模块初始化。支架/白模块/A/接触尚未验收，未发布生产。

冻结登记v002现保护15文件，五类部件/来源摘要核验通过。冻结了DJI/原8点、已验证source/纹理和本轮模块库；比例与安装校准继续单独推进。全场景训练接受仍false。


### 冻结与模块装配核验更新（2026-10-20T12:52:25.926855+08:00）

冻结登记v002的15文件已再次核对一致。modular_assembly_probe_v001已由独立SPEC及主控源码/证据复核通过接口范围：原DJI网格/材质/尺寸、共用W、C=BA、canonical8及真实Blender观察相机投影保持一致；独立实际八点重投影最大误差0.0000867px。此场景只用于诊断，仍有117对腕带—皮肤表面相交、安装支架未加入、标定未通过，不能用于训练。缺件盘点继续复用原资产。

缓存装配关系的九角度有限检查找到一例局部候选：固定Ty=-6mm，绕带+Y=-5deg，Tx=+0.5370859mm、Tz=-0.8072649mm，72边内孔最小余量0.1mm、完整缓存band-skin及core-skin表面交叉均0。未应用，独立复核进行中；角度触边界，不追加搜索，不把proxy当实测腕点。若标定A变化，core-skin必须重新检查。


### 联合诊断v003结果及主控视觉检查（2026-10-20T12:59:33.209595+08:00）

v003单初态运行实际68 solver nfev、1364次含有限差分调用、303.50秒，以ftol终止；session27117已terminal exit0。独立重算全部后验完全一致，并用独立Rodrigues/原canonical顺序核24角投影差≤4.55e-13px。f0最差局部边18.8358→9.8273px进入12px，三帧brace分别13.2456/8.4863/9.9881px均在18px内，方向21.6643/22.2350/16.5696px均在40px内。f0 FN0.2144%、已知背景超出1.5227%均在原5%内；不能说所有指标改善。

独立SPEC及rootQUALITY仅签收有界诊断：主控实际查看0/540/960三张原视频对照，f0上左镜头凸起蓝轮廓仍高于可见实物，unknown区/完整外形仍未建立；f540无背景GT且旧FN门槛未定义；540/960参数触逐轴边界；这些帧均参与拟合，不能当独立验证。保留固定尺寸、原角序和所有门槛，不重启/扩大优化。后续仅以该结果作为实际模块组合检查候选。证据：joint_semantic_blocks_result_review_v001/controller_quality.json。新accepted合成数据仍0，未启动pilot训练。


### 原安装件及左侧资料复用盘点（2026-10-20T13:04:04.396829+08:00）

原v007左右各8附件mesh可按CAD固定（快拆/USB）、brace固定（白模块/球）、跨frame连接（link/cable）复用。后者依赖旧A，不能搬到新A后仍假称接通；只处理影响遮挡和连接外观的必要细节。球原固定childscale=.006对应12mm，不可直接reset1；旧28mm绿管不叠到34mm材质绿条；附件非目标。盘点没有派生资产。

左侧f0L已有3K×2安装候选及对应C，可与f0R在同帧/commonK诊断组合；f540L已有两环+轴和partialbody观测，但没有已审CAD曲线对应/C。R-only多帧审查不覆盖L。暂不让补C阻塞f0预览，后续跨姿态检查按现有素材做最少必要补充，不混f0L与f960R冒同一帧。原始canonical8全程复用。证据分别位于outputs/action4_mount_module_inventory_v001与outputs/action4_left_multiframe_readiness_v001。


### 三姿态实际模块和整机8点完成验证（2026-10-20T13:17:22.057506+08:00）

唯一一次CPU Blender采集成功应用0/540/960三姿态，未保存或改写冻结资产。四组件三角拓扑与源一致，独立逆实际矩阵回原顶点最大差2.78e-16m；实际角点与固定C最大差0.000257px，与Blender观察相机最大差0.000161px。实际新A/S下band-skin、core-skin、core-band完整表面BVH均0；三帧同局部几何得到验证。267个肘口代理采样均出画，不是遮挡证明，也不是连续开口可见性证明。源scene/冻结源身份不变、原场景合同恢复通过。独立SPEC与rootQUALITY已通过上述工程范围。

主控实际查看三张raw/geometry对照，完整0–7虚拟角点覆盖整机，而非显示屏碎片；粗自然手姿与实拍弯腕/捏取不同，前臂轮廓和真实佩戴仍未视觉验收，不能据此直接断言手尺寸错误，也不恢复精细指关节拟合。f0镜头凸起轮廓偏差保留。缺少必要快拆/连接件以及环境、材质光照/实际遮挡预览，仍不能生产训练数据。后续最小顺序为原安装件必要复用→f0双侧同帧同K组合→渲染与遮挡/尺寸复核→参数范围冻结→2000图与pilot。证据：outputs/action4_modular_relation_actual_v001/actual_threeframes_v001、geometry_previews_v001，以及outputs/action4_modular_relation_result_review_v001/controller_quality.json。


### 左34组件与佩戴核查的实际进展（2026-10-20T13:47:24.377521+08:00）

左原尺寸独立模块实际生成于data/action4_scene_assets/modular_left_brace_v001/build_v001。首次构建在保存后回读合同阶段exit1：6处材质links由内存tuple变JSON list，值/序列化字节均相同。根因实际只读验证后，主控另执行只读recovery，原validate、活动材质/packed图像、实际顶点面材质NPZ全等与原来源SHA通过；实际scale1.01被拒绝，改合同width仍不等。没有重建/改资产/补写旧成功回执。独立SPEC及主控QUALITY签独立组件初始化，当前冻结v003显式核18文件全部匹配。

f0双机同K缓存已核完整16点；左两A分支原identityS均124对band-skin交叉，9个固定roll中+3.75/+5两小位移S可行且独立全几何复算通过，但未应用/未签真实佩戴。用户再强调佩戴后，三帧实拍与粗手代理指出A/手姿歧义：左start1拇指侧更贴近f0，右整体pi径向替代仅f0改善而540/960变差；没有选取任何新A。必须区分固定相机安装和粗腕关节姿态，不能以0BVH或环贴合替代实拍。原CAD/尺寸/8序不改。

右安装件准备S上下文漏绑定被独立SPEC阻断；v002仅修为当前S乘原skin缓存并记录来源，15pure通过，未actualbuild。因wearing待确认，当前A下派生连接暂不构建。渲染接入已确认复用原RGB/visible/amodal/canonical8流程，仍需独立新模块scene合同及freshprofile。新accepted训练图0，未启动pilot。


### 佩戴检查后收敛到原局部腕控制（2026-10-20T13:55:39.399064+08:00）

原/Pi对照没有跨帧一致改善，root实际查看f0/f960三联图后未采用替代。已有right_hand控制及source模块映射可复用，±25deg guard保持；后续只准备f0三种局部Z整体腕掌状态，以分离neutral手姿差异与固定佩戴，不做指关节拟合、不改A或尺度。右mount的S上下文缺陷已独立SPEC/root关闭，真实wearing未验收因此尚不构建连接。左独立组件已冻结v003，其完成不等于wearing完成。


### 实际RGB与单Y腕姿的收尾记录（2026-10-20T15:15:41.555549+08:00）

原HANDZ三状态实际完成且复核：固定8/core/band不变，0新增交叉，原95poly/163tri仍在，±25 slab位移约2.541mm，未解决f0方向。新的Astra-Medium解析轴检查解释Z近沿视线；6.1Sol-High薄适配经独立SPEC/root后仅实际提取Y25一次，恢复成功。掌代理normaldot由-.059变-.453，thumb-palm dx由-289变+11.68px，band/core交叉0、新增poly0；方向改善但手仍高/伸展，root看三联图后仅允许下一组合诊断，不签wearing。源REST/尺寸/权重/原8不改。

模块RGB已实际GPU5运行三单样本，各RGB64+visible1+amodal1；v001/v002/v003分别约27.38/27.54/27.95秒，峰值3168/3168/4522MiB，均conclusive，当前没有运行中的GPU任务。最终v003墙三个块约126/130/136、127/131/138、125/129/135，对应raw119/124/132、123/125/131、118/122/127；前臂参考块118/100/85对raw109/94/73，剪裁归零。所有三次完整8和两mask SHA完全相同，可见与amodal-inframe234023px完全一致，标签覆盖整机。此为单视图外观初值改善，非实测albedo/灯光或最终场景验收，不据此推最终存储/吞吐。原图底部nativecrop实际格间距约70–80px，与初值77.6px相近，先前未量化的格尺度担忧已收窄，不错误据此改尺寸。

右mount六件已按既审v002完成一次CPU构建/同进程cleanreload，原源字节不变。36pair仅ball-band50、ball-link236、band-link1、link-quick26处在预先定义的名义接合区候选，区外/不明0；不当作机械测量通过。该接触上下文仍是neutral S，Y25接触将在组合前用已存cache核查。库保持outputs诊断身份，尚未加入生产资产。

用户子智能体默认已落实到remote AGENTS：新分析用Astra/Medium，新实现用6.1Sol/High、精简任务包。一个分析智能体额外产生的3张本地查看副本已由root仅删除其新建副本，服务器原件保留；后续包明确禁止本地图片镜像。当前仅Sol-High准备composed入口，旧高档实例不再续派。新accepted训练图仍0、未启动pilot；下一步同场RGB检查Y25+mount+v003光照，继续原计划。


### 实际组合RGB闭合与左单姿态取舍（2026-10-20T15:55:41.069631+08:00）

v001组合CPU在新法线分量阈值2e-4处失败，原记录保留。诊断确认连冻结neutral也会误拒；新v002恢复原库已有0.002rad最大方向角合同，并保留shape/finite/unit检查。独立9相关pure及单loop反例通过，主控审差异/源身份后一次实际CPU preflight通过，原8与旧预览完全相同。随后canonical GPU5实际3render完成33.462秒、采样峰3168MiB，源资产与临时状态恢复通过。原visible/amodal各234023像素、截断和遮挡0；完整目标mask处于原8投影bbox内，amodal文件与旧版本相同。root实际看原生mask面板和实拍同crop对照：完整机身框正确，腕带/支架/线缆已可见；仍有手掌展开、座体外观和前臂轮廓差异，只签工程组合诊断，不签真实场景/训练。证据outputs/action4_modular_composed_prepare_v002/runs/native_profile_v002/controller_result_quality.json。

左HAND+Y25经独立准备SPEC/root后唯一实际提取成功，35骨/左前臂/双core恢复，source不变、原8不变；99polygon/173triangle旧self集合保留且无新增self/core交叉，但新增28对band-skin表面pair，涉及6皮肤triangle/3源polygon。独立结果复核后root拒绝此姿态，保留neutral+左S3.75诊断候选，无额外搜索。pair数量不是独立穿透处数/深度或体积；原实际成功不等于佩戴通过。已看RAW/neutral/Y25几何对照。新accepted训练图仍0，未开始pilot。


### 左安装件实际失败与根因（2026-10-20T16:12:39.459579+08:00）

已实际单次CPU构建及clean reload左六件库，源固定件/尺度/材质和重载几何一致；36类完整碰撞检查中band-link44、link-skin97、quickrelease-white34共175对未解决，root明确拒绝组合使用。结果保留outputs/action4_left_mount_modules_v001/actual_build_v001/controller_rejection.json。独立诊断确认前两组最接近跨brace正负Z的77.7mm连接段，源矩阵链并无毫米级实现误差；phase-v002候选仅修附件基座相位，尚未实际验证。

此前同f0左neutral core/band/skin对右Y25三件+六mount的27对AABB全部分离，最小距离下界13.8996mm；此证据不含左mount，不能宣称双机全部不碰撞。批量接口继续按原plan/shard/validate/assemble适配，未生成正式数据。


### 左附件修正闭合与实际双机场景验证（2026-10-20T16:59:32.532561+08:00）

左phase-v002保留旧175问题并实际消除，随后只剩长rail的cable-skin66对。基于完整17280三角、Y区间[-303,17.835]mm选2731面，支持54.495650mm+原半径1.7mm+工程余量1mm，统一rail外移至57.195650mm；未改USB起点/lead、stationYZ、phase/A/S/其他尺寸。经独立SPEC/root后v003一次实际build/reload，36类pair unresolved0，来源及回读几何一致。新电缆路径是工程估计，不是实测；旧失败保留。证据outputs/action4_left_mount_modules_v003/actual_build_v001/controller_result_quality.json。

双机CPU v001实际发现Blender libraries.load原地修改调用方names list，使Object成为映射键。单库实际复现后只以tuple保存原名/list副本导入修复，旧失败保留。v002实际已完整导入且16点通过，但右ball-band两个BVH候选被原分类归unresolved；独立Decimal70核实际三角在同zone坐标分别分离3.09786/3.53413nm，映射数值误差约6e-17m。v003仅在原classifier前保留证书并排除按原1e-12仍严格分离的BVH候选，overlap/touch/近边界保留，zone不扩；原50候选=2分离+48保留。

实际CPU v003终态exit0，左右ID1/2共16原角点、source/材质/变换链通过，81跨侧无交叉、两侧各36内部pair无unresolved，源资产不变。root从v002实际contract另独立重投影最大偏差左0.000075529px、右0.000256992px；这只是几何链一致性，不是真实模型精度。实际结果outputs/action4_modular_bilateral_prepare_v003/runs/cpu_probe_v003/result.json。

整帧pixels接口完成0/左/右/双及excluded8保留、错误pass0负例拒绝、输出角色分流与异常恢复CPU/mock；真实GPU首次profile在render前被旧env.actual_K硬编码g1b_env_observer拦截，peak0、exit1，不宣称已出图。正在最小改用既验assembly_observer内参与原canonical导出，保留失败outputs/action4_modular_bilateral_prepare_v003/runs/native_profile_v003。当前新accepted训练图0、pilot未启动，尚待实际双机RGB/掩码后继续有界采样。


### 双机原生RGB与完整标签链实际闭合（2026-10-20T17:36:49.515656+08:00）

像素v003最小相机适配经独立SPEC后实际CPU pixelprobe通过，完整16点/遮挡范围/恢复得到验证。GPU native_profile_v004实际完成4renders，但Blender中PIL缺失导致末尾PNG校验失败。root另在.venv独立恢复诊断：实际scene contract与CPUpixelprobe完全一致、实际16点/矩阵一致、原canonical verifier通过、全部5PNG完整解码与尺寸/原DataAdapter默认检查通过；原job仍failed。恢复时首次把float mask传入bool-only API被拒，未发布结果，纠正为原>127二值规则后通过，记录在recovery目录，不改原输出。

已把运行边界修为child仅结构校验+pending record/render_receipt，host已有Pillow执行默认DataAdapter PNG验证及RGB/两visible/两expandedamodal的SHA/完整解码/尺寸/同run归属检查，全部通过才写最终diagnostic_report。原投影/8/mask/渲染数学不变，独立26CPU反例通过。新native_profile_v005真实端到端succeeded/exit0，4renders，66.691秒，采样峰3180MiB；host验证与最终scene/source检查通过。左visible146326/fullamodal147486px、遮挡0.786515%，右234023/234023px无遮挡，双侧截断0。4mask与前次真实渲染文件SHA完全相同；RGB重复最大差1/255、平均0.001692灰阶，非bitwise相同。实际legacy角点与恢复所用pure投影最大差0.00008644/0.00016054px，在原重投影容差内。

root实际查看当前16点标注原生crop及实拍/合成对照，两个完整机身均处于原8 bbox内，佩戴连接/两机间距可作为首个有界预览基底。仍有粗手掌展开、简化球座和实拍外观差异；不重新引入精细手指门槛，也不签全场景/标定或训练接受。证据outputs/action4_modular_bilateral_prepare_v003/runs/native_profile_v005/controller_result_quality.json。当前正式新训练图0，pilot未启动。


### 参数/消费基础接口与非零参考证据（2026-10-20T18:09:03.544324+08:00）

已实现sample_parameters(entry,config,source_context)与AppliedScene基础消费/恢复接口，分别14和7项CPU检查通过，尚未实际调用动态Blender。原几何/A/S/F/K不改，presence真正隐藏core/decal/专属mount并保留非目标skin/band。初始90%reference=delta0的错误解读在封存前已纠正；未知reference姿态范围保持null且拒绝正式计覆盖。当前wall/hardnegative基础控制存在，6族具体效果未完成，未冒称128已运行。源码和交接在action4_modular_parameters_prepare_v001及action4_modular_scene_sampling_prepare_v001。

独立视频包围范围审计核对原/demo2850帧PTS一致，7帧非框配准相关0.9965–0.9985；14个wholebody proxy的归一化宽0.0779–0.2063、高0.0653–0.1889、面积0.00598–0.03696，双机横距0.0884–0.3958画宽。root直接查看960/1200/1590共6个raw/demo整机crop并核全部输入SHA，签仅诊断成对布局支持；暗帧和8–12demo像素不确定性保留，单目3D量/K/A/S不因此获得GT。证据outputs/action4_reference_envelope_audit_v001/controller_quality.json。

条件capture平移/距离提案、coarse cloth effects和near/self/truncation几何effect正分工实现；尚无新训练图接受，未运行pilot。


### 七布局实际检查前的跨运行时来源一致性修复（2026-10-20T19:03:32.477848+08:00）

首次CPU检查在prepare阶段因source_context摘要不同退出，实际布局0、渲染0；保留outputs/action4_modular_layout_actual_v001/runs/cpu_probe_v001。逐字段比较仅9个base_B派生浮点数不同（最大1.1102230246251565e-16），84个来源身份无漂移。采用固定Python双精度运算顺序生成身份metadata，不舍入、不忽略字段，实际bilateral矩阵运算不变。两次factory prepare-only验证后host/Blender上下文全字段精确一致，33项pure检查通过；证据outputs/action4_modular_context_portability_v001/preparation_handoff.json。

下一步仅刷新版本化布局/执行入口身份，原7个C/B与布局求解结果保持；独立复核后一次执行0/540/960/1200/1590/2250/2550的实际scene、完整16点、动态碰撞及恢复验证，再做RGB对照。正式2000张接受数量仍0，未启动pilot。


### 前臂遮挡来源语义的纠正（2026-10-20T19:05:48.333492+08:00）

此前将f1980直接概括为“自身前臂遮挡”的记录证据不足。对1950/2010及1970/1975/1985/1990邻帧的实际查看未能明确分离同侧前臂皮肤遮挡与黑色安装件/腕带遮挡，因此该归因当前标为未确认；不能将黑色配件当皮肤，也不能宣称整段视频不存在前臂遮挡。1980/2820本轮未打开，但相关邻帧已用于分析，不能宣称该片段完全未见。证据outputs/action4_modular_geometric_effects_prepare_v002/raw_semantics_gate.json及nearby_semantics.json。

保留self_forearm_occlusion场景能力要求；已有shaft_y-0.06_edge+1只作为固定安装关系下的工程增强预览候选，42条有效稀疏射线中25条前臂优先、16条本体可见、1条安装件优先，不能代替全像素mask或真实视频频率。原8族覆盖要求不变，未经实际渲染不能计接受；新2000计划生成前须公开登记精确场景配额，未证实来源的工程方向只计轻微扩展，不沿用旧v015的337项数量后又谎称均属90%参考。原1600/200/200和0/1/2实例配额保持。


### 按用户新截图纠正佩戴基准及参数层次（2026-10-20T19:16:38.815760+08:00）

用户明确指出当前相机相对腕部径向偏转偏大，应位于腕正下方或仅轻微偏转，图像左侧掌背方向疑反。保留已好的相机几何/尺寸/纹理/原canonical8；旧A/S是诊断装配候选，不是已验真实佩戴，允许独立新版本修正。暂停按旧A扩展渲染和训练准入，原结果保留。root再次实际并排查看raw0与native_profile_v005：合成掌面更侧立，图像右拇指相对腕中心的左右关系与实拍有差异。单独thumb-minus-palm标量不能签佩戴通过。

下一最小诊断：依据源掌侧/背侧法向，每侧最多两个局部径向装配候选（总4），仅缓存前向，不优化/Blender/渲染；保持C/K、单位尺寸，参数化A的径向相位并用B=C inv(A)保持机身投影，先固定S避免同时改变多个关系。检查掌面法向、拇指相对腕中心、前臂轴、腕带/机身整体位置及几何关系。候选角度属于源轴构造而非实测，不能拿原A的约90度相位差直接声称实拍修正角。安装件仍需新A下复核及必要route适配，不能复用旧穿插通过结论。

统一采样模块必须分三层：固定资产尺寸/canonical8；经多帧核对的左右安装基准及小幅安装偏移；整套腕-相机共同运动、双机相对关系和环境参数。安装偏移在同一连续序列/同一装配实例内保持一致，不能逐帧抖动或让相机独立漂移；新基准确定前不虚构具体角度上下限。参数范围合法仅是必要条件，每样本仍需几何/碰撞/视锥/完整8点/真实mask及场景效果检查，不能声称范围内任意组合都可靠。

本轮7布局CPU检查已实际进入首case0，动态合同与16点/0render探测通过，但恢复后对历史base合同的独立检查失败（RIGHT state mismatch），因此该case未整体通过，其余6未执行。原日志在outputs/action4_modular_layout_actual_v002/runs/cpu_probe_v001，正在定位具体字段；不放宽比较、不重复原7、不与本次佩戴问题混为一谈。


### 佩戴径向候选的实际缓存接触检查（2026-10-20T19:27:59.447096+08:00）

每侧掌/背共4个缓存候选已保存outputs/action4_wearing_revision_v001，root已实际查看原帧与两掌侧投影对照并核C=BA、正旋转、同侧core顶点/拓扑完全不变。两掌侧候选纠正掌面/拇指的粗方向，但手掌整体仍比实拍竖直，未签完整佩戴。源proxy对齐角不是实测范围，不能作为后续±90度随机增强。

单次CPU纯缓存153对检查：左右各36内部对、81跨侧对；跨侧表面候选0，左右core-skin/core-band/skin-band均0。保持安装件world位置的显式相位假设Bnew P=Bold允许复用固定几何；新变化关系中发现right skin-cable44对，须修必要线缆路径，不能隐藏/缩细线缆或直接接受。固定硬件的498个BVH候选另列，其中29个triangle-ID相对旧world记录变化，坐标原点/float32候选敏感性仍保留，不把原设计接合全判新穿插或自动放行。皮肤/机身网格非全封闭，不能据表面0交叉签完整solid包含验证。证据contact_probe_v001/report.json。

后续顺序：只修有证据的right cable局部路线并复核；用原bilateral初始化后建立显式新佩戴装配层，冻结相机mesh/尺寸/原8不变；再做新基准实际RGB对照和必要粗腕屈伸，之后才冻结安装小偏移与整腕运动范围并接2000样本流程。当前训练接受0。

此前restore误报已唯一CPU观察定位：fresh premutation与restored完整state精确相同；历史JSON与runtime仅48处shader links list/tuple表示差，canonical值全同，未发现实际恢复漂移。仅观察副本恢复links类型后原验证通过；正式类型规范化patch仍待整合，旧失败不改为成功。证据outputs/action4_modular_restore_diagnosis_v001/observation_result.json。


### 新佩戴基准的单一线缆修正与可复用装配层（2026-10-20T19:32:35.767146+08:00）

右皮肤/线缆44个表面候选集中于原控制折线[4,5]长段。按完整新右皮肤17280三角在原mount坐标系的全路线Y slab支撑，复用已有左rail算法，得到单一外移候选X=-50.447793995674mm（原-40.699144427347mm）；仅控制站公共rail X外移9.748649568327mm，USB端点/首lead/全部Y/Z/1.7mm半径保持。余量1mm为工程约束非实测，AUTO曲线实际扫掠仍须检查。已授权一次CPU构建/cleanreload及新cable对另外17件完整检查，输出新outputs/action4_wearing_revision_v001/right_cable_actual_v001，失败不扩搜索；不修改冻结asset、不GPU/render。

装配实现复用原bilateral初始化和原完整验证，新增显式wearing层：P=Anew inv(Aold)，Bnew=Bold inv(P)，skin/band根按新B、core局部按新A以保持C/8；安装件父级记录相位P，因此原world位置可以保持。只允许独立新cable实例替换已检查路线，原材质/半径/USB接口继承。新装配合同从变换前state解析预测，禁止以变换后snapshot反定义期望；保存恢复原状态，仍用原验证和规范表示的历史合同验证。两侧相同语义但独立参数，不镜像右手替代左手。新代码准备/pure/独立复核后执行一个实际新双机CPU检查及GPU原生对照，再确定后续参数范围。


### 新佩戴原生结果与角点数据流专项核验（2026-10-20T20:05:37.488268+08:00）

新wearing base已实际CPU通过完整153对、16角点及原内存/typed历史恢复；GPU5 wearing_palmar_f0_native_profile_v001原生4次render成功，exit0，107.7564秒，采样峰值4522MiB。host实际解码并核验RGB/2visible/2amodal共5PNG。root查看原片/新RGB/16点12边叠图，掌面和外侧拇指方向改善，完整框包住DJI机身；手掌抓握与raw不同，按用户要求，未影响目标/遮挡的细指差异不重新加入关键路径。此为单f0诊断，不签所有场景或真实视频预测通过。

角点证据：CPU/GPU16点数值完全一致；两实例全部visible及expandedamodal正像素严格处于8点投影凸包内，越界像素0（不只是轴对齐bbox）。独立由mm→m、实际物体/相机矩阵/K重投影，对训练消费corners_px最大差左0.00007978px/右0.00017485px。native→ROI256→HM64→native连续逆变换误差<=2.27e-13px；理想热图经过真实top20离散解码最大3.523nativepx，是量化/解码偏差，不冒神经网络预测结果。当前hidden角点仍全部valid；将corner_visible全false不改变target/valid，16通道CPU loss梯度皆非0。证据outputs/action4_wearing_dataflow_audit_v001/report.json及native_profile_v001/root_corner_mask_audit.json。

实际数据流缺口已明确：原backend patch尚未应用且仍调用旧base，不能继承新A/B/P/cable；候选formal=false而旧patch正式接受要求true导致晋升不可达；实际diagnostic默认dataloader拒绝，训练配置四manifest不存在。因此下一步直接落实源码：local负责原sampling接新wearing factory及0/1/2实例、恢复/ID正确性；left负责沿原synthesis/backend应用最小接入和严格可达状态机；完成代码后由local统一发布新配置/source闭包，旧来源/失败保留。正式accepted由独立已审generation release、固定plan、实际128和全部record验收共同决定，不能改候选标志自声明通过。6pendingfamily仍阻断生产，原2000/split/presence/group不变，不创建伪正式清单。


### 用户调整双机比例：2000张的92/5/3精确配额（2026-10-20T20:18:22.557216+08:00）

用户要求两台相机同时存在的合成图占90%以上。新执行比例设为2/1/0=92%/5%/3%，以满足高于90%且保留少量单机和困难负例。

| 划分 | 总张数 | 双机 | 单机 | 零目标 |
|---|---:|---:|---:|---:|
| train | 1600 | 1472 | 80 | 48 |
| eval | 200 | 184 | 10 | 6 |
| test | 200 | 184 | 10 | 6 |
| 合计 | 2000 | 1840 | 100 | 60 |

总量、三个split、group防泄漏和8family范围保持，hard_negative只消耗60张零目标配额，不能另加旧100张或恢复旧80%双机。新recipe.instance_weights按0/1/2顺序为[0.03,0.05,0.92]，planner/参数quota_contract/assemble分别核exact counts；单机覆盖左右两侧。当前只更新计划，不创建真实2000计划或称已有1840图。新参数发布必须绑定新entry authority，不能继续用旧v015 entries只改统计标题。最终双机占比以实际通过验收的两实例ID/掩码/完整8点记录计数，不能把一台忽略或缺失的样本充数。场景参考支持/轻微增强90/10是另一维度，不能与双机92%混淆。已有历史plan、失败数据和检查记录不改写。


### 用户调整正式测试集为全双机（2026-10-20T20:36:23.586086+08:00）

最新要求“测试集只保留双机”立即覆盖此前test184/10/6。训练/验证的数量和92%双机保持，新2000如下：

| 划分 | 双机 | 单机 | 零目标 | 总数 |
|---|---:|---:|---:|---:|
| train | 1472 | 80 | 48 | 1600 |
| eval | 184 | 10 | 6 | 200 |
| test | 200 | 0 | 0 | 200 |
| 合计 | 1856（92.8%） | 90（4.5%） | 54（2.7%） | 2000 |

planner及参数契约需支持显式per-split配额：test权重[0,0,1]，不能把全局aggregate比例再次均摊到test；首批场景覆盖保留逻辑不能强迫test生成hard_negative。8family在整体数据/检查中覆盖，test只包含合法正目标场景，困难负例仅占train48/eval6共54张。正式test每张须实际两实例有效、ID1/2和各8点完整；缺失、ignored/unobservable或不满足原可见支持规则的实例不能凑双机数。

当前运行的none/left/right/both四模式属于工程接口诊断，其样本不发布为正式test，不受test全双机分布限制，也不计入2000配额。当前source/session保持直到该次诊断结束，修正补丁先在outputs准备再统一应用与重绑；旧v002政策、计划及日志原样保留。新policy_v003已落盘，但未生成任何新2000数据或正式manifest。


### 四种目标数量的实际数据流与test全双机实现（2026-10-20T21:00:26.941609+08:00）

四presence CPU及GPU实际诊断均完成。none/left/right/both分别导出ID集合[]/[1]/[2]/[1,2]及0/8/8/16角点，实际GPU调用1+3+3+4=11，host完整校验1+3+3+5=12PNG，每case双层恢复成功。root实际查看四图，零目标无DJI、单机只显示对应侧机身、双机同时显示，腕带/固定底座保留；白模块保留在scene，是否被腕遮挡另计，不因肉眼未见误称删除。GPU5 job wearing_presence4_native_profile_v001 exit0、370.2795秒、采样峰值3180MiB；仅诊断，不纳入正式test或2000配额。证据outputs/action4_wearing_presence_actual_v001/runs/native_profile_v001。

GPU终态后已一次应用经过独立SPEC的combined+parameters补丁，旧源完整快照在outputs/action4_testsplit_support_apply_v001/source_before。实际canonical backend14项CPU通过，新采样release_v004真实host prepare通过117来源；相对已渲染v3，W/K/A/B/S/P、环境和隐藏策略逐值相同，geometry8a54未改，不重复渲染未变几何。

正式晋升现在按原10%可见比例/16个full-input像素规则，从host已核mask面积与letterbox仿射重算支持度，拒绝ignore/unobservable、低支持、伪造支持值或缺visible bbox，不能按ID数充双机。已把四张实际native记录送入新增support guard，返回有效IDs与真实presence一致；这只验证guard，不授真实accepted数据。

新per-split权重已接入原planner及backend：train48/80/1472、eval6/10/184、test0/0/200，总54/90/1856。首8family覆盖不再强塞test负例，旧无override计划保持相同行为。新policy43cda生效；新正式recipe/plan、实际generation review和accepted数据仍未生成，6pendingfamily继续阻断生产。后续以新palmar几何重新生成有限layout提案，并实施桌面/衣物等效果，不沿用旧佩戴的碰撞/遮挡结论。


### 首个桌面候选失败与有界解析修正（2026-10-20T21:22:04.139055+08:00）

f540 image-conditioned base+左foreground cloth单次CPU已执行，0render，未进入正式数据。原60度桌面与左band/skin/cable/white产生1073项表面候选/box内点；cloth没有新增冲突；动态/原场景均已恢复。原始failure在outputs/action4_table_foreground_actual_v001，不能因纯测试通过签场景成功。f540 LEFT方向来自RIGHT f0工程fallback、RIGHT取960方向，明确不是实拍同帧3D。

只调整桌面倾角的一个解析候选：固定table1.4×.65×.025m、top backedge Z=.8及像素v616、25mm物理格纹、全部actor和cloth；根据18组件既验缓存和f540刚体delta，计算每顶点正面半空间2mm工程间隙要求，在预声明60–85度区间取最大 atan2(dy,dz)+asin(.002/hypot(dy,dz))，得到68.8815246941度。该角度为工程几何解，非实测桌角。代码9732不变，仅新增配置与derived notes，先窄复核，再唯一一次实际CPU重新核完整table/cloth/actor/wall接触、effect remove及原场景恢复；不降低碰撞门槛、不缩相机/手腕/桌面、不自动扩大搜索。实际像素遮挡/桌面覆盖仍待后续RGB，6family仍pending。


### 修正桌面候选的实际CPU结果（2026-10-20T21:34:54.876947+08:00）

唯一68.8815246941度候选在outputs/action4_table_foreground_actual_v002实际CPU通过：新table/cloth与active actors/wall共43对、cloth self与box内部检查无未解决项；base/effect两次真实像素几何探测的16个角点坐标完全相同，0render；effect移除、wearing及原场景恢复全部通过。原60度失败保留，不改为成功。

该结果只证明当前f540候选的几何/恢复/角点合同。桌面实际可见覆盖、cloth是否真正减少目标visible mask、背景衣物范围与RGB真实性仍未签收。下一受控native对照固定同一f540源：base无effect与foreground新table+cloth各RGB/visible/amodal共4render，分别fresh profile，root可在GPU5/6并行。保存不同诊断entry并明确共同parent参数ref，不冒真实同帧3D或正式2000条目；最终用实际mask变化判断遮挡，不能用10%bbox控制值冒真实遮挡率。


### 2026-10-20T22:22:22.580701+08:00：f540双原生渲染完成，外观拒收

base及table+cloth两组原生渲染已产出；实际掩码对照LEFT少1867可见像素（2.24097%），RIGHT不变，两台canonical8及各自完整amodal文件一致。该对照仅归因于组合效果，不单独归因cloth。root重看真实对照图，发现RIGHT相机可见面反向、整体腕部姿态不符、布片悬空；因此外观验收失败，禁止进入训练。证据`outputs/action4_table_cloth_native_prepare_v001/runs/root_visual_rejection_v001.json`及同目录`root_combination_mask_comparison.json`。下一步修正候选朝向与AABB粗筛的职责，再做真实几何验证；不改冻结机身/点序。


### 2026-10-20T22:38:12.311337+08:00：已应用朝向与数据流补丁，首次实际检查止于数值比较

经独立SPEC、QUALITY及root检查，五个源文件精确合并见`outputs/action4_orientation_geometric_apply_v001/application.json`。root实际active allocator核验2000配额通过；fixture采用历史family权重，只验证分配算法，不是正式新计划。全新source closure及f540候选已建立，物理字段与v004完全一致。首次Blender CPU在创建场景前报`Observed-face evidence failed or changed`，结果见`outputs/action4_layout_orientation_actual_prepare_v001/root_first_attempt_failure.json`；未运行实际碰撞或渲染。正在针对host/Blender派生浮点比较做有界修正；源SHA、原始矩阵及离散字段不放宽。


### 2026-10-20T22:47:09.214482+08:00：浮点兼容修正后实际几何拒收

layout-only派生数值比较修正已独立SPEC及rootQUALITY通过并应用；host/Blender元数据各6/6测试，固定源、选择和原几何门保持。沿用相同factory/context、重生参数，物理参数与首次候选完全一致。实际Blender对81跨侧+左右各36组件对进行检查，发现493对未获允许三角面相交：skin__skin204、skin__usb_connector24、skin__cable70、band__usb_connector79、band__cable116；同侧无违规。原实际BVH正确拒绝场景，0渲染，未到canonical16 probe。证据`outputs/action4_orientation_actual_v002/root_geometry_rejection.json`。LEFT直接借RIGHT540朝向仅满足背面符号，并非真实左侧姿态；下一步依据原视频及已有body/wrist证据独立优化LEFT540，不能改尺度/删几何或用反面朝向规避碰撞。构造函数未返回，因此probe的restored=false只表示未拿到句柄，不可误称实际恢复失败；详见证据限定。


### 2026-10-20T22:59:51.514818+08:00：依据用户澄清调整主线

停止将单帧精确重建作为生产前置。原视频改用于场景覆盖和范围估计，已验证模块与真实几何/标签检查继续复用。LEFT任务已转为粗方向与合理装配候选，body像素差仅作诊断；metadata接口兼容任务继续短期收尾。此前493跨侧三角相交是物理无效，仍拒绝；与原视频姿态不完全相同本身不再是失败。计划顶部及新增参数化合成章节已同步，正式2000尚未生成/训练尚未开始。


### 2026-10-20T23:38:08.521643+08:00：独立装配预设与滑条模块实际六例通过

已应用`action4_modular_pose_controls.py`及sampler两处显式schema分支，保留原BVH/像素/恢复路径；Astra规格检查及root代码检查通过。新source closure与六例参数在`outputs/action4_pose_controls_actual_v001`。实际Blender三进程、各2线程、CUDA隐藏：f0双机、新table预设none/left/right/both、table组合滑条全部exit0。none28、single各78、dual各153组件对，均0未解决表面交叉；物理ID分别[]/[1]/[2]/[1,2]，0/8/8/16点，全部wearing与original恢复通过。root独立由实际相机/物体矩阵和毫米转米重投影，误差<1e-6px，实际/请求变换差<2e-6；组合滑条使左/右点最大位移68.7921/60.2845px，原点序/尺寸不变。证据`root_batch_review.json`及各run/root_review.json。

这仅验证六个实际状态，不代表整个滑条范围、环境接触、真实掩码/遮挡或全部场景已通过。原尺寸GPU渲染仍在准备。native runner初版独立审查发现bank直接来源未完整纳入结束时SHA检查，正以v002补齐，保留初版，不以CPU验证代替PNG或生产接受。正式2000接受计数仍0、训练未开始。


### 2026-10-20T23:49:41.375228+08:00：双机预设及滑条组合原生图像/掩码验证

两个dispatcher profile均exit0，GPU6预设/GPU5组合，运行约113.64/113.76秒、每卡峰值3180MiB；总8renders、10PNG经host实际解码/尺寸/SHA验证。root实看`outputs/action4_pose_controls_native_prepare_v001/runs/root_pose_controls_native_comparison.jpg`；两台机身完整可见，8点12边覆盖机身，参数变化可见。每实例visible和完整amodal逐像素均在canonical8投影凸包内，越界数全0（`root_corner_mask_audit.json`）。这是合成几何标签，不是网络预测或真实视频准确率。

基础图LEFT遮挡2.1583%、RIGHT0；组合图LEFT1.6037%、RIGHT0，未达到请求5%–50%，故终态`REJECT_NATIVE_POSE_OCCLUSION_INTERVAL_RGB_RETAINED`正确拒绝该区间并保留RGB，不改标签造通过。当前可作为已验证代表性装配与标签输出链；全范围、桌面衣物/更多遮挡、全部场景、正式2000与训练仍未完成。完整结论`root_native_review.json`。


### 2026-10-21T00:04:20.351757+08:00：完整衣物与独立桌面实际组合通过

`outputs/action4_modular_environment_prepare_v002/runs/both`与`sliders`两个CPU尝试均exit0、0renders。全153 actor组件对、实际garment几何与自交均无未解决项；evaluated Solidify/接缝测得衣物底面距桌面2.852mm、最高点21.911mm，actor到衣物保守平面余量10.089/13.131mm，衣物到墙117.393mm。原有16点逐值不变，garment→table→outer恢复全部通过。保持对象独立且不穿透，未生成旧小矩形布片。实际图像/掩码尚未渲染，不据此宣称外观或全场景完成。root证据`root_actual_review.json`及各run/root_review.json，准备后续原生两例环境检查。


### 2026-10-21T00:17:34.196325+08:00：完整桌面/衣物原生两例通过代表性背景检查

GPU6/5两个profile均exit0，约175.49/174.98秒、各峰值3236MiB；8renders/10PNG完成host解码及SHA/尺寸核验。root实看`outputs/action4_modular_environment_native_prepare_v001/runs/root_full_environment_comparison.jpg`，桌面与真实纹理完整衣片已组合，不再使用悬空mini-square。两台机身与有序8角框完整可见；逐像素掩码仍全部位于canonical8投影凸包，越界0；相对无衣物pose-only，原8点逐值不变，可见/完整掩码SHA对照记录于`root_native_review.json`。当前背景变化保留标签，组合滑条5%–50%遮挡目标仍未达，作为拒绝区间结果保留，未假称前景遮挡实现。

这是两例背景预设、几何及标签链验证，不是所有参数组合或全部场景的接受。继续准备近距、侧/前视、截断和实际前臂/衣物遮挡的有限物理候选，以及正式entry/family适配；正式2000与训练仍待完成。


### 2026-10-21T01:42:46.281170+08:00 九个原生覆盖案例与主控逐图复核（阶段仍进行中）

实际GPU4–7完成9任务：8成功，right_edge_25入口失败；每例已真实4renders，共36renders，单进程采样峰值3180MiB，约124–133秒。失败阶段为actual_mask_metrics，原图和清理记录保留，无自动重试。

主控从实际PNG重算18实例的可见/全amodal/画内amodal计数，并独立用object/camera矩阵和K重投144点：最大误差9.095e-13px；可见及完整amodal像素在8点凸包之外均0。right_edge_25右机可见93628、画内amodal93627、完整105804，有且仅1个已记录边界不嵌套像素；原renderer允许该差异，新consumer错误使用visible<=inside，修正候选正在准备。right_edge_15/25实际截断率为4.428%/11.509%，不把预设名称15/25解释为实际掩码百分比。两组前臂预设右机实际遮挡为68.003%/57.016%，但视觉确认开放前臂截面空洞，所以不能据此验收该场景。

证据：`outputs/action4_registry_native_prepare_v003/runs/root_review_v001/root_pixel_corner_audit.json`（SHA256 554c9332cbebfd2af7ad820dd68c4a344af8d1ffa8b07df8ce3f122e92fffb2e）、`visual_review.json`、`native_nine_overview.jpg`和逐例canonical8检查图。全部检查图存outputs，未混入data。

衣物10/15度候选的projection摘要修正位于`outputs/action4_rigid_garment_lift_metadata_v004`；独立SPEC+QUALITY见`outputs/action4_rigid_garment_lift_spec_v003/review_v004.json`（df511bceb600268b5eb3e7a2cae56d090a10d5f2ed0abfe91416711ce03c14b2）。只修正投影元数据，未改网格/阈值；实际组合与原生像素仍未执行。

正式pose入口候选`outputs/action4_pose_formal_adapter_prepare_v001`有14个CPU测试通过，尚未激活，限基础wall_rear。正式2000接受数0、训练未启动、最终2850帧预测视频未交付；目标保持未完成。


### 2026-10-21T02:07:54.393510+08:00 增量缺陷修复与独立复核收敛

- 掩码语义v003通过独立11项host回归及Blender二值/方向/清理检查。主控再用该候选host验证器读取全部9份既有实际record，45个PNG均通过，包含此前1px误拒案例；0次新渲染，旧失败记录保留。证据`outputs/action4_mask_semantics_fix_v003/root_quality/all_nine_host_semantics.json`（f71be57988fa66ef8fe1b676a0cc3f5e779062745639a7cb5bbad112b6c1b691）。
- Formal adapter v001独立审查暴露姿态proof错配、near-plane遗漏、空磁盘plan读回三处漏洞；v002根因修复后独立21/21 CPU测试通过。该候选仍只支持wall_rear基本组合，尚未激活；大批量逐record重复来源扫描的成本另做有限计数审计。
- 前臂独立延续候选覆盖9bank/18模块；原44点精确接缝、6336接触交集窄证明、实体环境、自交、彼此与原actor检查由独立SPEC复核。最小深度0.102067m、整尾环画外余量至少88.55px。接口NaN绕过已在新v002有限值检查修正，18正例与9非有限值反例通过。这里只证明CPU候选，实际接缝法线/材质/visible mask仍待原生组合审核；相关runner正在准备，未渲染。
- 冻结登记v003的18个资产文件哈希再次全部一致。临时资产、检查图和报告均留在outputs；未生成正式训练样本。
- 当前9个诊断样本RGB+双机四mask平均6,870,937.67字节，按同分布2000张约12.798GiB，仅为图像/掩码初估，不含annotation/cache/checkpoint/video或新场景差异；正式先导后重估。

主控候选quality收据：`outputs/action4_registry_native_prepare_v003/runs/root_review_v001/root_candidate_quality.json`。所有候选与实际运行、正式接受严格分开；本阶段继续进行。


### 2026-10-21T03:03:21.086621+08:00 前臂实际组合闭环完成，两个有效遮挡预设确认

第一组三个实际任务各4renders（共12）完成几何与像素后，均在严格inventory清理终止，失败保留。root实际查看确认空洞/断口消失；near-left与source0目标支持有效，默认partial1只有3.17563757%可见因此拒绝。没有按原68%遮挡假定继续验收。

通过3个4×4 CPU对照实验定位：use_empty/save/reopen的首次render新增内建Render Result。新窄所有权事务经独立59项检查，随后near-left实际native重验exit0，149.895秒、采样峰值3180MiB；精确删除一个新增无引用Render Result，所有原ID集合恢复。新旧16点和四mask完全相同；RGB并非字节一致，8bit最大差1、均值0.00180847，已重新实际查看，未将初始RGB相等假设当通过。证据`outputs/action4_continuation_cleanup_fix_v001/runs/near_left_100mm_back_native_v001/root_quality.json`（eecc52570bf30a3a8846341f62b2d930ca823022019c500a2148c8ad1ed37a45）。

随后在原控件范围内预声明12个缓存候选，选bank2/1 local-X+6°。新增typed接口从当前参数及不可变基线计算Dworld并核对实际skinworld；原fixed接口不改、不伪传旧world。经独立56项检查，两个实际native任务exit0，分别约157.863/158.870秒、采样峰值3180MiB。原153+新增41对/自交/接缝/末端/near均通过，cleanup完整；右机真实可见比例64.25015906%与35.88834130%，两机均有效。root解码mask、独立重投16点（每例最大4.547e-13px）、凸包像素检查和实际图像复核通过。

结果：`outputs/action4_continuation_controls_actual_prepare_v001/runs/root_review_v001/root_native_review.json`（3084d73d6a774397b0d1aa304cbbce56f4213c538dd672c4583c8da5c0c911f1）、同目录`typed_controls_native_comparison.jpg`/`visual_review.json`。原RGB/标签仍为outputs诊断，正式训练接受数仍0。

Host批次上下文候选已实现并进入独立审核，计数确认整release扫描降为open/close各一次、authority每批一次；worker/pixels剩余开销保持未解决，不称全链路性能已达标。源上下文兼容桥只在独立候选中开发，active src保持。下一步是审后组合正式执行入口、跨场景环境层及批量控制域，继而128与2000先导训练，未宣称项目完成。


Host批次优化独立SPEC补记：原21+12+16检查通过不足以验收批次生命周期；新增公开接口反例证实跨线程require_plan/defer_report漏owner、shard前置解析异常漏poison、延迟多报告后项冲突时前项成功文件残留且receipt仍可读。SPEC_REVIEW SHA1c0a00e9cfdf613585d8f412befbb71c8e8e6821cbc06ad7ee3ecd27d1da7646。v001暂不合入，v002限定修复进行中；不降低几何/来源/支持门，不把source完整性PASS当训练通过。


### 2026-10-21T04:02:24.063128+08:00 完整场景原生通过与实际源码迁移

source540左local-Y+6°完整table/full-garment/双臂续接/双机组合完成4次原生渲染，实际任务exit0，耗时472.438秒、GPU采样峰值5012MiB。原153/16、组合45交互对与全20件半空间均通过；桌顶最小间隙33.895643mm，衣物顶部21.910980mm，二者余量11.984664mm；完整逆序恢复通过。root已查看真实RGB和重新解码mask/投影，两个目标可见比例98.65775%/99.99857%，完整amodal及visible凸包外像素0。证据root_quality SHA0c77f5ca8dc967b775c493fd6edcb9326d92255ae5009770f55dffc00b91a194。仍是保留原wall_rear entry的诊断，不冒充正式table-family数据。

全部项目任务和子agent停稳后，root按activation_plan SHA29974750adfe05f78367e07279a4ff25894e9932058ac6820323270f8f12fbee实际替换4个src文件并增加3个helper，保留原byte备份。fresh-process driver v002进入真实source迁移：neutral/contact兼容view/composed/bilateral/wearing/factory/bounds七级已创建，新source context c775bd43196ff10c2987bef98cb7f2431f36cbf69e35cf46e86570a03fec0809通过物理树同值及bridge检查，但在第一selected variant的exact派生重绑处失败。FAILED与全部中间文件保留。

root比较发现非metadata差异为派生矩阵约1e-16、投影约1e-12px的重放差异。现有typed validator原本支持有名路径的1e-9重放预算；用历史全部数值原样、只更新四个明确身份字段后，两variant完整typed及桥exact-preservation检查均通过。已派发最小rebind函数修复，保持现有容差、控制、固定值与所有旧文件不变。此诊断不等于迁移完成；旧current入口已标为None防止误用，新尝试将独占新目录而不覆盖失败v001。


## 2026-10-21T04:21:57.464504+08:00 当前执行来源迁移及独立复核

七个已审源码文件已激活；首次release_v001在历史参数浮点末位序列化精确比较处失败。仅修正rebind函数：保留历史数值，只更新四个明确来源身份，再通过原typed重构容差和bridge精确保留检查；未放宽容差，失败证据原样保留。新release_v002 fresh-process迁移exit0。

主控另起进程重新prepare factory/load配置，核查205项实际文件SHA、11个banks、18项冻结资产；两variant与两baseline在剔除四项允许身份字段后JSON序列化精确一致，完整当前typed校验和bridge闭合通过。报告 `outputs/action4_execution_source_release_v002_root_review/root_review.json`。此验证未创建场景、未渲染、未训练；此前完整桌面/衣物/续接实际图仍属历史context证据，不自动转移为当前来源准入。

后续按原计划继续：两个已审runner显式接入当前context和bridge，独立SPEC、实际native标签与主控图像检查，再做正式family/参数范围/128/2000与pilot训练、存储实测和2850帧视频。不重复建模、不改变固定佩戴尺寸、八角点顺序或10%/16px门槛。


## 2026-10-21T04:49:33.227320+08:00 当前来源两个遮挡案例实际通过及测试夹具修复

两个current-source continuation案例在GPU5/6并行实际运行，均exit0；四次render/案例，耗时255.574s与262.801s，显存采样峰值4522MiB与3180MiB。主控独立投影原32角点最大误差4.55e-13px；legacy float32记录与该投影差异最大0.000170px，符合原canonical_export独立0.003px容差，两者未混同。实际可见/完整机身掩码在八角点凸包外均0px；右机可见比例0.358883/0.642502，左机1.0，未ignore/排除。原153与续接41检查无未解决碰撞，严格runtime inventory与layer/source逆序恢复通过。主控已看两图：双机佩戴与遮挡可分辨，续接末端未暴露；细手指和简化环境仍为工程近似。仅证明两个离散合成诊断，未证明全域参数/真实隐藏角点/模型预测精度。证据 `outputs/action4_current_native_root_review_v001` 和 `outputs/action4_current_continuation_dispatch_v001/dispatch_observation.json`。

源码激活后的35项回归首轮4fail+4error来自测试临时根未传入新mask helper，以及旧CPU fixture缺完整plan/authority/launch和新异常cause包装。保留初始失败log；经独立SPEC和主控diff审核，仅更新两test文件，真实Host/index/record/poison/close/publication保持执行。新增source漂移关闭拒发反例，active重跑36/36通过（`outputs/action4_active_fixture_application_v001/active_tests.log`）；无运行源码/门槛修改，无pytest安装。

完整table/garment当前入口独立审查发现三个直接读取ref未闭合，正在版本化补齐；未运行被拒绝packet。正式accepted2000仍0、pilot未启动、完整2850帧预测待完成。


## 2026-10-21T05:26:44.896438+08:00 完整当前场景通过与同卡快照优化对照

完整source540场景current v002在GPU7实际exit0，4render，606.459秒，采样显存峰3394MiB；主控重新读取PNG与geometry复核16点、mask包络并查看原图。153+45检查无未解决碰撞，20个actor/续接的桌顶最小间隙33.8956mm、衣物包络余量11.9847mm，两机可见98.6578%/99.9986%，runtime/layer/source严格恢复。该样例原entry仍wall_rear历史诊断身份，不能自动签table正式family。root报告 `outputs/action4_current_native_root_review_v001/source540_complete_current_native_v002/`。

为降低重复扫描，新增独立候选只在单次OwnershipRegistry._verify内部共享fresh snapshot，原source_snapshot独立保留；原33与新增10mock、独立SPEC证明完整6owner扫描7→2，无跨调用缓存。独立集成SPEC确认全packet仅四组执行引用变化，物理/来源/参数/seed/render/GPU一致。随后同GPU7实际候选4render成功439.138秒，采样显存峰3236MiB。单次对照节约167.321秒（约27.6%），仅作本对实测，非稳定吞吐结论。完整实际几何/角点/掩码数组/支持计数精确相同。

RGB未逐字节一致：max单通道差1/255、平均差0.005494灰阶，1.6168%像素至少一通道变化。深入核对发现garment loop_normals摘要变化、geometry_uv和实际几何不变；mesh_state会调用calc_normals_split。旧/新未优化样例之间也有max1/255的RGB差异，但其normals摘要相同，因此不能直接把本次normals差异归因为GPU噪声。已保留实际差异，派发仅isolated garment的有界CPU诊断，无需重建冻结DJI或重跑完整场景；优化暂不晋升正式生产，未优化已通过路径仍可复用。证据 `outputs/action4_current_native_root_review_v001/snapshot_pair/`。

新参数化续接接口另行准备于 `outputs/action4_parametric_continuation_prepare_v001`，旨在复用已冻9个bank并支持新entry/seed，而不复制旧diagnostic身份；作者报告3代表bank与反例CPU通过，尚待封存、独立SPEC及实际集成。训练数据0/1/2配额、test全双机和完整任务目标保持；accepted2000、pilot、存储最终评估与2850帧预测均未完成。


### 2026-10-21T05:38:06.510748+08:00 新entry接口SPEC与法线有界诊断关闭

参数化双侧续接接口已封存并获独立SPEC：3个代表bank/六侧新entry正例和10个关键反例独立通过，作者3正29负证据另保留。接口支持已选9bank的精确geometry lookup，独立重放新entry authority、验证current参数；历史baseline仅供固定局部mesh/world，采用原Dworld公式，不以旧sample_id冒新样本。正进入versioned actual layer/runner接线；尚未执行该新接口的真实场景，未签formal acceptance。证据 `outputs/action4_parametric_continuation_spec_v001/SPEC_REVIEW.json`。

法线小型CPU实验已终止：同helper单panel4356法线三次读取逐字节一致/finite/近单位，geometry/evaluated SHA命中实际，但normal SHA为第三值，不能推算原两场景normal数组差。主控独立读取NPZ确认；实际差异已在garment注册前构造摘要中出现，彼时两版table-only扫描次数相同，且新fingerprint仍独立检查mesh_state，不存在删除法线校验的证据。精确原因与实际normal等价仍未知，报告 `outputs/action4_snapshot_normals_diagnostic_v001/root_review.json`。保留快照版为有实测收益的诊断候选，不据本对照批准正式量产；同时此细节不阻断新entry接口与正式数据流程，复用已验未优化完整scene作为基线，不继续只为追法线hash而重复完整场景。


### 2026-10-21T07:15:59.257779+08:00：新衣物姿态源码接入与正式数据链进展

- 实际执行 `publish_root_only_v002.py --execute`，exit0。只替换pose-controls与execution-context-bridge两份源码，其他5份保持原SHA；独立SPEC先关闭输入身份与双次读取的两项问题，v001仍保留为blocked历史。新receipt：`outputs/action4_execution_source_release_v003/MIGRATION_RECEIPT.json`，SHA69984a0a0b7728e2b60cf26baa1f70afae2a123409dc2766d823cc682b8ea2b4。主控实际核对267项依赖；typed重载由独立智能体继续复核。
- release_v003原11姿态保留，并加入cloth_lift10_source540、cloth_lift10_source960两组10度离散预设。共同位移来自全20个actor/续接部件与衣物间隙计算；不使用只含18个部件的错误间隙。局部pose滑动暂锁0，原机身尺寸、安装和canonical8不变；真实碰撞、掩码和原生效果仍须验收。
- release_v002已完成并由主控复核四presence原生案例及新entry完整桌面案例，见`outputs/action4_current_native_root_review_v001/presence_and_full_newentry_root_review.json`。0/1/1/2相机分别导出0/8/8/16点，单机保持原ID1或2；实际pair、掩码、投影与恢复通过。完整桌面案例包含18原部件及2续接，实际153+45对检查通过。它们是明确历史source下的实际诊断，不能冒充新source或正式训练集。
- 新source的桌面/整件衣物provider已另存`outputs/action4_parametric_environment_prepare_v002`，17项CPU合同检查通过；函数/类AST和几何构造不变，只更新来源绑定。等待窄SPEC后供后续实际组合使用。
- 并行推进derived预设到原geometry bank映射，以及worker批次验证复用。后者发现JSONL分隔符误写为字面转义，正在自有候选修正并加入逐行解析检查；未进入active或正式数据。
- 当前正式accepted=0；未启动pilot训练。后续顺序仍为新source代表性真实组合与八类/0-1-2正式接口、固定128检查、按1600/200/200生成accepted2000、D0/A0/B0、全视频预测和实际存储统计。测试集仅双机、总双机92.8%，不得用诊断样本填正式配额。


### 2026-10-21T07:22:27.553787+08:00：发布后重载接受与2000条目的配额落实

主控读取并检查独立fresh重载脚本及SPEC后，接受release_v003的源码/metadata限定结果：实际267项引用、18冻结文件、原11/新增2预设以及两derived的typed重放与历史baseline非身份字段精确一致。来源环境v002的14项独立检查（7反例）通过，176实际消费引用闭合。主控接受回执在`outputs/action4_execution_source_release_v003_root_review/root_acceptance.json`；真实cloth_lift场景仍未验收。

用原分配器生成`outputs/action4_formal_family_allocation_prepare_v001/ENTRY_AUTHORITY_205901.json`（SHAe8a127bfe0a1e482cd70085b10ad64be1741531271191da33a13c0ced7008fb0），完整重放一致且group不跨split。train为48/80/1472、eval为6/10/184、test为0/0/200。保留已有八场景权重；hard_negative仅0目标，wall_rear可1/2目标，其余六类限定双机以匹配当前场景实现范围。该分配只固定待生成身份，尚未绑定完整渲染计划或产生任何正式accepted图片；不可据此宣称2000数据生成完成。后续正式配方必须使用相同allocation_controls和身份，不能渲染后改ID或挪split。


### 2026-10-21T08:28:57.134781+08:00：cloth_lift真实任务启动、批次接口审查和性能定位

- derived mapping独立SPEC通过，current preset与immutable geometry bank分离，40个CPU canonical点重投影与215依赖闭合，不把缓存当实际场景。
- 场景层v001被独立审查拦下authority/baseline字段合同错误；v002只增创建前3ref精确绑定并补示例，原geometry/build/remove不变。独立20项检查（2正18反）通过，原失败保留。
- runner在封包前修正caller漏字段、独立review报告schema接线及profile finalizer掩盖主异常问题。最终H SHA814703868df699cadcbfa8c92560885d0d24b167950e50cb5656b122626139c8；有效native_v004两包及runner精确refs见H。独立25项CPU检查通过；root核对339依赖并部署原字节薄入口` scripts/action4_derived_lift_preview.py`。
- root任务`derived_cloth_lift10_source540_native_v001`与`derived_cloth_lift10_source960_native_v001`已在GPU5/6分派，现场确认worker/child活跃；只算实际运行中，尚不算成功或正式数据。原16点、153+45/full20、20mm−2e-6、4render、5PNG、10%/16px和严格恢复不变。前者同时输出execute cProfile；后者不加观察开销。
- 已检查旧fulltable诊断wall593.051秒与日志RGB render/save17.10秒、3mask Finished1.31/0.40/0.37秒，证明需定位渲染之外开销，尚不能归因到某具体函数。实证保留`outputs/action4_throughput_audit_v001/OBSERVATIONS.json`，不拿profilewalltime冒正常吞吐。
- worker batch候选已独立145断言及root quality通过，但未激活；只支持局部编排扫描复用，不宣称全worker O(N)。正式composition10文件候选已完成，独立SPEC发现侧几何proof未闭合immutable assets和同数错误pair名可通过两项问题；原作者只修受影响host合同，保持反例与v1blocked报告。
- 后续统一发布采用显式7个formal consumer来源，current metadata配置传入以消除hash循环；bridge拟仅允许指定wearing sampling factory追加这7个精确source refs，禁止通用扩展。实际active源码变更须等待当前GPU任务终止及独立SPEC通过。此项现在仅准备，不是已发布。


### 2026-10-21T08:59:02.943061+08:00：两组当前衣物抬起实际闭环通过，定位重复校验开销

两任务均succeeded/exit0：source540（带execute cProfile）wall1005.872秒、峰值4522MiB；source960（不带cProfile）wall968.369秒、峰值3236MiB。它们姿态不同，不作为profiling开销或优化加速率的配对实验。root独立原32点重投影最大误差4.55e-13px，legacy float32记录最大约0.000155px；可见及amodal掩码均无像素超出8点凸包。source960右机可见率约97.04%；左机沿原边界规则保留1个nonnested像素及raw计数，未改门槛。精确source153/extension45/self2/full20名单核对通过，actor/garment间距两例约0.0200001866m。

root实际查看了两张原生叠图：8角线框覆盖机身而非屏幕标记，双机分离、腕部连接与typed变换一致，前臂末端不露开口，完整桌面/衣物可见；不宣称精细抓衣动作、照片级真实或逐帧复刻。原生图与叠图分别在`outputs/action4_derived_lift_actual_prepare_v001/runs/`和`outputs/action4_current_native_root_review_v001/derived_lift_source{540,960}_v003/`。

Blender日志保留了删除其新建内部Render Result（users=1、user_map为空）的消息；root核对renderer before==after_cleanup、原ID保留、unknown_changes=False、无render primary error，随后layer/base恢复通过，没有据此隐藏未知源对象泄漏。

profile实测216次new_scene_state、5698次mesh_state、103次load_config、15次lift.validate_binding和178643次SHA256；部分cumulative时间互相包含，不能相加。当前优先仅把正式router的registry引用切到已审同次snapshot复用实现：每次_verify仍新读状态，6owner共享本次完整快照，保留独立source和字段检查；不实现跨call/typed cache。候选`outputs/action4_current_runtime_reuse_prepare_v001/HANDOFF.json`，等待窄SPEC与本次正式代表例实测，不承诺旧606→439的加速率或法线逐字节相同。

formal integration v002已关闭两项host不可变几何/精确pair绑定问题，独立报告`outputs/action4_formal_composition_integration_spec_v001/review_v002.json`、root质量回执`outputs/action4_formal_composition_root_quality_v001/ROOT_QUALITY.json`。当前active仍v003，尚未发布11文件候选；新required_jobs/FINAL_GATE已绑定本轮两个终止任务和root实际图像报告，执行时仍须fresh quiescence与精确发布SPEC。正式训练数据仍0张。


## 2026-10-21T09:56:06.728362+08:00 v004实际源码与元数据发布通过，原生候选准备中

> 当前执行位置（2026-10-21T09:56:06.728362+08:00）：release_v004的11项源码安装与3项保留源已完成，fresh独立校验及root复核通过；完整发布标记为`outputs/action4_execution_source_release_v004/FORMAL_PUBLICATION_RECEIPT.json`，source语义摘要e517b0f944c5598a87142ed1ac360b33607f94781566e10037af185ca4c49919。当前实际生成入口仍须验收：16-entry/8-family/四presence候选plan已由原create_plan生成，Host opening/closing及封存正在进行；经独立审查后由原scripts/action4_synthesis.py render-shard经Dispatcher调度GPU4–7。候选数据与正式2000分离，不继承v003实际场景结论。后续顺序保持实际16接口核验→固定128准入→合格2000→D0/A0/B0→2850帧预测与存储评估。正式accepted=0。


根验收证据：`outputs/action4_formal_publication_root_review_v001/ROOT_PUBLICATION_ACCEPTANCE.json`；独立证据：`outputs/action4_execution_source_release_v004_spec_review_v001/SPEC_REVIEW.json`。真实安装11、保留3、依赖346、Blender Python3.10 AST通过。基础MIGRATION_RECEIPT的pending字段保留，最终回执另行表示完整发布。六条metadata仅五个不同entry，不能当作六张渲染图。source540/960旧lift实际结果保持v003历史身份，不能移作v004验收。

本轮准备发现outputs辅助launch.py不符合现有Dispatcher的scripts入口与长参数白名单；实际入队将使用既有正式render-shard CLI及完全相同plan/shard/output参数，不新增入口或放宽调度校验。当前未入队新GPU任务、未生成正式训练图片。


2026-10-21T10:05:09.671396+08:00：完成v004→训练→视频窄只读梳理，24项证据引用经root当前哈希复核。未发现确定字段/坐标接口断裂；当前缺split属于尚未生成验收数据。来源报告SHA19a4fb0638b21bc564d6f98d124e109043ea321399fcaefda82f5f0e65161c62。未启动训练、模型、评测或预测；上述阶段数据来源绑定及规范B0调度作为启动条件补入同份plan。


## 2026-10-21T10:14:11.090214+08:00 当前16候选实际调度启动

独立17项SPEC完成（报告cc885e04...），root实际读取caller/计划/证据后质量通过（4dadaec5...）。原Dispatcher完成输入闭包核验并一次性入队4个profile=True任务，各4entry/2threads/8192MiB reservation，物理GPU4–7。session65418为唯一运行scheduler；通过实际PID/start/command身份确认worker与child四组均live。生成路径data/action4_synth/interface_candidates_v004，运行路径outputs/action4_synthesis_runs/action4_synth/interface_candidates_v004。当前运行中，不等于16图完成/场景接受，正式2000 accepted仍0。原健康CPU准备总1454.448946秒；没有重复创建plan或弱化门槛。


### 2026-10-21T10:16:52.314970+08:00 用户更新CPU资源授权

项目CPU总预算调整为72核，取代此前32核总上限；显存足够时允许同卡多进程。并发数量按实际负载profile、剩余显存和CPU预算确定，保留原GPU4–7范围及显存余量门。当前4个健康渲染任务保持原冻结source；调度器总预算32→72的单点修改先作为outputs/action4_cpu72_dispatch_prepare_v001候选，独立检查后在当前批次终止边界统一启用，避免运行中改变source。单个任务线程上限32暂不改变，通过增加独立任务利用总72核。训练接口的来源绑定与合成数据图像/标注/物理质量准入保持原要求。

本次未修改active调度器或重启任务；仅生成候选。


### 2026-10-21T10:23:29.898005+08:00 参数范围及检查辅助进展

当前16的四个原任务均进入Blender子进程，仍执行来源检查；worker/child实际身份存活，无重启，不计图像完成。离线叠图脚本已独立schema/18toy检查通过及root质量复核，证据outputs/action4_formal_native_visual_spec_v001/ROOT_QUALITY.json；实际16图像仍未检查。参数采样准备报告`outputs/action4_bounded_sampler_readiness_v001/READINESS.json`（SHAd98e5dabe5f8b7e7214ce0674ce21670b4b009b6f22ba6d13b36e86b79963e7d）明确工程limits不等于任意组合物理安全；128诊断entry子集不替代完整2000authority/plan。当前几何probe嵌入render流程，冻结前独立几何检查薄入口为真实接口缺口，已派Astra在独立outputs候选实现，禁止改运行中源。静态配置复用候选亦未合入，局部CPU计时不得当整体生成速度。


2026-10-21T10:25:42.651357+08:00：已将用户要求的全参数量化字段、输入limits与实证safe范围的区别、坐标系/关联约束以及边界验证顺序写入同一plan。现值来自outputs/action4_bounded_sampler_readiness_v001/READINESS.json；完整机器台账正在独立outputs目录整理，尚未替代active配置或声称全域安全。


2026-10-21T10:31:18.556549+08:00：用户指出距离、旋转与遮挡、双机相对位置三个参数语义不足。此前列出的数值属于现代码微调limits而非完整目标范围，主控已明确纠正并在同plan增补绝对距离/图占比、实际mask可见性与识别界限、双机3D关系定义。当前16 frozen候选不改，仍非最终覆盖证明。台账同步更新中，尚未实现新三维滑条或确认识别下限。


2026-10-21T10:34:18.395159+08:00：当前16的首4张分别wall/table/tableturn/near已真实生成RGB与backend_record；root仅作临时检查，叠图及actual距离/占比/相对向量/原mask统计写outputs/action4_formal_native_interim_review_v001。八角线框覆盖机身、独立重投影误差为浮点数值量级；原共享mask检查四条通过。near左前臂衔接与左图界skin wedge外观需定位，已记录FIRST4_VISUAL_FINDINGS，不能当全场景通过。每shard完整manifest/Host关闭未完成，正式accepted仍0。


2026-10-21T10:42:29.981559+08:00：完成参数台账引用与静态场景复用原则增补；当前实际多entry暴露composition预检目录共享导致第二entry失败，失败证据保留并正在单点修复。不能把224参数条目或静态复用方案当已实现稳定采样器。


2026-10-21T10:46:12.406840+08:00：用户明确避免重复哈希，已写入原plan。实际单row profile支持该优先级；同scope不可变数据复用候选仍未生产验证。三个实际shard已因共享preflight目录失败退出，单点目录隔离候选独立CPU red/green通过；第四shard1仍live，不改其source/不重启。正式accepted仍0。


## 2026-10-21T10:49:25.038000+08:00 原批次结束与单点修复安装

原4任务均failed/exit1、scheduler观察进程session65418正常结束；实际worker/child均已退出。四个失败stack都指向同一composition_preflight_data全shard共享mkdir冲突，第一entry成功、第二entry拒绝。首4图与临时测量保留，整批manifest未完成，不能转正式训练。独立目录red/green/同ID重复拒绝通过，root核对源码只变1行。composition已安装b1482ec3...，原f174前像保留；当前source metadata刷新尚待完成，不运行旧v004 plan。调度器总CPU72已安装d2e3ba13...，所有其他GPU/磁盘/profile/每job32规则不变。安装回执分别位于outputs/action4_composition_batch_path_fix_v001/INSTALLATION_RECEIPT.json及outputs/action4_cpu72_dispatch_prepare_v001/PUBLICATION_RECEIPT.json。

性能/复用只读报告outputs/action4_scene_reuse_readiness_v001/REPORT.json指出：每worker已是单Blender多entry，但每entry仍重新加载baseline/固定资产；首要已测热点为重复source读取/哈希，随后才评估warm scene，尚未实际实现，不声称提速完成。


### 2026-10-21T10:56:36.919110+08:00 v005来源更新实际完成

已审窄driver在fresh进程session86453执行exit0，CLOSING SHA f91bc431d269da2f1f437246bdf74e8e4fa55b999415b22a2f7b9205acaf9876；实际context为2500887c90a2e48706eee992016f1708f268b8b5165fe179e0b6166e01f8d4a1。直接比较v004物理tree及v4/v5 bank数值均等，root只核对14新增metadata引用，没有额外全大资产扫描。最终状态见outputs/action4_execution_source_release_v005/ROOT_ACCEPTANCE.json。当前只证明metadata成功，未证明新多entry渲染成功。下一步原16身份/controls使用8worker×2threads并行重绑，原create_plan/Host必要检查及4×4实际重跑保持，产物新目录interface_candidates_v005，旧失败完整保留。


2026-10-21T11:02:38.329109+08:00：已按用户要求收敛参数工作：停止额外精细范围研究作为前置，以粗范围+冲突拒绝推进训练闭环。原16 v005准备session78104已实际启动8workers，各worker2线程，8个进程实查live且首次8params已写，无失败；正式绑定/create_plan/Host仍待完成。


## 2026-10-21T11:23:59.392120+08:00 v005完整准备通过并实际重跑

并行prepare session78104 exit0：entry准备68.719s、create_plan340.549s、Host652.521s，总1061.806s；参数/entries不变，Host成功关闭，record_checks0明确仍非实际图验收。最终SPEC f9f546d3...、rootquality1bb4669c...。调度预检曾拦截caller缺scripts入口SHA，保留失败并仅补entrypoint/source两refs后原_validate通过；不重跑Host。当前root唯一scheduler session97036已实际入队v005四个4entry任务，GPU4–7，各2threads/profileTrue，新data/interface_candidates_v005。实际观察：[{"job": "formal_interface16_v005_shard0_v001", "status": "running", "worker_live": true, "child_live": true, "device": "GPU-1648afa2-3e25-b1c7-b149-38b07306eb8f"}, {"job": "formal_interface16_v005_shard1_v001", "status": "running", "worker_live": true, "child_live": true, "device": "GPU-8c93aadc-2a0a-665c-46c3-dbd3f063e0d8"}, {"job": "formal_interface16_v005_shard2_v001", "status": "running", "worker_live": true, "child_live": true, "device": "GPU-b1ae5be6-1dba-7343-e233-59df41402f33"}, {"job": "formal_interface16_v005_shard3_v001", "status": "running", "worker_live": true, "child_live": true, "device": "GPU-a68477c3-1c2a-d557-58f5-405d80e5383e"}]。所有旧失败保留，新的scene/第二entry/完整Host和视觉尚未通过，正式accepted仍0。


2026-10-21T11:28:41.164500+08:00：用户明确2–3主要场景即可，已将三现有模板及family兼容映射写入原plan。后续不增加环境，侧重粗参数大量采样与试训；当前冻结v00516运行不改参数或身份。


2026-10-21T11:44:43.564426+08:00：v005 shard0/3各已实际写出第2entry的backend_record（self_forearm及zero），确认旧mkdir冲突没有在这两条连续路径复发。root实际PNG读回、原8点独立重投影与原共享mask检查通过，并查看两张派生叠图；右相机受遮挡仍保留完整机身8角，无目标图无虚构框。证据outputs/action4_v005_first_records_review_v001/second_entry_INTERIM.json及second_entry_VISUAL.json。此为单张临时检查，所有shard Host关闭/完整16尚待，正式accepted仍0。


## 2026-10-21T12:27:18.646041+08:00 v005全部渲染完成，Host浮点统计比较错误与独立图像诊断

四shard各4张及worker完整关闭已完成；后续Host均failed/exit1，scheduler97036 exit0是观察进程正常返回，不是数据通过。原16几何投影全部通过1e-6px，而11条diagnostic残差dict因跨numpy环境的最大4.547e-13px差异被精确相等拒绝。源/原records/失败不改。残差修复独立SPEC52d088...保留原point1e-6/transform2e-6/legacy.003，比较诊断统计用32ULP且cap1e-10，待5source整包安装。

root单独读取16PNG/28实例及mask，独立8点重投影和mask/hull检查通过，查看完整index16；这些是合成GT而不是模型预测。独立诊断见outputs/action4_v005_independent_image_audit_v001/DIAGNOSTIC.json、ROOT_VISUAL_DIAGNOSTIC.json；不替代原HostPASS、不转训练accepted。用户最新粗范围与三模板优先级下，保留粗手部/续接外观限制，不扩精细建模。正式accepted2000仍0，当前所有旧jobs已terminal。


## 2026-10-21T12:32:49.576382+08:00 五源码统一安装与真实集成启动

全部旧jobs terminal后，root按v003 manifest da623c63...实际安装五份已审候选，含cached adapter残差统计修复0e85...；当前Py3.12和Blender Py3.10语法均通过，逐项before/archive/after匹配，精确安装journal与回执在outputs/action4_validation_reuse_root_install_v001。没有改原16数据、没有将原Host失败改为PASS。fresh metadata driver session87118已启动，真实2row复用/8entry create_plan/union closing尚待结果；当前source处于v006 metadata pending，旧v005计划不可作为当前source新执行入口。


## 2026-10-21T12:51:12.305085+08:00 v006真实集成完成并启动8张实际链路验证

真实2row scope复用和8entry create_plan通过，metadata总296.719秒、create_plan30.1905秒；这是metadata耗时，未证明整批渲染提速。独立最终SPEC outputs/action4_formal_native_v006_spec_v001/SPEC_REVIEW.json通过，root已检查并在原Dispatcher实际入队4x2，GPU4–7，session99134。复用现成8条计划，不再构造另一16条；实际scene、原8点/实例mask、完整Host以及可视化待完成。预冻结128原2000ID输入和真实Dataset/ROI/heatmap映射检查同步准备，三现有模板和正式2000配额不变。没有训练accepted；旧失败原样保留。


2026-10-21T13:00:37.369843+08:00：首128原2000ID typed参数与candidate/baseline绑定实际全部完成，4进程总约158秒，0失败；剩余1872已启动12worker各2threads（session88929），复用首128，不重复生成。actual v00516张/28实例的原Dataset、det targets、trainer ROI、corner heatmaps及eval inverse实际CPU检查通过，覆盖train8/eval4/test4，原图坐标往返最大5.084e-13px，保留原Host失败且formal Dataset拒绝未验收样本。证据outputs/action4_training_io_probe_v001/actual_v005/REPORT.json；不涉及训练模型预测或正式test结果。并行子智能体因额度中断，root已接手其必要实现和API检查，未切换Sol。


2026-10-21T13:04:08.137811+08:00：v006 shard0及3的完整geometry_validation已发布，manifest身份、2张/分片、合计6实例48角点核对通过，非正式2000验收。root以真实报告授权首128原始ID的CPU Blender预冻结几何诊断，16worker各2threads（session52449），实查16 PID均live且开始计算；与1872参数准备12worker各2threads及最多4render各2threads合计64线程预算，低于72。首批预冻结尚无结果，不宣称无冲突/128通过；其后仍需原实际RGB/mask和可视化门。主机可用内存实查约407GiB，GPU当前profile峰值约3.1–3.4GiB，后续生产同卡多进程仍按完成profile和真实余量准入。


2026-10-21T13:17:38.404304+08:00：用户再次要求用足72核。参数准备1872已完成exit0，用时745.488秒，连同此前128共2000输入，仍非2000图像。root将首128中尚未创建输出的后64原ID显式交接到新增36单线程Blender进程（session78648），已实查36PID均约100%CPU；保留原16×2线程任务。加计划构建2线程与剩余诊断2线程，总工作预算72。交接不覆盖原样本目录，实际单例由原独占输出目录保护；旧固定分片worker以后遇到已被新worker接管的tail目录会在新场景创建前停止，该调度结果保留且不能混为几何失败或擅自重跑。新增aggregate.py要求两个启动器结束、完整128真实SELECTED/GEOMETRY与原始8投影及清理全部成立，任何真实几何失败继续阻断。证据与交接清单位于outputs/action4_prefreeze128_scaleout_v001。2000候选plan同时只做metadata构建（session40627），实际128选择必须逐项等于plan，否则保留候选并构建新版本。


2026-10-21T13:21:21.521585+08:00：v006实际8的4个原render-shard任务全部succeeded/exit0（session99134结束），完整Host报告共8图14实例112角点。原独立8点重投影、mask-hull containment与未变10%/16px门通过；root查看index8和右机正常/遮挡/截断3张原生crop，确认全机身8点而非屏幕标志，隐藏点保留、出画不钳位、无目标无虚构框。actual Dataset/ROI/heatmap/eval inverse消费者8图14实例实际通过，坐标往返最大5.084e-13px；本8只train4/test4，没有冒充eval数据或模型预测。完整有限验收见outputs/action4_execution_source_release_v006/ROOT_ACCEPTANCE.json，仍不替代128正式RGB验收、训练或2000 accepted。几何预检截至此时41/128通过，0回退0失败。


2026-10-21T13:26:16.769427+08:00：首128几何预检已选中80条，0样本BLOCKED；3个旧固定分片worker已在明确转交的tail独占目录处停止，另11个新worker已完成。旧退出并非样本PASS证明，仍由scaleout/aggregate.py逐项检查实际128份GEOMETRY/清理/来源/投影后才能汇总。原128真实图像尚未启动，候选2000 plan构建session40627仍live。已准备实际128原始点/mask/Host检查及全景+机身局部放大的8页检查图入口outputs/action4_actual128_review_prepare_v001/review128.py，未宣称已执行。


2026-10-21T14:00:00.610830+08:00：128几何预检完整汇总通过（outputs/action4_prefreeze128_scaleout_v001/AGGREGATED128.json，SHA af919e88...），0回退/0样本失败，最大独立投影差9.0949e-13px，实际选择与已准备2000参数前128逐项完全一致。旧启动器16个目录独占交接退出完整保留，逐样本GEOMETRY及清理证明没有放宽。当前实际128RGB尚未启动。

统一检查代码已归位src/eval两个模块与scripts薄入口，11个原函数/类AST一致；最终真实8图/14实例四进程检查13.587秒，角点比较差0、mask测量完全相同，真实consumer坐标往返5.084e-13px。错误计划SHA、缺分片、Python优化禁用断言三反例均拒绝；root查看统一全景+局部图。证据outputs/action4_integrated_validation_checks_v001/ROOT_QUALITY.json。

发现旧外层render_shard仍使用无scope的全计划启动检查；新增cached launcher复用原index而不修改现有producer，实际8metadata预检32.63秒且4条Blender命令逐字符等同原成功命令，GPU0次。其新实际GPU启动将在原128阶段验证，不冒充已经运行。Dispatcher资源/路径规则同步支持新入口，76项回归通过；原31+2>32测试与早先已实施72核规则冲突，已保留失败证据并改成多job精确72/73边界。执行authority10项、cached launcher6项控制面检查通过。正式执行r001锁定准备态，旧计划和训练被拒绝。当前完整2000计划构建session40627仍在运行，原source和全部参数未改。


2026-10-21T14:03:24.449587+08:00：隔离控制最终补充阶段范围不可扩宽、训练计划必须绑定当前source/parameters、train/eval/test三份split必须齐全。控制面最终95项通过（76调度、6cached launcher、13authority），源/参数/几何五个既有producer源码SHA与v006完全相同，无物理source迁移。缓存启动器ROOT_QUALITY及统一检查ROOT_QUALITY已保存。r001仍准备态；完整2000计划构建尚未返回，不提前开渲染或训练。


2026-10-21T14:28:53.051326+08:00：完整候选计划f0f3aec5...实际构建exit0，用时3713.973秒；全2000分配与原e8a127...authority完全一致。128几何选择与计划参数release4c128...前128逐项相同，0回退，独立点误差最大9.0949e-13px。父启动器新增已批准身份快路径，实际2000核对10.031秒，错误expected SHA拒绝，原子进程全量检查未移除。r002已原子启用，仅允许原index0..127。一次误传PLACEHOLDER哈希的非执行调用在队列前被拒绝，记录在PRE_DISPATCH_REJECTION.json；更正为真实quality SHA后，session49671已入队并启动pilot2000_v006_preview_s0/s5/s2/s3_v001四任务，各4条共16条，GPU4–7、2threads/profileTrue。新缓存启动器实际GPU全流程结果尚待，不能把运行中写成完成。


### 2026-10-21T14:47:05.178153+08:00 — 20分钟检查提醒与独立优化候选

用户要求的20分钟检查已由应用创建，ID `action4-20`，首次实际检查后停用。现有profile128调度器及四个渲染任务保持运行，未重启或改写其绑定源码、配置及输入。最近检查尚无RGB产出、无作业失败；不得把存活或GPU整卡活动当作实际图像通过。

在 `outputs/action4_bridge_read_reuse_candidate_v001/` 独立实现和检验了桥接引用读取缓存。真实当前bridge输入的完整输出回执与原实现一致，9项不变量/反例检查通过，关闭边界仍强制读取并验证全部依赖SHA。两轮对照的原实现读取510次，候选483次（减少5.294%）；耗时受并行负载影响，未宣称端到端渲染提速。由于收益有限且没有消除每个renderer重复完整2000条校验，主控决定保留为未启用候选，不为其迁移当前正式来源。正式src/scripts与运行中的执行入口均未改变。详细数据和限制见该目录REPORT.json。


### 已生成图片与当前批次的复用边界复核 — 2026-10-21T14:51:41.520173+08:00

用户指出之前已生成图片。实际旧v015共1093条manifest记录，图片存在数1093，去重sample_id为1067；native128子集128条。不得把当前批次0RGB表述成项目历史未生成图片。旧准入在2026-10-19已因安装关系/比例/环境问题撤回，原recipe固定安装角按family变化，当前为palmar_wearing模块装配，且双机配额已调整，不能只改manifest把旧图当新版数据。此复核不宣称逐图不可救回，也未重新视觉验收旧图；保留历史成果，继续复用CAD/原视频帧/当前2000参数及128几何。源码或哈希差异本身不构成重渲染理由。清单与边界见outputs/action4_existing_data_reuse_audit_v001/REPORT.json。本次没有启动新生成任务。


### 2026-10-21T14:54:52.726601+08:00 — 共享环境准备消除重复计算的候选

在当前2000参数的绑定中查得918条桌面/衣物样本共享完全相同的四项静态环境输入。独立候选复用现有_PlanIndex按批次缓存环境准备结果，3条真实桌面/衣物样本输出逐项等于原environment.prepare；原4.971754秒，候选2.058869秒，额外结束依赖fresh SHA验证0.483448秒。这里只证明静态组件优化，不是端到端生成提速。返回值修改不污染缓存、错SHA拒绝、已seal上下文拒绝均通过。正式集成补丁仅改变dependency_refs/_dependency_refs并增加静态准备helper，其余21函数AST未变，逐样本姿态/几何/mask逻辑未删。源码仍冻结，未安装，尚需集成级验证和受控来源切换。证据outputs/action4_static_environment_reuse_candidate_v001/REPORT.json与integration.patch。128几何始终已完成，不重跑；当前4渲染任务仍在原有流程。


### 2026-10-21T14:58:25.226291+08:00 — 静态依赖复用集成对照与来源批次缓存

共享环境候选在3条真实composition依赖路径中完成对照，返回依赖maps逐项一致。原18.962084秒，候选16.656495秒另加0.459080秒结束校验；因此不能把局部environment组件的约一半耗时下降外推成完整链路加速一半。进一步定位sampling.verify_sources每次重复读取相同125冻结来源；独立候选仅在已有_ScopedPoseContext/owner内使用check_ref复用，普通dict保留原逐次完整读取，结束仍fresh验证。真实连续10次原4.456087秒，候选0.585510秒加结束0.472011秒；篡改context和已seal上下文均拒绝。首次反例脚本deepcopy整个准备context因包含module失败，已保留失败脚本与记录，改为只复制source_context后验证通过。两处正式源码均未替换，当前运行来源保持不变；新增source_scope.patch的接入需要受控来源发布，不能直接热替换。


### 2026-10-21T15:09:03.078643+08:00 — 后续链路99检查及真实网络标签梯度证据

现有checkpoint、training/eval、eval lineage/dispatch、video queue、video IO、workers共99项CPU工程测试通过（31.559秒），FFmpeg七帧真实编码/时间戳检查包含在内。此为工程测试，不能充当真实模型精度。没有正式训练启动、没有产生正式checkpoint。

真实当前合成ROI与正式ResNet50配置（decoder256/head64、ImageNet权重、ROI256/HM64）对照：两隐藏点仍valid，8通道logit梯度全非零；改corner0的x标签16px改变目标、loss和梯度；全部valid关闭loss/梯度为0；backbone/decoder/corner head经实际AdamW更新。一个诊断update loss0.016994832→0.016794236，不是收敛结论。首次测试假设单点扰动仅改一个通道错误：当前带宽依赖8点中心，其他热图宽度会变；已保留失败并修正测试假设，正式代码未改。

实际full768×1024+2ROI的B0型联合计算也通过：center/offset/box/heatmap四项均参与优化，四head均有非零梯度和更新；角点loss到共享conv1梯度L1=0.00840168，检测项=34.9898567。该诊断从ImageNet开始、不是正式A0后B0，梯度比例只提示初始检测项主导风险，不据此修改权重。ROI选择detached，角点loss到检测head无直接梯度是明确结构，检测由自身监督更新。正式阶段需观察GT-ROI与完整RGB分开eval以识别检测/角点瓶颈。

当前video仍association=None/smoothing=None，raw8独立top20均值解码，不得称刚体约束或时序锚定已实现。详细证据outputs/action4_downstream_chain_check_v001/REPORT.json及outputs/action4_corner_gradient_evidence_v001/{REPORT,JOINT_REPORT}.json。每20分钟监控已更新automation action4-20，不再持续轮询合成。


### 2026-10-21T15:16:19.640476+08:00 — 遮挡指标与固定长方体投影候选

在outputs/action4_corner_strategy_prepare_v001中独立实现occlusion_diagnostics.py和rigid_projection.py，未热替换正式代码。现有8真实标注图分层83画内可见/27画内遮挡/2画外角点。构造明确标注为控制夹具的GT复制预测，只对隐藏点加100px：隐藏PCK下降至0且EPE100px，可见点仍1/0px；全部漏检PCK0、EPE null且保留全部GT分母。缺帧身份/visibility语义拒绝。它们不是模型性能成绩。

固定8点刚性候选使用现有NumPy2.5.2/SciPy1.18.1；OpenCV导入失败因此未调用其API或增加依赖。14真实canonical导出实例的已知2D投影恢复最大0.000272205px；受控1px噪声样本残差最大1.303439px，500px错误点和全部低置信输入拒绝，错误corner_ids/非零畸变/unknown K状态拒绝。无尺度拟合、无点序重排、raw输入保持。默认仍raw展示。官方PnP方法说明参考https://docs.opencv.org/4.13.0/d5/d1f/calib3d_solvePnP.html；当前代码是独立的SciPy候选，不是OpenCV实跑。K仍估计值/零畸变假设，几何一致不证明模型精度或跨帧稳定性。详见OCCLUSION_REPORT.json、RIGID_REPORT.json、IMPLEMENTATION_STATUS.json。


### 2026-10-21T15:20:14.324006+08:00 — 保守双机关联候选

outputs/action4_corner_strategy_prepare_v001/temporal_association.py完成8项控制检查：顺序反转仍保持匹配ID、输入不改、漏检不画旧框且短暂返回可关联、远处新目标新ID、歧义交叉不声称连续、重复frame拒绝、超过双机拒绝、过期ID不复用。无真实模型/视频结果，未安装或开启，角点坐标不变。关联不能解决所有交叉/长遮挡，也不能赋予真实左右身份。详见TEMPORAL_REPORT.json。源码整理边界核查发现dispatcher会绑定整个src/*.py集合，新增src文件也会改变后续profile身份，因此当前仍将候选集中在该目录，未为了整理破坏活动执行来源。


### 2026-10-21T15:30:25.192612+08:00 — 修正可见性语义及边界诊断

用户提醒可见/不可见训练区分。核查action4_global_render.project_canonical及action4_canonical_export确认corner_visible实际为虚拟包围盒点ray-clear与原生inframe，不是物理表面角点可见性。前面报告的83/27/2计数应解释为画内射线畅通/画内射线受阻/画外，先前物理可见性措辞不准确；不意味着原标签或梯度接入错误。已在独立候选新增corner_support.py共用分类，核对ray/inframe/尺寸/点序；修正候选此前[0,W)边界为exporter的[-.5,W-.5]，原始数据与正式训练代码未改。5项语义/边界控制检查通过；遮挡诊断与覆盖测试重新通过，旧措辞报告保留为legacy_visibility_wording供追溯。不能用6角点硬上限删隐藏监督。

离线后处理CLI已实跑3控制帧/4实例，原始预测保留、漏检不补框、固定canonical与估计K来源绑定、几何候选单列；错误原始时间轴在发布前拒绝。POSTPROCESS_REPORT.json及GEOMETRY_CLI_REPORT.json为工程夹具检查，非真实模型视频效果。coverage.py已对当前诊断8图给出大小/遮挡/截断/对象局部朝向分层，区分declared/usable双机并验证重复ID、组跨split和test非双机反例；诊断8图不承担2000配额验收，不据其87.5%要求重生成。


### 2026-10-21T15:33:16.693663+08:00 — 覆盖统计接入已有统一验证结果

coverage_cli.py已消费实际8图的统一检查报告（d121441e...）及其精确manifest引用，未重读RGB/mask。诊断模式quota结果为null，不宣称正式配额通过；将实际8作为preview128时明确失败wrong_sample_count/usable_dual_below90percent，错误报告SHA在输出前拒绝。128模式检查数量、实际可用双机≥90%、test双机、组隔离及计划目标未丢失；2000模式额外核1600/200/200与54/90/1856。尚未用未完成128/2000数据冒充通过。coverage计数进一步区分planned_target_count、target_count、实际可用实例和excluded_instances。证据COVERAGE_CLI_TESTS.json与coverage_cli_actual8_v002/REPORT.json。上一次20分钟间隔观察时4原Blender仍存活、CPU约59分钟、0文件、无失败，未重启；之后继续后续链路工作，未持续轮询。


### 2026-10-21T15:36:52.294211+08:00 — 实际完整网络权重保存/加载闭环

使用当前decoder256/head64的ResNet50、既有ImageNet初始化、实际合成ROI256，按原save_checkpoint_atomic保存后通过原build_model严格重载。8通道logits逐位相同（最大差0），top20有序8解码也相同；网络结构变化、预处理变化拒绝。该权重显式component_diagnostic/global_step0/epoch_boundary=false/resumable=false，正式视频拒绝，即使allow-component也不能写final_inference。未进行正式训练。唯一诊断权重为models/action4/component_checkpoint_roundtrip_v001/component_weights_only.pt，138686852字节，SHA8c8a7262583deb7a797b8e85d059eebf7ef09741cc17f5b82286528dfa211f6f；不作为D0/A0/B0初始化或恢复文件。工程报告outputs/action4_downstream_chain_check_v001/REAL_CHECKPOINT_REPORT.json，optimizer/恢复语义另由既有99项测试覆盖。


### 2026-10-21T15:42:49.960730+08:00 — 部分可靠点与一致错误姿态反例

刚性候选增加14实例受控实验：以GT虚拟射线标记人为指定高/低质量，保持高质量点原位，其他点统一偏移(12,-9)px。14几何候选均返回，受扰点平均误差均较15px降低；这使用oracle质量标记，只是条件性几何试验，不代表模型能识别可靠点。更重要的反例：将同一机身在camera-X平移0.03m后投影，作为全高置信错误输入，候选几何残差仅5.33e-6px仍通过，但相对原始真实导出投影平均误差143.283px。证明刚性一致不等于图像定位正确，不能把几何检查当模型准确率或遮挡推断成功。证据outputs/action4_corner_strategy_prepare_v001/PARTIAL_SUPPORT_REPORT.json；未调整准入阈值、未开启rigid显示、未改训练策略或合成输入。


### 2026-10-21T15:48:18.540905+08:00 — 当前128批次首次实际RGB产出

按20分钟间隔于15:46:32检查，原4job继续running，RGB为s0=3/s5=1/s2=2/s3=3，合计9；峰值显存3236–4522MiB，原worker/Blender PID/starttime未变，Host报告尚未闭合。未重启或更换来源。root查看四个已打印SYNTHESIS_SAMPLE_RENDERED_PENDING_BATCH_CLOSE的首样本RGB（0/5/2/3），双机及墙/桌/衣物/近双机可见；不是角点/掩码验收，也不是当前128通过。粗手/前臂仍属已声明近似，不重开无关精细建模。日志含Render Result删除用户数提示但未见终态失败，待原关闭校验判定，不靠单独ERROR文本停止任务。证据outputs/action4_pilot2000_dispatch_v006/profile128/PARTIAL_RGB_INSPECTION.json。


### 2026-10-21T15:53:08.441489+08:00 — 同来源后续112入口准备

只读核查旧refresh_v006发现其为五源历史发布器，含新诊断8计划与来源包装重建，不直接套用当前任务。未执行发布、未改源、未重做2000参数/128几何。当前同v006 finish128的条件批准文件与命令已排他保存，现阶段未入队；原driver强制等待所有现存任务终态与四个当前profile conclusive后才能入队112，合计仍是原128，不授权1872或训练。证据outputs/action4_pilot2000_dispatch_v006/FINISH128_PREPARATION.json。


### 2026-10-21T16:38:00.158562+08:00 — 用户指出异常耗时后确认流程性能缺陷

不能再把进程存活当作继续等待的充分理由。每4图分片在opening/closing重展完整2000依赖图；首图在进程启动约65分钟后才出现，当前16图最迟16:00:33写成，任务16:35左右仍在CPU关闭校验。各分片日志可统计渲染完成段合计63–74秒，不能把两小时称作RGB渲染时间。root先前持续等待并未有效解决瓶颈，已明确纠正。

关闭快路径候选真实8计划对照：opening保持33.16/33.62秒；closing由33.0214秒降到0.547822秒，同313引用和完整source_vector SHA一致。先检查资产map完整登记，再fresh读取全部依赖及计划，去除同字节依赖图语义重建。11个控制反例通过，生产尚未安装。该候选不是跳过校验，也不是新图像接受。

已保留原112批准副本，撤回当前准备入口为WITHHELD_PENDING_VALIDATION_PERFORMANCE_FIX；旧执行命令在质量SHA/status门失败，未入队。现有4任务、16RGB和全部参数/128几何不动。下一步优先修复/验证批量流程再追加分片，不继续扩散旧全量重复校验。证据outputs/action4_pilot2000_dispatch_v006/PERFORMANCE_DEFECT.json及outputs/action4_static_environment_reuse_candidate_v001/FAST_CLOSE_REAL8.json、FAST_CLOSE_COUNTEREXAMPLES.json。


### 2026-10-21T16:53:24.903771+08:00 — 逐样本重建根因与常驻基础场景实际Blender验证

调用链确认每entry重新wearing.create→bilateral.create_scene→composed/neutral.create_scene→open_mainfile，并重新追加左右腕带/安装件/相机/线缆；并非场景常驻只调参数。composition前后重复dependency_refs和多轮快照/验证也增加CPU成本。

原导出器对固定CAD+已记录实际矩阵的双机16点投影，32次有界计时中位0.000539195秒，最大对旧导出差0.000126015px；明确不含射线、快照、mask和scene加载，不能把其周边耗时归给角点矩阵计算。

独立真实Blender进程完成PersistentWearingBase原型：metadata3.203s，基础装配一次33.489s；连续两个原参数应用含原collision/pose验证16.968s和18.075s。保持相同scene对象指针，原生角点对旧实际导出最大差0.000141511px。每次只恢复到wearing模板，base.restored仍false，不假称已恢复原场景；批次末真实base.restore一次并验证通过。render_pre观测0次，未调用GPU/渲染，未改旧图片/标签。当前只证明基础场景/姿态复用；组合层常驻与实际RGB快路径还待接入验证，生产尚未安装。证据outputs/action4_persistent_scene_repair_v001/REPORT.json、PURE_PROJECTION_TIMING.json，原旧112批准继续撤回。


### 2026-10-21T18:43:42.640985+08:00 — 旧慢路径收尾与新完整RGB路径实测启动

旧4分片的Blender实际已完成16图。仅在render_completion完整、独立FastHost16通过、目标Host无Blender子进程后受控SIGTERM旧Host，队列保留failed/exit -15及inconclusive profile；原16RGB、掩码、manifest未改写。新Host真实发布4份geometry_validation：opening56.444s、16条验证51.489s、closing2.710s，仍为视觉验收待完成。证据 outputs/action4_fast_pipeline_repair_v001/HOST_PUBLICATION_REPORT.json 和 OLD_HOST_CONTROLLED_STOP.json。

真实Blender性能根因：原lift组合检查196.009s主要消耗在反复mesh快照、JSON序列化与静态读取；显式memo候选同几何组合47.869s，原生角点最大差0.00015459px。基础创建36.390→12.465s，参数应用22.060→8.583s。均为带剖析的组件时长，不是完整RGB吞吐；对应 lift_profile_v002 与 lift_memo_v001 原始报告保留。

新实现已放 src/data/action4_execution_cache.py、action4_runtime_memo.py、action4_persistent_base.py、action4_fast_generation.py、action4_fast_launcher.py，薄入口 scripts/action4_render_fast.py。独立Astra审查指出初始化/异常恢复及新实现closing引用缺口，已修正并补5项生命周期/入口测试；原cache/authority/dispatcher75项通过，共80项。首次误用pytest因环境无pytest失败，随后按已有unittest运行，无安装依赖。失败日志保留。

r003已激活；首4张新样本通过dispatcher profile入队并启动，job pilot2000_v006_fast_s1_v001。此处不宣称新RGB已通过，也不宣称128或2000已完成。旧112批准继续WITHHELD，需新路径实测后重新明确放行。


### 2026-10-21T18:52:05.312211+08:00 — 快速生成首两组真实完整链路通过

快速分片1：4图/8实例/64角点，worker414.572s、独立Host73.967s，最大投影误差0.000141511px。分片4：4图/8实例/64角点，worker273.471s、Host73.226s，最大误差0.000153535px。两组均由canonical dispatcher记为succeeded，真实render_completion、geometry_validation与fast_launch_timing已落盘；基础装配批末恢复、runtime适配恢复和fresh source closing实际通过。原16图另行保留，新8图验收不替代全128视觉与数据流验收。证据 outputs/action4_fast_pipeline_repair_v001/FIRST_FAST_NATIVE_END_TO_END.json。

续跑控制器1188286只在4个profile全部succeeded/conclusive且已有32图报告和manifest一致后释放96个未生成ID；CPU每进程3、总预算72，显存按最大实测profile加25%余量。旧scheduler退出后由唯一scheduler续接，失败不重试；128完成后运行统一验证并等待root视觉检查，不自动越过1872/训练门。当前正式训练未开始，2000接受数仍0。


### 2026-10-21T19:35:38.864086+08:00 — 首128接受、GPU3组件通过与两线程生产启动

首128的36个分片全部完成，其中旧4分片保留受控失败及独立Host恢复记录，其余32快路径任务succeeded。统一检查40.017s，128图/250实例/2000角点，native→ROI→HM→native最大误差5.08423e-13px。root实际查看sheet01–08全128图及两机局部：框覆盖机身、转面/遮挡/截断未缩框，负样本无dummy框；重复基准视图与合成外观局限已记录。覆盖门通过，left/right分别23/30种按0.001px量化的投影；不据此宣称真实视频泛化已证明。原root接受状态不等于2000训练接受。

GPU3专用设备真实UUID核验通过，13项实际CPU/CUDA/真实双机与零目标对照及错误点序、1px偏移拒绝测试全部通过，耗时12.460s。实际GPU回执包含299613个真实mask像素核验。并发/权威/cache的77项CPU测试通过；因将8GiB额外余量改为3GiB，原两个测试的固定数值边界失效，已改为依据HEADROOM表达相同的不足/精确容纳反例，未删除断言。原失败日志与修正日志均保留。

当前r005，控制器1229139、scheduler1229212，quality job quality_gpu3_remaining1872_t2_v001。先16新图profile+GPU3 gate，之后自动1856。运行交接 outputs/action4_fast_pipeline_repair_v001/REMAINING1872_T2_PROCESS_HANDOFF.json；旧3线程continue_remaining1872.py仅保留准备历史，不执行。当前正式训练尚未开始。

数据链路优化候选：assembly复用草稿12项CPU反例通过，尚未生产安装；训练视图在一个真实候选上由3518776字节降至20790字节，原图ROI、检测目标、全8热图及valid逐张量相等，篡改角点被拒绝，原/轻量候选仍被正式loader拒绝。它不是正式数据转换或完整2000证明。原丰富审计记录保留。


### 2026-10-21T19:59:01.455370+08:00 — 多样性缺陷确认与受控撤回

用户观察近重复后，root审计全2000参数文件，确认不是渲染缓存复用了错误图像，而是输入几何大量相同：仅113状态，table_fastening440/1、table_turn383/1、forearm305/2、cloth95/2、truncation139/2、negative54/1。source1与37几何控制和两机矩阵相同，只改gain/exposure。根因在旧schedule photometric_only及至多一项毫米/1°扰动；旧pose控制typed硬界±4mm也不足。此前把标注/几何/配额通过提升为批量准入，缺少有效多样性门，判断不充分；已明确纠正，不把旧root接受回执静默改写。

19:42左右仅停止身份核实的放行controller1229139，1856尚未入队。四个2线程profile自然完成16图，GPU3四个子报告全部通过（16新样本）；随后核实quality active_ordinals为空，受控SIGTERM其idle child subtree，wrapper保留failed/-15。全部队列任务已终态，旧144图和证据保留。权威改为r006 diversity_repair，旧render/quality/train均禁用；goal仍推进参数修复，不是用户暂停整个任务。

证据：outputs/action4_fast_pipeline_repair_v001/DIVERSITY_AUDIT2000.json、DIVERSITY_RELEASE_HOLD.json、DIVERSITY_QC_CONTROLLED_STOP.json、DIVERSITY_ACCEPTANCE_WITHDRAWAL.json。新纯数值采样模块正在 outputs/action4_diverse_sampling_v007/ 准备；没有声称新宽域已通过实际场景或训练。


### 宽域29维采样实现与本次数值验证（2026-10-21T20:45:19.278940+08:00）

状态：参数/几何控制已实现，Blender变体adapter仍待接入；未生成新版RGB，未做新范围实际碰撞或实际mask测量，未放行生产/训练。

新增 src/data/action4_parameter_sampling.py（在已有候选实现上扩展）、configs/data/action4_random_sampling_v004.json、tests/test_action4_parameter_sampling.py。旧outputs候选与旧图保持不动。按用户明确的几何安全定义，移除拟议的双机间距/深度差限制，不将遮挡目标区间命中或参考视频相似度作为安全条件；未关闭原有生产工程校验。

实际21项CPU测试通过，0.228秒，覆盖固定安装/尺寸/点序、左右独立运动、可复现联合随机、三轴独立范围、背景独立选择、缺失目标无伪GT、遮挡各参数改变实际候选几何、遮挡片capture/world坐标及反投影、无效输入拒绝。测试夹具与正式数据明确分开。

使用3个现有模板的真实anchor与canonical8产生128组参数，读取anchor时一次验证来源；抽样/写出约0.4835秒，39墙面、34桌面、55提起衣物。左深度0.400672–1.397253m、右0.400931–1.386374m；左u0.080698–0.919337、右u0.096068–0.914741；左右v覆盖约0.12–0.90。左/右预计bbox面积占整图分别约0.130%–4.253%和0.128%–3.157%，是数值投影而非实际mask面积。2048个有序角点数值可计算，重算投影误差0仅证明同一几何数据的计算一致性，不是独立Blender/RGB正确性证据。没有逐对比较或全库相似度过滤。

证据：outputs/action4_diverse_sampling_v007/broad_joint_v004/REPORT.json 与 parameters128.jsonl。r006 hold保持；下一项按plan完成模板adapter与最终工程检查，不能把参数文件计为合成数据完成数。


### 当前固定模板接口、缺陷修复及真实输出证据（2026-10-21T21:17:03.411802+08:00）

状态：固定场景参数/Blender接口、增量碰撞、候选batch组件已实现并经规格后质量复审通过；新preview128尚未完成，accepted2000与训练未开始。

固定场景实际128次apply/export：wall43、table43、cloth42，每worker一遍模板加载；最大实际Blender与数值投影差0.000683225px，单workerapply+export0.739–1.131秒。证据outputs/action4_diverse_sampling_v007/fixed_scene_v005各REPORT。随后3张原生RGB+mask实测17–21秒/图，旧环境导致5/6ignore；原文件保留于fixed_scene_rgb_v005，不能称通过训练质量。修正固定环境后的相同3组参数6实例均进入ROI，48/48角点支持，maxroundtrip4.68743e-13px，见fixed_scene_rgb_layout_v006/HOST_CONSUMER_REPORT.json；root已看全景+角点crop，第三张实际碰撞由后续组件正确检出，该诊断不纳入接受集合。

独立审查修复：proposal frozen source/A/S/F/canonical绑定；NaN/Inf/少顶点先拒绝；两处同步错误D也拒绝；材质借用来源记录；原输出目录异常不能向旧COMPLETE追加FAILED。采样21项、runtime contract6项、batch边界/归属6项CPU测试通过。碰撞适配器13项BlenderCPU测试、3项源变更反例、6条实际proposal当前源码证据在collision_adapter_v001/evidence_current_v002.json；6条为wall2PASS、table1PASS1UNKNOWN、cloth2COLLISION且各有UNKNOWN，单次0.138–0.377秒。未知保留为未知，不包装成128通过。

已准备4worker×2图profiles，配置configs/execution/action4_template_preview_v001；此记录写入时尚未启动。最终质量接口单独实现中，r007质量/训练未放行。阶段完整目标保持2000真实接受、单帧D0/A0/B0、eval/test、2850原视频结果及存储评估，不以本组件测试替代。


### 新128闭合、当前full1872及下游候选（2026-10-21T22:11:43.841265+08:00）

新版128已完成并经root视觉接受用于2000先导生成。全部250实例/2000角点经真实GPU3、原生mask与consumer检查，max native roundtrip6.43110e-13px。证据outputs/action4_template_preview128_v001/root_review/SUMMARY.json、ROOT_ACCEPTED128.json及38个明确quality报告。成功120原样复用；修复8槽完成36.40秒。两例无目标scene_family错误及其原FAILED、一个未闭合中间RGB均保留，不将失败改标成功。

首4worker连续8图耗时38.5–46.6秒，GPU峰值3236–5012MiB。后续28worker的120槽调度历时332.67秒，112槽完整成功，两个失败分片涉及8槽随后补齐；不得用该带失败计时冒称128完整吞吐。候选不是模型预测，未完成正式accepted2000或训练。root全部128视觉记录明确合成简化与真实域效果未知。

r010及fullplanv009已经启动剩余1872与GPU3 streaming quality；当前进度以outputs/action4_template_full2000_v001中的实际进程和PROGRESS/QUALITY_PROGRESS为准，不把启动或排队当完成。每20分钟心跳action4-20已更新当前阶段，只在有意义变化/完成/失败时通知，不自行重启旧方案。

下游独立候选：assembly19项合成工程测试通过（0.046s），split protocol9项CPU测试通过（4.372s）；只写入各自outputs候选目录，未改live src、未实际发布正式数据或训练。CPU测试期间仅短暂暂停质量admission协调器以保留预算，既有GPU质量子进程和全部渲染不暂停，测试完已恢复协调器。候选仍需主控规格/质量审查和实际2000链路验证。


### 正式数据发布与新D0启动（2026-10-21T15:30:41.364138+00:00）

剩余1872的234个render分片全部成功，用时4199.364秒；其quality全部成功，用时4224.440秒。合并已接受preview128，实际272份batch质量证据、2000唯一槽位，1600train/200eval/200test；双机1856、左单机45、右单机45、无目标54，test全部双机。3802实例/30416支持角点，14286不可见角点和9个画外角点保留有效ROI监督，最大native roundtrip9.374856803373542e-13像素；这是标签/坐标变换证据，不是模型精度。

数据正式发布完成：data/train、data/eval、data/test及data/action4_dataset_commit.json（SHA915d620fbd7640b4e5e0ba579ef45112cbb262e203175a95e39acd7baad8fbd0）。9604个RGB/mask全部硬链接，源原样保留；实际像素15136058162字节，约14.0966GiB。最终17031输入来源闭合，不重新解码PNG或生成图像；三目录后统一commit已经完成。存储数值只代表选中像素，不冒充全项目/最终训练视频容量。

根与独立SPEC正式批准精确绑定同一ledger/来源/安装source/relocation/共享模板协议；root复用原128张已实际目检且SHA相同的图像，未把旧pilot回执直接冒充formal approval。selected2000与现实视频域效果仍不同，partial arm ends、平面遮挡片和三模板共享限制保留。

接口合入后，23项authority测试通过。受影响consumer套件53项最初9项错误、1项CUDA跳过：错误均为新增gate绑定要求config.data，破坏旧最小方法配置调用。root在gate调用处恢复缺失data时旧模式，无任何显式协议字段的降级，未改已发布split validator。修正后12项gate测试全部通过；其他原通过项未无意义重复运行。失败日志保留consumer_tests.log，修复日志gate_regression_tests.log。真实2000配置build_config及最终pixel/manifest admission已通过，r011真实authority验证12条引用通过。

当前四卡D0任务template2000_d0_v001已经由正式Dispatcher启动，GPU4–7，fresh ImageNet初始化、32ROI、最多500更新；运行不是D0通过。A0/B0、正式eval/test、2850帧原视频输出及最终总存储仍未完成。没有push/commit，无时序训练。所有新证据在outputs/action4_template_publication_v001与outputs/action4/template2000_d0_v001；模型结果在models/action4/template2000_d0_v001。


### GPU2–7及训练性能修正（2026-10-21T15:53:20.655001+00:00）

仅在项目进程中设置NCCL_P2P_DISABLE=1，不改服务器全局配置。默认四卡allreduce实际超时，P2P-disabled四/六卡求和分别正确为6/15；先确认通信再进行长训练。原D0首次集体通信前后未有训练步/ckpt，保留failed与受控停止证据，v002是fresh D0，不冒充resume。

Action4Dataset保持完整PNG解码、尺寸、角点/标签及RGB SHA检查，只在构造时使用ThreadPool并行。D0固定32ROI源图用最多32项LRU缓存，返回clone，文件inode/size/mtime/ctime变化立即拒绝缓存，不能隐藏输入修改。A0/B0每rank10个loader进程；主训练rank2个计算线程。串行validation共享主线程预算，多线程validation与loader阶段不同时运行，调度器按较大阶段计算总预算。六rank采样按ceil划分，真实窗口样本各一次、尾空rank零权重，实际平均梯度经CPU反例与尾批对照验证；原global batch、LR、总步数/epochs不改。

真实32图预检查4.9579s串行/1.9453s十线程，缓存填充7.2200s/缓存命中及逐张精确比较0.5762s。只证明当前文件缓存状态下组件提速，不是整体训练加速倍数。报告outputs/action4_training_performance_v001/PERFORMANCE_CHANGE_RECEIPT.json。

实际状态：v002已入队/启动，尚未声明D0通过。所有93项测试的初次失败、修正及64项复验记录保留；不把工程测试替代正式训练收敛。检索历史测试时grep -R误跟随越界软链接，已中止、不采用返回内容、无目录外修改；记录SEARCH_BOUNDARY_CORRECTION.json。


### D0失败根因与AMP几何修复（2026-10-21T16:29:08.042489+00:00）

D0 v002完成500有效更新、无nonfinite、六rank最终权重一致，骨干非零梯度且参数更新；最终gate失败，P95约27.8热图像素，后续A0/B0未启动。之前多次运行中检查属等待，不代表收敛通过。原检查点与日志保留。

最终固定32热图CPU诊断：top1/5/20的P95分别26.656/28.219/27.221，换解码不能消除问题。实际同权重GPU对照定位到AMP作用范围错误：prepare_rois中的destination @ inverse.T在外层autocast下降为FP16，native坐标量化改变采样图像，但numpy角点目标不变。FP32平均loss0.009563728，仅模型AMP0.009563282，几何与模型均AMP0.000105394；ROI像素最大差2.436695、标签差0。训练因此拟合了与评估不一致的裁剪输入。这是现有训练工程缺陷，不能归咎于数据量不足或不可见角点。

最小修复：仅ROI坐标矩阵乘法、grid构造和grid_sample置于autocast disabled且FP32，网络继续AMP。实际相同32ROI重测像素/标签差均0；两条AMP的loss均0.009563282，FP32一致至数值精度。24项受影响回归通过（1CUDA-only skip），新增CPU BF16外层autocast坐标反例通过。正式数据、标签、固定ROI、LR、全局batch、500更新阈值未改；没有重新生成数据或弱化gate。

当前r013，六卡GPU2–7/72CPU新D0 template2000_d0_v003按原v005配置fresh ImageNet初始化；旧v002不能作为gate或exact resume。条件接续outputs/action4_training_performance_v001/continue_a0_b0_v002.py，后续runIDs A0/B0均v002；必须等待新D0实际通过。最终test脚本尚绑定旧B0候选，正式执行前需要更新为本轮最终选定权重，禁止直接执行旧入口。证据：failed_d0_heatmaps/REPORT.json、amp_geometry/REPORT.json、amp_geometry_fixed/REPORT.json、amp_fix_regression.log、amp_geometry_regression_test.log。


### AMP几何修复后的真实D0通过

D0 template2000_d0_v003在127/500步正式提前通过，完整d0_gate.json已经validate_d0_gate复核，P95=1.8223980075热图像素、均值0.3667967539，32ROI/256角点，六rank最终权重一致、nonfinite更新0。原source/data closing完成，Dispatcher succeeded。上一轮相同数据方法但AMP坐标错误的v002失败保留。此结果证明固定32训练样本及真实训练几何链路可拟合，不是eval/test或原视频泛化结果。后续A0 fresh ImageNet两轮，使用这个gate而非D0权重；B0从A0权重另起优化器。


### A0两轮完成与B0接续

A0 template2000_a0_v002完成190有效更新，两轮，nonfinite0、六rank权重一致、Dispatcher succeeded。固定eval GT-ROI P95分别50.1706/50.4394，mean27.1720/26.2667，定位仍弱，不能以执行完成冒充精度达标。按既定eval P95选择第1轮best，未读取test调参。B0 template2000_b0_v002已经启动，五轮全图检测与角点联合训练，从A0 best仅加载权重并新建optimizer；D0通过并不保证本先导泛化成功。


### B0梯度缩放溢出与新配置验证

B0 v002第1次联合更新成功，center/offset/box/heatmap四项实际参与，随后第2次出现非有限梯度，六rank同步停止且未继续optimizer/scheduler。原失败及1条有效更新保留，不把partial作为可恢复epoch。实际A0best与前2全局窗口重放：scale8192第2窗口rank0非有限，scale512两窗口六rank局部梯度均有限；这是一张GPU重放局部梯度，不冒称完整DDP通过。初次诊断遗漏CUBLAS deterministic环境导致运行失败，第二次补齐与原训练一致的环境后完成，原日志保留。

唯一正式配置变化runtime.amp_init_scale8192->512，configs/model/action4_pilot2000_v006.json；源码、数据、ROI、LR、batch与预算完全不变，max_nonfinite_updates仍0，不跳过无效批次。旧D0绑定旧scale，因此r014按原标准fresh D0 v004验证新数值配置，之后直接复用已完成A0v002权重启动B0v003新optimizer，不重跑A0或数据合成。条件接续continue_b0_v003.py。完整目标仍为五轮B0、eval/test、2850帧真实视频及效果审核；不以本对照代替最终结果。


### FP16缩放512仍失败，改用FP32数值基线

D0 v004在127步正式通过，P95=1.9186351503、平均0.3675647058，六rank一致且nonfinite0。复用A0v002的B0v003实际完成73次有效联合更新，之后再次非有限梯度同步停止；这证明此前2窗口scale512对照只解决早期反例，不能证明完整训练稳定。原failure_step73及全部日志保留，partial epoch不可exact resume。

r015/v007仅将runtime.amp从true改false，采用FP32训练，停止试探更多FP16缩放值，严格非有限检查保持0容忍。现有源码支持FP32路径，不新增代码；GPU2–7实测空闲约47–48GiB，单卡预留32GiB并实际profile，不把空闲显存视为必然够用的证明。按数值配置绑定要求新D0v005验证FP32，随后复用A0v002权重接B0v004，不重做A0或数据。后续脚本continue_b0_v004.py，test准备run_final_test_v004.py。此前临时入口保留为历史，不自动重启。


### FP32 D0通过

D0v005实际125步提前通过，P95=1.9075603080、mean0.3659246618，完整gate复核通过，六rank权重一致、nonfinite0、Dispatcher succeeded。该数值配置下后续B0v004复用原A0v002，仅新建优化器，不重做数据或A0。固定32通过不能作为泛化结论。


### 先导失败后的有界角点预训练完善（2026-10-21T17:55:23.906881+00:00）

FP32 B0五轮1000更新完成，六rank一致、nonfinite0、同步吞吐14.5734img/s；五轮固定eval@5%联合F1均0。正式冻结test200双机/400实例，TP0/FP373/FN400，先导未达到完整锚定要求。原视频全片仍在生成；root已查看实际frame0，两台清晰可见但模型无预测，不能交付为成功。

同一正式train均匀128张/239ROI诊断：选定best(epoch1) mean22.9569/p9538.0748，last(epoch5) mean20.1026/p9537.8604。训练样本本身也弱，不支持仅归因域差或立即扩增数据。基于这些train/eval证据，在原模型与2000数据基础上恢复正文原A0十轮，只改变A0epochs2->10；四卡micro8替代六卡micro6，effectiveglobalbatch32保持，LR/损失/模型/FP32预处理/标注完全不变。配置configs/model/action4_corner_refinement_v001.json的training_revision明确覆盖旧server_extension的A0两轮预算注记，旧对象保留为先导来源协议，不冒称仍是原两轮实验。原D0方法绑定不变且入口仍实际验证，不重复D0。

r016仅授权A0，run template2000_a0_refine_v001；GPU4–7/48CPU，与GPU2原视频任务并行，总额度72内。A0 fresh ImageNet是既定入口，不拿D0固定32拟合权重作初始化，不重写已完成A0/B0证据。训练前声明：查看十轮固定eval与train诊断是否明显改善；不得依据test选择超参数，不自动追加B0或规模化采集，不放宽最终角点要求。正式test先导结果保留，后续若再用同一test必须披露重复评估，不声称全新盲测。


### 十轮角点训练完成后的实际诊断

A0 refine_v001十轮950更新完成，四rank权重一致、nonfinite0。固定eval mean8.7020/p9529.0229；相同均匀train128/239ROI诊断mean3.5252/p9515.4069。相比原B0训练子集mean20.10/p9537.86显著改善，但训练误差仍明显且eval差距大，不能将后续问题仅归因域差或盲目扩增数据。未自动追加B0；保留原视频输出与全部失败证据。依据为train和eval，未用test调参。


### 同2000角点优化步长单因素对照（2026-10-21T18:21:51.640590+00:00）

在十轮A0完成且train/eval仍有明显误差后，保留该基线，增加同预算单因素对照：仅A0新增层LR1e-4->1e-3；该量级在固定32 D0已能收敛，但不冒称全训练集也已验证。骨干LR1e-5、FP32、10轮、globalbatch32、四卡micro8、数据、网络、损失及seed均不变。使用同ImageNetfresh初始化，不修改旧ckpt。比较train128与固定eval完整八角点，选择只依据train/eval，不使用已看过的pilot test调参。配置corner_refinement_v002，run template2000_a0_refine_v002，r017仅A0授权，无自动B0/新数据。若失败保留并诊断，不提高非有限容忍。


### 新增层学习率单因素对照完成

A0 refine_v002十轮950有效更新完成，四rank一致、nonfinite0。只将新增层LR1e-4改1e-3，按固定eval选第6轮，eval mean5.28794/p9519.84090。相同train128/239ROI mean1.25001/p953.23167，较低LR十轮train mean3.52517/p9515.40694明显改善。训练学习问题缓解但eval仍有较大差距，不能宣布角点定位达标或把GTROI诊断冒充全图系统；没有自动追加B0或新test。原全部权重、预测与失败保留。


### 八角点刚性投影离线候选

outputs/action4_rigid_projection_candidate_v001准备固定机身尺寸的共同6DOF投影拟合，仅接收预测ordered8/confidence、fixedcanonical8、knownsyntheticK，禁止GT姿态/角点/可见性进入fit。normalizedDLT/有限姿态初值及Huber残差最小二乘；退化/残差大/未收敛时保留raw并标拒绝，raw始终保留。8项几何与非法输入测试通过0.750s，不是模型效果证明。GPU4/12CPU的固定eval GTROI对照已提交，拟合worker仅收到四项允许输入，真值只用于事后评分，未用test。真实视频内参未经实测，不能宣称该候选可直接提供真实6D准确性；正式src/infer未修改。


### 已改善A0后的有界联合训练与刚性候选对照（2026-10-21T18:55:09.612977+00:00）

刚性候选固定eval GTROI378实例，166拟合被接受，162平均误差改善、4变差；all8@5%33->117，84恢复/0通过退化。normalizederror平均0.08115->0.07584，p95保持0.29894，不能解决尾部大错误/漏检。raw始终保留，未安装正式src，真值姿态未进入fit，真实视频内参仍无实测。

基于完成的A0 train/eval改善，r018明确执行同2000五轮B0复用A0refinev002第6轮best，仅weights初始化、新optimizer；网络/损失/B0LR/5epoch/globalbatch8/FP32保持。四卡micro2是吞吐布局调整，不改变全局样本权重。配置joint_refinement_v001/run template2000_b0_refine_v001，GPU4–7/48CPU并行旧视频；已有D0方法绑定未变且入口继续验证，不重跑D0/A0/数据。此为有界后续验证，不将A0诊断当fullRGB通过，不自动新test；仍以完整eval和raw真实视频判断。


### 旧先导2850帧视频恢复验证与实际存储

原video job在全部2850预测与绘制后因裸ffmpeg不在PATH而失败，原失败保留；项目内runtime/ffmpeg/bin工具存在。独立encoder-only恢复使用原图片，12线程并行完整PNG解码/尺寸/hash准入，最终每张freshhash与完整MP4解码/时间轴验证保持；修复helper恢复函数引用的单元反例后5测试通过，初失败日志保留。恢复579.641秒，RECOVERY_COMPLETE明确accuracy_failed，2850帧、3248x2464、fps2997/100、timebase1/11988、每帧PTS/duration及输入未变化均verified。MP4358005055字节，SHAefea737d66d2ec1b5d45a5d5166cad912fb549d05e80f1abb8115b251ba903ed，本地副本一致。后续薄入口scripts/action4_video.py仅为新进程prepend项目工具PATH，未改server全局环境/训练src。

先导显式路径实际分配合计35.2582GiB：正式2000像素及标注逻辑14.1350GiB，real帧缓存9.1425GiB、A0权重0.7676GiB、B0权重1.6776GiB、旧视频drawn与预测9.1818GiB、编码与verification0.3340GiB；du -P不跟随软链接且共享inode在测量参数内只计一次。失败/诊断/后续refine额外运行不在此先导小计内，保留未删。仅像素扩展估计10k70.48GiB/50k352.41GiB，不能作为含所有checkpoint与video的总需求。报告PILOT2000_STORAGE_FINAL.json。旧模型97.72%帧无预测且六帧实际目检均漏检，视频IO完成不代表任务成功。

冻结A0共享特征候选5基础测试、真实3更新检测头更新且cornerlogits逐值不变、双rankCPU DDP机制均通过；当前仅outputs候选。已安排两组同A0权重/数据顺序/优化设置max64updates的component对照joint vs frozen，GPU3/4各12CPU。明确component_smoke=True，partialckpt不允许正式init/test，生产src未修改；不把诊断冒充五轮正式训练。


### 冻结特征64更新组件对照完成

同A0best/数据顺序/预算的两组组件均完成64更新：joint GTROI mean9.4483/p9531.5327，frozen_A0_features mean5.28794/p9519.84090，后者与初始化A0一致。两组fullRGB@5%F1均0，说明冻结能保持角点能力但尚不证明检测充分学习；不能提升component partialcheckpoint为正式模型。报告outputs/action4_frozen_feature_candidate_v001/COMPARISON_REVIEW.json，正式src未改，原失败保留。


## 2026-10-22 资源配置核验

实际检查发现r018仍为GPU4–7/4rank/48CPU，现已发布r019并更新active指针为GPU2–7/6rank/72CPU。修改前项目队列所有任务均为终态；本次未声称启动或跑满资源。既有训练配置每rank计算2、供数10、验证10与新预算一致，数据和训练超参数未改。已核对视频模块安装回归12/12通过；这些是工程测试，不代表定位准确率。后续训练须使用新源码对应门禁；同卡多任务按吞吐实测扩容，不将空闲显存等同加速保证。


## r020 冻结A0特征的正式有界验证

64更新组件对照证明冻结策略保留A0角点输出，而joint出现退化；两者当时完整检测F1均为0，不是成功证据。候选已安装正式src，46工程测试通过45/跳过CUDA1，视频模块12测试通过。新配置frozen_refinement_v001仅B0冻结backbone/decoder/corner_head，三检测头可训练，保留原数据/损失/LR/5轮/globalbatch8/FP32。先运行当前源码绑定的D0 v006；通过后从正式A0refinev002 best权重启动B0，新optimizer，不采用组件权重。GPU2–7六rank、CPU72。原D0不是失败重跑，而是trainer/gate源码变更要求新绑定；不重跑A0或生成数据。正式test不自动使用。冻结一致性、完整eval和真实视频仍需验证，不将冻结成功冒充任务完成。


### 冻结策略数据流边界核查

证据见outputs/action4_training_performance_v001/FROZEN_DATAFLOW_AUDIT.json。不可见但投影有效的角点在A0保持监督。B0冻结策略只更新center/offset/box检测头；corner heatmap损失为监测，不会再学习检测框偏差。ROI builder使用detach检测输出，因此corner损失也不通过裁剪反传检测头。完整eval必须同时看GTROI诊断与实际检测框下八角点，不能把冻结保持等同完整识别成功。未改变运行中的源码和配置。


### 当前源码D0 v006通过并衔接冻结B0

D0完成125更新并正常退出，固定32ROI/256角点，mean0.3659246618与P951.907560308热图像素，几何门禁通过。控制程序验证后提交template2000_b0_frozen_v001，已核实真实子进程运行。GPU2–7六rank/CPU72。证据outputs/action4_training_performance_v001/D0_V006_B0_HANDOFF.json；固定32拟合通过不代表完整eval或真实视频达标。


### r021 冻结检测头学习率单变量对照

冻结B0五轮1000更新完成，非有限0、六rank一致、冻结状态一致；五轮完整eval F1均0，末轮仅2预测且未匹配。epoch002中心诊断train64/eval64目标中心中位0.07785/0.07615，二者原阈值检测均0，支持训练响应不足，不证明单一原因。下一对照从相同正式A0权重fresh optimizer开始，仅B0 lr_new从1e-4到1e-3，仍五轮/原seed/FP32/global8/冻结状态，绝不改阈值或使用test。configs/model/action4_frozen_refinement_v002.json；D0 v006当前绑定实际重新核验通过，无需重跑。等待当前视频合并编码完成再提交template2000_b0_frozen_lr_v002，GPU2–7/CPU72。是否有效必须依据完整eval及真实视频，非仅loss下降。


### r022 编码与后续训练资源重叠

将待运行LR对照的供数/验证worker从10降为8，每rank计算2，六卡训练60CPU；已完成GPU分片及合并后，与12CPU编码并行，总计72。数据、seed、batch、优化、权重和评估不变。旧等待控制器PID/完整命令已核实，仅停止尚未提交训练的等待控制器，未中断任何训练/视频；新frozen_lr_controller_v003.py等待encoding_and_verifying或编码完成再提交相同runID。配置frozen_refinement_v003，权威r022；训练入口仍验证原D0绑定，不重复运行D0。


### 冻结B0真实视频失败与分片吞吐

全部2850帧预测实例均0，六个固定帧0/400/1000/1600/2200/2700实际目检均有两台可见相机但无框，记录outputs/action4_frozen_video_v001/ROOT_REVIEW_SIX_FRAMES.json。这是六帧人工检查加全量预测计数，不冒称全视频人工审核。剩余2818帧45分片全部成功，入队至末任务结束416.204秒，含排队/启动实测6.771帧/秒，最大worker显存838MiB；不含首32profile、合并和编码，不是对其他模型的因果速度对照。编码仍独立验证，模型效果不合格。


### 冻结检测头LR单因素五轮结果

run template2000_b0_frozen_lr_v002完成1000更新，非有限0、六rank一致、冻结状态一致；首轮TP0，第二轮TP1/F1约0.00483，第三四轮TP0，末轮TP1/FP29/FN377/F1约0.00490。相比lr1e-4完全无TP有少量变化，但不足以满足任务，不能据此认定单纯提高LR解决检测。完整逐轮对照FROZEN_LR_TRAINING_COMPARISON.json。独立固定32容量诊断frozen vs separated已排队/启动，保持当前正式结果，不产生正式checkpoint；两组原A0初始化、128更新、globalbatch8/micro1、检测loss、相同优化，不等同五轮正式协议。


### 独立ROI候选集成验证

候选网络默认state/logits精确保持原行为，显式独立分支可strict roundtrip且重复enable不重置；训练/resume/eval隔离overlay回归20项19通过1环境跳过。首版测试入口缺少main保护导致spawn重复执行，失败日志保留，已修复后新目录通过。新增独立策略双rank Gloo用3张含零ROI图、global4含尾部padding，1次更新后两rank模型一致、冻结ROI与A0逐值相同、checkpoint保存两rank RNG。证据outputs/action4_separated_features_candidate_v001/integration/MANIFEST.json。均为CPU机制/控制流，不是正式准确率；正式源码未合入，仍需真实尺寸GPU一致性及新源码D0后才可正式训练。


### r023 独立A0 ROI分支正式有界训练

当前全部旧任务已终态后合入network/stage_policy/trainer/eval四处，原源码已备份，安装回归58项57通过1环境跳过。新策略在B0复制A0 ROI分支并冻结，检测backbone/decoder/head可更新；恢复与eval先按显式checkpoint策略建分支再strict load，禁止静默漏载。固定32两组仅中心响应证据，不证明泛化。新配置action4_separated_refinement_v001，新D0 template2000_d0_v007当前源码实际验证后才允许template2000_b0_separated_v001五轮训练。保持原2000/A0权重/seed/globalbatch8/FP32/B0LR1e-3/5轮，GPU2–7 CPU72，未使用组件权重或新test。隐藏角点原A0误差仍在，不能仅凭检测改善判最终成功。


### 新源码D0 v007通过

D0 v007在125更新后正常完成，固定32ROI的mean0.3659246618/P951.907560308热图像素，几何与精度门禁通过。已自动提交template2000_b0_separated_v001五轮正式有界验证；此门禁不代表整图检测或真实视频成功。证据D0_V007_SEPARATED_HANDOFF.json。


### 独立分支五轮完成与ROI支持范围缺口

template2000_b0_separated_v001完成1000更新，末轮TP18/FP353/FN360/F1约0.048064，270额外实例定位但角点错误、90未定位，检测改善仍非达标。epoch2的240定位匹配中，168实例有真实点超出ROI热图支持范围，105实例即使支持范围内最优预测也无法达到all8@5%；精确生产affine分析见SEPARATED_ROI_SUPPORT_AUDIT.json。扩大ROI只是覆盖改善，不证明旧权重适配；已提交原ROI回放与2.25扩张下256/448分辨率的隔离配对实验，严格使用保存预测框、不用GT框推理，基线回放先验差异检查。


### r024 ROI适应正式阶段

ROI迁移候选CPU53项52通过1跳过，实际GPU2更新/64ROI全部512点得到监督（可见291、隐藏221），无非有限值，明确旧A0权重来源。任务全部结束后合入trainer/gate及新增initialization helper，原源码备份；安装60项59通过1跳过。新配置action4_roi_adaptation_v001：448/2.25、A0中心±15%尺度0.65–1.35扰动，从原正式A0refinev002 best权重热启动，新optimizer，10epochs，lrnew1e-4/backbone1e-5，global32/micro6/六卡72CPU。新D0 template2000_d0_v008保持固定32无扰动，通过后才允许template2000_a0_roi_adapt_v001；不从组件权重初始化、不自动B0/test，既有检测器保留。后续须比较GTROI与固定预测框下完整八角点，扩大覆盖不自动等于准确率提升。


### 新ROI D0 v008通过并衔接A0热启动

新ROI448/2.25的D0完成176更新，几何及精度门禁通过，P951.867358282热图像素；固定32收敛不是最终准确率。template2000_a0_roi_adapt_v001已自动启动并核实子进程存活，预定原A0权重热启动10轮，检测器和原数据保留。证据D0_V008_ROI_ADAPT_HANDOFF.json。


### ROI适应后刚性投影配对结果与真实图审核

更新时间：2026-10-22T08:07:16.340740+08:00。主控独立推进，不再派发或续派子智能体。adapted_rigid_saved_eval_v001已成功终态；复用200张eval的371个预测，不重新训练或渲染，25.52秒完成拟合，185个拟合通过既有门限。all8@5%：TP33→138、FP338→233、FN345→240、F1 0.088117→0.368491。按GT身份复核105个恢复、0个由通过退化为失败；严格2%下TP1→19，18恢复、0退化。此处零退化仅指阈值通过状态，不代表每个角点误差都减小。使用已知合成K、固定canonical8和预测点/置信度，GT框/姿态/可见性未进入拟合；未使用test，未安装生产推理。证据：outputs/action4_adapted_rigid_eval_v001/result/{REPORT.json,PAIRED_RECOVERY_AUDIT.json}。

剩余主要缺口：遮挡<10%的95实例有78达标，10–40%的119实例有53达标，≥40%的164实例仅7达标。必须保留高遮挡失败，不能以删除难例或改变指标声称完成。真实视频内参尚未校准，此合成收益不得外推为实机成功。

重复抑制IoU0.85候选对完整200eval计分与0.95基线相同。主控实际查看frame0：两台物体均出现独立预测，重复占用左相机的问题在此帧改善，但两框仍有错误连线/角点错位，拒绝作为准确锚定结果。证据outputs/action4_duplicate_decode_candidate_v001/result/ROOT_REAL_FRAME0_REVIEW.json。当前任务未完成；下一步需要严格组合原检测器与适应后的ROI分支，验证输出等价和父权重身份，再用真实帧检验几何修正。


### 统一推理组合候选接口验证

2026-10-22T08:11:15.240659+08:00：新增outputs/action4_composite_inference_candidate_v001/composite.py，显式接收正式B0检测器和A0 ROI父状态，严格检查阶段、数据身份、网络、独立分支策略及仅ROI预处理变化；只替换ROI分支，不伪装为可恢复训练checkpoint。真实父权重检查：347个非ROI状态张量完全不变，332个ROI状态张量与新A0逐值相等；组件权重、缺失数据身份、整图预处理改变和错误分支策略均被拒绝。实际网络CPU前向使用1024×768整图与448×448ROI确定性随机输入，center_logits/offset/ltrb及ROI logits与各自原模型差异均为0。首版验证器对检测路径正常空corner_logits取max失败，已修正为空张量形状/相等检查并保留记录；未修改模型以规避检查。证据STATE_VERIFICATION.json、FORWARD_PARITY.json。此为候选接口机制验证，尚非真实图准确率或生产接口合入。下一步用实际RGB和原保存预测配对，保持父权重来源与原检测输出不变。


### A0累计50轮与每5轮评估

2026-10-22T08:18:30.110329+08:00：已核对原10轮验证mean6.6067→4.8042、P9525.2834→17.0834；支持进一步训练，不能证明欠拟合是唯一原因。原20轮续训队列尚未启动，因用户最新50轮/每5轮eval要求标记DELIBERATE_SUPERSEDED_BEFORE_LAUNCH，未重复训练。实现A0可配置评估周期，实际CPU小模型6轮验证仅第5和最终第6轮生成评估/checkpoint，正确处理非5倍数终轮。新增r026与action4_roi_adaptation_extend_v002，D0v009及成功后A0 additional40控制器；旧数据、权重、失败记录保留。证据outputs/action4_eval_interval_candidate_v001及outputs/action4_training_performance_v001/ROI_ADAPT_EXTENSION_CONTROLLER.json，实际启动及结果以调度器为准。


### 每25 optimizer step异步评估与30分钟检查

2026-10-22T08:28:45.482602+08:00：已停止旧每5epoch方案的自动A0交接控制器，保留运行中的D0，不重复启动A0。已创建30分钟线程自动检查action4-25。异步评估队列/常驻worker候选位于outputs/action4_step_eval_candidate_v001/step_eval.py；队列机制验证覆盖独占领取、乱序结果选优、不重复处理完成回执、重复请求拒绝和非有限分数拒绝。QUEUE_TEST.json仅为人工夹具工程验证，不是训练结果；尚未安装生产、尚未启动25step连续训练。仍需接入训练快照发布、epoch恢复状态、实际GPU并发与末尾队列排空验证。


### GPU1八进程异步评估正式接入

2026-10-22T08:41:49.872543+08:00：已安装src/eval/action4_step_eval.py及训练器异步接口。真实CPU控制流26次optimizer更新，仅在25和尾段26发布快照；epoch恢复checkpoint与保留索引通过实际validate_resume_checkpoint/verify_retained。回归43项42通过1跳过。GPU1四进程实测回放原正式855/950step两个checkpoint，P95与历史结果差异均0，约63.92秒；八进程并发验证另存eight_worker_check，状态以REPORT.json为准。新sourceD0v010已交给控制器，只有新D0及八进程GPU验证通过后自动启动A0 async_v001，不再等待人工准备。快照序列化补齐checkpoint schema，完整loader读取验证通过。原每5epoch方案不再启动；全部旧产物保留。


GPU1八进程实际并发检查已完成：同一正式950step权重独立回放8次完整200eval，8次P95均与原结果差异0；这是并发一致性检查，不是8个训练checkpoint。fixture最终跨运行目录best别名被保护规则拒绝，全部eval已成功；保留失败说明，验证同运行目录别名及加载后通过，没有重复eval。证据outputs/action4_step_eval_candidate_v001/eight_worker_check/REPORT.json。自动监控action4-25已更新GPU1专用eval/GPU2–7训练的最新授权。r027控制器已启动，D0v010运行存活；A0 async_v001会在门禁通过后自动提交，当前不得把待启动描述为已训练。


### 停止重复D0并直接启动A0

2026-10-22T08:49:55.493969+08:00：已受控停止本任务D0v010及其自动交接控制器，保留运行记录，不标记通过。复用D0v009原始gate不改写，通过显式schedule-only复用校验；故意改变ROI方法、模型源码、receipt身份的三项反例均拒绝。A0 async_v001已直接入队并由后台调度器启动，GPU2–7训练/GPU1八进程eval，40追加epochs、每25optimizerstep异步eval、30分钟监控不变。


### 累计80轮与隐藏角点机制审查

2026-10-22T09:34:05.113112+08:00：只读真实16个高遮挡训练目标的标签/损失审查确认81/81有效隐藏点都有梯度，翻转可见性元数据不改变监督。独立热图损失缺少八点刚性联合约束；构造双峰热图下全局top20均值解码落在零概率区域，证明这一机制有失败可能，但尚不能证明实际高遮挡样本都由此导致。构造logit=-10时sigmoid SmoothL1梯度很小，仅是假设机制，不能冒充实际模型饱和。已启动CPU1实际checkpoint热图检查，证据ACTUAL_LOGITS.json将区分实测与构造测试。累计80自动衔接控制器已启动等待当前任务，未干扰当前训练。


实际step2875高遮挡16ROI复核：可见49点mean EPE1.8257热图px，隐藏79点6.5329；可见/隐藏平均峰值0.7098/0.4548。两组均无近零峰、无解码位置概率低于峰值一半的样本，因此本子集不支持把“全零饱和”或“双峰取均值落空”直接定为主因；构造反例只代表机制风险。隐藏点定位明显较差且仍有监督梯度，后续需实测几何一致性及遮挡姿态歧义。完整证据ACTUAL_LOGITS.json与ACTUAL_LOGITS_INTERPRETATION.json，未修改当前训练。


### 隐藏角点身份与显式几何补全

2026-10-22T10:14:49.813418+08:00：固定to80 step1475的32ROI对照完成：24高遮挡原预测1/24达标，局部5x5峰解码0/24，预测置信度刚性投影3/24，oracle任意身份匹配1/24、刚性对称重排1/24、GT可见点限定拟合3/24。轻遮挡8例原5/8、刚性投影8/8。高遮挡24例中18例仅2或3可见点，6例至少4点；10个刚性拟合通过中7个仍未达标。因此ID混淆非本子集主因，几何约束有帮助但错误姿态仍可几何自洽，可见点不足及错误观测需要图像姿态先验。全200eval共3024点canonical-ID重投影最大差0.000338px，支持标签ID与存储变换一致，不证明图像建模完美。对照正在扩至全164高遮挡，复用已完成32例，只增加140例CPU1计算。证据outputs/action4_corner_identity_geometry_v001/{REPORT.json,MECHANISM_BREAKDOWN.json,FULL_REPORT.json}（FULL_REPORT尚在运行时不得引用为已完成）。80轮后真实图检查控制器已设置，等待最佳选择文件与训练成功终态，再用GPU2–7空闲卡生成6帧原始预测供主控查看；不是整段视频完成。


### 独立几何候选实际训练与否决结论

2026-10-22T10:37:04.174371+08:00：全部164高遮挡对照完成：raw1/164、局部峰0、刚性投影31、oracle身份1、oracle对称1、oracle可见点拟合15；114例少于4可见点，83例刚性拟合接受但52仍未达标。独立geometry_completion.py已实际训练，隐点loss对image/pose/visibility梯度L1分别12.738/3.949/0.0972，隐藏标签扰动梯度变化24.705；全部分支更新，3D点间距离最大误差1.12e-7m，观察点扰动输出改变0.0457归一化单位；坐标闭环最大误差7.45e-7、checkpoint实际加载复现32训练例全部达标。但32训练/24验证出现0/24验证过拟合。随后使用冻结最佳A0完整3024train/378eval特征训练候选50轮，v001最佳3/378；发现标准化ROI下绝对深度预测未显式利用K尺度，v002接入K、固定尺寸和观测extent及ROI单位中心修正，受控焦距2倍试验深度响应由1倍改为2倍，完整训练最佳11/378（高遮挡3/164），仍劣于原82/378。两版均拒绝生产，未替换正式分支。80轮最佳ckpt的真实0/400帧已查看，重复漏检/畸形框仍存在；保存ROOT_VISUAL_REVIEW.json，不宣称视频成功。


### 保留姿态初值的受约束残差训练

2026-10-22T10:55:16.123854+08:00：残差候选已完成真实2400step训练、96次25step评估：原82/378，初始刚性投影229/378，最佳step1775为240/378；高遮挡1→42→50/164。注意本次刚性初值允许采用求解得到的姿态，即使旧残差门限未接受，所以不能将229与此前门控投影当同一方法。实际checkpoint重载复现240，较初始17恢复6退化，全部正深度，点间距离误差<1.2e-7m。机制验证隐藏loss三分支非零梯度，零修正复现输入几何。完整200图RGB-only检测框配对评估已启动，尚未报告其分数。32CPU初值77.45秒，GPU2候选训练52.69秒，复用缓存不重算图像特征。GPU3–7开始完整2850基线视频，非准确率通过，后续需编码完成才可宣称生成成功。


### 残差候选逐目标退化与训练充分性复核

2026-10-22T10:58:37.820325+08:00：固定step1775相比刚性初值17恢复/6退化，其中高遮挡11恢复/3退化、中遮挡6恢复/3退化、低遮挡0变化。step1800–2400的25次验证八点达标范围234–238，均未超过240，保留1775而非继续盲目加步数。推理输入消融：neutral可见性239/378，图像特征错配214/378，原240；可见性权重影响很小，不能声称强可见点引导收益。证据INPUT_ABLATION.json及TRAINING_SELECTION_REVIEW.json。完整RGB-only200eval与2850帧视频运行中，无错误输出；截至本次检查已生成1093帧，未编码完成。


### 完整RGB对照发现预测框下残差退化

2026-10-22T11:01:42.446438+08:00：完整200eval RGB-only预测框对照完成：raw TP75 FP296 FN303 F1.200267，rigid TP219 FP152 FN159 F1.584780，residual TP208 FP163 FN170 F1.555407。残差对刚性4恢复15退化，高遮挡32→26/164、中遮挡95→91/119、低遮挡92→91/95，因此拒绝当前残差替换刚性路径。GTROI240/378的优势未转移到预测框场景，训练ROI分布差异是优先核查假设，不能仅据关联认定唯一原因。实际查看rigid真实frame400：交叉线消失且两个刚性框存在，但机身边缘与框的大小/方向仍有偏差；真实K未标定，未达标。证据full_rgb/REPORT.json、PAIRED_DECISION.json、ROOT_REAL_FRAME400_REVIEW.json。


### 预测ROI与GTROI混合适配训练

2026-10-22T11:04:28.976908+08:00：预测框适配提取器及后续初始化、训练控制器已实际启动；当前train已处理101/1600图，预测199框、匹配161，尚无失败。尚未开始新候选参数训练，待特征和初值完成自动衔接。原完整视频已2177/2850帧，仍未编码完成。旧残差在预测框退化的证据保留；新方案不先声称解决分布偏移或泛化。


### 视频帧完整性与新候选自动验证衔接

2026-10-22T11:09:03.589838+08:00：最佳80轮基线视频的2850帧推理/绘制全部完成，5个分片完成，帧ID连续且无重复；复用6预览帧的时间信息从原始manifest补齐到独立predictions_with_timeline.jsonl，原预测不覆盖。PREDICTION_TIMELINE_AUDIT.json记录完成，MP4编码及最终时间轴验证仍进行，不提前宣称视频完成。新预测ROI适配当前已处理901/1600训练图，未报错；新增validate_cache.py自动检验全部200eval图及所有预测保留，并要求坐标还原后复现raw TP75/FP296/FN303。preview_after_training.py在新训练完成后自动生成6真实帧，不重复200图推理，不自动推广候选。所有新验证尚未运行的部分标记等待，正式源码未改。


### 残差退化样本的预测框偏移描述

2026-10-22T11:11:16.080094+08:00：完整RGB对照中15退化目标的绝对框中心偏移中位数为GT宽高的8.59%/7.48%，204保持达标目标为3.22%/2.82%；尺度比中位数分别0.889/1.005与1.079/1.114。支持进一步验证裁剪分布差异，但不是因果证明。证据full_rgb/REGRESSION_CROP_DIAGNOSTIC.json。新预测ROI提取已1301/1600图，后续仍等待提取/初值完成；视频MP4持续增长、无失败日志，最终时间轴验证未完成，禁止提前报告成品通过。


### 预测ROI适配完成及完整基线视频验证完成

2026-10-22T11:17:12.292843+08:00：新候选2400step/每25step完整eval已完成，最佳step250，TP233 FP138 FN145 F1.622163；刚性基线219/.584780，19恢复5退化，高遮挡32→41/164。独立重载best复现全部计数；eval缓存保留371预测及200图，原始坐标回放最大误差6.53e-5px，计分复现75/296/303。实际查看新候选frame400仍有机身框范围/姿态偏差，未通过真实锚定；不能把合成评估提升当最终成功。完整80轮基线视频outputs/action4_best80_full_video_v001/encoded/full_red_raw.mp4编码和时间轴验证通过，2850帧、425220484字节，SHA256 55eae906dc3e1b869501788bd238d56b7c177f049d34976056960ca8719a183b；这是已知精度失败的原始基线视频，不是新候选完整视频。


### 双机重复检测与几何初值误差范围

2026-10-22T11:20:34.602127+08:00：原完整2850视频统计：2621帧2预测、222帧1预测、7帧0预测；8帧符合IoU>=.85且中心距<=.05小框对角线的重复风险条件，仅是启发式非真实漏检计数。GTROI初始化姿态误差评分：高遮挡164例中位6.846度、P9536.479度、11例>30度、2例>90度；中遮挡中位2.692度、轻遮挡1.983度。GT姿态只用于诊断，不进入拟合。证据DUPLICATE_RISK_AUDIT.json和POSE_INITIALIZATION_ERROR.json。固定权重重复抑制对照进程已存活运行，尚未完成，不提前计为通过。


### B0追加80轮与每20step完整评估

2026-10-22T11:28:55.374627+08:00：B0异步20step/fullRGB计分与显式B0旧权重+A0最佳分支初始化已实现。实际CPU21更新验证20/21快照、epoch20与last恢复，46回归45通过1跳过；GPU1真实原B0快照完整200eval复现TP18 FP353 FN360，8worker队列验证通过。真实双来源state检验检测参数逐值不变/ROI逐值等于指定A0。B0 80轮已提交，具体运行状态以调度器为准。A0候选新增预测可见性权重拟合完整200eval241TP F1.643525，高遮挡46/164，相对233候选9恢复1退化；52缺少加权姿态目标回退原初值不筛除。该新逻辑尚需真实画面接受，不得称已安装生产。


### B0评估权重保留与容量约束

2026-10-22T11:31:51.979728+08:00：v001 B0实际未启动，因800全量权重加80epoch状态的300GiB估计无法满足调度磁盘安全余量，已标记启动前替代，未重复训练。增加opt-in保留策略并通过具体5任务/乱序评分测试：正确保留top3、待处理权重、全部5份评估记录。v002仍80epoch/20step，控制器不等待新D0。之前46回归45通过1跳过、CPU21step B0与GPU1实际B0完整评估已通过。最终启动状态需以v002实际PID核实，排队不算开始训练。


### B0直接A0初始化及完整机身检测验收规范

2026-10-22T11:39:45.548806+08:00：已受控停止不符合当前要求的warmstart旧B0+角点F1训练，保留PREVIOUS_RUN_STOP.json及全部已完成状态，不回滚。新增框指标模块并通过完美框、重复一对一、双机漏一、空图误检测试；将所有预测八角点替换为错误值后B0框成绩完全不变，证明已拆分。真实GPU异步bboxeval复现原B0保存的全部200图计数TP288 FP83 FN90；首次校验误用了后续组合推理289作为基线而失败，已以同一原B0记录逐图核对通过，保留说明，不改阈值。低阈值AP50=.675294、AP50:95=.224139仅开发诊断，maxDets2明确。新配置action4_b0_best_a0_bbox80_v001从bestA0 step1475直接初始化，80轮/每20step框eval控制器已接通，独立最终验收集和人工真实GT仍未完成。


### B0连续训练期间A0独立试验与框监督闭环

2026-10-22T11:50:51.011900+08:00：B0完整机身目标/解码闭环：16双机图32目标理想预测还原最大误差5.29e-5 fullinput像素；visible bbox只作锚点，LTRB仍回归完整canonical包络，测试不代表学习准确率。A0定位可靠性候选用2521匹配训练ROI、800更新/每25eval，CPU2无GPU，参数L1更新3480.33；最佳step650验证Brier.145591，原峰值.145112，未改善，不采用。B0持续运行，早期step520开发框F1.6158、AP50.5154/AP50:95.1597、双机65/184、重复预测1、空图0/6；仅早期开发结果，不验收。


### 结合公开工作的数据流机制改进

2026-10-22T11:59:18.456903+08:00：独立dense_vote.py已实现方向监督和加权直线交点：构造6可见支持点推断8个支持域外角点误差3.99e-7，8通道梯度均非零，平行投票退化标记正确；尚未训练图像预测头、尚无鲁棒RANSAC和真实泛化证据。独立GIoU候选验证同一相对20%收缩、物体尺寸.02到.2时GIoU均约.36，坐标SmoothL1由2e-6变2e-4；非重叠仍有非零梯度。说明机制匹配性，不证明当前B0误差唯一原因，也未中途改变B0训练。


### 可见区域方向投票候选实际训练结果

2026-10-22T12:05:17.364603+08:00：真实3024train/378eval机身mask采样均非空，CPU2完成800step、每25step评估，共64.72秒。最佳step800，预测前景IoU.58661、角点归一化平均误差.09555、八点全通过0/378，高遮挡0；无病态交点被检测到但位置仍不准。候选确实训练了前景与方向头，但不满足精度，拒绝接入。该实验采用4x4池化特征插值16x16及非RANSAC交点，不是PVNet复现，不能据此否定论文方法或直接归因单一因素。后续方向需要真实高分辨空间特征与鲁棒一致性，避免重复同一低分辨试验。当前B0正式训练未改动。


### 独立验收集封存与训练后生成衔接

2026-10-22T12:09:52.551188+08:00：已完成400样本独立验收计划、ID/group不重合检验和原worker接口校验，51个配置准备完成；原训练/开发清单不变。控制器已启动等待B0完成，当前未渲染任何验收图片、未产生验收指标。生成计划和封存状态位于outputs/action4_independent_acceptance_v001/GENERATION_PLAN.json及STATUS.json。该步骤补齐用户要求的独立验收准备，不改变当前B0源码/阈值，也不冒称最终数据已经完成。


### 原视频完整机身框人工验收准备

2026-10-22T12:12:38.751737+08:00：29帧原图ID/PTS/hash引用及未标注草稿已生成，未复制/重抽2850帧。独立标注准入检查器通过拒绝真实未审核草稿、未确认空目标、模型预测作GT、缺失审核人等反例；构造不确定范围帧不进入严格IoU、显式保留列表。测试仅metadata机制，人工标注0帧，不能报告真实检测P/R/mAP。B0长训和源码未修改。


### 真实帧人工审核工具交付

2026-10-22T12:18:47.988707+08:00：29帧内嵌离线HTML工具已生成并在本地实际浏览器加载，框选原图坐标、撤销、缺少审核字段拦截均验证；修正canvas边框导致的坐标偏移，测试后恢复未审核空白状态。模型预测框不预填；没有创建人工GT。工具支持逐帧场景/遮挡/不确定范围/无目标确认及JSON导出导入，导出导入文件往返尚未测试。服务器outputs/action4_real_acceptance_v001/review_frames.html，本地workplace/outputs/real_acceptance_review/review_frames.html，截图review_ui.png。此为审核工具完成，不代表真实验收完成，B0未改动。


2026-10-22T12:24:43.962429+08:00：现运行manifest再次确认直接A0best初始化、independent_A0_roi冻结分支。B0日志heatmap loss可以随裁剪变化，但冻结A0参数不被B0优化，不可把B0长训解释成A0继续训练。A0改进仍须独立训练候选。完整200eval失败分解CPU2任务运行中，已61图未报错，未拿部分样本作完整统计。


### 按用户要求以demo核对真实视频

2026-10-22T12:35:18.856059+08:00：完整200图固定step3660框失败分解已完成：高遮挡102/164 IoU>=.5，59有重叠但IoU.1–.5，3无重叠；中遮挡114通过/5框偏差，轻遮挡95全部通过。用户提供demo同帧6位置背景相关系数>.999、最佳时间offset均0；自动白色叠加差分在6帧各提取到2个参考线框组件，仍标candidate非GT。第400帧旧B0框与参考包络IoU .8246/.7734，经过A0几何候选的角点外包络为.3221/.3493；反映部分真实失败在A0/geometry阶段放大，而非检测器完全没框住。完整同步左右对照视频编码已启动（左demo右原80A0+旧B0失败基线），不代表当前B0训练结果。证据outputs/action4_demo_alignment_v001/{ALIGNMENT_REPORT.json,REFERENCE_OVERLAY_EXTRACTION.json,SIX_FRAME_REFERENCE_AGREEMENT.json}。


### 检测框软约束几何候选的实际否决

2026-10-22T12:41:30.648701+08:00：独立box_prior固定物理尺寸、正深度、有界旋转/中心/深度修正，机制测试零误差输入保持和错误深度修正通过。但完整200eval统一约束TP233→51/F1.13618；仅当角点包络与B0框IoU<.5时修正63预测，也仅215TP、0恢复18退化。两种策略均拒绝接入，说明检测框自身误差可损害正确角点，不能凭demo少量帧表象全局约束。启动固定当前B0 step3660的6真实帧CPU2对照，将二维检测框与原A0角点分开画，避免一直用旧5轮B0视频代表正在训练的新模型。同步demo对比视频已2850帧编码核验并复制本地outputs/pilot2000_review/demo_vs_baseline.mp4，右侧明确为旧基线，不是新B0结果。


2026-10-22T12:42:46.295802+08:00：较新B0 step3660的6真实帧二维框与demo包络对照完成，4/6双参考包络IoU>=.5；frame0两机恢复，但frame1000出现手间衣物误检并漏左腕相机（实际图已查看），不是重复框。frame2200一框参考IoU约.482。此为中途checkpoint参考对照，不能把frame0改善泛化为全视频稳定检测，也不能把全问题归为NMS或几何模块。证据outputs/action4_currentB0_demo_review_v001/REPORT.json与ROOT_FRAME1000_REVIEW.json。


### 衣物误检的负样本覆盖与监督审查

2026-10-22T12:46:57.336259+08:00：负样本清单确认train48空目标：wall18/table16/cloth14，eval仅6空目标，test全双机无空图。实际查看每背景1张，共3张：腕部/手保留，衣物例主要是远处平铺衬衫，没有这些示例中的近手捏持衣物外观。使用真实空目标标注构造高logit误检，focal梯度+1.12675会抑制误检，box/offset损失0，证明不是空图loss被忽略。覆盖疑点与数据流错误区分，不能据3图断言全部48无相关外观或认定唯一原因。证据outputs/action4_hard_negative_audit_v001/{COVERAGE.json,ROOT_VISUAL_REVIEW.json,NEGATIVE_LOSS_TEST.json}；当前B0未更改。


2026-10-22T12:50:46.485662+08:00：B0 step7436/epoch38仍存活，371已发布/330已评估/0失败；最佳step4660开发框F1.84615、AP50.76431/AP50:95.27829、双机130/184、高遮挡107/164。top3权重及全部指标正常保留。评估落后41不等于失败，队列上限64机制保持；GPU4/5虽各约43GiB空余，但利用率瞬时11%/100%，显存空闲不能证明DDP同步训练上加eval会加速。未新增竞争GPU任务，保留已有训练结束后的扩展清队列机制。更新耗时基线最近100step中位.4985s、P95.5344s仅host计时，不能当全链路吞吐。


### 独立验收质量检查与发布门槛衔接

2026-10-22T12:55:18.313873+08:00：已接通acceptance质量控制器，当前等待训练后生成的批次，无新增GPU任务，不修改B0源码。复用原质量函数，只有报告路径重定位，51批工程报告全部保留，任何失败禁止自动接受。尚未生成图片、尚无工程质量通过或最终验收成绩，不能将控制器启动算作质量验证完成。


### A0较高分辨空间特征与稳健投票候选

2026-10-22T13:00:31.002890+08:00：已启动28x28空间特征提取与后续训练控制器，不使用任何GPU、B0源码不变，额外CPU2。仅IRLS构造25%错方向反例平均误差由.5672到.4508，加入成对假设一致性初始化后为.08137；平行退化仍被拒绝。保留IRLS_ONLY_TEST及ROBUST_VOTE_TEST，都是构造机制测试，不是实际精度收益。尚无v002实际训练完成结果；不能用新机制通过冒称隐藏角点解决。


### B0完整评估区间与训练损失趋势核查

2026-10-22T13:09:41.806868+08:00：已完成的1–2000、2001–4000、4001–6000、6001–8000区间均100次eval，最佳mAP50:95依次.23766/.26819/.27829/.26434；同期train center loss均值.27477/.01747/.00580/.00149下降。8001–10000只有36次eval完成，禁止据未评完区间判最终收敛。该趋势提示潜在过拟合/分布问题，不等于证明原因，也不擅自早停80轮；继续保留全程best而非last。证据outputs/action4_training_performance_v001/B0_DEVELOPMENT_CURVE_AUDIT.json。A0v002特征提取CPU2仍存活，已526图/1001目标，80epoch控制器等待数据未报错，尚无参数训练成绩。


2026-10-22T14:41:40.129212+08:00：A0空间投票v002完成80完整epoch/15120更新，最佳step14525，39/378八点通过、高遮挡1/164，低于原A0的82/378，拒绝替换正式权重。独立验收400候选已全部生成，51/51批工程质量通过无失败；尚未视觉审核/发布/模型评分，不可报告最终验收成绩。B0冻结best及阈值保持，后续应完成视觉审核后执行一次独立验收。


2026-10-22T15:12:42.516888+08:00：独立400候选在查看模型分数前抽查12原始图（每模板空/单/低遮挡双/高遮挡双），实际看到多处画内断开的开放手臂截面、独立悬浮矩形衣物遮挡片，以及较远较小的桌面布局。全部工程质量通过记录保留，但不因此自动判视觉真实度通过；ROOT_VISUAL_REVIEW标记最终真实验收用途不通过，禁止发布/正式评分，400原文件不删不改。此结论不等于每张图都逐图审过，样例已足以阻止把同模板新采样包装成真实任务通过。后续应修复场景边界与遮挡物关系，并保持原冻结相机资产和已冻结B0权重。


2026-10-22T15:42:52.006284+08:00：冻结最佳B0 step4660文件SHA与FROZEN_SELECTION一致，实际6真实帧预测完成并分开绘制二维框/八点。5/6帧两参考线框包络IoU>=.5（仅demo一致性非GT）；frame1000实际查看仍把双手间衣物当作相机、漏左腕相机，右侧检出，未满足稳定双机真实验收。此结果来自最终冻结B0，不再引用旧5轮基线代替新模型。未修改阈值或模型，失败证据outputs/action4_frozenB0_demo_review_v001/ROOT_REVIEW.json。


## 2026-10-22 增量结果：独立权重与候选池

- 实现src/model/action4_components.py：A0/B0仅保留各自模块；严格角色加载；独立优化接口；组合推理桥接。导出models/action4/independent_components_v001/A0.pt与B0.pt，未修改旧权重。
- 已验证B0内冻结ROI权重逐张量等于最佳A0；拆分后真实帧0/400/1000/1600/2200/2700的完整预测与旧组合逐项相同。两个模块各自一次工程梯度更新不产生另一模块梯度/权重变化；错误角色加载拒绝。该工程探针不代表任务训练或精度改善。证据outputs/action4_component_separation_v001/REPORT.json。
- 初次组件加载因旧load_checkpoint严格只接受legacy schema而失败，保留verify.log；修复为组件专用weights_only=True加载与路径/角色/schema检查后verify_retry.log通过。没有放宽原loader。
- 旧训练控制器尚未迁移到独立组件，组件权重无optimizer/scheduler/RNG，不可直接续训。此项未完成不能表述为完整新训练流程已交付。
- 冻结B0step4660在200开发图及6真实图提取top20池，原解码在206图完全重放。legacy95/legacy85/hard50/linearSoftNMS50/GaussianSoftNMSsigma0.5均TP319、FP57、FN59、F1=.846154、双机130/184。当前重算AP50:95=.2782647，历史冻结记录约.2782939有微小数值差，不覆盖历史值；本次方法比较使用同一缓存，五法相同。证据outputs/action4_b0_postprocess_v001/REPORT.json。
- 开发集候选池在score>=.3时oracle匹配上限319，与基线相同；>=.001时333。这是仅诊断上限，不是可部署成绩。
- 第1000真实帧左相机位于第三候选，score=.719204、与demo参考包围框IoU=.849475；第二候选衣服误检并不与相机重复。因此普通去重不能修复该失败，需要目标辨别/质量排序及外观分布改进。demo非认证GT。证据POOL_COVERAGE.json。
- 可见框中心相对完整框中心位移/完整框对角线：train高遮挡1130目标median=.146448/P95=.240478；eval164目标median=.150915/P95=.244450。表明锚点有显著偏移，但尚不能证明替换中心必然改善。

独立验收400仍视觉失败，不发布、不评分；A0/B0拆分不改变现有精度，不宣称任务达标。GIoU、质量头和新独立训练控制器仍需实施/实际训练对照。


## 2026-10-22 独立实际训练已经启动（进行中）

- 六项80轮配对微调，配置与权威r033已固定；GPU2–7各一项，GPU1八eval，CPU68核另有4核余量。B0每20step、A0每25step；并非仅工程测试，也未完成80轮。
- RECOVERY_TEST.json：B0 quality/A0 joint用真实数据缩小尺寸，保存后下一更新与连续执行的参数、optimizer、scheduler和损失完全一致，错误身份拒绝。初版验证脚本缩进错误在verify.log保留，修复后verify_retry.log通过。
- LOSS_FLOW_TEST.json：128组箱体与Torchvision官方GIoU loss最大差0、梯度差9.31e-10；隐藏标签改变确实改变修正头梯度，隐藏梯度范数1.1683。仅机制证明，不是识别精度。
- PIPELINE_TEST.json：A0修正坐标确实回到原图输出，八点ID不丢弃；没有在视频链路静默退回旧热图。
- 2026-10-22 17:20+08时同step140，B0 baseline/GIoU/GIoU+quality mAP50:95为0.274308/0.290899/0.294964。同期A0 step200 baseline/noedge/joint全八点通过73/31/35（总378），隐藏误差0.08835/0.12546/0.12652，候选早期退步，未采用。各自最佳仍保留起点82/378，不能声称隐藏点问题解决。原始结果见PROGRESS.json和各eval/*.done.json。
- 发现重复PNG解码与共享队列偏斜，训练发生背压等待。仅受控替换8个自有eval进程，训练PID、模型与损失源码不变。新执行器每实验至少1worker、两弹性worker、RGB内存缓存；评分函数完全复用。CACHE_PARITY.json逐项相等；EVAL_REPLACEMENT.json保留8个未完成请求中断/重新领取证据。已完成评估不丢弃。
- source规范已整理，正式入口见docs/action4_model_interfaces.md。outputs中的初次launch/verify仅保留历史快照，不作为正式可维护代码。未提交/推送Git。
- 正在执行正式尺寸恢复探针与全378目标A0同权重raw/refined诊断；完成前不写通过。自动检查action4每30分钟继续。

正式尺寸恢复验证补充：LIVE_RECOVERY_TEST.json已通过。取运行中B0 quality step680与A0 joint step800的恢复状态（不停止训练），按保存cursor计算真实下一批，重复恢复执行后模型、optimizer、scheduler、RNG和损失完全一致；尺寸全图1024×768、ROI448。此验证补充缩小尺寸连续/恢复对照，不是新训练效果。

2026-10-22 17:28+08补充实际结果：B0配对step500 mAP50:95 baseline=.267393、GIoU=.312836、GIoU+quality=.330211；详见B0_PAIRED_STEP500.json，仍是开发集早期结果。
A0 joint step725同权重全378GTROI诊断已完成：原始热图全八点56/378、修正后55/378；原始隐藏误差.102244、修正后.101761，两者高遮挡均0。对比原冻结A0的82/378/.087917，退化主要已出现在原始热图特征分支，不能只归咎修正点位移。新增损失影响共享特征是待验证解释；后续优先冻结已好的A0主干/热图、独立训练修正模块做消融，不能据本诊断直接宣称因果已证实。本次80轮对照继续保留最佳起点，候选未推广。该诊断step725推理快照已用硬链接保存在a0_raw_refined_audit/frozen_step_00000725.pt，避免last.pt继续更新导致证据无法复现。


2026-10-22本次继续：确认六个训练PID、八个当前缓存eval PID均存活，未重复启动。新增正式src/eval/action4_component_real_review.py，固定当前B0 quality候选与原A0 baseline最佳权重硬链接，在原视频六帧做RGB-only组合推理。结果4/6达到双目标demo参考IoU>=.5，旧冻结B0为5/6；第1000帧实际图片确认衣服褶皱误检、左腕相机漏检仍存在，2200帧一个框参考IoU=.4836。该候选不推广为最终模型，证据outputs/action4_independent_trials_v001/real_review_v001/ROOT_REVIEW.json。模型身份{"A0": {"step": 0, "variant": "baseline", "role": "A0", "config_id": "f7d64d941f9d1923b3fc9002c56e8493fd4bf3b96542a73c1700bd3aa38c2f10", "snapshot": "outputs/action4_independent_trials_v001/real_review_v001/A0.pt"}, "B0": {"step": 740, "variant": "giou_quality", "role": "B0", "config_id": "7c1900f56344b738d5ba8f8a2f9f07a142714ecbe2de80e5beb4f05cbab726d6", "snapshot": "outputs/action4_independent_trials_v001/real_review_v001/B0.pt"}}。开发成绩上涨并未证明真实迁移改善。


模块化增强增量：src/model/action4_a0_enhancement.py及src/data/action4_direction_targets.py已实现；真实448ROI一次aux参数更新后，base全部state逐项未变、无base梯度、关闭增强前后输出完全一致，隐藏标签扰动改变辅助梯度。证据outputs/action4_a0_additive_v001/INTERFACE_TEST.json，仅工程机制，不是全数据训练结果。
PVNet投票移植的无噪声构造测试误差1.206e-5px；25%异常方向最大误差.115464px，未通过既定.01px断言，保留voting_test.log和VOTING_REVIEW.json；未将该投票接入正式预测替换，尚未完成GPU/full-data精度验证。
best初始化接口已用固定B0 quality step740验证：所有参数完全继承、旧optimizer/scheduler/RNG未转移、跨角色拒绝；无新训练启动。证据outputs/action4_independent_trials_v001/best_initialization_test/REPORT.json。正在运行的六项对照、数据和源码未改动。


双线推进补充：A0新增src/model/ops/action4_vote_refinement.py，在官方投票初始化后对射线法向残差作Tukey IRLS；原官方兼容循环及原失败日志保留，不下调原.01px异常方向测试门槛。五种种子(19/31/47/83/101)的25%异常方向构造测试、无噪声测试、平行射线和空mask拒绝测试通过，浮点输出误差均低于原门槛。此为理想合成方向算子验证，绝非模型学习到0误差。src/model/action4_a0_enhancement.py新增auxiliary_vote_evidence：仅预测mask参与推理；数值无效点回落base点；production_replacement_enabled=False。未设未经验证的自动融合权重。证据outputs/action4_a0_additive_v001/REFINEMENT_TEST.json及refinement_interface_test.log。

B0新增正式src/eval/action4_candidate_pool.py和src/eval/action4_quality_order_audit.py，固定当前best硬链接到outputs/action4_independent_trials_v001/quality_order_audit_v001/B0.pt，CPU2执行完整200图同权重解码顺序对照。比较当前product-before-peak与centre-peak-then-quality；不修改当前训练/阈值，不凭少数帧选规则。报告尚在生成，完成前不宣称某方法改善。


B0 quality-order全200图固定权重对照完成：质量先乘再取峰mAP50:95=.396156；先中心候选再质量排序=.398006；19图框发生变化，但两者TP328/FP40/FN50、F1=.879357、双机137/184完全相同。没有证据表明该顺序能解决主要漏检/误检；不热改原decoder。结果outputs/action4_independent_trials_v001/quality_order_audit_v001/REPORT.json。

A0辅助首次train_run_v001在第1次更新前因CUDA二维交叉熵不支持严格deterministic而失败，保留FAILURE及日志。改为完全等价的两通道logit差BCE，CPU double数值/梯度对照通过；又发现CUDA median返回索引不支持严格deterministic，改成稳定排序取同一lower median值，未关确定性验证。448ROI四样本GPU严格模式更新与构造投票验证通过，base全部state不变。预检最初未打开deterministic，故未提前发现；失败和修复证据都保留。

v002从相同最佳base重新初始化新head（v001零更新、无可恢复optimizer），已启动GPU5训练与GPU1评估，DISPATCH.json为当前PID依据。v001失败不计入完成训练。重试入口第一次因报告文件命名不符未启动进程，已核对严格测试日志并保存CUDA_DETERMINISTIC_TEST.json后受控启动；初次测试PASS日志仍保留。当前80轮尚未完成，不宣称隐藏点精度改善。


本轮完成全1600训练配方审计：0目标/0遮挡片48，1目标/1片80，2目标/2片1472；3024片均无attachment字段。源码明确按目标canonical8投影构造camera-facing四点矩形，TemplateRuntime创建无父对象quad；不是有手部连接的布料模型。桌面对象原点camera depth由.621257+1.05变为1.671257m；此为原点深度，未声称实际桌面覆盖面积。证据outputs/action4_scene_domain_audit_v001/OCCLUDER_COOCCURRENCE.json与ROOT_FINDINGS.json。发现的是数据生成结构与域差异风险，尚未用反事实渲染证实模型因果依赖，未修改训练数据。
辅助A0实际训练/eval PID2226978/2226977已核实存活；step25完整评估仍0/378辅助全八点通过（base82），不部署。


十二图四格诊断已实现正式renderer/评估/资源等待控制器，Python语法检查通过。控制器PID2234166实际启动，当前只等待原B0 baseline80完成和PID退出，未加载Blender、未渲染、未新增数据。原场景的CAD/安装关系不改，干预仅为明确的可见性变量；这是诊断而非修复后的场景验收。输出outputs/action4_scene_domain_audit_v001/counterfactual_v001。


## 2026-10-22T19:33:33.795799+08:00 数据流与有效信息复核（进行中）

保持独立A0/B0与原2000数据。三项B0及A0 baseline继续80轮；joint/noedges约24–25轮无提升后受控停止，保留负结果，不代表完成80轮。辅助方向场仅训练附加头，原A0冻结。执行权威r036。

完整200开发图、378目标、相同检测框/分数/候选数：八点0.05阈值raw74、rigid211、visibility_residual237；高遮挡0/164、28/164、45/164。这是已知合成内参条件下的组件组合结果，非真实视频达标。配对恢复与退步见outputs/action4_current_geometry_comparison_v001/DATAFLOW_REVIEW.json。

数据流：RGB→B0完整框→同一ROI仿射→独立A0固定ID八点→固定CAD/合成K刚性拟合→预测可见性加权及残差→原图逆变换。GT框、可见性与姿态不进入推理。预测可见性只改变拟合权重，不删除隐藏点。

12张三姿态四格干预：相机存在6图12目标均检出；机身移除但支架保留6图均误检（有patch共6FP、无patch共4FP）。此为配件混淆诊断，不能外推总体误检率。辅助前景头step1300对12机身ROI/10配件误检ROI呈分离信号，尚不据此设阈值或部署。优先完善机身/配件负例与真实遮挡连接关系，而非任意提高权重。

隐藏标签已进入所有8方向通道的损失；改变隐藏标签能改变梯度、辅助权重会更新，冻结主干不变（INTERFACE_TEST.json）。但当前辅助最佳仅3/378，不允许替换原角点；分割IoU高或数值投票有效不等于角点准确。后续针对方向误差、支撑区域和几何拟合逐项消融。


## 2026-10-22T19:38:27.756256+08:00 当前固定权重真实六帧验证

新增src/eval/action4_geometry_real_review.py，复用完整200对照固定B0/A0/residual权重。原始八点、刚性拟合、可见性残差共三列六帧实际推理并视觉检查。数据与结果位于outputs/action4_current_geometry_real_review_v001；不向推理输入GT框/点。沿用合成源K只是明确的假设，不是实测内参。

B0两台预测框与demo参考包围框IoU>=0.5的帧为5/6；frame1000一台匹配0.837、另一台0，存在漏检与手部附近误检。八点几何有局部收整但完整机身贴合、朝向和尺度仍不稳定，ROOT_VISUAL_REVIEW明确拒绝最终交付。不得以5/6参考一致当正式检测精度，不得以合成237/378推断真实达标。

首次启动因错误解析虚拟环境解释器软链接导致NumPy缺失，零推理退出，失败日志保留；使用未解析的项目.venv/bin/python后六帧全部完成。没有安装新依赖或修改训练源码。后续继续机身/配件/手部区分与可信几何约束，不直接融合当前辅助方向场。


## 2026-10-22T19:41:29.861629+08:00 成熟方法复核与方向场故障隔离

继续使用原2000数据和现有独立组件，不直接替换主线。PVNet原论文强调像素方向投票及关键点位置不确定性用于PnP（https://openaccess.thecvf.com/content_CVPR_2019/html/Peng_PVNet_Pixel-Wise_Voting_Network_for_6DoF_Pose_Estimation_CVPR_2019_paper.html），官方https://github.com/zju3dv/pvnet。GDR-Net以稠密对应、可见mask、表面区域注意力形成几何中间表示（https://openaccess.thecvf.com/content/CVPR2021/html/Wang_GDR-Net_Geometry-Guided_Direct_Regression_Network_for_Monocular_6D_Object_Pose_CVPR_2021_paper.html），官方https://github.com/THU-DA-6D-Pose-Group/GDR-Net。本项目现有稀疏方向分支不等于完整复现任一方法。

新增src/eval/action4_direction_failure_audit.py：固定辅助最佳权重，对全部378 ROI执行原预测/GT可见mask替换对照，前16 ROI另做GT方向+mask检查。标签仅故障诊断，结果不得当RGB模型成绩；输出outputs/action4_direction_failure_audit_v001。GPU6 CPU2，原训练不动。

代码审查发现归一化差异：官方smooth_l1_loss先逐样本按可见像素数归一化；现有evidence_loss对整个batch的support合并归一化，可能使可见区域大的目标贡献更大。尚未证明这是当前失败原因，待方向/掩码隔离结果后只在新候选中比较，不能热改现有训练。数值投票有效不能代替校准不确定性，不能直接提高融合权重。


方向故障隔离已完成：step625固定最佳头，原预测3/378，GTmask替换仍3/378，隐藏点平均归一化误差0.168567→0.162709；前16个ROI使用GTmask+GT方向为16/16，其中8个高遮挡。可见/隐藏方向平均角误差15.446/17.013度。证据指向方向预测瓶颈，不能把分割质量改善直接转化为角点融合权重。16样本oracle仅证明该范围坐标/投票链路可正确解码，不是模型成绩，也不能排除全部投票边界问题。下一候选优先比较逐ROI归一化方向监督；原训练不改，保持独立对照。详见outputs/action4_direction_failure_audit_v001/ROOT_REVIEW.json。


## 2026-10-22T19:45:25.132565+08:00 逐ROI方向损失同起点训练对照

新增src/model/action4_direction_losses.py和独立版本trainer src/model/training/action4_direction_trial.py，原运行源码不改。pooled_control/per_roi从同一选优辅助head初始化，主干冻结，优化器及schedule重新开始，lr0.0003、原3024训练ROI、80轮上限、每25step全378ROI异步测评。仅方向损失归一化不同；逐ROI平均含空ROI零贡献，避免微批次分割改变梯度。tests/check_action4_direction_normalization.py已验证含空ROI的微批一致、单ID标签梯度对应和空掩码零梯度；这些是工程正确性，不是精度。

GPU6和7各一训练+一测评，新增10CPU，最大保守总预算62/72；GPU1仍最多8测评。派发与配置见outputs/action4_direction_normalization_v001/DISPATCH.json，权威r037。注意复用评估器的EVAL_PROCESS.gpu为历史硬编码1，实际GPU映射以DISPATCH与启动环境为准，不因此误判占卡。计划比较相同步数隐藏误差/八点达标及恢复退步；无增益不得替换正式主线。


## 2026-10-22T19:48:30.895836+08:00 独立对照起点及checkpoint核验

启动时原辅助best已更新至step1625，两个候选均由该同一权重初始化（不是故障诊断所用step625）。两组step0全378ROI报告完全一致，all8=7、隐藏误差0.1225807，PAIRED_INITIAL_EVAL.json保留证据。实际step75检查确认头参数已更新；保存头+优化器状态在相同构造特征下复现下一更新完全一致，CHECKPOINT_TEST.json。这不等于完整DataLoader断点重放或精度验证。

初版审计脚本错误地试图读取已被异步评估器正常清理的step0快照；训练未受影响。改为核实并保留best硬链接的step0权重后验证通过，失败记录CHECKPOINT_TEST_INITIAL_FAILURE.json保留。没有为了审计停止、重启或修改训练。


## 2026-10-22T19:52:42.882138+08:00 当前组件完整视频导出

新增src/eval/action4_component_video.py，逐帧RGB推理固定B0/A0/geometry并保存原始三路预测JSONL，直接流式编码原始八点/几何修正双列视频，避免重复保存2850张运行图片。无跟踪或平滑，不改原始抽帧。权重与真实六帧审核相同，仍是未达标诊断；K为假设。

3帧编码预检发现MP4默认movie timescale将时长舍入为1199tick，原帧PTS0/400/800正确。设置movie和video track timescale11988后复核时长1200tick，TIMEBASE_REPAIR.json保留；未重复推理。完整2850帧任务outputs/action4_component_full_video_v001已启动，GPU7+4CPU（含2编码线程），总预算66/72。结束必须检查COMPLETE、帧数、帧率、时长及视觉结果，生成成功不代表精度验收。权威r038。


## 2026-10-22T19:55:55.669200+08:00 B0完整机身框坐标链路验证

新增tests/check_action4_detection_target_roundtrip.py，对正式train/eval/test共2000图3802目标执行标签→中心/offset/ltrb→正式解码→原图完整框。最大误差0.00020806px，数量一致。此为工程契约检查，没有运行模型或据test调参。证据outputs/action4_detection_contract_v001/REPORT.json。

高遮挡1436目标的可见中心/完整中心归一化偏移median0.145888/P950.242877；其余2366目标median0.025392/P950.092710。现有visible-center/full-envelope设计数学可逆，不是坐标缩放bug；对遮挡条件的学习难度需后续同起点中心策略对照，不能未经训练替换锚点。现有训练数据及运行源码保持不变。


## 2026-10-22T19:58:38.862019+08:00 独立组件组合绑定

新增src/model/action4_component_bundle.py及tests/check_action4_component_bundle.py。当前固定B0step5220、A0、geometry分别保存，通过bundle.json引用且不合并权重。对几何特征父权重不仅核对来源字符串，还与原父checkpoint抽取A0逐参数比对；同名来源但任意浮点参数变化、错误组件role均拒绝，测试通过。证据outputs/action4_bundle_contract_v001。当前视频启动后进行了只读绑定核验，COMPONENT_BINDING_REVIEW.json明确时序；没有热改运行推理代码。

后续选优须分别报告B0框指标、A0角点指标、组合真实效果；独立训练产生的新A0即便单项更好，也不得未经重新适配直接替换现有残差模块的特征父模型。当前组合仍有真实失败，不据绑定校验宣布精度达标。


## 2026-10-22T20:03:03.184066+08:00 辅助分支训练拟合与泛化隔离

固定step1625，train/eval各128ROI（各64高遮挡+64其他）：辅助2/128与2/128；原A0为75/128与26/128。辅助隐藏误差train0.117862/dev0.128235，maskIoU两组约0.873/0.872。支持辅助头训练拟合不足，不能只解释为分割或泛化问题；原A0有明显train/dev差距。不能据此认定唯一根因。证据outputs/action4_direction_fit_audit_v001。

已启动outputs/action4_direction_fit_probe_v001：从同step1625起点，固定16个train ROI（8高遮挡8其他），缓存冻结base特征，仅训练现有辅助头500更新，每100检查拟合，无正式checkpoint输出，不接触eval/test训练，不宣称泛化。GPU6 CPU2，临时总CPU68/72；当前正式训练和视频不变。


固定16ROI拟合probe完成，0/16→step100 4/16→200 7/16→300 8/16→400 13/16→500 16/16，平均归一化角点误差0.092083→0.010961。说明当前头可拟合该小集合，暂不因全量早期低分重建网络；该重复训练不证明泛化，也不能推出增加轮数必然有效。无正式权重输出。进程退出后将源文件内容不变归位src/model/training/action4_direction_fit_probe.py，CODE_LOCATION.json记录执行时路径与最终路径；GPU6临时诊断资源已释放，总CPU恢复66。


## 2026-10-22T20:07:09.728806+08:00 异步评估空槽再分配

实际检查原B0三组和A0 baseline均有15–16评估积压，无failed；训练进程仍存活。GPU1旧slot4/5随pruned任务排空退出，slot7此前退休，现实际5cached+1aux。确认身份/退出和GPU1空闲显存后，新增两缓存评估worker：B0三队列池和A0 baseline，GPU1总数恢复8。复用原O_EXCL领取及selection.lock，不改训练源码/评分/频率，不重复已完成评估。证据outputs/action4_eval_rebalance_v001/DISPATCH.json，权威r039；总CPU保守预算仍66。新增worker缓存预热后根据实际完成率判断效果，不能以进程数宣称提速。

本次实测A0 baseline最佳all8达88/378（step6300）；三B0约第46轮，giou最佳AP5095=0.3662385，quality最佳0.4284543；都是开发选优，尚无最终真实达标。完整视频175/2850且真实PID存活，无FAILURE，继续。


## 2026-10-22T20:11:20.540782+08:00 前臂开放截面定位

只读Blender审计3个真实执行模板、12个前臂对象：natural_hand各44开放边，continuation各88开放边；6个continuation姿态中5个存在投影在画内的开放边界。自然手与延长模块接合处也有边界，因此不能把所有开放边都当可见断口；结合此前实际渲染确认存在末端截面。src/data/action4_arm_boundary_audit.py与outputs/action4_arm_boundary_audit_v001保留证据。无渲染、无资产修改。

修复定位为独立手臂/身体连接及姿态约束，固定腕部远端几何和DJI安装关系；不简单拉长整只手改变比例，也不以封口当真实感达标。需区分接合环与末端环，再验证末端与身体/袖口连接和画面边界。用户允许继续用原2000训练，当前训练数据不变；未来候选通过视觉和角点核验前不发布。


## 2026-10-22T20:13:56.340787+08:00 独立前臂末端封闭候选

src/data/action4_arm_endcap_candidate.py已实际生成data/action4_scene_assets/arm_endcap_candidate_v001/wall_diagnostic.blend。先拓扑分离两个44顶点环，再按与自然手臂开放环的世界坐标距离识别：接合环距离0，末端约0.3156m。仅封闭真正末端，左右各88开放边→44（保留精确匹配的接合环）；所有原顶点位置不变，原资产不覆盖，独立mesh副本。运行模板transform检查与canonical8导出比较，角点及可见性输出逐项相同。

这是拓扑候选，不是完整人体连接修复。未渲染审核，画内末端仍可能成为封闭残肢，身体/袖口连接及姿态边界仍未完成；不得用于训练或替代正式场景。证据outputs/action4_arm_endcap_candidate_v001/REPORT.json，旧2000不变。


## 2026-10-22T20:17:56.043626+08:00 几何求解解析雅可比加速候选

新增src/model/action4_rigid_projection_analytic.py：保持相同DLT+24姿态初值、Huber损失、求解预算和输出规则，仅以解析投影雅可比替代数值差分。tests/check_action4_projection_jacobian.py覆盖零角、小角、近pi、随机旋转及非零skew的15例，最大导数绝对差4.79e-8。

实际6真实帧12预测目标对照，原11.864s/候选7.776s（1.526倍），最大ROI角点差8.954e-6px，接受/拒绝原因全部一致。src/eval/action4_geometry_speed_audit.py与outputs/action4_geometry_speed_v001保留逐例证据。这是局部算子加速与等价性，不代表整视频提速或精度提高；现有视频源码/进程保持不变。退化/低置信等边界和全链路验证后才能作为后续默认。


## 2026-10-22T20:20:07.235917+08:00 解析几何边界与完整链路验证

7个边界案例通过：无置信、仅3点置信、点塌缩、共线均保持相同拒绝原因；零角/微角/近pi加噪投影的拟合输出最大差约2.2e-8px。证据outputs/action4_geometry_speed_v001/BOUNDARY_TEST.json。

独立src/eval/action4_geometry_pipeline_analytic.py与既有候选只替换solver import；原视频使用的pipeline文件不改。已结束的geometry_comparison入口增加显式--solver选择。启动outputs/action4_geometry_analytic_full200_v001，硬链接同一B0/A0/residual，完整200图与原参考比较原始预测、检测身份、每点偏差和最终计分，GPU6 CPU2，临时总68。该检查完成前不默认推广；r040记入任务，原视频和训练继续。


## 2026-10-22T20:24:41.758024+08:00 几何配对验证自动收尾

新增src/eval/action4_geometry_equivalence.py，等待已核实身份的full200进程完成后自动核对：200唯一同序ID、实例数/ID/框/分数、三路全部角点最大差<0.001原图像素、两档阈值TP/FP/FN/精确率/召回/F1。任何差异保留EQUIVALENCE_DIFFERENCES，不自动放宽标准。源进程在REPORT前退出则记录失败且不重启。输出EQUIVALENCE.json，当前POSTPROCESS.json记录真实进程；临时CPU上界69/72。

构造夹具tests/check_action4_geometry_equivalence.py通过：完全相同输出通过；即使所有汇总分数相同，只改一个角点0.02px也被拒绝。该夹具不是模型成绩。当前200实际推理继续，不能提前宣布全量等价或推广。


## 2026-10-22T20:30:02.279209+08:00 全200解析几何对照收尾（严格等价未通过）

全200唯一同序样本、实例数量/ID/框/分数一致。raw坐标逐项相同；rigid最大原图坐标差0.000242692px，满足0.001px门槛；visibility_residual最大差0.00373984px，超过事先门槛。两档0.02/0.05下TP/FP/FN/精确率/召回/F1全部相同，但不得将分数一致冒充严格输出等价。EQUIVALENCE.json及ROOT_REVIEW.json保留失败，不放宽门槛，不替换当前视频/default。

该结果限制了12实例局部通过的推广范围，未证明真实精度退步，也未证明候选完全等价。后续如研究数值实现允许差异，需另行明确协议，不能回写本次为通过。全量任务与收尾结束，临时GPU6/CPU资源释放，权威r041，保守常驻CPU66。角点解码检查另确认主线/辅助/视频均top20均值，未发现参数冲突。


## 2026-10-22T20:32:57.920689+08:00 B0同卡评估补充

旧空槽再分配后，B0积压15–16→6–8，A0baseline→1，无failed。GPU1约93%利用，GPU2/3/4瞬时12%/42%/9%，各余显存约40GiB。新增三B0同卡eval各1CPU，GPU1仍8，全部复用atomic claim+selection.lock；无训练源码更改、无重复队列或放松评估。全局CPU69/72，派发outputs/action4_eval_training_cards_v001/DISPATCH.json，权威r042。需根据实际完成和训练吞吐核验收益，启动不是速度证明。


## 2026-10-22T20:38:19.972752+08:00 利用现有2000数据推导稠密表面对应可行性

已确认正式数据无深度/XYZ图，但有精确合成姿态、内参、可见mask和冻结CAD。新增src/data/action4_surface_probe_inputs.py及action4_dense_surface_probe.py，以ROI112网格采样可见mask、原图射线逆变换进object坐标、BVH与冻结mesh求交。先核对冻结mesh与模板left_Action4_core顶点一致，然后对前2个train图4目标求交，2720有效mask采样中2717命中、3无效保留，最大回投影0.00033014px。源RGB和已有标签不变，无渲染。

这是由已知合成几何推导的表面对应，不是实测深度，也不能据4目标宣布全量可用。未发布训练标签，输出outputs/action4_dense_surface_probe_v001。后续若扩展，需核验左右机身、全场景模型版本、mask边界/贴花无效像素、GT信息仅进监督不进推理。可为GDR-Net式稠密几何中间监督准备sidecar，保持A0/B0独立及当前主线不变，不宣称完整复现。


## 2026-10-22T20:43:03.679672+08:00 全量合成表面标签候选生成

3模板×左右core共6项冻结mesh绑定通过（顶点、三角面、无modifier），见outputs/action4_surface_labels_v001/MESH_BINDING.json。新增surface_inputs/surface_labels/surface_dispatch三个data模块，先用1CPU顺序准备mask可见ROI112射线，随后3个CPU-only Blender BVH worker生成3802目标附加标签。分别写data/train|eval|test/surface_v001_candidate，保留源划分，原RGB/annotation/mask不覆盖，未进当前训练。最大额外3CPU，总72；仅求交，无RGB重渲染。

输出仅保存实际命中的object XYZ/native像素/depth/face索引和未命中grid索引，无效位置不填造XYZ。每个有效点必须有限、正深度、回投影<=0.005px，否则整个worker失败并保留记录；不能默默丢弃错误对象。完成回执仍CANDIDATE_NOT_TRAINING_ACCEPTED，后续需全量完整性、序列化回投影、覆盖率及少量可视核验，再做新训练候选。权威r043。


## 2026-10-22T20:54:15.013926+08:00 表面标签精度失败与v002修复

v001生成3802对象、2820528命中点、8172未命中、57835684字节；独立几何验收失败，不能发布。只读扫描定位s209005_0000647_a00_i1的grid7443/face51399：BVH返回点距三角面平面1.0417758e-6m，超过1e-6m门槛，使用float32网格重算仍1.0411e-6m，非单纯mesh类型差。FAILURE_DETAILS与REGRESSION_RED.log保留复现。

新data/action4_surface_intersection.py用原像素射线/命中三角面双精度精确求交，再检查正射线参数、平行退化和重心坐标有效性；实际失败样本写盘后平面误差4.9242e-10m、回投影1.18765e-6px，原门槛不变。支持BVH浮点求交精度是故障来源，旧标签不就地改。

通过action4_refine_surface_labels.py生成data/train|eval|test/surface_v002_candidate共3802文件，旧v001全部保留。179个BVH命中在精确射线/三角形检查中不成立，显式precision_rejected_grid_indices并入无效索引，未伪造XYZ或静默丢弃对象。v002独立全量验收正在运行，未接训练，权威r044，CPU70/72。


## 2026-10-22T21:16:25.796100+08:00 用户要求收敛重点：成熟方法复用与B0质量融合

已使用anysearch学术检索和浏览器官方仓库，来源outputs/action4_method_research_v001/SOURCES.json。A0新增主线只保留GDR-Net/GDRNPP的RGB稠密几何中间监督方向：RGB ROI→预测可见/完整mask与表面XYZ/区域→几何姿态→固定canonical8投影。保留当前独立A0/B0与backbone，不默认整体替换ConvNeXt，不引入所需观测深度的refinement。ZebraPose表面编码是连续XYZ歧义严重时的备用；PVNet现有辅助不代表完整复现，继续完成对照但不再无证据融合；VAPO可见性重要性仅后续小模块。FoundPose为可选独立参考，公开实现无featuremetric refinement，不作为立即替换。公开BOP权重不等于现成Action4权重。

B0实际梯度检查2真实训练图：GIoU对offset/ltrb/共享backbone有非零梯度；quality对quality_logits/共享backbone有非零梯度，对offset/ltrb无直接梯度符合IoU target detach。现实现为CenterNet式中心+GIoU+独立QFL-inspired质量头，不是完整GFL/VFNet。训练hard pool按center峰，推理按product峰；quality头没有显式读取预测框ROI/边界特征；GTanchor可能与hard pool重复。这些是需要对照的设计风险，不把所有差异直接称bug。

真实frame1000冻结step5220：正确相机候选中心0.991519、q0.346705、乘积0.343764、demo参考IoU0.865497；误检中心0.978694、q0.369284、乘积0.361416、参考IoU0。正确候选被排序挤出前2，非该帧根本没有候选；不能用提高阈值或固定左右位置解决。证据action4_b0_signal_audit_v001。

同权重全200评分消融：center-only AP50:95=0.3890665/F1=.906667/双机147/184/FP32；quality-only=.4274894/.890339/148/184/FP47且重复9；product=.4284543/.909091/147/184/FP30且重复1。保留GIoU+quality收益，针对真实质量排序和机身/支架/手部难负例做修复；不因为单帧失败移除质量头。下一B0候选优先框感知质量特征+统一候选监督/校准，参考VFL/GFL联合语义，不把q直接当IACS使用。

后续只推进一个A0几何增强和一个B0评分修复；现有80轮训练继续，不再围绕微小数值一致性或次要手臂封口扩展零散实验。稠密表面v002已3802数值验收及6例视觉核验通过，可用于独立候选监督，旧v001失败保留；这不代表场景真实感或模型精度达标。每次增强必须报告单帧/真实双机识别、隐藏点、恢复与退步；最终全视频仍是目标。


## 2026-10-22T21:35:14.655570+08:00 GDR表面分支真实接口与固定特征缓存

从upstream/gdr_net固定commit1be9fe7的cdpn_rot_head_region.py复用no-concat预测头，src/model/heads/gdr_region_head.py保留层次及输出，去除不使用concat路径，将mmcv初始化/归一化替换为Torch等价函数；完整Apache2许可及PROVENANCE随代码保留。应用采用2048输入、64filters/GN8、3级上采样、2独立mask通道、XYZ3通道、64区域+background。非完整GDRNPP复现。

新增action4_surface_head与data/action4_surface_data，后者从v002精确坐标构造normalized XYZ、64个确定性CAD FPS区域、可见与完整amodal掩码；监督与RGB/骨干输入分离。测试先失败再实现通过：构造梯度验证无效XYZ位置无梯度；真实train首ROI607有效点产生新头梯度norm17.9585，原A0骨干无梯度，坐标网格检查通过。均非精度结果。

锁定当前选优A0step6300作为base_best.pt，新增surface_cache一次缓存原train3024/eval378的固定GT ROI特征与独立标签，保持FP32、无jitter首轮控制条件；这是GT ROI候选学习，不能代替完整B0检测效果。缓存约需数GiB，避免每epoch重解码大RGB/amodal掩码及重复冻结骨干。CACHE_PROCESS真实GPU6 CPU2，总71。缓存后仍须接姿态求解、正式训练与25step评估；本轮未宣称已启动新分支训练。原训练/视频不改，r046。


## 2026-10-22T21:50:24.731118+08:00 表面预测到八点与连续训练自动衔接

OpenCV headless4.13.0.92安装在runtime/surface_deps，--no-deps不改现有venv，缓存和临时目录限项目内。src/model/action4_surface_pnp.py使用预测XYZ与预测visible概率，通过RANSAC EPnP+LM估计姿态，再投影固定canonical8；接口不接收GT姿态/掩码/坐标。K和CAD是已知几何先验；真实K仍为假设。少于6对应、塌缩/共线、无一致性、机身负深度等拒绝并保留失败，不填真值。

几何测试真实合成对象oracle对应最大角点误差1.84e-5heatmap像素，20%错误对应0.0460，空mask/塌缩拒绝。只是解码验证。GPU严格确定性head测试通过，不需要更改上游双线性操作；2个实际缓存ROI执行loss/backward/optimizer/PnP评估链路通过，loss3.1959、gradnorm29.9225，不是精度。

新增surface_train/surface_eval/surface_dispatch，cfg action4_gdr_surface_train_v001：80完整epoch、batch16/micro4、lr.001 cosine、XYZ1/region.1/mask1、每25step全378ROI评估，invalid_pose计失败，隐藏均值仅有效解且必须同时报告coverage。head独立last含optimizer/scheduler/epochcursor/RNG，best为选优权重；骨干固定step6300。评估只输入features，GT targets不进入网络或PnP。原图检测效果需后续B0组合验证，不能用GTROI成绩替代。

控制器真实PID已启动等待CACHE_COMPLETE和缓存PID退出，通过计数/工程测试/显存检查后自动派发GPU6的一train+一eval，替代缓存的2CPU，常驻预算保持71。缓存失败不启动；新分支一方失败仅停止本控制器拥有的另一方并保留错误，不重启旧任务。当前尚未开始新分支正式训练，r047。


## 2026-10-22T22:06:21.707179+08:00 B0框感知IACS分支与一致候选缓存

新增model/action4_box_quality.py：从预测框九个位置采样冻结B0 decoder特征，加宽高/比例/面积特征，轻量评分头输出联合质量；旧center/q不作为新头输入，避免只学习旧分数组合捷径。VFL公式参考官方原论文，独立适配而非完整VFNet复现。构造测试先失败再实现通过，验证框改变会改变采样特征、连续IoU目标产生梯度。

训练与推理共用eval/action4_box_quality_pool.py的20center峰+20product峰去重并集，不添加GT候选，不把同一grid当重复监督。训练目标采用完整机身maxIoU>=.5为正，target=IoU；其他背景为0，匹配ignore对象且未匹配有效对象的候选不参与监督。预测框不回归，骨干冻结。新增training/action4_box_quality_cache.py，以实际B0 forward hook捕获框特征；真值仅用于监督匹配。

锁定当前best B0 step5220，configs/model/action4_box_quality_v001.json；GPU4 CPU1缓存原train1600/eval200，当前A0cache2+原69+新1=72。后续新分支训练须等资源释放，至少80轮、每20step框指标eval，不能声称缓存即训练。当前正式B0与A0/视频不改，r048。


## 2026-10-22T22:53:32.404807+08:00 新B0训练完成及真实得失

原三B0组件对照完成80轮16000step；框感知评分头亦完成80轮8000step，0..8000每20step共401次评估完整无失败，best800。旧去重下AP5095 .424019/F1 .883355/双机145/184/重复20，不能替换原模型。

固定2x2后处理对照：原并集评分+IoU-NMS.5 AP.428509/F1.909091/双机147；新评分+硬NMS.5 AP.435452/F1.908851/双机151，但s209005_0000755_a01两真实GT框IoU.5158，硬NMS误删一台。复用旧已验证Soft-NMS代码，非参数搜索：Gaussian sigma.5的新评分AP.434773/F1.915119/双机152/184/FP31/重复0，并保留该重叠双机反例。证据action4_box_quality_nms_audit_v001与action4_box_quality_soft_nms_v001。

真实6帧视觉审核及demo参考匹配：原方案5/6，新评分+Gaussian仍5/6；修复frame1000手部误检及相机漏检，却在frame400漏掉原本检出的右侧相机。ROOT_VISUAL_REVIEW明确不推广。不能只报恢复不报退步，不降低阈值补单帧。后续B0继续同一评分分支，优先保留原检测置信先验的残差校准/针对性数据修复，避免绝对新分数覆盖有用原信息；不单纯加训练次数。

A0新分支实际训练与25step评估进行中，当前best 11300，GTROI all8 247/378，高遮挡 58/164；不可与完整B0链路或真实视频指标混报。原文档与r049记录实际状态，旧终态B0不重启。


A0表面分支正式收尾核实：80轮15120 optimizer steps，0..15100每25step及最终15120共606次评估全部完成，无缺失/failed，控制器正常退出。best11300，GTROI247/378、高遮挡58/164。FINAL_REVIEW标记READY_FOR_REAL_PIPELINE_REVIEW而非生产通过；下一步必须在预测框及真实RGB上验证。


### 2026-10-22 B0候选瓶颈与残差细化工程验证

CANDIDATE_COVERAGE.json已对184张双机开发图完整归因：oracle169，实际152；候选不足15、排序/抑制9、分数不足8。该结果支持同时改框回归和质量评分，不能仅改阈值。

独立候选模块已实现；preflight使用实际训练缓存16张执行12次更新，核验框初值等于原框、初始评分保留旧乘积、GIoU/SmoothL1有框分支梯度、quality不通过detach目标偷更新框、外观与旧质量头参与梯度，以及权重往返输出一致。首次测试错误地要求动态IoU监督下总loss单调下降，已保留失败；实际框GIoU loss从0.52438降到0.48088、回归从0.15027降到0.13510，质量目标随框变化而改变，总loss不用于该固定目标断言。质量梯度/有限值要求保留，正式效果仍必须看eval，不以工程通过代替。

A0 GDR候选step11300在6张真实参考图12个预测ROI中11个数值有效姿态，但可视化存在过大、偏移和朝向错误，未达到稳定锚定标准；数值有效不等于准确。outputs/action4_surface_real_review_v001仅诊断，未默认替换。


B0 residual candidate interim step4220实际200开发评估：mAP50:95=0.46776067、F1=0.94986807、双机166/184=90.2174%、重复2、空图误检0/6。仅开发达90，不代表任务达标。固定同一step4220真实6帧比较，demo参考一致双机从原5/6降为3/6；第1000帧手部误检仍在，第1600帧背景误检，第2200帧手部误检。已肉眼查看comparison.jpg，ROOT_VISUAL_REVIEW拒绝默认推广，失败不筛除。当前80轮连续训练不改动；最终best须重新真实验证。证据outputs/action4_box_refinement_real_review_v001/REPORT.json及ROOT_VISUAL_REVIEW.json。


### 2026-10-22 GDR完整几何训练链路适配

原版GDRN.py将预测XYZ、2D坐标、region attention、mask送入ConvPnPNet，并对输出位姿施加point matching等损失。旧本地surface分支只有XYZ/region/mask损失，region输出及amodal mask没有直接进入OpenCV求解，最终8点误差不反传；这是确认的实质方法差异，尚不能单独证明全部真实误差由其引起。

新增src/model/action4_gdr_pose.py，复用上游ConvPnPNet与6D旋转代码（upstream commit1be9fe73292fd748087aa88d7bf987434f271ebb）。保留原层结构；替代mmcv初始化/归一化，限定drop_prob=0，避免原inplace改调用方输入。应用侧输入64网格，预测XYZ+相机射线+64region+可见/完整mask，ego6d和正深度相对参数化；输出固定8点刚性投影。位姿loss采用固定ID八点相机坐标point matching、可见/隐藏2D分别归一化、正深度约束，保留原surface监督。不是完整GDRNPP复现，不假称采用原版所有参数。

configs/model/action4_gdr_pose_v001.json：从surface best11300、原A0骨干best6300继续，1600图3024ROI，80轮=15120steps，前5轮仅新pose，后75轮联合surface+pose；骨干保持冻结以复用已有缓存，GT ROI仍是已知局限。每25step独立异步eval378ROI，选all8@0.05及明确次指标最佳。推理只用预测features、K和固定CAD；GT pose/visibility/corners仅loss。旧A0及已完成B0不改。GPU4训练、GPU1评估，CPU新增2。训练后的预测B0 ROI和真实视频验证是必需后续，不可用GTROI结果替代。

工程preflight检查原版前向精确一致；实际2ROI GT位姿重投影误差4.58e-5px；仅几何loss对XYZ/regions/masks梯度范数分别3.12e-5/1.24e-4/9.19e-6，双mask通道均非零；严格确定性CUDA反传也通过。证据outputs/action4_gdr_pose_candidate_v001/preflight_deterministic.log，工程证据不代表精度。


### 2026-10-23 A0子智能体交叉审查及版本化修正

按用户本次明确授权派2名Astra只读审查：一名high查上游方法/梯度，一名medium查数据/日志，主控复核并写入修正。审查没有修改运行源码。有限抽查32ROI支持canonical8/米制pose/ROI K一致；不发现GT pose、corners、mask输入网络。固定378条eval身份与父模型一致，仍仅GTROI诊断。

确认microbatch归一化问题：原可见/隐藏各micro求均值再按ROI比例累积，会使监督随分块变化。主控新反例旧micro16 loss0.190305/grad0.3348、micro4 loss0.17631275/grad0.30768928；修正后micro16/8/4/1完全一致。新src/model/action4_gdr_pose_losses_v002.py使用整optimizer-batch可见/隐藏分母，角点项不再重复乘micro比例。

参考上游rotation-only PM+独立centroid/z，修正版把耦合RT-L1拆开，保持当前ego6d/softplus坐标参数化与固定ID。大平移偏差时独立旋转梯度0.69603，GT位姿几何loss为零，全可见/全隐藏有限。此为失败模式修复验证，不是已证明当前精度失败全部由它引起。又用实际2ROI与冻结step525权重验证v002总loss和几何输出反传，actual_data_preflight.log通过。

异步恢复修正：独占worker锁后回收无done/failed孤儿claim；ACTION4_ATTEMPT_ID隔离旧终态；BEST.json指向不可变selected_step权重，done最后提交。CPU文件故障注入验证before_alias/before_done重放。明确fail-closed：捕获异常的.failed保留，不自动重试；显式resume若有failed默认拒绝，需明确--retry-failed并归档历史。不可把helper硬退出测试说成所有异常自动恢复。

当前v001不改不停止，继续完成80轮；v002控制器已排队等待其成功完成，从实际best权重及对应step receipt绑定后续训80轮、每25step异步eval，warmup=0联合训练，独立新loss目标与新优化器状态；不是等预算纯消融。声明configs/model/action4_gdr_pose_v002.json，父完成后生成configs/model/action4_gdr_pose_v002_resolved.json。GPU4训练/GPU1测评，新增并发2CPU，仅继任前一run；权威r051。模型/data/eval源码职责分开，outputs仅证据与状态。

最新一次主控读取v001为step5951 epoch32，best5650 all8=119/378 heavy17/164，低于父surface+PnP247/378与58/164；不得宣称已改善或真实达标。新版本尚未训练，效果待检。v002的spec审核PASS，quality按明确fail-closed边界限定PASS，主控通过源码检查及实际回归验证。

B0本次完整80轮/8000step/401eval已结束：最终按mAP选step6300，AP50:95=.47032656、F1=.94583884、双机164/184=.891304，未达90%。中间4220双机166/184不能与6300最高AP拼成绩。对最终6300同6真实图同候选2x2诊断：旧框旧分5/6，旧框新分3/6，新框旧分4/6，新框新分3/6（demo参考非GT）；排序回退与框修正均有贡献，不能称为单一NMS问题。保留原B0正式基线，新候选不推广；不得以开发集替代真实验收。


### 2026-10-23 B0图像特征与框质量联合训练

问题依据：离线refinement best6300在开发集AP=.47033/双机164/184，真实同6图2x2显示旧框新分由5/6降3/6，新框旧分4/6，冻结特征上的重排未纠正真实语义混淆。独立新版本保留原B0+refiner结构，从B0best5220及refinerbest6300初始化，重新在线读取原1600训练图，让refined质量/GIoU/SmoothL1通过9点双线性特征采样真实反传到B0骨干。离散候选索引、框位置和旧置信先验仍detach，dense框/中心/质量损失继续监督基础检测器；不把目标IoU回流为推理输入。训练/推理复用相同候选并集与特征采样。候选纯数学从既有评估代码提取到src/model/action4_proposal_pool.py，避免模型依赖评估运行器。

原候选并集与新共用实现实际逐项一致；新确定性双线性采样对既有grid_sample最大差0.0001202（float32特征单位）；实际RGB refined质量/GIoU/回归到backbone梯度范数分别.54547/1.20278/.46428，实际更新有限。batch2前反向1.56s/3.44GiB；batch16前反向3.74s/26.01GiB，明确只是单次资源测量非稳定吞吐预测。

配置configs/model/action4_online_b0_v001.json：2rank GPU5/6，每rank16、全局batch32、micro16，无padding或重复采样，80epoch=4000optimizersteps，每20step完整200开发eval，test不参与。GPU1专用eval，2rank各2计算+6loader，加eval最多3CPU，新增预算19（总仍小于72）。沿用已验证NCCL_P2P_DISABLE=1，max_restarts=0，观察实际训练更新而不重复D0。最佳按mAP50:95/F1；双机>90%与真实验收仍独立必须满足。新方法是现有模型的在线联合增强，不宣称完整VFNet复现。A0现有训练/修正版队列不变，不改原2000数据、不新增渲染或隐含标签。

新源码src/model/action4_online_b0.py、src/model/training/action4_online_b0_{batch,train,dispatch}.py、src/eval/action4_online_b0_{pipeline,eval}.py。输出outputs/action4_online_b0_v001。训练启动与效果以DISPATCH/真实PID/STATUS/eval为准，工程验证不是精度。



2026-10-23 B0在线联合运行已验证实际双rank更新：GPU5/6 local16/global32，step6与50均完成，无NCCL超时；约2.9–3.1s/optimizer step（非全程统计），每rank峰值约26.4GiB。首step40完整200开发eval：双机167/184=90.7609%，mAP50:95=.47106950。真实同6帧仍3/6，1000手部、1600背景、2200手部误检未消除，不推广，不宣称真实90%。原B0参考5/6，demo不是GT。证据outputs/action4_online_b0_real_step40/REPORT.json与ROOT_VISUAL_REVIEW.json。

新增src/eval/action4_staged_pipeline.py串接独立online B0与GDR A0，固定阈值.3、权重config身份和A0骨干身份检查，使用B0预测ROI与crop K，固定8点ID；无GT框/位姿/mask输入。实际真实RGB frame0用固定B0step40+A0step525做接口检查成功，STAGED_INTERFACE_CHECK.json明确仅接口、不作为accuracy。等待两条训练完整选优后再进行真实最终验证与视频。


2026-10-23进展：A0 GDR-ConvPnP v001已80轮15120step/606eval完成，best12550为172/378、高遮挡26/164，仍低于父surface+PnP247/378、58/164。v002自动从实际best12550完整head初始化并进入训练，严格新loss目标，权威r053；CONTROLLER_STATUS中的WAIT是启动前历史，DISPATCH、真实PID、STATUS已经证明运行，不得据该旧字段重复启动。B0 online初期best60开发AP=.483319、双机168/184；实际真实检查仍仅step40=3/6，不能把60成绩与40真实结果混为同一权重。


### 2026-10-23最终交付、封存和Git发布要求

用户明确要求：所有相关训练和验证完成后，继续完整真实视频生成与效果检查，封存各正式阶段最佳模型，整理源码与文件结构；无用过程输出删除，其余历史资料统一归档，编写conclusion并推送https://github.com/woaijieyang/0003_Action4_Detect。

执行次序：完成当前A0/B0及相关试验→固定最佳权重与指标身份→全量eval、真实视频及可视化检查→制作自包含A0/B0 best（每阶段各一份；包含实际使用的父模块参数）→验证独立加载/继续训练入口→按依赖清单保留/归档/删除→整理最终conclusion及README/配置/源码→显式Git清单审阅与发布。当前不清理运行依赖。

统一归档目录archive/legacy，原相对布局与迁移清单保留。最终模型拟存models/A0/best.pt、models/B0/best.pt，其他checkpoint仅在确认不再依赖且封存验证完成后删除。没有optimizer的历史best仅支持从best权重初始化新优化器继续，不伪称精确恢复。原始数据、已接受标签、冻结Blender模型、必需runtime和环境保留；Git只收源码/配置/文档/小型清单，数据/权重/大输出/历史资产不提交。

准备证据outputs/action4_release_preparation_v001：INVENTORY、SOURCE_CLOSURE、CHECKPOINT_INVENTORY、TEMP_CHECKPOINT_AUDIT、CLEANUP_PLAN。静态闭包58文件、未被该闭包引用候选160文件不是自动删除许可，仍需动态CLI/渲染与文档引用审核。models逻辑575852956745B，但按inode去重分配245183930368B；1013.tmp名字对应273inode，禁止把逻辑大小当可回收空间。当前无删除、无归档移动。

远端origin正确，服务器main与远端main均d88d61eede18382b788fa452054a19450bc5e19a。禁用全局凭据的服务器Git dry-run失败；已通过GitHub连接器确认当前账号对此仓库具有push/admin权限，后续可使用连接器发布，不读取服务器目录外凭据。未执行任何提交或远端写入。AGENTS已补齐GPU1测评授权及收尾规则，.gitignore已排除archive资产。


2026-10-23补充闭环证据：同A0v002 step3050，GTROI网络179/378 heavy32，预测对应PnP231/378 heavy51，网络初始化迭代PnP229/378 heavy55。PnP相对网络恢复75、退步23，非全面胜出，未更改训练/默认推理。B0step180+A0step3050+iterative在完整200开发图预测ROI得224/378、heavy42/164，bbox匹配360/378，漏框18，invalid pose0；框与角点指标独立。3402 train/eval目标full bbox与canonical8投影包围框差为0，旋转矩阵det接近1且正交误差<=1.11e-6，未发现反射/非刚性标签问题。

视频链src/eval/action4_staged_video.py及parallel.py已验证3帧串行和2进程4帧分段合并，2997/100fps、1/11988timebase、duration_ts分别1200/1600正确；全局frame_id/PTS/模型身份逐条验证。actualRGB首帧已视觉查看，机身角点仍有偏差、B0框包含过多支架，不能当准确验收。完整2850帧尚未生成，等训练完成与最佳选择。parallel支持最多10进程，按每进程6CPU+既有占用严格<=72，失败只停止自有进程组；故障取消测试通过。


### 2026-10-23 A0完整训练结果、解码一致性与收尾加速

GDR-ConvPnP v002已80轮完成，最佳step13350。对齐原评估的确定性设置（torch deterministic + CUBLAS_WORKSPACE_CONFIG=:4096:8）后，同378 GT ROI：直接网络190/378、heavy35/164；预测对应点PnP236/378、heavy53/164；网络初始化迭代PnP232/378、heavy55/164。均未超过原surface11300+PnP的247/378、58/164，不因新版本就推广。保留原surface11300、只接入新pose13350的组合也未改善：直接网络166，PnP247，iterative235。组合已明确标为非单一训练时刻，不混称step13350原权重结果。

发现比较脚本未启用与训练评估相同的确定性设置，导致原surface基线复算245而非247；对齐后恢复247/58。已将staged推理固定到相同设置并记录在identity；不改正在运行的B0评估源码。最终成绩须在冻结发布运行配置下重算，不混用旧非确定性比较报告。

已匹配360个框的中途完整链路（B0step180/A0step3050/iterative）中224成功，136失败分成69仅隐藏点超阈值、67可见与隐藏都失败；允许长方体180度对称置换也没有恢复失败样本，置换仅诊断，不改变正式固定ID标准。不能将这组诊断外推为全部真实失败原因。

当前实现仍有明确适配差异：A0 RGB骨干冻结，几何头联合训练并不等同原GDRNPP端到端RGB完整训练协议；B0为现有CenterNet式骨干/解码加框细化与质量评分，不是完整VFNet的FPN/ATSS/DCN原配置。这些差异记录为限制，不虚称复现论文指标。按照用户本轮收尾要求，不在此时另开新的训练分支。

为完成既有旧A0对照，新增只读冻结特征评估缓存：outputs/action4_aux_eval_acceleration_v001。全部378 ROI完整报告逐项相同；一次原流程首用91.77s（含首次图片加载），缓存准备36.81s，缓存评估10.71s；原活动worker近期稳态约43–57s，不把首用8.57倍宣称为普遍稳态加速。4个GPU4辅助worker已真实完成多项评估，约10–13s/次。它们仅领取未完成队列、写真实done、保留snapshot，不与原worker竞争BEST；全部80轮/606eval、原writer退出后统一核对并更新最终best。缓存是临时运行资产，不是正式模型，所有消费者结束后纳入清理。

## 2026-10-23 训练收尾、冻结模型、实际视频与交付

状态：本轮主训练及完整视频已执行；**真实任务验收失败，不能声明完成了可靠双机锚定**。本节覆盖此前“仍在训练”等过程状态，保留历史原始记录。

### 完成的实际训练与选优

B0 online 联合检测/质量修正训练完成80轮、4000 optimizer steps，201次开发评估（step0及每20步）。按 mAP@0.5:0.95、再框F1选出的 best 为step2200。A0 GDR v002完成80轮、15120steps、每25步评估，网络解码选出的 best 为13350；不把最后一步当best。

在最终B0预测的同一批ROI上，A0 joint＋PnP八点成功222/378、高遮挡48/164；joint＋迭代PnP为219/378、47/164；旧surface＋PnP为219/378、46/164。选择joint＋PnP作为本次开发集候选包。该选择不覆盖此前真实审核失败，也不证明它优于所有历史模型在真实视频中的表现。

| 指标 | 开发eval | 历史test |
|---|---:|---:|
| 图像数 | 200 | 200 |
| 框precision | 0.984085 | 0.987469 |
| 框recall | 0.981481 | 0.985000 |
| 框F1 | 0.982781 | 0.986233 |
| mAP@0.5 | 0.979111 | 0.973173 |
| mAP@0.5:0.95 | 0.548464 | 0.561451 |
| 双机同时检出 | 177/184（96.20%） | 194/200（97.00%） |
| 重复框/预测框 | 0/377 | 1/399 |
| 空图误检 | 0/6 | 无空图，无法估计 |
| 固定ID八点全部误差≤框对角线5% | 222/378 | 245/400 |
| 高遮挡八点成功 | 48/164 | 38/142 |

test曾参与历史项目评估，本次仍明确标为历史test，不是新盲测。独立400候选的视觉失败门禁仍有效，没有发布或用于正式验收。分场景/距离/遮挡详情保留各REPORT.json，八点成绩不替代B0框成绩。

### 真实画面检查

B0step2200在6张同帧demo参考中仅2/6双机匹配IoU≥0.5，旧B0参考为5/6；demo不是认证GT。候选边界诊断表明，第1000、1600、2700帧的相机候选存在，但质量排序落后于手/衣物/背景；第2200帧一台相机候选最高IoU约0.412，修正后约0.275。已定位具体失败阶段，尚未证明单一根因或修复成功，不能仅用更多训练轮数解释。

已生成完整2850帧、1900×750、2997/100fps的demo左/模型右视频；帧ID、400ticks/帧、1/11988timebase及最终时长均验证。所有帧均保留真实模型预测。5688个检测中828个位姿无效，分布在771帧：704为预测表面不足，116为重投影不稳定，4为负深度，4为退化对应点。检测框数量为2的2838帧并不意味着两台真相机检出，不能把该数量当召回率。

主控实际检查0、250、500、750、1000、1250、1500、1750、2000、2250、2500、2849帧对照图：多帧衣服/手误检、夸大的错误cuboid；起始及最后帧较接近也不能覆盖中间失败。明确结论为VISUAL_REVIEW_FAILED_REAL_TASK_NOT_ACCEPTED。真实K仍为估计，未实测；无时序训练/平滑。

### 封存、后续训练与代码

封存候选为服务器models/A0/best.pt（包含冻结backbone和GDR任务头）、models/B0/best.pt（完整检测器与refiner），两个阶段独立，均标注真实验收未通过。保留历史最佳参考，避免新候选的真实退化覆盖原基线。

独立包在两张实际图像上与封装前输出完全一致。A0与B0各执行一次实际训练损失、反向传播、参数更新和保存后加载一致性检查。检查更新没有进入正式best；best无优化器历史，后续明确新AdamW热启动，不拼接last优化器。当前训练器新增best_bundle可选分支，下一实验配置已准备但没有再次启动80轮。A0保留surface/pose分别学习率。

代码目录继续为src/data、src/model/training、src/eval。README由模板说明更新为实际项目状态与入口。补齐model/heads/__init__.py，修正wheel漏打包GDR头的风险；打包包含上游许可证和来源说明。构建成功及语法检查是工程结果，不是精度证明。

### 主要证据

- outputs/action4_final_selection_v001/FROZEN_SELECTION.json：同一B0下A0对照与冻结选择。
- outputs/action4_final_test_v001/REPORT.json：冻结后的历史test。
- outputs/action4_final_proposal_audit_v001/REPORT.json：真实候选与排序边界。
- outputs/action4_final_video_v001/：完整comparison.mp4、逐帧predictions.jsonl、IO回执、统计和ROOT_VISUAL_REVIEW。
- outputs/action4_release_bundle_check_v001/REPORT.json：封装等价与实际更新检查。
- outputs/action4_release_preparation_v001/：归档/删除清单、打包与发布核查。

### 未解决项与结论边界

真实双机检测稳定性、完整机身框精确覆盖、隐藏八点可靠性均未达到任务要求。合成场景域差异、真实负样本与质量评分排序仍是后续改进方向；本轮没有通过换阈值、筛失败或使用demo答案来修饰结果。代码发布是可复现的研究阶段交付，不是达标模型发布。

### 本轮全部运行闭合与已执行整理

三个辅助试验均完成80轮/15120step，各606次异步eval已完成统一选优：additive best14650为50/378，pooled best13750为48/378，per-ROI best13150为50/378，均低于对应基线82/378，不采用。辅助进程全部退出后才删除共享缓存。

实际归档408项（336个历史输出目录、32个旧脚本、1个历史模型目录、39个旧配置），路径统一为archive/legacy；删除2038个非best快照/临时硬链接/关闭的共享缓存/已合并分片。逻辑unlink字节包含硬链接重复，不当作真实磁盘释放量。原始data、冻结Blender、现有环境未修改。历史各run只保留best与必要依赖，canonical models/A0与models/B0各保留一个best。保留的现行源码/配置所需历史依赖没有为了减少目录数而强行删除。

清理后真实第0帧输出与完整视频记录完全一致；独立加载A0/B0和1600/200/200数据加载通过。记录POST_CLEANUP_CHECK.json、SEALED_MODELS.json及cleanup_journal.jsonl。归档清理完成不改变真实验收失败结论。

发布前再次完成自包含包与原视频第0帧一致性验证。wheel实际构建成功，修正model/heads未成为常规包导致的遗漏；许可证与provenance随包。8个历史CRLF源码/测试用.gitattributes逐项-text保留原字节，防止Git换行转换破坏冻结SHA。未为消除无功能影响的历史空行告警而修改冻结源码。未将wheel内容验证冒称全新机器安装运行验证。

action4的30分钟自动监控已删除：所有相关训练/异步eval已经结束，没有继续空轮询的价值。Git发布身份以outputs/action4_release_preparation_v001/PUBLISHED.json及远端main为准；科学验收仍为失败。

## 2026-10-23 冻结模型后的 B0 排序修复与 A0 误差分解

本轮不重新训练，不修改已封存权重。对同一B0step2200拆分候选框/修正框与center/product/learned三种评分，共6种固定对照。保留修正框、使用center评分时，开发集双机171/184（92.93%）、mAP50:95=0.529753、框F1=0.973333；A0原PnP八点221/378、高遮挡45/164。相比learned评分的开发集177/184、真实6参考2/6，center在6参考达到5/6。质量分数的真实排序已被定位为明显问题，框修正的收益仍保留。模块为eval/action4_b0_scoring.py，策略只由显式配置b0_scoring_policy选择；原默认推理第0帧逐字段等价检查通过，封存配置不被覆盖。

已完成center策略全2850帧视频及帧/时间轴验证。无效位姿由828降至332，涉及帧由771降至319。这是输出有效性统计，不是精度率。主控另检查12个此前六帧之外的帧，看到多处相机检出恢复；1427/2473仍有错误目标，593/1163/1963/2639仍有角点/尺度偏差，尚不达标。

### 扩展到完整视频的demo参考对照

使用与原六帧相同的白色线框差分阈值，独立于任何模型预测提取2850帧demo包络。2773帧满足“两组件且背景对齐误差小”的参考条件，77帧无法可靠评分，全部保留并报告。与原人工查看过的六帧相比，一对一几何匹配后的bbox边界差均≤1px；初次2700帧位置顺序检查把左右组件配反，已用匹配核对更正，提取像素和模型预测没有修改。

在同一2773帧上，IoU≥0.5双机参考包络同时匹配：旧B0step5220为1367/2773（49.30%），新权重learned排序732/2773（26.40%），新权重center排序1862/2773（67.15%）。阈值0.45/0.55下新center仍优于两者，改善不是单个阈值偶然翻转。该指标是demo线框参考一致性，**不是认证真实召回率、不是mAP GT，也不能作为训练标签**。仍未达到用户要求的可靠双机检出目标。

### AP接口纠正

整图A0流程此前先用0.3筛掉B0框，再传给AP计算，会截断低分候选。现将工作阈值框与完整AP候选分开：AP使用Soft-NMS后阈值0.001、最多2框；工作P/R/F1及角点仍用0.3。回归用例验证低分真阳性保留AP且不进入工作检出率，并检查过滤后的索引一致。B0训练异步eval原来已保留全部候选，因此best选择不受影响。对**原封存learned策略**重新核算历史test，mAP50:95仍为0.5614511313，工作计数也完全一致；未拿该test选择新的center策略。

### A0证据与被拒绝的改动

实验性固定旋转/机身尺寸、仅调整平移深度以贴合B0框的约束，使八点221降至93，高遮挡45降至9，仅恢复1个、损失129个。已拒绝并归档代码，不接入默认推理。这证明不能把IoU0.5层面的框当作精准角点几何真值。

随后仅作oracle诊断，在已保存预测后替换GT旋转或GT平移：359个有效匹配目标中，原八点221；仅替换旋转变283，仅替换平移变265。失败组旋转误差中位数9.70度、平移误差41.16mm；其PnP重投影中位数仍只有0.627px，说明对自己预测的表面重投影较小不能证明真实姿态正确。**oracle数字不属于可部署精度或实际改进成绩**。旋转及平移均须继续改进，不能依靠简单框约束补救。

证据：outputs/action4_b0_frozen_factor_review_v001、action4_b0_center_scoring_candidate_v001、action4_center_scoring_video_v001、action4_full_demo_reference_v001、action4_frozen_test_ap_correction_v001、action4_pose_error_decomposition_v001。失败约束位于archive/legacy/postrelease_box_anchor_v001。A0/B0权重保持原封存身份，任务仍未完成。

### 2026-10-23 摄影增强续训前验证

新增摄影增强在8张真实训练RGB上实际改变模型输入（归一化RGB平均绝对差0.29216），原始图像、padding与全部标签记录保持不变；实际反向传播和参数更新通过，峰值显存13493MiB。四rank样本覆盖、persistent epoch更新、断点后缀与seed复现验证通过。这些是工程证据，没有新的精度结论。

同时复现并修复best硬链接别名的临时文件残留：同inode两个名字的POSIX rename可能无操作，旧helper重复4次残留3个临时硬链接。新代码先判断samefile并用finally清理；重复别名与非获胜eval均不残留temp，已有故障重放6项回归通过。第一次旧回归因历史固定fixture目录已存在退出，原fixture保留，新增显式测试目录后成功；未跳过回归。

新正式配置configs/model/action4_b0_photometric_v001.json；输出outputs/action4_b0_photometric_v001。训练效果待后续实际完成。

A0已完成3024train/378eval的预测表面输入缓存。2个实际ROI的重构前向最大差0，几何损失等价；XYZ/相机射线/区域/掩码输入及R/t头均有实际梯度。16个实际ROI的优化器更新、保存加载后下一更新逐参数完全相同。上述是数据流与恢复工程检查，不是新准确率；正式80轮效果仍待训练评估。仅pose分支训练，冻结表面头不承担新梯度，边界明确。


摄影增强与冻结表面姿态两项均已进入实际训练。截至2026-10-23T06:35:13.022923+08:00：B0 step2214、第45轮，完成110次eval；从全部worker报告核对的临时最佳step1700，开发mAP50:95=0.613484，双机179/184。初始化同协议mAP=.529753、双机171/184。A0 step11711、第62轮，best8625为196/378、高遮挡36/164；本试验初始化166/378、27/164。均为训练中间结果，最终还需全量eval闭合及真实视频验证。

B0原GPU1评估落后，已增GPU6/7各一独占claim辅助worker，原训练源码未动；辅助只写done、不并发写BEST，最终ALL_EVAL_SELECTION必须核对201次评估。原GPU1初始两供数worker已实际退出。B0稳态compute67+A0compute3=70，控制器/监控预算2，总72；30分钟监控action4已更新。新增推理导出工具没有改动运行绑定源码，两个配置的source_identity逐项匹配。

完成阶段导出门禁与A0组合包接口：训练未完成时禁止正式导出；用一次实际中间pose权重作临时组合检查，冻结surface11300/原backbone6300与pose头完整键形一致、冷加载成功，cached-feature到RGB-head预测差2.67e-5px，临时权重已删除。此检查没有重跑raw RGB backbone，也不是最终准确率；最终选优后仍须完整RGB/视频检查。当前新训练代码及结果尚未作为完成阶段推送。


## 2026-10-23T07:49:42.983420+08:00 本轮80轮闭合与实际链路结论

B0摄影增强完成80轮/4000step，201次评估全部成功并由ALL_EVAL_SELECTION统一核对，最终best3220（不是中间最高双机率的快照）。开发集mAP50:95=.6162104443、框F1=.9867374005、双机178/184；历史test mAP50:95=.6493924639、F1=.9937264743、双机196/200。同一保留A0PnP组合的八点开发221/378、高遮挡45/164；历史test253/400、高遮挡36/142。历史test仍非新盲测。

完整2850帧新视频已完成时间轴/模型身份核查。全片5695个检测，303个位姿无效，涉及291帧。独立demo包络的同2773帧、IoU0.5双机参考一致性2268/2773=81.79%，比上一center策略67.15%提高；0.45/0.55阈值为84.31%/77.21%。77帧参考不可判定，未删除失败。12帧实际视觉复核仍看到1699漏检/误检、2473巨大错误cuboid、1163/1963无效或偏移角点，因此**真实任务仍未达标**。证据outputs/action4_photometric_validation_v001。

A0冻结surface试验完成80轮/15120step、606eval，best8625在GTROI为196/378、heavy36/164；完整RGB预测ROI评估仅42/378、heavy4/164，拒绝替换保留A0。已导出完整研究候选验证链路，裸pose-only权重没有误用。评估profile此前含被bundle分支忽略的旧pose_method字段，实际REPORT身份为network；完成后保存原profile并去掉无效旧字段，预测未改。

### 明确的A0训练/推理裁剪差异

3024训练目标都经精确GT框裁剪，真实机身最大边在ROI中的比例固定44.4444%。365个匹配的B0预测ROI中，该比例1%–99%为35.43%–61.97%；283/365的真实中心落在训练中心1%–99%范围之外。训练GT框中心/尺度先验过强，不能把GTROI指标当作实际锚定。证据outputs/action4_training_domain_audit_v001/CROP_CONTRACT.json。

已复用原A0的center±15%、scale0.65–1.35裁剪扰动，实现真实RGB输入、K矩阵、八角点同步变换。实际2ROI投影一致误差5.72e-5px；R/t/visibility不变，身份不重排；与生产prepare_rois完全一致。新RGB工作集每epoch改变裁剪，冻结backbone/surface且只更新pose头，实际前向等价与更新验证通过。这个在线裁剪训练尚未启动，不能声明已改善精度。

新固定B0best3220的预测ROI验证输入已从实际RGB生成：GT仅供目标关联/计分；后续须把B0漏检/FP保留在联合分母里，不再只用GTROI选优。准备代码和输入位于src/data/action4_pose_rgb_batch.py、action4_online_pose_data.py、action4_prepare_predicted_pose_maps.py、src/model/action4_online_pose_rgb.py及outputs/action4_a0_roi_contract_v001。

两个已结束运行的非best快照/last及重复视频分片已按清单清理，保留best及同inode不可变别名、完整预测/视频和后续需用输入。旧封存父模型未覆盖。当前研究选择见models/CURRENT_RESEARCH.json；新B0候选仅是当前较优研究模型，不是最终验收通过模型。

在线RGB修复检查已完成：4个实际ROI的CPU/GPU裁剪最大归一化RGB差0.000371933，实际反向传播、pose参数更新、冻结参数无grad、backbone buffers不变均通过，峰值显存约517MiB。分布式visible/hidden全局归一化参考loss=.352738291、rank合并=.352738261、梯度均=.773963153；错误rank平均=.348933756。初始测试反例扰动太小不足以超过测试检出阈值，保留失败日志后增大人工反例位移，正确实现和错误实现被明确区分，未放宽断言。联合指标测试确认漏检和FP未从分母剔除。

新配置configs/model/action4_a0_online_rgb_v001.json，输出outputs/action4_a0_online_rgb_v001；这些验证仅为工程证据，正式训练效果尚待实际评估。

在线RGB训练已确认四rank实际更新。截至2026-10-23T08:24:46.661354+08:00，step6200、第33轮，已完成248次预测ROI评估，临时best5425为145/378、高遮挡12/164；本run step0为72/378、1/164。无eval失败。所有数字仍是合成预测ROI中间结果，未作真实通过判断。

新增完成后衔接控制器outputs/action4_a0_online_rgb_validation_v001：等待80轮、606评估及真实worker退出后导出完整包，再运行全开发/历史test/2850帧视频；默认只生成待主控审核结果，不自动接受、不覆盖封存。另对B0摄影增强后权重重新做固定6种评分方式开发对照，先由开发mAP/F1选方法，暂未查看该新候选真实结果作选优。运行源与配置保持冻结。


## 2026-10-23 在线 RGB 裁剪训练、实际视频与阶段封存

状态：本轮训练和完整评估结束，真实任务仍未通过；未因产物完整而宣布精度达标。A0在线RGB训练完整80轮、15120个optimizer step、606次异步评估，按预先确定的预测ROI缓存协议选择step14000。模型冻结后执行开发200图、历史test200图及完整2850帧原视频，未用test选择checkpoint。

| 同一B0 photometric best3220 | 开发集全八点/378 | 开发高遮挡/164 | 历史test全八点/400 | test高遮挡/142 |
|---|---:|---:|---:|---:|
| 保留 A0 step13350 + PnP | 221 | 45 | 253 | 36 |
| 在线RGB A0 step14000 + network | 171 | 20 | 201 | 21 |

全八点使用固定canonical ID、每点误差≤完整GT框对角线5%，漏检和无效姿态计失败。B0未改变：开发P=.989362、R=.984127、F1=.986737、AP50=.979000、AP50:95=.616210、双机178/184；历史test P=.997481、R=.990000、F1=.993726、AP50=.990024、AP50:95=.649392、双机196/200。测试是已知历史test，不能冒称新盲测。

缓存选优173/378、重跑完整RGB171/378存在两项边界翻转：s209005_0000012_a02/ID2从.0498767到.0500509；s209005_0001552_a01/ID2从.0497654到.0500290。372个匹配ROI身份一致；逐点归一化误差差异P95=.000723885、最大.0108773。配对诊断确认CPU/GPU裁剪和冻结网络数值路径都有贡献，单目标与双目标批次也不能声称逐位相同；尚未完全分离各项贡献。严格执行原.05阈值，最终整链路成绩报告171，不改阈值、不改选优结果。未来应统一缓存和部署的裁剪设备、批次形状和特征计算路径，再做新实验；不把本次缓存当部署精确等价。

完整视频1900×750、2850帧、2997/100fps、timebase1/11988、1140000ticks已核实。相同2773帧demo参考上，两机检测框一致性仍2268/2773（81.79%），77帧参考不可评分；两机八点投影包络一致性从保留PnP的273/2773提升到在线network的1356/2773（48.90%）。后者仅是包络IoU≥.5代理指标，不能证明角点ID、旋转或真实召回率；所有漏检/无效姿态仍保留为失败。新视频数值无效姿态0，不代表定位正确。

主控实际查看73、307、593、877、1163、1427、1699、1963、2237、2473、2639、2807帧。2473巨大错误立方体明显改善；1163、1963缺失或偏移改善；1699仍漏真机并选背景；73、877、2237、2807仍有朝向/深度/角点偏差。因此保留合成集更优的原PnP模型，同时独立封存真实视频改善的网络候选，不宣称任何一版全面达标。视频及ROOT_VISUAL_REVIEW位于outputs/action4_a0_online_rgb_validation_v001。

B0六策略复核：refined_learned在开发AP50:95=.636888优于center=.616210，但预先确定的真实60帧对照center48/60、learned46/60，保持center，不因合成分数提高自动替换。完整框误差分解另保留，不能证明所有失败均由排序造成。

场景反事实仅改变环境：桌面深度从原.621m移至1.671m导致尺度缩小；独立矩形cloth mesh造成悬浮遮挡。恢复桌面/衣物原矩阵并隐藏独立布片后实际渲染改善比例，actor顶点/矩阵不变、八点投影差.000287754px。但手臂出画连接仍未修复，删除布片还减少遮挡覆盖；未做新场景碰撞验收，不发布这批候选、不用于训练。保持原2000数据和冻结资产不变。

代码继续按src/data、src/model/training、src/eval分责。新增在线RGB训练/DDP全局可见隐藏损失归一、固定预测ROI异步评估、完整模型导出与自动验证、只读demo包络比较；工程测试与实际模型分数分开。独立包的旧selection_scope文案误称GTROI，已写旁证澄清，未来导出文案修正，冻结权重不改。

后续最有证据支持的工作：先统一CPU缓存/部署计算契约；针对B0真机与支架/衣物混淆、场景尺度与连接缺陷改进数据；A0保持可见几何→位姿→固定八点链，检查旋转/深度与ROI分布，不仅增加轮数。当前用户要求的阶段封存整理不等同科学目标完成。

本轮消费者终态确认后，归档46文件到archive/legacy/release_r067，删除651个非best权重、last、过期claim和已合并视频分片文件；逻辑unlink字节1,366,770,175包含硬链接，不报告为真实释放空间。保留best与同inode step14000别名、606份评估报告、整段视频、原始预测和全部失败证据。首次清理预检发现旧photometric分片已经被上一轮清理，在任何删除前停止；缩小为本轮新产物后执行成功，未重复旧清理。

封存后5份当前/必要父模型严格加载、数值有限性检查通过；真实73/2473帧与已保存完整视频预测完全一致（角点最大差0px）。新参考包络回归和DDP归一回归通过，396份Python源码AST通过。这些工程证据不覆盖真实精度验收。当前所有训练/验证终态，action4监控已删除。


## 2026-10-23 r069：最后一轮评估契约修复与第二阶段准备

固定B0step3220、A0step14000，在同一RGB前向上比较三种姿态解码，未重训、未用test调参。完整200开发图：network171/378、heavy20/164；网络初始化iterative228/378、heavy47/164；同surface直接PnP218/378、heavy40/164。预先按时间分布选择60真实帧，两机框一致性三者同为48/60，两机投影包络一致性分别26/60、9/60、7/60。因此不能因合成分数上升而替换真实更稳的network；iterative本轮不采用。此60帧不是全视频验收，更不是认证角点GT。

实际GPU推理同时捕获预测XYZ/区域/mask与K/canonical，保留同一原图的batch形状，禁止合并不同原图的目标形成新的eval batch。全200图的PoseFromMaps重放与原网络角点差0，和已保存完整RGB baseline输出差0。新image_grouped_pose_eval完成实际异步队列回放step14000；随后完整核对372匹配目标的逐点归一化误差与判定，最大差0，all8=171、heavy20，376预测/378GT、4FP/6FN均保留。监督标签与模型inputs分开，GT只参与计分；一对一目标、匹配下标及分母均检查。旧CPU缓存173/378记录不删除，也不再标记为部署等价。新接口支持声明评估manifest，后续数据/冻结模块变化必须重建对应缓存；不能复用本轮2000数据缓存冒充第二阶段。

用户短暂提出红框必须在绿框内。隔离候选曾实现固定R/CAD、只调3D平移/深度的可行区间投影，通过100数学例和2真实帧工程预检；没有改权重或正式视频。用户随后撤回该方向，正式pipeline已精确恢复到发布HEAD；候选源码/配置/测试及预检证据统一归档到archive/legacy/rejected_box_constraint_v001，禁止自动采用。

本轮按用户要求规整文档：原plan顶部r018/r016等过时“当前”段落移入固定归档，保留原文完整快照；更新当前GPU、数据预算、发布许可，原方案与逐次记录不重写。conclusion区分当前摘要和历史时间线，ARCHITECTURE修正“只有GT ROI、无jitter”的旧描述；执行索引列明唯一入口与第二阶段前置事项。不启动第二阶段合成/训练，不添加新的持续轮询。
