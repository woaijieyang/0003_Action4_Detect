# Action4第二阶段识别与八角点增强：进行中结论

日期：2026-10-24，Asia/Singapore。计划：[完整八阶段执行稿](2026-10-24-1-plan-Action4第二阶段识别与八角点增强.md)。用户已批准主稿和五处定点修订，授权阶段一检查通过后1B→固定1B→1A。

## 已交付文档和代码

完整原稿保留并适配，八阶段、约19万字节；五处修订放入各自阶段。阶段一代码复用现有DDP/异步eval/封存框架，新增统一pose评分、整update有效分母、零监督/无监督独立头策略、不可变源码及数据根分离、正式6500准入和有序RGB预取。未改正式数据、CAD和models/default，未覆盖另一数据任务修改；未提交或推送Git。

## 实际工程验证

- 6500张全部实际解码；train/eval/test=5200/650/650，实例10400/1300/1300，ignore/unobservable为0。receipt：outputs/action4_phase2_stage1_v001/data。
- 8项针对性测试通过；12项既有epoch partition检查通过。见UNIT_CHECKS.json、unit.log、partition.log。
- 真实双rank和四rank NCCL/AdamW更新对照通过，最大梯度绝对差2.98e-8；建立动量后零监督不改变参数/optimizer/scheduler。见DDP_UPDATE_CHECK*.json。
- 真实默认A0、4个跨数据批次实例、两次更新：surface/pose均有梯度，冻结参数及buffers不变，严格重载输出一致，实际模型微批相对梯度误差4.69e-8和4.30e-6。见A0_ACTUAL_UPDATE_CHECK.json。这不是精度通过。
- 实际修改本任务拥有的工作树测试模块/配置后，延迟进程仍读取冻结版本；data root仍在授权项目根。测试后工作文件恢复。见SNAPSHOT_ISOLATION_CHECK.json。
- 有序预取与串行读取，在B0 step0/20/40/60的全部650图预测逐项完全相同。见PREFETCH_EQUIVALENCE.json。

## 保留的失败与修复

1. 首次A0逐位重载比较未启用完整确定性设置，最大差1.53e-5；随后实际微批验证仍发现3.48e-5相对梯度差。定位为FP32几何头批量形状引起的算术差异（backbone差0；FP64角点差8.88e-14），统一instancewise_head_v1后按原容差通过，没有放宽阈值。
2. p2_b0_integration_v001在启动前因GPU2/3被其他任务占用而被资源门阻止；无训练worker启动。保留日志，改用空闲GPU4–7，没有停止其他工作。
3. p2_a0_integration_v002第2批发生received0items of ancdata；进程FD软限1024不足以承载多ROI预取。仅提高训练进程软限到16384；p2_a0_integration_v003已完成60实际更新和受控恢复。

## 小规模实际训练与全量开发eval

B0 integration v002/v003：60次实际更新，step20受控停止并精确恢复，step0/20/40/60均650图完整eval。mAP50:95由0.618260到0.739287，框F1由0.973826到0.993072；这是小规模开发对照，不是80轮正式结论，也不代表真实视频达标。预取版完整eval耗时77.4–87.1秒。

A0 integration v003：60次实际更新、step25受控恢复，step0/25/50/60均完整650图；All8@2%分别145/107/170/170（GT1300）。最终best仍须遵守发布precision和各困难组约束，不能仅按成功数抬高忽略退化。eval每次约303–312秒，当前同卡双eval服务能力不足以直接按该配置放行正式1A；须在固定正式1B之后完成更合理的eval卡分配/有序预取吞吐检查。

B0正在以不变global32/micro8验证四rank容量（40updates、3次完整eval），用于充分利用空闲GPU4–7并缩短正式80轮。正式1B必须以该容量回执通过及实际资源检查为前提，不把工程run的checkpoint当成正式parent。

## 当前边界

代码通过：以上项目已证实，完整阶段一准入仍由集成/吞吐回执决定。训练完成：只完成明确标为integration的有限更新；正式80轮尚未在本次记录时启动。效果改善：B0小规模开发对照有改善，A0分组约束和真实效果未通过最终验收。阶段六/七仅在文档中闭合，未伪装已实现。

真实相机标定未知，配置明确assumed pinhole/null distortion；边界清单标MISSING_SOURCE，不编造空图或真实标签。最终test、外部模型、视频跟踪、完整2850帧增强均未运行。

## 正式1B已启动（2026-10-24，Asia/Singapore）

三rank、GPU4/5/6训练，GPU0/7共4个eval worker，CPU预算43/72。40步真实容量检查和受控恢复完成，评估服务按每物理卡一个流的保守估计/训练compute-only产生率=1.2324，通过1.2门。

正式run=p2_b0_formal80_v001，80轮/5200图，无跳过时13040更新与653次eval。已观察实际optimizer step 68，不是仅提交PID。配置与源码由runtime/action4_runs/p2_b0_formal80_v001/RUN_SPEC.json固定；当前权威configs/execution/action4_execution_r089.json。默认权重和正式数据保持不变。

A060步集成虽有All8@2%=170/1300候选，但小目标、大小不等及距离组退化，准入best仍为step0的145/1300；不宣称A0增强通过。正式1A等待1B完整选优后固定B0，并重新完成适合A0的eval资源吞吐门。

最后交付核验：{"train": {"step": 206, "epoch": 2, "cursor": 1376, "no_supervision_update_count": 0}, "best_step": 160, "mAP50_95": 0.8117465112731468, "F1": 0.9976923076923077, "eval_complete": 9, "eval_failures": 0, "overall_keys": ["images", "tp", "fp", "fn", "precision", "recall", "f1", "dual_images", "both_detected_images", "dual_both_detection_rate", "duplicate_predictions", "duplicate_rate_per_prediction", "images_with_duplicates", "targetless_images", "targetless_false_positive_images", "targetless_image_false_positive_rate"], "report_keys": ["schema", "scope", "operating_score_threshold", "matching", "AP_input_contract", "overall", "AP_by_IoU", "mAP50", "mAP50_95", "per_scene", "per_image", "per_target_stratum", "stratum_note", "scoring_policy", "AP_candidate_policy", "evaluation_wall_seconds", "evaluation_images"]}

正式6500三manifest与默认A0/B0权重SHA再次核对保持不变，证据DELIVERY_BOUNDARY_CHECK.json。已建立本线程每30分钟自动检查（action4），继续1B并在完整选优与A0吞吐准入后推进1A；不运行最终test或整片视频。


### 并行准备：主控第一轮复核（2026-10-24）

用户明确授权两个 Astra 子智能体，分别负责阶段二冻结 YOLO 参考/评分接口，以及阶段六、七稀疏跟踪/几何合同 CPU 工具。原正式 1B 使用不可变源码快照继续执行；准备模块尚未接入该 run，不修改已发布数据或默认模型。CPU 预算主任务43、两项准备各4，合计上限51/72；并行并不以抢占其他任务或重复训练为代价。

主控实际阅读了两项代码。YOLO 初版将识别与 support 质量混算、空可见 mask 的识别分母处理不充分，已退回定点修正。单张预览实际查看发现银白色/过曝风险，不能凭工程渲染成功宣称参考库视觉通过。跟踪初版存在跨实例材料合并、缺失响应误作遮挡正确、共同域缺少完整候选身份的问题；第一轮修复后40项CPU测试通过，主控继续要求排除同一材料不同generation同时入拟合，并将窗口投影差固定于预登记材料集合，避免动态交集缩小评分域。

这些是提前实现与审查，不是阶段二99%达标、阶段六跟踪精度或阶段七几何改进的证据。主控复测与代码质量复核完成后另写回执；完整模型推理、solver接入和真实稀疏标注仍在各自阶段验收。

正式1B读取到optimizer step740、epoch字段5；已完成评估最佳step700，在650张双机合成开发图上mAP50=1.0、mAP50:95=0.8722152359、F1=1.0、两台同时检出650/650。无目标样本数0，对应误检率未测。该结果不得当作真实视频或独立test验收；80轮仍在进行。


阶段六/七准备代码经两轮主控审查修复后，主控独立复跑42项CPU检查全部通过（15 tracking、15 sparse scoring、12 pose diagnostics）。回执：outputs/action4_parallel_preparation_v001/TRACKING_ROOT_TESTS.json。确认同一材料不同generation不重复计点，预登记固定F缺成员时不缩小集合，缺响应不充当遮挡正确，heldout隔离和共同域分母有反例测试。此结论仅覆盖纯CPU合同模块；尚未接入实际CoTracker/PnP/LM，也没有真实轨迹或隐藏角点效果证据。主稿的loss权重从等价简写补成原批准的完整配置键，不改变数值或阶段结构。


### 并行准备闭环：79项主控独立CPU复测通过

YOLO准备模块最终37项CPU测试经主控独立重跑通过；跟踪/几何合同工具42项通过，共79项。先规格审查再质量审查，修复均保留原失败证据：识别与support联合指标分离；unknown/ignore不冒充负类，原发布数与可评分分母分别记录；复制mask按ID预删导致合法TP消失的问题已修正，新增重命名与复制双目标反例。源码身份/固定资产/输入通道/输出合法性已有接口检查，但真实官方模型加载、prompt存取等价性仍待隔离环境完成后验证。

主控查看30W与3W两张实际预览：3W下机身恢复深炭灰，说明过强照明是原银白偏差的贡献因素；没有据单视图批准六方向参考库。未修改冻结几何资产。完整48RGBA/144RGB库、模型推理和99%门不在本次CPU通过结论中。

| 准备范围 | 代码证据 | 尚需所属阶段完成 |
|---|---|---|
| 阶段二YOLO/reference/scorer | 37项主控CPU复测、官方资产身份、两张照度对照 | 隔离运行环境、实际模型与profile重载、六方向参考审查、99%真实评分 |
| 阶段六稀疏跟踪/heldout | 共同域、缺响应、跨实例/代际隔离等合同测试 | 实际CoTracker调用、真实稀疏标注、预测消费链 |
| 阶段七窗口诊断 | 固定F、heldout不参与、变时间间隔与接缝夹具 | 实际PnP/LM/窗口求解及接受/回退入口接入、真实效果验证 |

根复核回执在outputs/action4_parallel_preparation_v001/ROOT_REVIEW.json；实际复测见YOLO_ROOT_TESTS.json及TRACKING_ROOT_TESTS.json；仅这14个新文件的身份封存在ROOT_REVIEWED_FILES.json。正式1B继续原不可变快照，没有导入上述新模块。两个子任务本次准备范围已完成，后续不得把CPU准备通过冒充完整阶段或效果通过。

### 定时检查最新状态（覆盖更新）

2026-10-24 13:57 Asia/Singapore：正式1B step2434、第15轮进行中，121次完整650图eval完成，step2420已领取执行，无未领取积压、失败回执或非有限loss。最佳step2300，mAP50:95=0.9209960670（较上次0.9068870236提升0.0141090434），F1=1，两台同时检出650/650。训练与4个eval进程身份一致；80轮未完成，1A等待既定准入。仍仅为合成开发集成绩，不代表真实视频或最终test。完整状态见outputs/action4_phase2_stage1_v001/HEARTBEAT_LATEST.json。


### 2026-10-24 GPU0授权撤回与无训练重启迁移

用户要求停止使用GPU0。已核验原controller及两个GPU0 eval的PID/命令身份后，结束原监督进程，再结束两个指定eval；原三卡训练PID3044002保持运行，GPU7两个eval保留。未完成step2520 claim归档后重新入队，未删除请求或快照。新监督脚本scripts/action4_supervise_resource_handoff.py接管既有训练，并通过原冻结launch.py在GPU5/6各建立一个替代eval；总计4eval，CPU预算仍43/72。当前权威r091，实际映射以gpu0_retirement_v001/RESOURCE_DISPATCH.json为准；原冻结config保留原始身份，不覆盖其历史GPU配置。

GPU0上的本任务eval PID3044003/3044005已消失；其他任务PID不作处理。监督逻辑继续验证80轮遍历、每20step完整eval覆盖、统一rank选优并调用原best-only封存逻辑。新阶段1A明确禁用GPU0，仍需固定最终B0并通过A0吞吐门，每25step完整eval不变。


### 后续接口与原生零样本准备（2026-10-24）

GPU0撤回后替代eval已完成多次650图整评，实际耗时约88–93秒，训练PID保持不变；未重复训练、未丢step2520请求。后续新run资源校验新增6个针对案例通过。新增逐角色封存入口scripts/action4_seal_completed_role.py，核验完整训练/评估/控制器身份后，使用各自不可变源码中的loader严格加载并字节复制到models/candidates；7个新增metadata反例通过。当前真实1B尚未结束，--check-only返回WAITING_FOR_COMPLETION，未提前封存或改默认权重。

YOLO真实原生CPU接口已接受：官方视觉profile(1,1,512)、文字profile(1,7,512)保存重载预测等价，实际加载冻结/eval，4CPU；已核查真实NPZ产物。112个视觉raw候选、文字0候选只是接口结果，不算识别成绩。六方向低照度比较已由主控实际看图；相机相对3W主灯+0.6W补光保持深灰机身并恢复背面边框/右侧细节，批准进入1024候选参考构建，不等于正式参考库/99%通过。GPU1空闲后批准最多2个YOLO eval进程，各8GiB显存预算、4CPU，采用独立GPU runtime合同，不翻改CPU-only合同。新开发对照预登记固定6张eval、三种配置（单ref、固定六ref合并、文字），未碰test，不依单图GT挑ref；仍仅camera-proxy合成开发证据。

SAM阶段五状态合同25项主控CPU复测通过，修复非法响应被捕获后仍可发布、失效后published仍为真的漏洞。随后取得固定官方SAM3 commit，核实NumPy int对象ID及outputs=None等真实接口差异；原生响应桥接新增10项主控CPU检查通过。官方sam3.1_multiplex.pt直链实测HTTP401/GatedRepo，需要合法权重或授权认证；尚未加载SAM3.1真实模型，不用其他模型替代。

阶段三独立SAM→C5 adapter采用原稿结构，真实CPU两次更新证明投影/编码器按预期接收定位梯度，质量loss与RGB/adapter隔离，缺mask严格回退；bbox extent及未知GT NaN资格问题经定点修复，主控只重跑2个相关新测试通过。尚未接入现A0正式训练/eval/bundle，不把工程读出变化称为八点精度提升。三个子任务并发上限3，CPU总预留含主控峰值69/72，GPU0禁用；SAM权重等待期间不占GPU。


### 固定六图YOLO原生开发对照：保留负结果

GPU1上两个独立eval进程已完成42次原生预测；调用总推理时间20.829秒、单调用中位0.326秒，显存reserved峰值约3250/3086MiB。实际输入为预登记6张合成eval、12实例；单front参考和固定六参考组合在发布score0.25下均0输出，文字camera-proxy配置10个发布输出、识别6/12、严格joint0。主控读取DEV_RESULTS.json核对，未改阈值、未用GT逐图择ref、未碰test。图像外观改善和profile接口通过均不能冒充零样本识别性能。

下一项仅在同一已登记样本上核查官方直接visual提示与NPZ重载两条路径、参考自识别、RGB/BGR和letterbox映射，保留原负结果；确认接口是否有误后，再预登记下一轮参考方案。SAM3.1官方权重HTTP401/GatedRepo问题已告知用户，等待项目内合法权重/授权方式，不能据源码桥接35项CPU测试宣称已验证真实原生SAM。


### 视觉提示接口定点诊断

官方直接visual_prompts+refer_image与NPZ重载路径，在同一预登记p2_eval_00000均无输出，重新提取embedding与保存向量差0；参考1024→1280缩放gain1.25/pad0、BGR→RGB张量误差约2.98e-8。六ref向量均finite、L2norm1。该例没有发现存取、通道或参考bbox缩放差异。与此同时，六参考在自身图上的top1 mask机身覆盖仅约0.35–0.55，虽最高score可0.79仍多为局部，不算可靠整机自识别；不能仅以跨域解释失败。下一步先核对官方是否支持reference mask及参考尺度控制，再预登记有限对照。保持既定0.25发布阈值与原失败记录。


### SAM3.1资产阻塞已解除：ModelScope原PT

用户要求查其他公开来源后，已从ModelScope facebook/sam3.1固定revision616acbee0b9ed4177f1f389e3c13594a0a1f6398单份下载原sam3.1_multiplex.pt，落models/external/sam3。实际3,502,755,717bytes、145.38秒、平均24.09MB/s；一次完整SHA256=0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6，与ModelScope manifest及另一公开原PT分发一致。不是ComfyFP16重打包，发布元数据USER_UPLOAD/CreatedBy=systemd如实保留，不冒称已独立获得Meta gated原件认证。官方许可随资产保存。

weights_only=True/map_location=cpu/mmap读取实际根state_dict1623张量，tracker457/detector1166；源码定义的三个multiplex参数均[16,256]。仅有界有限性抽样，未声称全模型推理或全量数值验收。主控读取回执并核对文件stat后，批准隔离SAM环境与GPU1一进程/一session短原生测试准备。用户随后确认HF也已授权，既有ModelScope文件有效，不重复Chrome下载。

当前状态已从HF资产阻塞改为隔离环境/真实原生测试准备；不能继续引用旧BLOCKED状态。GPU0上仍有其他任务进程，已核验它不是本项目，未停止或修改其工作。


### 封存接口复核收口与零样本第二轮

封存工具追加逐request/done身份、唯一图像覆盖、原始报告、不可变rank重算、复制副本strict-load、parent/data/counterpart依赖闭合；pair改为各角色独立immutable loader进程，并强制A0所用B0 SHA/step/config等于配对封存B0。主控发现并修复sys.path误删项目venv依赖的P1，真实-I子进程验证torch/numpy来自受控venv而data/model/eval来自指定snapshot。主控4项针对性复测通过；真实当前run check-only仍WAITING_FOR_COMPLETION，尚未封存或覆盖默认模型。证据outputs/action4_role_sealing_review_v001/ROOT_REVIEW.json。

YOLO第二轮固定两个因素对照完成：mask75在官方单通道LetterBox产生5D shape错误后，用项目薄适配仅squeeze末尾单通道维度，原失败与独立attempt2保留；原生池化/网络未改。mask75发布TP0/12，bbox25发布4个、BoxTP3/12、Joint1/12，严格逐图/双机0。bbox25参考自身top1覆盖97.4%–99.2%，说明参考尺度对局部化有实质影响；未达到零样本性能门。72次查询总推理计时30.791秒、GPU1两worker峰值约2.77/3.00GiB，完成后释放。未增加第三配置或降低阈值。

SAM3.1隔离环境CPU官方builder import已通过，numpy1.26.4来自runtime/sam31_deps而主torch/vision只读复用；冻结B0_parent在dev820..824五帧各产生两个真实候选，822分数0.890705/0.809970。这些仅用于短原生工程测试，不是当前正式B0最佳模型的真实验收。GPU1完整构建/单PNG/5PNG结果仍待实际worker回执，不提前写通过。

### 非 DJI 误选与近邻实例检查：进行中

本轮只读复查了原六图YOLO诊断图和原始汇总。text_fixed7发布10个预测、完整框IoU诊断匹配6个GT（12个GT总数），联合支持成功0；该差额包含重复及定位问题，不能直接宣布4个非DJI误识别。bbox25候选仍只在六图有3/12完整框诊断TP、1/12联合TP，不是全量成绩。

已将全650开发集、全部GT、near_pair及遮挡组、非目标候选与unknown分离的要求纳入同一主稿。现有scorer v3把完整框IoU诊断同时作为99%联合门条件，与主稿允许宽框且按可见支持判断的规定不一致；正在定点修复并保留历史版本语义，不重写旧结果，不放松联合99%门。全650编排与SAM条件/端到端分母检查并行准备，尚未取得全量zero-shot成绩。正式B0训练保持原不可变源码，不受上述评估器修改影响。

### YOLO/SAM实际接入修复与真实输入消费（2026-10-24，进行中）

- 已确认真实822帧text_fixed7:16（camera分数0.678139）把双手捏住的牛仔衣料作为实例支持。主控查看放大图复核；这是具体视觉失败案例，不是独立GT或全量误检率。原预测和证据在outputs/action4_sam_session_preparation_v001/full_dev_review_v001。
- YOLO评分v4已完成定点修复并由主控复核7项新测试；99%联合门继续按可见实例支持/提示/严格双机/逐图，不把完整框IoU诊断混入，不改历史v3结果。新声明式文字接口4项主控测试通过，原生class ID/角色/语义/profile契约绑定，避免改text后仍使用固定0/1类别。已批准固定两配置13次查询，尚不等于全650结果或正式准入。
- SAM公共ID存在但mask全空时已改为target_visibility_unknown、available=0，不冒充有效目标或训练负例；session/response的42项CPU测试由子任务完成，真实非空输出未重算。
- 主控发现并修复StagedPipeline未支持新SAM增强bundle、未消费mask的问题。新增显式native预测→同RGB ROI→增强C5→surface/pose→canonical8路径。独立审查又复现过滤后位置ID领取另一实例mask的问题，已改为过滤前稳定候选身份；原错配反例现被拒绝，新增4项工程测试通过。
- 实际复用了真实822原生SAM gen1 mask（75,144像素）与冻结B0 parent预测，加载工程训练step3增强bundle执行两次CPU forward。右机available从0变1，8点ID0–7保持，原图投影变化0.275–1.059像素；左机仍缺mask，未伪造。证据outputs/action4_sam_pipeline_integration_v001/REAL_CONSUMPTION.json。该变化仅证明真实输入进入网络并影响输出，不能称精度改善：增强bundle仅工程检查、真实K未标定、没有GT精度评分。
- SAM增强同run恢复2项测试由主控重新核验通过。首次主控命令未显式屏蔽CUDA，CPU-only测试在setUpClass拒绝且0测试执行；补充CUDA_VISIBLE_DEVICES空值后2项通过。没有GPU训练或伪称首次通过。
- 全650标签审计为1300目标、0ignore，heavy160图、severe15图、small43图、near_pair10图（组可重叠）。无独立其他相机/非DJI语义负类、纯空场景标签。既有6图已暴露全部3场景家族，按家族隔离没有未暴露prompt_gate；完整650作为开发评估，不能冒称新独立验收。全量原生缓存正在准备，不修改正式split/test。

### 局部证据面积比修复与全量评分续接

原单侧匹配使用visible mask像素/完整候选框面积作为[0.5,2]尺度门，混淆遮挡与尺度。650标签几何敏感性分析中，1300实例的720例低于0.5，高遮挡192/192例低于0.5；这是GT几何敏感性，不是实际模型拒绝率。现按主稿补充定义区分bbox尺度比与visible_fill；真实mask覆盖、中心、唯一归属仍须通过，negative不硬删除。主控亲跑25/25工程测试通过，原失败与修复记录保留在outputs/action4_b0_yolo_guidance_preparation_v001/ratio_revision_v002。

全650 native任务已由root冻结runner/config并启动两GPU1 worker。完成控制器也已独立审查并启动，记录outputs/action4_full_zeroshot_eval_v001/COMPLETION_CONTROLLER_LAUNCH.json；仅在4550原生任务与650/1300覆盖齐全、无失败且worker退出后，使用同冻结runner完成CPU评分及汇总，不重启或重复原生推理。

新的真实机身外观参考已由主控实际查看并批准为开发候选，仅保持原像素、按预测mask排除背景并pad到25%占比。822及820–824作为enrollment，不再用于此候选查询成绩。bbox与mask两种官方提示方式正在CPU注册，未来完整650+56真实开发困难帧的对照尚未运行；无真实GT部分不产生伪准确率。新文字13次对照保留负结果：真实822衣料输出消失同时真右机也丢失，不予替换旧已用支持。

### 新增头实算复核、完整原生预测结束与真实参考对照启动

- 主控复核B0 guidance heads时发现新增框/质量损失原先也计算B0行，会与原refiner重复计权；现限制YOLO-only。无YOLO rank保留新头连图零，独立头更新决策使用全局有效计数。12项CPU测试由主控实际运行通过；两次AdamW工程更新与严格重载、真实B0共享feature消费证据在outputs/action4_b0_guidance_heads_preparation_v001。没有正式融合训练或精度收益结论。
- v001全650原生4550任务已全部完成且0失败，CPU controller已接续评分；完整分数尚待汇总。
- 真实参考bbox/mask两profile通过12项编排复核后，v002全650+56已启动。r095允许旧v001 native退出后由2增至4个GPU1查询worker，每worker CPU4/8GiB预算，四YOLO期间不运行SAM GPU。扩容记录明确是当前显存观察与4×8GiB保守预算，不声称取得未暴露的历史allocator峰值。GPU0保持禁止，正式B0运行源码/默认权重/6500数据不改。
- 新SAM全量诊断提示来自冻结1B step0既有650预测：原1300 post-NMS记录保留，.3阈值发布1298，2个漏检不补GT。producer本身max2，不能叫阶段二uncapped池；step0临时bundle已退役，无其字节SHA或重放声明，复用依parent初始化与冻结源码/配置/成功回执链。原生SAM full-dev实现准备中，尚未新GPU执行。

### 官方提示复核、全量结果与最新资源覆盖

已查Ultralytics官网、固定上游实现及THU-MIG官方示例，并实际查看真实后屏候选和crop-context参考。官方资料与项目建议区分保存：outputs/action4_stage2_preparation_v001/real822_reference_enrollment_v001/official_usage_audit_v001。

v001完整650评分已完成：text fixed7完整框诊断precision91.21%/recall54.31%，严格联合双机成功6.15%；六合成视觉参考对应96.42%/35.23%，严格联合双机8.15%。这些不是99通过，完整框诊断与支持联合指标未混称。v002真实参考两profile完成650评分及56未知真实诊断，.25下56帧均0发布；读取raw进一步发现bbox参考4/56有raw、最高.11078，mask参考38/56有raw、最高.23953。raw候选不等于正确目标，但0发布也不能说模型完全无响应。CPU有限阈值对照已启动（controller3284914），不新增GPU推理，不改原.25结果。

用户最新仅允许GPU4–7，GPU0–3禁止。已核实旧GPU1自有原生任务在新指令时均已结束，残留GPU1进程不属于本项目，未触碰。r096已原子生效，实时guard的8个禁用train/eval案例均拒绝；30min自动检查已同步。服务器96逻辑CPU/48物理核，以24–95作为最多72逻辑CPU池，优先高编号；已对本项目21个活动进程组的217个现有线程设置并核验affinity，无训练重启、无其他任务改动。CPU核验和资源guard证据见outputs/action4_resource_transition_r096。后续SAM候选仅GPU7、14GiB allocator、fresh24GiB及运行余量8GiB保护，尚未GPU启动。

### 模块作用链与视频里程碑更新（2026-10-24）

用户授权阶段一新A0/B0选优后先生成完整基线，后续稳定模块逐项同帧视频对照。已定点纳入阶段一与阶段八；训练期间不生成整片、不使用test选参，后续整片观察登记曝光。GDR-Net实际forward、SAM-Adapter、SAM-6D和CoTracker3官方代码的适用边界已补入对应阶段；MLP/adapter须有真实监督与strict加载路径，不自动微调冻结外部模型。

SAM r096_v005实际首批2任务完成，最终根模型load零missing/unexpected、2次session均关闭、峰值reserved6328MiB。主控实际查看FIRST_BATCH_VISUAL.jpg：目标对应正确，附件混入仍需量化，未宣称质量通过。已在GPU7/CPU47–48继续同身份1298任务，复用已完成2条，PID3304140；只用GPU4–7，不触碰其他进程。

YOLO真实参考有限阈值CPU开发已完成，.05/.10/.25平均joint F1分别0.09629/0.05299/0.00898，.05被开发规则选择；正式99门未过、不自动替换默认。该成绩为识别与支持联合指标，不是纯分类准确率；56真实未知GT帧不参与选阈值。

当前连接层代码检查不等于训练收益。阶段四SAM→角点FPN断链正在最小opt-in修复；完整SAM独立scorer与视频selection入口正在并行补齐。正式1B继续运行，1A仍等待1B封存与本阶段吞吐准入。

主控新增复核：SAM→角点FPN与视频基础身份合同共9项测试通过（8.697s）；两次adapter更新确实收到定位梯度，默认参数键、缺mask回退与冻结buffers保持。阶段四完整训练入口仍缺，不宣称效果改善。B0最新读取step9826/epoch61、490完整eval无failed，best step9500合成开发mAP50:95=0.9498853879，650双机场景同时检出650；全数据均双机，空图误检与真实视频泛化仍未测。

### 连接层数据消费与完整SAM原生运行收口（2026-10-24）

主控完成本轮34项针对测试：corner与原视频合同9、视频资源增量5、SAM独立评分14、provider接线6。provider真实train-unit→core与单图eval→strict bundle pipeline通过；检测/mask工程夹具明确标识，不是正式准确率。质量target来自同一实际ROI的IoU，available/probability不能替代。正式native-v2 train/eval缓存与RGB变体仍缺；当前full-dev raw不可直接作为训练缓存。

完整SAM原生1298/1298任务完成，650图/1300GT固定域保留2上游未提示目标；全部返回available不等于质量通过。已开始冻结源码CPU全量评分。首启快照遗漏data.__init__所需action4_adapter，失败发生于导入前，原日志保留；主控补入新source_snapshot_v002并实测模块导入后重启评分(PID3327179)，未重跑SAM推理。最终质量以scorer_v001/full_v001/RESULT.json为准。

视频原生入口13项测试通过，新增成对seal/hash/资源/缓存合同，当前renderer仍demo|model，连续SAM/CoTracker与baseline|enhanced面板仍待所属阶段实现；不冒称已生成新视频。

### SAM完整650评分与实例链反例（2026-10-24，已完成）

固定旧B0父模型step0提示的1298次SAM推理及CPU完整评分已结束。1300GT中1296达到可见机身coverage≥0.5；同时满足coverage≥0.5、target_fraction≥0.8、camera_purity≥0.95、其他目标leakage≤0.05的为615/1300。650双机图中两机均满足覆盖646图，均满足严格support152图。条件IoU均值0.7778189只对应上游box唯一匹配域1265提示；另33提示关联未测，2上游无提示保留。高遮挡/重度遮挡/近邻/小目标为图像分组，保留每图两GT：严格support分别80/320、5/30、14/20、44/86，不能当纯遮挡目标级指标。

主控实际看图并回溯原始日志：680份bound mask为目标加背景污染，1份跨目标污染；示例可见支架/附件被包含。跨目标p2_cov1000_E_eval_0016的第二候选旧B0 score=.077489未发布/未提示，冲突检查只有一个已发布候选；SAM唯一raw mask仍含第二机残片，probability=.93361。该证据说明丢候选和分离约束有缺口，但不证明修改某阈值必然修复。新B0封存后应在相同SAM配置下验证上游改进，旧结果保留。报告/图在outputs/action4_sam_full_dev_eval_v001/scorer_v001/full_v001/；这些不是新B0、真实视频、DJI类别特异性或八角点精度成绩。

provider文件验证表的3项主控增量测试通过：首次PNG/NPZ共2hash，LRU驱逐后未变文件重读新增0hash；修改/替换/哈希竞争、fork/pickle重新验证，实际RGB与ROI检查不缓存省略。本轮主控累计37项针对测试通过，正式增强训练仍等待真实cache与阶段准入，不覆盖默认权重。

### A0并行启动与10step评估（2026-10-24）

按用户最新要求，A0不再等待B0最终完成。已复制并strict加载当前B0 step9500为不可变counterpart；父A0仍step14000。首次复制校验误读bundle的step键导致检查退出，权重复制未损坏，读取实际selected_step后核验通过；未修改B0运行。

p2_a0_parallel_integration_v004已启动控制器3341891，60实际更新/每10step完整650eval，共7次；GPU7训练，GPU4/6 eval，CPU总预算61，snapshot1134文件。当前为新资源拓扑与吞吐准入，非80轮正式A0。默认权重/6500/已运行B0冻结源码不变。子智能体额度错误退出，未反复派发，未完成的导出与YOLO配置准备保留待root接续。

### 实际原生SAM缓存到模型消费（2026-10-24）

新增data/action4_sam_native_export.py，保持原生已绑定mask/obj/session/generation，按保存预测文件SHA+sample/grid映射pipeline的稳定candidate ID，不用ordinal/GT挑mask。B0-only源合同显式声明模式并核验原生完成/源码/prompt回执，不伪造不存在的3项YOLO SHA；旧九项依赖合同保持。此6项B0-only导出仅限旧父B0开发eval，provider明确拒绝作为train cache。

5项定点测试通过，包括实际4个mask逐像素相同、ID映射、改index拒绝、隐式缺YOLO身份拒绝、训练用途拒绝。实际两张图用真实defaultB0预测与原生SAM缓存进入工程A0，4/4实例available，路径native_mask→same_RGB_ROI→enhanced_C5→surface_pose→canonical8，无假检测器/合成mask；A0仍是工程step3，不称正式精度收益。全650导出已用冻结data源码、CPU36–37启动PID3346953，无新增推理，输出outputs/action4_sam_native_export_v001/full650。

### 并行A0评估吞吐与缓存导出完成（2026-10-24）

A0 integration已完成60实际更新，7次full650 eval仍在收尾；前3次耗时279.97/259.95/292.49s。每10step生产速率高于原2worker服务能力，因此在GPU7用相同固定step0和不可变源码启动2个容量测试worker（PID3349954/3349955，CPU28–31/32–35），不进入选优或原队列；它们与原GPU4/6 evaluator及B0共存，A0训练已结束，不能声称已完整测得80轮时的联合负载。服务能力及实际准确输出等价性需记录后再放行正式配置。

原生SAM full650缓存导出完成：650图、1298个唯一稳定实例ID，2上游未提示实例明确登记。ROOT_EXPORT_REVIEW通过；仅旧B0开发cache，不自动成为正式train cache。candidate_overlap明确是SAM mask落在B0预测框内的像素比例，不是GT IoU/类别置信。

YOLO cropcontext_v003的两个CPU官方profile已完整登记，root准备了650合成+56真实未知标注图的同域配置，复用旧cohort而不重算GT。仅改变参考图tight-crop内背景，body25%/框/mask不变；native conf.05、CPU评分.05/.10/.25。新runtime只允许GPU4–7，旧GPU1硬限制已隔离在历史快照；AST检查表明原生预测计算除资源检查外不变。当前未启动新GPU查询，A0吞吐优先。

### 正式A0并行80轮已启动（2026-10-24）

integration_v004完成60更新、7个完整650eval，TRAIN/EVAL/CONTROLLER config与attempt一致，无failed。step0/10/20/30/40/50/60 All8@2%分别136/108/131/121/165/164/166；非0候选均未通过部署困难组约束，best仍step0，不宣布效果改善。额外GPU7容量评测256.46/267.11s，两次All8均136；按最慢已测292.49s、请求间隔中位90.68s，4worker容量比1.2401，达到预登记1.2。保留联合负载尚需正式运行观测的限制。

正式p2_a0_parallel80_eval10_v001控制器3357320已启动，80epoch/10400有效train实例，预算17360updates/1737full650eval，无train子集，新optimizer从A0step14000开始；固定中期B0step9500不随livebest变化。训练GPU7，evalGPU4/6/7/7，CPU69；源码来自已验证integration不可变snapshot，未使用之后开发中的SAM/exporter修改。阶段一执行门和launch位于outputs/action4_a0_parallel_admission_v001。后续最终B0如果改变，另建立固定配对复核，不能混淆原训练选优身份。

### B0融合模型所有权与严格checkpoint接线（2026-10-24）

新增GuidedB0 opt-in模型；现有construct/optimizer/bundle/load_bundle已支持显式增强schema，原B0detector/refiner strict初始化与共享appearance参数保留。一个综合CPU测试(7.450s)包含两次实际更新、完整optimizer覆盖去重、梯度到共享backbone/identity头、增强权重保存重载bitexact、缺guidance参数/错图像/旧消费者拒绝。输入为显式工程128RGB/candidate packet，非实际YOLO候选性能；未启动融合训练。当前正式A0/B0的不可变源码未改。

具体仍缺当前B0实时proposal与外部冻结证据的正式provider、guidance全局分母与原B0loss合并、部署候选阈值及guided async eval；正式run gate保持明确拒绝，不以组件成功代替全链。证据outputs/action4_b0_fusion_wiring_v001。

### 两个子智能体审查闭环（2026-10-24）

按用户最新要求同时最多2个子智能体。SAM缓存只读审查实际核对1300保存候选与冻结step0预测，0错配，1298唯一导出+2未提示，未重算mask或虚构正式train来源。B0当前候选审查发现singleton歧义提升、模型来源token未约束参数更新；root已修复并通过反例/真实更新续前向检查，二次发现pre_joint trace残留也已修复。规范审查通过后完成质量复审，无新增阻塞项；DDP/正式融合训练/完整部署仍未验收。

当前forward由同一次dense特征构造top100×2候选，不从旧B0缓存取候选，不先裁成2。所有保留候选联合关联，并显式区分父checkpoint来源与初始化+更新身份；错误输入拒绝，缺失外部证据仍可走RGB特征。正式候选loss与guided eval后续接入之前继续禁止融合run启动。

### 本地适配研究与实际关联加速（2026-10-24）

查阅官方CNOS、SAM-6D、Tip-Adapter、SAM-Adapter及YOLOE/SAM原生文档，定点补入阶段二参考证据候选和阶段三已有q_mask门的有限对照。明确RGB-D几何不能借用signed-distance D；Tip-Adapter官方test选epoch实现不照搬。外部权重继续冻结，新候选尚未训练，99门与最终指标不变。

真实current B0+保存native YOLO首图链路已运行：167B0假设+5YOLO-only、54条歧义、0唯一可用外部证据，原耗时102.2358秒，保留为负结果。修复每pair重复整图nonzero/面积/质心与交集统计，使用每轮mask统计及严格ceil切片。5项等价/歧义测试通过0.164s，包含20组完整pair诊断与305边界框；真实同输入再测3.3066秒，约30.9倍，trace/候选顺序/availability完全一致。改善速度，不宣称识别改善；候选假设组织的歧义仍需专项验证，不能单独放宽门来变造available。证据outputs/action4_local_evidence_performance_v001。

### YOLO可选采用与继续研究（2026-10-24用户更新）

用户允许YOLO多种方法后仍不达标时暂不接入，并立即补充“继续尝试，不是直接放弃”。已将该边界写入阶段二：99门不降低；YOLO未准入不阻塞必做SAM3.1/CoTracker3/八角点几何。继续cropcontext、参考视角/上下文、短文本原生对照；采用前真实来源与联动复核仍必要。B0-only缺失外部证据不是负类，旧step0开发cache不能冒充正式traincache。

独立reference evidence实验已经从研究进入运行：冻结B0step9500，train中三scene各64图共192，只用于建立参考候选；eval完整650，test不读。5标签/覆盖测试及独立规范/质量审查通过；CPU36–37提取PID3395800已真实产出features。固定reference cache、alpha0原始分数、labelshuffle负对照与逐image/scene完整分母评分代码已完成4测试并复审通过，尚未产生实际分类效果结果。该probe是项目侧可分辨性诊断，不是YOLO99或完整检测验收。

### 必做CoTracker3原生桥接已落实（2026-10-24）

官方指定commit源码已取得，HF权重经本机固定revision下载、核对官方LFS SHA后上传服务器；资产101890938字节，SHA2670d456...，strict加载0missing/0unexpected。复用现有Torch2.8，无新CUDA/PyTorch安装，无GPU操作。服务器直连失败已保留原因，不重复下载。

新增原生core适配器真实处理裁剪/resize/query/support/bidirectional/还原；独立审查发现query_role与PTS未进入snapshot、NaNPTS可通过，root修复并新增timebase/adapterprotocol/constructor绑定。7测试通过0.487s，真实原片首2帧新协议CPU双向forward5.129s，raw形状2×38×2/2×38/2×38。原始查询和强制写回分开，不作为材料观测。工程点不是DJI角点标注，未执行自动材料选择/CAD绑定/几何，未声称实际跟踪精度或120帧性能通过。证据outputs/action4_cotracker_native_preparation_v001/ROOT_REVIEW.json。

### Surface运输、暂定CAD绑定与参考对照收口（2026-10-25）

surface导出由同一次网络forward提供实际A/K/XYZ/logits/pose和完整模型身份，root6测试0.679s通过；实际p2_eval00000两实例default/opt-in pose完全相等，XYZ越[0,1]原样保留。新增坐标语义拒绝防projective/shear/负scale与非法bbox/K重算hash绕过；原NPZ未改。尚不是持久track/material确认。

材料mesh绑定独立审查修复距离预筛选漏掉歧义面、仅meshSHA不足的来源问题。7测试0.306s通过；实际冻结CAD构造100点验证8暂定/92歧义（按face顺序有偏，不是实际模型材料率）。成功仍provisional不进入几何接受。源、策略、区域和单位可复查，自动seed与时序确认仍未完成。

参考实验842图提取及完整650dev分类诊断结束：eligible52018、ignored62173；true reference F1=.9986769，原single-shuffle=.9967300，不能直接当纯语义增益或YOLO99。进一步机制检查发现原shuffle每类仍37/64同标签（57.8%相关）；8组精确每类32/32平衡打乱后F1=.387555–.394625，接近center基线.387555；全翻转在原选参数下F1=0。支持正确标签在该受限proposal域提供信息，但这是posthoc描述性机制诊断，非独立测试/显著性/完整检测AP或真实视频收益；正例含bboxIoU和纯单目标资格，不能当泛化身份标签。原结果与初步谨慎解释均保留，未晋升任何模型。

### YOLO继续尝试：参考背景对照已实际启动（2026-10-25）

cropcontext_v003已在新资源检查后启动单个GPU7 worker，CPU36–37，allocator7GiB/进程8GiB预算，启动free23625MiB、A0pending3，未更改其他进程或原训练源码。预期706图（650dev+56真实unknown）×2新profile=1412native任务，不重跑旧profile。首次核验已52任务、0failure，GPU7余量20333MiB。控制器3463187 CPU24在native完整退出后自动CPU36–37跑固定6policy评分与aggregate；总CPU预算72。当前仅在运行，尚无新完整性能结论，YOLO可选采用但继续研究。

### 已完成B0封存与新SAM训练输入准备（2026-10-25）

B0正式80轮完成13040次更新、653次完整650图评估；TRAIN/EVAL/CONTROLLER/BEST一致后通过既有sealer封存step9500，路径models/candidates/action4_improvement/stage1_B0_completed_20261024/SEALED_ROLE.json，bundle SHA ce6ba1a58e9f16608d4a44923c5e36914fc764f05b4ba8764f2e9c6c0e5f2ca3。开发集mAP50:95=.9498854、双机650/650；不等于真实视频/test达标。该SHA与当前A0固定counterpart相同，无需只为封存身份重复配对评估。models/default未改。

冻结该B0另生成全部5200train图的10400候选，完成回执outputs/action4_selected_b0_prediction_preparation_v001/run/COMPLETE.json；650eval复用step9500已保存预测，未重跑。只使用RGB推理，GT框/mask/pose不作为提示输入。

新SAM eval source采用显式completedB0 v2，区分旧step0 v1；1300任务正在GPU5/CPU94–95运行。独立scorer18项CPU测试及实际首批评分已过，控制器3519598等待原生完整退出后在CPU86–87进行全650评分，与旧输出按相同eligible GT配对；pending不从共同条件分母删除。mask available不是质量准入。

训练集输入也已封存到outputs/action4_sam_selectedB0_train_v001：5200图、10400原保存候选、10400合法提示，CPU prepare通过，lock=13a2356948e7f15468f62a59dc242d19025a1dc7bb70a27b7404aaaa85f9f112。source snapshot=e8f28f524038085ffed5593c809cefb5c476cf6cf6a85a045e63471dcd47891f。只抽取图像元数据及已保存预测，不包含GT监督。独立输入审查及真实首批尚待，原生train尚未启动。completedB0正式cache exporter/provider正在独立实现，不能把准备通过写成训练消费闭环。

资源权威更新为r097，GPU仍仅4–7、CPU仍24–95最多72逻辑CPU。已结束B0/YOLO任务不重启；原生SAM同一时间一个worker/session，不与训练集任务重叠。A0原不可变run继续80轮，每10实际step完整eval；本次观测1814updates、179done、0failed，best仍step0，无已准入提升。

### YOLO参考上下文对照完成与阶段五材料准备（2026-10-25）

cropcontext_v003完成706图×2=1412原生任务及六个预登记阈值策略的完整650开发集评分。bbox参考conf.05识别precision=.72358、recall=.20538；mask参考conf.05识别precision=.44814、recall=.71462，联合support recall=.30538，严格逐图成功率=.01846。mask conf.10识别precision=.60592、recall=.51923。仍未达到99联合门，没有采用，也没有将camera代理类别当作已验证DJI专属识别。56真实帧无独立GT，不能从此报告真实准确率。保留全部原生输出，后续不重复本轮查询。

provisional surface材料选择和真实video frame binding已补入；16项CPU测试通过。实际静态工程输入两实例在各512几何试验预算下产生19/0暂定点，分别还有486/207候选未访问，不能解释为第二台永远无有效材料。主要拒绝为display/glass、投影不一致和法线歧义。这仍是旧B0/工程SAM-A0静态诊断；正式视频三帧确认、持续身份、跟踪评分及几何准入尚未通过，禁止把暂定点直接当正确材料观测。

训练SAM独立规范审查后质量审查均通过：5200元数据、10400原始候选、211 snapshot文件，206代码/官方源文件与eval版本一致。首批控制器3526978在CPU84等待eval原生进程完整退出，随后仅运行两张不同图的真实首批，root复核前不启动全量train。

A0 step1800存在实际数值进步：All8@2%从136/1300到356/1300，合法匹配角点P95从95.23px降至35.37px，高遮挡0/192到5/192。但small从14/127降为13/127，depth>=1m从7/39降为3/39，未通过冻结困难组约束，best仍step0；未调低约束或替换默认。详见INTERIM_STEP1800_ANALYSIS.json。这是开发集阶段性结果，不是训练完成或正式效果验收。

新B0的SAM eval原生1300/1300完成，0上游未提示；完整CPU评分进程3533974已接续，不把available1300当质量达标。trainSAM首批2图真调用strict完整加载、0缺失/多余参数、session正确关闭；root查看原图/选定mask对应关系，通过的是输入工程检查。全10400训练提示原生任务PID3535731已在GPU5/CPU94–95启动，复用2首批，不重复推理；未启动SAM-A0正式训练。

completedB0 exporter/provider新增实现root13项CPU/旧cache回归通过4.515s，但独立规范审查发现3处需补：消费cache实际mask未与原生registration像素闭合、hash后parse/decode竞态、官方snapshot实际源文件核验遗漏。已返修，正式cache导出保持未准入；保留原失败审查与原生输入，不用工程tests通过掩盖审查缺陷。

完整650图新SAM评分已结束：固定1300GT的可见支持覆盖1300/1300（旧1296），严格support709/1300（旧615），+94目标。严格门是coverage>=.5、target_fraction>=.8、camera_purity>=.95且other_target_leakage<=.05，不是身份分类准确率。同一1265个双方合法且available的匹配域，mask IoU均值.777819→.804994，提升.027175；另35个新合法上游对象单独登记，不混入配对增益。高遮挡图组按原定义包含两目标：strict94/320；近邻组17/20；未声称所有真高遮挡目标都有提升。主要瓶颈仍是机身与配件/背景的分割污染，需要质量监督与RGB回退，不能强制mask作为真值。证据outputs/action4_sam_selectedB0_metrics_preparation_v001/full_v001/ROOT_ANALYSIS.json。

### completed-B0 cache证据链修复与实际全量消费（2026-10-25）

独立审查S1/S2/S3已闭合，保留原失败review：provider逐payload将实际bool像素与原生registration绑定，unprompted/unavailable不得伪带mask；JSON/RGB/NPZ的hash到parse/decode读前后fstat/pathstat一致，memo命中也校验；官方snapshot文件集合、SHA、只读和commit真实核查。root17项测试3.234s通过，随后SPEC_REVIEW_v002及QUALITY_REVIEW_v002均PASS。

使用7文件冻结source（4034ab...）实际完整导出650图/1300候选，474.819s；producer全payload原生绑定验证及provider初始化58.336s。真实首/末图的RGB和候选ID消费通过，改动一个RGB像素会拒绝。该验证没有重跑SAM/B0，没有使用GT挑mask；actual source modules均来自冻结目录，旧v1仍开发专用。证据outputs/action4_sam_selectedB0_cache_v001/ROOT_CONSUMPTION.json。数据导出和工程正确不等于质量或正式训练通过。

### 训练监督归属v2已实现、实际全5200框归属核查（2026-10-25）

新增data/action4_sam_training_associations.py与固定policy；绑定annotations/predictions/native cache/policy/source合同，逐nonignore GTunit完整行域。框IoU>=.5、发布score>=.3；候选多GT时不强配，同GT多个唯一候选按预测score与稳定ID排序，不按GT IoU或SAM质量试到成功。native available独立，缺mask或歧义不删除原A0unit；provider正式路径拒绝旧fixture association并核验实际record身份。

10项测试及独立规范/质量审查通过；root联合该模块与角点模块15项测试7.374s通过。读取实际5200train/10400保存预测作bbox-only全域诊断，10400非ignore单位均matched、ignore0；未读GT mask或SAM mask，不把它称正式关联文件。完整cache生成后仍必须按完整合同重新绑定产物。证据outputs/action4_sam_training_association_preparation_v001/ACTUAL_5200_BBOX_OWNERSHIP.json。

已启动冻结接续控制器3552948（CPU85，后续worker86–87）：等待10400原生train提示完成且进程退出，随后导出训练cache、生成v2association并验证provider。source identity ea17daf...共9文件（8源码+policy）；不会自动启动SAM-A0正式训练。50/30/20 ROI route、20%mask dropout/扰动、匹配光度变体与实际Stage2Units小训练仍是后续必要门。

### 阶段四头与224监督增量通过，整条训练链仍待接入（2026-10-25）

新增model/action4_corner_supervision.py，复用enhancedC5→P5/P4/P3/P2定位FPN；旧head默认返回与state keys保持。新增valid、像素b和detached quality MLP；GT与forward分离，合法隐藏点仍监督，已知错误/无证据/域外为分类负例，unknown/ignore/dummy跳过。coordinate以同网格GT完整框对角线归一；Laplace dx/dy/b保持224像素。quality标签合法且detach误差<=1%GTdiag，输入也detach，1-based epoch1–5权重0、第6起.2，不自动quality_ready。

独立SPEC→QUALITY均PASS，工程5测试含两次真实CPU AdamW更新、定位梯度、quality不回定位、独立无监督头不更新、strict完整head重载；追加known-negative NaN、unknown/inactive、分组sums/counts可加反例通过。root联合15测试再过。此处尚未接stage2core/trainer、完整bundle/loader、最终pose decoder、异步eval及整checkpoint准入；不宣称阶段四完成或八点效果提升。证据outputs/action4_corner_stage4_preparation_v001/HEAD_SUPERVISION_IMPLEMENTATION_REPORT.json及两份review。

受影响旧回归补查：provider wiring/file-validation的9测试通过；合并运行时corner-connection在模型检查前因PyTorch interop线程重复初始化报错，保留ROOT_AFFECTED_REGRESSION.log。按该旧测试原本独立进程约定重跑，1测试通过13.841s（ROOT_CONNECTION_REGRESSION_ISOLATED.log），未改测试或降低断言。已将完整650图SAM辅助分割成绩登记results/results.md，工程测试与进行中的训练不混入正式成绩。

### 阶段三mask增强实际消费与crop_specs统一坐标（2026-10-25）

opt-in sam_mask_augmentation path/SHA经Stage2Units→provider→ROI输入实际消费：20%dropout，剩余25%在ROI448上做半径1–3方形腐蚀/膨胀、每轴<=3平移或最多87像素小孔；不修改原生cache/RGB/GT/相机变换。预测mask增强后重新裁限图像支持并计算signed distance，quality loss按实际输入mask监督。20000固定seed的dropout4076/20000=20.38%，retained15924内perturb4022=25.257%。

独立审查先发现baseline/B0携带该配置被忽略，第一次guard又遗漏baseline+SAMTrue，均保留CHANGES_REQUIRED记录。最终Stage2Units/model config/construct/run gate拒无实际SAM/corner variant或flags不一致的配置，SPEC_REVIEW_v003与QUALITY_REVIEW_v003 PASS，9项augmentation tests通过。root联合模型与增强12测试36.584s通过；最终新增variant guard另1测试0.048s通过。真实eval缓存首实例的mask输入核查：identity quality target=.6272727，按预先选定seed膨胀后=.6009989，D改变；dropout available=0、quality count=0，RGB/A/K与原cache SHA不变。此处q=.5仅为执行真实loss函数的工程占位，无模型调用、参数更新或效果提升结论。证据outputs/action4_sam_mask_augmentation_preparation_v001/ACTUAL_MASK_CONSUMPTION.json。

新增TrainingCropSpec与prepare_pose_rgb/prepare_online_surface_batch可选crop_specs入口，绑定连续native边界及GT-unit、请求/实际route、seed/policy/reason；与旧boxes/seeds/jitter互斥，既有2.25 expansion只施加一次。5测试通过0.093s；初始padding测试错误期望归一化后为0，依据现有黑色padding+ImageNet归一合同改为-mean/std，原失败保留，无实现或断言弱化。独立spec/quality通过。

实际p2_train00000实例1与B0grid24398同框路径对照：RGB、masks、XYZ/regions和geometry逐tensor相同，1331surface点、投影误差1.208e-6px；完全出图crop保留unit但pose/surface与visible/hidden角点counts关闭，合法背景visible-mask监督仍保留。没有改data、调用GPU或新模型。该入口只是统一传输，50/30/20 route sampler与候选来源校验仍需实现，不当作采样策略已完成。

### 阶段四模型归属与严格candidate bundle（2026-10-25）

a0_corner_image/corner_enabled显式候选由JointSurfaceRGB拥有corner_head，同一次RGBbackbone提供C2/C3/C4与SAM增强C5；return_corner=True返回raw_pose/sam_aux/corners，旧调用明确拒绝，未替换正式pose。optimizer覆盖全部可训练参数且不重复，冻结backbone参数与BN buffers保持。candidate bundle保存完整backbone/surface/pose/SAM/所有角点分支及协议/父模型/counterpart；默认loader拒，只有显式allow_corner_candidate研究加载可strict恢复。

审查发现hash父文件后重新打开加载的竞态，现通过expected_sha256绑定同一resolved句柄hash→seek→torch.load并检查前后fstat/pathstat；合法同step不同权重原子替换反例拒绝，公共parent字段与corner_parent也交叉核验。MODEL_WIRING_SPEC_REVIEW_v002、QUALITY_v002通过，root完整CPU双更新/strict reload/freeze/default兼容测试随上述12测试再次通过。

仍未接现有trainer的corner sums/counts及epoch转换、完整data variant、单帧decoder、whole-checkpoint quality_ready与async eval，validate_run_gate因此明确拒绝全部corner调度。通过的是模型/保存接口工程验收，不能当成阶段四训练完成或精度提升。

### 阶段三ROI路由与真实训练计数消费（2026-10-25）

50%GT-jitter/30%prediction/20%extra-offset已通过Stage2Units→crop_specs→online surface→SAM provider接入；prediction内full/coarse各半请求，当前冻结来源只有full，coarse请求明确fallback full，不伪造coarse。Extra-offset起点±.35×GT框宽高、scale[.65,1]，仅integration，未校准不得正式训练。20000固定seed请求计数为GT10001/full3009/coarse2997/extra3993；不是实际训练统计。SamRoiRouter返回trace原曾暴露内部dict，深改trace可污染后续框；现deepcopy并以反例验证，SPEC/QUALITY_v002通过。另root发现provider返回association也共享内部对象，BEFORE实际复现污染，现取出时deepcopy，AFTER通过1.938s及独立RETURNED_OWNER_ALIAS_REVIEW通过。原native数据与冻结cache控制器未变。

Trainer实际消费固定11项route计数：wholeupdate/micro前一次int64 allreduce，active==global_n，dummy不计，无监督数据消费也累计；STATUS/train日志和last.pt/complete保存累计及epoch/cursor，恢复要求累计active==epoch*N+cursor，重放日志按logical_update_key去重。独立spec/quality与4项CPU测试通过，未启用路由时无新增collective。仍未运行正式stage3 sampler训练，完整train association等待原生10400完成。

### 阶段四loss接线、监督身份修复与真实Gloo对照（2026-10-25）

五项corner损失已进入既有loss_sums_on_batch；local_counts按配置显式corner flag，整update先统计allrank分母再分micro；epoch loop仅一次ep+1，1–5 quality权重0、第6起.2。实际owned模型两次CPU更新使corner/SAM projection/quality变化，base/BN保持。root身份修复后再次运行该真实模型loss测试通过15.192s。

独立审查发现corner_gt224无源身份，另一帧/GT metadata可被贴上当前身份监督；现packet绑定image/GTid，与外meta、FeatureIdentity和SAMowner/association交叉校验，预测ID与GT ID分开，unmatched显式None，缺字段拒。原CL-S1反例已关闭，TRAINING_LOSS_SPEC/QUALITY_v002通过。该修复没有改变GT、分母或训练算法。

新真实2进程CPU/Gloo工程对照已完成，使用实际corner target/loss/global denominator/AdamW/route函数及小型2参数预测器，非完整backbone DDP或GPU准确率。与单进程同globalbatch相比参数最大差0，optimizer状态最大差3.725e-9，objective最大差2.980e-8。4次data attempt中3次optimizer更新、1次全局无监督跳过，质量预热独立参数/动量保持，route累计12active，与checkpoint恢复一致。首版root fixture漏roi字段在梯度测量前失败，日志保留；修正完整ROI+身份fixture后v002通过。诊断float(Parameter)日志有PyTorch警告，未改变目标或反传，未为静音重跑。证据outputs/action4_corner_gloo_integration_v002/ROOT_REVIEW.json。

### 单帧刚体角点解码增量及跨身份拒绝（2026-10-25）

新增model/action4_single_frame_pose.py与固定policy：head→固定ID观测/有效概率/质量/b/多峰标志；逆A224到native，S=J b²I Jt+4I；至少6非共面ID建立SQPNP候选，与raw作双起点、右SE3共10LM，q-weightedHuber1.5+rawposeprior；检查10%重投影改善、全8深度>1e-4、full/history IoU或coarse支持、30度/.5CADdiag变化门。输出8点始终来自同一刚体pose，观测改变能够改变被接受pose。quality未全checkpoint准入时最终仍raw，diagnostic候选另存，不借逐图GT选择。

首轮审查发现helper丢image/instance/ROI身份，相同pixelspace下B观测能修改A的pose；现obs保留完整identity，decoder同时核raw/ROI/route/obs，原跨frame/instance/ROI反例均拒，8数值/身份tests通过，SPEC/QUALITY_v002通过。7x7外峰比.8及近5%cost多解判断只是未校准启发式，不认证多峰/全局唯一；零畸变未被证明，投影明确为注册针孔假设。该decoder尚未接完整async eval/quality_ready封存与正式部署，不宣称隐藏八点效果提升。

### 角点预测桥、整checkpoint校准与GT评分适配（2026-10-25）

StagedPipeline增加显式corner研究入口，完整candidate严格加载，实际一遍B0及一遍A0backbone产生raw、diagnostic candidate和紧凑八点观测；正式instances仍raw、quality_ready=False。原默认入口不接受未准入candidate。实际CPU工程模型测试验证head tensor被消费、missing-mask RGB回退及旧SAM raw角点等价，不是精度提升。审查发现公开context与provider map共享引用可重封任意mask注入；现公开对象隔离、验证meta和actualpixels，forward只消费owned context，原全1mask/GTorigin反例拒，CP-S1 spec/quality_v002通过。

纯calibration函数复用原All8主协议，完整650/GT/published域含FP和missing；known-invalid高q计负，unknown未测阻ready；100valid-selected、90%knownselected、10%eligibleGT六点覆盖与candidate>=raw全局一次选择。quality gate不代替原eligible/rank。审查发现P95、分组recall、invalid与005附加统计可伪造，现按原perGT errors/outcomes和GTmembership重算，root17tests通过2.137s、独立复审通过。原GT分组代码仅抽为gt_groups，使用既有完整650/1300输出验证前后report字典exact。

GT scoring adapter已实现冻结预测后才读GT、同框IoU归属、不按角点误差选GT、GTmaskSHA和224变换、生产source envelope/closure。review发现五处问题并返修：offline binding不冒执行证明；prediction-only decoder重放/物理投影；report-bound fullboxgate；scripts closure；纯metadata模型语义。root18tests通过3.183s，但复审又发现有效raw可被任意标False/None压低基线（S2b），当前仍在修复，未放行正式评分/async。

### B0当前细化框质量信号恢复与开发校准（2026-10-25）

源代码核实：BoxRefinement的quality目标是当前细化完整框IoU.detach（负例0），但center排序导出时丢弃了已计算的quality，只有旧dense quality保留。新增include_refined_quality=False可选导出，默认原字段不变，True时同次forward附refined_quality_score；不以center概率或old_quality_score冒充完整框可靠性。

root用原immutable B0算法只观察proposal logits，首次2图检查后续采集完整650图/1300候选；所有旧字段逐图EXACT，无B0重训、原box/score/ID改变。新q范围[.8917892,.9865182]，对匹配完整框IoU的MAE=.02068135。预登记阈值.5/.6/.7/.8/.9，要求precision>=.95、GTcoverage>=.90，按coverage/precision/低阈值择优；.5–.8完全等价，.9少2目标，按平局规则选.5。这是合成开发集operatingpoint，不证明.5优于其它等价值，未测真实/空场/nonDJI特异性。原始字段exact和全部统计经独立CPU复算核对。

冻结FULL_BOX_QUALITY_GATE.json（SHA e000e7db...）绑定真实B0/refined-quality协议/calibration_report/capture/prereg数据。Bridge新增validator消费完整链和实际SAM reference来源，不仅信status，actual gate正例与错依赖/计数反例通过；独立spec/quality已通过。该信号后续供decoder预测侧保护，不改当前B0发布阈值或默认权重。

### 同进程SAM payload proof复用性能（2026-10-25）

新增有界PID隔离proof memo，首次仍验证全payload；复用仍检查actualpath/stat/SHA及完整source/marker/result/期望字段，修改或重封异常照常拒绝，LRU16384只存摘要不存mask。23tests通过，独立6小测试及审查通过；既有full650实测首次125.861s/1300decode、同进程重复8.398s/0新增decode，约14.99倍。首次比先前58.336s更慢，不宣称冷启动或端到端14.99倍；只减少同字节同来源的重复准入。未更新正在等待的冻结train-cache controller源码，没有重跑推理或新增export。

训练SAM native10400/10400已完成（5200图，全部target_present只表示transport/binding，质量未测），原进程退出。冻结接续controller按约定启动CPU86–87 worker3690263，实际已有cache records写出；不重跑SAM、不改已冻结导出源码、不自动启动正式训练。

GT评分adapter最终S2b已闭合：按原FP32 camera/K112/uv112 clamp_min1e-4重算raw_valid，两flags和corners存在性必须一致；正z低于clamp仍尊重原raw合同，负z合法invalid，overflow/符号不确定拒绝。原有效pose伪False降级反例现拒，旧单图两实例cornersdiff0。SPEC/QUALITY_v003通过，root21测试再次通过5.096s。Adapter只验证source binding，execution_verified始终False；真实冻结producer/async完整运行与模型效果仍未验收。

Corner数据integration变体已接同一Stage2Units/ROI/provider/targets，root3新数据检查+7关联回归共10通过4.534s，正式调度门未打开，待独立复审。Async corner分支与无eligible终态代码已实现，root9测试0.497s通过；false-quality不建BEST，全部未准入时明确NO_DEPLOYABLE_CHECKPOINT并保留parentfallback。执行器/源码freeze及前置资源门仍在完成测试，尚未做实际完整corner eval或放行训练。


### 2026-10-25 角点评估联调与step3750真实开发诊断（进行中）

本轮未下载重复权重、未重启A0、未重跑SAM原生推理，未动正式数据/default。新执行器与async联合16项CPU合同测试通过（0.820s）；另新增实际scoring adapter→calibrator→selection→commit的两实例解析夹具，1项通过（0.389s），夹具即使All8全对仍不准入best。该证据没有真实模型forward，不是完整650图效果。实际当前src/scripts的490个Python文件按生产闭包规则检查，缺失项目import为0；不代替冻结运行验证。

主控复现并修复两处连接问题：corner worker原先在资源检查前调用CUDA allocator，现已前移检查；worker原先改写已封存REPORT耗时，现把worker计时放到result顶层并绑定原报告。独立审查又复现post-bind篡改内存report仍可进入commit的缺口；已在corner_result_signature统一核对实际report_artifact路径/SHA/同handle读回内容，覆盖提交、已有BEST及最终done检查。修复后14项相关测试通过（1.628s），包含新反例；独立复审尚待闭合。原失败与修复日志保留在outputs/action4_corner_async_preparation_v001。

首轮实际10400训练单元请求路由为GT jitter5236（50.35%）、full1564、coarse1539（预测合计29.84%）、extra2061（19.82%）；下一epoch有6884个实例改变请求类型。仅验证请求抽样，coarse未准入、有效路由与mask/RGB消费须在正式train cache完整后另查。依据：outputs/action4_sam_roi_routing_preparation_v001/ACTUAL_TRAIN_REQUESTED_DISTRIBUTION.json。

A0当前step3750的完整650图已保存预测：All8@2%=467/1300（初始136），匹配角点P95=29.8913px（初始95.2310）。小目标17/127（初始14）；高遮挡4/192（初始0）；>=1m只有3/39（初始7），故deployment_eligible=false且selected best仍step0。没有降低困难组准入门。当前训练仍第18轮，尚非80轮完成。

对两份已有预测做posthoc开发诊断，不新增模型推理。两阶段共同匹配的151个高遮挡GT、641个隐藏点，其归一化隐藏点中位误差19.6169%→4.3539%，P95 49.1272%→13.9358%；小目标高遮挡仍0/40。所有匹配失败中，放开八点ID进行Hungarian配对的诊断也没有新增2%全点成功例。这不能证明完全没有ID问题，但表明仅重排点序不足以修复当前误差；应继续推进可见mask输入/几何/角点定位增强，不能把工程通过当效果解决。完整匹配/缺失分母及逐实例证据：outputs/action4_a0_parallel_admission_v001/INTERIM_STEP3750_PAIRED_DIAGNOSTIC.json。


本节后续闭合：executor与async均已取得独立SPEC→QUALITY PASS；主控最终联合19项测试全部通过（0.735s），源码SHA与回执核对一致。原post-bind反例已独立复测为拒绝，未创建BEST/done。证据：outputs/action4_image_guided_eval_preparation_v001/ROOT_CLOSURE.json。该状态仅为上述代码合同闭合，不表示真实full650 corner推理、阶段三/四正式训练或效果准入；相应运行门保持未放行。另按原稿范围继续实现native RGB original/light/blur确定性变体合同，仅准备模块和小型像素检查，未批量生成、未混用旧mask、未修改正式数据。


### 2026-10-25 native RGB变体与真实SAM消费准备（进行中）

RGB generator和像素manifest两个增量均已完成独立SPEC→QUALITY复审。主控分别重跑8项（0.172s）和10项（0.406s）CPU tests通过，独立复测也通过。期间保留并修复两个实际反例：内部source symlink改变后receipt路径漂移；manifest自报图像集合缩小后重封装READY仍误过。当前实现分别固定首次resolved path，并在manifest/READY中强制绑定和读取预登记EXPECTED；旧失败报告保留。

最终修复版对固定真实训练图p2_train_00000、seed0实际生成并核验original/light/blur共3条记录，35.861s。original引用原文件；light/blur为无损PNG；shape均2464×3248×3，GT/几何不变，未调用模型或GPU。回执outputs/action4_rgb_variant_manifest_preparation_v001/ROOT_ACTUAL_PIXEL_PILOT_V002.json。旧actual_one_image_seed0缺expected_registration，保留且不准入，没有静默迁移。该pilot不证明变体SAM缓存接通或模型准确率改善。

完整原图SAM导出已完成：5200图/10400候选，实际导出耗时3840.710s，EXPORT_COMPLETE.json已出现。训练association v2也已写出10400行（ignore=0）；原worker3690263仍在继续最终provider校验，尚无总COMPLETE，未当作训练准入。未重启该健康进程或更改其冻结源码。

新增tests/test_action4_sam_real_consumption.py，已通过独立静态启动前审查，但尚未执行，不把skip算PASS。总COMPLETE后应在CPU39、空CUDA、单线程下显式ACTION4_REAL_SAM_CONSUMPTION=1启动：固定8个实际train单位做Stage2Units/GT/预测mask/ROI/计数核验，再固定首对做2次真实loss/backward/AdamW更新、冻结参数与buffers检查、mask消费差异及严格重载。首对无可用mask或无梯度应保留失败，不换样本；这仍是实际接口小检查，不是正式80轮或效果验收。启动前回执outputs/action4_sam_training_association_preparation_v001/REAL_CONSUMPTION_PRELAUNCH_REVIEW.json。

最近实测A0到3923update、第19轮，390次完整650eval、0失败，best仍step0；继续原冻结run。后续stage3/4训练、RGB变体原生模型链、视频及整体交付仍未完成。


### 2026-10-25 变体真实B0输入与供数性能诊断（进行中）

主控对固定p2_train_00000的original/light/blur，使用封存B0 step9500及其原不可变RunSpec源码完成3次实际CPU forward，17.199s。输入前逐文件/像素/shape核验，backbone pre-hook记录实际[1,3,768,1024]输入。三版本均返回2候选；light的真实模型输入摘要变化，原grid对齐候选最大框差0.83081px；本次blur概率未启用，其源像素、模型输入和完整原始预测与original均相同。未向模型传GT、未改变阈值、未调用SAM，也不把该单图train诊断当完整开发或真实视频成绩。证据outputs/action4_rgb_variant_b0_probe_v001/RESULT.json及ROOT_CONSUMPTION_ANALYSIS.json。生产版变体producer正在独立审查返修，不能将此诊断直接充当SAM新版本输入合同。

另由独立CPU38测量同一原生图（1预热+3次、暖缓存）：PIL RGB读取中位329.03ms，NPY eager8.42ms，readonly mmap后完整复制4.11ms；未含首次SHA成本。实际像素及prepare_pose_rgb的448RGB、A112/K和全部112geometry/metadata逐位相同。现有Action4Dataset有<=128 decoded LRU，但ComponentData/Phrase2Dataset没有使用；ordered_items仅预取，SAM provider没有另一次PIL解码可删。不能将39.1/80.1倍的单项读取差异写成训练加速倍数。证据outputs/action4_rgb_decode_performance_v001/{MEASUREMENTS,ASSESSMENT}.json。

按该单图尺寸估计（未扫描全尺寸、未分配缓存），原图train+eval decoded RGB约130.81GiB；若train三变体但eval仍原图，约363.35GiB。当前观察的系统可用RAM约413GiB、项目文件系统空闲约2495GiB只是时点资源，不能据此长期预占。下一步可在新源码候选中增加显式decoded artifact后端，复用现有reader返回结构，绑定actual input/source SHA、pixel/variant/recipe、NPY身份与完整预登记域；首次同句柄验证后用stat/PID校验复用，返回独立副本，禁止缓存可训练特征或越变体复用mask。必须完成真实loader端到端吞吐后再准入后续训练，不热改当前A0。

用户最新取消后续子智能体派发。已核实两个现有子智能体均completed；未续派。主控独立继续，训练/eval程序不受此调整影响。训练cache总COMPLETE已到达（5200/10400/ignore0，association约479.807s，provider约952.591s）；已按静态审查启动实际SAM消费检查PID3745784，CPU39/空CUDA，不代表已通过。B0变体producer最终spec/quality与主控12tests通过，真实production pilot待主控执行。


### 2026-10-25 全量训练候选与部署门分离：已接入活动运行

用户明确调整：全量All8@2%用于训练候选，困难组单列并作为部署门，后续针对远距/高遮挡继续训练。主控新增独立candidate observer，不改当前冻结训练/评估源码或旧BEST。9项CPU测试通过（0.094s）：覆盖困难组失败仍保留高All8候选、在原worker unlink前保留inode、历史权重缺失不伪造、与原rank完全等价、外来/非全量报告拒绝、恢复/陈旧终态/损坏权重拒绝。未派子智能体。

实际观察器PID3759830，CPU94/空CUDA，源码冻结在outputs/action4_training_candidate_selection_v001/source。首次处理421份完整评估，真实选出step4170，All8@2%=475/1300（36.5385%）、P95=28.0381px；far>=1m2/39、高遮挡3/192，因此部署开发门仍失败。选中.pt已保留并按SHA/role/step/config_id实际重载元数据核对，SHA35c232ecfc122932647a8c2e500588df440a5267a9bfccc3ea3dcb96041de376。当前训练候选恰与全部历史评估的最高rank一致；418个旧评估权重已被旧策略清除这一事实仍记录，不宣称恢复了它们。旧部署BEST仍step0，default未改变。活动候选尚未最终封存，后续按新政策持续更新。

真实SAM消费检查已完成：固定8个train单位读取，固定首对[0,1]两次实际CPU更新；目标0.205197→0.158358，adapter projection梯度L1分别30.2299/33.7677，13个可见点与3个隐藏点均进入相应有效分母，mask质量有效数1。更新后移除mask使112坐标角点最大差0.0859833；冻结backbone参数/buffers不变，严格重载输出精确一致，default不变。测试总739.982s（包括全train cache首次验证），这是实际工程消费证据，不是完整训练或效果提升。outputs/action4_sam_real_consumption_v001/COMPLETE.json。

B0变体生产入口最终spec/quality与12tests通过后，实际启动被旧B0 source中的41个未登记Python字节码缓存拦住。主控核对其均对应已封存源码，将433075bytes缓存可恢复归档至archive/legacy/20261025_B0_unlisted_python_bytecode_v001；所有RunSpec已绑定文件SHA前后完全不变，未改源码/配置/权重或当前A0。保留失败记录，未放宽inventory验证。随后冻结light单图worker实际完成2候选，源身份、变体ID、row.run_identity均核对，raw输出与此前同B0 CPU诊断完全一致；新入口未再次产生这些缓存。证据outputs/action4_rgb_variant_b0_preparation_v001/ROOT_ACTUAL_LIGHT_REVIEW.json。仍非full5200/SAM v3或正式训练准入。


### 2026-10-25 第2400帧衣物误检：实际候选池定位

用户指出的错误成立，不能以合成开发成绩或过去局部改善声称“B0不再误检衣物”。已确认本次视频确实加载B0 step9500/A0 step4270，temporal_mode=independent_frames_baseline，没有tracking。第2396/2398/2401/2402帧第二候选在右机附近，第2397/2399/2400帧跳到衣领附近；b0:1是每帧候选槽位，不是持续track ID。

主控用同一冻结B0做一次第2400帧CPU完整候选诊断（无新视频、无GT入模型）：30个raw候选，经同Soft-NMS顺序第1左机center0.95497/refinedQ0.93976，第2衣领0.79021/0.65313，第3真右机0.76591/0.86895。当前选择器while len(out)<2把第三个目标截掉；正式视频是GPU推理，CPU小数差异已保留，不将CPU重算覆盖原视频预测。A0没有身份确认门，把错误ROI当实例继续给出数值合法pose；pose_valid只检查有限值/正深度，不证明对象是DJI。

Tracking当前只具原生CoTracker两帧实调、session/关联/几何合同和部分接口；连续SAM会话、身份关联、材料轨迹到几何的完整视频闭环未接通。后续必须在候选预算截断前进行track条件关联；SAM/历史ROI要保持同实例，不能把新高分错框直接重置旧track；CoTracker跟踪可见材料点，经可靠几何约束影响A0姿态/八点，不把隐藏虚拟角点直接当可见像素query。跟踪可能稳定地跟错对象，不能替代B0身份鉴别。单帧quality乘积或阈值改善只是假设，未据此改默认策略。原始证据outputs/action4_real_failure_audit_v001/ROOT_CAUSE.json及FRAME2400_CANDIDATE_ORDER.png。


### 2026-10-25 SAM职责区分与当前真实验证边界

用户要求先完成B0＋SAM3.1视频跟踪，暂停YOLO；其他跟踪接入等A0训练结束并选定checkpoint后再推进。A0训练不得停止。图片训练/eval中的时间状态关闭；视频用近5帧粗位置约束选择并平滑更新框，SAM按其实际功能分路，不全部串接。

已实际运行连续120帧820–939：v001因14GiB进程配额OOM而非整卡耗尽失败，保留记录；v002提高至32GiB后正向传播完成，反向原生out_probs=-1e4哨兵被旧概率校验拒绝；核对固定官方源码这是removed-object值，现精确映射visibility_unknown，非空残余mask不注册为可用、不钳制成概率。21项相关CPU测试通过。v003完整运行两会话，mask可用率分别110/120与2/120，不是稳定跟踪通过。

源码核对显示text/box调用是概念提示并reset状态，propagation_full走VG；points+obj_id则走指定实例路径，可触发propagation_partial。两者不是同一功能。v004基于既有预测mask自动取内部正点与框外负点，在新会话直接point seed；原生返回无可绑定目标，已保留失败，不宣称PVS接通。其SciPy/NumPy版本警告也记录，后续不能把此环境路径直接当已验收生产依赖。下一步应核对正确实例初始化/衔接，而不是扩大概念搜索或每帧重发框。

新增B0 association pool保留30候选及同一次forward的128维外观，原top2输出数值保持；受控历史回放选中原第3名右机，但不等于SAM已稳定。轻量video_box_history只供video_inference，保留5次有效观测、位置关联、平滑bbox_for_roi、有限预测/失锁；图片construct/dataset/eval前置拒绝时间模块，6项测试通过。该平滑框必须进入实际A0 ROI与K变换才算接通，不能只改变画面绿框。

原视频现为用户允许的主要开发/专用优化数据，可针对其困难帧调参和训练；不再称为未见测试，不复制参考标注冒充模型输出。原始预测、失败、各模块结果分开保留。图像SAM、概念发现、实例跟踪、纠偏及概率/质量语义分别记录于CONTINUATION.json。


### 2026-10-25 新会话点提示输出组装的定点追踪

固定官方源码 _build_sam2_output（1241–1258行）在帧不属于cached_frame_outputs时直接返回{}，忽略传入的refined_obj_id_to_mask。主控提取实际函数AST执行反例：相同新mask输入，无帧缓存返回[]，存在空帧缓存返回[0]。这证明输出组装存在新会话边界缺陷，尚不能证明v004实际mask非空或语义正确；没有修改官方源码、成功门槛或默认模型。

证据：outputs/action4_sam_instance_seed_audit_v001/ROOT_CAUSE.json。另启动仅一次点提示的原生诊断（LAUNCH.json，PID3850951），固定原点坐标和官方方法返回，只记录组装前mask及返回ID，不传播整段、不替换预测；结果需以native/ASSEMBLY.json及完成/失败回执为准。GPU5、CPU88–93，A0训练PID3357408与候选观察器3759830已核实仍在运行。采用sup-systematic-debugging先区分计算结果与输出过滤，不用空响应假造目标缺失。

原生诊断已结束：同一提示得到1×2464×3248的bool mask，前景75115像素；frame_cache_present=false，官方组装returned_ids=[]。保留未修改原始返回，close成功，0活跃会话。预览确认前景位于画面右腕相机，非衣物；没有真实分割GT，不声明边界精度。候选组装函数src/model/action4_sam_seed_output.py修复fresh-cache合并，5项CPU测试通过，保留红/绿测试记录。该候选尚未安装到原生sequence，更未通过连续跟踪；下一步新快照实例级兼容装配后仍执行原绑定门和短片传播。证据ROOT_REVIEW.json、native/ASSEMBLY.json、native/SEED_MASK_REVIEW.png。

### 2026-10-25 实例输出修复装配与连续传播v005

在已确认的新会话mask丢失根因上，增加实例级make_seed_output_bridge：仅无帧缓存且有实际refined mask时合并；有缓存或无新mask仍调用原函数。不修改官方文件、分割网络、概率阈值或绑定门。安装器核对固定官方方法源SHA和模型类型，只有sequence显式instance_points及instance_output_bridge开启；单帧图片加载器不启用。7项组装/兼容测试与6项sequence回归通过。

新不可变快照outputs/action4_sam_b0_sequence_v005/source（8088b1ba6e400bf5ec1ecd7d5de94153d7312bd5b9e490c2c574bef0065a006c）启动PID3853179，GPU5/CPU88–93。复用原120帧B0池与相同预测mask点提示；B0未重算，A0训练不改。待原绑定成功及完整120帧真实传播证据后再评价，不能以安装回执或13项CPU测试声称效果通过。旧v004失败与native单点诊断保留。

v005已完成并退出：两条指定实例均120/120返回可用mask，会话正常释放。保持原绑定门，track00初始mask在目标B0框内占0.9942355、对另一候选覆盖0，唯一绑定通过。抽查820/824/850/880/910/939六帧，绿色/橙色分别留在对应腕部相机，未见切换衣物。原生手动实例概率1.0只是接口赋值，未作为身份或质量证明。完整原始输出与snapshot SHA保留于run/NATIVE_COMPLETE.json；ROOT_REVIEW.json明确仅连续传播及有限视觉检查，不是全视频精度或B0→A0联动验收。下一步消费该snapshot进行五帧候选关联和平滑ROI，再接A0裁剪与相机变换。不可把mask外接框直接当完整机身框。

### 2026-10-25 SAM→B0五帧关联真实回放及框视频

新增src/eval/action4_sam_b0_box_replay.py，消费绑定B0权重/像素/帧/时间/track/obj/generation的SAM快照；保留全候选与history-only对照，SAM可见框仅辅助关联，完整框仍来自B0。首次v001因消费端漏掉像素hash的dtype/shape头而拒绝，修复到既有生产者合同，保留失败，不改原数据。

v002真实回放种子824之后825–939共115帧/230实例：关联230、预测补帧0、lost0、SAM支持使用230；相比history-only有1次候选改变，相比单帧top2有28次选中top2之外候选。4项支持边界测试通过。该统计不是GT准确率。保存双向SAM语义，不能称实时因果追踪。

已生成115帧同帧框对照视频，输出见outputs/action4_sam_b0_box_replay_v002/VIDEO.json。第910帧目视：单帧top2框到墙面且漏左机，SAM+历史候选关联框住双机。仅此帧检查不能代替全片验收。视频明确box-only/no A0；下一步必须将bbox_for_roi送入真实A0裁剪与K变换，不把绿框变化当八角点改善。结果ROOT_REVIEW.json、FUSED_BOXES.jsonl、COMPLETE.json。

### 2026-10-25 平滑B0框实际进入A0 ROI与相机变换

新增data/action4_video_roi_context.py不可变视频输入，绑定RGB像素、帧/候选ID、B0权重、回放身份及K；observed轨迹用平滑完整框，保留原检测框；无当前检测的轨迹显式withheld，不能假造新测量。StagedPipeline新增predict_video，消费同一已有prepare_rois、A112@K与八点回原图路径；图片predict默认行为不变，增强SAM特征输入不与视频选框混淆。

冻结诊断源码以旧试验A0 step4270/B0 step9500在GPU5实际运行825/910/939三帧，各2实例均有限正深度，bbox_used与回放精确相等，K112与A112@K在rtol2e-6/atol2e-5内一致。第910帧实际红框由墙面移回左腕相机；仍有机身角点/姿态偏差，不能当八角点精度通过。完整产物outputs/action4_video_roi_consumption_v001/COMPLETE.json及ROOT_REVIEW.json。当前A0训练源码/权重不改。

视频相关测试80项：69通过、11按既有条件跳过。首次运行临时目录配置不符导致25错误，固定项目内TMPDIR后通过，保留旧日志；原生probe首次CUBLAS环境在CUDA初始化后设置被正确拒绝，第二次仅将环境前置后实际完成，失败保留。尚需115帧A0联动短片、2400衣领失败段与后续正式选优比较。

### 2026-10-25 115帧A0联动短片与残余姿态跳变

outputs/action4_video_roi_sequence_v001完成825–939全部115帧：复用115帧既有同权重基线与3帧已有联动结果，只新增112帧A0推理；B0/SAM不重算。所有实例ROI和K消费断言通过，230姿态有限正深度；这只证明数值和输入一致。视频VIDEO.json指向outputs/videos/20261025-112420-stage5-SAM-B0-A0-ROI-pilot，ffprobe验证115帧、29.97fps、1280×510。

连续诊断发现track01在899→900帧最大角点位移/机身框对角线0.6211，而框中心位移/对角线0.00738。实际两帧目视同一机身及相近绿框，红色立体框方向显著改变，证实B0框稳定不能直接保证A0角点稳定。位移不是GT误差，不把更平滑当更准确；未加EMA或改当前A0训练来掩盖。保存A0_JUMP_899_900.json和PNG，后续A0训练选优及材料几何调试需纳入此反例。下一优先继续2400衣领误检区域SAM/B0验证。

### 2026-10-25 衣领误检困难片段2360–2479启动

依据用户允许原视频开发调试的最新要求，将2360–2479明确登记为开发片段；该范围跨原guard边界，不能再称未见锁定评分。既有基线2360–2364双候选score约0.94–0.98，第2400帧衣领误检已确认。新120帧完整候选池用于连续关联，不用单帧top2截断后结果替代候选池。

增加discovery_seed_only显式功能分支：概念text+box仅初始化种子mask并关闭会话，不执行传播；保留原种子绑定门，随后独立指定实例路径处理。6项sequence测试通过，其中新增反例确保seed-only绝不调用propagate。新不可变快照outputs/action4_sam_collar_bootstrap_v001/source（301a3d5b2808b021551ea77de04f7480cc0921cfb747deb00ed89ed906a80850），控制器3866753先B0候选后种子发现，CPU86–87控制、88–93模型，GPU5；不干扰A0训练。实际结果以各回执为准，尚未宣称衣领误检解决。

困难片段B0全120帧已完成（38.29秒），v001仅确认右机种子，原生seed-only绑定成功并正常关闭。追踪SEED_ASSOCIATION_AUDIT发现左机2362帧被两个邻接grid36977/36978的重复假设触发0.15歧义门拒绝；二者框IoU0.96221、中心距/对角线0.00459、外观cos0.99830。未降低身份/质量门；新增仅种子确认阶段的重复假设归组：邻接grid、IoU≥.95、中心距≤.05对角线、cos≥.995全部满足，保留代表与alias记录，原完整池不删。真实前5帧回放恢复两条链，后续帧不参与种子决策。

8项sequence测试通过，包括重复峰不冒充第二实例、重叠但外观不同不合并、seed-only不传播。仍需真实近邻双机验证，归组不是已认证实例真值。v002只请求新恢复的track00左机种子mask（PID3869453），右机已有mask与B0池不重算；下一步两份来源分别绑定后进入指定实例传播。尚未证明2400衣领误检修复。

左机v002种子唯一绑定obj1，mask40545像素；同时返回右机obj0被原归属门排除。右机v001种子obj0保留复用。两份预测来源分别绑定后，启动outputs/action4_sam_collar_tracking_v001连续指定实例传播，PID3870478；不重算B0/概念mask，不将启动当跟踪通过。

### 2026-10-25 衣领片段完成、近邻反例与全2850帧启动

collar_tracking_v001两条轨迹各120/120可用。collar_box_replay_v001种子后115帧/230实例均关联，无predicted/lost；SAM使用220次，10次与历史冲突被忽略，history-only对照候选改变0次，相比原单帧top2选中其他候选38次。第2400帧目视原框选衣领，新输出框落在左右腕部机身，不能将全部改进归因SAM，也不证明全片正确。视频outputs/videos/20261025-113958-stage5-collar-SAM-B0-box保留box-only身份。

用户明确追加全2850帧检查，已启动outputs/action4_sam_fullvideo_bootstrap_v001（控制器3875632、B0进程3875633），最新观察253/2850，A0进程3357408仍存活。此任务先全B0候选，再仅种子发现；连续全片SAM需后续绑定快照，尚未开始，不能称全片已接入完成。显存/CPU独立限制沿用GPU5和尾部核心，不使用GPU0–3。种子发现只需前5帧解码，整片帧身份不缩减。

新增近邻测试实际暴露shared-large-box被错误标observed；修复为记录共享大框歧义并保留两条轨迹。9项测试通过。对两段已有真实数据回放共230帧/460实例，候选与平滑框相比旧结果均未改变（shared_box_rejections=0），证据outputs/action4_close_pair_guard_v001/REAL_REPLAY_REGRESSION.json。这仅证明已检查短片无回归，不代替全片近邻效果。

### 2026-10-25 全片帧完整性补齐

新增replay(include_prefix=True)显式全片模式：初始化前保留每帧B0 fallback并标unconfirmed，不伪造track；种子帧验证当前候选ID/框后发布seed_confirmed；后续每帧原轨迹记录保留，包括lost。默认旧短片模式不变。渲染器按完整输出帧数核验，初始化阶段用灰色原始B0框并明确标注，不能误称已跟踪。A0结果渲染器改为读取连续帧清单，不再硬编码825–939或模型step；当前只做语法核查，完整视频仍需实际验证。

8项prefix/support测试通过；复用已有真实820–939预测进行120帧回放，前4帧unconfirmed，第824帧两条真实种子，后115帧候选/框与已通过回放完全一致，无新模型推理。证据outputs/action4_fullvideo_contract_v001/ACTUAL_PREFIX_REVIEW.json。这不是2850帧完成证据；全片B0进程仍按原不可变源码运行，未重启。

### 2026-10-25 全片逐帧审计与独立mask证据

新增action4_video_track_audit.py，按完整帧/PTS/像素身份清单检查遗漏、重复track、一个检测分配给多track、lost仍发布ROI、伪当前观测、复制mask充当两份支持。计数two_observed明确不等于two_correct，不生成无GT的P/R。5项审计反例测试通过；旧真实120帧回放从已验证native snapshot补齐pixel身份后审计通过，4帧initializing、116帧two_observed，实际证据ACTUAL_TRACE_AUDIT.json。

replay新增实际pixel_sha256归组：同一像素mask即使session/obj/mask_identity不同也不作为双机独立支持；记录shared_identical_mask_pixels，历史B0分支仍按自身证据运行。与prefix/support合计11项测试通过；两框重叠但mask像素不同不会仅因IoU被合并。全片代码后续冻结使用，当前运行的B0提取源码不改。

既有基线全帧预测仅用来定位人工视觉复核候选，实际查看1088/2072确认双机靠近；1374预测框中含裤子区域，1178边界含义仍有不确定性，不将预测距离小直接标成真实双机成功。预览与选择依据保存在outputs/action4_fullvideo_contract_v001/NEAR_PAIR_CANDIDATES.*；这些重点复核不缩减2850帧检查范围。

### 2026-10-25 全片不可变交接与训练进度

src/eval/action4_sam_fullvideo_dispatch.py实现一次性交接：等待已有bootstrap完成回执，核对2850域、同源B0模型/区域、两条唯一绑定seed、native严格加载和关闭回执，再创建独立只读tracking快照。拒绝已存在输出、缺种子、模型不一致、未释放会话；无自动重启。5项测试通过（初版测试文件字符串转义错误日志保留，修复测试语法后执行）。冻结调度副本outputs/action4_sam_fullvideo_dispatch_v001/dispatch.py，PID3884444已等待；实际SAM子进程以CHILD.json及真实PID为准，不把等待当已传播。

最近观察B0全片2817/2850，原进程健康。A0候选观察器584份报告，训练候选step5740 All8@2%=559/1300（43.0%），部署困难组门未通过；A0进程3357408仍运行。此为合成开发选优，不等于真实全片检测或八点达标，默认模型保持不变。

全片B0完成2850/2850，实际候选推理1057.82秒（不含前置读取验证），B0_COMPLETE.json明确source/model/region/pool SHA。两条种子于第4帧唯一绑定并关闭；seed-only每会话只解码5帧，DISCOVERY_RESOURCE.json可核查，未传播概念分支。

交接器已生成outputs/action4_sam_fullvideo_tracking_v001只读快照，source identity eff0bc9965ea4ac730455689138783f047bd927fadc516fa6c22215737828ca2，子进程3885882已实查存活。该进程复用全B0池与两份种子mask，正在执行全片实例任务；尚未有完整2850×2响应/质量回执，不能当全片识别成功。运行源码保持冻结。

### 2026-10-25 全片后处理自动衔接与等待状态

新增action4_fullvideo_postprocess.py：等待原生dispatch退出回执后，验证同source、完整2850帧与已绑定track集合，再执行include_prefix全片回放、逐帧审计及完整框对照视频。原生失败则保留错误并停止，缺mask不伪造准确率；输出为REVIEW_REQUIRED，不自动发布达标。5项准入测试通过。只读快照outputs/action4_fullvideo_postprocess_v001已启动等待PID3888450，CPU86–87，无GPU、无模型重推理。

SAM进程3885882实查仍在官方帧加载阶段，最新1195/2850，约3.1–3.6fps；这是RGB加载进度，不是已跟踪帧数。健康任务不重启，不修改运行源码。A0四个eval进程3357372–3357375全部实查存活，队列与实际update见outputs/action4_fullvideo_contract_v001/A0_RUNTIME_CHECK.json；没有证据证明当前eval服务积压，不为填满显存盲目新增重复worker。

### 2026-10-25 全片基线发布阈值一致性修正

检查发现Soft-NMS内部select默认floor=.001，而正式StagedPipeline发布score≥.3；框对照renderer及prefix fallback此前直接画single_frame_selected。全2850候选审核确认仅2247帧受此差异影响，第二候选raw/selected均0.28985035，refinedQ0.35324，不是Soft-NMS额外压低。正式基线应有1帧单观测、2849帧双观测；该计数不表示正确率。

新增published_baseline复用既定.3门，修正对照绘制、prefix fallback和基线候选比较，不改原候选/模型/训练阈值。13项回放测试通过。尚未执行任何回放的postprocess_v001经PID/命令/UID及无输出核验后受控取消，CANCELLED.json保留；新不可变postprocess_v002 PID3892086等待同一原生SAM。A0与SAM均未停止或修改。全片SAM最新仅RGB加载2384/2850，尚非传播响应，禁止据此宣称跟踪已完成。

全片SAM第一条轨迹已从RGB加载进入实际传播，原生track00种子唯一绑定成功，当前进度{"responses": 554, "last_local_index": 557, "status": "target_visibility_unknown"}。子进程3885882已核实存活，未观察到失败；此为真实响应进度，不等于逐帧视觉正确。第二条轨迹尚未完成，保留同一运行，不重启。

### 2026-10-25 全片首次实际效果失败：第116帧起丢失实例

第一条轨迹在已处理771响应时，仅112份selected_mask，首个无目标公开响应为global116，此后到当次观察末尾659响应缺mask。raw输出out_obj_ids/probs为空，不是进程错误，也不是目标真实缺席证明。实际查看115/116原图，右腕DJI仍明显可见；115绿色mask已缩到很小局部，116完全无公开目标。记录EARLY_TRACK_FAILURE.json及FIRST_TRACK_LOSS_115_116.png。此结果否定“一次初始点提示即可全视频稳定跟踪”的假设；不能用原120帧短片成功或进程健康覆盖此负结果。

保留原全片无纠偏基线继续完成，不修改运行源码、不把缺失mask视作训练负例。后续独立候选必须接入原阶段五的受控B0确认重绑定/generation纠偏：区分支持退化/目标未取得/推理失败，连续证据触发，不每帧概念重搜。SAM可用布尔值不足以代表支持质量；B0位置历史分支输出必须与SAM实际贡献分开报告。完整结果和纠偏版本均需全2850验证，不能只补116附近后即宣称全片通过。

### 2026-10-25 全片终态OOM与恢复路径缺口

原生fullvideo_tracking_v001已真实终止：track00收到1512响应（最后local1515），32GiB进程限额内申请162MiB失败；当时整卡仍约15.18GiB空闲，不是GPU0物理卡工作，日志GPU0为CUDA_VISIBLE_DEVICES映射后的逻辑0（物理GPU5）。CLOSE确认释放会话；dispatch与postprocess均明确FAILED，无全片视频、无自动重启。全B02850候选、部分SAM原始结果、失败日志保留。此前“继续运行”记录为当时状态，本条覆盖为终态失败。

复用全片B0候选做无SAM历史控制回放：track00从193、track01从744进入吸收态lost，随后没有重初始化。observed916/predicted11/lost4763是算法状态计数而非GT误差。首次SAM丢失116附近，B0历史仍有正确位置候选可用于受控纠偏；仅无期限位置外推不能满足全片要求。

固定官方VideoTrackingMultiplexDemo底层支持offload_state_to_cpu，顶层_init_new_sam2_state未传入。新增实例级、固定源码SHA桥启用其官方CPU历史存储参数，不改网络/阈值；3项桥测试+6项sequence回归通过。实际50帧候选已执行，INITIAL_STATE_PROOF确认offload=true/storage_device=cpu，但末尾状态张量检查发现cpu与cuda:0混合，原严格检查失败，未宣称内存修复通过，也未保留到最终parity回执（需补逐字段诊断）。FAILED.json和关闭回执保留。CPU选项不能解决第116帧的感知失锁；下一步要分别闭合内存与B0引导重绑定，最终仍需2850全片验证。

### 2026-10-25 用户要求最近15帧：内存与接口约束

固定源码确认：普通mask memory与object pointers主要消费近邻/筛选出的少量历史，但交互式实现仍保存non_cond_frame_outputs及每对象视图；max_obj_ptrs_in_encoder默认16不是只存16帧，memory selection还可回查更早高质量状态。不能把原生默认保留策略当本项目必须保存全历史。磁盘全片证据与运行态张量分开管理。

新增action4_sam_history_window.py候选：跨帧只保留最近15帧普通状态＋最多2个conditioning锚点；计算当前帧期间允许额外当前状态，计算后收敛15。同步清理主output、per-object视图、临时non-cond视图、consolidated索引与公开mask缓存，正反向保留方向分别处理；不改磁盘结果，不静默删除超预算锚点（要求显式重绑定）。5项测试通过，尚未装配原生模型、未证明15帧策略效果或全片显存上界。

CPU卸载的第二次16帧逐字段诊断补足证据：12个可比较正向帧mask/ID/概率逐项完全一致；大块maskmem/image features/image pos均在CPU，普通pred_masks也在CPU。GPU仍保留小object pointer/logits及种子cond mask。此前“所有字段必须CPU”断言过强，不能据其失败推断大块卸载无效；保留FAILED回执，新增字段证据与PARITY.json，不改写失败。此诊断依旧不是全2850内存或跟踪恢复通过。

跟踪接口明确B0完整候选→实例关联（允许近邻框重叠）→observed/predicted/lost分离的完整框→A0 ROI/K；SAM输出是可见支持，不是完整框或类身份认证。下一步将15帧状态管理与受控纠偏接到原生调用，再用真实数据验证，不能以平滑或缓存测试代替定位准确。

### 2026-10-25 15帧原生装配与首个重绑定提案

install_history_window以固定源码SHA在_prepare_backbone_feats读历史前清理，SequenceBackend在每个原生yield之后同步清理普通状态/对象视图/公开缓存，记录实际数量和显存；图片路径不启用。5项状态测试+6项sequence回归通过。独立只读160帧GPU候选已完成，普通历史最大15、公开缓存最大见ROOT_REVIEW.json、peak allocated6398.21MiB（约6.25GiB），正反向全部160返回。可用mask116帧，失锁未因内存修复自动消失；与父原始前向mask比较结果逐帧保留。此为160帧状态上界证据，不是2850全片或纠偏效果验收。

新增RecoveryGate：明确present_unscored/degraded/missing/inference_error/ambiguous；缺失/退化至少3帧且B0连续5帧score≥.9/refinedQ≥.8、位置一致才生成纠偏提案。pending期间不每帧重复请求；原生唯一绑定明确接受后才推进generation，拒绝保持原generation并冷却，错误与歧义不当作目标缺席。6项测试通过。复用真实0–159失败记录和HISTORY_ONLY候选，生成track00:g1:frame121，证据117–121共5帧，generation仍0，未调用原生纠偏，不能把提案当恢复完成。后续必须实际获取该帧预测mask、唯一绑定、点提示纠偏与传播，并核查旧/新generation责任域。

### 2026-10-25 首个原生纠偏完成（离线固定提示对照）

第121帧依据117–121五帧B0证据实际运行SAM单图概念提示，133015像素mask唯一绑定目标，另一机身候选保留并参与排他检查。mask视觉位于右腕DJI；预测mask生成2前景/2背景点，使用本项目CPU环境计算，无GT点。该功能为纠偏证据获取，不把概念发现直接当连续跟踪。

新160帧只读候选在初始frame4点提示外，额外预登记frame121纠偏点，再双向传播；保留15帧普通历史＋两个conditioning锚点。原生纠偏响应再次唯一绑定obj0，完整160帧返回。可用mask由同15帧无纠偏基线116增加到158，剩余无mask帧116/117；max ordinary15，peak allocated6859.84MiB（约6.70GiB）。实际查看115/116/127/159，后两帧mask回到目标机身，115仍退化，116仍缺失。结果outputs/action4_sam_recovery_propagation_v001/ROOT_REVIEW.json。

该实验在传播前提供固定纠偏点，属于离线重放，早于121的结果可能使用未来conditioning；不能冒充在线响应时延或全片真实精度。尚未提交全局generation或默认流水线，必须将连续纠偏/锚点退役/失锁重识别接入后再全2850验证。保留无纠偏、OOM、剩余缺失及全部原始记录。

### 2026-10-25 前向逐帧触发纠偏已实际跑通

新增显式candidate-only propagate_forward，原双向入口不变，必须关闭活动stream后才能调用纠偏；8项backend测试通过。因官方多次无新提示传播可能转为propagation_fetch，不能把任意分块当持续推理；本候选仅在实际新纠偏后续跑。

冻结原生候选在frame4开始前向传播，RecoveryGate于frame121依据当时missing及117–121五帧B0观测触发；关闭stream后调用原生点纠偏，唯一绑定通过后才提交候选generation1，并从121前向续跑。最终frame4–159共156记录连续唯一，generation0责任[4,121)，generation1[121,160)，旧121失败响应单独留档不重复发布；0–3明确初始化。可用151/156，缺失116–120。没有提前向模型输入121提示，但提示mask预先由该帧单图推理准备，未测实时端到端30fps，不能称实时能力通过。输出、绑定、触发和显存见outputs/action4_sam_causal_recovery_v001/native/COMPLETE.json及ROOT_REVIEW.json，默认流水线未发布。

新retire_previous_anchor调用官方clear_all_points_in_frame(need_output=False)，仅在新纠偏唯一绑定后退役旧纠偏，初始化锚点永不删除；未知额外锚点拒绝处理，随后同步15帧引用清理。4项接口测试通过，尚未经过多次原生纠偏验证。不能将此单次成功或纯B0历史控制的长期lost吸收态当已解决；接下来闭合重识别与全2850流程。

### 2026-10-25 按用户简化：五帧B0主链与全片重新确认

新增FiveFrameTracker复用原VideoBoxHistory：五帧位置关联和平滑不变，失锁后由五帧连续score≥.9/refinedQ≥.8、位置连续的当前候选重新确认；同候选不能填两个轨迹，近期覆盖双机的大框拒绝，重叠独立候选可保留。恢复调用明确reinitialize_confirmed接口，重置旧位置而非与失锁前位置混合，association_epoch递增，物理身份连续性标未验证。6项新增测试+9项原核心测试通过。旧默认核心仍不自动恢复，新wrapper显式候选使用。

实际全2850候选回放完成，无新模型推理：跟踪域2845帧/5690实例记录observed5349、predicted26、lost315；旧无恢复控制observed916/predicted11/lost4763。恢复发生215(track00)、765/1445(track01)，lost区间[193,215)、[744,765)、[1173,1445)。这些是状态覆盖，不是准确率。首次冻结驱动生成时换行转义语法失败，v001保留，v002修正后运行。全片视频outputs/videos/20261025-134552-stage5-full2850-B0-fiveframe-reacquire已完成2850帧，明确SAM OFF/no A0。4线程有界图像预取＋2线程编码，源帧4/2072/2400与串行显示像素完全一致；16项渲染/回放测试通过。

对照demo与原画面1374，左机仍可见；B0候选grid38544 score0.623688/refinedQ0.630090，框偏向裤子，其他附近候选更弱。新流程没有通过强确认恢复左机，故全片准确性未通过；不能将长失锁解释为目标出画，也不能放宽阈值重新画错框。下一步保持五帧位置主线，让SAM在B0退化/缺失时辅助当前机身位置；不把复杂多锚点流程作为主链前置条件。旧多锚点代码/准备证据保留，不再因此阻塞B0推进。

### 2026-10-25 用户重申A0连续训练与后续扩展倍数

实际核验A0进程3357408及4个eval worker、候选观察器3759830均存活。当前step6878、epoch32/80，每10optimizer step完整eval；请求688、完成687、仅6870待完成，失败回执0。训练候选step6670 All8@2%=586/1300（45.08%），困难组部署门未通过，默认权重不改。继续原不可变任务，不重启、不混接新optimizer。

用户计划稍后调整B0框扩展倍数，本轮不调整。必须分开原始完整机身检测框、A0裁剪ROI扩展、SAM搜索支持范围；后续只修改明确指定的控制量并做同权重/同帧对照，不将搜索范围当机身预测框，也不更改当前训练源码。

### 2026-10-25 A0实际供数瓶颈与12-loader对照

实际40.66秒采样step7920→7924约10.17s/update；4个loader各近满一核，主训练多次do_poll等数据，pending仅2。旧STATUS.seconds约2.6–3.1s从loader交付后计时，不能当端到端。真实RGB为2464×3248，PNG解码中位258.6ms，完整target准备369.1ms/实例。优先提高供数并行度，不通过重复训练进程挤占同一run。

当前工作树visible mask重复解码已改为同item一次打开、两次原规则采样。3tests及16真实案例完整pre-edit输出精确一致；frozen共同契约一致。局部准备中位365.0→354.4ms，约2.9%改善，未应用到运行中冻结源码。Root已先spec再quality检查，见ROOT_REVIEW.json。

4/12loader CPU对照覆盖每种拓扑1424单位/30batch，正常、尾32和跨epoch的完整tensor、metadata、micro2、顺序与增强种子精确一致。正常16batch含hash/资源检查130.610→71.279s；尾段/启动不提速，不能当长期训练吞吐。

实际GPU5训练对照从独立读取的同一step8090 checkpoint开始，各16次真实更新后model/optimizer/scheduler/RNG摘要全部一致。含loader启动136.092→78.484s（1.734倍）；本小段排除前3更新后7.231→2.716s/update（2.663倍），不包含正式保存/发布/full eval，不承诺长期同比。首版探针import了新版才有的函数而失败；修正import后还发现共用内存optimizer state使第一组原地修改第二组初始step，已用CPU别名反例复现。新版每组独立读不可变checkpoint通过；失败全部保留，未影响正式训练。证据outputs/action4_a0_loader_training_probe_v001/v002。

正式完整650eval最近约136–153s/项，四worker平均服务约36s/项；若提速后每10更新约27s，必须同时评估6–8worker服务能力并遵守72CPU/GPU4–7及不挤占其他任务。不能仅加loader而忽略eval反压。

当前正式运行不支持DataLoader热增，也无formal cooperative stop；直接停止trainer会触发旧controller终止eval。已向用户确认既有“不停A0”是否允许一次受控checkpoint切换，答复前原任务继续。新增纯loader-continuation校验器10tests通过，严格限制训练语义/数据/源身份/完整状态/CPU，只返回launch_allowed=false和未满足的运行门；它不是控制器或迁移完成。不得改checkpoint config_id伪装原生恢复，不得混写旧eval。

本次只读更新：step8208，epoch38/80，eval 820/821，failed0；训练候选step8120 All8@2%=622/1300，部署门False。代码候选与短程实际更新验证通过，正式80轮尚未完成，不代表真实视频精度提升。

### 2026-10-25 双ID原生连续传播与分对象历史修剪

原生fixed-forward候选v001在frame35失败：obj0的纠偏20被当作全局conditioning豁免，未纠偏obj1的ordinary20也未老化，造成16项ordinary。CPU结构反例修复前复现[15,16]；不是通过更改计数掩盖。修复后每个native core仅保留自身conditioning豁免，共享public cache仍保留全局真实conditioning并集；7项测试通过。

Root检查最小diff后冻结v002并实际重跑44帧、四段forward、仅纠偏obj0：两ID在44/44帧均有新计算score元数据及public mask输出，15帧ordinary断言通过，peak约6507.69MiB，session关闭。结果与ROOT_REVIEW在outputs/action4_sam_native_forward_v002。这验证了局部纠偏后两ID继续计算与历史有界的接口；不是mask质量、完整B0框或2850全片验收。原SAM视频mask→A0 enhancedC5/surface及CoTracker调度/几何仍单独待闭合，没有用B0框track替代。

### 2026-10-25 双机支持框修复、原计划接口和完整joint运行

SamSupportedBoxes已修复真实散点污染：1178原mask仅16个远处杂点就把bbox从主体约202×198拉到1430×1488，错误放行旧B0尺寸/偏移校准。现在只对几何支持清理小于最大连通组件0.5%的远处组件，原始mask不改；校准使用真实框内支持像素/候选框面积。历史框裁切转动机身的反例已复现并加入当前可见支持enclosure，极细长局部支持不触发完整extent更新（aspect限4）。输出仍明确full_box_estimated，不把mask当amodal真值。

独立审查额外复现：总支持比0.4而框内仅0.32的校准误放行，以及两份IoU0.99875近复制mask被误当双独立支持。现已分别修复；cleaned-mask IoU≥0.9标记ambiguous_shared_instance_support，保留raw并不认证SAM当前支持。仅B0框重叠不因此拒绝。16项针对测试通过，独立spec/quality复审通过；真实356帧mask对IoU均0，反例不是本段已观察到的失败。

实际回放v007覆盖1124–1479全部356帧，两ID框finite/正尺寸，323observed/389sam_supported，均非精度分母。与已生成v006的bbox/status/candidate/source/calibration字段完全一致，复用视频不重编码。主控查看16个登记代表/最大运动帧，包括1195/1289/1374：右机错位改善，1195左机原机身裁切已覆盖。相邻中心位移P95由45.45到39.86native px，只是动态统计而非GT误差。仍存在估计margin，未获得完整2850准确率或A0接线通过。视频outputs/videos/20261025-175450-stage5-SAM-B0-geometry-gap/comparison.mp4，356帧，ffprobe验证；首次PATH没有ffmpeg的编码准备失败保留，改用已有runtime/ffmpeg/bin工具后完成。

原计划功能单独推进：VideoSamSnapshot复用实际ROI/mask输入类型；主控要求并修复显式ROI policy（默认2.25不变），10测试通过。MaterialLifecycle直接消费selector packet；主控要求并修复qualified_binding的当前frame_id约束，15测试通过。两者均为接口准备，未声称当前baselineA0消费mask、真实材料激活、CoTracker或几何精度通过。

新完整原视频joint候选outputs/action4_sam_joint_full2850_v001已由主控审核后启动PID4095999，GPU5/CPU88–93，32GiBallocator。复用原2850PNG和初始frame4的真实bootstrap predicted masks，明确obj0/1映射track00/01；一个forward流4..2849，0..3显式unconfirmed。无GT/demo/gap1124提示，无新B0/A0推理。每core15历史和原生双ID连续计算已在44帧验证，本次验证完整片段的覆盖/内存/失败分布；auto_correction=false，完成也不等于准确率通过。启动时GPU及CPU已核查，不影响A0。首次实际检查仍在PNG加载420/2850、约4帧/s，PID真实存活，尚未开始完整传播，不能把加载计为跟踪完成。后续须对实际失跟定位纠偏，不能以该baseline代替最终跟踪要求。

A0本轮最近核查step8491、第40/80轮，训练候选step8440 All8@2%=622/1300，部署门仍false；没有受控切换或默认权重覆盖。


## 2026-10-26T01:21:21.872987+08:00 全片跟踪交付、机身边界诊断与 SAM ROI 到 A0 实际消费

- 完整对照视频已生成并核验2850帧、29.97fps，无丢帧/补帧：`outputs/videos/20261026-000314-stage5-full2850-B0-SAM-tracking-v002/comparison.mp4`。左侧demo仅供参考，右侧两个固定ID的2D跟踪框，无A0。原生SAM第4–2849帧两ID均有mask，前4帧为真实B0证据经第4帧确认后的离线回填，不能称从第0帧因果在线识别。当前仍有支架/线缆混入机身框，未判定全程准确。
- 全片原生推理已结束，不重新启动PID12578。主控抽查1195/1289/2400已不再复现旧屏幕内部种子导致的大段缺失及2400衣领误框，但框范围仍需修复。原完整失败、原始预测和候选均保留。
- 拒绝把形态学body-core候选作为生产机身：1195/obj1只保留29.35%mask并切掉机身；不得降低门槛掩盖失败。证据`outputs/action4_sam_body_core_v001`。
- 完成15次独立SAM PCS查询及12次使用现有跟踪框的修正查询。第一轮raw mask极值受离散小岛污染，第二轮改用既有跟踪框。文字-only在三个困难帧的9次查询均无输出；camera/action camera+框仍包含支架，部分查询返回两台。这些均不是可靠完整机身修正，不接入已有PVS、不重渲全片。原始请求、原生输出与视觉证据在`outputs/action4_sam_body_semantic_probe_v001/v002`。
- 完成SAM-only当前证据到A0的主控代码审查和真实NN验证。第1195帧旧默认路径0个ROI，新显式准入2个ROI，A448/A112/K112实际消费并输出两组有限canonical8；第2400帧正常B0路径保留2个ROI。没有当前B0时score仍为null，不冒充B0识别。旧A0step4270和当前best step14550各做同帧接口检查，无新整片A0候选视频。见`outputs/action4_sam_supported_roi_actual_v001/COMPLETE.json`。
- 这只是SAM跟踪几何影响RGB裁剪，不是mask注入C5。当前best14550的可见/amodal mask在1195仍偏移、漏覆盖，不能反过来当作机身真值裁框。证据`outputs/action4_body_scope_a0_diagnostic_v001`。
- A0继续80轮训练、每10实际optimizer step完整650图eval。本次读数step15416、epoch72，续跑后缀请求596、已完成591、失败0、待处理5。全程最佳按唯一ONELINEAGE指针为step14550，All8@2%=700/1300；困难组仍未通过。没有改运行源码、正式数据或默认权重。
- 状态分别为：ROI连接代码与实际模型消费通过；A0训练未完成；全片跟踪连续性已有实证，但完整机身框及真实八角点效果未验收。后续先闭合机身范围/近邻与接缝检查；A0结束选3–4候选视频；再按原稿接SAM特征、CoTracker材料和几何，不以本次B0框跟踪替代其他阶段。


### 2026-10-26T01:26:42.242299+08:00 机身范围的条件诊断
在原图1195的网格放大图中明确选择约略机身框及2正/2负点，与既有跟踪框PVS提示作同模型2帧对照。人工点用于开发诊断，不是GT、不用于训练或自动识别评分。机身范围提示在1195/1196明显减少两侧支架覆盖，而旧跟踪框提示仍分割较大的黑色装配体；另一台保持原提示。这支持“提示的物体层级不够明确”这一原因，尚不能证明自动纠偏已经可用。正在`outputs/action4_sam_explicit_body_clip205_v001`对1195–1399延长检查，固定新generation和两个ID，后续帧无新增提示；不覆盖原完整视频，不宣称自动全程准确。


### 2026-10-26T01:36:08.093677+08:00 从人工条件诊断推进到B0派生提示
人工范围的1195–1399共205帧诊断结束，205帧均有两ID；选定时间点显示被纠正目标主体更集中，另一目标仍有附件。不将其计为自动识别成绩。
随后使用1163帧的两个实际B0框（均score≥0.9、refined_quality≥0.9且既有校准已接受），按框中心生成正点、框外四边中点加0.15宽高偏移生成负点，未使用人工像素提示。1163–1399共237帧、无后续提示，两个ID均在237帧输出；主控查看9个时点，1195大装配体混淆明显减轻，转腕后仍能带入细线缆/连接件，因此未通过完整机身边界验收。这个开发片段的起点由主控选定，不冒充自动事件调度已经完成。
已将提示构造整理为`src/model/action4_sam_seed_extent.py:b0_body_scope_prompt`，先保留6项缺功能失败，再实现并通过6项新测试和4项既有测试。新接口生成的两个请求与实际237帧原生运行请求逐字段一致。图边裁剪后落入机身范围的“负点”直接丢弃，无合法负点则拒绝，避免把机身当负提示。现有2/3角点提示接口未改变；训练冻结源码未改。
证据：`outputs/action4_sam_b0_reanchor_clip237_v001/ROOT_REVIEW.json`、`outputs/action4_b0_body_prompt_contract_v001`。尚须接自动触发、joint-session/generation重放及全片复核；不直接覆盖现有完整视频或默认模型。
A0最新step15740、epoch73；续跑后缀eval完成622/629，失败0，待处理7。当前候选All8@2%=700/1300；仍在训练，不提前封存为最终best。


## 2026-10-26T03:15:04.379977+08:00 A0完整训练封存、自动重锚全片与后续变体原生接口
- A0累计80轮/17360实际更新已完成，完整1737次eval（9450前缀946次＋续跑后缀791次）闭合，全程observer sealed=true。训练全量All8最佳为step15770、701/1300；legacy部署门BEST=11710不是用户要求的全程训练最佳，禁止混用。独立封存器从不可变续跑源码复核prefix/suffix全部报告与原有rank，严格加载后封存到`models/candidates/action4_improvement/stage1_A0_completed_20261026/A0.pt`；回执schema为`action4.sealed_lineage_training_candidate.v1`，不是默认部署通过。默认权重未改。top4训练候选为15770/14550/12990/12810，后续按用户要求进行视频比较。
- 新增BodyReanchorGate与CandidateReplay，8＋6项CPU测试通过。实际自动流程完成2850帧、12次原生重锚及旧会话关闭，generation逐帧归属完整。1195难例改善来自自动1170事件读取过去1163的B0校准证据，无人工像素提示。第一次快照缺少依赖在GPU启动前失败，保留v001；新v002通过不可变导入预检。SAM隔离环境中原SciPy要求NumPy≥2而SAM固定1.26.4，使用项目内独立SciPy1.14.1依赖，不改已装环境，6个真实mask几何结果与原环境逐位相同。
- 新视频`outputs/videos/20261026-022351-stage5-B0-SAM-auto-reanchor/comparison.mp4`已完成2850帧/29.97fps并传回本地，SHA256为`3f5fc80ae9c861f955af390b5d156f58a5e66a5eddf13573d2f5b2b9e075e408`。四帧初始化仍为实际B0证据的离线确认回填，不称从0因果在线。最终编码复用预测，不调用模型。
- 不判定全程稳定通过：原生/融合输出无lost不代表机身准确；完整位移统计发现track00最大归一中心步长由约0.211增至0.402，1636处mask包络把框拉向图像底边。源码中包络还写回尺寸/历史状态，是明确的问题路径。新增独立BodyBoxProjection只影响发布框/ROI，不改关联和native事件；6项测试通过，12CPU进程32.52秒重放全片、未重跑SAM。但严格v001会切掉1195转腕后的机身，拒绝；放宽旋转范围v002虽恢复1195并拒绝远处异常区域，1637仍有机身覆盖疑点，暂不采用、不另生成视频。
- 后续数据状态复核：5200图/10400条原始SAM训练缓存及关联、650图新B0评估缓存、真实两次更新与严格重载此前已完成，不重做。新增RGB变体输入合同与原有native worker显式schema分支，6项合同＋4项native/legacy回归测试通过。一张实际light变体的两个B0候选完成原生SAM推理、都关闭会话，变体文件SHA与原图不同且独立variant ID贯穿请求和回执。证据`outputs/action4_sam_rgb_variant_native_pilot_v001/ROOT_CONSUMPTION.json`。仍需variant exporter/provider、对应监督关联和Stage2Units按图选择，未开始阶段三正式训练。
- 当前边界：A0训练完成与全程训练选优封存完成；新跟踪全片是候选，框范围/稳定性未最终验收；变体原生接口通过不代表正式训练准入或效果改善。


## 2026-10-26T03:55:49.631626+08:00 四候选完整视频比较已完成
按已完成80轮训练的全程候选清单，分别冻结15770/14550/12990/12810，固定B0 step9500。为此显式扩展completed-training-comparison授权类型，并验证候选必须属于只读lineage seal清单、B0必须匹配固定seal，原during-training路径保留；7项合同回归通过。使用GPU4–7共10个视频worker，60工作CPU＋4控制CPU＋1主控CPU，未重启训练。
四个输出各2850帧、合计11400视频帧，逐帧B0检测完整字典在四组结果间严格相同，数值无效pose为0。视频位置见`outputs/action4_completed_video_comparison_v001/COMPARISON_INTEGRITY.json`，均已传回本地outputs/videos。模型/帧/PTS/配置检查通过不代表角点精度通过。
主控同帧查看0/300/1195/2400：开头清晰双机较合理，但共同B0袖口/衣领误检和1195合并/过大ROI问题仍明显。没有证据把其中任何一份宣布为demo级部署结果；15770保留为训练指标最佳，不覆盖默认权重。此组SAM/CoTracker关闭，不能与已启用SAM的跟踪视频混淆。下一步应在真实绑定的跟踪ROI上验证A0，并推进原稿SAM特征/几何流程，不能仅再加训练轮数。
额外几何候选采用预测机身窗口选择连通区域，6项CPU测试通过，保留原始mask。复用已有原生输出检查25个包络冲突，仅3个通过有界恢复，22个仍未解决（多为1638–1656及1532/1533/1543）。未作为生产采用，也不把更小框当准确性。证据`outputs/action4_body_support_components_v001`。


## 2026-10-26T06:50:22.576989+08:00 跟踪ROI全片接入A0及阶段三真实变体链路
- 已完成不可变v003自动重锚全片（2850帧、5次原生纠偏），通过明确的B0质量、外观、运动与另一实例支持约束解决“正确B0被过大SAM mask硬拒绝”的接口问题。保留原始预测和负对照；不将无lost等同准确。新机制包含冲突持续状态及连续三帧干净证据恢复，避免一帧证据把异常mask重新写回框尺寸。
- A0固定15770、B0固定9500，使用已绑定session/generation/native ID的真实跟踪ROI完成2850帧：实际A0 head 5692次，B0/SAM模型不重复运行。首四帧复用同身份基线并明确离线确认来源。完整视频见outputs/videos/20261026-060751-stage5-A0step15770-tracked-ROI/comparison.mp4，本地副本SHA与服务器一致。实际CPU配置55逻辑核，旧描述50已通过RESOURCE_ACCOUNTING_CORRECTION纠正；不修改已完成运行配置。
- 主控查看编码后的1195/1636/1950/2400：1195袖口、2400衣领误选得到改善，两台机身分离；1950仍有明显姿态偏转，1636也不能判八点准确。POSE_JUMP_DIAGNOSTIC表明框中心稳定时角点仍会大跳，且不能仅解释为角点ID互换。该视频没有SAM特征注入、CoTracker或角点平滑，不作为后续阶段完成证据。
- 已完成三个限定数据接口：实际light原生预测导出与严格provider；独立GT监督归属；Stage2Units按原始image ID/epoch选择变体并在ROI采样前替换RGB。主控分别独立运行13、12、13项针对测试通过。只有一张light实际图/两实例，不称全量多变体完成。关联任务误触旧CPU模型工程测试的越界范围已记SCOPE_DEVIATION，未用于本接口效果证明；无GPU、无optimizer更新或权重写入。
- 继续使用封存A015770启动实际light→SAM→loss两次更新与严格重载检查。第一次启动未显式传ACTION4_PROJECT_ROOT，数据准入前失败，原日志保留；第二次使用同一不可变源码并设置正确code/data root。正式80轮未启动，默认模型与6500发布数据未改变。

阶段三上述真实变体检查已结束：物理GPU5/CPU90–95，一张light图两实例从封存A015770实际更新两次，adapter梯度非零断言通过，backbone参数与buffers相对父权重完全不变。原重载测试误将SamGeometryHead作为RGB模型调用，保存后失败；复用已有两次更新权重，以训练包装器和正式部署head/backbone两条路径严格重载，rotation/translation/camera_corners/corners/depth逐位一致。原保存前输出与梯度数值未落盘，不声称保存前后逐位对照，也不重训伪补记录。证据outputs/action4_variant_actual_updates_v001/COMPLETE.json；这是实际训练链路检查，不是模型效果改善或阶段三全量训练完成。


## 2026-10-26T07:01:26.796907+08:00 blur原生链路与主稿抽样规则对齐
冻结B09500完成实际blur变体一图预测，SAM3.1随后完成两个预测提示且会话关闭。证据outputs/action4_rgb_variant_blur_actual_v001及outputs/action4_sam_blur_native_v002；已导出blur cache和独立监督关联，未改变发布数据。GPU B0首调用缺CUBLAS_WORKSPACE_CONFIG而失败，已保留并设置:4096:8确定性要求后同源恢复；SAM第一版因CPU affinity不符合当前authority在推理前拒绝，新不可变v002使用88–93执行成功。
发现integration选择器与主稿已有RGB选择函数不同：新增显式native_policy_one_based_v1，必须original/light/blur三叶等概率，复用action4_rgb_variants.select_variant，Stage2Units epoch只+1一次；旧integration_weighted_v1保留并明确不能冒充主稿正式规则。两图×80epoch逐项与原函数对照、非法权重/未知规则及原13项接口回归，共16测试通过。新配置PLAN_SELECTION_CONFIG已落盘，仍是integration，未提前开放formal。
实际三叶数据消费进程308292仍在验证完整原图cache依赖，CPU实测工作中；原图结果不重算推理。完成前不宣称三叶消费已通过。下一步读取outputs/action4_sam_three_variant_consumption_v001/COMPLETE.json及session95374；不因耗时或工具观察超时重复启动。


## 2026-10-26T07:05:26.086008+08:00 训练输入重复解码优化
三叶实际消费PID308292仍在运行（核查时6分41秒，约78%单CPU占用），没有重启。源代码核查显示完整原图cache准入会遍历10400实例并验证payload，此前只用一个light小缓存不暴露这项启动成本；当前不能把这个耗时归为GPU训练慢。尚未取消或弱化完整依赖验证。
对实际逐实例重复取RGB，增加按PID、解析路径、SHA、stat指纹绑定的64MiB有界解码缓存。每次命中验证路径/身份稳定，返回副本防止增强污染；fork、文件变化、错误SHA仍拒绝或重新验证，不缓存任何可训练输出。新增3项加既有variant export13项，共16测试通过。2464×3248真实图首读0.3199秒，后续三次0.0167/0.0162/0.0156秒，像素逐位相同，占24,009,216字节。这只证明解码复用收益，不称训练总吞吐提升20倍。证据outputs/action4_sam_three_variant_consumption_v001/RGB_REUSE_BENCHMARK.json。新实现未热替换当前已导入旧源码的消费进程，后续新快照使用。


## 2026-10-26T07:10:38.251341+08:00 三叶消费完成及全量RGB变体启动
PID308292已正常完成：一张图两个实例在6个epoch选择中实际消费original两次、light三次、blur一次，两个实例choice identity一致，native prediction ID分别绑定对应变体，均available；这是数据消费而非模型效果。该进程使用历史integration selector，主稿canonical selector另有16项对照测试，正式运行将使用PLAN_SELECTION_CONFIG。
64张分散训练图像的8个生成分片全部完成，函数在READY发布前逐图重新按固定recipe核验像素和来源；8图分片约105–112秒。证据outputs/action4_rgb_variant_scale64_v001/ROOT_COMPLETE.json。全量控制器315917核验64图域、READY和manifest身份后已启动64个单核worker，处理其余5135张（原单图＋64图复用），目标合计5200原图引用＋10400派生图。工作CPU30–93，控制29，主控28；先前原图检查94–95已结束，实际当前合计66逻辑CPU，上限72。无GPU推理、无新数据标签或正式训练，发布6500和models/default不变。
全量目录outputs/action4_rgb_variant_full5200_v001，完成后核查COMPLETE的5200唯一图像全集与73个分片manifest。后续才按各变体执行冻结B0/SAM和监督关联；不得把RGB READY当作SAM完成或训练准入。当前不重复启动控制器或已运行分片。


## 2026-10-26T07:17:14.740096+08:00 冻结B0分片并行及变体mask鲁棒化接入
已在物理GPU4/5/6/7分别启动一个冻结B09500变体生产rank，CPU24–27；与64个RGB生成worker、控制及主控合计70逻辑CPU，不使用GPU0–3。消费已通过的64张图×light/blur，共16分片/128图次；已看到各rank第一分片完成回执，其余继续，未重复原图B0。入口复用action4_rgb_variant_b0.prepare和不可变旧B0源码/原predict，不另建检测器。目录outputs/action4_rgb_variant_B0_scale64_v001。
变体provider新增固定mask augmentation路径：必须训练域及已验证监督关联，按原政策对预测ROI448 mask做dropout/扰动；selector显式验证policy binding并传给当前选中叶子的provider。RGB、K、GT可见mask及实例身份应不变。routing仍拒绝，未绕过正式准入。新增实际light数据的对照测试，连同原selector和association回归正在运行（session13864）；完成前不得称通过。当前运行的RGB/B0进程使用各自不可变快照，不消费这些新改动。


## 2026-10-26T07:23:19.557904+08:00 分片SAM来源读取接口
新增action4_sam_cache_collection，通过原图ID唯一映射实际cache/association分片，拒绝重复和缺失，不把多个原生source identity压成一个伪来源。选中分片才构建原SamCacheProvider并执行其完整原生/监督验证；metadata索引本身不等于mask准入。worker最多保留两个provider，避免随分片数无界增长。SamVariantSelector显式支持shards配置，仍仅integration，formal gate未开放。实际单light分片与原直接叶子的ROI、mask、GT和identity对照及重复/缺失/来源失败测试正在运行，session61797；前一mask augmentation回归session13864继续，无失败日志，尚未完整结束。当前RGB/B0不可变任务不消费新接口。


## 2026-10-26T07:29:46.568798+08:00 B0分片闭合、SAM队列启动及ROI变体接口
128图次的16个B0叶子全部完成，主控核对每个PREDICTIONS SHA和各variant的64唯一图全集；不是准确率评分。旧4个B0rank已结束。r101仅把SAM CPU从正在做RGB生成的88–93调整到释放的24–27、94–95，GPU4–7边界不变；与64生成CPU＋主控28/控制29的并集72。SAM控制器324573已启动16个原生分片队列，已完成首light叶子并进入blur，逐叶保持独立来源身份，未重复B0。
前次mask augmentation＋关联/selector回归26项通过（658秒），collection4项通过（235秒），均是当前CPU竞争下的工程耗时，不是模型吞吐。
继续补齐原50/30/20 ROI路由到实际选中变体：路由器支持已验证variant association及variant-qualified candidate ID；Stage2Units改为调用当前选中provider，而非固定self.sam_provider；full/coarse缺源仍按原policy记录回退，不伪造coarse。原预测mask和GT监督仍分别处理。新增actual light四种requested route并联mask augmentation的测试，连同原router/count回归正在运行session63901。未开放formal训练，extra-offset校准及coarse来源仍需后续闭合。

上述variant ROI routing测试已结束：实际light图四种requested route、原router/count及collection contract，共14项通过（46秒）。prediction_coarse因当前缺独立coarse源而明确回退prediction_full，日志和计数保留；不把回退称为coarse功能完成。全量RGB生成核查时已写9601张派生PNG，但64分片尚在完整recipe/像素校验，文件数量不是READY，不提前消费未完成分片。


## 2026-10-26T07:39:24.535567+08:00 同分片重复验证优化与后续B0自动队列
发现prepare light/blur时同一manifest会重复执行完整recipe/pixel验证。增加进程内成功proof复用：首次仍完整验证；key包括PID、manifest路径/SHA及全部来源图、派生图、annotations、policy、EXPECTED/READY的解析路径与stat指纹；FAILED出现立即拒绝，前后指纹变化拒绝，返回深拷贝防止调用方污染。缓存最多4份，不信任磁盘自称PASS作为首次验证替代。测试执行23次（含导入基类导致10项重复；13项唯一用例），全部通过，覆盖原篡改/缺域等及新复用/返回值变更/进程隔离。实际一图三行manifest首次34.97秒、相同成功proof再次0.044秒，当前CPU高负载下测得，不称GPU预测或训练整体加速倍数。
新增B0全量后续控制器330502，仅等待RGB COMPLETE并验证5135唯一新增图/64分片，再启动12个worker在GPU4/6/7（每卡4），CPU80–91；GPU5保留SAM。每个worker将同一分片的light与blur顺序处理，共享已验证proof，不重跑先前65图。当前等待状态不是预测完成；原生成进程继续不可变source，未热修改。证据outputs/action4_rgb_variant_B0_full5200_v001。


## 2026-10-26T07:45:40.067921+08:00 原生分片缓存导出与父权重消费修复
CPU导出控制器332859已开始逐个消费原生完成分片，不重复B0/SAM推理；按export_completed验证原生session/mask/prompt来源，独立构建variant association，最终产生原图＋light/blur分片组合的128unit integration配置。核查时已导出并关联5个分片，目录outputs/action4_sam_variant_scale64_cache_v001。全量RGB此时32/64分片完成，其余继续，GPU后续队列仍等待完整门。
已创建只读configs/model/action4_stage3_parent_registry_v001.json：实际加载并校验A015770/B09500的角色、结构、config identity、SHA与字节数；只供新阶段初始化，不改默认模型和部署selection。发现construct对普通SAM A0虽声明parent_bundle_sha256却只在corner分支传给严格加载器，已改为只要配置提供SHA就传入同文件句柄hash＋反序列化的已有加载器。实际错误SHA拒绝、正确封存parent构造与backbone冻结两项测试通过。此前实际检查已独立核验父文件SHA，未发现加载错误；修复是把这一合同落实到正常构造入口。


## 2026-10-26T07:55:20.187434+08:00 全量RGB闭合与CoTracker原生GPU接口
全量RGB COMPLETE已写入：73分片（旧1＋扩展8＋新增64），主控核对manifest字节和5200唯一图全集/15600行variants（含5200原图引用和10400派生图）。315917已结束，原STATUS最后循环残留61/64，终态以COMPLETE为准；确认进程结束后修正STATUS并记录原因。B0后续控制器330502已自动启动12rank，GPU4/6/7，未重跑已有65图。
阶段六原生adapter原先仅CPU loader，视频/query张量也固定CPU。新增显式GPU4–7/UUID/CPU affinity/显存配额/当前authority准入，模型严格核验官方commit和权重，video与双向query转到core设备；raw输出仍保留，无GT或CAD输入。9项CPU接口/拒绝测试通过。实际GPU4首2帧运行发现导出backward query直接numpy()的CPU假设，v001失败留存；修复detach.cpu后用新不可变v002重跑短窗，2帧1.168秒，4帧含负crop补边1.325秒，support分别36/16，原生输出数值有效，峰值490804736字节。旧CPU两帧raw与GPU工程query native坐标最大差0.001532px，visibility差约1.01e-5；此为数值实现对照，没有真实跟踪GT，不能当定位精度。查询角色明确engineering_input_not_material_seed，不进入PnP、材料binding或几何验收。证据outputs/action4_cotracker_gpu_bridge_v002。


## 2026-10-26T08:08:04.223387+08:00 阶段三真实小训练与完整eval启动
64图light/blur的16原生叶子及16缓存/association均已完成，原生与导出控制器已退出；最后STATUS残留15/16不代表未完成，终态读取COMPLETE。
发现普通A0工作树rank仍将deployment_eligible放在All8之前，新增显式full_all8_training_candidate_v1契约：worker持久化契约，普通A0训练候选按原All8排序；困难组eligible继续单独输出。旧无契约历史结果语义不改，corner仍不能绕过质量准入。3项排名测试通过。
复用stage1_prepare/run_snapshot/stage2_dispatch启动p2_a0_sam64_integration_v001，A0父15770、固定B09500、新optimizer，原图/light/blur规范选择＋50/30/20路由＋预测mask dropout/扰动。64训练图128单位，7次遍历预计21updates，step0/10/20/末次完整650eval；本轮不重复已有受控恢复实验，不做test或视频，不称正式80轮。训练GPU5/CPU92–95；两个eval在GPU4、6/CPU72–79；12个并行B0变体worker仍CPU80–91，另控制/主控27–29，总预算40逻辑CPU，实际分配约27。控制器347021、train launcher347045、eval347046/347047，已写DISPATCH。
本集成输出沿用原prepare生成的stage1_baseline目录前缀，run scope与本节明确属于stage3 integration，不是重跑阶段一；原正式阶段一结果不变。不可变源码runtime/action4_runs/p2_a0_sam64_integration_v001/RUN_SPEC.json，工作树后续改动不影响它。此时尚无训练完成或新精度结果，后续检查实际更新、完整eval、最佳候选和吞吐。


## 2026-10-26T08:14:25.620811+08:00 实际启动等待与后续材料接口边界
核验当前train子进程347062（launcher347045下）仍存活，约98%单CPU活动，尚在SAM数据准入初始化，未见首step，不因日志空或观察超时重启。新eval与控制器均存活；此时无新增效果证据。
后续材料流程核查：surface_snapshot v1严格绑定当帧B0 grid，StagedPipeline的video_context明确拒绝return_surface；当前材料模块仅支持此v1，不能把SAM维持框或track_id假扮成新的B0 grid。后续必须通过显式video surface身份扩展，并将track/session/generation与同帧mask关联保留，再接材料seed/lifecycle，当前未实施此扩展，也未宣称材料跟踪完成。
为避免后续阶段产物继续误落stage1_baseline前缀，原prepare新增可选output_group=stage3_a0_sam，仅A0允许，默认历史路径不变；只做语法解析验证，未改当前不可变run或移动已有产物。未来正式stage3使用此组选项。
已在确认旧SAM/导出控制器终止且COMPLETE存在后，将其顶层STATUS修正为终态引用，避免15/16残留循环状态误导；原生PROGRESS、缓存绑定和模型结果未改。


## 2026-10-26T08:29:50.028964+08:00 视频ROI surface数据流实际闭合及训练首证据
新增surface_snapshot.video_roi.v1，保留原图片v1合同。视频surface经VideoROIContext当前RGB/K/模型/replay验证，记录track、frame、真实B0来源（有才保留）、SAM支持回执、native session与mask generation；sam_supported时grid和score必须None，拒绝伪造B0。project_track_generation尚无独立登记，明确null，不把mask generation冒充project generation。所有数组/上下文hash、ROI/K/canonical关系继续验证。
StagedPipeline.predict_video显式支持return_surface/surface_frame，只允许原baseline joint_pose，不把此功能称为SAM feature injection。CPU合同首次夹具误用numpy CHW，被既有RGB转换合同拒绝；改为实际provider采用的Torch CHW后15项通过，失败日志保留。将纯surface数据合同迁入src/data/action4_surface_snapshot.py，eval保留兼容导出，model材料模块改为依赖data，消除model反向依赖eval；函数实现未因迁移改变，相关21项回归通过。
实际冻结A015770/B09500对原视频4/1195/2400各两个实例执行默认与surface输出对照，六次backbone、零B0调用；原角点与pose结果逐位不变，六份surface NPZ保存重载逐位一致。1195两台均为SAM维持ROI，metadata明确grid=None且mask generation2；4/2400保留各自真实B0 grid及当前session，未重跑SAM。证据outputs/action4_video_surface_actual_v001/COMPLETE.json。此为真实模型数据消费，不是材料绑定/几何精度验收。
复用六份实际surface做未改阈值的上界诊断：第1195帧track00 visible probability最大0.617，低于0.7，尚未叠加SAM/mesh/spacing就已经0个候选；track01上界49。第4帧分别1197/1094，第2400帧265/126。这些不是最终材料数，说明困难帧无法保证重新seed，后续应从清晰帧建立材料并通过实际lifecycle/carry验证，不降低阈值伪造成功。
阶段三实际STATUS/train.jsonl已有step3、epoch1完整128单位、零监督跳过0；末更新hidden69/visible187、sam quality27有效监督。step0完整650图/1300实例All8@2%=701/1300，与封存父一致；训练后精度未返回，不以loss下降代替改善。空train.log不能作为未更新依据，后续以STATUS/train.jsonl和eval done读取。


## 2026-10-26T08:47:20.174611+08:00 真实供数profile与重复验证改进
阶段三仍运行旧不可变源码，核查时已到step9/epoch3；不根据空stdout判停。三个DataLoader PID352972/352974/352976持续工作。另对同一8图真实variant provider构造两次进行cProfile（无模型）：15.804/13.686秒，主要为重复PNG解码、两次重复原生receipt验证、每次解析全5200条标注。没有把猜测当作磁盘/GPU故障。
新增进程内成功variant pixel proof，绑定实际文件SHA/stat和完整预期entry，首读仍验证PNG格式/像素、任何变化失效；标注索引按PID/路径/SHA/stat缓存完整解析结果，只深拷贝当前分片记录，所有原文重复ID仍拒绝。实际同分片构造12.649/2.810秒。
发现collection原LRU会连同已验证metadata一起淘汰，随机跨分片反复重建provider。改为保留声明分片的轻量provider元数据，只允许两个hot provider保留解码frames；每次选择仍检查index和association字节身份，不缓存可训练输出。实际三分片首次11.221/6.881/6.936秒，再回第一分片0.000417秒，复用同一已验证provider。不是整个训练加速倍数，首次成本仍存在，当前训练未热修改。
29项验证复用/实际association/export回归及4项collection测试通过。首次新增测试脚本换行转义语法错误发生在执行前，已修正并保留日志；另两项旧测试要求“全部不准入”错误文案，实际仍正确拒绝缺失监督关联，更新为当前功能的“requires verified”断言，未取消负例或放宽授权。正式训练前须以新快照确认实际单位/更新等价及吞吐。证据outputs/action4_sam_provider_startup_profile_v001。


## 2026-10-26T09:04:32.267061+08:00 全量B0闭合与注册并行SAM队列
剩余5135原图×light/blur的128个B0叶子全部完成，主控核对全部PREDICTIONS字节及两variant各5135唯一图且同域；各发布10270候选只是候选数量，不是GT匹配准确率。原65图保持复用。B0旧STATUS仍10/12，终态以COMPLETE及ROOT_COMPLETE为准。
统计已完成64图的256个SAM提示：peak reserved最大6352MiB、peak allocated约5908MiB。依据实际值设置每worker8GiB配额、启动free至少16GiB（保留8GiB余量）；r102登记10个独立slot，GPU4/6各3、GPU7共4，每slot2CPU（总20），排除当前训练/评估CPU，GPU5留给训练。每卡SAM配额24/24/32GiB，加GPU4/6已有eval8GiB及8GiB余量仍在卡容量内。新增登记严格匹配和内核flock lease，避免同slot两个活跃模型；5项注册/互斥测试通过。旧无slot原生启动在注册池生效期间拒绝，不能绕开池占用。
控制器373714已启动两个真实新分片canary（rank0 GPU4、rank1 GPU6），不是重复旧数据；两叶完整完成后才启动其余8slot。源代码与每叶配置独立冻结，逐叶继续保留来源/关闭回执，失败不得发布空成功。目录outputs/action4_sam_variant_full5200_v001；当前只确认canary调度，未称全量SAM完成。现有A0集成训练和eval不停止、不改源码。


### 2026-10-27T13:19:32.737233+08:00：SAM集成闭合、频率更新与六面失败定位

SAM四卡集成28次实际更新/2轮及0、25、28步完整650 eval均完成，step25恢复成功；All8@2%依次899/879/881，best仍0，不能报精度改善。当前正式80尚未启动；首25更新请求发布时间差91.759秒包含loader/保存，compute合计29.126秒不是训练全墙时。16并行分片650推理36.341秒尚未计正式评分/训练争用；正在单持久进程测两轮正式650评分服务，不重复原生SAM推理。

按用户新要求已接入epoch31从25切到10有效step的统一调度，trainer、controller、observer和不可变封存检查共用实际日志。55项相关CPU检查通过，见outputs/action4_stage3_eval_schedule_v001；这是代码证据，非80轮完成。SAM封存补齐adapter strict load、332个冻结backbone tensors/buffers逐项比对，工程bundle15 adapter tensors加载通过，integration不能冒充正式seal。

六面前视808候选相对raw旋转26.409度、深度仅-0.669%、宽增加34.151%；圆盘15度轴向反例轮廓集合差0.006767px而角点最大79.727px，说明该轮廓不足以锁定整机转角。修复了未经审核区域被无条件full_visibility=true升级的问题；partial原始边缘仅单向research候选，正式final保留raw。27项CPU测试及808回放见outputs/action4_front_region_rootcause_v001，不能称角点效果已修复。下一项定点验证为M3T原生RGB纹理对应的复用可能，用户允许原视频开发；不以demo当认证GT。


### 2026-10-27T13:36:03.253303+08:00：M3T机制定点复用及真实反例

官方DLR M3T固定commit f02106186a1e756450226b3b793e309506817c58、MIT源码定点存入upstream；只参考RGB region+texture联合刚体机制，没有宣称native M3T已集成。复用现有OpenCV、Blender BVH及精确交点：808–822固定15帧×2实例保留30条。track00 ORB参考210点，14对共334几何内点且在lens区域之外；track01参考9点，后14帧不足，未造结果。

808固定CAD首交点绑定177/210及8/9，后续不重绑，XYZ_source明确A0_estimated_keyframe_CAD_raycast。独立EPNP在812出现明显错误（相对raw角点差218.82px并非GT误差）；现有partial region+typed固定纹理联合8/15数值合格、7触界，812仍拒绝。所有final_corners保留raw。不能将更接近raw、低重投影或更平滑等同精度。主控审核代码与真实结果，补tiny-residual稳定计算，相关18项CPU测试通过，见outputs/action4_face_texture_probe_v001/ROOT_REVIEW.json。下一步是成熟区域轮廓可见性与纹理联合观测的进一步核验；B0+SAM的全片身份准确也未由这些短片证明。

分段eval规则的恢复日志补齐：checkpoint保存有效update ledger，恢复不把未提交的旧日志尾巴重复计数；原train.jsonl保留失败历史，最终ledger SHA绑定TRAIN_COMPLETE。57项调度/封存CPU测试通过；正式SAM80无零跳过时1346次完整650评估，共874900图次。


### 2026-10-27：目标16完整eval容量实测未通过

16×650共10400次预测、全部正式scorer完成且16份报告完全相同，0.0258765checkpoint/s，低于按集成参考产生率加20%后的0.0326943。全体墙时618.321秒，不剔除最慢worker；未同时训练，因此不能声明联合负载通过。未启动SAM80，未重跑原生SAM或更改默认权重。当前用一次头32图性能归因定位确定性输入处理瓶颈，按证据优化而不继续盲目加进程。证据outputs/action4_stage3_eval_capacity16_v001。


### 2026-10-27T14:07:24.207583+08:00：确定性加速通过完整复验，独立SAM80已派发

性能归因找出每ROI全原图常量valid构造/isin/float转换的冗余。复用完全相同float32 grid和nearest/zeros/align_corners=False内核，改4字节零stride常量输入，并每ROI共用grid。真实32图64ROI原/新全部输入、backbone/head和最终预测严格相同；同计时准备10.0417→5.0837秒。主控独立29项ROI测试、15项训练/冻结上下文消费测试通过。未启用备用prepared输入缓存。

目标16×650原计分复验v002耗时478.981秒，服务率0.0334042/s，预登记门0.0326943/s，超过门2.1714%；16份报告/原始预测全部与v001字节相同。这是评估服务门通过，不是联合训练实测或精度改善，仍保留max_pending20背压和真实积压监控。

已从实际sealed A0 step1450/B0 step9500构建run p2_a0_sam_prediction15_formal80_v001，不可变RUN_SPEC共1393文件。控制器1399656、trainer1399714、16eval进程实际存活；启动检查无failure，仍在初始化，尚未确认optimizer更新。4卡GPU4–7训练+各4eval，正式65CPU(31–95)，其他工作仅24–30总7，禁止GPU0–3。按80epoch/10400预测ROI、ROI1.5、global48、每rank micro2、前30轮25step/后续10step完整650执行；旧baseline不追加/default不覆盖/test不消费。active authority已更正为r109，旧r107顶层历史baseline“training”描述不再冒充当前状态。30min action4-sam监控已更新。

六面复用另比较了单应性预过滤与直接描述子对应进入局部刚体联合拟合；两路仍各8/15数值合格、7触界，无精度准入，见DESCRIPTOR_JOINT_REVIEW.json。不因成熟方法名、纹理匹配或更小raw差就声称全视频六面跟踪成功。


### 2026-10-27：首次正式有效更新与后续接口缺口

SAM正式run已核实step221/epoch2，10400 units四rank均存活，损失/gradient有限、无failure，不再是仅初始化。后续按30min监控。六面[11,12,13]单变量对照结果为负：30记录保留，joint1/15数值合格、14触原tz=-0.6界，未采用。证据outputs/action4_front_region_lens_assembly_v001；该负结果说明仅改mask/CAD部件对应仍不能解决固定纹理与区域冲突，不调阈值凑通过。

阶段四只读审查实证：现parent input guard未覆盖SAM-native父模型，corner predict默认2.25且未走当前冻结B0预测回放，存在重新引入训练/eval gap的风险；正在工作树注册独立corner-native合同、显式ROI1.5与冻结预测路径，正式gate保持关闭。当前SAM80消费不可变snapshot，不受这些修改影响。原生M3T Python封装/构建准备另行受限CPU进行，不宣称已集成；现系统pkg-config未找到Eigen/GLEW/GLFW/OpenCV开发包，不做全局安装。


### 2026-10-27T15:45:56.996915+08:00：30分钟实际训练/eval复核

正式SAM80已到step2182/epoch11；controller、trainer及16eval命令身份已核查存活，完整eval已完成80次、待8次、失败0，已发布请求0..2175按25有效step完整连续。当前best step1875，All8@2%=947/1300（72.846%），相对同CPU采样step0 899/1300增加48个、3.692百分点；FP/FN各353，合法匹配P95从16.210到15.202px。high_occlusion53/192、small49/127、near_pair12/20、small_high_occlusion1/40。此为合成开发集中的阶段性改善，不是80轮完成、最终封存或真实六面准确性通过。

后续证据已落盘待主控完整闭合：完整650图质量sidecar真实复用32+新forward618，未校准；原生M3T软件渲染及15帧运行已通，初始化敏感性仍明显，4–18两seed在frame18角点分歧约126.89/119.47px（不是GT误差）。两个子智能体后续因服务额度限制退出，未把退出视为服务器训练失败，不重启训练；后续校准/最终审查由主控继续，不派新agent绕过。


### 2026-10-27T15:52:28.257069+08:00：质量sidecar与旧校准复用闭合

发现并纠正重复计算：outputs/action4_refined_box_quality_calibration_v001已有同B09500、同650图的完整1300质量分数和校准，根逐行核对与新sidecar原base及q完全一致。因此最近新增618次模型前向本可由旧产物迁移避免；保留事实与成本，不再重推/重搜阈值。本轮0模型调用，以旧校准0.5为来源，新增显式等价桥接，把新sidecar接入native角点输入消费。

完整CPU诊断：q相对max-nonignore完整框IoU的MAE0.02068135、signed_bias+0.01791306、Pearson0.69407529；这是回归诊断不是一对一检测指标。预先限定{.4,.5,.6}全选1300，三者TP1300/FP0/FN0、双机650/650相同；空图/单机/真实衣物特异性未测，不能据此证明quality分数可泛化拒绝错误物体。复用原校准0.5，不新签一个扩大范围的gate。

新增src/eval/action4_corner_quality_bridge.py与diagnostics模块，实际消费时验证当前RGB/K/base/model，保留原B0字段，旧capture→新sidecar逐值等价后才使用原开发阈值。28项桥接及原合同测试通过；实际650来源验证+第一图2实例消费通过，改1字节RGB拒绝。接口是可选启用、未绑定任何正式stage4模型、stage4训练gate保持关闭。证据outputs/action4_refined_quality_development_review_v002/{REPORT,QUALITY_BRIDGE,BRIDGE_ACTUAL_CHECK}.json。当前SAM80不可变源码/数据/default均未改变。


### 2026-10-27T16:02:57.557450+08:00：当前SAM支持进入原生M3T纹理观测

独立原生构建runtime/m3t_support_v001复用已有依赖/软件renderer，原native二进制未覆盖。新增opt-in当前support按同frame原生像素域筛选descriptor/keypoints；数据侧认证SAM frame/PTS/RGB/generation和mask摘要，两个实例共享像素从纹理观测中剔除而非合并实例，空mask不回全图。原生clone输入防外部alias，过期frame返回false。真实接口全mask180keypoints、空mask0、stale拒绝；最初cv::Point2f Python转换失败保留，显式tuple caster修复。

同808–822、两实例共30记录开关对照完成，无native调用失败。关闭support30个pose与原版逐值相同；开启后5320个候选keypoints保留2390，逐点核验在各自当前support中，17个pose发生变化，证明信息实际进入原生求解而非仅渲染。原RGB、初始化、model cache、参数不变，无SAM重推。根检查CONTACT的808/812/818/822：track01方向/范围仍不准，不能宣称精度改善；raw final与default保留，region模态仍未消费SAM，六面正式准入不通过。证据outputs/action4_m3t_texture_support_v001/{NATIVE_SUPPORT_TEST,PAIRED_NUMERIC,ROOT_REVIEW}.json。


### 2026-10-27T16:15:17.716330+08:00：120帧连续原生跟踪暴露真实失跟

已执行注册4–123共120帧、两实例240记录，只frame4初始化，后续119次真实native tracking更新；当前SAM纹理支持240次实际消费，保留关键点49089/56294，native调用失败0，raw final不变。根实际查看4/18/40/65/90/123：前段尚能靠近机身，转腕后末端红框漂至手部上方而当前B0仍落机身，效果拒绝。track00在116帧native/B0 IoU首次<.1且仅3纹理点，track01在120帧<.1且仅2点；这是预测分歧诊断不是GT指标。区域分支缺失跟阻断，不能只靠mask筛选持续更新。

严格几何接口162条拒绝来自native float32旋转累积超过已定表示预算，不放宽门；原矩阵和错误逐条保留。首次v001原生环境无Python cv2在0帧退出，原失败保留；v002复用15张已存BGR数组并一次解码其余105张，4/18像素与旧输入逐值一致，无训练env修改/无SAM重推。新增src/eval/action4_native_tracking_audit.py校验完整有序帧/ID/PTS/RGB消费、仅首帧初始化、支持域和raw保留，实际15帧正例及漏帧/重复/过期support/暗中重置反例均通过。证据outputs/action4_m3t_support_continuous_4_123_v002/{STREAM_AUDIT,DRIFT_LOCALIZATION,ROOT_REVIEW}.json。下一必要修复是在已有sequence controller增加当前证据失跟/恢复，不能每帧用A0硬重置刷平滑。


### 2026-10-27T16:29:07.378060+08:00：原生跟踪当前证据恢复真实对照

新增model/action4_native_recovery.py，固定开发触发：当前B0/SAM存在，native与B0框IoU<.1且纹理点<6连续2帧；合法当前A0正深度/刚体、投影框IoU≥.3、mask落B0内≥.5、score≥.3方可reseed，冷却10帧。此片段已看过，明确开发策略，不冒充独立验收。只重启所选body的region+texture，renderer按新pose刷新；另一body pose不改、跨body/过期support原生反例拒绝。5项状态测试通过，缺Python package export首个接口失败保留修复；现有正式训练snapshot不动。

4–123共240记录、119原生更新完成，仅track00@117、track01@121发生恢复。首次恢复前226记录与旧无恢复pose逐值相同，记录/事件证明真实消费当前A0 seed且非逐帧重置；源/ID/全部raw final保留。新增审计拒绝暗中reset及错误seed消费反例。根实际查看116/117/120/121/123：恢复后候选回到机身附近，旧框仍漂在手部。第123帧与当前B0预测框IoU由0/.076到.626/.740；这是预测一致性和视觉恢复证据，不是GT角点准确率。152条数值表示门失败保留、不放宽阈值，正式最终效果尚未准入。证据outputs/action4_m3t_recovery_v001/{STREAM_AUDIT,ROOT_REVIEW}.json及RECOVERY_CONTACT.jpg。

定时训练核验：step3077/epoch15，124次已提交完整650eval、116完成、8待、失败0；best2725为953/1300，比前次947增6，但high_occlusion49/192低于前次53/192，near_pair13/20、small_high_occlusion1/40，保留退化；非最终训练完成/真实视频验收。


### 2026-10-27T16:37:26.092376+08:00：原生候选接入现有ROI视频保存链

新增eval/action4_native_candidate_bridge.py并在现有action4_roi_video_job按序合并后、只读渲染前加入显式可选hook；不另建图像推理框架，不改当前已运行snapshot。真实2850帧基线与120帧240条native恢复结果完成关联：检查当前frame/PTS/RGB、唯一track、A0实际raw pose/corners、B0原框、ROI配方、模型step/config、K/CAD及0–7投影一致，withhold/数值失败不会标可用。移除候选字段后全部2850帧逐对象与原stream相同，原corners_final不变。

12项candidate/recovery/support CPU测试通过；首次生成源字符串换行SyntaxError在任何流读取前失败，已修正并保留工具错误，实际运行后产生完整ATTACHMENT/COMPLETE/ROOT_REVIEW。该接口只是经核验候选附着，尚未准入或生成新全片视频。源outputs/action4_native_candidate_bridge_v001。原生float32矩阵漂移继续定位：Link::UpdatePoses在float矩阵上重复exp与乘法且不重投影SO3，造成累积非正交；应在独立native版本修复数值状态保持后验证，不能降低现有几何有效性检查。


### 2026-10-27T16:53:26.094243+08:00：原生刚体数值保持修复及真实反例

独立runtime/m3t_rigid_v001新增默认关闭的Link刚体保持，仅独立刚体支持，保持原始theta/平移，按原32eps输入近刚体及16eps元素修正预算做doubleSVD最近proper rotation，再castfloat。反射/尺度/剪切/nonfinite fail closed，不降低原SO3/正深度门。实际Eigen反例发现Affine.rotation会先去掉1%scale，改用raw.linear检查，旧构建未用于真实片段；Link.SetUp测试夹具顺序错误亦保留后修正。

10000次真实native helper旋转，最大正交误差5.9866e-8，原float累积6.8690e-5；actualLink正常seed通过、1%scale拒绝。4–123两臂各240记录：关闭240/240与旧恢复版逐值同；开启后数值拒绝152→0，最大正交误差2.5020e-7、最大det误差2.4040e-7、每link1191次投影，单次最大元素修正2.3784e-7。所有帧/currentSAM/两ID/原final保留，无native失败。

效果分开：微小数值改变影响后续局部匹配，恢复从track00@117/track01@121变为两者@120。根实际查看40/90/117/120/123：117时track00新版本仍漂在手部，未证明效果更佳；保留旧结果，新版本仅数值通过不正式发布。证据outputs/action4_m3t_rigid_state_v001/{NUMERICAL_TEST,PAIRED_RESULT,ROOT_REVIEW}.json及PAIRED_CONTACT.jpg。

另复现RecoveryController在B0缺检但有当前SAM纹理支持时无条件withhold的问题；新增默认关闭的allow_sam_only_continuation，显式开启时保留当前支持连续性、禁止用缺失B0做reseed，空/弱SAM仍withhold，不受“纯外推”时限误杀。14项CPU回归通过；这项独立开关未混入上述两臂对照，也未改正式模型或视频配置。


### 2026-10-27T17:09:22.795727+08:00：关键点代理与真实优化对应的区分

原生read-only observation_audit导出最后线性化实际data_points的观测/投影像素、残差、variance及bodyXYZ；权重按原公式复算，仅诊断，不声称逐bit读取gradient累积标量。原目标函数在20px外仍有非零cost/error²权重，本轮未偷偷删除这些点或更改优化目标。关键点数还可能在CalculateResults刷新关键帧后变化，因此不能当本次优化对应数。

实际接口同帧274keypoints仅253对应，空支持对应0、读接口不改pose、像素差与残差一致。新read-only版本keypoint控制臂240/240位姿与前一rigid版逐值一致。两个120帧臂都保留240记录、119native更新、0数值/调用失败；恢复支持改为真实匹配在原20px尺度内计数，未改其他阈值。第115帧track01仍32keypoints但实际匹配0；匹配臂track01恢复120→119，track00仍120。根实际查看119/123：第二台119提前回到物理机身、另一台仍漂移，未证明全片六面效果通过。证据outputs/action4_m3t_observation_audit_v001/{PAIRED_RESULT,WEIGHT_SEMANTICS,ROOT_REVIEW}.json。

最近30min训练检查step3776/epoch18，完整eval已完成145、待7、失败0，best3450 All8@2%=970/1300，比前次953增17；仍是合成开发成绩，正式80未完成，不生成整片视频或默认发布。


### 2026-10-27T17:20:49.674800+08:00：有限恢复阈值对照与原生PNG输入复用

提前登记仅比较框分歧IoU .1/.3，已有.1匹配计数臂直接复用，新.3保持所有模型、匹配尺度、连续2帧、seed资格与10帧cooldown。120帧240记录、0调用/数值失败；恢复提前到track00@114、track01@115，但根查看114/115/119/123发现提前接续后再次漂移，123明显不如.1，候选拒绝，不自动选更早触发、不扩大盲目阈值搜索。证据outputs/action4_m3t_early_recovery_v001/ROOT_REVIEW.json。下一重点是区域模态尚未消费当前SAM支持，而不是继续微调恢复数字。

为后续片段避免重复展开2.88GB/120帧NPY，独立native311环境增加项目内runtime/m3t_image_io（Pillow12.3.0，与主环境同版本但兼容311的独立wheel），未导入错误312扩展、未修改训练.venv。新data/action4_native_png_io.py读取同句柄PNG bytes先SHA/尺寸核验再BGR解码。实际120图与已存BGR数组逐值相同，拒绝错误source/shape，0模型调用/0新增NPY；42.96秒为读解码加旧数组对照总耗时，不当作单图推理提速。证据outputs/action4_native_png_io_v001/{COMPLETE,RUNTIME_IDENTITY}.json。


### 2026-10-27T17:43:41.930256+08:00：SAM区域特征独立输入真实试验（负结果保留）

独立region-camera将当前原生SAM可见mask编码为0/255三通道特征，原RGB纹理输入不变，K/相机pose严格相同。原生RestartSupportedBody只有显式aligned_aux_camera并且K/pose相等才接受不同camera，错误K/pose反例拒绝。两路buffer分别owned，240条辅助图像前后checksum通过，原mask不裁边/不补amodal/不造概率。

真实gradient消费检查：固定当前RGB与seed，仅将辅助mask平移20px（明确counterfactual测试，不伪称SAM预测），区域gradient改变、纹理gradient及pose不变且全finite，证明输入实际参与区域求解。额外RGB双camera15帧控制30/30pose与旧版逐值相同。120帧SAMregion分支0数值/调用失败，仅track00@120恢复一次，但根看40/90/114/119/123：track01最终仍偏移扩大、比原RGBregion已恢复机身的对照差，拒绝采用，不能用恢复次数下降当效果提高。证据outputs/action4_m3t_region_support_v001/{GRADIENT_CONSUMPTION,PAIRED_RESULT,ROOT_REVIEW}.json。

当前可见mask不等于认证的完整机身轮廓，这一限制在native二值区域输入中仍存在。后续应检查当前A0姿态能否通过有界不确定性先验参与每帧几何，而非只init/lost时使用；属于待验证下一步，不冒充已接入。保持正式SAM80 snapshot、data和models/default不动。最近定时状态step4660/epoch22，187eval已提交/181完成/6待/0失败，best4375 All8@2%=988/1300，仅合成开发成绩。


### 2026-10-27T18:04:24.837265+08:00：用户要求固定面ID，实体与会话分离

现canonical8/六面拓扑原本已固定；检查发现FaceSeedBinding.track_id包含face_generation，实际是会话代次ID。新增persistent_face_id仅绑定已登记body_scope/body_identity_generation/face0..5，session_track_id保留旧四元组hash，track_id别名和fingerprint不变。FaceInputs新增六个稳定实体ID元数据，未登记机身不伪造ID；两台真实seed构造12个唯一逻辑face实体。16项新/旧face-input/session合同测试通过。遮挡、frame变化、面会话重建不换实体ID；真正机身身份代次变更仍另记，不能把未知重识别硬认作原机身。语义面定位尚未通过，固定ID本身不证明位置准确。证据outputs/action4_persistent_face_identity_v001。

并行当前A0prior诊断发现工程调用错误：fixed anchor Link只SetUp未初始化Jacobian，故v001两臂120帧均native调用失败；原结果保留，不能算先验效果。v002用零DOF固定anchor专用Optimizer.SetUp只初始化（不加入tracker、不优化target），实际原生单位检查通过；新诊断遇任何native工程失败即停止后续帧/下一臂。当前v002限定120片段已启动，不影响formalSAM80。


### 2026-10-27T18:16:13.551567+08:00：固定实体ID实流验证及当前A0先验对照

两台frame4真实FaceSeed的scoped body均属于同sequence_identity，逐源SHA/seed/prompt/body对应验证后形成REGISTRY。实际120帧240记录导出12个persistent_face_id/16个persistent_corner_id，跨恢复不变；几何无效明确unavailable，合法六面标inferred_from_body_pose，directly_observed=false，不把固定ID当真实物理面分割。产物outputs/action4_persistent_face_identity_v001/{REGISTRY,ACTUAL_120_ID_CHECK}.json及LABELS_120.jsonl。

复用upstream M3T SoftConstraint已实际接入：每个可优化刚体以固定零DOF虚拟anchor接收当前A0，角度rad/平移m，先验为相关图像预测非GT。单位检验正确向+10mm目标移动2.5mm，改-10mm目标后方向反转，关闭方向不更新，anchor不被优化。v001因anchorJacobian未初始化全部调用失败保留；v002单独初始化固定anchor后两臂各240当前packet/119trackingstep、0调用/数值失败，且固定anchor全过程不被solver改写。弱(.1rad,.01m)与强(.035rad,.003m)名义正则尺度均已预登记，不冒称真实校准不确定性。根看90/114/119/123：两臂未解决快转失跟，strong第二台123仍上漂，不采用。证据outputs/action4_m3t_a0_prior_v002/{PRIOR_UNIT_RESULT,CONSUMPTION_AUDIT,ROOT_REVIEW}.json。后续应检查错误区域观测的作用权重与当前支持，而非无限增大A0先验。

18:15定时状态：formalSAM80 step5672/epoch27，227eval提交/221完成/6待/0失败，best5425 All8@2%=997/1300；high_occlusion65/192、small_high_occlusion2/40，仍属合成开发，训练未完成。第31epoch将按冻结配置切为10step，重点监控真实积压。


### 2026-10-27T18:36:33.940113+08:00：当前证据约束区域梯度，快转片段获得有限改善

新增RegionObservationGate复用原生n_lines_max：当前B0+SAM且合法当前A0先验与native框IoU<.3时停用区域线，≥.5再启用，滞回中保持状态。不删除验证，真实native反例证明0线使region gradient/Hessian全0、texture目标不变，恢复线预算后区域目标逐值恢复；空临时histogram在既有learning_rate<1路径保留历史统计，若显式reseed则先恢复线预算重建当前histogram。

单变量strong-prior父对照120帧240记录完成，仅track00@114/117和track01@117/122共4条停用，0硬reseed、0数值/调用失败；每条停用的实际梯度和Hessian都核对为0。根看90/114/117/119/123：快转阶段比父对照更贴近机身，但角度/范围仍有偏差，属于有限视觉改善候选，不正式发布。证据outputs/action4_m3t_region_gate_v001/{CONSUMPTION_AUDIT,ROOT_REVIEW}.json。

扩展输入核查发现真实frame186–188 track00的score和bbox_b0_raw均None，ROI来源SAM_tracking_geometry、A0仍valid；原短片runner对bb无条件索引将失败。新增eval/action4_native_recovery_inputs.py保留缺失B0及不可计算overlap为None，不把SAM框伪装B0。实际3帧mask/source输入验证及控制器分支检查完成；其中用于分支的20支持数明确为工程夹具，尚未执行这些帧原生tracking，不冒充实际匹配数。证据outputs/action4_native_sam_only_inputs_v001。下一步需把helper用于更广片段消费者。


### SAM增强视频输入与surface输出实际检查（2026-10-27T19:30:09.413274+08:00，进行中）

新增并接入 src/eval/action4_video_sam_inputs.py；现有predict_video与run_part消费同一已认证当前帧上下文。实际第4/186/232帧验证了RGB、ROI、K、mask和实例身份；第232帧两台都无当前B0检测，仍通过各自合法SAM跟踪ROI调用A0，共两实例，B0 forward次数0。工程step2的mask开关对照改变有效C5及角点输出，缺mask准确回退RGB C5；这是信息消费证据，不是精度提高证据。

第232帧两份增强surface快照严格保存重载通过，导出前后实例预测完全相同；模型/机身/当前帧/RGB/ROI/K/PTS/time_base与SAM来源均绑定。0–3帧真实输入和缺mask路径通过，并以第0帧实际前向及surface导出检查。共32项相关回归通过。保留原缺测试模块失败日志；新增时间验证发现旧VideoROIContext仅在登记机身时保存time_base，已修为保留所有明确提供的time_base，随后回归通过。

证据目录：outputs/action4_video_sam_feature_bridge_v001，主要回执ROOT_REVIEW.json、BOTH_SAM_MODEL_CHECK.json、surface_frame232/CHECK.json、PRESEED_CHECK_v002.json、FINAL_REGRESSION_v003.log。正式80轮源码及models/default未改；尚未用最终SAM best运行完整视频，六面全视频准确性仍未验收。

19:28:11现场快照：正式训练step7339/epoch34；343次eval已请求、324完成、19待完成、0失败。选优best7130，All8@2%=1005/1300，上一记录1002；高遮挡64/192，小目标且高遮挡1/40。仅合成开发集结果，不能外推真实视频达标。四GPU训练及16eval worker继续运行；CPU正式31–95，主控/接口24–30，未派子智能体。


### 决策ROI驱动的六面原生跟踪开发片段（2026-10-27T19:47:05.199524+08:00）

原生区域/纹理/当前A0先验流程已从历史实验脚本迁移至src/eval/action4_native_clip.py，薄入口scripts/action4_native_clip.py；输入准备及只读复核同放src/eval。实际4–6帧迁移检查的6个姿态与旧runner逐值一致；不重复生成模型或RGB NPY。相关门控与输入检查21项通过。

冻结源码下并行执行180–299和1180–1299两段，各120帧、240实例记录，CPU分别25–27、28–30，正式任务65CPU和主控24合计预算72，无新GPU模型任务、无子智能体。两个片段各119次实际跟踪更新，均0调用失败、0数值无效、0硬重置。决策完整ROI参与region可靠性门控；原B0框缺失合计256实例帧仍全部具有原A0输出，其中111次满足当前SAM纹理接续条件，其余六面候选不足的情况保留原A0，不等于整条链丢失。

12面ID及16角点ID持续登记；它们来自固定CAD拓扑和同一机身位姿，不是12份直接观测。已查看180/186/232/299、1180/1195/1289/1299同帧对照：两台ROI保留，原生候选仍有朝向/厚度/范围差异，没有证明整体更准确，故未替换正式输出。原生框与决策ROI的中位IoU分别约0.612/0.592，原A0约0.652/0.700；这些仅是预测间一致性，不是GT准确率。

证据：outputs/action4_m3t_broader_clips_v001/ROOT_REVIEW.json，两个子目录的TRACK_COMPLETE.json、ENGINEERING_REVIEW.json、PERSISTENT_ENTITIES.jsonl、PAIRED_CONTACT.jpg。近邻段还运行时曾提前读取完成文件而得到FileNotFoundError，属于观察未完成，不是任务失败；未重启，等真实进程exit0后完成复核。正式80轮与默认模型均未改变。


### 原生纹理对应与鲁棒权重有限对照（2026-10-27T20:05:09.987517+08:00）

实际记录显示：180–299片段有201个非空纹理实例帧，其中75帧没有20px尺度内对应；1180–1299片段为198个非空实例帧，其中50帧没有尺度内对应。对应并非充分独立观测；按当前估计CAD绑定作诊断，分别114/92实例帧没有尺度内唯一UV对应，至少3个估计面有对应的实例帧仅18/23；该映射使用估计XYZ，不能当独立物理面标注。证据在outputs/action4_m3t_broader_clips_v001/TEXTURE_OBJECTIVE_AUDIT.json和FACE_OBSERVATION_COVERAGE.json。

保留上游及原二进制，新建runtime/m3t_tukey_v001默认关闭候选。复用源代码、配置和原生构建环境；构建入口在src/eval/action4_native_robust_build.py及薄scripts入口。实际原生梯度测试253个对应验证了受控超尺度残差的权重、gradient和Hessian为0，关闭后原gradient/Hessian逐值恢复，标量有限差分一致；默认关闭时6个实际姿态与原二进制逐值相同。没有把上游公式差异直接断言为已证实错误。

候选两段共240帧/480实例记录全部跑完，0原生调用失败、0数值无效。与决策ROI的框IoU中位数由0.612→0.614和0.592→0.653；这是预测间一致性，不是GT精度。实际放大对照仍有混合效果，例如232帧track01相对机身上移，故未提升为默认、未替换原A0输出，也不继续围绕此权重无界调参。复核记录outputs/action4_m3t_tukey_derivative_v001/ROOT_REVIEW.json。

19:57训练现场：step7951/epoch37，完整eval请求405/完成388/待17/失败0；best7700 All8@2%=1018/1300，前次1005；高遮挡67/192，小目标高遮挡0/40。22个既有controller/trainer/rank/eval进程身份和存活已核验。正式冻结训练与默认权重未修改。当前关键剩余是最终SAM增强A0、角点分支及有效面观测约束，仍未完成全视频六面精度验收。


### 阶段四真实输入完整模型工程接续（2026-10-27T20:47:35.790526+08:00）

复用现有construct/optimizer/loss_sums/global count/step_if_supervised/bundle/load接口，以实际原生预测ROI1.5的训练单元0、1执行两次完整角点分支更新。分别测试epoch1/epoch6权重分支，不是完成六轮训练。16个角点参与corner监督，其中原pose监督13个可见、3个隐藏；corner KL到SAM projection梯度绝对和285.5052非零。冻结backbone参数/buffers未变；质量头预热时无隐式更新/optimizer状态，启用后更新；建立AdamW动量后全局零监督不改变模型/optimizer/scheduler；严格重载forward逐值一致，默认loader拒绝未准入候选。

v001在两次真实更新之后因检查脚本对非Tensor extra_state调用detach而失败，已保留；v002修正复制逻辑后全项通过。随后独立真实eval图p2_eval_00000通过，B0 forward0/A0 backbone1、两实例，quality_ready及deployment_admitted仍false。工程父模型为已有SAM step2，不能冒充当前正式80轮best，也未执行完整650角点eval。

实际发现并修复两处接入漏洞：完整eval前置检查未识别已验证的quality sidecar桥；配置与SAM producer引用的两份B0预测内容相同但路径不同被误拒绝。现在桥检查实际B0元数据，预测副本须两份实际SHA一致并保留双来源。实际全650/1300记录与别名组合preflight通过，缺桥/错误来源/错误模型仍拒绝；相关37项回归通过。

证据：outputs/action4_stage4_real_input_updates_v002/ROOT_REVIEW.json、COMPLETE.json、ACTUAL_EVAL_CHECK_v002.json；outputs/action4_stage4_quality_preflight_v001/REAL_ALIAS_PREFLIGHT.json及TESTS_v003.log。SSH曾超时/reset；核查真实进程和最终回执后继续，没有据此重启正式任务。阶段四正式gate保持关闭，仍需实际SAM80best、完整模型DDP/恢复、完整650异步eval吞吐及阶段准入。


### 阶段四完整模型Gloo及恢复实际对照（2026-10-27T21:24:25.366586+08:00）

复用已验证真实单元0/1，不再扫描全量SAM输入缓存。单进程micro1累积参考与两个实际Gloo进程对照，共4次数据尝试/3次优化更新；两rank在step1后严格恢复完整model/optimizer/scheduler/RNG。覆盖一个rank仅dummy但仍同步、全局零监督一致跳过、冻结backbone不变。

首轮v001两个rank完全一致，但与连续参考的少量角点参数超出既定atol1e-6/rtol1e-5，最大4.3245e-5，失败保留。独立v002仅显式冻结空间softmax中冗余的heatmap.bias并过滤optimizer中的冻结参数，未放宽误差门；最终训练参数最大差异0，optimizer moments/per-parameter steps、scheduler及全局counts/更新决定也逐值一致。它是数值训练合同通过，不是精度提高证据。

同时修复从corner父bundle新建训练时沿用推理loader冻结状态的问题，实际工程父模型验证除显式bias外其余角点参数可训练且各归属一个optimizer组。15项相关回归通过；旧默认配置及正式阶段三运行源码不变。证据outputs/action4_corner_full_gloo_bias_v002/ROOT_REVIEW.json、ROOT_COMPARISON.json、CONTINUATION_OWNERSHIP.json；v001失败记录保留。后续仍需实际SAM80best、四GPU集成与完整650 eval吞吐及正式准入。

21:06现场核查：正式SAM训练step9388/epoch44，eval完成532/请求548/待16/失败0；best9070，All8@2%=1022/1300，高遮挡67/192，小目标高遮挡1/40。22个既有进程存活，尚非训练完成或真实视频达标。


### 阶段四独立准入代码与收尾审计（2026-10-27T21:49:06.045868+08:00）

已实现src/model/training/action4_stage4_prediction_protocol.py并接入现有validate_run_gate，未创建正式放行文件。26项相关测试通过；实际已有baseline封存记录被正确拒绝为阶段四父模型。首次回归仅因历史测试要求错误信息前缀“Corner candidate”而失败，保留日志后恢复兼容前缀，拒绝行为未放宽。

控制器已有全eval完成但无合格corner时的明确终态和父阶段回退，无需重写。实际审计发现通用封存仍需补corner专属合同：不能对未准入报告调用corner rank；原始报告比较须保留corner完整不可变payload；strict loader须绑定质量/decoder之后显式加载corner候选；最终selection需保存整checkpoint分支和独立父回退。以上尚未完成，因此stage4_execution_gate继续缺失、正式训练不能启动；当前SAM80运行源码未改。证据outputs/action4_stage4_formal_protocol_v001/ROOT_REVIEW.json。


### 阶段四封存接口扩展与回归（2026-10-27T22:40:40.597786+08:00）

已扩展现有封存工具及src/eval/action4_corner_sealing.py，支持合格corner筛选、完整报告校验、实际校准域与数据manifest对齐、immutable worker重算选优资格，以及decoder/校准/reference/父回退绑定。初版按绝对路径索引evidence导致3个新元数据用例报错，修正为现有相对路径格式；失败日志保留。另保留历史不可变loader三参数调用，不向旧SAM/ROI加载器传新增corner关键字。

最终56项回归通过，包括baseline、训练候选、ROI、SAM和新的corner元数据/绑定反例。它们是工程合同及历史兼容检查，不是真实正式corner模型封存。未签发corner seal或阶段四正式放行，models/default和当前训练snapshot未改。仍需selection读取与部署分支连接、GPU完整集成/吞吐，以及真实训练完成后的独立封存与视频验收。证据outputs/action4_corner_sealing_preparation_v001/ROOT_REVIEW.json。

22:36正式SAM现场：step11273/epoch52，完整eval请求737/完成720/待17/失败0，22进程存活。best10680 All8@2%=1030/1300，与前次best10080相同，FP/FN均270；P95由13.90334降至13.82609，故按既定次级排序更新。两者deployment_eligible均false；当前仍为训练候选，不宣称真实视频或困难组通过。


### 封存selection到现有corner推理的连接（2026-10-27T23:21:56.363020+08:00）

新增src/eval/action4_corner_selected_inference.py并接到StagedPipeline显式方法；没有另建检测器、backbone或decoder。合同读取检查immutable source、权重、配置和终态身份，全局分支后处理只复制现有候选/原始结果；校准未成熟、模型/ID不匹配、实例缺失/换序、fallback被改写、非有限候选均拒绝。研究输出保留，最终已选输出使用独立schema。

41项相关回归通过；schema补充后8项针对性测试再次通过，数量不相加。实际baseline封存记录被正确拒绝为corner seal，模型前向0次。当前没有真实已合格corner seal，因此没有宣称正向正式模型部署验证完成；视频和test provider仍待接入，本轮未使用test、未改默认权重、未改运行中训练源码。证据outputs/action4_corner_selected_inference_v001/ROOT_REVIEW.json。

23:16现场：SAM正式step12108/epoch56，完整eval完成803/请求820/待17/失败0；best11820 All8@2%=1031/1300，前次1030，高遮挡69/192、小目标高遮挡2/40，22进程存活。此为合成开发结果，正式训练及真实视频验收尚未完成。


### 阶段四完整模型双GPU NCCL对照（2026-10-27T23:47:40.803046+08:00）

复用原单元0/1、固定预测ROI1.5及冻结bias配置，不重扫全量缓存。独立不可变源码下，单GPU参考与GPU4/5双rank NCCL完成4次尝试、3次实际更新，包含局部dummy及全局零监督；参数最大差异0，optimizer moments/step、scheduler、有效counts和更新决定一致。reference峰值分配0.876GiB，各rank1.177GiB，allocator每进程6GiB；CPU25–27、28–30，未用GPU0–3、未派子智能体，既有正式训练继续推进。

范围纠正：最初ROOT_COMPARISON的full_state_resume措辞过宽。实际是在同一进程内重建模型/optimizer/scheduler并恢复RNG，未验证新trainer进程从真实DataLoader游标继续；已保留INITIAL_COMPARISON_REVIEW并在当前回执澄清，不能据此宣布完整scheduled恢复或四卡正式集成通过。证据outputs/action4_corner_full_nccl_v001/ROOT_REVIEW.json。


### 评估并行与CPU调度实际调整（2026-10-28T01:44:18.257700+08:00）

已增加4个原版冻结eval worker：16/17/18/19分别使用GPU6/7/4/5，原16eval和四卡trainer未重启。三个安全退场子进程测试通过，包含commit异常不被隐藏；新增控制代码只管理资源生命周期，没有替换计算或提交函数。已核查额外worker的真实PID、原RunSpec消费、650图完整报告、当前claim和零失败。01:29辅助worker已完成16次完整eval，非挂空进程。

初始19worker短窗口未见明确提速：更新发布率约0.33574→0.33485 step/s，后窗仅418秒，不能宣称并行数增加就更快。15秒CPU时间采样显示trainer约3.26、loader23.74、原eval21.35、辅助3.06个逻辑核等效负载。随后补到队列上限20，并将辅助worker调为共享评估CPU池、nice19，训练CPU绑定和优先级不变；新调度的稳定收益仍待持续测量。

执行权威更新r112，30分钟自动检查已同步两个资源扩展及退场条件。证据outputs/action4_eval_acceleration_v001与v002，当前ROOT_REVIEW在v002。01:29训练step14792/epoch69，eval完成1072/请求1089/待17/失败0；best13180 All8@2%=1038/1300，高遮挡71/192、小目标高遮挡2/40。没有使用GPU0–3、没有派子智能体，也没有改数据/default权重。


### 2026-10-28 资源扩展收尾及完整650角点链验证（进行中）

原训练及16个完整eval worker未改动。辅助worker16–19完成各自在手任务并全部exit0，两个COMPLETE回执保留；资源登记更新r113，释放CPU25–30给后续准备，正式训练仍使用64–95。短时吞吐对照未证明增加进程提速，不报告加速百分比。当前训练step 16389、epoch 76，best仍以实际BEST记录为准。

阶段四完整650工程eval v001/v002/v003在实际推理前分别暴露runpy启动别名、Torch虚拟模块文件名、Torch动态生成代理与不可变源码检查的冲突；失败均保留，未重复650推理。修复采用精确来源及内容校验，真实项目外来模块仍拒绝。v004已通过预检并进入实际完整650推理，PID1684928；使用现有两次更新工程权重，不能当正式训练或效果验收。产物：outputs/action4_stage4_full650_engineering_v004。


### 2026-10-28 完整角点评估进度与耗时可观测性

在现有action4_image_guided_eval入口增加每25张及阶段边界的原子PROGRESS记录，分别记录读取等待、上下文准备和预测墙钟时间；不插入CUDA同步，不新增模型调用或哈希。记录始终complete=false，成功仍仅以原完整COMPLETE及评分产物判断。失败记录真实已完成数，不把失败图片计入完成。6项针对性测试通过，包含注入预测失败后无成功回执、来源校验回归。当前v004不可变运行未修改、未重启；此改进供后续快照使用，不能宣称当前吞吐已经改善。证据：outputs/action4_corner_eval_observability_v001/ROOT_REVIEW.json。


### 完整650运行状态表述修正

核查现有_execute：EVALUATION_STARTED在pipeline factory之前写入。因此，仅有该回执及存活进程只能确认预检通过后进入执行过程，不能进一步证明模型初始化完成或已有图片完成预测。此前“已进入实际推理”的表述证据不足，现修正为“实际进程运行中，初始化/预测内部进度未测”。v004不注入调试、不重启；新进度记录仅供后续不可变快照，最终以完整预测及完成/失败回执核对。


### 2026-10-28 完整650预测结束，评分ROI旧值缺陷修复

v004已保存完整650图预测和INFERENCE_EXECUTION_RECEIPT，进程正常结束于评分异常，不是成功完成：评分器两处仍固定2.25，导致实际1.5 ROI身份/224仿射及GT支持区域不一致。已让评分读取登记input_contract，并与真实checkpoint配置核验；历史2.25路径保留，错误比例拒绝。23项回归通过。新不可变p2_corner_roi15_rescore_v001复用原冻结预测在CPU25–30重评分，不运行模型、不改原预测、不覆盖原FAILED。实际结果待新COMPLETE/FAILED；工程权重仅两次更新，不能作为正式模型效果。证据outputs/action4_corner_roi15_scoring_fix_v001/ROOT_REVIEW.json。


重评分启动记录：v001因不可变launcher未登记新离线入口，在评分前拒绝；已显式登记eval.action4_corner_rescore并冻结v002。v002第一次启动错误地解析Python符号链接导致丢失venv、在导入numpy前退出，原日志保留；改用项目venv绝对路径后同一未执行快照启动成功，PID1694023。该进程仅CPU25–30，无GPU、无模型forward，真实进度见outputs/action4_corner_roi15_scoring_fix_v001/RESCORE_v002_attempt2.log。不得将失败启动写成已完成评分。


### 完整评分质量侧车合同接通

v002离线重评分通过ROI校验后，在完整框质量检查处拒绝：旧评分只认可B0原记录内联quality字段；现有推理已采用不改B0原记录的验证sidecar。评分现复用CornerQualityBridge及其RGB/K/实例/权重绑定消费接口，逐实例检查同一threshold、scope、passed、reason；同时校验checkpoint绑定的sidecar引用。不默认填质量、不修改原B0、不放宽准入。24项回归通过，覆盖错实例、缺侧车、错误门控拒绝。v003不可变CPU离线重评分PID1695660已启动；v002 FAILED保留，未重复模型推理。完整结果仍待回执。


### 2026-10-28 完整650角点工程链评分完成

离线v003完整评分完成，650图/1300GT/1300正式输出，无GT或prediction记录缺失。使用原工程两次更新权重；原始及候选All8@2%均676/1300=52%，All8@5%=93.6923%，FP/FN各624，无无效姿态，合法匹配角点P95=17.52175原图像素。质量已选点0、六可用ID实例覆盖0，quality_ready=false，全checkpoint回退raw正确，未发布best/default。该成绩不能同当前SAM80正式best混为同一模型，也不表明角点增强有效。推理未重跑；原ROI/sidecar评分失败保留。完整证据outputs/action4_corner_roi15_rescore_v003/ROOT_REVIEW.json。正式阶段四仍需实际SAM80封存父模型、完整scheduled DDP恢复与异步eval吞吐检查，通过后才启动。


### 阶段四准备入口接入结果

prepare_corner已接入现有stage1_prepare模块，7项CPU准入/准备测试通过，覆盖原配方及父模型不变、重复目录拒绝、缺正式SAM80封存时不创建输出。正向准备测试明确使用模拟父封存校验，未冒充真实正式父模型检查；未启动训练。真实SAM80封存和完整阶段四执行门仍须随后完成。证据outputs/action4_stage4_prepare_v001/ROOT_REVIEW.json。


### 阶段六非运行式准备：CoTracker冻结状态核查

原适配器只核对参数version，未覆盖子模块train/eval与buffer。现原生加载记录参数和buffer的名称、storage、version、device/dtype/shape；直接core调用前后检查全模块eval、参数无梯度及冻结状态一致。10项CPU接口/资源回归通过，实际官方权重CPU严格加载成功（187参数张量、1buffer），未调用模型forward。没有提前启动CoTracker视频或绕过六面验收；跟踪精度未测。证据outputs/action4_cotracker_frozen_state_v001/ROOT_REVIEW.json。


### 正式SAM80一次性封存等待已登记

为了完成后无需等30分钟轮询，已启动PID1702257：仅每60秒检查本run的TRAIN_COMPLETE、EVAL_COMPLETE、CONTROLLER_COMPLETE、BEST四文件；全部存在后核对固定RunSpec和现有sealer工具SHA，再单次调用严格封存。工具固定只读副本及命令记录在outputs/action4_sam80_seal_queue_v001，不改选优、不覆盖默认、不重复训练。目标models/candidates/action4_improvement/stage3_sam80_completed_20261028；CPU25–30，无GPU，最长等待两小时。成功后仍需主控审查真实封存及指标，不能把等待进程当已封存；其他监控不得重复启动同一sealer。


### SAM80封存到原视频配对入口的实际缺口修复

原freeze_pair仅接受baseline ROI50/extension，无法绑定SAM80。现同入口增加显式native SAM80分支，要求严格封存、80轮完整回执、冻结backbone、同B0与ROI1.5、精确SAM输入合同；配对产物声明current_native_v1视频特征路径，不更改默认，不宣称track输入分布已验收。train/eval各自prompt/cache来源分别保留，仅共同B0/SAM模型身份要求一致，不错误要求整个cache来源字典相同。12项回归通过，正向使用当前正式config加模拟完成回执，尚未使用实际完成seal，也未生成视频。证据outputs/action4_sam_video_pair_preparation_v001/ROOT_REVIEW.json。


### 2026-10-28 正式SAM辅助A0的80轮训练完成

实际TRAIN_COMPLETE已写入：80epoch、每轮10400实例、17360有效optimizer更新，17360次尝试，零监督跳过0。最终1346个eval请求已全部发布；本次核验1331完成、15待完成、0失败。训练进程已退出，现为尾部eval排空阶段，不能提前称全部评估/封存已完成。一次性封存队列继续等待其余终态回执。资源权威r116登记训练结束，不重启、不扩增epoch。


### 正式SAM80全部1346次eval完成，最终best13180

TRAIN_COMPLETE、EVAL_COMPLETE、CONTROLLER_COMPLETE同一attempt齐全。实际best13180，全650/1300分母；与同run step0比较All8@2%899→1038（69.1538%→79.8462%，+10.6923pp），合法匹配P95 16.20999→13.55902px（降低16.3539%），高遮挡44/192→71/192、小且高遮挡0/40→2/40。所有原有组的实际对照见FINAL_DEVELOPMENT_COMPARISON.json。收益是SAM/surface/pose联合训练系统收益，未做独立SAM因果归因；仍非test/真实视频验收。一次性封存任务已自动触发，实际sealer子进程1710732运行；尚未写SEALED_ROLE，不提前宣称严格重载通过。


### SAM80严格封存及阶段四真实集成启动

SAM80 best13180独立strict reload和frozen backbone核验通过；A0 SHA=a6e2c9dc88322bf27283de898ccd75f484edfe91bc1586824d7ed923f0f960bd，位于models/candidates/action4_improvement/stage3_sam80_completed_20261028/A0.pt。原默认不变。A0/B0视频pair通过原sealer并行复核并写MODEL_PAIR，非真实视频验收。

阶段四p2_a0_corner_native_integration6_v001从上述真实封存父模型启动：216固定预测ROI、6小epoch/global48/micro2/4rank、预计30更新，step25受控恢复；step0/25/30完整650eval。先前recipe_v001继承阶段三adapter1e-4被准入拒绝、无训练；v002改为主稿阶段四1e-5并通过实际父封存/配置准入。新的不可变RunSpec已生成，controller PID 1717888，启动不等于更新/测试通过。GPU4–7，训练64–95、eval49–54、controller55；预留视频24–48、主控56–63，最多72逻辑CPU。阶段四80正式训练未启动。


### 新封存SAM80完整视频与阶段四集成并行

视频controller1719076，output=outputs/action4_sam80_fixed_video_v001：新sealed A0 step13180+B0step9500、ROI1.5、已有同身份B0/SAM轨迹；明确current_native_v1辅助特征输入，逐帧RGB/ROI/K来自解析后的完整机身ROI。8分片worker使用GPU4–7每卡2个、CPU24–47/controller48，先第0真实分片通过才派余7；不会重跑B0/SAM模型，不在渲染中改点。新源码/配置只读并绑定SOURCE_MANIFEST的configuration；1项实际配置只读/篡改拒绝测试通过。训练集成用64–95/eval49–54/controller55，主控56–63，总72；两者独立模型/快照/输出，不覆盖默认或数据。视频尚未完成，不称效果通过。


### 阶段四实际质量预热和新视频输入链检查

四rank真实训练已完成step25/前5epoch，并由原控制器派发受控resume。实际checkpoint25检查：epoch=5(cursor0)、4rank RNG齐全，corner_quality六个张量与初始step0完全相同，optimizer中该独立头尚无任何state；证明预热期没有隐式AdamW/动量更新。第6epoch启用及完整恢复结果尚待终态。证据outputs/action4_stage4_native_integration_review_v001/WARMUP25.json。

新SAM80视频实际已保存第4、180、232帧。6个实例逐项核对bbox_for_a0=实际bbox_used=RoiSpec.full_body_bbox、固定1.5/448、A448及K112一致、mask available、corner IDs0–7及raw/final不被渲染修改。第232帧两台均无当前B0，但合法SAM tracking geometry仍进入新A0。此为实际保存数据的局部链路核对，不能外推全片正确/角点精度。证据outputs/action4_sam80_fixed_video_v001/EARLY_INPUT_CHAIN_REVIEW.json。首次读取脚本误把instances当顶层字段导致KeyError，未影响生产输出；按真实prediction嵌套结构重读通过。


### 阶段四受控恢复完成与真实视频负例

实际integration已6epoch/30更新完成；TRAIN_PROCESS确认从25恢复，末次checkpoint epoch6/cursor0。六个corner-quality参数均有5次AdamW更新且exp_avg非零，与前5epoch完全冻结的WARMUP25证据闭合；完整650eval尚未全部终态，不能宣布全部集成通过。证据outputs/action4_stage4_native_integration_review_v001/RESUME_QUALITY30.json。

已目视核查保存预测的4/180/232及相邻259/260帧，遮挡区仍明显不稳定：259→260同track_01实际机身/绿色ROI变化小，raw姿态旋转差178.817度，红色框形状明显变化。首356帧诊断筛出200/710相邻track对（阈值仅筛查，不是真实错误率）；针对D2长方体符号对称的简单ID翻转解释未得到支持（按登记规则0例），不能靠重排ID修复实际姿态跳变。已保留原输出和视觉负例，未覆盖默认；六面应继续依赖当前图像/材料观测及连续身份，而非绘制层平滑。证据outputs/action4_sam80_fixed_video_v001/early_review/ROOT_VISUAL_REVIEW.json。


### 阶段四完整真实集成及新SAM80全视频完成

6epoch/30更新/step25新进程恢复已完成，step0/25/30三次全650评分及controller齐全，All8@2%=1038/1020/1019，三次quality_ready均false，NO_DEPLOYABLE_CHECKPOINT，正确保留SAM13180父模型。step0与封存SAM父模型的全650/1300实例/20800坐标标量逐值完全一致(max差0)，不是仅比摘要指标。正式80尚未启动，需实测吞吐容量；三次实际完整eval耗时635.88/633.44/559.38秒，不用阶段三旧容量自动放行。证据outputs/action4_stage4_native_integration_review_v001/ROOT_REVIEW.json。

新SAM80开发视频已2850帧/1900×750/2997÷100fps、time_base1/11988/duration_ts1140000验证完成；所有8片齐全，渲染读取保存结果。视频位于outputs/videos/20261028-040831-SAM80-best13180-roi15-B0-SAM-features-e1p50-track/comparison.mp4；SHA e1c5d3d80c7ec71cb0a323c184c17af5fa4f730d37798487b47fe6ea0c460045。已复制到本地workplace/outputs同时间阶段目录并核对同SHA，便于用户查看。早期259/260真实画面复核失败仍成立，不能称最终达标；继续完整视频失败段分析及六面图像证据跟踪。


### 六面连续跟踪受控对照启动

已参数化既有native_clip_prepare，显式绑定新SAM80完整视频/INPUT_LOCK并核对A0/B0 step/config；注册4–355连续352帧，避免困难段冷启动。两组仅改变持续A0 SoftConstraint是否启用，raw来源有效性/region门/恢复规则相同，仍保留合格重置及全部失败。模型、RGB、SAM支持共用已验证输入，不重推A0/B0/SAM。

现有逐帧重写完整FRAME_EVENTS数组改为追加JSONL，终态/失败再输出兼容数组；稀疏debug仅8帧，内部跟踪渲染不跳过。实际3帧/两体6姿态与全debug逐值相同、JSONL与兼容数组相同；7项选项/先验CPU测试通过。两组nativeCPU作业[1746792, 1746793]分别84–89/90–95，GPU不使用；此为诊断候选，不是六面准确性验收。证据outputs/action4_m3t_sam80_continuous_v001/PREREGISTERED.json及PARITY_REVIEW.json。


### 阶段四完整评估服务容量检查启动

使用实际run_corner_evaluation全650预测/评分/质量准入链，19并发、GPU4–7按5/5/5/4分配，每worker2CPU24–61、4GiB allocator；controller62。无训练、无BEST提交、无test。样本为已完成集成step30模型参数，仅用于服务性能测量，绝不是selected best或后续训练父模型。首次准备误把last.pt当可加载推理bundle，被header检查拒绝、未派发；随后用原不可变training core严格恢复training_model并导出容量用bundle，模型参数逐值一致、新更新0。新源快照p2_corner_capacity19_v001，controller 1753242；配置/预算3项测试通过。保留完整cold-start评分成本，未测联合训练负载和成熟decoder开销，不提前宣称正式准入。


容量v001在真实CUDA模型推理前失败：process_cpu_affinity要求train保留组，eval-only配置未列该组。19worker均在相同前置校验退出，没有650图推理结果；未算容量通过。新v002补未启用train=[63]并在controller扇出前调用同一生产资源校验，未弱化原guard；3项配置测试及实际guard通过，冻结新RunSpec后controller 1755284启动。原失败保留。两个352帧原生对照已完成704姿态/组、无原生调用或数值失败；连续先验组重置0/0，无连续先验组重置1/3，尚需效果复核，不能以完成/低重置宣称精度。


### 连续先验对照的真实视觉结论

两组均完整352帧/704姿态，实际代码调用与数值无失败、12面/16角ID保持，但均未准确性准入。目视4/180/260/355：去连续A0先验在180/260/355明显漂离机身，保留先验组仍有方向/厚度错误。native与相同预测ROI的IoU中位数有先验0.6944、无先验0.4138，raw0.7457；这些仅一致性代理，不是GT精度。不能接受移除先验，也不能把固定ID当准确。原生SAM仍仅进入texture独占像素支持及既有region门，并非六面分别得到SAM观测；下一步应检查真实观测不足和约束，而非继续任意调参或绘制平滑。ROOT_VISUAL_REVIEW与PRIOR_COMPARISON已保留。比较图首次因新labels参数缺Path导入失败，仅影响后处理；补import后复用原跟踪产物生成，无跟踪重跑。


### 阶段四19路完整服务容量测量完成

19×650实际预测与评分全部完成，所有worker exit0，总843.9218秒，0.0225139次完整eval/秒；19次固定权重重复仅用于容量，不作精度重复实验或best选择。最大allocated=0.7504873275756836GiB、reserved=0.787109375GiB。按25step频率和20%服务余量，孤立服务测量对应训练update平均间隔至少2.1320129906198892秒；实际联合训练与成熟decoder开销尚未测，不能仅此放行正式80。原v001前置CPU合同失败保留。证据outputs/action4_corner_capacity19_v002/ROOT_REVIEW.json。


### 32单CPU评估与有界真实训练联合检查

基于19路峰值GPU分配0.75GiB与CPU配额约束，注册32worker/每卡8个、每worker1CPU且eval_loader0、allocator3GiB，复用同一完整650评估器。其与19路的数学输出需后续逐值核验，不预先宣称更快。资源24–55 eval、56 controller；已有真实trainer以SAM13180初始化、相同stage4配方和216真实预测ROI运行到20有效step受控停止，仅用于性能，不是正式训练完成或best候选。训练64–95、观察器57、root58–63，共72。只统计同epoch、前后均global48的连续更新间隔，至少两个3间隔窗口需32eval进程存活且至少16处于prediction阶段；否则联合证据不足，不放行。两份RunSpec的project_code_sha相同，6项预算/时间汇总测试通过。所有新worker和observer的HOME/TMP/CUDA/Torch/HF/XDG缓存均显式绑定项目内。容量controller1779338，joint observer 1780682。


### 32路服务完成与同步就绪联合检查

32×650服务全部完成/1205.697秒/0.0265407次eval每秒；首worker全650/1300八点坐标与19路逐值完全相同。此次伴随故障训练器的初始化资源占用，但无有效optimizer重叠证据，原controller固定training_concurrent=false不能解释为完全没有后台资源占用；不做纯拓扑因果提速归因。v002短IPC训练配置在发现原cohort已有6worker结束后未启动，NOT_LAUNCHED已登记。

新capacity32_v002与joint20_v003增加性能专用就绪门：4rank必须拿到真实首批12单位tensor，32eval必须完成模型/依赖初始化，再验证36个同experiment live PID后GO。正式模式禁止启用此门，不改数学/随机状态；12项门控、真实tensor IPC、时间汇总及原eval链回归通过。固定就绪目录outputs/action4_corner_joint_gate_v001。wall包含初始化及等待，不隐藏扣除成本。控制器1798854/观察器1798917已启动；尚未测得联合结果。


### 联合测量完成及评分重复RGB解码定位

同步GO正常，原trainer实际20有效更新结束；12个全48同epoch间隔、4个合格联合窗口，最快窗口均值1.459327秒。32完整eval全部完成，包含初始化/同步等待总1451.886秒，0.0220403调用/秒；按预登记20%余量要求0.0328919，当前口径余量未通过，不启动正式80。不能隐藏扣除同步等待改变结论。

实际冷后处理cProfile 261.936秒中_v2_rgb约132.649秒/650次，主要为评分再次PIL解码RGB；仅评分未重推模型。已新增同进程原始像素验证凭据，在成功consume后缓存身份/文件指纹，不缓存预测或GT；每次快路仍检查文件、K、B0和完整候选域，换进程/来源变化/无凭据回原解码或拒绝。35项回归通过，full650原/新评分逐值比较进程1837841正在运行，尚不算服务容量已改善。

另观察真实6D头（CPU，第4/259/260帧共6实例）：最小x范数0.520、正交y范数0.269，非接近归一化epsilon。CPU相邻旋转175.60度与GPU178.82度均异常，CPU/GPU角点最大差3.895px，不能把CPU观察当GPU逐像素一致性。冻结资产+y=rear screen，frame4两体均+y几何朝前，与真实后屏可见相符；原生正常图覆盖大体对应机身，无明显整体前后轴翻转证据，但并非标定或全片准确证明。只读诊断v001挂钩错误地用训练封装head而加载结果为SamGeometryHead；v002按geometry.pose观察成功，原失败保留无模型改动。


### 完整650评分复用验证通过

新旧全量scoring/domain/quality target/result逐项完全相同，650图/1300GT，候选评分RGB重新解码0次。先验证当前RGB145.012秒（非模型推理），原评分181.748秒，证据复用评分38.381秒；前置验证成本明确保留，收益适用于正常预测阶段已验证相同原图的同进程链，不能宣称新进程从零评分只需38秒。没有缓存A0预测、GT或训练中间特征，模型调用0；35项回归及源变更/PID/K/候选域反例通过。全服务容量尚待用新不可变源码核验，当前正式80仍未启动。证据outputs/action4_scoring_rgb_proof_v001/ROOT_REVIEW.json。


### 优化评分的新不可变联合服务检查启动

23项门控/资源/计时/sidecar/旧eval链回归通过；新cap32_v003与joint20_v004同project_code_sha，controller 1858141、observer 1858142。训练先预热实际first batch，eval随后初始化，最后共用GO；仍有界20有效更新、不提交best、不正式80。源码包含已通过全650逐值比较的输入证明复用，服务数值一致性及容量尚待实际结果。所有缓存项目内、训练IPC短目录、CPU分区与GPU4–7授权不变。


### 2026-10-28 六面观测链复核

复用352帧结果，无新增推理。continuous_prior中track00/01分别102/119帧无原生尺度内匹配，217/230帧不足12对应；此为诊断数而非准确性门。第260帧track01有B0/SAM当前支持及43个区域内特征，但有效纹理对应为0；不能将本例漂移归因于B0漏检。下一步核查原生关键帧生成/保留与几何绑定，上游淘汰顺序风险尚非证实根因。证据：outputs/action4_m3t_sam80_continuous_v001/OBSERVATION_CHAIN_REVIEW.json。


### 2026-10-28 原生关键帧只读诊断接入

已确认当前a0prior二进制构建配置来自runtime/m3t_observed_src_v001。新增独立keyframe_audit构建，只暴露关键帧age和各缓存点数；action4_native_clip在跟踪调用前后采集，旧二进制明确记录unavailable，不伪造0。5项接口测试通过。独立编译PID1866717、CPU61–63，不改运行中的评估源码或旧二进制；尚待原生导入、同输入姿态一致性及连续失败片段验证，不能称跟踪修复完成。首次文本插入因缩进匹配断言中止、未写native_clip；修正后通过编译语法及测试。


关键帧诊断原生版本已独立编译完成，实际导入age/counts接口通过。首个回放准备复制了只读源文件，替换诊断入口时PermissionError，发生在原生推理前；保留失败副本与日志。修正为仅调整新副本权限、替换后恢复只读，未改父快照。v002控制器PID1869016，CPU58–60，先3帧同输入严格姿态对照，通过后才连续352帧记录；结果仍需复核，不构成跟踪效果通过。


### 2026-10-28 阶段四正式启动与关键帧诊断闭合

32路完整650评估均完成且计分字段与旧固定权重逐实例一致，929.646秒、0.0344217次/秒，超过真实joint20要求的20%余量门0.0310082；该门非长期稳态证明，正式队列max_pending40保留背压。冻结运行源码22项gate/封存/selected推理回归通过；首次pytest不可用、随后单测环境及准备路径范围错误均保留日志，最终运行时测试采用正确project root与tests路径，prepare入口仍在工作树使用原有验证。

正式p2_a0_corner_prediction15_formal80_v001已固化已测源码，父SAM13180/B09500、ROI1.5、80轮global48，每25更新全650，预期无跳过17360更新/696eval。首次r131资源权威漏cpu_policy，创建worker前失败；r132补齐后33个拥有进程已实际启动，controller1875647/train1875675。此时尚未确认首个更新，不宣称训练完成；未覆盖default。

只读关键帧回放352帧/704姿态与旧版最大差0。观察到频繁替换；track00有7帧空bank，track01无空bank却有119帧无有效匹配，空bank并非全部问题解释。证据ROOT_KEYFRAME_FINDINGS.json，下一步须检查错误姿态派生XYZ绑定与关键帧替换的作用，不通过附加平滑掩盖。


原视频translation_only对照已启动，PID1878285/CPU60–63，无GPU。10项接口测试通过；实际原生约束测试确认anchor旋转90度不改变translation-only优化结果、平移仍生效。首个测试遗漏固定anchor优化器初始化而有原生警告，补齐与正式接口一致的初始化后无警告通过，两个日志保留。仅关闭旋转先验，不是最终跟踪修复结论。证据outputs/action4_m3t_translation_prior_v001。


translation_only原视频352帧/704记录已完成，无数值失败。实际查看4/120/180/232/260/355同帧图：180/232/260部分框更贴近机身，但120及后段仍有方向/厚度错误；候选保留、未部署，不量化无GT准确率。ROOT_VISUAL_REVIEW.json记录限制。

正式corner80启动随后出现真实NCCL初始化超时，rank1等待rank0 unique ID180秒；尚无STATUS或有效更新。controller已停止该次拥有进程，不得沿用“训练中”状态。完整10400units缓存校验在各rank NCCL前独立执行，且32eval同时启动，与已通过的train-first容量流程不同；当前仅为可检验的原因假设，下一次需对数据准入到NCCL的启动同步进行修复验证，不能重跑旧配置碰碰运气。失败日志与ROOT_STARTUP_FAILURE_REVIEW保留，default/data不变。


### 2026-10-28 启动同步修复与三关键帧对照

已加入opt-in host数据就绪同步（各rank完成同一attempt/10400输入后才进入NCCL）及controller等待TRAIN_PROCESS再启动eval。5项CPU测试含真实双进程、迟到rank、域不一致/缺失超时/失效trainer拒绝通过，未更改通信180秒超时或训练数学。完整10400实际数据＋四卡collective验证PID1882974运行中，无训练更新。旧formal失败不重启，后续需新验证快照。

translation_only关键帧重建从139/171降至77/34；无对应帧从102/119变为94/129，故不能宣称两机均改善。新单变量n_keyframes=3候选PID1884041，CPU60–63，同4–355帧原视频、其余不变，10项参数/接口测试通过。目标验证单缓存淘汰对跨视角对应的影响，尚未效果通过。


用户进一步明确阶段五直接基于前三阶段推进，固定SAM A013180/B09500/ROI1.5，不依赖stage4增强。三关键帧候选352帧704记录完成，无数值调用失败；实际复核120/180/232/260/355画面结果混合，120/260仍有明显方向问题，因此未替换默认或声称跟踪成功。根复核：outputs/action4_m3t_translation_bank3_v001/ROOT_VISUAL_REVIEW.json。


完整10400预测ROI四rank启动同步实测通过，各rank NCCL求和均10，未训练；最早rank主机等待173.786秒，说明预检到达差已接近旧通信超时。新train/dispatch仅加入ready门，移除新增片段后与原已测数学源码字节一致，core/data/prediction/scoring未变。基于新准入门固化formal80_v002并启动controller1888306，先初始化训练再32eval，首更新尚待确认。阶段五继续固定前三阶段权重独立推进；30分钟闹钟更新到v002，原失败v001保留且不得重启。


### 2026-10-28 正式父模型登记修复与空关键帧事务保护

formal_v002已通过数据同步/NCCL，但正式construct要求的父registry元信息缺role/bytes/architecture/config身份，四rank一致失败、0更新。先前integration不走此分支，是准入覆盖不足；现已从实际封存权重生成完整entry，保留strict verify_reference/verify_loaded。14项回归通过；实际full_training模式构建、81可训练参数张量唯一覆盖6optimizer组、严格clone重载通过；冻结v003中再次实际构建通过。新controller1894678已启动初始化，旧v001/v002失败保留。

原生Body源码确认YAML body_id/region_id会读取，未据此认定ID错误。六面当前实际关键帧日志证实4次非空参考被空更新替换（208/221/223/257）。独立guard候选默认OFF，开启后仅在新3D点非空时淘汰旧关键帧；不可见或空更新保留旧参考，不改点序或直接修姿态。构建PID1895366 CPU60–63，待默认OFF实际一致性与原视频候选检验。此保护仅针对已复现参考丢失，不承诺全部姿态漂移解决。


30分钟检查：v003实际4rank模型初始化通过，EVALUATOR_STARTUP_READY确认等待462.989秒后启动评估；尚未观察有效optimizer更新或新完整eval。空关键帧guard独立版本编译/6接口测试通过，defaultOFF3帧6姿态与旧版差0；guardON原视频352帧回放运行中，不能预先称效果通过。


### 2026-10-28 空更新保护真实回放结果

352帧704记录已完成，defaultOFF短序列差0；guardON累计拒绝6次空更新，非空到空转换消除、两机均无空bank。实际查看208/221/257/260/355仍有明显方向/厚度错误；track00无对应帧104（原102），track01仍119，不能称定位改善。保留可靠性修复候选但未部署，下一步重点仍是有效图像对应和错误姿态派生3D绑定。阶段四v003已产生真实有效更新，最新观察step276，未跳过零监督更新；全650异步结果需完成后另报，不以loss下降代替精度。


### 2026-10-28 结果深度与纹理逆姿态缓存对照准备

本地实际M3T源码中Tracker::CalculateResults先按当前body渲染results深度；TextureModality::CalculateResults没有刷新最后一次优化前缓存的camera2body_pose，而ComputeKeyframeData/Reconstruct3DPoint使用该缓存。建立独立默认OFF候选：记录结果前缓存差，并在ON时于重建前调用既有PrecalculatePoseVariables。未改A0/角点ID/损失；12接口测试通过，独立编译CPU60–63。需实际OFF数值一致性、缓存差记录及ON原视频视觉验证，源码不一致不能直接当已证实精度根因。


结果姿态缓存候选：352帧/704记录完成，刷新后结果阶段cache与当前body变换全相等；刷新前矩阵元素最大差0.0382951、中位0.00374064（混合矩阵数值诊断，非角度/米制GT误差）。实际同帧图仍有方向/厚度错误，故一致性修复不等于精度通过。

同一冻结SAM80 forward的dense surface诊断8帧16实例完成，复用现有PnP固定参数；16个数值求解均有效，但180/232/260明显偏移/膨胀，图示常劣于learned pose。XYZ直接来自surface head、未由姿态反投影、无GT参与。不能再假定只有pose输出错而XYZ已正确；预测对应质量和可观测性同样限制六面恢复。未部署、未训练新模型，证据outputs/action4_sam80_surface_pnp_probe_v001/ROOT_VISUAL_REVIEW.json。

阶段四已收到完整eval：step0复现1038/1300；已检查至step225的训练快照目前未超过step0，quality均未ready。继续既定80轮，不将早期loss下降或raw回退评分称为新增强best。


复用已保存XYZ、零新模型调用，按当前SAM可见支持重放16实例PnP，实际A112/K112身份一致。第260帧track00候选点1247降608；它们是“预测SAM外点”，不能称已经GT确认的无关背景。视觉仍有显著方向/尺度问题，未采用。五帧原生旋转历史对照已实现，15项测试覆盖变帧间隔、身份/时间域、无支持不刷新、过期与快速转动拒绝，PID1922352/CPU60–63原视频回放运行中，尚未效果验收。


### 2026-10-28 稳定V1预测与交付准备

17项测试通过，包括单次方向反转、身份隔离、恒速延迟夹具、静止噪声、当前ROI位移/尺度直接跟随、原始字段保留、视频入口实际消费与渲染前篡改拒绝。2850帧/5700实例全域已生成并逐实例验证唯一刚性R/t投影，未重跑模型。大角变化且ROI小位移诊断事件1394→259，相邻角变化中位16.48→3.80度、P95 85.12→30.64度；不是GT抖动错误率或精度提升。所选历史旋转来源中位1帧、P95 4帧、最大5帧，真实视频延迟未有GT测量。实际复核4/120/180/232/260/1195/1289/2400/2778/2849同帧图：保留当前机身ROI位置；部分方向/厚度误差仍在，本版主要抑制突跳。完整CPU编码进行中。

源码冻结准备首次遗漏configs快照导致重定向校验失败（未生成预测）；新source_v002包含确切配置后成功，未绕过路径/源码校验。薄入口scripts/action4_stabilize_video.py已实际验证同身份复用不重复生成。模型组合登记models/candidates/action4_improvement/stage5_stable_v1/SELECTION.json，不复制权重、不改models/default。


### 2026-10-28 基础稳定V1完整视频交付

完整视频编码与ffprobe通过：2850帧、1900×750、2997/100fps、time_base1/11988、duration_ts1140000，保留完整原片时间域。全域5700条最终R/t形状、SO(3)、固定ID复核通过；A0行身份为step/config_id（旧原始行不含A0 SHA），精确权重SHA继承已绑定MODEL_PAIR；B0行SHA直接核对，未虚构额外原始字段。首次交付审计读取不存在的A0行SHA字段中止，已按实际字段及源模型组合回执分层核对。

视频：outputs/videos/20261028-121116-stage5-A0-stable-v1-e1p50-track/comparison.mp4；SHA256 31a5a2774b95b8b0208ae25e846dc11ee995600ba111728c83350456d6e9ffef。本地工作区outputs/videos同名目录已复制并计算SHA完全一致。左侧demo仅参考，右侧稳定V1。17项接口/数值测试、全预测刚性验证、完整编码检查分别成立；没有真实姿态GT，未宣称最终定位精度通过。默认模型未改；精细方向/厚度误差和剩余突跳继续优化，阶段四训练继续。


### Stage5 稳定性失败定位与图像约束候选（2026-10-28T13:33:02.817898+08:00，进行中）

已复核 V1 原视频：1163/1428/2777/2523 附近单帧最终旋转约137/131/131/129度，ROI中心移动为对角线6.77%/1.74%/7.98%/0.59%。当前 SAM mask 在这些帧 available；这不等于姿态可靠。逐帧图见 outputs/action4_stability_failure_audit_v002/TOP_JUMPS.jpg。V1.1 source-clock 全2850仅将定义内事件259降至233，仍有明显方向错误；不发布为已解决版本。
图像运动小探针16目标中，1428右机实际23个双向纹理点、覆盖0.643；可绑定22个估计表面点。错误跳变重投影中位73.64px，对照前姿态方向＋当前平移2.57px，可形成拒绝证据。其余多处纹理不足，保留 unknown。原生 SAM 概率不作为姿态质量。
代码已加入 src/model/action4_video_motion_evidence.py、action4_video_motion_observation.py，默认关闭地接入既有视频姿态处理及 predict_video；不改运行中源码。35项CPU/接口测试通过，含实际合成像素平移、跨实例排除、当前平移保留、真实转动对应、缺纹理不等于静止、旧路径与身份兼容。这里的测试通过不是原视频准确性通过。
六段真实成对重放在 outputs/action4_video_motion_veto_probe_v001；初版冻结包遗漏 data.__init__ 的 action4_adapter 依赖，启动前导入失败、无预测；source_v002 补齐并先完成冻结导入检查后启动，保留失败日志，不修改旧快照。正式训练未停止，data/default 未更改。


#### 实际成对重放与拒绝部署（2026-10-28T13:39:16.590953+08:00）

六段共306帧完成；核心66帧/132实例。第一候选 image veto 的 >30度跳变33→22。发现拒绝后下一帧纹理 unknown 直接恢复候选的问题，新增最多5帧方向保持，当前平移不冻结；37工程/接口测试通过。只重放相同缓存纹理对应（未重复模型或光流），第二候选跳变33→14，59/132实例具备纹理运动支持，17次拒绝、12次有界保持；这些计数不是真实角点误差或全片改善率。
实际检查 ROOT_REVIEW_0/1.jpg：1428附近有改善，但1162–1164仍可能稳定保持错误长轴；2523段8→9次跳变反而退化。因此 outputs/action4_video_motion_veto_probe_v002/ROOT_VISUAL_REVIEW.json 明确拒绝稳定版部署，不覆盖默认，不生成新的全片“成功”视频。原因是上一估计姿态派生XYZ与低重投影残差不能证明初始姿态正确。下一步必须同时审查当前候选和历史参考的可见面/轮廓相容性，不能只增加时间滤波。
阶段四当前step4590、epoch22，已完成171次完整650图eval，失败0；最近已完成step4250 All8@2%=1028/1300，尚无deployment-eligible增强快照。正式运行未受修改。


### 历史参考资格审计与SAM80面输入修复（2026-10-28T14:41:57.233741+08:00）

outputs/action4_reference_shape_audit_v001核完同一132核心实例：原A0必要轮廓相容60/132、原平滑54/132、运动候选53/132。该检查使用预测SAM mask，不是GT精度；反证“更稳等于更正确”。1163右机原A0无容差可见覆盖92.7%，保持候选77.8%。模块已接入可选当前mask输入；低可见面积、错误方向、180度对称同轮廓、跨实例/缺mask、实际消费者变化等已测。动态显示区域未排除的旧纹理探针不能作为刚体观测；已增加显式资格并保持默认关闭。
当前封存SAM80有20份真实surface快照可直接复用，全部RGB/K/实际ROI/模型比对后生成11份可提示实例，其余9无提案，不补假面；证据outputs/action4_sam80_saved_face_inputs_v001/COMPLETE.json。A0/SAM新增forward均0。旧face_pilot_inputs.run遗漏增强视频sam_video_context，现改为正式VideoSamFeatureContext.load同桥；2接口测试通过。连同稳定性/支持检查共52项已分别通过。
原生face试验v001在session前缺minimum_uncommitted_memory_margin_gib，已保留失败；r137/v002补齐8GiB余量后实际SAM严格加载0missing/0unexpected。种子返回whole_body_iou=0.898、negative_coverage=0.5，被原门拒绝，未传播15帧，进程exit0不等于跟踪通过。基于真实frame1195提示图，第3个正点位于bezel而非显示内容；v003仅把该点改为负点，记录manual_refinement，目标限定动态屏幕排除，不用于假设物理面GT。当前r138资源扩展只加一GPU7/CPU60–63/8GiB任务，不改变正式训练源码及数值配置；旧原始结果与默认权重保留。


#### 原生显示区域试验结束（2026-10-28T14:51:02.120472+08:00）

v003完成1195–1209共15帧、原生对象0身份连续，会话正常关闭、进程exit0。实际全15缩略图逐帧审核：显示内容区域保持，首帧whole-body IoU0.496、正点覆盖1/负点0；全段预测机身包含率最小0.94995、另一机像素占比最大0.000119。只认定一个实例/15帧显示区域候选，不是完整后面或全片六面精度。ROOT_VISUAL_REVIEW.json保留明确边界。
复用这15份原生输出做14相邻帧对照：不排除屏幕的纹理14/14被门槛接受，保守排除显示区域后0/14达门。不能把旧“available”当可靠刚体观测。本次排除是保守策略；屏幕上可能有固定显示/标记，是否与机身刚性一致还需独立验证，不能把整个屏幕中每个像素都断言为动态。已有全2850 marker审计是历史候选，只有361个当前支持关联，不自动升级物理绑定或全视频解法。
阶段四实际step6169/epoch29，完整eval235次、失败0；最近step5850 All8@2%=1040/1300，部署准入候选0。训练、默认组合和发布数据未更改。当前原视频稳定性仍未验收，后续必须围绕可靠的物理面/静态材料观测继续，不能再仅靠低抖动指标发布。

显示排除实现随后复用已有action4_face_motion_evidence.border_support的保守凸包排除，避免显示片段间空隙重新作为刚体纹理。原14对真实对照的代码版本单独绑定在v003/motion_audit_source，未用新代码覆盖历史观测。受影响22项测试通过；另加显示区域孔隙反例。全阶段仍未精度验收。


2026-10-28T15:12:43.361858+08:00：固定编号监督源码核查未发现无序匹配；全5698相邻姿态对中253次>90度，但虚拟长方体四个方向分支均无一使这些跳变降至20度以下。排除简单角点编号/180度分支修补，不改推理ID。阶段四step6672/epoch31，eval256/267、待11、失败0；6350 All8@2%=1049/1300，仍非部署准入增强。后续核查已有CoTracker材料接口与真实种子资格，不以背景补点，不自动调用PnP。


### 同批质量审计完成、All8候选已保留（2026-10-28T15:58:15.436488+08:00）

outputs/action4_corner_quality_alignment_v001/REPORT.json：使用不可变正式训练源码的build_corner_targets，对同一step6375完整650图1300实例10400点重建目标，读取并采样全部真实GT可见mask。A224差0，known/valid资格差0，训练与评分二元标签差0，评分UV与导出UV相同，decoder实际准入ID差0；float32训练与评分误差最大差0.000243像素。该批没有坐标/点序/标签/准入实现不一致的证据。
实际差别是概率过度自信及指标事件不同：q>=.6选8977点，平均q=.91782，但1%正确率.68553；decoder可用8767点，1%正确率.69613。排除的210点正确率.24286，说明valid/域/多峰筛选有实际作用。q>=.9组6166点，平均q=.97517，1%正确率.84350；排序有用、概率未校准。质量头针对单点1%阈值，最终姿态按八点2%评分，这两者不能直接等同。
同一step6375 raw All8=1040/1300，corner_candidate=1115/1300。按用户新规则完成候选排序，保留 models/candidates/action4_improvement/stage4_all8_trial_20261028/training_snapshot.pt（与实际已完成eval快照SHA一致），身份和选择见outputs/action4_corner_all8_trial_selection_v001/SELECTION.json。严格重载/视频尚待执行，不宣称正式训练已结束或质量校准已通过。
材料子任务由主控复查真实结果：未降门槛，补足预算后frame4左右仅2/4种子，1195和2778均0；都是provisional，未确认3个非query帧且每机不足12几何点。当前数据链已消费，不可强塞背景或把6个不同机身点合并做一台PnP。主控审核记录outputs/action4_material_sam80_readiness_v001/ROOT_REVIEW.json。


### All8候选实际视频入口推进（2026-10-28T17:09:40.876954+08:00）

step6375已在正式不可变loader独立strict reload通过，完整backbone/SAM/pose/corner均存在，quality_ready/deployment仍false。既有predict_video显式corner_trial_binding接入当前VideoSamFeatureContext+resolved ROI，补齐RoiSpec/A448/A112/A224/K112保存回放检查。run_part、controller、renderer复用既有工程，增加绑定模型/源码/配置/逐帧预测的独立trial receipt，默认raw guard不变。14受影响CPU入口/receipt/回归测试通过，独立review发现并修复候选误标raw及receipt配置关系缺口；actualforward不能以stub测试替代。
v001首段在B0 counterpart路径检查失败，尚无forward；记录SHA相同但路径不同。v002显式改为角点快照实际绑定的不可变B0副本，selection保存alias核验，未放宽guard。首6帧完成6次backbone/0B0、峰值约806MB、8.81秒。0–3无SAM按既有bootstrap明确raw回退；4–5有当前mask及角点头实际消费，因多解/可靠ID不足保留raw。原图复核未发现绘制篡改，但不能由首6帧声称精度通过。
全2850现由GPU5/6、CPU60–63两worker继续，controller57。截至检查782完整保存帧、15实例几何候选接受；主要回退为可靠ID不足1457实例。12512点中q>=.6为5361、valid>=.8全部、multimodal1945、最终usable4996，显示真实域角点置信/分布仍限制精修。统计为非连续已完成分片前缀，不是全视频验收；不改运行decoder阈值或叠加平滑。
正式训练9198step/43epoch、355eval已完、待13、失败0，8875 raw最终All8=1049/1300；独立候选视频使用6375不随训练动态换权重。


### 材料原生跟踪实际消费（2026-10-28T17:29:40.565882+08:00）

已审核其余7帧双机14条材料记录，全部候选访问：仅20帧各3点、808左2点、1289左5点、2489左3点，其余0。连同之前4帧左右2/4，没有任何记录达到12点/4格，禁止降低门槛或补背景。
复用既有CoTracker3 core适配器、预训练scaled_offline和固定模型配置，实际frame4–20共17帧、两机分别2/4个已核暂定种子，固定crop，36context支持点不参与CAD；正反向原始xy/v/c及query标记全部保存，未用query强制写回评分。GPU4/8GiB、CPU58–59单线程，12.07秒运行完成。96个非query点帧通过当前SAM归属/非另一机/visibility>=.9/confidence>=.6的部分门。原图4/8/12/16/20可见点基本停留机顶标签，仍没有真实点GT。
复用现成frame20 SAM80surface，只采样实际预测UV经A112后的同点XYZ/region，不重新跑A0；多个固定XYZ偏差超过2%CADdiag，另有面法线歧义/背面或投影不一致。因此没有3帧完整确认，且点数分布不足。保存ROOT_REVIEW.json为二维实际消费通过、三维几何未准入，未将6点混机求解，不改A0输出。证据outputs/action4_material_track_native_v001。


### 有限q门候选与完整视频（2026-10-28T17:36:37.578921+08:00）

原0.6角点候选全2850/5700实际预测完成，18实例接受几何修正，其余保留原姿态及原因；完整trial receipt已绑定保存产物，正在只读编码，不能称已经稳定。真实域主要q不足6点，不是valid分类。
限定q0.4/0.2两候选只改独立decoder q门及研究schema，其余min6/多峰/非共面/正深度/姿态变化/boxsupport全部不变，未改运行中源码。真实0.6 accepted/fallback两实例先与既有结果复现一致，然后全650/1300计分；原模型调用0，GT仅最后score_pose使用。All8@2%1115→1121→1125，P9512.4179→12.1176→11.8012；高遮挡80→83→88/192，近邻15→14→15/20。0.4退化反例保留。Root核对报告与输入，允许0.2作开发视频对照，不覆盖默认或把质量分数改为已校准；0.6完整视频继续不动。下一步只重放整片保存corner observations及同帧SAM masks，避免重复GPUforward。


### q0.2实际全视频重解码完成（2026-10-28T17:49:29.033547+08:00）
复用6375全2850/5700实际网络输出和当前SAM mask完成q0.2解码，0新增A0/B0/SAM；首0–5及1163共14条q0.6实际复现通过，逐帧maskSHA/数量/owner/PTS一致，全部final单一刚体投影核验。原raw/ROI/固定ID不改，研究schema/decoder source明确区分，未调用旧validator伪通过。18→153接受，158条final变化；140由fallback变接受、5由接受变fallback、13两路接受。多数仍回退：可靠ID不足2395、多PnP解914、优化多解808、姿态变化超门1016、可见支持325等，失败完整保留。
root实际检查4/1163/1195/1289/1950/2400三列原图：4一机有变化，后五帧均raw回退，方向/尺度问题仍存在；不能把接受数提升当稳定锚定达成。ROOT_FINAL_REVIEW明确未通过稳定性验收。原q0.6完整对照视频继续编码，source不改变；q0.2逐帧结果已保存，尚未另编码整片。


2026-10-28T17:53:03.494076+08:00：q0.6候选完整视频编码完成并传回本地，SHA b51ebff5c0d2e7b666c2f4a1d109f0d81f094921228f741f7d68fc1558b89702 两端一致；2850帧1900x750、2997/100fps、time_base1/11988、duration_ts1140000。actual raw/candidate同帧及编码抽帧均保留，DELIVERY.json明确未通过稳定性验收，不覆盖默认。

2026-10-28T18:05:31.851804+08:00：10416step/48epoch、405eval已完、待12、失败0；10125 raw1061、candidate1082，不高于当前已完成角点候选最优step7300的1116/1300。已核旧controller在quality未过时保留全部快照依赖，completion可写NO_DEPLOYABLE_CHECKPOINT；新All8候选封存需独立回执，不修改旧BEST或其判定历史。


2026-10-28T18:37:16.537130+08:00：step6375的64个预登记训练unit前向诊断已由root复核逐点记录、固定索引顺序及参数/buffer版本。512标签独立重算差0；q>=.6共454点，431满足1%标签(94.93%)，平均q.92175，对比完整dev同阈值68.55%/meanq.91782。抽样train隐藏点71/82正确(86.59%)，而完整dev已记录28.61%。该证据更支持训练分布到开发分布的校准/泛化差异，不支持普遍的target/decoder计算错位；训练64与完整dev组构成/augmentation不同，不能把差值当完整因果量。原配方15dropout/15perturb保留，无backward/optimizer/新训练，GPU峰值634MiB，进程已退出。ROOT_REVIEW.json记录限制。


### 收尾选择工具与实际姿态通路诊断（2026-10-28T19:49:36.303896+08:00）
All8-first终态研究sealer新增3文件完成，root核查身份/域/严格加载/不伪造quality，12工程反例通过，真实check-only只返回WAITING；80轮及全部eval后才能复制seal，不更改旧controller/BEST/default。
同step6375同12失败帧RGB/ROI/K，SAM辅助ON与显式aux dropout实际24forward比较：native与原保存结果像素完全一致，大于30度旋转均15次，median60.18/64.66、P95153.34/171.38；不能认定关闭SAM能解决抖动。旋转6D捕获30实例最小轴sin约.3166、xnorm约.44，没有逼近数值epsilon，不改规范化公式。首捕获脚本错误假定posehead批量2，真实instancewise调用为单实例，v002按实际hook顺序拼接并核域，旧错误保留。
复用SAM80已存20surface做pose输入几何support对照，RGBbackbone/SAM/B0新增调用0；只将可见支持外XYZ设归一化中性.5、region归零，rays及原visible/amodal mask保持，不遮隐藏角点热图。CPU重放出现小数值差超过固定容差，v003换原GPU算子而非放宽门后全部重放通过。真实4/180/1195/2778三列图结果混合，未部署；该证据证明背景几何影响pose，但不证明简单屏蔽就是正确修复。后续另按原稿ray-mesh暂定绑定来源检查frame4真实静态图像特征，不降低原surface门。


### 用户提前停止阶段四并固定阶段五父模型（2026-10-28T21:57:46.105050+08:00）
用户明确不再继续stage4训练。Root核对trainer/controller PID及进程组后停止本任务训练并保留32eval，实际STATUS13454step/epoch62；539提交eval至13450全部完成0失败后核验并停止32个owned闲置eval。USER_STOP_REQUEST/STOPPED/EVAL_DRAINED真实记录保留，未伪造80轮completion。新--user-stopped封存实际执行完整报告身份核验与独立strict loader/backbone检查，选step7300 corner_candidate All8@2%=1116/1300，FP/FN184、P9512.3143；SEALED_USER_STOPPED_ALL8_RESEARCH，formal_training_complete=false。实际A0路径models/candidates/action4_improvement/stage4_user_stopped_all8_20261028/A0.pt，SHA b8ce72185e7ce46be29a1b62cf8d7086bb6167669959a76f870365b6f09e70b9。原质量标记保留、不覆盖default。
已固定configs/model/action4_stage5_selection_v001.json，实际新父7300接当前SAM/ROI1.5，对4–9六帧真实forward和12份surface导出通过，0B0重推，严格绑定raw与candidate/ROI/K；后续不继续从旧SAM13180作为当前parent。新初始ray绑定也是7300实际frame4pose；旧13180证据保留只作历史对照。
root发现旧LCD排除global hull会跨离散小岛遮掉静态标签；新默认OFF的per_component模式保留每个native mask像素和同3pxmargin，只不填跨岛空隙。9组件/旧motion测试通过；真实旧parent第二机5→9点/4→7格，第一机15/9不变。以新7300parent重新绑定仍15/9点、9/7格，全部provisional，第二机未达12点。原native mask与dataset没修改。ray辅助脚本已固定从执行snapshot导入exact_surface_hits，旧live helper入口不继承到新执行。
新父第一次surface-only index缺body_mask字段被阻止，v002补充与当前原生像素逐值核验一致的既有mask引用，未放宽身份检查。当前GPU4/CPU88–95进行frame4–63实际60帧轨迹验证；不按数量预签几何准确，不把第二机9点当已满足12点。

2026-10-28T22:00:10.929086+08:00：新封存7300实际初始pose的15+9暂定射线材料点已完成frame4–63共60帧原生CoTracker轨迹。第一机885/885、第二机506/531非query点帧通过当前SAM归属、v>=.9、c>=.6等部分条件；所有raw轨迹和失败保留，未做PnP/姿态写回，尚缺闭环、固定绑定多帧确认及实际视觉复核，不能称几何可靠。运行27.64秒，native probe已终态，旧train/eval未重启。


### 阶段五60帧闭环与固定绑定复核（2026-10-28）

阶段四保持用户停止及step7300研究封存，不恢复80轮。实际60帧原生CoTracker闭环已完成：track_00为15/15，track_01为8/9；第9点终点无可靠观测，记未测而非零误差。闭环结果未改变A0姿态。

复用当前7300的5—9帧surface，以实际A112将原始跟踪UV映射后采样XYZ，分别与固定ray-CAD绑定比较。24点中没有点达到至少3个非query帧的既定2%CAD对角线距离条件；双线性与最近邻误差均存档。二维轨迹自洽不代表三维身份正确，目前不得进入正式材料PnP或宣称六面稳定。ray绑定和surface均来自模型估计，尚不能据此认定哪一方是真值。

证据：`outputs/action4_stage5_ray_cycle60_v001/COMPLETE.json`；`outputs/action4_stage5_ray_surface_consistency_v001/RESULT.json`。本检查0次新模型调用、0次姿态修改，无最终test或默认权重变动。


### 阶段五固定估计绑定与120帧有限对照（2026-10-28）

独立坐标审计在query frame4已发现ray与dense surface距离中位4.638/4.910mm，surface同pose重投影中位20.900/24.843px；12份snapshot身份、frame/track、米制CAD、A112与ray重心重构均一致，未发现坐标链实现错误。差距不能归咎于后续CoTracker漂移；两条估计均非真实GT。

新增eval诊断复用FitPacket与heldout过滤。3个实际CPU测试通过：heldout任意改变不影响求解、有效UV改变能影响求解、11点拒绝（本环境无pytest，runpy逐项执行全部3函数，未安装新依赖）。60帧第一机59个非query帧可数值求解，177个heldout点帧median0.3931px/P950.8969px，59帧纯观测条件数通过；第二机不足12fit未求解。root已看6帧CONTACT，短段框随物体稳定，但机身下侧范围仍含初始偏差、该片段运动较小，不准入正式姿态，不称真实角点精度提升。

提案质量0.003的有限对照保留原门，候选20/19点、12/14格。120帧实际原生跟踪54.37秒完成，partial支持2164/2380和2037/2261；闭环仍在运行，后续需独立复核。证据outputs/action4_fixed_material_pose60_v001、outputs/action4_sealed7300_ray_seeds_v003、outputs/action4_stage5_ray_track120_v002。


### 120帧固定crop根因修复与转面失败保留（2026-10-28）

相同20/19点在120帧initial crop测试中末帧全无支持。读取冻结上游框发现track00第117帧已完全离开frame4 crop，track01第123帧也完全离开。新增默认不变的offline_frozen_prediction_union，窗口内每帧B0+SAM完整预测框各自一次expand1.5后取union，同一crop贯穿core输入与native回映；无GT、无重复扩张。3个crop测试通过；独立审阅确认来源、采样、负边界及回映匹配。

实际同输入120帧修复对照：partial支持track00由2164/2380至2345/2380，track01由2037/2261至2096/2261，运行52.95秒。这只是支持计数，不是准确率。root看第5/110/123帧两机CONTACT，123帧机身已大幅转面，仍存在错误位置。末帧8/5点可发起回查，但无点通过闭环：第一机7点cycle误差超门、1点返回不可用，其余12点末帧不可用；第二机2点超门、3点返回不可用，其余14点末帧不可用。cycle像素误差最高161.21，绝不把落在同一mask内视作同一材料。原失败完整保留，不发布新pose。

新增固定绑定诊断两个P2已修复：强制预登记前三个heldout ID，并验证cycle/track/surface/model/frame/PTS/anchor顺序和维度。4项可移植测试通过，独立审阅通过；此前60帧来源实际符合规则，原结果未失效。

下一步重点是转面处旧材料停用与新面重绑定、按观测区间闭环；不得降低原有12点/4格、改写已观察失败、或将共享A0初始化的低残差冒充真实三维准确。阶段四保持停止，默认权重和最终test未动。新增证据：outputs/action4_stage5_ray_track120_v003、outputs/action4_stage5_ray_cycle120_v003、tests/test_action4_material_prediction_crop.py、tests/test_action4_fixed_material_pose_diagnostic.py。


### 当前best全片与刚体限速滤波（2026-10-29）

当前sealed7300全2850帧完成并已传本地，5700实例、37次corner候选接受。复用已有同身份6帧，新A0帧2844，B0/SAM0新调用；8workers在GPU4-7、64CPU实际并行，全部终态。该未滤波baseline在100/110/123/130/150/180视觉检查仍有明显方向/厚度错误，不能作为改进后稳定交付。

刚体限速滤波已全2850/5700处理和只读编码、本地SHA一致。同当前selected-A0输入的V1控制：大于30deg且ROI运动小事件1389->239，旋转P95 82.037->29.298deg；新候选1389->0，旋转median3.555/P956.006deg，选用历史来源age中位1、P954、最大5帧。0事件主要由显式限速保证，不证明准确；root看6帧转面contact，快速转腕滞后及原方向/厚度问题仍在。26测试通过，含旧身份/默认行为、selected真实消费、双ID隔离、限速/刚性、raw保留与图像模式拒绝。输出outputs/action4_stage5_bounded_stable_v002/DELIVERY.json；默认权重未覆盖、无最终test。

native关键帧事务独立源码实际编译，34原生检查及root三帧defaultOFF exact parity(6pose,delta0)通过。host41帧A/B都82pose，无native/数值失败；ON仅2commit、40reject且旧bank字节完全不变，但root看100/110/120/140，两路都在转腕时脱离真机身，因此只接受工程修复，不部署效果。当前已选pose接入验证5tests及真实3帧6实例通过，其中3实例selected和raw不同并被正确消费。

编码PNG供给已新增默认OFF的有序8线程预取，24真实帧逐像素完全相等、顺序/尺寸错误处理通过；串行4.439/4.458秒，8线程1.009秒，仅解码准备测得约4.4倍，不冒充整片编码速度。当前已运行encoder源码未改，后续新版本可显式启用。


### 当前图像方向证据真实消费与全片启动（2026-10-29）

新增默认OFF的HOLD/MOTION证据producer和现有stabilizer真实consumer。30项滤波/消费回归加9项producer检查通过；真实25帧50实例中13次保持、19输出相对同输入bounded控制发生变化，关闭证据后姿态逐值复现控制，原ROI/raw未改变。静态参考固定，缓慢累计旋转会退出保持；真实转面100→110/110→120均unknown不保持。当前mask认证不表示语义/材料或姿态准确性。

新4reader只预取数据，state仍严格顺序消费，各线程独立ReceiptVerifier；实际4..15共12帧与既存顺序观察逐字段完全相同，1.807秒完成。当前完整2850候选已从不可变源启动，PID2309484需实际核验；CPU64-95、0GPU/0新模型。源与参数outputs/action4_stage5_image_stable_preparation_v003；同模型当前像素的新效果尚待完整诊断及视频复核。

另对完整原片ROI源做了只读核查：5700实例无重复/缺失track域，无完全相同的双机mask SHA，无IoU>=0.9的采用框对；727实例当前没有B0支持，但SAM支持的完整ROI确实输入A0。此为数据通路证据，不等同于727次真实正确识别的人工GT成绩。见outputs/action4_stage5_tracking_roi_audit_v001。


### 图像证据及弱轮廓增益完整视频闭合（2026-10-29）

V3全2850/5700完成：有效图像证据1455、直接保持293、unknown4245，最大连续保持80帧。全片角度中位3.537/P956.006deg，大跳事件0；边界特征覆盖有限，不能宣称全片准确。已复制完整视频到本地且SHA一致cd84e5627fc43135121d41aa15206580d365315de1dbfa4e92f170273e266524。

固定48相邻对whole纹理+silhouette候选仅补4/28unknown，且1400→1401 track00与boundary已知motion冲突，未采用。仅质心平移弱hint在同48对补7/28、无已知motion冲突；全量4CPU块mask-only 142.85秒，912eligible、1455known排除、3313阈值拒绝、20身份/缺失/代际unknown。首次preflight误把派生回放身份当文件SHA，尚未读mask即停；修复为回执绑定raw逐行对齐，失败和原源保留。

V4复用V3全部图像观察和固定raw预测，0RGB/LK/模型重跑（渲染按需要读取RGB）。完整2850/5700实际消费审计：912条均真实应用<=0.75Hz/90deg/s，最大该分支步进3.003deg；1455强证据flags/选择未覆盖，全部raw/ROI/ID/translation不变；强静止保持293仍有当前像素支撑。整体角度median3.003/P956.006deg，大跳事件0，属于滤波输出统计不是姿态GT。完整视频已本地SHA一致fa9570b6bfdec6ad7c3516272c7d2180700ac3f5838757d05d51f2a005141091。root实际解码110/180/1195/2400查看，1195和2400两机分别有跟踪框且未框衣物，但方向/厚度/快速转动仍有局部误差，未做全片真实姿态验收，不作默认晋级。

另缓存比较stage3_13180与stage4_7300：原片raw旋转median/P95约16.485/85.116与16.444/82.037deg，图像quiet子域293对约2.637/28.996与2.438/23.901；RGB/ROI/mask来源相符，K112 JSON记录的float32/64差异最大约3.05e-5，不能据此称新模型比旧模型显著变差。此为运动输出诊断，不是真实角点准确率或完整input tensor parity。

当前训练、所有该轮probe和V4编码均终态；保留同一技术计划/结论。后续继续修正可见面的几何约束与转面对应，当前稳定候选不等于八阶段全部完成。

### 2026-10-29 稳定姿态消费断点修复与未通过的几何假设

发现并修复实际接口问题：V4的final_pose缺finite、顶层R/t保留前滤波值，导致直接下游姿态先验将5700条全部拒绝。renderer读取final角点正确，因此这不是证明已交付视频未滤波。新增只读legacy适配与v2生产合同后，全5700条恢复合法消费、重投影误差0，原几何和原文件不改。三帧真实原生配对中，先验实际启用从0/6变为6/6，float32 anchor与预期完全一致，原生输出确实发生变化；这只证明信息进入求解，不证明精度。

证据：outputs/action4_final_pose_contract_fix_v001、outputs/action4_final_pose_native_bridge_v002/ROOT_ACTUAL_CONSUMPTION.json。prepare→native→audit→attach闭环6记录通过，原2850帧去附件后逐行相等；raw与selected路由独立，selected恢复用对应已选输入。附件仍deployment_admitted=false。新增root封存就绪标志负例检查11项通过；此前44项合同/稳定器检查及14项消费者检查记录保留。首次缺source manifest、旧MODEL_PAIR缺顶层几何字段的失败均保留。

其它有价值负结果：历史候选可改善部分mask相容，但54个可替换实例中候选相对前帧角差中位25.30°、P95 83.85°，不能硬切换；屏幕mask不是认证物理LCD轮廓，冻结mesh区域与标称有效显示区域不能混为同一几何；7帧桌布消失点仅2帧产生条件数值候选，不能据此更换K。24实例冻结pose-head焦距敏感性中当前2639重放R/t完全相等，2000/1370候选方向视觉混合、严重错误未解除，K/CAD未修改。对应outputs/action4_history_support_probe_v001、action4_screen_plane_hypotheses_v001、action4_grid_intrinsic_feasibility_v001、action4_intrinsics_pose_probe_v001。所有均诊断，不升级为校准或精度证据。

正在将修复后真实连续先验扩大到三类短片段配对，输出位于outputs/action4_corrected_pose_prior_clips_v001。使用现有原生跟踪器、固定输入，不重复网络推理；结束后单独登记实际消费、稳定性和视觉结果。当前准确稳定全视频仍未验收，默认未覆盖、训练不重启、最终test未使用。

三类配对实际完成：每组103帧/206机身记录，6进程全部exit0，连续先验实际206/206启用，无先验对照0；0网络新调用。根审110/140/1195/1220/2400/2420对照图：连续先验避免了无先验原生解在转动、遮挡时严重离开机身，但方向/厚度仍有偏差，未统一优于V4，故只接受接口修复，不替换视频输出。证据outputs/action4_corrected_pose_prior_clips_v001/ROOT_REVIEW.json。

增加默认关闭的五帧旋转方向一致性增益：往返冲突时降低滤波带宽和角速度上限，连续同向旋转维持原行为；只作用于同一刚体姿态，ROI/平移/raw/ID不改。22项包含恒速精确对照、振荡抑制、反转恢复和来源保留的测试通过。全2850已保存预测重放正在进行，复用已有SAM/图像证据，不重跑网络；尚不证明原视频精度改善或接续正确。

五帧一致性候选完整重放完成2850帧/5700实例：旋转步长中位2.037858°、P95 3.363448°，V4分别3.003003°/6.006006°。50项实际相关测试通过（33 unittest＋17 standalone），另对冻结旧版本60帧验证默认OFF角点与遥测完全相等；恒速转动夹具不引入额外滞后，反转后能恢复。全片raw/ROI/ID/平移不变，4854实例的预测旋转一致性低于0.5；这是预测噪声指示，非真实运动检测。抽查100/110/140/180/1195/1220/2400/2420仍见方向、厚度误差，故仅允许生成开发候选视频，不覆盖默认或宣布准确率提高。初次快照缺ACTION4_PROJECT_ROOT，在重放前失败，保留v005/FAILED；v006启动环境修复、相同模型与输入完成，未更改运行中源码。

V5完整开发视频已交付：`outputs/videos/20261029-072034-stage5-sealed7300-coherent-stable-track-e1p50-track/comparison.mp4`，2850帧、1900×750、2997/100fps、1/11988时间基、1140000ticks；本地与服务器SHA一致。编码后110/1195/2400帧实际抽查两机与编号存在，未在所审帧重新锚定衣物；方向/厚度仍有误差，未声称全片视觉验收或真实误差改善。全部本轮原生及重放/编码进程终态，不重启。保留V4、V5及原始预测，默认不覆盖。

### 2026-10-29 SAM轮廓分支、相机合同与标记绑定独立诊断

SAM此前真正进入原生纹理支持，但RegionModality仍消费RGB颜色。新增默认关闭、每实例独立SAM可见支持相机，仅给region，texture/渲染仍原RGB。三帧真实默认OFF逐值复现原native结果，独立SAM相机所有权/字节核对通过、后续实际tracking梯度改变。三个片段103帧/206实例：原V4相容113、原生RGB63、原生SAM92；root审110/140/1195/1220/2400/2420仍发现方向和厚度问题，不发布。当前二值法会把可见边界作为轮廓观测，不能保证完整机身，尤其手/衣物遮挡；发现共享像素时新增拒绝，避免把unknown当确定背景。全部已跑片段共享像素数0。证据outputs/action4_native_sam_region_v001及action4_native_sam_region_clips_v001。

唯一子智能体只读审查确认相机K/外参和numpy生命周期没有发现错接，并发现初始化StartModalities后读取尚未计算的gradient/Hessian诊断错误。主控已修复为首帧computed=false、norm=null，不再读原生未初始化内存；真实ExecuteTrackingStep后才读取、检查有限性及禁用零梯度。7项测试通过，旧记录首帧数值不作证据，旧真实pose不重写。不能把这项诊断修复说成精度改善。

相机线索：复用361条已有当前机身归属的AprilTag矩形观测，无新检测。中心固定、方形像素、未拟合畸变条件下，318个正f²解的焦距中位1301.37px（43个负/无穷解保留），两机分别1302.15/1297.90。全部361条矩形轴正交残差在原2639下中位22.61°，1301假设下2.38°。30帧块等权、主点与焦距联合四折拟合焦距1054.5–1285.4px，留出角残差中位3.75–4.87°，仍有长尾。该证据说明原假设值得怀疑，不是正式相机标定；不把未知畸变写为已证明零畸变。

冻结A0实际验证24实例、72次geometry-head调用，原2639基线R/t严格重放误差0，两个新K同时进入ROI K/射线/相对深度与回原图投影，不做简单t缩放。视觉仍混合，180/2778等可更差，默认K未改。首次GPU试验因缺CuBLAS deterministic workspace在baseline前失败，v002补环境配置，未关闭确定性。来源outputs/action4_marker_intrinsics_feasibility_v001与action4_marker_camera_pose_probe_v002。

旧显示标记CAD绑定来自透视截屏映射，19号四边约20.67/25.52/20.75/24.42mm，不能当物理正方形实测。独立361条矩形方向＋预测完整框平移候选保留固定canonical ID、所有IPPE分支，未使用物理标记尺寸；较小K降低标记重投影误差，但1195/2013等与机身方向仍不一致，marker到body的固定轴假设未准入。不能用屏幕标记低残差或预测框拟合成绩替代机身真实姿态精度。该候选不进入默认链路，见outputs/action4_marker_orientation_camera_v001。

本轮没有重新训练、修改6500数据或默认权重，也没有消费最终test。V5仍为最新已交付完整开发稳定候选；准确方向/厚度、快速转动与全片接续的最终验收仍未完成。

独立审查已由根读取并登记：206个派生相机checksum重建核对、103帧RGB对照一致；SAM相对原生RGB31赢/2输，相对V4仅3赢/24输；初始化6条梯度不合法作证据，后续200条tracking记录190条region开启且非零。原始报告保留，另存AUDIT_STRICT.json将旧初始化非有限诊断值显式标为不可用，不修改原native记录。当前最小下一步锁定1197/track_01，核查实际轮廓搜索线是否把手/衣遮挡当背景，不凭新增权重或平滑发布。

### 2026-10-29 ROI1.15独立75轮正式启动

用户最终选择保留并适度增强PnP/跟踪/平滑，取消专门的PnP有无对照视频；直接检测点和direct_head_report仅留诊断，不删PnP或替换原pose指标。直接点可视化代码有6项测试但没有启动推理/渲染，也未交付该对照视频。

ROI1.15新段67项CPU测试、真实两次更新与完整415键父继承、严格CPU重载、真实RGB/A/K及native SAM辅助/encoder等价经根审阅；1.15几何域全10400单位仅15角出域，标签不夹边。最初CUDA逐bit比较差4.77e-7保留为失败，随后CPU逐bit和state精确比较闭合，不放宽容差。两图4GT的实际执行检查不冒充完整650或正式提升。

根已核验最终gate证据SHA，调用原prepare_corner/freeze生成不可变run p2_a0_corner_roi115_finetune75_v001，冻结运行准入再次通过。execution更新r141，controller PID2537286、torchrun2537325及4rank2537347–2537350已核验真实命令；目前首次数据校验/host同步阶段，尚未观察到参数更新，32eval按现有controller在TRAIN_PROCESS后启动。没有恢复旧stage4。GPU4–7、CPU24–95，总预算72，正式组65，主控7。新30分钟监控action4-roi1-15-75已创建。

固定75epoch、ROI1.15、global48、微批2、四rank，计划16275更新机会与1629次完整eval（无skip条件）。surface/SAMadapter2e-6、pose1e-5、SAMquality/corner/cornerquality2e-5、warmup200+cosine1e-7。保留PnP研究best单独原子alias并在controller/封存核验；部署BEST与quality保持原语义。约2TB可用空间的启动检查已记录，过程中继续监测。当前状态：代码/接口通过；正式任务已启动但训练未完成；效果改善未知。

原生跟踪增量：新audit-only库复现旧3帧和1190–1197 native姿态逐值相同；真实1197 track01的3517模型前景样本中1821在SAM可见区外、3574模型背景样本中1087在SAM区内，短线搜索偏离支持。新显式直接mask似然/unknown域及更宽scales独立对照已执行；三片段206实例，learned-wide支持相容160、unknown-wide172，原native92，V4为113，V5为102。属于同源mask必要支持诊断，非准确率；raw native角步长P95仍约13–19度，未发布替换V5。OwnedNativeCamera减少重复hash但保留两份buffer不变检查，5负例/等价测试通过，实际尺寸buffer检查中位0.271→0.111秒，仅是buffer开销，不是整体FPS。所有这些原生实验均终态，不抢新训练资源。

正式75轮启动已越过数据预检/NCCL并真实更新：首次固化状态step190、epoch1；配置的32个eval已登记，当前请求20、done0、failed0。最新抽查单次更新约1.312秒，GPU瞬时占用曾为35–97%，不把利用率快照等同长期吞吐。authority r142记录真实运行状态；没有完整新eval时不声称效果提升。证据outputs/action4_improvement/stage4_corner/p2_a0_corner_roi115_finetune75_v001/FIRST_LIVE_PROGRESS.json。

### 2026-10-29 首次完整评测失败、修复及完整状态恢复

v001实际达到STATUS261，但step0完整650预测后，scoring入口_scoring_roi_expansion只允许1.5，拒绝1.15；controller按真实失败停止全部owned进程。此前2图实际预测/手工评分没有覆盖此完整入口，是启动前核验遗漏；原代码通过/输入等价仍属各自范围，不能声称当时完整eval链已通过。原冻结source、预测和失败记录均保留。

最小评分修复通过42项回归；复用原650预测、0新forward的真实build_scoring_inputs→calibrate完成，raw All8@2%=36/1300，PnP候选549/1300，quality_ready=false。该结果仅1.15初始权重，不是训练收益，且明显不能据此宣布1.15优于原1.5。证据outputs/action4_roi115_scoring_repair_v001。

完整恢复点last.pt为step260、epoch index1、cursor2064、4rank RNG齐全，原SHA1046c332337a0783c5e62c5f98a0a6cdc07fd6fdfc7cae92c0b2018dd9c100e5。根新增显式评分修复接续合同，旧state/config_id从未改写；独立审查发现4rank排他读锁竞态，已改为共享非阻塞锁并测试。16项恢复/评分/研究best测试通过；新不可变v002的实际CPU构造+strict load确认模型、优化器moments和scheduler完全相等，RNG保持原四份。27个旧评测snapshot四组件逐键相同重导出新配置，全部记录SHA；不是伪造新推理，不重复训练前260次。

v002 controller PID2574672已经启动，run p2_a0_corner_roi115_finetune75_v002，source code SHA8110f42e5545f6f12f89b67ef048f1fe494a3fd2cc54c2e4a92a5d9715d1423b，authority r144。目标仍原75轮/16275更新机会；当前等待真实恢复与下一次更新回执。原第261次更新未入last，仅此一步需重放。监控action4-roi1-15-75已更新到v002，不自动恢复旧v001。关键证据outputs/action4_roi115_recovery_v001/{ROOT_STATE_RESTORE.json,PREFIX_REEXPORT.json,RECOVERY_TESTS_v002.json}。

v002已在真实GPU训练中加载step260完整状态，SCORING_REPAIR_RESUMED记录原config_id未改写、model/optimizer/scheduler/RNG保持；首次固化恢复进度step375、epoch2、cursor7584。新评测队列请求38，done0，failed0。原75轮预算未增加，未宣称新完整eval效果。authority r145更新实际运行状态。


## 2026-10-29 用户暂停、整理与发布

用户明确要求暂停工作、清理无用文件并push GitHub。本轮停止继续训练、eval、模型/跟踪增强和视频实验；只做必要的终止、保留、目录索引、清理和源码发布。当前权威r147，自动检查action4-roi1-15-75已PAUSED。

核验并终止44个本项目进程（含v002遗留loader）；停止回执无剩余live进程。ROI1.15 v003从step400完整状态恢复后仍在历史eval队列容量等待，无新的STATUS更新；保留v002 last.pt的模型、优化器、scheduler、RNG和数据cursor。原75轮没有完成，v003规定eval尚未完成，不伪造TRAIN_COMPLETE或最终best。

v001完整650评分入口未登记1.15导致失败；复用保存预测修复后初始PnP候选549/1300。v002在step400保存后因rank0队列等待、其他rank提前进入NCCL而失败；v003用CPU主机等待令牌协调发布，三项线程测试和真实双GPU超时对照通过，但暂停前未取得完整生产eval结果。单独复用step50预测的PnP候选455/1300低于起点549，不能据raw改善宣称总体收益。

阶段四A07300保持研究封存，650开发All8@2%=1116/1300，quality_ready=false、formal80=false。V5完整2850视频已交付但方向/厚度/遮挡仍未准确验收。后续ROI1.15视频统一入口的41项工程检查不是新模型或完整视频验收。

清理只针对已终态、已完成评估、无外部消费者或保留用途的中间快照，以及可重建缓存；历史失败、逐项成绩、选中模型、恢复依赖、正式数据与并行生成任务资产保留。逐文件动作与实际统计以outputs/action4_pause_cleanup_20261029为准。源码推送仅代表整理后的研究工程发布，不代表项目效果通过。

实际清理：删除0份终态无当前消费者中间快照，逻辑大小0字节；删除6617个可重建字节码/下载缓存文件，逻辑大小4136575111字节。硬链接与文件系统回收会影响实际释放量，未冒充磁盘实测。错误数0。保留当前step400恢复和原ROI115全部未完成eval依赖。

清理最终核验：删除1322份已评估、无现行操作依赖的历史中间快照，逻辑大小239004588654字节；另删除6617个可重建缓存文件，逻辑大小4136575111字节。此前0份为保守第一遍结果，本行及SUMMARY.json为最终实际结果。历史回执对删除快照的引用保留为退役证据，不再用于派发。当前ROI115全部恢复/eval依赖保留。
