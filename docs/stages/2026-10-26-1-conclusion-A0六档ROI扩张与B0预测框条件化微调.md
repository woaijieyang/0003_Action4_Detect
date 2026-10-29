# A0六档ROI扩张与B0预测框条件化微调阶段结论

状态：进行中。用户已明确使用上一阶段正式A0 best；视频开启共同固定track，图片eval关闭track。

目前仅完成计划登记和来源定位，尚未完成六档eval、六视频或条件微调，不宣称比例收益。
对应计划：2026-10-26-1-plan-A0六档ROI扩张与B0预测框条件化微调.md

## 实施进展
- 已按真实完成/选择/seal回执解析并独立strict reload正式A0/B0，FROZEN_SELECTION通过；当前解析到A015770/B09500，但执行不以文档数字为依据，未使用last或旧legacy BEST。
- 六档声明配置已生成，完整输入锁与2.25基线复现前标记不可运行；不是六次eval已完成。
- 已将固定系数传入现有RGB采样与已有SAM辅助裁剪入口，默认2.25不变；图像/视频系数进入输出身份。训练紧裁剪路径仍待条件阶段的显式冻结B0来源资格接入，不绕过旧GT-jitter检查。
- VideoROIContext新增决策字段；pipeline真实裁剪显式读bbox_for_a0，raw/final角点分开保存，无后续network融合时相同。provider从已认证当前mask派生visible框，并由实际历史frame/PTS记录最后可信完整框，不覆盖原B0/track replay文件。
- 19项ROI几何与上下文测试通过，另3项实际张量消费专项测试通过：同RGB/B0改变合法resolved ROI后像素和K变化；SAM维持无B0仍走A0；错session拒绝。测试采用工程模型夹具，不冒充真实定位效果。完整2.25复现、六档eval/视频、track同系数对照和条件微调尚未完成。
- 部分遮挡的质心偏移风险保留为待验证/修复项；不得以新增字段或稳定框图示宣称已解决。

## 六档完整图片eval已完成
输入锁差异检查发现仅evaluation_images/evaluation_wall_seconds两个回执扩展字段，与原报告共享字段全部一致；记录来源差异后通过，未跳过身份或分数冲突。2.25检查在用户要求复用时已完成，未再次启动：650图、All8@2%=701/1300、与原始预测最大坐标差0、B0前向0。后续五档通过同一不可变源码及同一输入锁并行完成，所有档B0额外前向均0。

|系数|All8@2%成功/1300|All8@5%成功|FP002|FN002|
|---|---:|---:|---:|---:|
|1.0|1|17|1299|1299|
|1.1|4|210|1296|1296|
|1.2|48|589|1252|1252|
|1.25|118|773|1182|1182|
|1.5|601|1136|699|699|
|2.25|701|1173|599|599|

目前数值领先2.25；不是更紧必然更好，也不能据此证明ROI过大是漂移根因。视频同帧、GT覆盖/空surface/边缘诊断及同系数track对照尚未完成，不发布最终比例、不启动微调。参考SIX_EVAL_PRIMARY_SUMMARY.json，EPE只在合法匹配中统计并保留完整失败分母。下一步优先六份标准视频，共用冻结track决策；2.25已有完整预测优先复用，仅必要时重新只读编码。

## 六份完整视频及覆盖诊断完成
16个实际推理分片完成六个2850帧域；每帧只读取一次RGB/track，其他五档实际A0，2.25逐帧核对ROI/模型/源图后复用旧预测。B0/SAM新推理均0。六视频共用一次原图/demo解码，只读保存的corners_final，全部ffprobe验证2850帧、2997/100fps、1900×750、duration_ts1140000/time_base1/11988，已传回本地并验证SHA。
GT覆盖诊断完整650图×1300实例×六档，7800行，未回流推理。1.0平均可见像素保留99.99456%，输入目标占用40.077%；2.25保留100%，占用7.9166%。所有档空surface和无目标448支持均0。1.0的448域外角点1593/10400，2.25为0。当前无224热图头，域外统计不能据此屏蔽合法pose投影监督。与既有prepare_online_surface_batch做实际2.25样本核对，surface和448可见支持计数完全一致。退化不能仅归因于可见机身被裁掉；输入尺度/分布适应需进一步证据，不直接宣称唯一原因。
从编码MP4提取并查看4/1195/1950/2400六档同帧画面。第4帧2.25完整范围较合理；1195均保留双机但姿态仍不理想；1950/2400存在某些较紧档局部更合理而2.25偏转的实例，没有统一真实赢家。故2.25只作为既有配置/数值基线保留，不宣布新部署达标。全2850帧按19个固定段输出诊断标志，区分真实转动与错误仍需视觉复核；不把标志数称为抖动改善率。
后续保持本任务优先：补齐同系数B0-only/track-assisted对照及预测框训练清单与分布分析，按证据进行唯一预测框适配；不因小ROI失败就把训练/eval来源gap搁置，也不转去其他增强阶段。

## 2026-10-26 固定1.5适配检查进展（未正式训练）
冻结B0训练预测一对一匹配得到10400个合法关联；全训练裁剪资格检查尚待完成。实际两次更新验证第一次因配置a0_variant误写baseline在构建前失败，未更新权重；修正为已有a0_baseline枚举，保留原失败日志。
actual_updates_v002实际两次更新成功：surface/pose均有梯度并变化；backbone参数/buffers不变，冻结特征提取无梯度；严格重载的rotation/translation/camera_corners/corners/depth逐项相等。这是代码集成证据，不是50轮完成或效果改善。
独立step0/step2完整650eval已启动，结果待实际完成。后续track路径比较遵循计划最新修订，本次不宣称全部部署gap消除。

### 本轮实际新增结果
- 16项数据裁剪、LR端点、评分/全局分母/零监督与物理误差报告测试通过。
- 独立完整650图：原best在1.5复现601/1300；两次工程更新为596/1300，不认定改善；B0没有重新forward。
- 全部10400预测训练ROI实际读取审计通过：10400有合法pose与surface监督，无空surface，无112域外点；可见点61434、隐藏点21766。最大surface投影一致性误差2.4841e-6个112网格像素。
- 双GPU现有训练器完成96实例×2epoch、4次更新，step2受控checkpoint恢复到step4；异步完整eval仍在消费。此为集成检查，不是正式50轮。
- train/eval框平均IoU为0.9774506/0.9565005，中心误差除以GT对角线均值0.00471835/0.00848601，证实相同框来源仍有剩余分布差异。统计见TRAIN_EVAL_PREDICTION_DISTRIBUTION.json，不将bbox面积称为可见mask占用率。
- 正式50轮配置已准备为草稿，等待本次集成全eval完成后以其不可变源码及实际输入资格放行。默认模型和已发布数据未改。
- 后续先完成固定best、固定1.5、同一RoiSpec/采样/K/forward的登记开发片段双路径比较；不以SAM紧mask替代完整机身ROI，不宣称已覆盖跟踪输入分布。

### 正式50轮已经启动（2026-10-26）
双卡集成的TRAIN/EVAL/CONTROLLER完成回执一致；完整eval step0=601/1300、step4=608/1300，step4困难组门未通过，不作为正式父权重。正式run仍从封存A0 best初始化新优化器，复用已验证集成源码快照。
run：p2_a0_roi15_prediction50_v001；GPU5/6训练，GPU4/7各4个eval进程；CPU61（训练68–95，eval36–67，controller35）。固定10400预测ROI、50epochs、globalbatch48，计划10850次更新、435次完整eval（无零监督跳过条件下）。实际已执行有效更新；不把已启动写作完成。
不可变身份：runtime/action4_runs/p2_a0_roi15_prediction50_v001/RUN_SPEC.json。
输出：outputs/action4_improvement/roi15_prediction/p2_a0_roi15_prediction50_v001。
八worker并行吞吐将由实际队列消费核实；已有无损背压，不丢eval。不修改运行源码，不覆盖默认模型，不改变训练框来源。

### 后续统一RoiSpec接口（工作树，未替换正式50轮快照）
共享prepare_rois现在实际消费不可变RoiSpec的完整机身范围、单次扩张和连续像素仿射；训练准备及B0-only/track-assisted输出记录真实消费的spec与K。已有采样数值、padding、归一化保持原样。28项接口/训练输入检查通过，受影响ROI replay/视频入口回归另见outputs/action4_shared_roi_spec_v001/REGRESSION.log。
首次测试包含随机图布局夹具错误，以及旧来源字段逐字比较不兼容新增追溯字段；已修正夹具，保留旧数值字段精确相等断言并新增bbox_for_a0与raw provenance断言；原失败日志保留。当前训练相关四文件快照SHA仍全部匹配RUN_SPEC。
这里只证明代码通路和输入消费，不证明SAM跟踪完整框估计精准，也不证明八点更稳定；固定50轮best之后的登记片段对照仍待执行。

### 固定best后的双路径对照评分准备
新增src/eval/action4_roi_path_comparison.py及6项CPU测试通过。两路径必须绑定相同权重/源码/相机/像素域/decoder/1.5，完整frame/PTS/RGB域一致；有效八点必须固定ID0–7，不允许重新排序。
共同实例先按同一当前B0候选源记录（这本身不证明物理身份）；SAM补回、另选候选、B0未采纳分别保留。未复核的补回不计作“正确恢复”，缺帧/重复实例/改动八点的结果拒绝。无真实姿态GT时只报告原始输出差异，不称定位误差改善。实际封存best的开发片段推理尚未执行，须等正式50轮及全部eval完成。

### 50轮封存兼容性修复
发现scripts/action4_seal_completed_role.py仍硬编码正式运行至少80轮。现仅对已登记action4.fixed_roi_prediction_adaptation.v1增加严格50轮、1.5预测框、冻结baseline A0、25step eval分支，并验证执行准入的源码/清单/父模型绑定；其余阶段原80轮限制不变。
38项封存相关测试通过，其中真实运行的只读check仍返回WAITING_FOR_COMPLETION，没有提前封存。未来回执将保留1.5/B0/cache/score合同，不自动认定track分布已经验收。正式训练源码不受影响。

### 开发片段边界核查与训练中期观察
正式训练实际PID591866/591867核查存活；观察时step685、第4轮，28请求/21完成/0失败，当前训练候选best step525=715/1300。该数值仅为进行中的开发集选优，50轮与所有eval未完成，不提前发布。
固定双路径对照已登记6个原计划开发核心片段，共720帧，每段最多前后40帧上下文，裁至dev_A/dev_B区域内；不包含guard/locked。配置configs/eval/action4_roi_pair_devclips_v001.json。
发现旧fullvideo reanchor运行器要求2850帧并继承全片session；不能将其缓存伪称独立片段冷启动结果。后续需复用既有区域sequence controller/backend创建独立片段会话或严格同身份缓存，保留seed失败；本轮未启动SAM或新A0片段推理。

### 两个子智能体独立审查及资源核查
封存审查指出选择合同、外部ROI引用、精确650域三处缺口；主控已修复并执行40项测试，独立复核确认关闭。实际运行尚未封存；相机字段通过eval annotations整体SHA绑定，不称新标定，不重复全量RGB哈希。
10秒实际采样GPU4/5/6/7平均利用率13.7/38.2/24.9/9.1%，显存不是约束；41请求34完成7pending，7项均已被worker领取，0失败，未达12背压上限。不能据显存剩余直接断言增加worker会提速。
源代码性能审查发现每rank每update至多120次loss日志scalar主机转换、逐实例GT finite同步及细粒度head调用等候选；STATUS.seconds不含loader-next/checkpoint/publish，不能据其排除供数等待。尚无实测加速结论，运行快照保持不变。下一步先独立测量同算术实现，不改batch、micro、精度或数据来源。

### GPU日志合并候选实测（不采纳）
两个子智能体分别完成封存复核及性能源代码审查；性能候选仅新增独立loss logging helper和测试，未接入正式trainer。
主控用真实两个预测ROI的A0 loss在物理GPU4独立验证：12micro日志Python浮点bit一致，实际A0梯度不变，optimizer更新0；原日志中位0.60011ms/micro，合并候选0.60734ms/micro。并发GPU负载下未见稳定收益，不采纳、不切换健康训练，不把减少拷贝次数当作已加速。
开发集进行中候选step1325 All8@2%=773/1300，原2.25参考701；高遮挡29/192 vs16/192，近邻12/20 vs1/20，小目标28/127 vs27/127，大小不等33/122 vs17/122。仍非50轮完成/真实视频验收，保持训练配方。

### 预测校准与可信完整范围的来源语义修复
旧ReanchorROIProvider直接把calibration_frames最后一项写为_roi_last_trusted_full；现有tracker的自动seed/一致性校准并不证明完整机身范围可靠，因此该命名过强。
工作树现在单独输出last_prediction_calibration（预测一致性证据，verified_full_body_extent=false），不凭该历史填写last_trusted_full时间；新字段进入ROI决策摘要校验。帧时间通过global frame映射查询并复用映射，未改变bbox_for_a0数值或训练快照。29项针对性与受影响接口检查通过。
历史视频/回执保持原状。本修复只校正证据语义；部分遮挡质心引起的完整框偏移仍需在封存best后的独立片段实测与算法检查中处理，不能据此声称跟踪准确性提升。

### 静止机身、遮挡质心变化反例已实际复现
输出outputs/action4_track_occlusion_counterexample_v001。固定完整机身框[35,35,85,75]，前10帧完整可见并有高分B0，随后B0缺失而SAM可见mask仅保留原右半边；物理机身位置始终不变。
sam_primary_v1和sam_b0_fallback_v2均在最终帧保持sam_supported/current_support=true，却将完整ROI中心移偏约10px。该CPU几何夹具证明当前mask质心加历史offset机制存在遮挡诱导假运动；不是原视频误差测量，也不是训练效果结论。
后续必须区别可见mask形变与刚体运动，保留完整范围及来源不确定性；需要同时覆盖静止遮挡、真实平移及平移伴遮挡反例，不能仅通过增加平滑或删除困难帧制造修复。当前50轮训练、已有视频与默认权重均未改。

### 独立clip输入实际准备完成
outputs/action4_roi_pair_clip_inputs_v001已建立6段隔离PNG目录、REGION与B0清单：1080上下文帧、720核心评分帧。PNG使用项目内samefile硬链接，不重新抽帧/编码；全量1080 local/global/PTS映射和冻结B0 pool字段逐项与原始来源核对一致。初次helper导入因换行字面量SyntaxError失败，尚未创建输出；修正为chr(10)后实际执行并复核通过。
本轮模型调用0、SAM session0、A0调用0；未继承原全片跟踪状态，且不声称已完成clip跟踪。来源像素域与time_base从已绑定原region读取，不另猜默认值。后续仍须等待本次50轮best封存，按native sequence入口重新核验图像SHA并独立建立会话。

### 用户要求30min闹钟与并行接口实施
已创建本线程heartbeat action4-50-eval，每30分钟核验当前50轮训练/eval并汇报真实新证据；不再逐分钟手动轮询。仅在故障或明确依赖需要时额外检查。
并行任务：独立clip原生session结果到VideoROIContext的适配仍在实现，待主控审核；mask运动证据模块已交付并经主控阅读和10项CPU测试复核，字段明确为visible_support_change而非RGB外观判断。纯平移具有shape-compatible正向证据；静止遮挡、等面积形变不会直接当作物理刚体运动；该诊断尚未接入生产tracker，不宣称修复完成。
主控补齐完成50轮后配套A0/B0开发组合入口src/eval/action4_roi_pair_selection.py及薄scripts/action4_freeze_roi_pair.py，4项metadata测试通过，拒绝旧ROI/错B0/旧A0。实际入口必须调用既有独立seal verifier，且后续推理源码仍须单独冻结；没有提前创建正式pair或覆盖default。

### 独立clip适配器主控复核闭合（离线双向功能）
原生native_tracks结果到现有VideoROIContext的适配代码已完成，主控发现并修正隐式legacy几何模式、npCHW与真实采样入口不兼容问题。显式tracker配置、真实torch CHW与K112消费，以及两个独立session各obj0的归属均有CPU反例。
独立复核随后指出FAILED可能早于finally/CLOSE、失败原因/native状态丢失、双向回放preseed未来信息三个问题。现要求真实已退出Popen回执及config/mode argv绑定、CLOSE released确认；公共输出保存带SHA的失败原因、全部core与逐track状态，明确offline_bidirectional_clip/causal=false/preseed_replayed。消费端实际模块身份纳入replay identity；完整推理源码仍须冻结。
主控最终42项相关测试通过，回执outputs/action4_clip_replay_preparation_v001/ROOT_ACCEPTANCE.json。该入口适用于原计划SAM离线双向功能，不得代替前5/10帧因果B0辅助track；对照评分器已拒绝混用。真实SAM/A0片段推理、因果clip入口以及遮挡几何修复仍未完成，不能宣称整条视频效果已通过。

### 因果forward主链与同源B0对照入口
既有sequence/backend/ClipReplay新增显式forward分支，保留原both路径。实际forward请求、native clip local→tail责任local→global/PTS映射、seed前initializing无ROI、缺尾/乱序/缺响应失败均有producer synthetic backend验证。独立review发现point种子可能来源于同帧双向mask，已补prefix seed-only discovery/config/resource/raw操作/close身份门并复核关闭。
主控54项组合回归通过。新src/eval/action4_roi_pair_inference.py两路都取已认证clip B0_POOLS.single_frame_selected，绝不重跑B0或分别挑框；同图同框RGB/K完全一致，选不同ROI则实际输入变化，track初始化时保留B0-only输出但不伪造track A0。
全2850帧历史源对比发现B0候选池与旧视频原始detections有8帧grid选择不同，同grid框最大差0.659424px；不能混作完全同源。旧视频仍为历史参考，不改写；新paired两路固定同一候选池身份。
源码/验证回执outputs/action4_clip_replay_preparation_v001/ROOT_CAUSAL_ACCEPTANCE.json。尚未运行真实SAM/A0片段或封存当前50轮best；因果API/prefix限制不是vendor全部内部算子的独立在线证明。异步新增seed、遮挡质心几何偏移仍是明确限制。

### 后续启动与父模型输入兼容性准备
新增显式冻结sequence启动入口，检查只读config/源码manifest、未登记可导入文件、CUDA UUID及CPU affinity，然后才进入原sequence模块；3项metadata启动检查通过，不代表native模型执行。
现有注册SAM资源门此前固定8GiB图片worker，不能复用已验证的32GiB视频额度；新增仅由显式authority启用的8/32档、每GPU总额度<=32GiB及8GiB余量检查，7项资源测试通过。configs/execution/action4_causal_clip_resources_v001.json为未激活草案，拟4卡各1sequence worker，当前active r102及正式训练完全未变。
父模型输入契约增加检查：新1.5预测框适配parent不能静默回到GT/2.25，也不能只声明roi_source却没有实际dataset consumer；11项相应回归通过。后续SAM增强数据路径仍需正式实现/验证后放行，不把阻断错误启动称为后续训练已接通。

### 配置准备器与封存权重检查补齐
prepare_discovery/prepare_forward已实现：discovery复制当前实际测试清单代码、复用官方source/asset，forward只继承完成prefix discovery的冻结执行代码与实例point来源，均不dispatch。独立review发现静态资源注册未交叉检查policy禁用GPU，主控补齐禁止/允许GPU及CPU总预算拒绝，17项组合测试通过。生产配置仍未准备/激活。
封存工具新增真实selected backbone参数和buffers对父模型逐项tensor/dtype/shape一致性检查，仅fixed50协议强制，旧seal保持兼容；41项封存测试通过，并用已有真实step2工程checkpoint验证一致。该工程结果不代表正式50轮best已检查；正式封存时会再次对实际最终best执行。

### 第一段真实配置准备与六段种子预检
使用实际官方snapshot、权重receipt、冻结clip/B0输入及主控已执行的针对性测试清单，成功准备dev_00 discovery只读源码/config。没有执行模型；active仍r102，所pin的未来资源policy仅为未激活草案，必须等50轮封存与现场资源核查才可启动。
6段真实B0缓存按现有前5帧score>=.9/quality>=.8确认得到种子数2/1/2/0/0/0。这些只是provisional B0种子，不证明语义正确。0/1种子反映初始化覆盖限制，不能强补第二台、忽略失败段或归因于A0 ROI。后续仍需保留完整core分母并单独评估初始化/重获策略。
证据outputs/action4_causal_clip_dispatch_preparation_v001/ROOT_PREPARATION_REVIEW.json与REAL_B0_SEED_PRECHECK.json。

### 不降阈值的滚动初始化诊断与真实画面核查
复用冻结B0候选池，保持5帧/.9分数/.8质量，只对连续窗口做独立诊断；未改变生产seed策略、未调用模型。dev_01在808帧出现两候选，dev_04在2489帧出现两候选，dev_05在2778帧出现两候选；dev_03到1445帧仅出现一候选、整个context没有双候选窗口。后续出现候选说明固定最初5帧会漏掉初始化机会，但不等于已经安全实现动态add-track。
主控查看9张真实帧叠图（4/784/808/1144/1445/2477/2489/2698/2778）：所选位置在腕部相机区域，不能据缩略图认证完整机身边界或全部窗口的语义正确。未将人工观察作为训练GT或改种子选择依据。需在后续有限因果初始化/重获实验中保持ID与native确认，不能只选已知好帧回填过去。
证据ROLLING_SEED_DIAGNOSTIC.json、SEED_REVIEW_CONTACT.jpg与ROOT_SEED_VISUAL_REVIEW.json。

## 六面辅助锚定新增方法（用户授权，尚未运行）
已只读核查冻结CAD canonical8与轴定义，以及现有A0 112网格XYZ/mask/region/pose surface输出。由canonical8派生固定6面，验证8角、12边、每角3面/每边2面、向外法向及平面性；配置configs/model/action4_six_face_anchor_contract_v001.json，证据outputs/action4_six_face_anchor_preparation_v001/TOPOLOGY_CHECK.json。
主稿阶段五内加入F子阶段，保留八阶段结构并明确在CoTracker之前实施。方案区分虚拟包围盒面与真实物理面片、当前可见观测与隐藏推断、五帧历史与预测自反馈；首先F0/F1/F2固定输入对照，再全2850帧开发复核。没有启动面跟踪/新训练，没有宣称准确性或稳定性改善。

### 六面几何模块与真实CAD证据推进
src/model/action4_face_geometry.py实现单SE3下固定ID角/面/边投影、front/back/grazing/outside诊断，明确不代表观测。主控12项测试通过，补齐真实surface snapshot的精确float32 canonical编码兼容（不放宽任意扰动）、正extent和未标定pinhole说明。
复用已经完成的旧parent在1.5下保存的1300实例原始R/t，不调用模型，六面共享角投影与原八点最大差0.00028088 native px；几何front-facing数量1/2/3分别38/374/888。这是同一姿态投影代数一致性，不是预测精度改善。
实际CAD法向审计与六向first-hit有限采样显示：虚拟包围面不能直接当物理支持平面，front方向有约-10.905与-15.954mm多层；normal方向会误包含内/背表面，first-hit排除了具体54三角反例。首样本1318个XYZ/face_indices与同源mesh三角几何一致，但仍未建立完整语义patch registry。
已核查src/data/action4_surface_mesh_binding.py存在，但其无序face集合比较不证明索引顺序；phrase2实际生成器直接以相同r3 mesh.faces顺序构BVH。所有审计在outputs/action4_six_face_anchor_preparation_v001，未改mesh/发布数据/当前训练。

### 六面输入、物理patch候选与五帧历史组件
真实mesh连通/平面候选生成后，独立复核确认820簇、原triangle顺序/面积/残差一致。root又补齐逻辑面轴与plane normal轴一致性，保留侧向first-hit为观测证据但不误标FaceID，同时拒绝bool/float view ID。v001原数据及源码保留，v002有85个带逻辑候选的patch，仍全部semantic_gt=false/exterior_verified=false。
FaceInputs桥认证实际snapshot/RoiSpec/model/frame，恢复未夹紧米制XYZ并保留mask/region logits，标记A0 prior非独立面观测。6个旧已完成parent的真实视频缓存（ROI2.25）桥接通过，错误1.5映射会拒绝；不称当前新best验证，缺body generation仍保留None。
五帧history实际使用全部保留观测的真实PTS相邻段速度median，最后真实观测为外推锚点；prediction/missing不入库但推进watermark/代状态，避免晚到过去帧进入因果历史。仅诊断prior/innovation，不直接输出或平滑8点；同pose八点保留共享evidence group。独立复核通过，并说明最终caller仍须认证source/group。
root42项组合测试通过。首次FaceInputs单测期望数组未broadcast造成shape失败，已修正测试形状并保持精确数值比较，失败log保留。尚未完成真实面观测、共同刚体求解或全视频效果验收。

## 原50轮已完成封存，用户追加10轮已dispatch
实际完成50epochs/10850updates/435eval/零跳过。best10125 All8@2%=891/1300，All8@5%=1228/1300，P95(valid matched only)=16.4291px；原2.25分别701/1300、1173/1300、22.1371px。全部已登记组相对原2.25未退化，但小目标+高遮挡仍0/40，真实视频未验收。
独立immutable重载与真实冻结backbone参数/buffers比较通过，seal位于models/candidates/action4_improvement/roi15_prediction50_completed_20261026；A0 SHA d86147adaf8cd76506ae9470e8fea9d8e23d0aa3a89800b4b0bd1a84146e157a，default未动。
用户随后授权仅追加10、batch64、eval10。新合同/原50兼容性独立审查通过；实际4GPU integration完成1312units/21updates(含32尾批)、step10恢复、4次完整eval，step0精确复现891；工程后各步883/886/887，故不使用工程权重。正式段从原sealed50best重新初始化optimizer。
正式run p2_a0_roi15_best_extension10_b64_e10_v001已dispatch；目标1630updates/164eval；此处尚不声称新段训练完成或改善。证据outputs/action4_roi15_extension10_v001。


### 2026-10-26T23:23:52.469353+08:00：当前best完整视频及六面原生试验
- 用户停止追加训练；使用已封存A0追加10轮best step1450（All8@2%=900/1300）和B0 step9500，固定ROI1.5。20轮未启动、已取消；不覆盖默认权重。
- 18个A0分片运行于GPU4–6、54逻辑CPU，全部2850帧完成；每分片75–131秒，显存张量峰值约0.513GiB。已有B0/SAM重放复用，新增B0/SAM全片推理调用为0。
- 完整对照视频：outputs/videos/20261026-230106-A0best1450-roi15-B0-SAM-e1p50-track/comparison.mp4。2850帧、1900×750、2997/100 fps、time_base=1/11988、duration_ts=1140000已核验。此为开发结果；demo不是真值或性能上限。
- 全片数据流核查：5700实例，4973当前B0关联、727由当前SAM支持续接。所有实际A0 bbox与决策bbox完全相同；ROI/K采样链已消费；raw八点与保存final相同，无渲染补点。K浮点重算差≤3.06e-5、canonical8投影复算差≤0.00022原图像素，只证明几何/数据流一致，不证明定位准确。证据：outputs/action4_fixed_best_video_v001/ROOT_DATAFLOW_REVIEW.json。
- 目视frame1195、2400两机均在腕部区域；1195仍有姿态方向/厚度偏差。未据此声称全片身份或角点精度通过。
- 六面真实输入已使用当前best重新导出6帧12个surface与当前body mask。face ID与corner ID不同命名空间，12面/16角点；project generation与SAM native generation分开。不是将旧snapshot改metadata冒充新输出。
- 首轮真实SAM面seed/15帧试验已执行两机4面假设：3个被现有污染/机身外支持门拒绝；1个仅跟到顶面按钮局部。ROOT视觉复核未通过整面效果，不进入几何融合。原始结果完整保留于outputs/action4_face_native_seed_pilot_v001。
- 发现物理patch身份与logical FaceID提示资格混淆：多层同向CAD表面会造成严格patch唯一性排除大量面内点。后续只允许同FaceID一致性作为SAM假设提示；物理depth/XYZ仍未知，不据此开放PnP或单平面homography。
- 实际prefix→forward曾因NumPy1/2 float32标量提升差异在精确复核时失败；按已绑定producer版本重放原运算，全部字段仍exact比较。真实两track通过，原raw/source未改。新dev00因果forward已exit0，精度另核查。
- 代码通过、native执行、完整视频生成与面跟踪效果分别报告；六面/材料几何/CoTracker尚未完成。


### 2026-10-26T23:47:27.261805+08:00：真实面提示对照与B0-only/track-assisted消费复核
- v1/v2/v3原生结果分别保留，均非完整六面验收。v2使用logical FaceID一致性，保留多层物理depth未知；v3仅把正提示点从2个局部点改为4个相隔80native px的面内点（当前frame4开发试验参数，不作全视频通用阈值）。
- v3两个独立session/obj0各完成15帧真实forward，30条响应完整且关闭；首末帧可见屏幕区域持续被支持，仍不是整个rear机身面，不能宣称六面/三维姿态或隐藏点精度已通过。部分非连接小区域污染保留。输出outputs/action4_face_native_seed_pilot_v003。
- 全片姿态变化诊断记录1060个相邻实例对>45°触发项（5698对分母），不是抖动错误率。实际检查2258→2259：图像机身运动较小，而A0旋转变化约178.89°，确认至少此处存在明显姿态翻转，单纯bbox稳定不能解决。证据ROTATION_FLIP_2258_2259.jpg和TEMPORAL_REVIEW_EVENTS.json。
- dev00独立因果120核心帧的真实双路径A0推理完成，使用相同B0池/当前best/ROI1.5。B0-only240输出，track-assisted232输出，前4帧初始化明确保留，无SAM-only补回实例；不得把该简单片段当漏检恢复证据。详细来源对应及原始八点变化见outputs/action4_fixed_best_roi_pair_dev00_v002/results/PAIR_DIAGNOSTICS.json。
- paired v001首次认证失败仍保留；NumPy1/2重放helper同时修复prefix与完整ClipReplay两处精确诊断，v002使用新不可变consumer源码，不重跑SAM或修改原raw。
- 主控组合50项CPU测试通过（unittest，PYTHONPATH=src:tests），覆盖身份/提示/原生接口mock/ROI实际消费/跨NumPy一ULP拒绝/资源dispatch。CPU合同通过不替代原生面精度验收。
- 完整视频已复制到本地workplace，SHA与服务器5c998d1f82ac5eaf94b5ef297c6e742d9aa13942357be2768558bb2e702c300e一致。


### 2026-10-27T00:45:21.602580+08:00：五F当前图像证据、单刚体候选与先验冲突查询
- 用户将后续子智能体上限改为1；当前仅一个已有agent分别负责CAD/源码独立审查，主控执行代码及真实验证。未追加A0训练，未改默认权重/6500数据。
- 新增src/model/action4_face_motion_evidence.py及src/eval/action4_face_motion_eval.py。保守排除屏幕凸包及margin，仅在body内屏幕外侧寻找真实RGB特征，前后向LK+RANSAC输出图像配准证据，不能直接升级为CAD姿态。
- 真实两机各14相邻帧对，28对中仅5对满足当前图像配准门（track00=2、track01=3）。主要问题是排除动态LCD后多数帧仅有1–10个可用特征；未降低12点当前pilot门去制造成功。该门属于本次图像配准pilot，不等同CoTracker材料PnP的12点准入。
- 新增src/model/action4_face_edge_pose.py与src/eval/action4_face_edge_eval.py：真实CAD环曲线、RGB梯度法向对应、单SE3左扰动，translation用CADdiag缩放。无先验残差，观测Jacobian独立检查。实际30实例候选全部未准入；normal residual虽下降，仍有错误面边缘、支持域离开和局部范围超限。没有修改最终八点。
- frame4投影图显示rear环顶部落到顶面白标签附近，说明两维nearest edge即使方向相似仍可匹配错误部件。新增默认关闭的方向多邻居和LCD附近带约束；真实两机方向冲突消除但候选仍不通过。不同观测域不比较残差声称准确率提高。
- CAD审查：rear patch078为真实窄环，815为带圆角填充面；没有认证LCD active四角。top button component10与canonical Face5同坐标系，但其XY近圆轮廓不是固定三维平面圆。证据REAR_SCREEN_PATCH_EVIDENCE_v001.json、TOP_BUTTON_EVIDENCE_v001.json在outputs/action4_six_face_anchor_preparation_v001。
- 修复候选查询的先验闭环：显式support_policy v2允许当前surface/body支持与A0可见性先验冲突时发起native查询，旧默认不变；仍不允许空支持、跨FaceID歧义或把查询直接当几何准入。新测试用真实forward式fixture验证错误姿态先验不阻断当前支持，空XYZ仍无proposal。
- 真实v004仅查询先前被A0背向先验阻断的track01 Face5，15帧原生输出完整、session关闭；首末帧观察到稳定的顶面按钮局部。它是part-on-face支持，不是完整Face5或已认证3D对应。outputs/action4_face_native_seed_pilot_v004/ROOT_VISUAL_REVIEW.json。
- 本轮组合25项CPU测试通过，另外原生15帧查询与30个真实RGB几何候选均实际执行。native成功、单元测试、几何可用与实际角点改善继续分开。
- 只读对照OpenCV官方LK示例与DLR M3T/ICG源码；成熟region tracker沿轮廓法向构对应线和概率，并有限外层重建对应。内部黑bezel两侧同属body，不能直接套整机foreground/background直方图。官方源码浮动master未固定commit，未编译或声称复现；详细来源SOURCE_COMPARISON.json。
- 后续优先让可见屏幕区域/顶面部件的真实观测共同约束同一SE3，先解决当前物理边对应与局部初值问题。单面低残差、mask更平滑都不作为角点准确证据；材料/CoTracker依赖阶段尚未完成。


### 2026-10-27T02:17:02.513403+08:00：观测域纠错、噪声隔离与完整视频面内特征审计
- 原normal profile把候选中心归属与RGB采样有效域共用allowed，导致真实边界另一侧RGB被置零。新增显式profile_sampling_domain=valid_rgb，仍保持候选中心/peak ownership/法向/距离/几何门，另保存未置零RGB profile与截断理由；旧默认保持。尚未把所有外轮廓中心的边界角色建模完成，不能声称全边界已可用。
- 实测旧初值frame4单峰有效数11/17→16/20，仍未达24。normal_profile_mixture保留全部合法峰及FWHM不确定性，直接minimize混合NLL；观测信息用真实投影Jacobian与marginal曲率，不用sqrtNLL或正则补秩。真实4/6px宽度有限候选为16/19、21/21条，仍未进入优化；不能将CPU多峰夹具通过当真实收益。
- 新增view-dependent真实CAD mesh边候选（含全mesh遮挡），明确是A0姿态条件下的几何proposal，不是固定三维圆或图像观测。按钮component10候选投影显示A0初值使按钮接近侧视，和真实可见区域有偏差。
- 对current screen/button mask拟合仅生成有限初始化，不替换pose；发现少量远离提示点的连通分量强烈干扰平方误差。新增seed-only positive-prompt component eligibility，原mask及原生质量分母不改；未知分量不作为part几何观测。track00屏幕/按钮未知像素115/41；track01为1876/23。部分区域小试proposal分数改变不称真实精度提升。
- 纹理方向怀疑已被正式审计推翻：实际UV包含1−u，marker与button本来处在object X异侧。5个正式render检测的正确UV中心误差0.129–0.689px，无flip为65–175.6px；不改发布数据/纹理/模型。
- 原始RGB及真实素材可解码36h11 ID18/19。导出的CAD四点仅是existing_appearance_texture_mapping_estimate；通过component6全mesh first-hit及barycentric，非真实metric标定/GT。NONE与SUBPIX-inverted各自身份独立，后者素材角点内移且synthetic四角误差更大，不能混配。
- 完整2850帧只读特征审计已完成，24CPU worker，现有tracked body windows内：track00检测ID19 146次、ID18 1次；track01检测ID18 225次。115帧两窗有标记、142帧一窗有标记、2593帧无标记。它不能覆盖或代替全片B0/A0/track；跨窗ID18须进一步查重复窗口/身份冲突。outputs/action4_display_marker_full_audit_v002/results/COMPLETE.json。
- v001完整审计因快照仅复制OpenCV Python wrapper而缺native binary，在0帧时失败；失败保留。v002使用已登记项目内shared native runtime并在fork前加载，project code仍冻结，配置显式NONE variant，未重跑成功样本。
- 原视频4/18/1195帧已用同variant估计CAD点运行IPPE双解，仅保存候选和固定canonical8投影，未覆盖baseline。低标记重投影不证明实物八角点准确；缺marker保持缺证据，尤其1195一机无标记，另一机两解误差接近。outputs/action4_marker_pose_probe_v001。
- 本阶段继续未准入最终角点融合，不追加训练，不使用最终test，不覆盖models/default。当前视频仍使用上轮已交付best+A0/B0/SAM实际ROI链。


### 2026-10-27T05:31:52.774265+08:00：完整开发增益、真实面失败边界和后续ROI合同
- 完整650影子几何测评已登记results/results.md：固定A0 step1450/B0 step9500/ROI1.5，baseline复现900/1300，候选955/1300；修复59、退化4，FP/FN均400→345，有效匹配P95 16.196564→15.406371px。候选修改565个预测；高遮挡44/192、小目标高遮挡0/40未变。GT只在保存候选后计分，不进入worker白名单；没有新神经训练或B0/A0推理调用。
- 真实marker候选独立区域审查与R/t因素分离不支持统一替换A0。R-only可改善部分body外包一致性，却可恶化按钮位置；完整IPPE有40–51px整体偏移反例。低四点误差不是整机pose通过。ROOT_SELECTION_REVIEW保留候选、不更新默认或视频输出。
- 原视频372条crop观测经完整当前支持归属检查为371条独立观测：361条归属一致，8条在seed前缺native支持、2条支持不足。1315是同一quad被重叠窗口重复读取；当前mask覆盖track00=0、track01=1，正确归并到track01，未交换body IDs。所有2850帧、未归属和缺失保留。
- 修正F1净位移参数化的范围语义：新object_origin_delta允许纯旋转保持物体中心，保留legacy默认及等效left增量。真实frame4四个对照仍未准入，说明该接口修复不等于姿态改善；条件数仅同chart比较。
- 用GPU6/7补做808正侧面与1195近邻遮挡source-XYZ策略真实15帧试验，获得局部lens/display/label区域跟踪；没有证明其canonical FaceID完全正确。
- projected-face-region新策略不再要求denseXYZ落入plane容差，并由当前几何polygon∩native body支持提4个分散正点。实际808/1195双机共4jobs/9面假设，GPU4–7各一进程、CPU64–95共32逻辑核并行完成。2个数值seed门通过，但图像复核没有认证完整物理面；出现整机吞并、负点污染、越界及错误先验区域。所有raw和失败保留，不改门以宣称成功。ROOT_VISUAL_REVIEW在outputs/action4_projected_face_native_v001。
- 后续SAM消费者原有enrich_training硬编码2.25已修为实际RoiSpec；主控9项针对检查通过。RGB/A448/A112/K112逐项一致，prediction缺身份或错association拒绝；原baseline禁止SAM和新阶段训练准入guard保持。旧5200/650 native original与当前B0同源，已完成light/blur不重算。
- 主稿阶段3/4/5相关入口已就地修订，继承实际best＋预测ROI1.5，旧GT-jitter/动态扩张为历史候选而非当前默认；stage4旧六卡假设改为GPU4–7实际分配。没有新stage3训练或adapter best。
- 已异步询问“停止更多A0训练”的范围是否包括后续独立增强分支；答复前不启动新训练。当前继续跟踪、几何及接口工作；该范围澄清不构成已有模块效果通过。


### 2026-10-27T05:57:46.742561+08:00：SAM 当前预测 ROI1.5 真实输入消费验证

新增 src/eval/action4_sam_prediction_roi_probe.py，复用实际 native payload 校验、prepare_pose_rgb、resolve_training_roi 和 prepare_sam_roi_inputs；未创建部分验收的生产 provider，未修改原缓存准入规则。4张真实合成训练图、8个实例均通过 RGB tensor 完全相等、A448/A112 和 K112 一致、双机请求换序后各自输入保持一致、缺mask时RGB/K不变且available=false。包含 continuous_bounds 顶边 -63.834px 的实际图外补边。耗时 9.756秒。

所检8份原生mask逐一核对缓存SHA、producer mapping、不可变提示记录、native completion/result、session/obj/generation及实际像素身份。复用既有输出，未调用B0/A0/SAM模型、未训练、未重做5200图全库payload验收。证据：outputs/action4_sam_prediction_roi_probe_v001/CONFIG.json、RESULT.json、SOURCE.py。

范围：这是有界真实数据接口验证，不是全量cache/训练准入或精度提升。10400实例来源对应关系沿用 CURRENT_PREDICTION_ASSOCIATION_PARITY.json 的全量核对；stage3正式consumer协议、后续训练授权和真实六面身份/共享姿态效果仍待闭合。最多一个子智能体并行，当前未追加派发。


### 2026-10-27T06:23:41.794286+08:00：五F共同刚体种子拟合及输入归属修复（未准入）

在现有 action4_region_pose_initializers 中增加清晰帧多区域单SE3候选，复用原生SAM整机及按钮观测、固定CAD和当前A0原始姿态。先保留无marker组及0.25/1.0组权重负对照，再固定1.0权重比较已有IPPE0初始化。只检查实际frame4两实例，未训练、未重新调用SAM、未生成整片或修改默认。

无marker条件曾出现整机预测区域IoU提高但marker投影更差，证实轮廓拟合不是角点精度证明。固定IPPE0初始化后，track00整机预测区域一致性IoU 0.827→0.888、按钮0.178→0.805；track01分别0.782→0.866、0.141→0.691。marker RMS45.59→19.54及44.02→16.68px参与拟合，不能称留出或真实姿态误差。track01触平移分量边界，仍未部署；不是所有八角点已准确。对照图为 outputs/action4_joint_region_seed_comparison_v003/COMPARISON.jpg。

8项CPU单元/回归检查通过，包含已知GT夹具、实际信息消费、缺少清晰可见性假设/错误绑定/非有限输入拒绝。唯一子智能体完成只读审查，指出入口缺帧和实例归属门、双向完整边界是假设、分量界不等于向量范数界、robust权重不具严格采样数不变性。模型注释和诊断口径已明确，未降低准入。

已修复诊断入口：显式清晰seed帧清单，并绑定FACE_SEED、PROMPT_REQUEST、native session/obj/generation、frame/PTS/RGB/K、scoped body身份和实际raw receipt，重放正提示连通component并要求逐像素相等。新增真实反例检查：两机分别错帧/跨机共4项均拒绝。首次绑定错误地引用track00 v002 prompt-response，被精确像素检查拒绝；实际源为v001同帧add_prompt响应。原失败及修正CONFIG均保留；旧v003实际mask输入未改，因此没有重算拟合。特别区分add_prompt响应与propagation frame0，同帧也不可混用。证据 outputs/action4_joint_region_seed_binding_v004/{FAILED.txt,SOURCE_CORRECTION.json,CONFIG_CORRECTED.json,RESULT.json}。

完成范围：当前种子候选、代码检查和来源修复；连续15帧拟合、六面/固定八角点效果准入、CoTracker及最终发布仍未完成。旧v002/v003诊断函数版本已按记录SHA留档，归档发生在运行后且范围仅两模块，不能冒充正式run完整不可变快照。后续连续任务必须使用完整冻结源码。


### 2026-10-27T06:57:15.256153+08:00：五F当前传播观测、五帧初始化与局部失效处理

完整15帧(4–18)、双机30实例按当前propagation mask/session/obj/generation/PTS/RGB/K重放，未复用seed mask或重新调用SAM。原始输入contact逐图核对，frame9 track00整机mask含另一台机身片段；该失败保留。初始0.3 CADdiag逐轴搜索界有18个触界，单次0.6诊断范围后全部消失；这是数值搜索对照，不是放宽发布门。

最近最多5个current-supported数值候选仅作为第二初始化；在同一个当前IPPE0 anchor、同一范围和当前观测目标内重新优化。没有历史loss/EMA/历史mask或逐角点自由修补。history_v003中完整fallback过渡出现21.76/22.49度接缝；不能只引用排除fallback后的5.06度最大值。partial_v004改为逐区域资格：污染body边界停用，当前button+marker继续输入，30/30候选均记录，0触界、0优化失败，相关过渡为5.06/5.80度。

完整14对邻接(每机)统计：track00 raw中位1.124、最大4.167度，candidate中位0.974、最大5.800度；track01 raw中位2.052、最大4.784度，candidate中位0.822、最大1.626度。这只是预测姿态变化，不能称真实误差或全面优于A0。全30结果仍published=false、production final保持A0。

固定ID导出180个面候选、240个角点；每帧12面/16角点，各角点恰属3面，从同一SE3投影一致。Face5的按钮区域和Face3估计marker不冒充完整面分割，其余面为推断。证据 outputs/action4_joint_region_sequence_partial_v004/{results/COMPLETE.json,ROOT_DATAFLOW_REVIEW.json,IDENTITY_REVIEW.json,FACE_CORNER_LEDGER.jsonl,PARTIAL_FRAME9_REVIEW.jpg}。

唯一子智能体独立确认当前观测和因果历史数据流，并指出先择优后top-facing可能误丢可用fresh解、世代变化缺reset。已在工作树补同一数值/正深度/朝向gate先于择优、世代更新清历史/旧世代拒绝；13项核心/历史测试及1项模拟状态机generation测试通过。对v004保存的28对双初始化候选重放新gate，选择改变0；未重跑优化、未修改旧冻结source。所有序列运行使用完整src/config冻结副本，CPU40编排、94/95求解，无GPU或模型重推理。

当前只通过短序列接口和候选检查，未通过六面跟踪精度/全视频发布。人工区域资格尚需自动化，困难段、原生前后面身份变化、材料跟踪、最终test/video及归档push继续保留未完成。


### 2026-10-27T07:25:14.210013+08:00：更正frame9判断，修复几何距离计算的裁剪依赖

更正上一节：frame9 track00原始mask共211645像素，其中主体211641像素，两个远处分量各2像素。预览按mask范围裁剪被这4个像素拉大，不能据此称大面积跨机污染；此前直接停用完整body约束过于保守。原始记录、失败对照保留，见 outputs/action4_joint_region_sequence_component_v005/PRIOR_REVIEW_CORRECTION.json。

新增当前多证据连通分量归属：当前button和marker同时支持同一分量才用于候选几何，不按最大面积选择、不修改raw SAM mask、不改变原生质量分母，未选区域保持unknown，缺失/分歧/复制cue不冒充正确负例。30实例均按当前观测归属；清晰全可见仍为本开发短片假设，未声称自动遮挡识别。

v005仍出现19.24度候选变化，随后检查几何目标。顶点倒序假设在3份实际输入上未发现目标差异，故没有按该假设改采样。真正失败来自局部distance-map超裁剪区域被赋max(shape)常数：同一frame4观测、同一R/t，画布补128像素且K/UV同步平移后，initial_cost 17907.787→6173.471。该错误在新增几何候选中，不影响已有A0训练、最佳ckpt、2850帧raw baseline或独立marker-only650评分。

已用当前观测边界像素的精确欧氏最近距离替换人工常数外部域，保留原裁剪/像素尺度与源数据。实际frame4/9/10画布平移等价差约9.09e-13/1.82e-12/9.09e-13。21项针对测试通过，另1项复制support反例通过。v006复用v005输入、参数和历史策略，只修距离域；全部30结果保留、0优化失败、0触界。track00相邻候选旋转中位0.838度/最大2.137度，track01中位0.602度/最大1.240度，均覆盖14对邻接，不将这些量称真实pose误差。

15帧只读开发对照视频已编码：outputs/videos/20261027-072041-six-face-dev4-18-distance-fix/comparison.mp4。1900x750，2997/100fps，time_base1/11988，duration_ts6000；源帧4–18，源首PTS1600，视频时间归零。左原A0，右未发布候选；encoder只读取预存VIDEO_PAYLOAD，0模型调用、0几何重算、不在绘制层选择姿态。GPU7 NVENC、CPU92–93，未使用GPU0–3；本机副本SHA核对相同。只代表0.5005秒开发短片，不是完整2850帧效果验收。

证据集中 outputs/action4_joint_region_sequence_distance_v006：ROOT_REVIEW、REAL_CROP_EQUIVALENCE、COMPLETE、IDENTITY_COMPONENT_REVIEW、FACE_CORNER_LEDGER、VIDEO_COMPLETE。12面/16角点每帧均由同一SE3投影，180面/240角点记录完整；physical face身份及真实精度未验收，默认输出保持不变。下一步对较难开发段验证区域资格/多视角约束；CoTracker、最终test/video、归档push仍未完成。


### 2026-10-27T08:16:56.855222+08:00：正面困难段808–822的实际分支验证（效果未准入）

复用已完成原生Face2-hypothesis传播，不重新调用SAM。当前best A0原始姿态作为初值，不使用背面marker或GT。冻结component13前镜头盖几何及法向−Y仅为显式对应假设；经mesh检查该部件为约29.93mm圆形薄盖，并非仅由bounding range误判的方形。当前输入15帧×双机=30条全保留，track00有15条实际面区域，track01因无已登记合法面prompt保留15 unavailable，不代表该机身不存在，原A0输出均保留。

v001整机与部件双向约束得到15个数值候选，但图像复核立体框明显偏移/扩大，拒绝发布。v002只使用当前B0预测完整框加0.1长边margin内的原始SAM轮廓，body为observed-to-model单向部分边界；不裁改mask、不创造框边界、不约束八点必须留框内。15输入的原mask/K逐值一致，后续无marker组；错误K和网格在solver前被拒绝。v002仍有明显投影范围/角度偏差，不准入。

track00相邻旋转：raw中位7.715/最大16.436度，v001中位1.743/最大10.678，v002中位1.727/最大4.917；这是完整14对相邻候选变化，不是姿态GT误差。更平滑但位置不可靠仍失败。来源与图像见 outputs/action4_front_region_sequence_v001 与 outputs/action4_front_region_partial_v002，ROOT_REVIEW明确不发布，默认权重/正式数据不变。

工程改动复用现有区域求解和历史链：显式前/顶面法向、可选marker输入、部分原始轮廓资格与残差分母、计算crop/K一致性检查。旧顶面/历史分支13项回归通过；2项部分边界测试验证未凭eligibility窗口制造新轮廓；1项正面法向测试通过。首次新增测试命令误写旧类名TrainingRoiTests，实际未执行该类测试，已纠正为JointRegionTests并完成2项检查。后续守卫的generation测试亦通过。

唯一子智能体正在补阶段六真实视频材料seed接口，只做CPU源码/数据准备，不启动CoTracker、GPU或新训练；其交付未复核前不记代码通过。F5整体效果仍未准入，不将材料接口准备写成跳过F5或阶段六效果完成。


### 2026-10-27T11:26:30.712223+08:00：用户明确后续独立训练，SAM阶段三实际链路推进

用户明确只停止baseline追加；后续独立阶段按主稿训练。保留baseline best，未批准旧追加段重启。已登记独立native-original SAM prediction ROI1.5协议，旧baseline保护门保持；独立训练gate不能借旧stage1/baseline回执。训练缺mask回退仍依靠显式预测记录；RGB变体同源ROI合同未闭合前不混入，遵循当前3.7原图分支。

真实Stage2Units完成8实例×2epoch数据消费（16条）：原图冻结B0预测1.5 ROI、SAM mask增强、真实GT监督，未调用模型；构造耗时613.076秒，总625.334秒。随后GPU4使用现有construct/optimizer/loss_sums/global-count/step接口实际2次更新，冻结base所有参数与buffers保持相同，严格bundle重载逐bit相同，峰值allocated953.5MiB，总6.43秒。此为实际输入工程更新，不是正式80轮或效果改善。

异步eval已接同一冻结B0和SAM原图合同，两份B0缓存虽然路径不同但SHA均e02169350abf56db3a3a227ae7b8fbfd8199e758331f18170756e2bd2a96f5e6，650图内容一致。新增协议/框-mask身份/async profile检查通过。全650评估v001错误使用Phrase2Dataset直接构造，被回执守卫拒绝、0图计分；v002抽出并复用正式worker原有ComponentData回执构造入口，未弱化守卫。当前step0/step2分别GPU5/6运行，尚无完整报告，不预报分数。

缓存验证增加可选1–8worker有界并行，元数据完整域先验证，每个payload原有内容/来源/变更校验全部保留，最终再扫stat。代理32项检查通过，root复核源码并独立跑14项专用检查通过。真实eval650/1300冷进程8worker验证74.45秒；没有同域串行对照，不与train613秒混算加速比。训练/eval调用方已显式接validation_workers，正式资源仍需吞吐准入。

另已主控复核真实视频材料接口，使用已认证VideoFrameBinding/VideoROIContext/native bytes而非伪造still缓存；补精确已哈希字节解码和显式timing参数一致性，12测试通过。保持质量门，扩大有限检查预算后6帧×双机全候选完成，共14386前置eligible、14371几何尝试、10个provisional点，各实例0–4；没有任何实例达到阶段七≥12点条件，不启动材料PnP/CoTracker效果验收。首次失败原因计数不能解释成独立根因；初始化自一致门与跟踪噪声门的耦合仍需按dev校准核查，不直接降门凑数。


### 2026-10-27T12:39:05.745331+08:00：完整650初始化复核与四卡28步集成启动

SAM初始化完整650得到899/1300，真实8单位2次更新后720/1300，后者不是效果提升，不能作正式父。历史baseline900/1300与初始化最大差0.159149 native px；定位为baseline历史GPU RGB采样与训练/SAM CPU采样的浮点差异。最差样本同GPU模型下，CPU crop预测精确复现SAM初始化，GPU crop精确复现历史baseline；矩阵相同。另完整650原baseline同CPU采样得到899/1300，全部1300实例八点与SAM初始化逐值完全相同。历史GPU路径与900结果不覆盖，未更改评分阈值。

新增显式roi_sampling_device用于公平同输入参考，新SAM评估记录CPU路径。独立阶段三parent seal/registry指向真实1450；冻结器复制parent后须由RUN_SPEC中的相同权重SHA证明，不允许旧15770/14000混入。新source snapshot完整保留runtime/surface_deps原生binary，避免只有cv2包装文件的历史递归导入风险。

已用现有freeze/dispatch/train/eval启动 p2_a0_sam_prediction15_integration28_v001：随机632预测ROI×2epochs=28真实更新，四卡GPU4–7训练、GPU5/6/7各1eval，CPU53，step25受控同run恢复，预期eval0/25/28。controller实际PID 1333587；DISPATCH已记录trainer和3eval进程。此为独立SAM集成，不是正式80轮，也不是baseline追加。源冻结后不修改运行代码。

已创建本线程每30分钟检查 action4-sam（首次工具请求缺destination被拒绝，修正destination=thread后创建成功，无重复任务）。完整阶段准入还需实际28更新/规定eval/恢复/吞吐及SAM封存证明；不能仅凭进程运行宣称通过。
