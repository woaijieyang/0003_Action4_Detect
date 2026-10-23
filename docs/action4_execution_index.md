# Action4 当前执行与第二阶段交接索引

本页是导航和状态说明，不另建一份与原plan竞争的计划。唯一plan/conclusion仍在docs/stages。当前r071：本线程工程链路验证、归类与归档完成；真实任务尚未通过。第二阶段数据合成由用户安排的另一任务推进，本线程不改动其文件和依赖。

| 事项 | 权威位置与含义 |
|---|---|
| 活动执行 | configs/execution/action4_active.json，禁止按mtime启动旧配置 |
| 当前计划 | docs/stages/2026-10-19-1-plan-Action4角点复现增补.md |
| 实际结论 | 同目录2026-10-19-1-conclusion-Action4角点复现增补.md |
| 模型选择 | models/CURRENT_RESEARCH.json；权重独立，B0框→A0 ROI→R/t→canonical8是串联推理 |
| 合成基线 | configs/eval/action4_photometric_best_v001.json |
| 真实视频候选 | configs/eval/action4_a0_online_rgb_best_v001.json |
| 原2000数据 | data/action4_dataset_commit.json及data/phrase1/train、eval、test；1600/200/200，test全双机 |
| 原片抽帧 | data/action4_real/cache_v001/frames_manifest.json，2850帧身份与时间戳 |
| 本轮准确评估接口 | src/eval/action4_image_grouped_pose_eval.py |
| 实际整链路一致性证据 | outputs/action4_pose_decoder_audit_v001/EXACT_EVAL_ACCEPTANCE.json |
| 几何增强负结果 | 同目录development与real60的三种方法REPORT，未采用iterative |
| 归档根 | archive/legacy；所有旧控制器/撤回方案只能作历史 |
| 撤回方案 | archive/legacy/rejected_box_constraint_v001；禁止硬裁剪/挤压角点 |
| 文档原快照 | archive/legacy/document_snapshots/r069，完整保存整理前原文 |

## 第二阶段开始前需要落实的事项

1. 保留DJI实体尺寸、canonical8编号、已验证腕部安装关系；新版本场景修复桌面尺度、手臂出画/连接和实际衣物遮挡。现有counterfactual只验证了部分改善，未获新数据准入。
2. 三个主要场景按worker驻留；距离、姿态、双机相对位置、光照及遮挡参数按合理范围抽样，保持不穿插、可投影、双机占比≥90%、test全双机。不把已失败400候选或旧场景工程PASS当新版本视觉通过。
3. 新数据先独立版本化，核对RGB、full-body bbox、八点、visible/amodal mask和姿态的一致性，再明确train/eval/test拆分。原2000和原视频保持可回查。
4. B0从独立best3220继续；A0从明确选定的独立best继续。训练前声明采用哪个A0及选择理由，不把新旧模块随意拼成“同一个checkpoint”。
5. A0每25step、B0每20step异步评估。若冻结backbone/surface，新数据按部署GPU前向及原图batch构建新缓存；若这些模块参与更新，不能沿用静态maps缓存，必须跟随checkpoint重新执行RGB评估。
6. 评估manifest、B0框来源、backbone/surface身份、角点ID、坐标变换和缺失目标分母绑定到新配置。实际RGB回放必须与缓存逐点一致，再启用缓存选优。
7. 冻结best后做完整框与固定ID角点分别报告、同帧demo参考及完整视频。模型数值有效、红框落在绿框附近、合成指标提高都不能替代真实准确性。

本线程完成既有数据上的两步工程训练对照，没有启动第二阶段正式全量训练或重复原80轮。资源总预算CPU72，GPU1专用eval最多8进程，GPU2–7训练/工作，GPU0禁止；各次启动以实际空闲为准。


## 第二阶段已有定因证据补充

archive/legacy/organization_v002/outputs/action4_surface_domain_audit_v001记录了60帧实际表面输入诊断：106目标内部拟合改善却有83目标外部参考IoU下降，暗色/侧背视图mask经常零碎并混入衣物，不能继续只加强PnP拟合或只训练pose头。原train3024目标按canonical观察主导方向，-Y44、-Z21，开发各5例且network0/5；参数随机不等于投影视角覆盖充分。第二阶段需先映射轴向与真实可行佩戴视角，再在合法运动内补足实际所需的稀缺视角；不要机械六面均衡。表面分支若参与更新，必须逐checkpoint评估实际RGB，不沿用冻结maps。


## 已准备的裁剪与监督衔接模块

- data.action4_online_surface_targets：旧native表面对应点→本次ROI连续采样坐标；重新采样原visible/amodal mask，同步K和八点目标。
- model.action4_sparse_surface_losses：在真实变换位置监督XYZ/region，保留现有表面损失含义；不以GT输入模型。
- 4个实际train目标/2855点已通过无jitter等价、jitter重投影及CUDA梯度检查；隐藏角点损失可以反传到XYZ/region/mask。随后已接入两步真实工程训练和完整开发集评估，不代表真实精度改善。
- 新数据不能默认落入旧surface_v002路径；需在第二阶段绑定对应的准入sidecar版本和manifest。已接入独立角色调度器；新数据仍须独立准入。


## 已闭合的独立训练与交付链路（r071）

1. 原始RGB与同步几何标签 → action4_stage2_data → 在线ROI/摄影增强；表面点按同一仿射变换，XYZ/区域ID不变，原mask重新采样。
2. A0 JointSurfaceRGB：RGB → backbone → XYZ/regions/masks → pose → 固定canonical8投影。GT只进入损失；surface与pose均实际更新。B0独立从best3220继续，仍使用完整框、GIoU及质量/细化损失，评估只按框指标。
3. action4_stage2_pair/dispatch核算两角色总CPU与GPU1评估数；action4_stage2_train处理任意数据数量、DDP不整除尾批、独立best初始化及精确故障恢复。
4. 每25个A0 /20个B0 optimizer step发布不可变快照；action4_stage2_eval使用当前RGB前向、共享claim队列和串行选优锁。终态等待所有eval后，只保留best。
5. action4_stage2_finalize校验完整训练/配置/角色身份，分别封装A0、B0；staged_video_parallel生成同帧demo参考视频。工程模式不能冒充80轮训练封存。

实际证据：outputs/action4_stage2_integration_v001。每角色3个训练单位、2 optimizer steps，step1中断续训与不中断对照逐张量相等；完整200开发图初始/末次各评一次。A0 171→167/378，B0 AP50:95 .616210→.570893，均选择初始best；没有新精度提升。16帧32实例视频与原父模型框/八点/时间戳完全一致，视觉仍有支架混入与框偏移。
本轮短试验未跨过25/20step周期，不能声称新调度器已跑完80轮。周期请求规则已验证，后续正式运行必须用新的准入配置与独占输出，不能重启历史integration配置。
正式入口保留双重约束：数据manifest与surface来源版本绑定；full_training至少80轮。工程gate无法授权全量训练，test不用于选优。参考configs/model/action4_stage2_integration_*仅为历史已执行工程配置，源码身份仍对应source_snapshot，不静默刷新为新代码。

归类：data/README.md、outputs/README.md、archive/README.md；精确迁移映射outputs/action4_organization_v002/MOVED.jsonl。保留模型入口为models/CURRENT_RESEARCH.json，集成重复best已退役。视频如下（时间仅为原mtime排序）：
- outputs/videos/20261023-105001-stage2-integration16.mp4（原 outputs/action4_stage2_integration_v001/short_video/comparison.mp4）
- outputs/videos/20261023-045455-B0-center-scoring.mp4（原 outputs/action4_center_scoring_video_v001/comparison.mp4）
- outputs/videos/20261023-090112-A0-onlineRGB-best.mp4（原 outputs/action4_a0_online_rgb_validation_v001/video/comparison.mp4）
- outputs/videos/20261023-073233-B0-photometric-best.mp4（原 outputs/action4_photometric_validation_v001/video/comparison.mp4）
- outputs/videos/20261022-123533-demo-old-baseline.mp4（原 outputs/action4_demo_alignment_v001/demo_vs_baseline.mp4）
- outputs/videos/20261022-230101-component-full.mp4（原 outputs/action4_component_full_video_v001/comparison.mp4）
- outputs/videos/20261023-012554-parallel-smoke.mp4（原 outputs/action4_parallel_video_smoke_v001/comparison.mp4）
- outputs/videos/20261023-041716-historical-final.mp4（原 outputs/action4_final_video_v001/comparison.mp4）

第一阶段数据现已物理归类到data/phrase1/{train,eval,test}，旧三个根路径仅作兼容链接。原manifest、所有文件stat身份及2855表面点重投影复核通过；新阶段沿用原接口时可解析兼容路径，不能误把phrase1计为新增样本。


## 候选数据诊断（未启动新训练）

outputs/action4_next_data_baseline_v001：只读3个已完成train预览，当前best双机框IoU≥.5为3/3，但固定ID八点0/6；GT框裁剪及任意角点重排诊断均未达到八点门槛。不要把3图成绩当总体精度或用GT ROI成绩代替真实推理。下一步是等待独立数据任务发布新清单和geometry准入，按既有stage2训练器显式绑定surface_source与manifest，再进行联合surface/pose训练。
另一任务接口中的geometry目录与本线程默认phrase1/surface_v002不同，必须显式指定label_roots；同名/存在不表示身份绑定成功。正式launch gate还要求三split manifest SHA一致，不能只传一个admission布尔值。
