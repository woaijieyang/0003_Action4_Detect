# Action4 当前执行与第二阶段交接索引

本页是导航和状态说明，不另建一份与原plan竞争的计划。唯一plan/conclusion仍在docs/stages。当前r069，本轮修复和整理完成，真实任务尚未通过；第二阶段等待用户正式推进。

| 事项 | 权威位置与含义 |
|---|---|
| 活动执行 | configs/execution/action4_active.json，禁止按mtime启动旧配置 |
| 当前计划 | docs/stages/2026-10-19-1-plan-Action4角点复现增补.md |
| 实际结论 | 同目录2026-10-19-1-conclusion-Action4角点复现增补.md |
| 模型选择 | models/CURRENT_RESEARCH.json；权重独立，B0框→A0 ROI→R/t→canonical8是串联推理 |
| 合成基线 | configs/eval/action4_photometric_best_v001.json |
| 真实视频候选 | configs/eval/action4_a0_online_rgb_best_v001.json |
| 原2000数据 | data/action4_dataset_commit.json及data/train、eval、test；1600/200/200，test全双机 |
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

本轮没有启动第二阶段、没有创建新训练或自动重复原80轮。资源总预算CPU72，GPU1专用eval最多8进程，GPU2–7训练/工作，GPU0禁止；各次启动以实际空闲为准。
