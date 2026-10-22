# 模型测评结果

更新时间：2026-10-22T18:52:58.876627+08:00（Asia/Singapore）。

## Action4 固定模板数据：开发集

正式系统测评成绩：尚无满足最终真实视频验收的结果。下表仅登记已完成的 **B0 完整机身框模块开发测评**，不代表 B0+A0+几何模块的八角点效果，不代表真实视频准确率。

协议：200 张 eval 图、378 个目标、184 张双机图、6 张无目标图。操作阈值0.3；完整机身框 IoU≥0.5一对一匹配计算P/R/F1。AP采用101点插值，IoU .50:.05:.95；输入候选阈值.001，每图最多2个框。开发集已用于选优，不能当作独立盲测。

### 辅助模块测评：同一冻结权重的解码顺序对照

| Record ID | Scope / Samples | Group / Model | Version | Status | AP50 ↑ | AP50:95 ↑ | Precision ↑ | Recall ↑ | F1 ↑ | 双机检出 ↑ | Benchmark SHA | Model code path | Dataset/manifest path |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
| [b0-order-001](#证据与适用范围) | B0 eval全200图 / 378目标 | independent giou_quality | step1980; product_before_peak | complete（模块范围） | 0.839060 | 0.396156 | 0.891304 | 0.867725 | 0.879357 | 137/184 | `791ac3469dc4a05b3a07a6200dce8db6c11fc6af68205efdf41677fcd9cfe34e` | [audit](../src/eval/action4_quality_order_audit.py)、[model](../src/model/action4_component_experiments.py) | [eval manifest](../data/eval/annotations.jsonl) |
| [b0-order-002](#证据与适用范围) | B0 eval全200图 / 378目标 | independent giou_quality | step1980; centre_peak_then_quality | complete（模块范围） | 0.838724 | 0.398006 | 0.891304 | 0.867725 | 0.879357 | 137/184 | `791ac3469dc4a05b3a07a6200dce8db6c11fc6af68205efdf41677fcd9cfe34e` | [audit](../src/eval/action4_quality_order_audit.py)、[model](../src/model/action4_component_experiments.py) | [eval manifest](../data/eval/annotations.jsonl) |

相对 b0-order-001，b0-order-002 的 AP50:95 增量为 **0.1850 个百分点**；TP328、FP40、FN50、F1和双机检出数相同。两者重复检测0、无目标误检0/6；仅6张无目标图，不能外推高可靠性。19图框发生变化，未据此修改正式解码。

### 证据与适用范围

- artifact_path：[完整指标](../outputs/action4_independent_trials_v001/quality_order_audit_v001/REPORT.json)、[逐图预测](../outputs/action4_independent_trials_v001/quality_order_audit_v001/PREDICTIONS.json)。
- 冻结checkpoint：[B0.pt](../outputs/action4_independent_trials_v001/quality_order_audit_v001/B0.pt)；step=1980；SHA256=`334bbc09a6b31c9ae35f2e6c23b0561b6610e00c0607708debfc7b0202563824`。
- 配置：[b0_giou_quality](../configs/model/component_trials_v001/b0_giou_quality.json)；checkpoint config_id=`7c1900f56344b738d5ba8f8a2f9f07a142714ecbe2de80e5beb4f05cbab726d6`。
- 数据与当前源码身份、200唯一ID覆盖和GT计数核验：[RESULT_REGISTRATION.json](../outputs/action4_independent_trials_v001/quality_order_audit_v001/RESULT_REGISTRATION.json)。源码hash记录的是登记时版本；不伪装成运行时新增的seal。
- 两行使用同一冻结权重、同一数据与评分函数，仅解码顺序不同。不得与其他checkpoint直接混为同预算增益；这是单次开发对照，不是多种子均值。
- 当前真实帧仍存在衣服误检与相机漏检；失败400候选未通过视觉审核，未进入本表。训练loss、构造测试、运行状态不登记为成绩。
- 后续训练仍在进行；本表的complete只表示上述200图模块测评完成，不表示80轮训练或项目完成。过程见[阶段结论](../docs/stages/2026-10-19-1-conclusion-Action4角点复现增补.md)。


### B0残差框与质量适配器：完整开发评估（非最终验收）

| Record ID | Scope / Samples | Model / Version | Status | AP50 | AP50:95 | F1 | 双机同时检出 | Model code | Dataset |
|---|---|---|---|---:|---:|---:|---|---|---|
| action4_b0_refinement_v001_final6300 | B0-only，200开发图/378目标 | residual adapter，B0父5220，best6300，80epoch | complete | 0.93174376 | 0.47032656 | 0.94583884 | 164/184 | [模型](../src/model/action4_box_refinement.py) / [评估](../src/eval/action4_box_refinement_eval.py) | [开发标注](../data/eval/annotations.jsonl) |

Benchmark SHA: 791ac3469dc4a05b3a07a6200dce8db6c11fc6af68205efdf41677fcd9cfe34e。重复2，空图误检0/6。
证据：outputs/action4_box_refinement_v001/eval/step_00006300.done.json；权重outputs/action4_box_refinement_v001/final_best_frozen.pt；配置及源码身份configs/model/action4_box_refinement_v001.json。400个20step评估加step0共401份完整完成。claim_scope仅开发集完整框，非真实验收；双机89.13%未达90%。真实6帧回退保留于outputs/action4_box_refinement_factor_audit_final/REPORT.json，demo非GT；不把中间step4220的90.22%拼入此行。



### A0完整几何链第一版：GT ROI辅助诊断

| Record ID | Scope / Samples | Model / Version | Status | 全8点@0.05 | 高遮挡全8点@0.05 | 数值有效pose | Code | Data |
|---|---|---|---|---|---|---|---|---|
| action4_gdr_pose_v001_final12550 | 全378 GT ROI，来自200开发图 | GDR-derived ConvPnP v001，best12550，80epoch | complete | 172/378 | 26/164 | 378/378 | [模型](../src/model/action4_gdr_pose.py) / [评估](../src/eval/action4_gdr_pose_eval.py) | [开发标注](../data/eval/annotations.jsonl) |

Benchmark SHA:791ac3469dc4a05b3a07a6200dce8db6c11fc6af68205efdf41677fcd9cfe34e。作用域仅GT ROI辅助诊断，非真实识别或端到端成绩。固定ID8角点误差各自除以GT框对角线，全部≤.05为成功。父surface+PnP为247/378、58/164，当前退步，不推广。80轮15120step/606eval全部完成，失败0；已确认microbatch损失权重问题和方法差异另见阶段结论，不能作为最终正确算法成绩。

配置/源码身份：configs/model/action4_gdr_pose_v001.json；artifact_path：outputs/action4_gdr_pose_candidate_v001/eval/step_00012550.done.json。选定权重已固定于outputs/action4_gdr_pose_candidate_v002/selected_parent.pt，PARENT_SELECTION.json绑定其step/config/hash，后续修正版不得与本行混淆。


## 2026-10-23 final frozen developmental candidate (not accepted on real video)

B0 online80epochs/4000steps/best2200; A0 GDRv00280epochs/15120steps/best13350 with predicted-surface PnP. Dev bboxAP50=.979111,AP50:95=.548464,F1=.982781,dual177/184; fixedID all8=222/378,heavy48/164. Historicaltest bboxAP50=.973173,AP50:95=.561451,F1=.986233,dual194/200; all8=245/400,heavy38/142. Test is not new blind acceptance. Raw grouped reports: outputs/action4_final_test_v001/REPORT.json and outputs/action4_final_selection_v001/joint_pnp/REPORT.json.

Real demo-reference6frames: B0 both2/6 vsoldbaseline5/6, not certifiedGT. Full2850frame comparison completed but visually FAILED; 828/5688 detections invalid pose across771frames. No production promotion. See stage conclusion and outputs/action4_final_video_v001/ROOT_VISUAL_REVIEW.json.


## 2026-10-23 frozen B0 center-scoring candidate and protocol correction

No new training; B0step2200/A0step13350 unchanged. Refined boxes +center score: dev dual171/184,AP50:95=.529753099,F1=.973333333,all8=221/378,heavy45/164. Full2850video complete, invalid poses332 across319frames. Same2773 eligible demo-reference envelopes: oldB0step5220 both1367 (49.30%), learnedscore732 (26.40%), centerscore1862 (67.15%).77referenceframes unscorable; NOT certifiedGT/realrecall/mAP; no production acceptance.

AP caller now retains low-score candidates separately from operating0.3 threshold. Original sealed learned-score historicaltest recheck unchanged AP50:95=.5614511313 and sameTP/FP/FN. New center candidate NOT selected ontest. Box-translation anchoring rejected (all8 221→93); GT pose substitutions are oraclediagnostics only. Full receipts instageconclusion.


## 2026-10-23 photometric80 and frozen-surface-pose80 closure

B0best3220: dev AP50:95=.6162104443,F1=.9867374005,dual178/184; historicaltest AP50:95=.6493924639,F1=.9937264743,dual196/200. With retained A0PnP: devall8=221/378,heavy45/164; historicaltest253/400,heavy36/142.
Full2850video, demo-reference IoU0.5 both2268/2773=81.79% (77unscorable), not certifiedGT/realrecall. Visual acceptance still FAILED.
A0frozen-surface80 best8625: GTROI196/378,heavy36/164; actual predictedROI42/378,heavy4/164. REJECTED for deployment.
