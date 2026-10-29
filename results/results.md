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


## 2026-10-23 online RGB A0 completed80, fixed detector full-image evaluation

All results use B0 photometric3220 center and A0 onlineRGB14000 network. Development200 images/378GT: all8=171/378,heavy20/164. Historicaltest200/400GT: all8=201/400,heavy21/142. Both are weaker than retained PnP221/378 and253/400; no synthetic promotion. Protocol is fixedID all8 normalized error≤.05 and unmatched objects fail. Code: src/eval/action4_staged_eval.py, src/model/action4_gdr_pose.py; data: data/eval/annotations.jsonl and data/test/annotations.jsonl, benchmark identities inherited unchanged from outputs/required/DATA_VALIDATED_ALL_SPLITS.json (eval SHA791ac3469dc4a05b3a07a6200dce8db6c11fc6af68205efdf41677fcd9cfe34e). Raw reports: outputs/action4_a0_online_rgb_validation_v001/{development,historical_test}/REPORT.json. Frozen profiles: configs/eval/action4_a0_online_rgb_best_v001.json.

Auxiliary full-video reference consistency:2850 frames,2773eligible77unscorable, both boxes2268/2773, both projected corner envelopes1356/2773 vs retainedPnP273/2773. Envelope metric is NOT cornerID/pose accuracy or certifiedGT; real task still fails. Cached development selection173/378 differs from actualRGB171/378 at two threshold-near targets; full disclosure and paired numeric diagnosis in stage conclusion, thresholds unchanged.


## 2026-10-23 r069 fixed-weight decoder audit (development + real diagnostic)

SameB0photometric3220/A0online14000, full200development378GT, no test selection: network171/378 heavy20/164; network-initialized iterativePnP228/378 heavy47/164; surfacePnP218/378 heavy40/164. Protocol fixedID8each≤.05GTdiag, missed/invalidfail. Prespecified60real demo-reference frames: bothboxes48/60 unchanged; bothposeenvelopes network26/60,iterative9/60,surfacePnP7/60. This proxy does not establish cornerID accuracy or whole-video performance; iterative not promoted. Source: src/eval/action4_pose_decoder_audit.py; data: data/eval/annotations.jsonl, exact identity inherited from unchanged accepted2000; artifacts: outputs/action4_pose_decoder_audit_v001/{development,real60}/*_REPORT.json.

Exact image-grouped GPU cache replay reproduces all372 matched target errors and171/378 decisions exactly (maxdifference0). This is an engineering contract repair, not a new accuracy gain or training result.

## phrase2 开发集：SAM 原生分割支持辅助测评（2026-10-25，Asia/Singapore）

任务范围为冻结B0提示下的可见机身mask支持，不是Action4类别精确率、A0八角点或真实视频验收。完整650图/1300原始GT保留，包括上游漏提示；不以条件域丢弃失败。严格support要求可见机身coverage≥0.5、target_fraction≥0.8、camera_purity≥0.95且另一机身leakage≤0.05，按预登记一对一协议计分。两次SAM权重/官方实现相同，B0提示来源不同。

| Record ID | Scope / Samples | Group / Model / Version | Status | 支持覆盖↑ | 严格support↑ | 双机同时严格support↑ | Model code path（服务器） | Dataset/manifest path |
|---|---|---|---|---:|---:|---:|---|---|
| [p2-sam-parent-b0-full650-20261025](#p2-sam-parent-b0-full650-20261025) | full650图 / 1300GT | 旧父B0 step0 + SAM3.1；官方commit `7f473cfd983d` | complete；仅辅助测评 | 1296/1300 | 615/1300 | 152/650 | `/data/zy/vln/a-test/outputs/action4_sam_full_dev_eval_v001/r096_v005/source/src/eval/action4_sam_image_eval.py` | `/data/zy/vln/a-test/data/phrase2/eval/annotations.jsonl` |
| [p2-sam-completed-b0-step9500-full650-20261025](#p2-sam-completed-b0-step9500-full650-20261025) | full650图 / 1300GT | 完成80轮B0 step9500 + SAM3.1；官方commit `7f473cfd983d` | complete；仅辅助测评 | 1300/1300 | 709/1300 | 205/650 | `/data/zy/vln/a-test/outputs/action4_sam_selectedB0_eval_v001/source/src/eval/action4_sam_image_eval.py` | `/data/zy/vln/a-test/data/phrase2/eval/annotations.jsonl` |

Benchmark cohort SHA-256：`ec9d897dec5fd6947d1e5b9b3d0217f6c2f1ace7c2fa24320d083e89e642218a`。发布eval annotation SHA-256：`5cc8cb48f504039ada59847fda3768fc53b19572fda7e5d7508e7770f56d30da`。SAM权重：`0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6`；新B0 bundle：`ce6ba1a58e9f16608d4a44923c5e36914fc764f05b4ba8764f2e9c6c0e5f2ca3`。原生兼容层`action4_sam_native_compat.py`身份包含在对应source snapshot中。

同一1265个可配对目标：平均mask IoU从0.7778189升至0.8049935，绝对差+0.0271746；新增35个上游合法目标不混入该配对差。新完整域中591个mask仍被诊断为机身夹带背景/配件；没有可用非DJI类别GT，类别specificity保持未测。该正结果不能替代SAM辅助A0训练或最终精度准入。

原始artifact（服务器）：旧`outputs/action4_sam_full_dev_eval_v001/scorer_v001/full_v001/{RESULT.json,PER_IMAGE.jsonl}`；新`outputs/action4_sam_selectedB0_metrics_preparation_v001/full_v001/{RESULT.json,PER_IMAGE.jsonl,ROOT_ANALYSIS.json}`。实际scorer源码SHA：`f94ec285b337c92618a4afda5fc243e44d5544245d45cca62839c9782e272d35`。实现过程、失败修复与后续边界见[本阶段结论](../docs/stages/2026-10-24-1-conclusion-Action4第二阶段识别与八角点增强.md)。

<a id="p2-sam-parent-b0-full650-20261025"></a>

**p2-sam-parent-b0-full650-20261025 证据：** 服务器artifact为`/data/zy/vln/a-test/outputs/action4_sam_full_dev_eval_v001/scorer_v001/full_v001/RESULT.json`；native配置与父模型、权重、source身份见`/data/zy/vln/a-test/outputs/action4_sam_full_dev_eval_v001/r096_v005/run/RUN_LOCK.json`。该行是旧B0父模型提示，不能当作80轮新B0。

<a id="p2-sam-completed-b0-step9500-full650-20261025"></a>

**p2-sam-completed-b0-step9500-full650-20261025 证据：** 服务器artifact为`/data/zy/vln/a-test/outputs/action4_sam_selectedB0_metrics_preparation_v001/full_v001/RESULT.json`及`ROOT_ANALYSIS.json`；native配置与完整模型source合同见`/data/zy/vln/a-test/outputs/action4_sam_selectedB0_eval_v001/run/RUN_LOCK.json`，B0选定step9500封存回执见`models/candidates/action4_improvement/stage1_B0_completed_20261024/SEALED_ROLE.json`。


## phrase2开发集：固定A0/B0预测与面内对应几何候选（2026-10-27T05:04:55.956360+08:00）

完整650图、1300非ignore GT，评分action4.pose_projection_all8_002.v1；一对一匹配、固定canonical8 ID、失败分母不变。图像模式不使用track。新行是缓存神经预测后的完整开发候选测评，不是新增训练、最终test或真实视频准入。

| Record ID | Scope / Samples | Group / Model / Version | Status | All8@2% ↑ | All8@5% ↑ | FP / FN@2% ↓ | 有效匹配角点P95 px ↓ | Δ All8@2% pp | Benchmark SHA | Model code path（服务器） | Dataset/manifest path |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|---|
| p2-roi15-best1450-full650-20261027 | full650 / 1300GT | A0 step1450 + B0 step9500 / ROI1.5 | complete；开发基线重放 | 900/1300 | 94.615% | 400 / 400 | 16.196564 | — | 5cc8cb48f504039ada59847fda3768fc53b19572fda7e5d7508e7770f56d30da | /data/zy/vln/a-test/runtime/action4_runs/p2_a0_roi15_best_extension10_b64_e10_v001/source/src/eval/action4_staged_pipeline.py | /data/zy/vln/a-test/data/phrase2/eval/annotations.jsonl |
| p2-marker-geometry-shadow-full650-20261027 | full650 / 1300GT | 同权重＋estimated display-marker geometry / frozen policy v001 | complete；未部署 | 955/1300 | 94.692% | 345 / 345 | 15.406371 | +4.231 | 5cc8cb48f504039ada59847fda3768fc53b19572fda7e5d7508e7770f56d30da | /data/zy/vln/a-test/outputs/action4_marker_geometry_full650_v001/source/src/eval/action4_marker_geometry_development_eval.py | /data/zy/vln/a-test/data/phrase2/eval/annotations.jsonl |


证据artifact：服务器outputs/action4_marker_geometry_full650_v001/results/{BASELINE_REPORT.json,CANDIDATE_REPORT.json,PREDICTIONS.json,OBSERVATION_AUDIT.json,PAIRED_DELTA.json,COMPLETE.json}。配置为同run的CONFIG.json，源码为source/SOURCE_MANIFEST.json，identity=b51864bac7be7f6ff7b72eef10e6e117da57fa5d2a1750f0cb4d96075799dac5。A0 SHA=528e4d3c0b66cceb624f2b714885bdd003462c293c6e0aa95cc9a2013e0068b2，B0 SHA=ce6ba1a58e9f16608d4a44923c5e36914fc764f05b4ba8764f2e9c6c0e5f2ca3；默认模型未变。

565个预测采用候选几何；逐GT修复59、退化4。高遮挡仍44/192，小目标高遮挡仍0/40；不能把整体净增55掩盖困难组未改善。全域无非有限发布、1300输出保留；数据与K从白名单输入，GT仅在worker候选保存后计分，未用于选择对应或pose。检测器与template binding固定为OpenCV4.13.0/36h11/default NONE，不与SUBPIX混用。

该binding是既有appearance纹理映射估计，不是实物metric标定；真实帧独立区域审查已发现marker拟合低误差但整机偏移的反例。因此本记录不构成真实视频替换A0的证据，也不代表冻结神经网络自身性能提高。详细过程和后续门见2026-10-26-1-conclusion-A0六档ROI扩张与B0预测框条件化微调.md。
