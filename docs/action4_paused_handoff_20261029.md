# Action4 暂停交接：模型、推理入口、视频与剩余问题

更新时间2026-10-29，Asia/Singapore。当前保持暂停。本页为原八阶段计划的续接说明，不另立方案；本次未启动训练、eval、模型推理、跟踪或编码。

服务器唯一项目根为 /data/zy/vln/a-test。下述相对路径均相对于该根；实际访问必须先cd并核验pwd -P。Git只包含代码、配置和文档，模型/数据/回执须在服务器使用。

## 1. 当前最佳研究组合与身份

当前续接组合是A0 step7300＋B0 step9500、ROI1.5。这里的“最佳”指当前阶段四已封存并进入阶段五的研究选择，不表示已通过真实视频部署验收。

| 项目 | 位置 |
|---|---|
| A0自包含权重 | models/candidates/action4_improvement/stage4_user_stopped_all8_20261028/A0.pt |
| A0封存回执 | 同目录SEALED_RESEARCH.json |
| B0自包含权重 | models/candidates/action4_improvement/stage1_B0_completed_20261024/B0.pt |
| B0封存回执 | 同目录SEALED_ROLE.json |
| 当前组合及解码参数 | configs/model/action4_stage5_selection_v001.json |
| 视频输入合同 | outputs/action4_stage5_sealed7300_inputs_v001/INPUT_LOCK.json |
| 研究候选与指标 | 同目录SELECTION.json |
| 成对身份 | 同目录MODEL_PAIR.json |
| 归档别名及全部权重SHA | configs/model/action4_pause_asset_registry_20261029.json |

A0完整服务器路径：/data/zy/vln/a-test/models/candidates/action4_improvement/stage4_user_stopped_all8_20261028/A0.pt
SHA256：b8ce72185e7ce46be29a1b62cf8d7086bb6167669959a76f870365b6f09e70b9

B0完整服务器路径：/data/zy/vln/a-test/models/candidates/action4_improvement/stage1_B0_completed_20261024/B0.pt
SHA256：ce6ba1a58e9f16608d4a44923c5e36914fc764f05b4ba8764f2e9c6c0e5f2ca3

当前selection实际加载B0的同SHA副本：
runtime/action4_runs/p2_a0_corner_prediction15_formal80_v003/source/inputs/B0_counterpart.pt。
现有loader同时绑定路径与身份；不能只因SHA相同便随意更换路径。

A0训练配置：runtime/action4_runs/p2_a0_corner_prediction15_formal80_v003/source/resolved_config.json；
对应RUN_SPEC.json在同名运行根。
B0训练配置：runtime/action4_runs/p2_b0_formal80_v001/source/resolved_config.json；
对应RUN_SPEC.json在同名运行根。以SEALED_ROLE.run_spec核验，不按mtime选择。

A0是surface/pose＋SAM辅助＋角点分支的完整包，保留RGB骨干和父模块；不能用旧纯pose loader吞掉新增模块。
阶段四实际由用户在13454更新/epoch62停止，539个eval完成；step7300为已评估研究选择，不称完成原80轮。
650图/1300目标的固定pose-associated协议：
All8@2%=1116/1300（85.846%），All8@5%=95.846%，FP/FN各184，合法匹配角点P95=12.3143原图像素，严格逐图成功率73.846%。
困难组All8@2%：小目标76/127，高遮挡85/192，小目标且高遮挡3/40，大小不等72/122，近邻16/20。这些是合成开发成绩，不是真实视频角点GT精度。
quality_ready=false；研究试用允许保留该候选，不伪写部署通过。

models/default/A0.pt仍是旧14000，B0.pt仍是旧3220，未覆盖。不能在当前组合加载失败时静默换回默认。
ROI1.15尚无最终best；完整恢复状态为：
outputs/action4_improvement/stage4_corner/p2_a0_corner_roi115_finetune75_v002/last.pt，step400，
SHA256 f29eab4e4866def2fed92119543a97aa2888594928f4777c5fdcc6035859d6b0。
待恢复v003身份为runtime/action4_runs/p2_a0_corner_roi115_finetune75_v003/RUN_SPEC.json；75轮未完成，last只供精确恢复，不是选优权重。

## 2. 数据、ROI、相机与评分

phrase2保持5200/650/650，原6500发布身份不改。CAD在data/DJI_Action4_construct/r3_v001/asset_manifest.json，canonical8固定0–7，物理尺寸不改。
原视频帧清单data/action4_real/cache_v001/frames_manifest.json：2850帧，3248×2464，2997/100fps，PTS time_base=1/11988。
K由INPUT_LOCK.video.K绑定：fx=fy2639、cx1623.5、cy1231.5，目前为未标定针孔假设，不是认证标定或证明零畸变。
完整机身框中心不变，连续正方形长边=1.5×max(w,h)，原图直接采样448；mask、点、K共用仿射，K_roi=A@K_native；不截ROI再拉伸、不二次扩张。
评分版本action4.pose_projection_all8_002.v1：按实例几何框一对一匹配，原图固定点ID计All8@2%；再以FN+FP、P95和较早step选优，漏检/无效发布保留分母。

解码完整参数在stage5_selection.decoder：p_valid_min=.8、q_corner_min=.6、minimum_ids=6、LM10次及完整框/支持/正深度约束。q=.2是另一个开发组合，不能静默与7300当前合同混称同协议最好。

## 3. 图片推理和完整eval入口

图片不使用track。实际实现：
src/eval/action4_staged_pipeline.py：
StagedPipeline.from_corner_bundles(a0_bundle,b0_bundle,eval_sam_cache,device=...)
→ corner_context(image_id,rgb)
→ predict_corner(rgb,K,canonical,sam_context=context,diagnostic_candidate=True)。

SAM缓存必须对应同一图像/冻结B0提示；新图不能冒用旧image_id。GT只供监督/评分，不用GT裁剪补漏。
完整650评估实际由src/eval/action4_stage2_eval.py消费不可变快照、RGB和合法SAM缓存，分别产生raw/corner候选报告。
src/eval/action4_corner_scoring_inputs.py与action4_corner_calibration.py负责评分和quality回执。
优先读SELECTION.selected.metrics、SEALED_RESEARCH及现有完整eval报告；本次不重跑。
通用python -m eval.action4_staged_eval --selection ...属于旧selection格式；不能直接把stage5 fixed_parent_selection文件当成其配置。阶段四使用专用corner pipeline/worker。
新eval须独立身份、明确请求和输出，不重启旧无限队列worker；最终test不用于调参。

## 4. 完整视频实际数据流

当前B0原始候选＋SAM当前传播＋实例可信历史
→ 实例关联及完整范围ROI决策
→ bbox_for_a0
→ 当前RGB/mask采样与K更新
→ A0原始姿态/角点
→ PnP候选或粗姿态回退
→ 跟踪/平滑
→ 保存浮点结果
→ 只读渲染。

核心实现：
- src/data/action4_video_roi_context.py、src/eval/action4_reanchor_roi_provider.py：当前支持/实例身份/完整范围与ROI合同。
- src/eval/action4_roi_video.py::run_part：实际RGB、ROI、K准备和模型前向。
- src/eval/action4_corner_video.py::predict_corner_video：SAM辅助、固定ID角点与几何解码。
- src/eval/action4_roi_video_job.py::run_job(config_path)：分片、身份检查、完整合并和编码。
- src/eval/action4_video_stabilization.py::generate、src/model/action4_video_pose_stabilizer.py：读取保存预测，因果历史处理，不重新调用模型。
- src/eval/action4_roi_video.py::render_standard_six：只读保存结果和demo同帧编码；当前ratios=[1.5]，函数名six不代表重做六档。

bbox_b0_raw、bbox_sam_visible、bbox_for_a0、track_id、frame/PTS及ROI来源要保留。SAM可见紧框不是完整机身；可靠当前SAM支持可续接本帧B0漏检，缺可靠当前观测的外推受限。允许双机ROI重叠，禁止复制mask或换身份。帧0–3采用frame4离线确认身份，不能称全链完全因果实时。

### 已完成的完整推理入口与复用

实际配置：outputs/action4_stage5_sealed7300_video_v001/CONFIG.json
不可变源码清单：同目录source/SOURCE_MANIFEST.json
实际controller：同目录source/controller.py
实际worker：同目录source/worker.py

历史实际入口是项目内.venv/bin/python -I -B <上述controller.py>，controller按CONFIG为worker分配GPU UUID和CPU。
**该目录已经完成，不可原地再运行**：日志/分片采用独占创建，会冲突。新实验须复制配置到新输出并按现有不可变源码打包机制登记，重新验证资源，不能直接重跑旧controller。

已有完整原始预测：
outputs/action4_stage5_sealed7300_video_v001/e1p50/predictions.jsonl
SHA256 67d08b2491328fa528d2db0687e8d48004ee8e2dc60dda9bdbf51a457cb159c1。
同根CORNER_TRIAL_RECEIPT.json绑定权重和解码。
B0+SAM实际轨迹来自INPUT_LOCK.video.track_replay：
outputs/action4_sam_reanchor_full2850_v003/run/FUSED_BOXES.jsonl。
该视频复用已完成B0/SAM结果，不能说每次A0推理重新运行了SAM。

### 当前V5完整视频及后处理入口

配置：outputs/action4_stage5_coherent_preparation_v006/CANDIDATE.json
源码：同目录source/SOURCE_MANIFEST.json
入口：scripts/action4_stabilize_video.py
参数：--source-manifest <上述清单> --config <上述CANDIDATE.json> --reuse。
--reuse会核验同身份回执后直接复用；本次未执行。
新后处理须新配置/新输出，不能覆盖旧文件。V5使用5帧历史、旋转2–3Hz自适应与180°/s上限，弱观测和不一致运动降低增益；精确参数以CANDIDATE.filter为准，不称已验证最优。

最终预测：outputs/action4_stage5_coherent_stable_v005/e1p50/predictions.jsonl
回执：outputs/action4_stage5_coherent_stable_v005/STABILIZATION_RECEIPT.json
只读编码API：
render_standard_six(video_root,input_lock,demo_reference,new_output_root,stabilization_receipt=receipt,encoder="libx264",encoder_threads=4)。
binding从COMPLETE/INPUT_LOCK读取，必须新输出；渲染不能改pose、插值或调用模型。

已交付完整视频：
/data/zy/vln/a-test/outputs/videos/20261029-072034-stage5-sealed7300-coherent-stable-track-e1p50-track/comparison.mp4
2850帧，1900×750，2997/100fps，
SHA256 8e8272c8ef15b1d55ce48abaeaecfa77e080be2621a696f6192629126b6519dd。
同身份直接复用。
scripts/action4_video.py是早期通用入口；scripts/action4_encode_parallel_video.py依赖旧专用产物；scripts/action4_tracked_roi_infer.py仍硬绑定A015770且依赖旁置CONFIG，均不能代替当前7300链路。

## 5. 尚未解决与恢复顺序

| 未解决问题 | 实际证据与下一步 |
|---|---|
| 质量过度自信 | 同批10400点标签/ID/仿射未发现错位，但q>=.6时1%真阳率约.686，隐藏点约.286；区分单点1%与All8@2%，校准/改进后同批复核，不伪设quality_ready |
| 原视频方向、厚度和抖动 | V5旋转变化P95约3.363°但不是精度；错误方向仍在。复核困难段原始/最终预测与来源，平滑不能纠正错误朝向 |
| 材料和六面绑定 | ray种子继承A0偏差，翻面闭环失败、第二机可靠点不足；先建立可靠静态材料XYZ/当前支持，固定ID不证明准确，不补背景点强行PnP |
| 原生SAM/region增强 | 宽搜索改善部分轮廓但结果混合、旋转仍跳变；保持候选与默认分离，同帧验证后再消费 |
| K及marker绑定 | 未标定相机；焦距和marker假设试验结果混合，不能仅凭低重投影残差改K或声称真实位姿正确 |
| ROI1.15未完成 | v001评分入口失败；v002队列背压导致NCCL超时，v003主机等待修复后用户暂停；恢复须核对step400状态、队列和claim，目标总计75轮 |
| 训练/部署ROI差异 | 1.5训练使用冻结B0，视频使用track-assisted ROI；统一RoiSpec/RGB/K后固定模型比较共同实例、跟踪补回和错误接续，不称全部gap已消除 |
| 全视频验收 | DELIVERY记录full_video_visually_watched=false、accuracy_accepted=false；完整观看并登记失败，冻结后才test，demo仅为参考 |

恢复顺序：
1. 等用户明确恢复；先读configs/execution/action4_active.json（r147）、CONTINUATION与STOP_RECEIPT，重新核查GPU4–7/CPU24–95和其他任务占用。
2. 固定7300/9500复用V5和原始2850预测定位问题；新身份才生成新视频。
3. ROI1.15从完整step400和v003身份受控续接；不重启终态v001/v002，不追加另一套75轮；规定eval消费完再选best和严格重载。
4. 删除的历史中间快照已退役，旧eval请求只能作证据，不盲目派发；当前恢复和队列依赖保留。
5. 新模型在真实track-assisted ROI上核查兼容性，再材料/六面几何及后续CoTracker；不拿真实视频预测pose冒充GT。
6. 分别报告代码通过、训练完成、真实效果；失败时保留父组合，默认不自动覆盖。

## 6. 本次核查范围

重新读取实际selection、INPUT_LOCK、封存登记、CONFIG/COMPLETE/DELIVERY和源码函数签名，当前A0与实际B0副本SHA一致。未做forward、完整eval或视频调用。
工程和效果结论沿用已登记证据，没有因本页整理升级验收状态。


## GitHub下载补充（2026-11-08）

视频、当前权重与实际源码快照下载及依赖边界见[action4_github_assets.md](action4_github_assets.md)。保持暂停，未启动模型任务。
