> 当前入口已更新：见 [ARCHITECTURE.md](ARCHITECTURE.md) 和 execution active 指针。下文为按时间追加的历史记录，旧 D0/ROI 控制器不得据此重新启动。

# Action4 当前模块与执行入口

唯一机器权威为 `configs/execution/action4_active.json`。当前机器配置以active指针为准；r024使用已发布2000数据，新ROI的D0 v008通过后衔接A0权重热启动的10轮ROI适应；原检测器保留。历史A0/B0、一次正式test及两版2850帧视频已运行，但定位效果未达标，不能视为任务完成。阶段计划/结论统一在 `docs/stages/2026-10-19-1-*-Action4角点复现增补.md`。

| 职责 | 当前实现 | 范围 |
|---|---|---|
| 28维独立参数采样 | `src/data/action4_parameter_sampling.py` | 两侧位置、0.4–1.4m深度、整体佩戴姿态、遮挡几何、光照；场景不随机 |
| 完整模板常驻应用 | `src/data/action4_template_runtime.py` | 每worker一次加载；冻结DJI/A/S/F/K/点序，实际更新root、布片和灯光，导出实际角点 |
| 一次增量碰撞检查 | `src/data/action4_template_collision.py` | 缓存基线、实际曲面与包含检查，UNKNOWN拒绝；不冒称完整人体机械仿真 |
| 固定场景候选批次 | `src/data/action4_template_batch.py`；`scripts/action4_render_template.py` | 同场景连续8张，保留拒绝/失败、明确attempt与来源；不直接接受训练数据 |
| 原生RGB/visible/amodal | `src/data/action4_modular_pixels.py`、`action4_mask_semantics.py` | 复用原实际渲染/掩码语义，源文件冻结 |
| 最终GPU3/Host/consumer | `src/eval/action4_template_quality.py`；`scripts/action4_template_quality.py` | 新template证据协议；不用伪造旧pose proof；像素/来源闭合和真实训练目标接口 |
| GPU数值核 | `src/eval/action4_gpu_quality.py` | 显式GPU3 FP64原始8点与mask hull；CPU仅测试oracle |
| 数据加载与全8点监督 | `src/data/action4_adapter.py`、`action4_dataset.py` | 原坐标/ROI接口；隐藏点不被visibility掩掉；原正式入口拒绝unaccepted |
| 模型与训练 | `src/model/action4_network.py`、`action4_losses.py`、`action4_decode.py`；`src/model/training/` | 现有ResNet50任务网络、D0/A0/B0与ckpt约束；已完成多轮有界训练，当前改进冻结A0特征，效果待验证 |
| eval/test/视频 | `src/eval/action4_runner.py`、`action4_video.py` | 正式整图推理不读GT；GTROI另标诊断；无跨帧训练/平滑；已有2850帧视频IO完成但定位不合格 |

当前配置：`configs/data/action4_random_sampling_v005.json`、`action4_template_environment_v001.json`、`action4_template_pilot2000_v009.json`。full运行专用调度脚本和状态在 `outputs/action4_template_full2000_v001/`；这些是本轮编排产物，不是复用算法模块。配置分片位于 `configs/execution/action4_template_full2000_v001/`。

候选数据在 `data/action4_synth/pilot2000_v007_candidate/`；各次诊断、叠图、统计、日志、失败和检查在 `outputs/`。完整正式数据目标为 `data/train/`、`data/eval/`、`data/test/`，各含 `images/`、`masks/`、`annotations.jsonl`；现已正式发布。完整性提交描述为 `data/action4_dataset_commit.json`。

历史候选在 `outputs/action4_template_assembly_candidate_v001/` 与 `outputs/action4_template_split_protocol_candidate_v001/`。正式实现已合入 `src/data/action4_template_assembly.py`、`action4_split_protocol.py`，入口 `scripts/action4_publish_template.py`；正式发布证据在 `outputs/action4_template_publication_v001/`，历史发布阶段训练配置 `configs/model/action4_pilot2000_v005.json`，当前训练配置以active引用为准（`configs/model/action4_roi_adaptation_v001.json`）。三个模板跨split共享是明确协议，不能伪改背景ID；旧strict-background默认必须保留，新协议验证独立样本/组/采样身份。隐藏8点监督保持原实现。

历史隔离：`action4_render_cached.py`、`action4_render_fast.py`、旧v006窄幅计划与旧统一quality controller均不是本轮启动入口。旧源部件/冻结资产仍可按明确引用作为依赖，不能删除或按mtime自动选用。旧`action4_synthesis.py assemble`和旧modular/physical-pose release不直接兼容新template候选，禁止补假字段冒充兼容。发布、训练或推送均以当前阶段实际授权/证据为准。

参数/算法/ID冻结清单：`configs/data/action4_sampling_freeze_v001.json`；总体检查频率为每小时一次，内部批次自动质量检查继续。

## 当前训练与后处理入口（2026-10-22更新）

- 当前D0：`template2000_d0_v008`，当前配置FP32。此前AMP几何错误已修复；随后B0 AMP梯度出现非有限值，正式后续改FP32。历史失败保留，不再以“模型继续AMP”的旧说明执行。
- 当前唯一训练衔接实例：`outputs/action4_training_performance_v001/roi_adaptation_training_controller_v001.py`。等待当前D0正式通过与源码/初始化权重身份验证，才提交`template2000_a0_roi_adapt_v001`。CPU72/GPU2–7，新optimizer，五轮；不重做A0/数据，不自动新test。
- `src/model/training/action4_stage_policy.py`：默认joint不变；显式B0 frozen_A0_features冻结backbone/decoder/corner_head，检测头可训练。冻结前后参数及buffers校验；不代表检测准确。
- `src/eval/action4_video_shards.py`，薄入口`scripts/action4_video_shard.py`：原生帧分片推理与绘制，必须经过完整合并和编码验证才能作为完整视频。
- `src/eval/action4_video_merge.py`：拒绝帧重复、缺失、时间轴/模型身份不同和非法角点；保持原始预测。
- `src/data/action4_png_admission.py`：单次编码调用内并行验证PNG；最终重新校验输入。不是跨运行永久缓存。
- 上述视频模块正式位置12项回归已通过，记录`outputs/action4_parallel_video_candidate_v001/installed_module_tests.log`。既有训练VideoQueue仍调用串行视频入口；安装分片模块不等于已自动改为并行视频调度。后续须显式采用已验证分片/合并编排，不能声称集成已完整完成。

## 历史隔离与尚未合入候选

`continue_a0_b0_v002.py`、`run_final_test_v002.py`及旧同名脚本只保留历史，不再作为当前执行入口。之前test已消费，未来重复评估不得冒充全新盲测。旧训练不因配置更新自动重启。

刚性投影位于`outputs/action4_rigid_projection_candidate_v001/`，仍为离线候选；真实内参未实测，未进入正式预测。保留独立角点raw作为对照，不把离线GTROI改善当作真实整图成功。当前验收必须同时核对完整eval、冻结状态及真实视频中的八角点包围效果。


独立ROI正式接口：`Action4Net.enable_independent_roi_features()`在A0加载后建立冻结副本，检测特征可训练；stage policy为`independent_A0_roi`，目前只支持B0，C1不自动兼容。resume/eval按checkpoint策略建立拓扑后strict load，重复enable不得重置ROI。安装回归58项57通过1环境跳过；GPU单张实际尺寸一致性和双rank CPU机制通过，不构成准确率证明。旧冻结策略及其失败结果保留。


当前ROI适应：`src/model/training/action4_initialization.py`显式验证A0旧权重与ROI-only预处理迁移，原网络及full_size必须一致；新optimizer记录迁移，不冒称exact resume。A0允许声明的ROI扰动，D0固定ROI不允许扰动。当前448/2.25和扰动设置需正式训练评估，扩大覆盖不保证准确；组件权重禁止正式消费。


### 独立组件训练对照 v001（2026-10-22）
正式职责、checkpoint语义和文件索引见[action4_model_interfaces.md](action4_model_interfaces.md)“独立组件实际训练接口”。当前套件configs/model/component_trials_v001，运行outputs/action4_independent_trials_v001；当前eval进程身份为EVAL_REPLACEMENT.json。运行中模型源码冻结，后续官方算子替换/格式整理必须新版本验证，不能破坏恢复身份。
