# Action4 暂停交接与文件索引

本页只导航；阶段事实和偏差继续维护原 plan/conclusion。更新时间：2026-10-29，Asia/Singapore。

| 入口 | 实际用途 |
|---|---|
| configs/execution/action4_active.json | 当前用户暂停权威，禁止自动恢复 |
| docs/stages/2026-10-24-1-plan-Action4第二阶段识别与八角点增强.md | 八阶段主计划 |
| docs/stages/2026-10-24-1-conclusion-Action4第二阶段识别与八角点增强.md | 实际执行、失败、修复和未完成项 |
| docs/stages/2026-10-26-1-plan-A0六档ROI扩张与B0预测框条件化微调.md | ROI实验及适配要求 |
| configs/model/action4_stage5_selection_v001.json | B09500＋研究A07300、ROI1.5的阶段五父组合 |
| models/candidates/action4_improvement | 各阶段自包含封存模型及实际选择回执 |
| models/default | 历史默认A014000/B03220，未被研究候选覆盖 |
| runtime/action4_runs/p2_a0_corner_roi115_finetune75_v003/RUN_SPEC.json | 暂停的ROI1.15不可变运行身份 |
| outputs/action4_improvement/stage4_corner/p2_a0_corner_roi115_finetune75_v002/last.pt | step400完整恢复状态；不是最终best |
| outputs/action4_stage3_native_prediction_v001/CONTINUATION.json | 最新暂停及后续依赖记录 |
| outputs/action4_pause_cleanup_20261029 | 停止、保留、删除、检查和发布清单 |
| outputs/videos/20261029-072034-stage5-sealed7300-coherent-stable-track-e1p50-track/comparison.mp4 | 已交付V5完整开发视频，未通过真实准确性验收 |
| data/phrase2 | 正式6500数据，不重生成、不重新划分 |
| archive/legacy/pause_20261029/document_snapshots | 本次整理前旧导航原文 |

## 实际数据流

图片：冻结B0原始预测 → 固定比例RoiSpec → 当前RGB与mask共同采样、K更新 → A0 → 原始姿态/角点与PnP候选 → 原图坐标统一评分。GT只提供训练监督或评价，不修正推理裁剪。

视频：B0原始检测＋SAM当前支持＋实例历史 → 关联与完整机身ROI决策 → bbox_for_a0 → 当前RGB/mask/K → A0原始输出 → 已实现的PnP、跟踪与平滑 → 保存 → 只读渲染。SAM可见紧框不等于完整机身范围；固定ID不等于跟踪准确。

ROI1.15视频接口已有工程准备，尚无完成75轮的模型或对应完整效果验收。继续前要验证实际选中ROI进入模型、与训练采样一致，以及完整视频效果；不得仅因接口通过就宣布gap全部消失。

## 暂停后恢复

必须由用户重新授权。先读暂停回执和RunSpec，确认step400状态、未完成eval及资源；原75轮未完成，不称本次完成。恢复不得启动已终态v001/v002；应使用受验证的恢复流程并核对队列claim。已删除历史中间快照只保留报告，不能再把历史eval请求当可重跑任务。默认权重、封存模型、恢复状态及被引用缓存均保留。
