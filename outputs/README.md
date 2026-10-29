# Action4 输出目录索引

## 当前数据交付

| 目录 | 用途 |
|---|---|
| coverage1000_v001/ | 新1000及合并6500的交付报告、实际分布、准入证明、代表图和来源记录 |
| phrase2_occlusion500_v001/ | 追加500的历史验收、遮挡分布、来源与去重缓存 |
| phrase2_generation_v001/ | 原5000的历史验收、来源与去重缓存 |
| phrase2_cleanup_v001/ | 本次实际删除清单、保留身份基线、清理后真实接口检查 |
| action4_phrase2_analysis_v001/ | 原5500分布及demo对比分析 |

正式数据仅从 data/phrase2 读取。上述三个已完成合成批次的临时RGB/mask、无用预览、失败候选与日志已直接删除，未将大体积废文件改名归档。
批次workers/仅保留正式标签直接引用的FRAME_SNAPSHOT/CANDIDATE及少量身份/终态元数据，属于来源记录，不是待运行进程。
当前正式状态见configs/execution/action4_active.json。不要重启历史PID、队列或已退役候选接口。

## 模型与历史研究

- 默认模型：models/default/A0.pt、B0.pt，未因清理改变。
- videos/：此前交付视频和INDEX；不是本批数据的最新模型效果。
- action4_stage2_integration_v001/：独立A0/B0训练、恢复、异步eval及视频工程证据。
- required/以及其余action4_*目录：保留现有研究结果、冻结资产和其他任务依赖；本次没有扩大为全项目删除。
- 历史归档仍在archive/legacy，现有路径和迁移记录保持不动。

阶段结论集中在docs/stages原conclusion；本目录只放输出，不放正式源代码。数据、权重、大日志和逐样本产物不进入普通Git。


## 2026-10-29 暂停交付索引

研究、训练、评估均已暂停。action4_pause_cleanup_20261029记录进程停止、删除计划/回执、保留模型验证及发布；videos保留已交付整片，action4_stage5_coherent_stable_v005保存V5来源与失败审查。ROI115三次运行的失败、未完成eval队列和恢复依赖保留，不可重启旧controller。无引用中间权重被明确退役后，旧eval请求只作历史记录；不以它们宣称可复跑。data中的正式6500未改。
