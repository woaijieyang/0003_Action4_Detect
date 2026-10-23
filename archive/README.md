# 历史归档索引

统一位置：服务器项目根下archive/legacy，保留原相对目录结构。
真实失败与旧方案仅作证据，不是当前启动入口。
迁移/删除逐项记录：outputs/action4_release_preparation_v001/CLEANUP_MANIFEST.json及cleanup_journal.jsonl。
当前权威：configs/execution/action4_active.json。当前模型：models/A0/best.pt、models/B0/best.pt。
原始数据与冻结Blender不参与清理；旧运行各best保留在归档，非最佳历史快照清除。
Git只跟踪本索引，历史资产和过程产物不上传。要恢复归档项，按清单原路径恢复并检查依赖，禁止覆盖当前代码。

本轮在线RGB整理：archive/legacy/release_r067，保留原相对布局；逐项清单与日志在outputs/action4_online_rgb_release_v001。当前选择以models/CURRENT_RESEARCH.json与r067为准，前述models/B0/best.pt为历史父模型。已合并视频分片及过期claim、非best/last只在消费者退出和覆盖核对后删除；数据及模型父依赖未删。
