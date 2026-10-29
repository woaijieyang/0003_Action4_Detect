# Action4 数据资产分类

第一阶段正式数据的唯一物理位置为 data/phrase1/train、data/phrase1/eval、data/phrase1/test，1600/200/200图；data/action4_dataset_commit.json保存原2000图的发布身份。整理没有改写这些图像、标签、角点ID或split。

| 类别 | 保留位置 | 用途 |
|---|---|---|
| 原始视频与逐帧数据 | final_test_data、action4_real | 原视频与2850原生帧、时间戳 |
| demo参考 | action4_reference、reference | 参考视频/逐帧；不是认证GT |
| DJI本体 | DJI_Action4_construct、initial_DJI_Action4_construct | 原始模型与冻结规范尺寸、canonical8 |
| 场景组件 | action4_scene_assets、action4_scene_templates | 冻结腕部/腕带/纹理和版本化场景 |
| 第一阶段正式标注 | phrase1/train、phrase1/eval、phrase1/test | RGB、完整机身框、8角点、mask及表面监督sidecar |
| 仍被引用的合成来源 | action4_synth | 只留下发布/运行依赖链仍需的7组来源 |
| 候选与诊断输入 | action4_scene_candidates、action4_diagnostics、action4_surface_regions_v001 | 场景候选、检查标注、规范区域定义 |

60组无当前依赖的旧预览和失败候选已迁至 archive/legacy/organization_v002/data/action4_synth；失败验收数据没有静默混入正式集。逐项位置见 outputs/action4_organization_v002/MOVED.jsonl。仍被引用的冻结资产保留原路径，避免破坏另一任务的运行输入。

第二阶段 phrase2_v001 由另一任务负责，相关文件、配置、产物和依赖保持隔离，本次不归档、不删除、不提交。其是否已准入不能由目录存在判断。检查标注与训练标注分离，运行可视化/视频在outputs，不放data。所有数据资产除本README外均不进入普通Git。

用户最新要求已落实：三个split通过原子目录交换迁入phrase1。旧data/train、data/eval、data/test仅为项目内部兼容链接，不是数据副本；现有manifest内部逻辑路径保留，以维持历史哈希及并行消费者。17,211个文件的inode/大小/mtime/ctime逐项不变，三个manifest内容哈希不变。迁移与接口复核见outputs/action4_organization_v002/PHRASE1_MIGRATION.json和PHRASE1_POSTCHECK.json。新文档与人工访问统一使用phrase1，另一任务的新数据不要写入这些第一阶段目录。


## 2026-10-29 当前引用状态

上文“第二阶段尚待准入”为历史记录。现有phrase2已发布6500张，train/eval/test为5200/650/650；训练消费者已有相应准入回执。本次暂停整理不移动、重划分或改写任何正式RGB、标签、Blender资产、原视频与抽帧，也不删除并行数据生成任务资产。当前实际训练与模型状态见docs/action4_execution_index.md。
