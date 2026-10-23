# Action4 模型存放与默认加载

## 默认权重：只保留两份

服务器目录 /data/zy/vln/a-test/models/default：

- A0.pt：onlineRGB，step14000，joint_pose/network。
- B0.pt：photometric，step3220，online_b0/center。

本目录只有这两个checkpoint。权威登记configs/model/action4_best_registry_v003.json，默认推理配置configs/eval/action4_default_v002.json。继续训练从各自best权重初始化新优化器；默认权重不含完整优化器恢复历史。
models/CURRENT_RESEARCH.json明确默认和历史比较用途。严禁按mtime或模糊搜索best选择模型。

## 历史权重：models/archive

用户最新要求将确有价值的阶段历史最优和参考模型放models/archive，覆盖此前archive/legacy/checkpoints位置。文件继续使用原mtime时间（UTC+08）＋阶段名，不使用best文件名，不冒充训练完成时间。
当前仅保留9份：5份阶段历史结果/父模型，以及4份完成训练的基线与损失消融参考。明确索引configs/model/action4_archive_registry_v001.json：

| 文件 | 保留理由 |
|---|---|
| 20261023-041150-A0-PnP-step13350.pt | Strong retained synthetic geometric baseline; independently loadable package |
| 20261023-041150-B0-online-step2200.pt | Previous completed online B0 stage best; photometric comparison parent |
| 20261023-071308-A0-frozen-surface-pose-step8625.pt | Completed frozen-surface stage best and explicit online-RGB parent |
| 20261022-095743-A0-heatmap-through80-step1475.pt | Selected across cumulative80 epochs; older heatmap baseline |
| 20261022-122404-B0-fromA0-box80-step4660.pt | First completed80-epoch bbox-protocol B0 stage best |
| 20261022-193452-A0-independent-baseline-step6300.pt | Completed80-epoch baseline; frozen backbone provenance and method comparison |
| 20261022-170928-B0-ablation-baseline-step60.pt | Completed80-epoch ablation best; control for GIoU and quality changes |
| 20261022-200702-B0-ablation-GIoU-step9300.pt | Completed80-epoch GIoU ablation best |
| 20261022-185434-B0-ablation-GIoU-quality-step5220.pt | Completed80-epoch GIoU plus quality ablation best |

其余109个旧checkpoint、恢复/初始化/中途/工程测试快照及重复别名已删除；失败报告、训练/评估结论继续保留。删除文件逻辑大小合计15,280,224,282字节，包含硬链接，不能当作实际磁盘释放量。
预训练ResNet骨干仍在pretrained，属于基础初始化资产；特征/maps/probability张量缓存不是checkpoint，没有按.pt扩展名误删。原始数据、标签和并行phrase2工作不动。

## 加载与审计

model.action4_best_registry核对默认文件角色、架构、step、SHA和配置身份；当前两默认模型与9份历史模型均完成严格CPU加载、参数有限性验证。
configs/model/action4_checkpoint_locations_v002.json区分保留路径迁移与明确删除。resolve_checkpoint只解析已登记的保留项；已删除项明确报intentionally retired，不静默回退默认模型。历史权重存在不同架构，应使用对应历史加载器，不把历史包直接套到当前网络。
保留的5个阶段模型不是全都优于当前默认；4份对照也不能作为默认。真实视频精度仍未通过。
清理证据outputs/action4_checkpoint_retention_v001，目录与接口说明docs/action4_checkpoint_layout.md。
