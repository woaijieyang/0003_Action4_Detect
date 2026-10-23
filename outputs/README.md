# Action4 运行输出

- videos/：已完成的8段可视化视频及INDEX.json；文件名为“时间-阶段.mp4”。时间取原文件mtime（Asia/Singapore），仅用于排序，不冒充准确生成事件时间。原始/demo输入视频仍在data。
- action4_stage2_integration_v001/：独立A0/B0训练、精确恢复、异步eval、封装与视频的本轮实际工程证据。两步训练未改善best；重复工程权重已清理，ARTIFACTS_RETIRED.json指向保留父best。
- action4_online_surface_contract_v001/：在线ROI/稀疏三维表面对应点与梯度检查。
- action4_organization_v002/：依赖闭包、整理计划、迁移清单与完成回执。
- required/与仍被引用的历史输出：当前数据准入、冻结组件和另一任务的依赖，保留原路径。
- phrase2_generation_v001/：另一任务拥有，本次不修改、不归档。

25组无当前依赖的旧运行证据已归档到archive/legacy/organization_v002/outputs。原路径可按MOVED.jsonl定位；历史报告不重写成“新结果”。重要结论集中在docs/stages原conclusion，目录身份与当前入口见docs/action4_execution_index.md。
真实视频仍存在机身/支架混淆与立体框偏移；工程通过不表示真实精度达标。权重、视频、日志及逐样本预测不进入普通Git。
