# 默认与留档checkpoint布局

本页只说明文件布局和加载合同，不另立训练计划。用户明确要求：留档文件使用时间＋阶段名，默认目录只保留默认加载的checkpoint。

## 当前默认

- 服务器根：/data/zy/vln/a-test
- 默认目录：models/default，仅A0.pt、B0.pt两文件。
- A0：14000，joint_pose，network解码，自包含backbone/surface/pose。
- B0：3220，online_b0，center评分。
- 登记：configs/model/action4_best_registry_v002.json。
- 推理：configs/eval/action4_default_v001.json。
- 训练：model.training.action4_stage2_core在full_training构造时默认使用上述登记；指定其他新父模型必须提供显式新版本parent_registry，不靠扫描best或mtime。

## 归档与来源

历史文件统一archive/legacy/checkpoints/YYYYMMDD-HHMMSS-阶段名.pt；时间取原文件mtime UTC+08，仅用于排序，不推断实际训练时间。历史权重含正式实验、候选失败及明确的工程夹具，不能因扩展名相同就作为正式模型。
本轮84条旧路径迁到80个实际文件位置：默认2份、归档78份；4组同inode别名合并。inode/大小/mtime核验通过，未重写tensor或checkpoint元数据，未删除独有参数。
旧models/A0、models/B0、candidates目录已清空移除。原README保留至归档metadata，旧目录不残留best别名。

configs/model/action4_checkpoint_locations_v001.json记录完整映射。model.action4_checkpoint_paths.resolve_checkpoint首先读取确实存在的请求文件；缺失时只允许精确的已登记项目内迁移项，不按名称猜测，不跳转目录外。
load_bundle及best身份校验均接入该解析器；当前默认profile使用直接新路径。旧冻结配置不机械全局改写，因修改配置会改变历史config_id。需要重放直接torch.load旧路径的历史脚本时，先按映射显式解析。原历史回执里的文件路径是历史记录，不是新的启动入口。

## 验证范围

默认训练模型构造、默认推理加载、3份历史自包含包CPU加载通过；角色互换、历史错选、step/SHA/config不一致的七个反例拒绝。84条迁移引用均能解析到实际文件；默认目录恰好两文件，归档权重文件名无best。
本轮只改布局和解析，不更改参数、模型数值或训练结论。第一阶段data/phrase1及另一任务phrase2生成文件不动。原checkpoint内来源字段保留，真实视频精度仍未通过。

记录：outputs/action4_checkpoint_layout_v001/PLAN.json、MOVED.jsonl、COMPLETE.json、VALIDATION.json。发布身份以PUBLISHED.json与远端main核验为准。
