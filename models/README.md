# Action4 权重位置

## 默认加载目录

服务器 /data/zy/vln/a-test/models/default 只存放两个checkpoint，不放日志、历史包或说明文件：

| 文件 | 阶段与来源 | 用途 |
|---|---|---|
| default/A0.pt | onlineRGB，step14000，joint_pose/network | 下一轮A0联合表面/位姿训练父权重及默认推理 |
| default/B0.pt | photometric，step3220，online_b0/center | 下一轮B0训练父权重及默认推理 |

权威登记：configs/model/action4_best_registry_v002.json。
显式默认推理profile：configs/eval/action4_default_v001.json。
models/CURRENT_RESEARCH.json记录默认与历史比较用途；不要按目录mtime或搜索best.pt选择权重。
默认训练从选定权重初始化新优化器，同run精确中断恢复是独立临时状态，不能把旧last优化器嫁接到这些权重。

## 留档权重

留档集中archive/legacy/checkpoints，按原文件mtime（Asia/Singapore）＋阶段名命名，不再使用best文件名。mtime用于排序，不冒充未知训练完成时间。

主要历史包：
- archive/legacy/checkpoints/20261023-041150-A0-PnP-step13350.pt
- archive/legacy/checkpoints/20261023-041150-B0-historical-step2200.pt
- archive/legacy/checkpoints/20261023-071308-A0-frozen-surface-pose-step8625.pt

其他历史实验与工程测试夹具同样按时间和阶段标识留档，不能误作默认模型。预训练ResNet位于pretrained，不属于当前A0/B0默认权重。

旧目录不留best软链或硬链。configs/model/action4_checkpoint_locations_v001.json提供84条原路径→实际位置映射；自包含包加载器使用model.action4_checkpoint_paths.resolve_checkpoint解析历史引用。已完成实验的配置和权重内原始来源字段保持原文，不重写旧实验身份。
旧脚本若直接调用torch.load原物理路径而绕过加载器，重放前须按映射显式解析；不得因旧路径消失而自动重训。

加载前校验角色、位置、step、文件大小、SHA与config身份。新默认目录只保留A014000/B03220，三份历史包实际CPU加载通过，七类防误加载反例通过。归档保留参数内容，4个重复硬链接合并，不是删除4份不同模型。
默认只是明确的继续训练选择，真实精度仍未验收通过。

迁移及核验：outputs/action4_checkpoint_layout_v001。
详细说明：docs/action4_checkpoint_layout.md。
