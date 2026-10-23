# 独立模型资产与选择

**下一轮训练唯一父权重登记：configs/model/action4_best_registry_v001.json。A0=step14000，B0=step3220。models/A0与models/B0是保留的历史基线，不按这些短目录名推断当前选择。**

当前登记见服务器 models/CURRENT_RESEARCH.json；所有模型均为研究候选，真实任务尚未验收通过。
- 合成集较优的 A0：models/A0/best.pt，step13350，固定表面 PnP。
- 真实视频改善的 A0 候选：models/candidates/a0_online_rgb_v001/best.pt，step14000，网络预测刚体位姿再投影八点。
- 当前 B0：models/candidates/b0_photometric_v001/best.pt，step3220，refined boxes + center scoring。
- models/B0/best.pt 保留旧step2200父模型，不是最新推荐B0。
- models/candidates/a0_frozen_surface_pose_v001/best.pt 是在线RGB试验的必要父依赖，不能按“旧文件”直接删掉。

各包包含推理所需参数、配置、阶段身份；A0/B0无共享参数存储。正式使用应通过configs/eval中的明确profile选择，不按mtime取最新文件。没有全面优于所有指标的A0，不把真实轮廓改善当成角点ID正确。
继续训练从各实验best权重初始化新优化器；best不含完整优化器历史，不能冒称精确中断恢复。在线RGB冻结策略用src/model/action4_online_pose_rgb.py及其训练器。
原输出目录只保留各run best、完成回执和评估证据；本次已移除当前A0/B0同inode选优别名，非最佳过程权重已清理。
本轮封存清单：outputs/action4_online_rgb_release_v001/SEALED_MODELS.json。权重不进入普通Git提交。

本轮工程集成没有产生更优模型：A0/B0均选step0，与上述父best参数逐值一致。恢复last、非best控制快照、预检包与重复工程封装权重在所有消费者结束后删除；工程selection/SEALED历史回执由ARTIFACTS_RETIRED.json标明已退役，不能作为活动权重入口。正式新实验从best权重开始；同实验意外中断才使用临时完整状态做精确恢复，终态清理该临时状态。


当前加载入口：
- 下一轮训练A0：models/candidates/a0_online_rgb_v001/best.pt（step14000，joint_pose，network解码）。
- 下一轮训练B0：models/candidates/b0_photometric_v001/best.pt（step3220，online_b0，center排序）。
- 固定父权重推理profile：configs/eval/action4_next_training_best_v001.json。
- CURRENT_RESEARCH.json使用next_training / synthetic_baseline / real_video_candidate明确用途，不再提供容易误读的顶层A0字段。

model.action4_best_registry核对路径、阶段、step、字节大小、SHA、包配置和来源训练配置。stage2_core在full_training构造时默认绑定该版本登记；改用新的best必须显式发布新登记，并在新配置中指定parent_registry，不能只改路径绕过。文件身份不变时同进程复用摘要，避免反复哈希。明确profile也在加载前校验两阶段身份。
A0自包含包config_id与原在线训练training_config_id属于不同层次；登记同时绑定两者，实际17个训练pose张量与原best逐值相同，不能把这两个合法身份差异误判为权重混用。
此选择为下一轮联合surface/pose训练的明确起点，真实精度尚未通过，合成PnP基线仍单独保留比较。
