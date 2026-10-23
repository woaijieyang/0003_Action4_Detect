# 独立模型资产与选择

当前登记见服务器 models/CURRENT_RESEARCH.json；所有模型均为研究候选，真实任务尚未验收通过。
- 合成集较优的 A0：models/A0/best.pt，step13350，固定表面 PnP。
- 真实视频改善的 A0 候选：models/candidates/a0_online_rgb_v001/best.pt，step14000，网络预测刚体位姿再投影八点。
- 当前 B0：models/candidates/b0_photometric_v001/best.pt，step3220，refined boxes + center scoring。
- models/B0/best.pt 保留旧step2200父模型，不是最新推荐B0。
- models/candidates/a0_frozen_surface_pose_v001/best.pt 是在线RGB试验的必要父依赖，不能按“旧文件”直接删掉。

各包包含推理所需参数、配置、阶段身份；A0/B0无共享参数存储。正式使用应通过configs/eval中的明确profile选择，不按mtime取最新文件。没有全面优于所有指标的A0，不把真实轮廓改善当成角点ID正确。
继续训练从各实验best权重初始化新优化器；best不含完整优化器历史，不能冒称精确中断恢复。在线RGB冻结策略用src/model/action4_online_pose_rgb.py及其训练器。
原输出目录只保留各run best及同inode选优别名、完成回执和评估证据，非最佳过程权重已清理。
本轮封存清单：outputs/action4_online_rgb_release_v001/SEALED_MODELS.json。权重不进入普通Git提交。
