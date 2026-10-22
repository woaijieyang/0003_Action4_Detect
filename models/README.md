# 模型资产

正式保留位置为A0/best.pt和B0/best.pt，阶段权重独立。
当前包是开发集选择的研究候选，真实视频验收未通过，禁止自动升级为生产模型。
A0包内有冻结RGB backbone与几何任务头；B0包内有检测器与框/质量修正头。
load_bundle和warm_start位于src/model/action4_release_bundle.py。后者从best重新初始化优化器。
历史各运行best及必要父模型按archive/legacy清单归档；非最佳过程快照不保留。
pretrained保留原预训练资产及其来源。权重不会进入普通Git提交。
