# 模型资产

本目录保存模型权重、checkpoint、adapter 及其本地资产。正式模型代码位于 [src/model](../src/model/README.md)，测评时的加载和设备设置位于 [src/eval](../src/eval/README.md)。

根据真实模型需求建立子目录，记录来源、版本和校验状态。已有 checkpoint 不覆盖，新训练输出使用新身份；不自动下载模型或修改共享权重。
除本 README 外，全部内容默认被 Git 忽略。目录骨架不代表已有模型或完成模型验证。
