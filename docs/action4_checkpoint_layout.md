# 默认与精选历史checkpoint布局

用户最新要求：当前默认独立存放；阶段历史最优与有参考价值的历史模型放models/archive；其余历史checkpoint删除。此次范围是模型权重，不将原始数据或特征张量缓存按扩展名清除。

## 当前布局

- models/default/A0.pt：step14000，在线RGB联合位姿模型，下一轮默认父权重。
- models/default/B0.pt：step3220，摄影增强B0，下一轮默认父权重。
- models/archive/：9份经过筛选的历史checkpoint，时间＋阶段名。
- models/pretrained/：预训练ResNet基础初始化资产。
- 登记：configs/model/action4_best_registry_v003.json（默认）、configs/model/action4_archive_registry_v001.json（历史）。
- 推理：configs/eval/action4_default_v002.json。

默认目录仅两个文件。历史9份分别是A0PnP13350、B0online2200、A0frozen-surface-pose8625、早期A0累计80轮选优step1475、早期B0框协议80轮选优step4660，以及A0独立baseline6300、B0baseline60、B0GIoU9300、B0GIoU+quality5220四份完成训练的对照。阶段内step不能跨不同run直接比较优劣。
档名日期来自原文件mtime UTC+08，仅用于排序，不声明未知的实际训练完成时间。当前9份均按各自真实网络拓扑严格CPU加载且参数有限。

## 清理与引用语义

从上一版归档中移入9份保留权重。其余旧checkpoint、D0工程运行、step0失败候选、恢复/初始化/中途快照及重复诊断别名，共109条文件路径已删除。逻辑文件大小合计15,280,224,282字节，含硬链接；不宣称该数值是磁盘实际释放量。结论、失败记录与删除明细保留。
特征缓存和模型权重在内容层面分开，未因后缀.pt相同而误删数据缓存；另一任务及其生成输入保持隔离。
configs/model/action4_checkpoint_locations_v002.json为当前解析表：29条保留迁移引用，173条历史原路径/中间迁移路径明确退役。已删除项报FileNotFoundError并注明intentionally retired，不自动读成默认权重。旧版本映射仅是历史记录，当前resolver已经切换v002。

## 验证

默认模型真实构造/加载及七项错误身份反例通过；历史9份严格CPU加载通过；默认目录2文件、历史目录9文件及全部29/173条引用行为核验通过。
初次历史加载检查误用共享Action4Net直接加载早期独立ROI分支，出现unexpected roi_branch键；检查后改用已有prepare_checkpoint_topology恢复原拓扑，严格加载通过。仅修正审核脚本，未改权重、未放宽strict校验，也未影响默认模型。
工程测试与文件整理不表示真实视频精度通过。完整证据在outputs/action4_checkpoint_retention_v001，发布身份以PUBLISHED.json和远端main核验。
