# eval：模型测评

存放测评时的模型实例化与 checkpoint 加载编排、设备设置、环境初始化、推理调度、指标计算和成绩汇总。Action 4 已实现 `action4_runner.py`（RGB 推理、eval/test 与有限解码调参）、`action4_metrics.py`（固定八角点的最大一一匹配）、`action4_video.py`（完整原始预测与精确时间轴红框视频）和 `action4_feedback.py`（候选事件与有证据的 S1 请求）。具体命令和验收见 [训练与测评执行说明](../model/training/README.md)。代码和工程测试不等同于模型性能已通过验收。

按需要再建立 loaders、environments、metrics 等模块，不提前创建空子目录。正式模型定义调用 [model](../model/README.md)，数据读取与预处理调用 [data](../data/README.md)。

## 责任边界

- checkpoint 选择、加载顺序、测评设备和运行配置归这里；模型结构及其状态接口由 model 定义。
- 数据转换、解析和划分归 data；测评协议、有效样本判定、计分和聚合归 eval。
- 测评器可读取答案计分，模型输入不可因此包含隐藏答案或测试集目标。评测标签与模型输入的路径和接口应分离。
- 导入模块不启动环境、不下载、不加载模型。实际运行显式调用，并使用任务独占的输出目录。

原始产物按 [实验规范](../../docs/EXPERIMENTS.md) 写入 outputs；核验后的成绩按 [结果规则](../../results/README.md) 登记到根 results。
