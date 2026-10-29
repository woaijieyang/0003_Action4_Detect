# DJI Action4 双机检测、八角点定位与视频跟踪

**2026-10-29 状态：用户暂停研究、训练与评估；本次仅整理和发布代码。真实全视频的角点准确性和稳定性尚未验收。**

项目复用固定尺寸 DJI Action4、canonical8 点序与已发布合成数据，采用 B0 完整机身检测 → 实例 ROI → A0 姿态/角点 → 几何与视频跟踪链路。数据、模型、环境和运行产物保留在服务器，普通 Git 只发布源码、配置、测试和文档。

## 当前可靠入口

- [执行与文件索引](docs/action4_execution_index.md)
- [第二阶段主计划](docs/stages/2026-10-24-1-plan-Action4第二阶段识别与八角点增强.md)与[进行中结论](docs/stages/2026-10-24-1-conclusion-Action4第二阶段识别与八角点增强.md)
- [结果登记](results/results.md)、[模型说明](models/README.md)、[输出索引](outputs/README.md)
- 当前暂停权威：configs/execution/action4_active.json → action4_execution_user_paused_r147.json。历史配置及 PID 不是重启授权。

## 模型与未完成工作

B0 已封存 step9500；阶段四 A0 的研究选择为 step7300、ROI1.5、corner_candidate，完整650图开发集 All8@2%=1116/1300。阶段四由用户提前停止，不能称完成原定80轮；quality_ready=false，研究试用与部署准入分开。

新 ROI1.15 的75轮适配在用户暂停时保留了完整 step400 状态，尚未完成训练与异步评估，也没有可替代1.5父模型的最终 best。评分入口和异步队列等待的修复有独立工程证据，不等于定位效果改善。models/default 未覆盖，仍保留历史默认组合；后续必须读取明确 selection，不能按文件修改时间挑权重。

已有2850帧 B0＋SAM、A0/PnP、跟踪和平滑开发视频；V5减少了诊断中的旋转跳动，但方向、厚度和遮挡问题尚存，不能把平滑当准确。六面/材料跟踪研究候选尚未完成精度验收。相机内参仍为未标定假设，demo只作参考。

## 目录职责

| 目录 | 内容 |
|---|---|
| src/data | 数据准入、预测框来源、ROI采样、K与标签变换 |
| src/model | A0/B0、SAM辅助、角点、几何与跟踪 |
| src/model/training | 优化器、有效监督分母、不可变运行、恢复和异步队列 |
| src/eval | 数值评分、推理编排、逐帧保存与只读渲染 |
| scripts / tests | 薄入口 / 工程验证 |
| configs | 无凭据的可追溯配置；旧配置不代表当前授权 |
| data/phrase1 | 历史2000张，1600/200/200 |
| data/phrase2 | 已发布6500张，5200/650/650，未重新划分 |
| models | 默认、封存候选、历史、外部预训练模型 |
| outputs/videos | 按时间与阶段命名的视频 |
| outputs | 需要保留的预测、缓存、回执和失败证据 |
| archive/legacy | 历史说明及整理前文档快照 |
| runtime / upstream | 项目内环境、不可变运行源码、外部实现 |

服务器项目根：/data/zy/vln/a-test。每条命令先进入并验证真实路径。仅物理GPU4–7及CPU24–95（最多72逻辑核）曾获授权；**当前已暂停，资源额度不构成自动恢复授权**。未修改正式数据和并行数据生成任务的资产。

## 复用与验证边界

仓库克隆不包含权重、6500张图像、SAM缓存和原视频。先按阶段文档核对数据/模型/依赖身份；不可通过旧控制器直接启动。训练、工程检查、开发成绩和真实验收分别报告。暂停后的恢复必须从记录的完整模型、optimizer、scheduler、RNG及数据位置继续，不能混接旧优化器。

保留 Apache-2.0 [LICENSE](LICENSE)。GDR-Net等复用部分遵守各自许可证及 provenance；第三方数据和权重授权不由本仓库许可证替代。
