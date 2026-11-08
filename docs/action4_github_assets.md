# GitHub 视频、权重与复现资产

2026-11-08，Asia/Singapore。项目保持暂停；本次只打包、校验和上传既有产物，没有启动训练/推理或重新生成视频。

## 下载

Release：[Action4 research 2026-11-08](https://github.com/woaijieyang/0003_Action4_Detect/releases/tag/action4-research-20261108)。

| 文件 | 用途 |
|---|---|
| [stage5-v5-stabilized-2850.mp4](https://github.com/woaijieyang/0003_Action4_Detect/releases/download/action4-research-20261108/stage5-v5-stabilized-2850.mp4) | V5完整2850帧，B0+SAM ROI、A0/PnP、跟踪和平滑 |
| [stage5-baseline-2850.mp4](https://github.com/woaijieyang/0003_Action4_Detect/releases/download/action4-research-20261108/stage5-baseline-2850.mp4) | 同A07300/B09500的原始视频对照 |
| [A0-stage4-step7300-roi1p5.pt](https://github.com/woaijieyang/0003_Action4_Detect/releases/download/action4-research-20261108/A0-stage4-step7300-roi1p5.pt) | 当前封存A0研究候选，ROI1.5 |
| [B0-stage1-step9500.pt](https://github.com/woaijieyang/0003_Action4_Detect/releases/download/action4-research-20261108/B0-stage1-step9500.pt) | 固定B0完整机身检测器 |
| [reproduction-snapshots.tar.gz](https://github.com/woaijieyang/0003_Action4_Detect/releases/download/action4-research-20261108/reproduction-snapshots.tar.gz) | 实际视频/后处理/训练的源码快照与关键配置及native候选源码 |

另有 [SHA256SUMS.txt](https://github.com/woaijieyang/0003_Action4_Detect/releases/download/action4-research-20261108/SHA256SUMS.txt) 和 [RELEASE_MANIFEST.json](https://github.com/woaijieyang/0003_Action4_Detect/releases/download/action4-research-20261108/RELEASE_MANIFEST.json)，下载后按SHA核验；模型文件只加载可信来源。

## 代码在哪里

仓库src/data、src/model、src/model/training、src/eval、scripts、configs、tests包含当前正式实现。不要只复制某个outputs中的启动脚本。
Release源码包保留服务器原相对目录：
- outputs/action4_stage5_sealed7300_video_v001/source：实际整片推理controller/worker及源码。
- outputs/action4_stage5_coherent_preparation_v006/source：实际V5后处理源码。
- runtime/action4_runs/p2_a0_corner_prediction15_formal80_v003/source：对应训练源码与resolved_config。
- runtime/m3t_direct_mask_src_v001、runtime/pym3t_direct_mask_src_v001：native研究候选改动及许可证；未默认为已部署效果。
- 关键CONFIG、CANDIDATE、INPUT_LOCK、模型选择/封存与视频回执。

代码包共4539个文件。包括所需Python源码、配置、C++/头文件、CMake及现有许可证，不包含编译二进制、环境、权重输入目录或大缓存。
历史SOURCE_MANIFEST描述当时完整运行目录，可能包含包外二进制；**代码包不是完整环境备份**，不能解压后就声称通过原完整清单。重新运行须按现有机制重新冻结并核对实际依赖，不能关闭身份校验或复写旧清单。
包内原第三方许可证保留；它们不被本项目许可证取代。

## 使用顺序与仍需服务器资产

1. 获取本Release所绑定的仓库提交；主入口详见[暂停交接](action4_paused_handoff_20261029.md)。
2. 下载两份权重和代码包，核验SHA。权重恢复到交接文档中指定的models/candidates路径；B0副本路径也受loader绑定，不能只随意改名。
3. 源码包按项目根相对结构解压，先查看归属，禁止覆盖已有运行源码。历史路径硬编码/data/zy/vln/a-test，迁移需显式适配并建立新身份。
4. 外部SAM3.1、CoTracker、native依赖按各自许可获取/安装；SAM gated权重没有转发。本仓库pyproject及各source/config提供环境线索，但未提供经新机器验证的一键环境。
5. 要重新计算原视频，还需原始RGB/2850帧、PTS清单、CAD、冻结B0候选、SAM轨迹与辅助缓存、INPUT_LOCK引用的源资产。它们仍在服务器，**本Release不包含完整6500数据与这些大型缓存**。缺失时不能用伪缓存或GT框替代。
6. 同身份视频直接下载观看；新实验使用新的配置/输出和不可变源码，原完成controller不可原地重跑。图片eval关闭track，视频按实际resolved ROI进入A0。

## 成绩边界

A07300的开发All8@2%=1116/1300；它是研究候选，quality_ready=false，原80轮被用户提前停止。V5已交付但真实方向/厚度、六面材料绑定、相机标定和完整视频准确性仍未验收。上传视频不表示效果已经达标；demo只为参考。
ROI1.15仅有step400恢复状态，没有最终best，因此不发布为新赢家。不要把本Release模型替换成last。

本次仅验证文件SHA、源码语法与包路径安全，没有重新推理或在全新机器完成端到端复现。后续用户明确恢复后再按原阶段计划继续。
