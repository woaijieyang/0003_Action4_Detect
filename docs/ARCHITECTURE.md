# Action4 当前架构与数据流

当前可复现实验由两个独立阶段构成。模型封存与代码发布不表示真实验收通过；验收状态见阶段 conclusion。

## 推理链

1. 原始 RGB 按保持长宽比的 letterbox 转为 1024×768；变换矩阵供框还原使用。
2. B0 的 ResNet 特征进入中心、offset、LTRB、定位质量头，形成候选框并还原到原图坐标。
3. 对候选框在同一张特征图取 9 点双线性特征，结合框几何与已有分数，由独立 refiner 输出修正框和定位质量分数。
4. Soft-NMS 排序抑制后，使用冻结阈值 0.3；训练和推理复用同一候选构造逻辑。
5. B0 预测框扩大 2.25 倍形成 448×448 ROI。A0 冻结 backbone 提取 RGB 特征；XYZ、区域、visible/amodal mask 进入 GDR 派生姿态头。
6. A0网络根据预测XYZ、相机射线、区域和visible/amodal mask估计R/t，再将固定尺寸canonical8投影。保留预测表面PnP作为合成集基线，二者分别报告；不把红框硬限制在绿框内。失效时标记pose unavailable，不用GT补齐。
7. 帧内 ID 是检测实例序号，角点 ID 是固定物体坐标系的 0–7；没有学习跨帧左右身份，不把序号当作物理左右相机标识。

所有 GT 框、角点、可见性和位姿仅进入训练损失或评估计分；完整图像推理不接受这些答案。

## 训练信号

B0 保留中心 focal、offset、框回归，在完整机身框上计算 GIoU；定位质量使用停止梯度的 IoU 标签。refiner 的框回归/GIoU/质量损失通过采样特征回传到 backbone。离散候选选择、框采样坐标及已有分数先验停止梯度，避免错误地声称对离散排序可导。

A0 表面头用可见表面 XYZ、区域与双掩码监督；姿态头使用 rotation-only point matching、crop centroid、相对深度和 fixed-ID 八角点投影误差。可见与隐藏点各按 optimizer batch 的有效点数归一化，微批次不再次缩放这两项。隐藏点仍有数值监督；不将不可见标签解释为忽略点。

最新A0在线RGB试验冻结backbone和surface，真实读取训练RGB并作中心±15%、尺度.65–1.35的ROI jitter，仅训练pose。更早的固定GT ROI缓存是历史对照。网络在真实视频包络上较稳、PnP在合成固定ID指标更强，无全面胜出的模型。

评估新契约：按部署的GPU裁剪和冻结特征前向生成预测maps，保留每张图原有0/1/2目标batch形状；不把各图ROI重新拼成32目标batch。inputs仅maps/K/canonical，原图逆仿射与GT标签放在计分侧。每25optimizer step评估的队列语义保持，使用image_grouped_pose_eval；旧在线训练调度器是冻结历史实现，第二阶段调度配置必须显式选新worker并核对新的dataset/feature身份。B0每20step及框指标协议保持。

## 实现位置

| 职责 | 模块 |
|---|---|
| 数据及 ROI 仿射 | `data.action4_dataset`、`action4_component_data` |
| B0 主体/候选 | `model.action4_online_b0`、`action4_proposal_pool` |
| B0 损失和训练 | `model.training.action4_online_b0_batch/train` |
| A0 主体/损失 | `model.action4_gdr_pose`、`action4_gdr_pose_losses_v002` |
| A0 训练 | `model.training.action4_online_pose_train`（已完成的在线RGB基线） |
| 异步选优 | `eval.action4_online_b0_eval`、`eval.action4_image_grouped_pose_eval`、`model.training.action4_async_recovery` |
| 权重包及 best 热启动 | `model.action4_release_bundle` |
| 独立包组合推理 | `eval.action4_staged_pipeline` |
| 完整 eval/test | `eval.action4_staged_eval`、`action4_bbox_metrics` |
| 完整视频 | `eval.action4_staged_video`、`action4_staged_video_parallel` |
| 真实候选边界诊断 | `eval.action4_release_proposal_audit` |

## 指标与失败边界

框指标以完整机身框 IoU≥0.5 一对一匹配，mAP 单独按 IoU 0.50:0.95 计分；角点在框匹配后以固定 ID、全部 8 点误差≤GT 框对角线 5% 为成功，漏检和无效位姿都计失败。高遮挡单独汇报。test 全双机，空图误检率无法由它估计；开发集的空图只有 6 张。

最新真实边界诊断发现：第 1000、1600、2700 帧的正确相机候选仍存在，但其质量分数排序落后于手/衣物/背景；第 2200 帧一台相机在候选池阶段就没有 IoU≥0.5 的框，refiner 进一步偏移。前者是排序/目标身份泛化问题，后者还存在候选定位问题，不能单靠改变 NMS 一个参数解决。此结论基于 demo 参考框，未形成真实认证 GT。

## 保留与归档

原始数据、固定 DJI 实体尺寸、canonical8、已发布 split 与冻结场景不因模型失败被改写。历史失败保留在固定 archive/legacy 并有迁移清单。当前所需训练特征缓存按依赖保留；无消费者的可再生成缓存和非最佳 checkpoint 才能删除。旧配置和报告是历史证据，不是当前启动入口。
