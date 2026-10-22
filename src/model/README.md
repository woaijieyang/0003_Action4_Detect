# model：模型与算法

存放正式模型定义、前向计算、算法组件和可复用模块。根目录 `models/` 保存权重，本目录只保存代码与必要说明。

## Action 4 网络与接口

`action4_network.Action4Net` 使用 Torchvision ResNet-50（移除池化和分类器）及共享的三层 `ConvTranspose2d(4,2,1)`、GroupNorm、ReLU 解码器，默认宽度 256。全图和原始 RGB ROI 分别调用同一份主干与解码器；默认全图 `1024×768`、ROI `256×256`，对应 stride-4 输出。输入必须已经按 ImageNet mean/std 归一化，无分类中心裁剪。

- `Action4Net(weights=None, freeze_bn=True, decoder_channels=256, head_channels=64)`：默认不下载权重。显式传 `weights="IMAGENET1K_V2"` 才请求官方缓存/下载；V2 是权重版本。
- `load_backbone_weights(path, sha256=None)`：仅从本地加载完整 Torchvision ResNet-50 state dict，核验实际 SHA-256，只去除 `fc.weight/fc.bias`，严格加载全部空间主干参数和 buffer。
- `forward(images, roi_images=None, roi_builder=None)`：返回 `center_logits/offset/ltrb/corner_logits`；`ltrb` 是按全图宽高归一化的非负距离，offset 可以为负。
- `roi_builder(detached_dense_outputs) -> (roi_images, metadata)`：可在同一次 DDP forward 中完成离散预测框选择和原图裁剪，输出中的 `roi_metadata` 原样返回。只允许与 `roi_images` 二选一。
- `forward(None, roi_images)`：A0/D0 的 ROI-only 路径；普通单机也可用 `forward_rois`。
- `set_backbone_trainable(bool)`：阶段级控制主干梯度。`train()` 自动重新冻结主干 BN running stats；BN affine 是否训练随主干参数开关变化。

`action4_losses.detection_losses` 接收 `center[B,1,H,W]、indices[B,M]、instance_mask[B,M]、offset[B,M,2]、ltrb[B,M,4]` 和可选 `negative_mask[B,1,H,W]`，输出标量 `center/offset/box`。Focal 的对数在 FP32 中稳定计算，正样本优先于 ignore；有效锚点必须有标签峰 1，同一格冲突直接报错。完整图像先独立归一化，再求 batch mean，空目标图保留背景监督。

`corner_heatmap_loss(logits,target,valid,roi_batch_index,batch_size)` 使用峰区/背景各自的 SmoothL1 mean，再按有效角点、ROI、完整图像平均。`valid` 代表投影在 Dataset 固定的热图中心域（64×64 时为 `[0,63]²`）内且有合法监督，不是可见性；该有效域与画布像素边界 `[-0.5,63.5]²` 分别记录。没有 ROI 时返回与网络相连的零；隐藏的虚拟 cuboid 角点不会因不可见而被移除。`joint_loss` 的四项默认权重均为 1。

`action4_decode.decode_detections(outputs,image_size=(W,H),heatmap_to_image=None,...)` 从 RGB 模型输出产生 0/1/2 个实例。默认局部 NMS、top20、阈值 0.3，只有同时满足 IoU≥0.95 与中心距离≤较小框对角线 5% 才去重。完整框保留出画部分；返回 tensor 坐标，由调用方序列化。可选矩阵从热图像素中心映射到原图。

`decode_corners(logits,heatmap_to_image=None,topk=20,from_logits=True)` 按固定八通道解码最高 20 个像素位置均值。常量热图不代表可靠定位，边缘峰存在向内偏差；正式 ROI 应保留完整框的外扩边界，并通过实际标签的理想热图解码检查。不能把该组件的数值正确性当成真实视频精度。

## 代码来源与验证范围

主干直接调用 [Torchvision 0.23 ResNet-50](https://docs.pytorch.org/vision/0.23/models/generated/torchvision.models.resnet50.html)。Focal、局部 NMS、top-k 和 gather 参考 [CenterNet 固定提交](https://github.com/xingyizhou/CenterNet/tree/4c50fd3a46bdf63dbf2082c5cbb3458d39579e6c)，作独立实现，修改点为 logits/FP32、逐图归一化、可见中心＋四边距和严格重复框处理。角点 top20 数学参考 [BoxDreamer 固定提交](https://github.com/zju3dv/BoxDreamer/tree/106f3c16c746770cbad80720ce4a557eb31f30f5) 的 `recover_bb8_corners`，无引用图片、Transformer 或完整 checkpoint；源码模块保存出处，未复制其训练禁止梯度的 ResNet 包装。许可证原文见 `THIRD_PARTY_NOTICES.md` 的链接及 `upstream/` 固定源码快照。

CPU 工程检查：`PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p test_action4_model.py -v`。测试用真实 ResNet-50、64 通道解码器及小输入执行共享前向、ROI 梯度和参数更新、BN 统计冻结、空 ROI、极端 focal、逐图 loss、坐标/重复框及严格权重校验；这不属于正式 256 通道模型的 D0 收敛、训练或测评。

## 训练与测评

训练循环、损失、优化器组合和训练策略按需放入 training 子包，启动脚本位于 scripts。training 可以调用 data 和 model。标签按训练协议使用，不因测评结果登记而获得把测试数据用于训练的资格。

测评时的 checkpoint 选择、设备放置和实例化编排位于 eval。模型自己的结构、状态加载接口与可学习模块仍归 model，不把模型算法复制到 eval。
模型主体不依赖测评运行器，不读取测评答案。通用清洗归 data，方法特有的可学习编码归这里。

导入只定义模块；昂贵初始化、训练和加载由调用者明确触发。模型配置、checkpoint 来源、版本及必要哈希按 [实验规范](../../docs/EXPERIMENTS.md) 记录。
