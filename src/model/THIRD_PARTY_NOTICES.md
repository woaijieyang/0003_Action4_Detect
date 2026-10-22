# Action 4 model references

The Action 4 modules independently implement the mathematical operations and architecture adaptations described in the execution plan. They do not bundle upstream source trees or third-party pretrained weights. Keep the fixed upstream lock and the original licence files when redistributing the accompanying upstream snapshots.

| Component | Fixed source and licence | Use and modification |
|---|---|---|
| Spatial ResNet-50 backbone | Torchvision `0.23.0`, [source](https://github.com/pytorch/vision/blob/v0.23.0/torchvision/models/resnet.py), [BSD-3-Clause licence](https://github.com/pytorch/vision/blob/v0.23.0/LICENSE) | Imported public model; remove avgpool/fc; retain explicit optional ImageNet V2 weight loading; freeze running BN statistics during task training |
| Focal, spatial NMS, top-k, gather | CenterNet `4c50fd3a46bdf63dbf2082c5cbb3458d39579e6c`, `src/lib/models/{losses,decode,utils}.py`; [MIT licence](https://github.com/xingyizhou/CenterNet/blob/4c50fd3a46bdf63dbf2082c5cbb3458d39579e6c/LICENSE), copyright 2019 Xingyi Zhou | Mathematical reference; independently written stable logits and per-image losses, four normalized box distances, strict duplicate handling. CenterNet attributes focal to CornerNet, copyright 2018 University of Michigan, BSD-3-Clause. |
| Fixed-corner top20 position mean | BoxDreamer `106f3c16c746770cbad80720ce4a557eb31f30f5`, `src/models/utils/box_utils.py::recover_bb8_corners`; [Project Registration License v1.0](https://github.com/zju3dv/BoxDreamer/blob/106f3c16c746770cbad80720ce4a557eb31f30f5/LICENSE), copyright 2026 ZJU3DV Group at Zhejiang University | Mathematical reference; independently written NCHW decoder, one coordinate domain and optional explicit affine; no upstream network, feature wrapper or checkpoint |

ImageNet pretrained weight files are downloaded only by an explicit initialization request or external setup command. Their real source URL, file hash and bytes belong to each experiment's dependency/initialization manifest; importing a Python module does not download them.


### Independent component trial v001 references (2026-10-22)

- Torchvision ResNet50 is directly reused through the installed Torchvision dependency.
- GIoU reference: https://docs.pytorch.org/vision/main/generated/torchvision.ops.generalized_box_iou_loss.html ; current frozen experiment independently expresses the formula and passes value/gradient agreement tests against the installed official operator. No official detector weights were imported.
- QFL mathematical reference: https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/losses/gfocal_loss.py (OpenMMLab). Current implementation uses a separate IoU-quality branch and project candidate sampling, not the paper's complete joint classification/distributional localization pipeline. It is not a full GFL reproduction. No source code from that file was vendored in this trial.
- A0 relation head is an unproven project candidate, not a claimed reproduction of PVNet or a proven hidden-corner solution. Actual negative development results must remain visible.
