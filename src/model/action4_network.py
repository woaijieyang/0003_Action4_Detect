"""Shared ResNet-50 detector and eight-corner heatmap network.

Torchvision 0.23 provides the backbone architecture and optional ImageNet V2
weights. V2 denotes the weight recipe, not a pre-activation ResNet variant.
https://docs.pytorch.org/vision/0.23/models/generated/torchvision.models.resnet50.html

Constructing the class is explicit; importing this module never downloads or
constructs a model. Inputs are RGB floats normalized by ImageNet mean/std, with
no classification centre crop. Pixel sampling belongs to ``data``.
"""

from __future__ import annotations

import copy

import hashlib
import math
from collections import OrderedDict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


RoiBuilder = Callable[[Mapping[str, Tensor]], tuple[Tensor, Any]]


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    raise AssertionError("positive channels always have a divisor")


class Action4Net(nn.Module):
    """Two image scales call exactly the same backbone and upsampling decoder.

    Default full input is Bx3x768x1024; ROI input is Nx3x256x256. Other positive
    sizes divisible by 32 are accepted for explicit engineering diagnostics,
    and must be recorded as a different configuration from the formal run.

    ``weights='IMAGENET1K_V2'`` explicitly requests Torchvision's pretrained
    download/cache lookup. Checkpoint reconstruction uses ``weights=None``;
    the caller then strictly loads the complete Action4 state dictionary.

    An optional ``roi_builder`` allows detached detection selection/cropping
    inside one DDP forward. It receives detached dense tensors and returns
    ``(normalized_roi_images, metadata)``; metadata is returned unchanged. The
    network neither knows labels nor imports data/evaluation code.
    """

    def __init__(
        self,
        *,
        weights: str | None = None,
        freeze_bn: bool = True,
        decoder_channels: int = 256,
        head_channels: int = 64,
        corner_probability_bias: float = 0.5,
    ) -> None:
        super().__init__()
        if weights not in (None, "IMAGENET1K_V2"):
            raise ValueError("weights must be None or IMAGENET1K_V2")
        if (type(decoder_channels) is not int or decoder_channels < 1
                or type(head_channels) is not int or head_channels < 1):
            raise ValueError("decoder_channels and head_channels must be positive integers")
        if not isinstance(corner_probability_bias, (int, float)) or isinstance(corner_probability_bias, bool) or not 0 < corner_probability_bias < 1:
            raise ValueError("corner_probability_bias must lie strictly between zero and one")
        from torchvision.models import ResNet50_Weights, resnet50

        net = resnet50(weights=(ResNet50_Weights.IMAGENET1K_V2 if weights else None))
        self.backbone = nn.Sequential(OrderedDict(
            (name, module) for name, module in net.named_children()
            if name not in ("avgpool", "fc")
        ))
        layers: list[nn.Module] = []
        in_channels = 2048
        for _ in range(3):
            layers.extend([
                nn.ConvTranspose2d(in_channels, decoder_channels, 4, 2, 1, bias=False),
                nn.GroupNorm(_group_count(decoder_channels), decoder_channels),
                nn.ReLU(inplace=True),
            ])
            in_channels = decoder_channels
        self.decoder = nn.Sequential(*layers)

        def head(channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(decoder_channels, head_channels, 3, padding=1, bias=False),
                nn.GroupNorm(_group_count(head_channels), head_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_channels, channels, 1),
            )

        self.center_head = head(1)
        self.offset_head = head(2)
        self.box_head = head(4)
        self.corner_head = head(8)
        self.freeze_bn = bool(freeze_bn)
        self.configuration = {
            "architecture": "action4_shared_resnet50_v1",
            "backbone": "torchvision_resnet50",
            "weights_at_initialization": weights,
            "decoder_channels": decoder_channels,
            "head_channels": head_channels,
            "freeze_bn": self.freeze_bn,
            "output_stride": 4,
            "num_corners": 8,
            "new_head_initialization": {"final_weight_std": 0.001,
                                        "center_probability_bias": 0.01,
                                        "corner_probability_bias": float(corner_probability_bias),
                                        "normalized_ltrb_bias": 0.1},
            "input_normalization": {"mean": [0.485, 0.456, 0.406],
                                    "std": [0.229, 0.224, 0.225]},
        }
        # Only initialize new modules. Pretrained backbone parameters are kept.
        for container in (self.decoder, self.center_head, self.offset_head,
                          self.box_head, self.corner_head):
            for module in container.modules():
                if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        # Quiet initial dense maps; GN/hidden layers still receive gradients.
        for container in (self.center_head, self.offset_head, self.box_head, self.corner_head):
            nn.init.normal_(container[-1].weight, std=0.001)
        nn.init.constant_(self.center_head[-1].bias, math.log(0.01 / 0.99))
        nn.init.constant_(self.corner_head[-1].bias, math.log(corner_probability_bias / (1 - corner_probability_bias)))
        # A normalized distance starts near 0.1, not softplus(0) ~ 0.69.
        nn.init.constant_(self.box_head[-1].bias, math.log(math.expm1(0.1)))
        self.freeze_backbone_norm()

    def enable_independent_roi_features(self) -> None:
        """Create a frozen copy after A0 load; call before loading separated state.

        Idempotence is essential: resume must not replace the preserved branch
        with the adapted detector. Training/inference callers must explicitly
        bind this topology to checkpoint policy before loading its extra keys.
        """
        if hasattr(self, "roi_branch"):
            return
        if not self.freeze_bn:
            raise ValueError("Independent A0 ROI features require frozen BN statistics")
        self.roi_branch = nn.Sequential(OrderedDict(
            (name, copy.deepcopy(getattr(self, name)))
            for name in ("backbone", "decoder", "corner_head")
        ))
        self.roi_branch.requires_grad_(False)
        self.roi_branch.eval()
        self.corner_head.requires_grad_(False)
        self.configuration["roi_feature_policy"] = "independent_A0_roi"

    def freeze_backbone_norm(self) -> None:
        """Freeze running statistics only; affine parameters remain trainable."""
        if self.freeze_bn:
            for module in self.backbone.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()

    def train(self, mode: bool = True) -> "Action4Net":
        super().train(mode)
        self.freeze_backbone_norm()
        if hasattr(self, "roi_branch"):
            self.roi_branch.eval()
        return self

    def set_backbone_trainable(self, trainable: bool) -> None:
        """Explicit stage-level freeze/unfreeze, including BN affine weights."""
        self.backbone.requires_grad_(trainable)
        self.freeze_backbone_norm()

    def load_backbone_weights(self, path: str | Path, sha256: str | None = None) -> dict[str, Any]:
        """Strictly load a local Torchvision ResNet-50 classification checkpoint.

        Only the two classification-head tensors are removed. Every backbone
        key and shape must agree; nested/custom/partially matching checkpoints
        fail instead of leaving random layers without notice. The actual file
        digest is returned, even when the caller did not supply an expected one.
        This method performs no download and never changes the new heads.
        """
        source = Path(path).expanduser().resolve(strict=True)
        if not source.is_file():
            raise ValueError("Backbone checkpoint must be a regular file")
        digest = hashlib.sha256()
        with source.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        if sha256 is not None and actual != sha256.lower():
            raise ValueError("Backbone checkpoint SHA-256 does not match")
        state = torch.load(source, map_location="cpu", weights_only=True)
        expected = set(self.backbone.state_dict()) | {"fc.weight", "fc.bias"}
        if (not isinstance(state, Mapping) or set(state) != expected
                or not all(isinstance(value, Tensor) for value in state.values())):
            raise ValueError("Expected an unwrapped complete Torchvision ResNet-50 state dictionary")
        if state["fc.weight"].shape != (1000, 2048) or state["fc.bias"].shape != (1000,):
            raise ValueError("Unexpected ImageNet classification-head shape")
        backbone_state = {key: value for key, value in state.items() if not key.startswith("fc.")}
        self.backbone.load_state_dict(backbone_state, strict=True)
        identity = {"path": str(source), "sha256": actual, "bytes": source.stat().st_size,
                    "loaded": "all_backbone_parameters_and_buffers", "removed": ["fc.weight", "fc.bias"]}
        self.configuration["backbone_checkpoint"] = identity
        return identity

    @staticmethod
    def _validate_images(images: Tensor, *, allow_empty: bool = False) -> None:
        if images.ndim != 4 or images.shape[1] != 3 or not images.is_floating_point():
            raise ValueError("Images must be floating NCHW tensors with three RGB channels")
        if images.shape[0] < (0 if allow_empty else 1):
            raise ValueError("A full-image batch must be nonempty")
        height, width = images.shape[-2:]
        if height < 32 or width < 32 or height % 32 or width % 32:
            raise ValueError("Image height and width must be positive multiples of 32")

    def _features(self, images: Tensor) -> Tensor:
        return self.decoder(self.backbone(images))

    def _empty_corners(self, reference: Tensor, height: int, width: int) -> Tensor:
        # Keep a zero derivative to every otherwise-unused corner-head parameter.
        # A zero-ROI image remains valid in the complete-image loss denominator.
        anchor = sum(parameter.reshape(-1)[0] * 0 for parameter in self.corner_head.parameters())
        return reference.new_empty((0, 8, height, width)) + anchor

    def forward_rois(self, roi_images: Tensor) -> Tensor:
        """ROI-only diagnostics/A0; use forward(None, ...) under DDP."""
        self._validate_images(roi_images, allow_empty=True)
        if roi_images.shape[0] == 0:
            return self._empty_corners(roi_images, roi_images.shape[-2] // 4,
                                       roi_images.shape[-1] // 4)
        if hasattr(self, "roi_branch"):
            with torch.no_grad():
                return self.roi_branch(roi_images)
        return self.corner_head(self._features(roi_images))

    def forward(
        self,
        images: Tensor | None,
        roi_images: Tensor | None = None,
        *,
        roi_builder: RoiBuilder | None = None,
    ) -> dict[str, Any]:
        if images is None and roi_images is None:
            raise ValueError("Provide full images or ROI images")
        if roi_builder is not None and (images is None or roi_images is not None):
            raise ValueError("roi_builder requires images and is exclusive with roi_images")
        result: dict[str, Any] = {}
        if images is not None:
            self._validate_images(images)
            features = self._features(images)
            result.update({
                "center_logits": self.center_head(features),
                "offset": self.offset_head(features),
                "ltrb": F.softplus(self.box_head(features)),
            })
        if roi_builder is not None:
            roi_images, metadata = roi_builder({key: value.detach() for key, value in result.items()})
            if not isinstance(roi_images, Tensor):
                raise TypeError("roi_builder must return (ROI Tensor, metadata)")
            result["roi_metadata"] = metadata
        if roi_images is not None:
            result["corner_logits"] = self.forward_rois(roi_images)
        else:
            assert images is not None
            result["corner_logits"] = self._empty_corners(images, 64, 64)
        return result
