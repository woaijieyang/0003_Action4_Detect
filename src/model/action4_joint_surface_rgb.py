"""Trainable surface+pose A0; optional SAM C5 residual, disabled by default.

The opt-in module reuses the SAME backbone invocation and existing GDRPose head.
Enhanced bundles and parameter groups are handled by release_bundle/stage2_core;
StagedPipeline has a separate typed SAM input route. Corner features are opt-in.
"""
import torch
from torch import nn
from model.action4_release_bundle import load_bundle


class JointSurfaceRGB(nn.Module):
    def __init__(self, parent_bundle, train_backbone=False, device="cpu", *, enable_sam=False, enable_corner=False, parent_bundle_sha256=None):
        super().__init__()
        if enable_corner and parent_bundle_sha256 is None:raise ValueError("Corner parent requires verified source SHA")
        self.head, self.base, self.parent = load_bundle(parent_bundle, "A0", device,
            allow_corner_candidate=enable_corner,expected_sha256=parent_bundle_sha256)
        from model.action4_sam_joint import SAM_ARCHITECTURE, CORNER_ARCHITECTURE, SamGeometryHead
        if type(enable_corner) is not bool or (enable_corner and not enable_sam):raise ValueError("Corner requires explicit SAM opt-in")
        corner_parent=self.parent["architecture"]==CORNER_ARCHITECTURE
        restored_corner=None
        if enable_corner and self.parent["architecture"] not in (SAM_ARCHITECTURE,CORNER_ARCHITECTURE):
            raise ValueError("Corner initializes only from complete SAM/corner parent")
        enhanced_parent = self.parent["architecture"] in (SAM_ARCHITECTURE,CORNER_ARCHITECTURE)
        if self.parent["architecture"] not in ("joint_pose", SAM_ARCHITECTURE, CORNER_ARCHITECTURE):
            raise ValueError("Joint surface/pose parent required")
        if enhanced_parent and not enable_sam:
            raise ValueError("Enhanced parent requires enable_sam=True; refusing to discard SAM parameters")
        restored_adapter = None
        if corner_parent:
            loaded=self.head
            restored_adapter=loaded.sam_adapter;restored_corner=loaded.corner_head
            self.head=loaded.head
        elif enhanced_parent:
            if not isinstance(self.head, SamGeometryHead):
                raise ValueError("Enhanced bundle loader contract mismatch")
            restored_adapter = self.head.sam_adapter
            self.head = self.head.geometry
        if type(enable_sam) is not bool:
            raise ValueError("enable_sam must be an explicit bool")
        self.train_backbone = bool(train_backbone)
        self.enable_sam = enable_sam
        self.base.requires_grad_(False)
        self.head.requires_grad_(True)
        if self.train_backbone:
            self.base.backbone.requires_grad_(True)
        # None contributes no state_dict keys or parameters: existing callers
        # retain the original module/state contract when the opt-in is absent.
        self.sam_adapter = None
        if enable_sam:
            if train_backbone:
                raise ValueError("SAM image protocol requires frozen RGB backbone")
            from model.action4_sam_feature_adapter import SamFeatureAdapter
            self.sam_adapter = restored_adapter if restored_adapter is not None else SamFeatureAdapter().to(device)
            self.sam_adapter.requires_grad_(True)
        self.enable_corner=enable_corner
        self.corner_head=None
        if enable_corner:
            from model.action4_corner_supervision import CornerObservationHead
            self.corner_head=restored_corner if restored_corner is not None else CornerObservationHead().to(device)
            self.corner_head.requires_grad_(True)

    @classmethod
    def from_corner_components(cls,geometry,base,state,adapter,corner):
        """Strict loader assembly: no parent file reads or missing-key initialization."""
        model=cls.__new__(cls);nn.Module.__init__(model)
        model.head=geometry;model.base=base;model.parent=state
        model.train_backbone=False;model.enable_sam=True;model.enable_corner=True
        model.sam_adapter=adapter;model.corner_head=corner
        model._corner_config_identity=state["config_id"]
        model.base.requires_grad_(False)
        return model

    def train(self, mode=True):
        super().train(mode)
        # Preserve parent backbone BN policy; unused legacy heads never train.
        self.base.train(mode if self.train_backbone else False)
        return self

    def forward(self, rgb, K, canonical, *, mask_aux=None, identities=None, return_aux=False,
                return_corner_features=False, return_corner=False):
        """Return existing geometry output, or (output, SamFeatureOutput | None).

        Enabled SAM requires one explicit FeatureIdentity per RGB/K/CAD instance,
        including missing-mask samples. MaskAux carries the same ordered identity.
        Caller binds actual ROI affine/K/cache provenance; identity strings alone
        cannot prove those sources. With no mask, enhanced C5 equals RGB C5.
        The returned auxiliary object is accepted by mask_quality_loss and exposes
        enhanced C5 for later FPN reuse; no second backbone or pose pass is run.
        return_corner_features=True additionally returns typed C2/C3/C4/enhanced-C5
        as a third value; it requires return_aux=True and adds no model parameters.
        """
        from model.action4_geometry_forward import instancewise_geometry
        if type(return_corner) is not bool or return_corner!=self.enable_corner:
            raise ValueError("Corner candidate requires explicit return_corner=True; baseline consumer refused")
        if self.enable_corner and return_corner_features:raise ValueError("Choose owned corner output, not feature-only output")
        if not self.enable_sam and mask_aux is not None:
            raise ValueError("mask_aux requires explicit enable_sam=True")
        # Exactly one backbone call, including enabled/missing-mask and return_aux.
        if type(return_corner_features) is not bool:
            raise ValueError("return_corner_features must be boolean")
        from contextlib import nullcontext
        with nullcontext() if self.train_backbone else torch.no_grad():
            if return_corner_features or self.enable_corner:
                if not self.enable_sam or (return_corner_features and not return_aux):
                    raise ValueError("Corner feature export requires SAM opt-in and return_aux=True")
                from model.action4_corner_observation_head import backbone_stages_once
                stages = backbone_stages_once(self.base.backbone, rgb)
                features = stages["c5"]
            else:
                features = self.base.backbone(rgb)
        if self.enable_sam:
            from model.action4_sam_joint import sam_geometry_forward
            result = sam_geometry_forward(self.head, self.sam_adapter, features, rgb, K, canonical,
                                          mask_aux=mask_aux, identities=identities, return_aux=return_aux or self.enable_corner)
            if return_corner_features or self.enable_corner:
                from model.action4_corner_observation_head import SamCornerFeatures
                output, auxiliary = result
                corner_features=SamCornerFeatures(stages["c2"], stages["c3"], stages["c4"],auxiliary.c5, auxiliary.identities)
                if self.enable_corner:
                    from model.action4_sam_joint import CornerCandidateOutput
                    corners=self.corner_head(corner_features,identities=auxiliary.identities,
                        sam_available=auxiliary.available.reshape(len(rgb)),mask_quality=auxiliary.q_mask.reshape(len(rgb)))
                    return CornerCandidateOutput(output,auxiliary,corners)
                return output,auxiliary,corner_features
            return result
        auxiliary = None
        output = instancewise_geometry(self.head, features, K, canonical)
        return (output, auxiliary) if return_aux else output
