"""Opt-in Stage4 corner outputs and local sufficient statistics, not pose admission.

Inputs to forward are prediction features only. Targets are a separate loss-only
packet. Counts are local corner counts: callers globally reduce them before
weighted_objective; no sample filtering or independent training loop is supplied.
Epochs here are ONE-BASED. quality_ready is deliberately not an output.
"""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from model.action4_corner_observation_head import CornerLocalizationHead, CornerLocalizationOutput, local_corner_uv
from model.action4_sam_feature_adapter import _identities

LOSS_WEIGHTS = {"corner_heatmap_kl":1., "corner_coordinate":1., "corner_laplace_nll":.1,
                "corner_valid_bce":.1, "corner_quality_bce":.2}


def effective_weights(epoch):
    if type(epoch) is not int or epoch < 1:
        raise ValueError("Explicit one-based epoch required")
    return dict(LOSS_WEIGHTS, corner_quality_bce=0. if epoch <= 5 else .2)


def parameter_loss_dependencies(prefix=""):
    """Independent owners; caller also lists these losses for shared SAM adapter."""
    if prefix and not prefix.endswith("."): prefix += "."
    return [
        {"parameter_prefixes":[prefix+"localization."+k for k in ("p5","lateral4","lateral3","lateral2","localization")],
         "losses":list(LOSS_WEIGHTS)[:-1]},
        {"parameter_prefixes":[prefix+"localization.heatmap"],"losses":list(LOSS_WEIGHTS)[:3]},
        {"parameter_prefixes":[prefix+"valid_head"],"losses":["corner_valid_bce"]},
        {"parameter_prefixes":[prefix+"scale_head"],"losses":["corner_laplace_nll"]},
        {"parameter_prefixes":[prefix+"quality_head"],"losses":["corner_quality_bce"]},
    ]


@dataclass(frozen=True)
class CornerObservationOutput:
    localization: CornerLocalizationOutput
    uv224: torch.Tensor
    valid_logits: torch.Tensor
    scale_b224: torch.Tensor
    quality_logits: torch.Tensor

    @property
    def identities(self): return self.localization.identities

    @property
    def p_valid(self): return self.valid_logits.sigmoid()

    @property
    def q_corner(self): return self.quality_logits.sigmoid()


class CornerObservationHead(nn.Module):
    """Existing 128-channel localization FPN, global valid/scale, detached quality.

    Quality input is local128 + global128 + peak/entropy/valid/b/SAMavailable/qmask.
    MLP dimensions are 262->128->64->1. It never gates localization or raw pose.
    """
    def __init__(self):
        super().__init__()
        self.localization=CornerLocalizationHead()
        self.valid_head=nn.Linear(128,8)
        self.scale_head=nn.Linear(128,8)
        self.quality_head=nn.Sequential(nn.Linear(262,128),nn.ReLU(),
                                       nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))

    def forward(self,features,*,identities,sam_available,mask_quality):
        heat,spatial=self.localization(features,identities=identities,return_features=True)
        n=len(spatial)
        for name,value in (("sam_available",sam_available),("mask_quality",mask_quality)):
            if (not isinstance(value,torch.Tensor) or value.shape!=(n,) or
                value.device!=spatial.device or value.dtype!=spatial.dtype or
                not torch.isfinite(value).all() or ((value<0)|(value>1)).any()):
                raise ValueError(name+": finite same-device/dtype probabilities [B] required")
        if not ((sam_available==0)|(sam_available==1)).all():
            raise ValueError("SAM availability must be binary")
        global_feature=spatial.mean((-2,-1))
        valid=self.valid_head(global_feature)
        b=(F.softplus(self.scale_head(global_feature))+.5).clamp(.5,20.)
        uv=local_corner_uv(heat)
        # All quality inputs are detached, including pooled feature statistics.
        flat=heat.logits.detach().flatten(2)
        probabilities=flat.softmax(-1)
        peak,indices=probabilities.max(-1)
        entropy=-(probabilities*flat.log_softmax(-1)).sum(-1)
        local=spatial.detach().flatten(2).gather(2,indices[:,None].expand(-1,128,-1)).transpose(1,2)
        stats=torch.stack((peak,entropy,valid.detach().sigmoid(),b.detach(),
                           sam_available.detach()[:,None].expand(-1,8),
                           mask_quality.detach()[:,None].expand(-1,8)),-1)
        qfeatures=torch.cat((local,global_feature.detach()[:,None].expand(-1,8,-1),stats),-1)
        quality=self.quality_head(qfeatures).squeeze(-1)
        output=CornerObservationOutput(heat,uv,valid,b,quality)
        validate_output(output)
        return output


def validate_output(output):
    if not isinstance(output,CornerObservationOutput):raise ValueError("Typed corner output required")
    heat=output.localization
    if not isinstance(heat,CornerLocalizationOutput) or heat.grid!=224 or heat.logits.ndim!=4 or heat.logits.shape[1:]!=(8,224,224):
        raise ValueError("Fixed8 corner224 logits required")
    n=len(heat.logits);_identities(output.identities,n)
    for value,shape in ((heat.logits,(n,8,224,224)),(output.uv224,(n,8,2)),
                        (output.valid_logits,(n,8)),(output.scale_b224,(n,8)),(output.quality_logits,(n,8))):
        if (not isinstance(value,torch.Tensor) or value.shape!=shape or not value.is_floating_point() or
            value.device!=heat.logits.device or value.dtype!=heat.logits.dtype or not torch.isfinite(value).all()):
            raise ValueError("Nonfinite or incompatible corner output")
    if ((output.scale_b224<.5)|(output.scale_b224>20)).any():raise ValueError("Scale outside .5..20 corner pixels")


@dataclass(frozen=True)
class CornerTargets:
    uv224: torch.Tensor
    diagonal224: torch.Tensor
    known: torch.Tensor
    valid: torch.Tensor
    identities: tuple


def _bool(value,shape,device,name):
    if not isinstance(value,torch.Tensor) or value.shape!=shape or value.dtype!=torch.bool or value.device!=device:
        raise ValueError(name+": explicit same-device boolean tensor required")
    return value


def build_corner_targets(*,uv_native,depth,A224_from_native,gt_diagonal224,
                         gt_known,ignore,identity_known,identity_correct,image_evidence,
                         active,identities):
    """GT-only builder. Hidden points remain legal; no visibility mask is accepted.

    known-negative: wrong known instance, known no-image-evidence, nonpositive
    depth, or out of grid. Unknown identity/GT/ignore are skipped. Nonfinite GT
    projection/diagonal is unmeasured, never multiplied by zero after arithmetic.
    Caller supplies full-body GT diagonal in the SAME224 grid (not visible box).
    """
    if not isinstance(uv_native,torch.Tensor) or uv_native.ndim!=3 or uv_native.shape[1:]!=(8,2) or not uv_native.is_floating_point():
        raise ValueError("GT native fixed8 coordinates required")
    n=len(uv_native);dev=uv_native.device;dtype=uv_native.dtype
    ids=_identities(identities,n)
    for value,shape in ((depth,(n,8)),(A224_from_native,(n,3,3)),(gt_diagonal224,(n,))):
        if not isinstance(value,torch.Tensor) or value.shape!=shape or value.device!=dev or value.dtype!=dtype:
            raise ValueError("GT geometry shapes/device/dtype mismatch")
    for name,value,shape in (("gt_known",gt_known,(n,8)),("ignore",ignore,(n,8)),
         ("identity_known",identity_known,(n,)),("identity_correct",identity_correct,(n,)),
         ("image_evidence",image_evidence,(n,)),("active",active,(n,))):
        _bool(value,shape,dev,name)
    A=A224_from_native.detach()
    if (not torch.isfinite(A).all() or not torch.equal(A[:,2],A.new_tensor([0,0,1]).expand(n,-1))
        or (torch.linalg.det(A[:,:2,:2]).abs()<1e-12).any()):
        raise ValueError("Finite invertible corner224 affine required")
    native=uv_native.detach();z=depth.detach();diag=gt_diagonal224.detach()
    finite=torch.isfinite(native).all(-1)&torch.isfinite(z)
    safe=torch.where(torch.isfinite(native),native,torch.zeros_like(native))
    mapped=torch.cat((safe,torch.ones_like(safe[...,:1])),-1)@A.transpose(1,2)
    transformed_finite=torch.isfinite(mapped[...,:2]).all(-1)
    diag_ok=torch.isfinite(diag)&(diag>0)
    declared_negative=(~identity_correct)|(~image_evidence)
    known=gt_known&~ignore&identity_known[:,None]&active[:,None]
    known=known&(declared_negative[:,None]|(finite&transformed_finite&diag_ok[:,None]))
    uv=torch.where(torch.isfinite(mapped[...,:2]),mapped[...,:2],torch.zeros_like(mapped[...,:2]))
    in_grid=((uv>=0)&(uv<=223)).all(-1)
    valid=known&~declared_negative[:,None]&finite&transformed_finite&diag_ok[:,None]&(z>0)&in_grid
    # Unsupervised coordinates are inert placeholders, not clipped observations.
    uv=torch.where(valid[...,None],uv,torch.zeros_like(uv))
    return CornerTargets(uv,torch.where(diag_ok,diag,torch.ones_like(diag)),known,valid,ids)


@dataclass(frozen=True)
class CornerLossStatistics:
    sums: dict
    counts: dict
    weights: dict
    quality_target: torch.Tensor
    quality_eligible: torch.Tensor


def corner_loss_sums(output,targets,*,epoch):
    """Unweighted LOCAL sums + integer eligible counts; no local-mean objective.

    Merge with existing A0/SAM sums/counts, globally reduce counts once per update,
    then consume weights and parameter_loss_dependencies in existing optimizer guard.
    """
    validate_output(output)
    if not isinstance(targets,CornerTargets) or targets.identities!=output.identities:
        raise ValueError("Corner target/output instance identity mismatch")
    h=output.localization.logits;n=len(h);dev=h.device
    for value,shape in ((targets.uv224,(n,8,2)),(targets.diagonal224,(n,))):
        if value.shape!=shape or value.dtype!=h.dtype or value.device!=dev or not torch.isfinite(value).all():
            raise ValueError("Safe corner target geometry required")
    _bool(targets.known,(n,8),dev,"known");_bool(targets.valid,(n,8),dev,"valid")
    if (targets.valid&~targets.known).any() or (targets.diagonal224<=0).any():
        raise ValueError("Invalid target masks/diagonal")
    zero=h.sum()*0+output.uv224.sum()*0+output.valid_logits.sum()*0+output.scale_b224.sum()*0
    sums={key:zero for key in LOSS_WEIGHTS}
    sums["corner_quality_bce"]=output.quality_logits.sum()*0
    count=int(targets.valid.sum());known_count=int(targets.known.sum())
    counts={key:count for key in ("corner_heatmap_kl","corner_coordinate","corner_laplace_nll")}
    counts.update(corner_valid_bce=known_count,corner_quality_bce=known_count)
    if count:
        uv=targets.uv224[targets.valid].detach()
        axis=torch.arange(224,device=dev,dtype=h.dtype)
        yy,xx=torch.meshgrid(axis,axis,indexing="ij")
        target_logits=-((xx-uv[:,0,None,None]).square()+(yy-uv[:,1,None,None]).square())/(2*1.5**2)
        log_target=target_logits.flatten(1).log_softmax(-1)
        sums["corner_heatmap_kl"]=F.kl_div(h[targets.valid].flatten(1).log_softmax(-1),log_target,
                                          reduction="sum",log_target=True)+zero
        delta=output.uv224[targets.valid]-uv
        diag=targets.diagonal224[:,None].expand(-1,8)[targets.valid].detach()
        sums["corner_coordinate"]=F.smooth_l1_loss(delta/diag[:,None],torch.zeros_like(delta),reduction="sum")+zero
        b=output.scale_b224[targets.valid]
        sums["corner_laplace_nll"]=(delta.abs().sum(-1)/b+2*b.log()).sum()+zero
    if known_count:
        sums["corner_valid_bce"]=F.binary_cross_entropy_with_logits(
            output.valid_logits[targets.known],targets.valid[targets.known].to(h.dtype),reduction="sum")+zero
    with torch.no_grad():
        error=torch.linalg.vector_norm(output.uv224.detach()-targets.uv224.detach(),dim=-1)
        qtarget=(targets.valid&(error<=.01*targets.diagonal224.detach()[:,None])).to(h.dtype)
    if known_count:
        sums["corner_quality_bce"]=F.binary_cross_entropy_with_logits(
            output.quality_logits[targets.known],qtarget[targets.known],reduction="sum")+output.quality_logits.sum()*0
    return CornerLossStatistics(sums,counts,effective_weights(epoch),qtarget,targets.known)
