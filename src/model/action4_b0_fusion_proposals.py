"""Current-forward B0 pool plus explicit frozen external evidence; no GT input."""
from dataclasses import dataclass
import numpy as np
import torch
from model.action4_guidance_contract import FrameIdentity,SourceIdentity,CandidateRecord,BudgetRecord
from model.action4_local_evidence import LocalEvidenceBatch,build_local_evidence,AssociationPolicy
from model.action4_proposal_pool import candidate_union
from model.action4_b0_fusion import GuidanceProposals

@dataclass(frozen=True)
class FusionFrameInput:
    frame:FrameIdentity
    b0_source:SourceIdentity
    native_to_canvas:np.ndarray
    external_candidates:tuple=()
    supports:tuple=()


def build_current_proposals(dense,feature,input,config,*,topk=100,policy=AssociationPolicy()):
    if not isinstance(input,FusionFrameInput) or not isinstance(input.frame,FrameIdentity) or input.b0_source.producer!='b0':
        raise ValueError('Explicit current frame and B0 source required')
    if any(c.frame!=input.frame or c.source.producer=='b0' for c in input.external_candidates):
        raise ValueError('External source RGB/frame/pixel identity mismatch')
    affine=np.asarray(input.native_to_canvas,float)
    if affine.shape!=(3,3) or not np.isfinite(affine).all():raise ValueError('Native canvas affine required')
    # Current dense values only, no saved previous B0 predictions or two-object NMS.
    rows,_=candidate_union(dense,affine,config,topk=topk)
    candidates=[];priors=[]
    for row in rows:
        pid=f"b0:{input.frame.image_id}:grid:{row['grid_index']}"
        scores=tuple((k,float(row[k])) for k in ('center_score','old_quality_score','old_product_score'))
        candidates.append(CandidateRecord(input.frame,pid,pid,input.b0_source,tuple(row['bbox_full_xyxy']),scores))
        priors.append([row['center_score'],row['old_quality_score'],row['old_product_score']])
    b0_evidence=build_local_evidence(candidates,input.external_candidates,input.supports,policy=policy)
    candidates=list(b0_evidence.attach(candidates));evidence=list(b0_evidence.evidence)
    matched={sid for e in evidence for channel in ('text_positive','visual_positive','negative') for sid in getattr(e,channel).source_candidate_ids if getattr(e,channel).available}
    # Only positive external candidates may create an object proposal. Negative
    # evidence remains a feature; it never deletes B0 candidates by itself.
    extras=[c for c in input.external_candidates if c.evidence_channel in ('text_positive','visual_positive') and c.candidate_id not in matched]
    # The candidate universe must stay joint. A singleton reassociation would
    # falsely turn shared/merged support into one unique match per duplicate.
    candidates.extend(extras)
    joint=build_local_evidence(candidates,input.external_candidates,input.supports,policy=policy)
    candidates=list(joint.attach(candidates));evidence=list(joint.evidence)
    ids=tuple(c.candidate_id for c in candidates)
    if len(ids)!=len(set(ids)):raise ValueError('Candidate identity collision')
    boxes=feature.new_tensor([c.bbox_candidate for c in candidates]).reshape(-1,4)
    flags=torch.tensor([False]*len(rows)+[True]*len(extras),device=feature.device)
    aligned=LocalEvidenceBatch(ids,tuple(evidence),joint.pair_diagnostics)
    packet=GuidanceProposals(input.frame.image_id,ids,boxes,feature.new_tensor(affine),aligned,flags,
        feature.new_tensor(priors).reshape(-1,3) if rows else None,tuple(ids[:len(rows)]))
    # Budget hits are diagnostic, never 'correct rejection'. No final instance
    # confirmation or near-pair dedup is performed by this preparation function.
    from torch.nn import functional as F
    with torch.no_grad():
        center=dense['center_logits'].sigmoid();product=center*dense['quality_logits'].sigmoid()
        counts=[int(((p==F.max_pool2d(p,3,stride=1,padding=1))&(p>0)).sum()) for p in (center,product)]
    budgets=tuple(BudgetRecord(name,count,topk) for name,count in zip(('b0_center_peaks','b0_product_peaks'),counts))
    final_matched={sid for e in joint.evidence for k in ('text_positive','visual_positive','negative') for sid in getattr(e,k).source_candidate_ids if getattr(e,k).available}
    trace={'b0_count':len(rows),'yolo_only_count':len(extras),'pre_joint_matched_external_ids':sorted(matched),'matched_external_ids':sorted(final_matched),
           'candidate_status':'unconfirmed_hypotheses','final_instance_selection_ready':False,
           'b0_peak_counts_before_budget':counts,'b0_budget_truncated_counts':[max(0,n-topk) for n in counts],
           'b0_topk_per_branch':topk,'b0_budget_upper_bound':2*topk,'two_object_cap':False,
           'b0_weight_identity_kind':input.b0_source.weight_identity_kind,
           'b0_optimizer_step':input.b0_source.optimizer_step,
           'ambiguous_external_evidence_count':sum(getattr(e,k).match_status=='ambiguous' for e in evidence for k in ('text_positive','visual_positive','negative'))}
    return packet,tuple(candidates),budgets,trace
