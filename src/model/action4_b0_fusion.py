"""Opt-in B0 guidance owner; proposals explicit, baseline inference never silent.

Candidate construction remains a separate prediction-only adapter. GT has no
place in this forward packet. Full trainer admission waits for that adapter.
"""
from dataclasses import dataclass
import torch
from model.action4_online_b0 import OnlineB0,proposal_features
from model.action4_b0_guidance_heads import B0GuidanceHeads,guidance_head_dependencies
from model.action4_local_evidence import LocalEvidenceBatch

VARIANT='b0_fusion_image'
ARCHITECTURE='online_b0_guidance'
PROTOCOL={'version':'action4.b0_guidance_connection.v1','appearance':'shared128+16','local_evidence':6,
          'full_box':'YOLO-only prior-free delta; originalB0 keeps real priors','identity':'separate from full-box quality'}


def enabled(cfg):
    v=cfg.get('b0_variant')
    if v not in (None,'baseline',VARIANT):raise ValueError('Unknown B0 variant')
    return v==VARIANT


@dataclass(frozen=True)
class GuidanceProposals:
    image_id:str
    candidate_ids:tuple
    boxes_native:torch.Tensor
    affine_native_to_canvas:torch.Tensor
    evidence:LocalEvidenceBatch
    yolo_only:torch.Tensor
    b0_priors:object=None
    b0_prior_ids:tuple=()

    def canvas_boxes(self,feature,canvas):
        if not isinstance(self.image_id,str) or not self.image_id:raise ValueError('Explicit image identity required')
        b=self.boxes_native;a=self.affine_native_to_canvas
        if b.shape!=(len(self.candidate_ids),4) or b.device!=feature.device or b.dtype!=feature.dtype or not torch.isfinite(b).all():raise ValueError('Native proposal tensor mismatch')
        if a.shape!=(3,3) or a.device!=feature.device or a.dtype!=feature.dtype or not torch.isfinite(a).all():raise ValueError('Proposal affine mismatch')
        expected=a.new_tensor([[a[0,0],0,a[0,2]],[0,a[1,1],a[1,2]],[0,0,1]])
        if not torch.equal(a,expected) or a[0,0]<=0 or a[1,1]<=0:raise ValueError('Explicit positive scale/translation pixel map required')
        if len(self.candidate_ids)!=len(set(self.candidate_ids)):raise ValueError('Duplicate candidate identities')
        if self.evidence.candidate_ids!=self.candidate_ids:raise ValueError('Candidate/evidence identity mismatch')
        corners=b.reshape(-1,2,2);ones=torch.ones_like(corners[...,:1]);mapped=torch.cat((corners,ones),-1)@a.T
        return mapped[...,:2].reshape(-1,4).detach()


class GuidedB0(OnlineB0):
    def __init__(self,network):
        super().__init__(network);self.guidance=B0GuidanceHeads()

    def bind_prediction_source(self,source):
        from model.action4_guidance_contract import SourceIdentity
        if not isinstance(source,SourceIdentity) or source.producer!='b0':raise ValueError("Typed B0 source required")
        self._prediction_source=source
        self._prediction_versions=tuple(p._version for p in self.parameters())

    @property
    def prediction_source(self):
        if not hasattr(self,'_prediction_source'):raise ValueError("Model-owned source binding required")
        if tuple(p._version for p in self.parameters())!=self._prediction_versions:
            raise ValueError("Weights changed without recording the optimizer update")
        return self._prediction_source

    def record_optimizer_update(self,step):
        from dataclasses import replace
        if not hasattr(self,'_prediction_source') or self._prediction_source.weight_identity_kind!='initialization_sha256_plus_update':
            raise ValueError("Training update requires initialization/config identity")
        if type(step) is not int or step!=self._prediction_source.optimizer_step+1:raise ValueError("Optimizer update counter must advance exactly once")
        versions=tuple(p._version for p in self.parameters())
        if versions==self._prediction_versions:raise ValueError("No actual parameter update to record")
        self._prediction_source=replace(self._prediction_source,optimizer_step=step)
        self._prediction_versions=versions
        return self.prediction_source

    def forward(self,images,*,proposal_packets=None,canvas_size=None,image_ids=None,frame_inputs=None,config=None,topk=100):
        if frame_inputs is not None:
            if proposal_packets is not None or image_ids is not None or canvas_size is not None:raise ValueError("Choose one explicit proposal interface")
            return self.forward_current(images,frame_inputs=frame_inputs,config=config,topk=topk)
        if proposal_packets is None or len(proposal_packets)!=len(images):raise ValueError('GuidedB0 requires explicit current prediction proposals; legacy baseline consumer refused')
        if image_ids is None or len(image_ids)!=len(images) or any(packet.image_id!=iid for packet,iid in zip(proposal_packets,image_ids)):
            raise ValueError('Proposal packets do not match current image order/identity')
        if canvas_size!=tuple((images.shape[-1],images.shape[-2])):raise ValueError('Proposal canvas differs from actual input')
        dense,features=super().forward(images);outputs=[]
        for i,packet in enumerate(proposal_packets):
            if not isinstance(packet,GuidanceProposals):raise ValueError('Typed guidance proposals required')
            boxes=packet.canvas_boxes(features,canvas_size)
            pooled,geometry=proposal_features(features[i:i+1],boxes,canvas_size)
            outputs.append(self.guidance(self.refiner,pooled,geometry,packet.boxes_native,
                candidate_ids=packet.candidate_ids,evidence=packet.evidence,yolo_only=packet.yolo_only,
                b0_priors=packet.b0_priors,b0_prior_ids=packet.b0_prior_ids))
        return dense,features,tuple(outputs)

    def forward_current(self,images,*,frame_inputs,config,topk=100):
        """Same single dense forward supplies live proposals and their features."""
        from model.action4_b0_fusion_proposals import build_current_proposals
        if len(frame_inputs)!=len(images):raise ValueError("One current frame input per image required")
        source=self.prediction_source
        if any(input.b0_source!=source for input in frame_inputs):raise ValueError("Stale or unrelated B0 source token")
        canvas=tuple(config["preprocess"]["full_size"])
        if canvas!=(images.shape[-1],images.shape[-2]):raise ValueError("Current image canvas/config mismatch")
        dense,features=super().forward(images);all_outputs=[];all_records=[];all_traces=[]
        for i,input in enumerate(frame_inputs):
            one={k:v[i:i+1] for k,v in dense.items()}
            packet,records,budgets,trace=build_current_proposals(one,features[i:i+1],input,config,topk=topk)
            boxes=packet.canvas_boxes(features,canvas)
            pooled,geometry=proposal_features(features[i:i+1],boxes,canvas)
            output=self.guidance(self.refiner,pooled,geometry,packet.boxes_native,
                candidate_ids=packet.candidate_ids,evidence=packet.evidence,yolo_only=packet.yolo_only,
                b0_priors=packet.b0_priors,b0_prior_ids=packet.b0_prior_ids)
            all_outputs.append(output);all_records.append(records);all_traces.append(trace)
        return dense,features,tuple(all_outputs),tuple(all_records),tuple(all_traces)

    def independent_dependencies(self):
        return tuple(dict(row,parameter_prefixes=['guidance.'+p for p in row['parameter_prefixes']]) for row in guidance_head_dependencies())
