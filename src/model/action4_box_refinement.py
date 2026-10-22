"""Residual B0 box/quality refinement on cached predicted proposals.

Uses the existing selected appearance adapter. GT is accepted only by loss().
Inference consumes proposal features, boxes and detector priors.
"""
import torch
from torch import nn
from torch.nn import functional as F
from model.action4_box_quality import BoxAwareQuality, varifocal_loss

class BoxRefinement(nn.Module):
    def __init__(self):
        super().__init__()
        self.appearance = BoxAwareQuality()
        self.fusion = nn.Sequential(nn.Linear(147,128),nn.ReLU())
        self.delta = nn.Linear(128,4)
        self.correction = nn.Linear(128,1)
        for layer in (self.delta,self.correction):
            nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)

    def forward(self, features, geometry, priors, boxes):
        visual = torch.cat((self.appearance.appearance(features),
                            self.appearance.geometry(geometry)), -1)
        prior_logits = torch.cat((torch.logit(priors[:,:2].clamp(1e-4,1-1e-4)),
                                  self.appearance.score(visual)), -1)
        h = self.fusion(torch.cat((visual,prior_logits),-1))
        delta = self.delta(h)
        wh = (boxes[:,2:]-boxes[:,:2]).clamp_min(1e-4)
        center = (boxes[:,:2]+boxes[:,2:])*0.5
        refined_center = center + delta[:,:2]*wh
        refined_wh = wh * delta[:,2:].clamp(-2,2).exp()
        refined = torch.cat((refined_center-refined_wh*0.5,
                             refined_center+refined_wh*0.5),-1)
        logits = torch.logit((priors[:,0]*priors[:,1]).clamp(1e-4,1-1e-4))
        return refined, logits+self.correction(h).squeeze(-1), delta

def paired_iou_giou(boxes, targets):
    inter = (torch.minimum(boxes[:,2:],targets[:,2:])-
             torch.maximum(boxes[:,:2],targets[:,:2])).clamp_min(0).prod(-1)
    area = (boxes[:,2:]-boxes[:,:2]).clamp_min(0).prod(-1)
    target_area = (targets[:,2:]-targets[:,:2]).clamp_min(0).prod(-1)
    union = (area+target_area-inter).clamp_min(1e-8)
    iou = inter/union
    enclosing = (torch.maximum(boxes[:,2:],targets[:,2:])-
                 torch.minimum(boxes[:,:2],targets[:,:2])).clamp_min(0).prod(-1).clamp_min(1e-8)
    return iou, iou-(enclosing-union)/enclosing

def loss(outputs, boxes, targets, matched, valid):
    refined,logits,delta = outputs
    iou,giou = paired_iou_giou(refined,targets)
    # Match is based on ORIGINAL proposal, never allow background to chase arbitrary GT.
    positive = matched & valid
    wh = (boxes[:,2:]-boxes[:,:2]).clamp_min(1e-4)
    target_wh = (targets[:,2:]-targets[:,:2]).clamp_min(1e-4)
    target_delta = torch.cat((((targets[:,:2]+targets[:,2:])-
                               (boxes[:,:2]+boxes[:,2:]))*0.5/wh,
                              (target_wh/wh).log()),-1)
    zero = refined.sum()*0
    giou_loss = (1-giou[positive]).mean() if positive.any() else zero
    regression = F.smooth_l1_loss(delta[positive],target_delta[positive],beta=1/9) if positive.any() else zero
    # Quality measures the CURRENT refined box, with no target gradient shortcut.
    quality_target = torch.where(positive,iou.detach(),torch.zeros_like(iou))
    quality = varifocal_loss(logits[valid],quality_target[valid])
    return {"quality":quality,"giou":giou_loss,"regression":regression}
