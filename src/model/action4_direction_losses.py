"""Per-object direction supervision for independently controlled A0 trials."""
import torch
from torch.nn import functional as F
from model.action4_a0_enhancement import direction_supervision,evidence_loss


def direction_trial_loss(output,points,mask,mode):
    if mode=="pooled_control":return evidence_loss(output,points,mask)
    if mode!="per_roi":raise ValueError("Unknown direction normalization")
    target,support=direction_supervision(points,mask)
    error=F.smooth_l1_loss(output["direction_vectors"],target,reduction="none").mean(2)
    counts=support.sum((1,2,3));valid=counts>0
    per_roi=(error*support).sum((1,2,3))/counts.clamp_min(1)
    direction=(per_roi*valid).mean()  # Empty ROI contributes zero; partition-invariant.
    logits=output["foreground_logits"]
    foreground=F.binary_cross_entropy_with_logits(logits[:,1]-logits[:,0],mask.float())
    return {"direction":direction,"foreground":foreground}
