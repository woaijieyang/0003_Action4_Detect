"""Frozen RGB backbone/surface with trainable original ConvPnP at actual crops."""
import torch
from torch import nn
from model.action4_release_bundle import load_bundle
from model.action4_pose_maps import PoseFromMaps,surface_to_maps

class OnlinePoseRGB(nn.Module):
    def __init__(self,bundle_path):
        super().__init__()
        head,base,state=load_bundle(bundle_path,"A0","cpu")
        if state["architecture"]!="joint_pose" or state.get("frozen_surface_step")!=11300:
            raise ValueError("Completed frozen-surface pose parent required")
        self.backbone=base.backbone
        self.surface=head.surface
        self.pose=PoseFromMaps();self.pose.pose=head.pose
        self.backbone.requires_grad_(False);self.surface.requires_grad_(False)
        self.parent_step=state["selected_step"];self.train(False)
    def train(self,mode=True):
        super().train(mode)
        self.backbone.eval();self.surface.eval()
        return self
    def forward(self,rois,K,canonical):
        with torch.no_grad():maps=surface_to_maps(self.surface(self.backbone(rois)),K)
        return self.pose(maps,K,canonical)
