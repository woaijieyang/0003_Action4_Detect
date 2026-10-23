"""Trainable surface+pose extension of the sealed RGB A0, keeping independent weights."""
import torch
from torch import nn
from model.action4_release_bundle import load_bundle

class JointSurfaceRGB(nn.Module):
    def __init__(self,parent_bundle,train_backbone=False,device="cpu"):
        super().__init__()
        self.head,self.base,self.parent=load_bundle(parent_bundle,"A0",device)
        if self.parent["architecture"]!="joint_pose":raise ValueError("Joint surface/pose parent required")
        self.train_backbone=bool(train_backbone)
        self.base.requires_grad_(False)
        self.head.requires_grad_(True)
        if self.train_backbone:self.base.backbone.requires_grad_(True)
    def train(self,mode=True):
        super().train(mode)
        # Preserve parent backbone BN policy; unused legacy heads never train.
        self.base.train(mode if self.train_backbone else False)
        return self
    def forward(self,rgb,K,canonical):
        return self.head(self.base.backbone(rgb),K,canonical)
