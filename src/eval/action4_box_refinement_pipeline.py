"""Independent B0 base + residual adapter inference in native image coordinates."""
import json
import torch
import numpy as np
from data.action4_paths import PROJECT_ROOT as R,contained_path
from data.action4_dataset import prepare_full_images
from eval.action4_component_pipeline import load_trained_component
from eval.action4_box_quality_pool import candidate_union
from eval.action4_soft_nms import select
from model.action4_box_quality import box_features
from model.action4_box_refinement import BoxRefinement
from model.training.action4_runtime import canonical_hash

class RefinedB0:
    def __init__(self,base_path,head_path,device="cuda:0"):
        self.device=device
        self.base,self.config=load_trained_component(contained_path(R,base_path,exists=True),"B0",device)
        self.head=BoxRefinement().to(device)
        self.state=torch.load(contained_path(R,head_path,exists=True),map_location="cpu",weights_only=True)
        expected=json.loads((R/"configs/model/action4_box_refinement_v001.json").read_text())
        if self.state["config_id"]!=canonical_hash(expected):raise ValueError("Refinement model configuration mismatch")
        self.head.load_state_dict(self.state["head"],strict=True)
        self.base.eval();self.head.eval()
        self.feature=None
        self.hook=self.base.decoder.register_forward_hook(self._capture)
    def _capture(self,module,args,output):
        self.feature=output.detach()
    @torch.inference_mode()
    def predict(self,rgb,threshold=.3):
        tensor=torch.from_numpy(np.array(rgb)).permute(2,0,1)
        image,affines=prepare_full_images([tensor],size=tuple(self.config["preprocess"]["full_size"]),device=self.device)
        dense=self.base(image)
        candidates,canvas_boxes=candidate_union(dense,affines[0],self.config)
        f,g=box_features(self.feature,canvas_boxes,tuple(self.config["preprocess"]["full_size"]))
        boxes=f.new_tensor([p["bbox_full_xyxy"] for p in candidates]).reshape(-1,4)
        priors=f.new_tensor([[p["center_score"],p["old_quality_score"],p["old_product_score"]] for p in candidates]).reshape(-1,3)
        refined,logits,_=self.head(f,g,priors,boxes)
        proposals=[dict(p,bbox_full_xyxy=b,score=float(s)) for p,b,s in zip(candidates,refined.cpu().tolist(),logits.sigmoid().cpu().tolist())]
        return select(proposals,"soft_gaussian50",threshold=threshold)
    def close(self):
        self.hook.remove()
