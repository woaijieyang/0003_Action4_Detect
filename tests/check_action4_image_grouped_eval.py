"""Scorer contract: retain image batches, misses, false positives and target separation."""
import numpy as np
import torch
from eval.action4_image_grouped_pose_eval import evaluate

class Model(torch.nn.Module):
    def __init__(self):super().__init__();self.batches=[]
    def forward(self,maps,K,canonical):
        self.batches.append(len(maps));corners=maps[:,0,0,:16].reshape(-1,8,2)
        return {"corners":corners,"camera_corners":torch.ones(len(maps),8,3)}

def main():
    groups=[]
    for n,label_count in [(2,1),(1,1)]:
        maps=torch.zeros(n,71,64,64);labels=[]
        for j in range(label_count):
            labels.append({"prediction_index":j,"instance_id":j,"corners_px":([[0.,0.]]*8 if n==2 else [[10.,10.]]*8),"diagonal":100.,"heavy":n==1,"visible":[True]*4+[False]*4})
        groups.append({"sample_id":str(n),"inputs":{"maps":maps,"K":torch.eye(3)[None].repeat(n,1,1),"canonical":torch.zeros(n,8,3)},"inverse_affines":[np.eye(3).tolist()]*n,"labels_for_scoring_only":labels})
    model=Model();report=evaluate(model,groups,{"total_GT":3,"heavy_GT":1,"prediction_count":3})
    assert model.batches==[2,1] and report["all8"]==1 and report["total_GT"]==3
    assert report["joint_precision"]==1/3 and report["joint_recall"]==1/3 and len(report["rows"])==2
    assert report["heavy_all8"]==0
    print("PASS: original image batches, FP/miss denominators, labels only score, hidden errors retained")
if __name__=="__main__":main()
