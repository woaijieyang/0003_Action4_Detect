"""Fixed stratified train/dev diagnostic; no checkpoint selection or training."""
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Subset
from data.action4_paths import PROJECT_ROOT as R
from data.action4_component_data import ComponentData
from model.training.action4_auxiliary_train import build
from eval.action4_auxiliary_eval import evaluate
from model.training.action4_runtime import atomic_json


def run(output):
    torch.set_num_threads(2);out=R/output
    cfg=json.loads((R/'configs/model/action4_a0_additive_v002.json').read_text());model,preprocess=build(cfg,'cuda:0')
    state=torch.load(R/'outputs/action4_direction_normalization_v001/selected_best.pt',map_location='cpu',weights_only=True);model.evidence.load_state_dict(state['head'],strict=True)
    reports={};selection={}
    for split in ['train','eval']:
        data=ComponentData(cfg['validation_receipt'],split,'A0');rng=np.random.default_rng(20261023);selected=[]
        for heavy in [True,False]:
            candidates=[i for i,(ri,gi) in enumerate(data.units) if (data.records[ri]['instances'][gi]['occlusion_inframe']>=.4)==heavy]
            selected.extend(rng.choice(candidates,64,replace=False).tolist())
        selection[split]=[{'sample_id':data.records[data.units[i][0]]['sample_id'],'instance_id':data.records[data.units[i][0]]['instances'][data.units[i][1]]['instance_id']} for i in selected]
        reports[split]=evaluate(model,Subset(data,selected),preprocess)
        atomic_json(out/(split+'_REPORT.json'),reports[split]);atomic_json(out/'STATUS.json',{'completed_split':split})
    atomic_json(out/'REPORT.json',{'scope':'fixed128train/128dev stratified diagnostic, not official evaluation','checkpoint_step':state['step'],'selection':selection,'metrics':{s:{k:v for k,v in d.items() if k!='per_roi'} for s,d in reports.items()},'training_mutated':False})

if __name__=='__main__':
    import argparse,traceback
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    try:run(a.output)
    except BaseException:
        atomic_json(R/a.output/'FAILURE.json',{'traceback':traceback.format_exc()});raise
