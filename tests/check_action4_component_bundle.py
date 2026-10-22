"""Reject a changed A0 even when its provenance string stays unchanged."""
import json,copy
from pathlib import Path
import torch
from model.action4_component_bundle import create_bundle,verify_feature_parent

def main():
    torch.set_num_threads(2);root=Path.cwd();out=root/'outputs/action4_bundle_contract_v001';out.mkdir(exist_ok=True)
    directory=root/'outputs/action4_current_geometry_comparison_v001'
    bundle=create_bundle(directory,out/'bundle.json')
    a=torch.load(directory/'A0.pt',map_location='cpu',weights_only=True)
    parent=torch.load(root/a['provenance']['source'],map_location='cpu',weights_only=True)
    changed=dict(a);changed['model']=dict(a['model']);key=next(k for k,v in changed['model'].items() if v.is_floating_point());changed['model'][key]=changed['model'][key].clone();changed['model'][key].flatten()[0]+=1
    try:verify_feature_parent(changed,parent)
    except ValueError:pass
    else:raise AssertionError('Changed feature parent accepted')
    wrong=dict(a,role='B0')
    try:verify_feature_parent(wrong,parent)
    except ValueError:pass
    else:raise AssertionError('Wrong role accepted')
    result={'status':'PASS','exact_parent_parameters_verified':True,'same_label_modified_A0_rejected':True,'wrong_role_rejected':True,'merged_weights':False,'B0_step':bundle['B0_step'],'production_accepted':False}
    (out/'REPORT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
