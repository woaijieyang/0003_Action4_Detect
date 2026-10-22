"""Checkpoint identity and exact head-optimizer replay; no accuracy claim."""
from pathlib import Path
import json,copy
import torch
from model.action4_a0_enhancement import DirectionEvidence
from model.action4_direction_losses import direction_trial_loss
from model.training.action4_component_train import config_id


def main():
    torch.set_num_threads(2);root=Path.cwd();out=root/'outputs/action4_direction_normalization_v001'
    parent=torch.load(out/'selected_best.pt',map_location='cpu',weights_only=True)
    rows=[]
    for mode in ['pooled_control','per_roi']:
        run=out/mode;cfg=json.loads((root/f'configs/model/action4_direction_{mode}_v001.json').read_text())
        initial=torch.load(out/(mode+'_initial_evaluated.pt'),map_location='cpu',weights_only=True)
        assert initial['step']==0
        assert all(torch.equal(v,initial['head'][k]) for k,v in parent['head'].items())
        state=torch.load(run/'last.pt',map_location='cpu',weights_only=True)
        assert state['config_id']==config_id(cfg) and state['resumable'] and state['step']>0
        assert set(['rng','epoch','cursor','optimizer','scheduler']).issubset(state)
        assert any(not torch.equal(v,state['head'][k]) for k,v in parent['head'].items())
        torch.manual_seed(99);features=torch.randn(2,256,12,12);rgb=torch.randn(2,3,48,48);points=torch.randn(2,8,2)*15
        mask=torch.zeros(2,12,12,dtype=torch.bool);mask[0,2:6,3:7]=True;mask[1,1:11,1:11]=True
        results=[]
        for _ in range(2):
            model=DirectionEvidence();model.load_state_dict(state['head'],strict=True)
            optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=.0001)
            optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
            optimizer.zero_grad(set_to_none=True)
            loss=sum(direction_trial_loss(model(features,rgb),points,mask,mode).values());loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            results.append({k:v.detach().clone() for k,v in model.state_dict().items()})
        assert all(torch.equal(v,results[1][k]) for k,v in results[0].items())
        rows.append({'mode':mode,'selected_parent_step':parent['step'],'recovery_step':state['step'],'initial_head_exact':True,'actual_training_changed_head':True,'same_state_next_update_exact':True})
    report={'status':'PASS','scope':'checkpoint/head-optimizer replay using constructed features; not full dataloader/resumed-training or accuracy proof','trials':rows}
    (out/'CHECKPOINT_TEST.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
