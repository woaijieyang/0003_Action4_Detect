"""Real-sample gradient and immutable-base checks; not accuracy evidence."""
from pathlib import Path
import sys,json,torch,numpy as np
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'src'))
from model.action4_components import load_component
from model.action4_a0_enhancement import A0Enhancement,evidence_loss,direction_supervision
from data.action4_component_data import ComponentData
from data.action4_direction_targets import prepare_evidence_batch


def main():
    torch.set_num_threads(2)
    torch.manual_seed(22)
    base,state=load_component(R/'models/action4/independent_components_v001/A0.pt',role='A0')
    model=A0Enhancement(base)
    data=ComponentData('outputs/action4_independent_trials_v001/DATA_VALIDATED.json','train','A0')
    index=next(i for i,(ri,gi) in enumerate(data.units) if data.records[ri]['instances'][gi]['occlusion_inframe']>=.4)
    x,points,mask,objects=prepare_evidence_batch([data[index]],state['preprocess'])
    before={name:value.clone() for name,value in base.state_dict().items()}
    with torch.no_grad():
        reference=base(x)['corner_logits']
        disabled=model(x,enhancement_enabled=False)['corner_logits']
    assert torch.equal(reference,disabled)
    targets,support=direction_supervision(points,mask)
    assert support.sum()>0 and support.shape[1]==8
    out=model(x,enhancement_enabled=True)
    losses=evidence_loss(out,points,mask)
    grad=torch.autograd.grad(losses['direction'],model.evidence.direction.weight,retain_graph=True)[0]
    hidden=next(i for i,v in enumerate(objects[0]['corner_visible']) if not v)
    changed=points.clone();changed[:,hidden,0]+=5
    changed_loss=evidence_loss(out,changed,mask)['direction']
    changed_grad=torch.autograd.grad(changed_loss,model.evidence.direction.weight,retain_graph=True)[0]
    assert not torch.equal(grad,changed_grad)
    optimizer=torch.optim.AdamW(model.trainable_parameters(),lr=1e-3)
    optimizer.zero_grad();sum(losses.values()).backward()
    assert all(p.grad is None for p in base.parameters())
    head_before=model.evidence.direction.weight.detach().clone()
    optimizer.step();model.train()
    assert not torch.equal(head_before,model.evidence.direction.weight)
    assert all(torch.equal(value,base.state_dict()[name]) for name,value in before.items())
    with torch.no_grad():after=model(x,enhancement_enabled=False)['corner_logits']
    assert torch.equal(reference,after)
    output=R/'outputs/action4_a0_additive_v001';output.mkdir(exist_ok=True)
    result=dict(status='PASS',sample_id=data.records[data.units[index][0]]['sample_id'],roi_size=state['preprocess']['roi_size'],all8_direction_channels=True,hidden_label_changes_gradient=True,base_gradients_none=True,all_base_state_unchanged=True,disabled_output_exact_before_after_update=True,auxiliary_parameters_updated=True,visible_surface_pixels=int(mask.sum()),scope='one real synthetic training sample engineering update, not full training or accuracy',official_reference='PVNet visible-pixel directions; auxiliary native-resolution branch, not full PVNet')
    (output/'INTERFACE_TEST.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
