"""Actual cached proposal update and bbox scoring integration, not accuracy."""
from pathlib import Path
import json,torch
from model.action4_box_quality import BoxAwareQuality,varifocal_loss
from eval.action4_box_quality_eval import evaluate

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(23);root=Path.cwd();out=root/'outputs/action4_box_quality_candidate_v001'
    rows=[torch.load(out/f'cache/train_{i:05d}.pt',map_location='cpu',weights_only=True) for i in range(2)]
    model=BoxAwareQuality().to('cuda:0');opt=torch.optim.AdamW(model.parameters(),lr=.001)
    f=torch.cat([x['features'][x['supervision_valid']] for x in rows]).to('cuda:0');g=torch.cat([x['geometry'][x['supervision_valid']] for x in rows]).to('cuda:0');target=torch.cat([x['target'][x['supervision_valid']] for x in rows]).to('cuda:0');assert (target>0).any()
    before=model.score.weight.detach().clone();loss=varifocal_loss(model(f,g),target);assert torch.isfinite(loss);loss.backward();assert model.score.weight.grad.norm()>0;opt.step();assert not torch.equal(before,model.score.weight)
    records=[json.loads(l) for l in (root/'data/train/annotations.jsonl').read_text().splitlines()[:2]];report=evaluate(model,rows,records);assert report['overall']['images']==2
    result={'status':'PASS','actual_train_images':2,'proposals':len(target),'positive':int((target>0).sum()),'loss':float(loss.detach()),'head_parameters_changed':True,'bbox_metric_pipeline_completed':True,'scope':'one-update engineering check, not model performance'}
    (out/'CACHED_UPDATE_TEST.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
