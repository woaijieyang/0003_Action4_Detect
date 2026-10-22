"""Fresh full-size actual-network component check, never a training acceptance."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import torch
from data.action4_dataset import (Action4Dataset, prepare_full_images, prepare_rois,
    build_detection_targets, build_corner_targets, is_ignored)
from model.action4_network import Action4Net
from model.action4_losses import joint_loss
from model.training.action4_runtime import seed_all, environment_identity
from model.training.action4_trainer import build_optimizer


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',choices=('cuda',),default='cuda')
    args=parser.parse_args()
    root=Path.cwd().resolve()
    assert root==Path('/data/zy/vln/a-test')
    def bounded(p):
        p=(root/p).resolve()
        assert p.is_relative_to(root)
        return p
    config_path=bounded(args.config);manifest_path=bounded(args.manifest)
    out=bounded(args.output)
    assert out.is_relative_to(root/'outputs') and not out.exists()
    assert torch.cuda.device_count()==1
    gpu='GPU-'+str(torch.cuda.get_device_properties(0).uuid)
    assert gpu=='GPU-b1ae5be6-1dba-7343-e233-59df41402f33'
    config=json.loads(config_path.read_text())
    assert config['preprocess']['full_size']==[1024,768]
    assert config['preprocess']['roi_size']==256
    assert config['network']['decoder_channels']==256
    assert config['runtime']['amp_init_scale']==8192
    torch.set_num_threads(2)
    seed_all(config['runtime']['seed'],deterministic=True)
    out.mkdir(parents=True)
    data=Action4Dataset(root,manifest_path,allow_diagnostic=True)
    indices=[0,3,4,8,9,16,19,24]
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    sources={str(p.relative_to(root)):sha(p) for folder in ('src/data','src/model','src/eval') for p in (root/folder).rglob('*.py')}
    sources.update({str(config_path.relative_to(root)):sha(config_path),str(manifest_path.relative_to(root)):sha(manifest_path)})
    rgb_identity={data.records[i]['image']:sha(bounded(data.records[i]['image'])) for i in indices}
    model=Action4Net(weights=None,**config['network']).to(args.device)
    identity=model.load_backbone_weights(bounded(config['initialization']['backbone_weights']),config['initialization']['sha256'])
    model.train()
    optimizer=build_optimizer(model,config['stages']['B0'])
    scaler=torch.amp.GradScaler('cuda',init_scale=config['runtime']['amp_init_scale'])
    first=next(model.backbone.parameters())
    initial=first.detach().clone()
    bn_initial=model.backbone.bn1.running_mean.clone()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize();started=time.perf_counter();rows=[]
    for i in indices:
        item=data[i];images=[item['image']];records=[item['record']]
        full,affine=prepare_full_images(images,size=config['preprocess']['full_size'],device=args.device)
        detection=build_detection_targets(records,affine,full_size=config['preprocess']['full_size'],device=args.device)
        valid_indices=[j for j,x in enumerate(records[0]['instances']) if not is_ignored(x)]
        boxes=[records[0]['instances'][j]['bbox_full_xyxy'] for j in valid_indices]
        rois,roi_affine=prepare_rois(images,boxes,[0]*len(boxes),roi_size=256,expansion=config['preprocess']['roi_expansion'],device=args.device)
        target,valid=build_corner_targets(records,[0]*len(boxes),valid_indices,roi_affine,device=args.device)
        with torch.autocast('cuda',dtype=torch.float16):
            result=model(full,roi_images=rois)
            expected={'center_logits':(1,1,192,256),'offset':(1,2,192,256),'ltrb':(1,4,192,256),'corner_logits':(len(boxes),8,64,64)}
            for key,shape in expected.items():
                assert tuple(result[key].shape)==shape,(key,result[key].shape)
                assert bool(torch.isfinite(result[key]).all()),key
            losses=joint_loss(result,detection,target,valid,torch.zeros(len(boxes),dtype=torch.long,device=args.device))
        assert all(bool(torch.isfinite(v)) for v in losses.values())
        scaler.scale(losses['total']/len(indices)).backward()
        rows.append({'sample_id':records[0]['sample_id'],'source_index':i,'target_count':len(boxes),'valid_corner_count':int(valid.sum()),'shapes':{k:list(v.shape) for k,v in result.items()},'losses':{k:float(v.detach()) for k,v in losses.items()}})
        del result,losses
    assert set(x['target_count'] for x in rows)=={0,1,2}
    scaler.unscale_(optimizer)
    bad=[n for n,p in model.named_parameters() if p.grad is not None and not bool(torch.isfinite(p.grad).all())]
    assert not bad,bad
    assert first.grad is not None and float(first.grad.abs().sum())>0
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    assert bool(torch.isfinite(norm))
    scaler.step(optimizer);scaler.update();torch.cuda.synchronize()
    elapsed=time.perf_counter()-started
    assert not torch.equal(initial,first.detach())
    assert torch.equal(bn_initial,model.backbone.bn1.running_mean)
    peak_alloc=torch.cuda.max_memory_allocated();peak_reserved=torch.cuda.max_memory_reserved()
    model.eval()
    with torch.no_grad():
        reference={k:v.detach().cpu() for k,v in model(full,roi_images=rois).items()}
    state={'model':{k:v.cpu() for k,v in model.state_dict().items()},'network':config['network'],'claim_scope':'component_diagnostic','optimizer_updates':1}
    checkpoint=out/'component_state.pt'
    torch.save(state,checkpoint)
    del model,optimizer,first,initial
    restored=Action4Net(weights=None,**config['network']).to(args.device)
    loaded=torch.load(checkpoint,map_location='cpu',weights_only=True)
    restored.load_state_dict(loaded['model'],strict=True);restored.eval()
    with torch.no_grad():
        actual={k:v.detach().cpu() for k,v in restored(full,roi_images=rois).items()}
    diffs={k:float((reference[k]-actual[k]).abs().max()) if reference[k].numel() else 0.0 for k in reference}
    assert all(v==0 for v in diffs.values()),diffs
    assert all(sha(root/p)==h for p,h in sources.items())
    assert all(sha(bounded(p))==h for p,h in rgb_identity.items())
    report={'scope':'actual full-size ResNet50 component check on existing diagnostic R2 RGB, not formal data/training/accuracy acceptance','status':'PASS','environment':environment_identity(torch),'config':args.config,'config_sha256':sha(config_path),'source_identity':sources,'rgb_identity':rgb_identity,'backbone_identity':identity,'microbatches':rows,'global_batch':8,'micro_batch':1,'optimizer_updates':1,'amp_init_scale':8192,'nonfinite_parameters':bad,'gradient_norm':float(norm),'backbone_updated':True,'bn_statistics_unchanged':True,'checkpoint_strict_reload_max_abs_diff':diffs,'checkpoint_sha256':sha(checkpoint),'training_step_seconds':elapsed,'training_peak_allocated_bytes':peak_alloc,'training_peak_reserved_bytes':peak_reserved,'limits':['Not a convergence or generalization check','R2 diagnostic data do not satisfy S0 production acceptance','Existing trainer resume semantics separately checked with CPU control fixture']}
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ('status','scope','optimizer_updates','backbone_updated','checkpoint_strict_reload_max_abs_diff','training_step_seconds','training_peak_allocated_bytes')}))

if __name__=='__main__':main()
