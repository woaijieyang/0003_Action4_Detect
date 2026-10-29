"""CPU frozen-B0 appearance separability probe, not formal detector selection.

Train-only reference bank later; full development candidate domain is retained.
GT is used only after RGB feature/proposal computation for supervised labels.
"""
import argparse,json,hashlib,time,os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from data.action4_paths import PROJECT_ROOT as R,contained_path
from data.action4_component_data import ComponentData
from data.action4_dataset import prepare_full_images,is_ignored
from model.action4_online_b0 import proposal_features
from model.action4_proposal_pool import candidate_union
from model.action4_release_bundle import load_bundle


def sha(path):
    with contained_path(R,path,exists=True).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def ious(boxes,gt):
    if not len(gt):return np.zeros((len(boxes),0))
    inter=np.maximum(0,np.minimum(boxes[:,None,2:],gt[None,:,2:])-np.maximum(boxes[:,None,:2],gt[None,:,:2])).prod(-1)
    union=(boxes[:,2:]-boxes[:,:2]).prod(-1)[:,None]+(gt[:,2:]-gt[:,:2]).prod(-1)[None]-inter
    return inter/np.maximum(union,1e-12)


def coverage(boxes,mask):
    # Exact integer-centre [l,r) support; summed-area table is per GT image only.
    h,w=mask.shape;integral=np.pad(mask.astype(np.int32),((1,0),(1,0))).cumsum(0,dtype=np.int32).cumsum(1,dtype=np.int32)
    b=np.clip(np.ceil(boxes),[0,0,0,0],[w,h,w,h]).astype(np.int64);l,t,r,bottom=b.T
    counts=integral[bottom,r]-integral[t,r]-integral[bottom,l]+integral[t,l]
    total=int(mask.sum());return counts/max(1,total),total


def labels_after_prediction(boxes,record):
    objects=record['instances'];valid=[g for g in objects if not is_ignored(g)];ignored=[g for g in objects if is_ignored(g)]
    gt=np.asarray([g['bbox_full_xyxy'] for g in valid],float).reshape(-1,4);ig=np.asarray([g['bbox_full_xyxy'] for g in ignored],float).reshape(-1,4)
    ov=ious(boxes,gt);iv=ious(boxes,ig);label=np.full(len(boxes),-1,np.int8);target=np.full(len(boxes),-1,np.int32)
    hit=(ov>=.5);positive=hit.sum(1)==1
    if len(valid):target[positive]=ov[positive].argmax(1)
    ignore_touch=(iv>0).any(1) if len(ignored) else np.zeros(len(boxes),bool)
    positive &= ~ignore_touch
    label[positive]=1
    maxcov=np.zeros(len(boxes));unseen_touch=np.zeros(len(boxes),bool)
    for j,g in enumerate(valid):
        p=contained_path(R,g['mask_visible'],exists=True)
        if sha(p)!=g['mask_visible_sha256']:raise ValueError('GT mask changed')
        with Image.open(p) as image:mask=np.asarray(image.convert('L'))>0
        if mask.shape!=(record['height'],record['width']):raise ValueError('GT mask shape differs from native frame')
        cov,total=coverage(boxes,mask);maxcov=np.maximum(maxcov,cov)
        positive &= ~((target==j)&(cov<=0))
        # Pure single-target reference/proposal label: ANY visible second-body
        # support is mixed. This is deliberately not the negative5% tolerance.
        positive &= ~((target!=j)&(cov>0))
        if not total:
            unseen_touch|=ov[:,j]>0
            positive &= ~((target!=j)&(ov[:,j]>0))
    label[label==1]=-1;label[positive]=1
    maxov=ov.max(1) if len(valid) else np.zeros(len(boxes))
    negative=(maxov<=.1)&(maxcov<=.05)&~ignore_touch&~unseen_touch
    label[negative]=0;target[label!=1]=-1
    return label,target,ov,maxcov


def selected_ids(records,split,cfg):
    ids=[r['sample_id'] for r in records]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate dataset image IDs')
    if len(ids)!=(5200 if split=='train' else 650):raise ValueError('Full source domain changed')
    if split=='eval':return ids
    if cfg.get('selection_protocol')!='seeded_sorted_scene64_v1':raise ValueError('Selection algorithm identity required')
    groups={}
    for r in records:groups.setdefault(r['scene_family'],[]).append(r['sample_id'])
    if len(groups)!=3 or any(len(v)<64 for v in groups.values()):raise ValueError('Three sufficiently populated scenes required')
    rng=np.random.default_rng(cfg['seed']);expected=[]
    for key in sorted(groups):expected+=sorted(rng.choice(groups[key],64,replace=False).tolist())
    chosen=cfg['train_image_ids']
    if len(chosen)!=192 or len(set(chosen))!=192 or chosen!=expected:raise ValueError('Training reference sampling identity differs')
    return chosen


def extract(config):
    cfg=json.loads(contained_path(R,config,exists=True).read_text())
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU probe only')
    if set(os.sched_getaffinity(0))!=set(cfg['cpu_affinity']):raise ValueError('Probe CPU allocation differs')
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(cfg['seed'])
    if sha(cfg['B0_bundle'])!=cfg['B0_sha256']:raise ValueError('Frozen feature model changed')
    model,_,state=load_bundle(str(contained_path(R,cfg['B0_bundle'],exists=True)),'B0','cpu');model.eval().requires_grad_(False)
    if state['selected_step']!=9500:raise ValueError('Explicit fixedB0step9500 experiment required')
    out=contained_path(R,cfg['output'],exists=False);out.mkdir(exist_ok=False);(out/'features').mkdir();records=[]
    config_sha=sha(config);started=time.time();validation=json.loads(contained_path(R,cfg['validation_receipt'],exists=True).read_text());data_sources={'validation_receipt_sha256':sha(cfg['validation_receipt']),'annotations':{split:validation['splits'][split] for split in ('train','eval')}}
    data_sources['annotations']={k:{'manifest':v['manifest'],'sha256':v['sha256']} for k,v in data_sources['annotations'].items()}
    for split in ('train','eval'):
        data=ComponentData(cfg['validation_receipt'],split,'B0');selected=selected_ids(data.records,split,cfg)
        byid={r['sample_id']:i for i,r in enumerate(data.records)}
        if len(data.records)!=(5200 if split=='train' else 650) or len(selected)!=(192 if split=='train' else 650):raise ValueError('Probe image domain mismatch')
        for iid in selected:
            item=data[byid[iid]];record=item['record']
            if sha(record['image'])!=record['rgb_sha256']:raise ValueError('Probe RGB differs')
            x,A=prepare_full_images([item['image']],size=tuple(state['config']['preprocess']['full_size']),device='cpu')
            with torch.inference_mode():
                dense,features=model(x);rows,canvas=candidate_union(dense,A[0],state['config'],topk=100)
                pooled,geometry=proposal_features(features,canvas,tuple(state['config']['preprocess']['full_size']))
                appearance=model.refiner.appearance.appearance(pooled).detach().numpy().astype(np.float32)
            boxes=np.asarray([p['bbox_full_xyxy'] for p in rows],np.float64).reshape(-1,4)
            labels,target,overlap,visible_coverage=labels_after_prediction(boxes,record)
            if appearance.shape!=(len(rows),128) or not np.isfinite(appearance).all():raise ValueError('Feature shape/finiteness mismatch')
            dest=out/'features'/f'{split}_{iid}.npz'
            np.savez_compressed(dest,appearance=appearance,boxes=boxes,labels=labels,target=target,GT_IoU=overlap,
                max_visible_coverage=visible_coverage,center_score=np.asarray([p['center_score'] for p in rows],np.float32),grid=np.asarray([p['grid_index'] for p in rows],np.int64))
            records.append({'image_id':iid,'split':split,'scene':record['scene_family'],'file':str(dest.relative_to(R)),
                'file_sha256':sha(dest),'RGB_sha256':record['rgb_sha256'],'GT_count':sum(not is_ignored(g) for g in record['instances']),
                'candidates':len(rows),'positive':int((labels==1).sum()),'negative':int((labels==0).sum()),'ignored_or_ambiguous':int((labels<0).sum())})
            progress={'status':'EXTRACTING','completed':len(records),'expected':842,'last_image':iid,'split':split,'elapsed_seconds':time.time()-started}
            temp=out/'STATUS.tmp';temp.write_text(json.dumps(progress));temp.replace(out/'STATUS.json')
    if len({(r['split'],r['image_id']) for r in records})!=842 or len({r['file'] for r in records})!=842:raise ValueError('Duplicate/missing output coverage')
    result={'status':'FROZEN_FEATURE_EXTRACTION_COMPLETE_NOT_ADAPTER_TRAINING','model_sha256':cfg['B0_sha256'],'model_step':9500,
        'config_sha256':config_sha,'data_sources':data_sources,'label_source_sha256':sha(__file__),'label_protocol':'uniqueIoU50_visible_evidence_zero_other_visible_leakage_v1','train_images':192,'eval_images':650,'feature_dim':128,'geometry_excluded_from_cosine':True,
        'records':records,'GT_only_after_prediction':True,'no_test_read':True,'no_GPU':True,'seconds':time.time()-started}
    (out/'FEATURE_MANIFEST.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();extract(a.config)
