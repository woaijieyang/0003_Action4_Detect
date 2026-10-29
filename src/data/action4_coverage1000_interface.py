"""Exercise actual no-training A0/B0 batches and all detector target records."""
from pathlib import Path
import argparse,json,hashlib,time,concurrent.futures,multiprocessing,os
import numpy as np
import torch
from data import DataAdapter
from data.action4_paths import PROJECT_ROOT as R
from data.action4_stage2_data import Stage2Units
from data.action4_coverage1000_data import CoverageBatchUnits,surface_source as candidate_surface
from data.action4_phrase2_dataset import surface_source as formal_surface
from data.action4_dataset import prepare_full_images,build_detection_targets
from data.action4_occlusion500_similarity import write

OUT=R/'outputs/coverage1000_v001'
_A=_B=None

def pin_reader(counter):
    with counter.get_lock():index=counter.value;counter.value+=1
    os.sched_setaffinity(0,{4+index});torch.set_num_threads(1)

def inspect_index(index):
    item=_B[(index,0,True,1)]['item'];r=item['record'];rgb,A=prepare_full_images([item['image']],size=(1024,768),device='cpu')
    if tuple(rgb.shape)!=(1,3,768,1024) or not torch.isfinite(rgb).all():raise ValueError('Actual B0 RGB batch failed')
    samples=[]
    for j in range(2):
        unit=_A[(2*index+j,0,True,1)]
        if tuple(unit['roi'].shape)!=(3,448,448) or tuple(unit['surface_masks'].shape)!=(2,112,112):raise ValueError('Actual A0 RGB/mask input shape differs')
        if len(unit['surface_sparse']['xyz'])==0 or not torch.isfinite(unit['surface_sparse']['xyz']).all():raise ValueError('Sparse XYZ did not enter the actual A0 unit')
        if not torch.isfinite(unit['geometry']['corners']).all() or unit['geometry']['corners'].shape!=(8,2):raise ValueError('All8 corners did not enter actual geometry targets')
        samples.append({'sample_id':r['sample_id'],'batch_id':r.get('batch_id','base5000'),'instance_id':j+1,'ROI':list(unit['roi'].shape),
            'masks':list(unit['surface_masks'].shape),'sparse_points':len(unit['surface_sparse']['xyz']),'hidden_points':int((~unit['geometry']['visible']).sum()),
            'surface_projection_error_hm_px':unit['metadata']['surface_projection_error_hm_px']})
    return samples

def detector_targets(records):
    matrices=[DataAdapter.letterbox_affine((r['width'],r['height']),(1024,768))[0] for r in records]
    checked=0
    for first in range(0,len(records),16):
        subset=records[first:first+16];aff=matrices[first:first+16];targets=build_detection_targets(subset,aff,full_size=(1024,768),device='cpu')
        if int(targets['instance_mask'].sum())!=2*len(subset):raise ValueError('A complete camera was dropped by B0 target construction')
        for i,(row,A) in enumerate(zip(subset,aff)):
            if targets['indices'][i,0]==targets['indices'][i,1]:raise ValueError('Two targets occupy the same B0 heatmap cell')
            for j,obj in enumerate(row['instances']):
                box=np.array(obj['bbox_full_xyxy']);full=DataAdapter.transform_points(box.reshape(2,2),A).reshape(-1)
                visible=np.array(obj['bbox_visible_xyxy']);center=DataAdapter.transform_points([(visible[:2]+visible[2:])/2],A)[0]
                d=targets['ltrb'][i,j].numpy()*[1024,768,1024,768];decoded=np.r_[center-d[:2],center+d[2:]]
                if np.max(abs(decoded-full))>.005:raise ValueError('B0 regression target is not the complete body box')
            checked+=2
    return checked

def check(scope='candidate',unpublished=False):
    global _A,_B
    torch.set_num_threads(1);start=time.time();rows=[]
    factory=CoverageBatchUnits if scope=='candidate' else Stage2Units
    source=candidate_surface() if scope=='candidate' else formal_surface()
    expected={'train':800,'eval':100,'test':100} if scope=='candidate' else {'train':5200,'eval':650,'test':650}
    all_target_count=0
    for split,n in expected.items():
        base={'mode':'integration','allow_unpublished_interface_validation':scope=='candidate' or unpublished,'seed':2026102304,
            'dataset_root':'data/phrase2','roi_jitter':{'center_fraction':.15,'scale_range':[.65,1.35]} if split=='train' else {'center_fraction':0.,'scale_range':[1.,1.]},'surface_source':source}
        b=factory(dict(base,role='B0'),split);a=factory(dict(base,role='A0'),split)
        if len(b)!=n or len(a)!=2*n:raise ValueError('Actual dynamic factory counts differ')
        records=b.data.records;all_target_count+=detector_targets(records)
        chosen={0,n-1}
        seen=set()
        for i,row in enumerate(records):
            marker=(row.get('batch_id','base5000'),row.get('phrase2',{}).get('coverage_group'),row['scene_id'])
            if marker not in seen:chosen.add(i);seen.add(marker)
        _A,_B=a,b;counter=multiprocessing.Value('i',0)
        with concurrent.futures.ProcessPoolExecutor(min(16,len(chosen)),initializer=pin_reader,initargs=(counter,)) as pool:
            samples=[sample for result in pool.map(inspect_index,sorted(chosen),chunksize=1) for sample in result]
        rows.append({'split':split,'B0_images':len(b),'A0_units':len(a),'all_full_body_detector_targets_checked':2*n,'actual_image_and_target_samples':samples})
    report={'status':'PASS_ACTUAL_DATA_BATCHES_NO_TRAINING','scope':scope,'unpublished_integration_only':unpublished or scope=='candidate','splits':rows,
        'all_detector_targets_checked':all_target_count,'no_model_constructed':True,'no_optimizer_or_training_started':True,'seconds':time.time()-start}
    name='DATA_INTERFACE_VALIDATION.json' if scope=='candidate' else 'FORMAL_ROOT_INTEGRATION_VALIDATION.json' if unpublished else 'FORMAL_DATA_INTERFACE_VALIDATION.json'
    write(OUT/name,report);print(json.dumps({k:v for k,v in report.items() if k!='splits'},indent=2));return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scope',choices=['candidate','formal'],default='candidate');p.add_argument('--unpublished',action='store_true');a=p.parse_args();check(a.scope,a.unpublished)
