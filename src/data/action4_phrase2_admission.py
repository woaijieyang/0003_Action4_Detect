"""Read-only published-data consumption; no generation, relabeling or split changes."""
import hashlib,json,os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image
from data.action4_paths import PROJECT_ROOT as R,contained_path
from data.action4_dataset import Action4Dataset,is_ignored
from data.action4_phrase2_dataset import Phrase2Dataset

def prepare(output,workers=12):
    out=contained_path(R,output,exists=False);out.mkdir(parents=True,exist_ok=False)
    release=json.loads((R/'data/phrase2/DATASET_COMPLETE.json').read_text())
    geometry=json.loads((R/'data/phrase2/GEOMETRY_ADMISSION.json').read_text())
    if geometry['manifest_sha256']!=release['manifest_sha256']:raise ValueError('Surface/release identities differ')
    receipt={'schema':'action4.component_data_validation.v1','dataset_root':'data/phrase2','splits':{}}
    counts={};missing=[]
    for split in ('train','eval','test'):
        ds=Phrase2Dataset(split,role='B0');rows=ds.records
        def check(row):
            image=contained_path(R,row['image'],exists=True)
            before=Action4Dataset._file_identity(image)
            with Image.open(image) as im:
                if im.size!=(row['width'],row['height']):raise ValueError('Dimensions changed')
                im.load()
            if Action4Dataset._file_identity(image)!=before:raise ValueError('Image changed during validation')
            for g in row['instances']:
                for k in ('mask_visible','mask_amodal_expanded'):
                    contained_path(R,g[k],exists=True)
                contained_path(R,f"data/phrase2/{split}/geometry/{row['sample_id']}_i{g['instance_id']}.npz",exists=True)
            return row['image'],before
        with ThreadPoolExecutor(max_workers=workers) as pool:stats=dict(pool.map(check,rows))
        manifest=f'data/phrase2/{split}/annotations.jsonl'
        receipt['splits'][split]={'manifest':manifest,'sha256':release['manifest_sha256'][split],'stats':stats}
        instances=[g for row in rows for g in row['instances']]
        counts[split]={'images_declared':len(rows),'images_decodable':len(stats),'images_training_eligible':len(rows),
            'instances_declared':len(instances),'instances_ignore':sum(bool(g.get('ignore')) for g in instances),
            'instances_role_eligible':sum(not is_ignored(g) for g in instances),
            'instances_unobservable':sum(bool(g.get('unobservable')) or g.get('status')=='unobservable' for g in instances)}
    path=out/'PHRASE2_VALIDATED.json';path.write_text(json.dumps(receipt))
    gate={'schema':'action4.full_training_admission.v1','launch_allowed':True,'mode':'full_training',
          'formal_stage2_allowed':True,'validation_receipt':str(path.relative_to(R)),
          'role_units':{'B0':counts['train']['images_training_eligible'],'A0':counts['train']['instances_role_eligible']},
          'eval_images':counts['eval']['images_decodable'],
          'surface_source':{'admission':'data/phrase2/GEOMETRY_ADMISSION.json','regions':'data/action4_surface_regions_v001/centers.json',
              'label_roots':geometry['label_roots'],'manifest_sha256':release['manifest_sha256']},
          'scope':'data admission only; separate code/integration/throughput gates still required',
          'counts':counts,'data_release_sha256':hashlib.sha256((R/'data/phrase2/DATASET_COMPLETE.json').read_bytes()).hexdigest()}
    (out/'FULL_TRAINING_ADMISSION.json').write_text(json.dumps(gate,indent=2))
    (out/'COUNTS.json').write_text(json.dumps(counts,indent=2))
    print(json.dumps({'status':'PUBLISHED_DATA_CONSUMPTION_PASS','counts':counts}))
    return gate

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=12)
    a=p.parse_args()
    if not 1<=a.workers<=72:raise ValueError('CPU budget')
    prepare(a.output,a.workers)
