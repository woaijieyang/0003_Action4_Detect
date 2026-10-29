"""Fail-closed6500 publication; original5500 record bytes and artifacts preserved."""
from pathlib import Path
import copy,fcntl,hashlib,json,os,shutil,time,ast,importlib
import numpy as np
from data.action4_paths import PROJECT_ROOT as R,contained_path
from data.action4_coverage1000_similarity import new_records,fingerprint,OUT,ROOT,STAGE
from data.action4_coverage1000_consumers import check as consumer_check
from data.action4_coverage1000_interface import check as interface_check
from data.action4_occlusion500_similarity import write
from data import action4_coverage1000_queue as queue

COUNTS={'train':5200,'eval':650,'test':650}

def verify_preserved_artifacts():
    count=0
    for line in (OUT/'OLD5500_FILE_IDENTITIES.jsonl').open():
        row=json.loads(line);path=contained_path(R,row['path'],exists=True);s=path.stat()
        if (s.st_ino,s.st_size,s.st_mtime_ns)!=(row['inode'],row['bytes'],row['mtime_ns']):raise ValueError('Original artifact changed: '+row['path'])
        count+=1
    if count!=38500:raise ValueError('Incomplete original file identity set')
    baseline=json.loads((OUT/'BASELINE_IDENTITY.json').read_text())
    for role,entry in baseline['models'].items():
        s=(R/entry['path']).stat()
        if (s.st_ino,s.st_size,s.st_mtime_ns)!=(entry['inode'],entry['bytes'],entry['mtime_ns']):raise ValueError('Default checkpoint changed: '+role)
    return {'old_files_checked':count,'default_model_steps':{r:x['step'] for r,x in baseline['models'].items()},'method':'Manifest-byte preservation plus exact inode/size/mtime; reuse existing SHA proofs without repeating full-library hashing.'}

def record_release_authority(commit):
    current=json.loads((R/'configs/execution/action4_active.json').read_text());authority=json.loads((R/current['active_ref']['path']).read_text())
    if authority.get('status')=='PHRASE2_6500_DELIVERED':return
    freeze=json.loads((R/authority['source_freeze']).read_text());freeze=copy.deepcopy(freeze)
    for name in ['data','audit','similarity','interface','consumers','merge','postprocess','final_review','preview']:
        path='src/data/action4_coverage1000_'+name+'.py';freeze['files'][path]=hashlib.sha256((R/path).read_bytes()).hexdigest()
    for path in ['configs/data/action4_phrase2_consumer_v001.json','src/data/action4_stage2_data.py','src/data/action4_phrase2_dataset.py']:
        freeze['files'][path]=hashlib.sha256((R/path).read_bytes()).hexdigest()
    freeze.update(status='SOURCE_FREEZE_6500_DELIVERED',formal_data_published=True,time_unix=time.time(),dataset_identity=commit['similarity_dataset_identity'],manifest_sha256=commit['manifest_sha256'])
    freeze_path=OUT/'SOURCE_FREEZE_MERGED_v001.json';write(freeze_path,freeze)
    revision='r'+str(int(authority['revision'][1:])+1).zfill(3)
    authority.update(revision=revision,parent_authority=current['active_ref']['path'],status='PHRASE2_6500_DELIVERED',formal_dataset_release_complete=True,
        current_formal_images=6500,current_formal_split_counts=COUNTS,current5500_release_still_usable=False,gates_remaining=[],live_generation_processes=0,target_images=6500,batch_images=1000,batch_split_counts={'train':800,'eval':100,'test':100},split_counts=COUNTS,
        final_split_counts=COUNTS,source_freeze=str(freeze_path.relative_to(R)),consumer_configuration='configs/data/action4_phrase2_consumer_v001.json',
        completion='data/phrase2/DATASET_COMPLETE.json',original5500_release='data/phrase2/releases/base5500_v001/DATASET_COMPLETE.json',train_enabled=False)
    p=R/f'configs/execution/action4_execution_{revision}.json';p.open('x').write(json.dumps(authority,ensure_ascii=False,indent=2))
    if json.loads((R/'configs/execution/action4_active.json').read_text())!=current:raise ValueError('Concurrent authority writer; preserve its state')
    write(R/'configs/execution/action4_active.json',{'schema':'action4.active_execution_pointer.v1','active_ref':{'path':str(p.relative_to(R)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}})

def merge():
    if Path.cwd().resolve()!=R:raise ValueError('Project root required')
    lock=(OUT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'MERGE_COMPLETE.json').exists():return json.loads((OUT/'MERGE_COMPLETE.json').read_text())
    if queue.counts()!={'accepted':1000}:raise ValueError('Wait for every actual accepted slot')
    records=new_records();audit=json.loads((OUT/'FINAL_AUDIT.json').read_text());proof=json.loads((OUT/'SIMILARITY_COMPLETE.json').read_text());review=json.loads((OUT/'ROOT_FINAL_VISUAL_REVIEW.json').read_text())
    interface=json.loads((OUT/'DATA_INTERFACE_VALIDATION.json').read_text());cfg=json.loads((R/'configs/data/action4_coverage1000_v001.json').read_text())
    if (audit['status']!='PASS_ACTUAL1000_QUOTA_FILE_AND_HOST_GEOMETRY_CLOSURE' or audit['batch_identity']!=fingerprint(records)
        or proof['status']!='PASS_GLOBAL_SIMILARITY_AUDIT' or proof['images']!=6500 or proof['dataset_identity']!=audit['global_identity']
        or interface['status']!='PASS_ACTUAL_DATA_BATCHES_NO_TRAINING' or review.get('status')!='PASS_FINAL1000_VISUAL_REVIEW'
        or review.get('batch_identity')!=fingerprint(records) or review.get('global_identity')!=proof['dataset_identity']):raise ValueError('Publication admission identity/status mismatch')
    baseline=json.loads((OUT/'BASELINE_IDENTITY.json').read_text());backup=ROOT/'releases/base5500_v001';complete=ROOT/'DATASET_COMPLETE.json'
    # Idempotent recovery after the final commit, without appending again.
    if complete.exists() and json.loads(complete.read_text()).get('images')==6500:
        done=json.loads(complete.read_text())
        if done['status']!='FORMAL_DATASET_USABLE' or done['similarity_dataset_identity']!=proof['dataset_identity']:raise ValueError('A different6500 release exists')
        preserved=verify_preserved_artifacts();record_release_authority(done)
        result={'status':'FORMAL6500_DATASET_USABLE','images':6500,'targets':13000,'fixed_id_corners':104000,'splits':COUNTS,'preservation':preserved,'no_training_started':True,'recovered_after_commit':True}
        write(OUT/'MERGE_COMPLETE.json',result);return result
    if not backup.exists():
        if json.loads(complete.read_text())!=baseline['dataset_complete']:raise ValueError('Original release changed')
        backup.mkdir(parents=True,exist_ok=False)
        for name in ['DATASET_COMPLETE.json','GEOMETRY_ADMISSION.json']:shutil.copyfile(ROOT/name,backup/name)
        for split in COUNTS:
            p=ROOT/split/'annotations.jsonl'
            if hashlib.sha256(p.read_bytes()).hexdigest()!=baseline['manifests'][split]['sha256']:raise ValueError('Original manifest changed')
            shutil.copyfile(p,backup/(split+'_annotations.jsonl'))
        shutil.copyfile(R/'configs/data/action4_phrase2_consumer_v001.json',backup/'consumer_configuration.json')
    old=[];oldbytes={}
    for split in COUNTS:
        data=(backup/(split+'_annotations.jsonl')).read_bytes()
        if hashlib.sha256(data).hexdigest()!=baseline['manifests'][split]['sha256']:raise ValueError('Original release snapshot changed')
        oldbytes[split]=data;old.extend(json.loads(line) for line in data.splitlines())
    if fingerprint(old+records)!=proof['dataset_identity']:raise ValueError('Global proof does not bind unchanged old5500')
    preserved=verify_preserved_artifacts();converted=[];links=[]
    def link(name,split):
        src=contained_path(R,name,exists=True)
        if not src.is_relative_to(STAGE/split) or src.is_symlink():raise ValueError('Not an owned new artifact')
        suffix=src.relative_to(STAGE/split)
        if suffix.parts[0] not in ['images','masks','geometry']:raise ValueError('Wrong final artifact directory')
        dst=contained_path(R,ROOT/split/suffix,exists=False);dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():
            if not os.path.samefile(src,dst):raise FileExistsError('Never overwrite a different existing file')
        else:os.link(src,dst)
        links.append({'from':name,'to':str(dst.relative_to(R)),'bytes':src.stat().st_size,'inode':src.stat().st_ino});return str(dst.relative_to(R))
    for original in records:
        row=copy.deepcopy(original);split=row['split'];row['image']=link(row['image'],split)
        for obj in row['instances']:
            box=np.asarray(obj['bbox_full_xyxy']);T=np.asarray(obj['canonical8_geometry']['object_to_capture_cv_m']);t=T[:3,3]
            obj.update(bbox_center_uv=((box[:2]+box[2:])/(2*np.array([row['width'],row['height']]))).tolist(),
                actual_width_fraction=float((box[2]-box[0])/row['width']),optical_Z_m=float(t[2]),euclidean_CAD_origin_distance_m=float(np.linalg.norm(t)))
            for key in ['mask_visible','mask_amodal_expanded']:obj[key]=link(obj[key],split)
            obj['surface_supervision']['path']=link(obj['surface_supervision']['path'],split)
        row.update(batch_admission='coverage1000_reviewed_and_merged',dataset_release='phrase2_6500_v001',
            transform_authority='instances.canonical8_geometry.object_to_capture_cv_m is actual exported GT; actor_pose_capture_by_side retains the checked reproducible control proposal.')
        converted.append(row)
    if len(links)!=7000 or fingerprint(old+converted)!=proof['dataset_identity']:raise ValueError('New artifact/identity conversion failed')
    temporary={}
    for split,count in COUNTS.items():
        rows=[r for r in converted if r['split']==split];p=ROOT/split/'annotations.jsonl.coverage6500_tmp'
        data=oldbytes[split]+(b'' if oldbytes[split].endswith(b'\n') else b'\n')+b''.join((json.dumps(r,allow_nan=False)+'\n').encode() for r in rows)
        if not data.startswith(oldbytes[split]) or len(data.splitlines())!=count:raise ValueError('Original records were rewritten or final count differs')
        p.write_bytes(data);temporary[split]=p
    first_gate=consumer_check()
    if first_gate['status']!='PASS_NO_ACTIVE_OLD_MANIFEST_TRAIN_OR_EVAL_CONSUMERS':return {'status':'PUBLICATION_BLOCKED_KEEP5500','consumer_gate':first_gate,'candidate1000_preserved':True}
    blocked=OUT/'MERGE_IN_PROGRESS_BASE5500_COMPLETE.json'
    if complete.exists():
        if json.loads(complete.read_text())!=baseline['dataset_complete']:raise ValueError('Unexpected current release')
        if not blocked.exists():os.rename(complete,blocked)
        else:complete.unlink()
    # Close the new-consumer window before touching a single manifest.
    second_gate=consumer_check()
    if second_gate['status']!='PASS_NO_ACTIVE_OLD_MANIFEST_TRAIN_OR_EVAL_CONSUMERS':
        shutil.copyfile(backup/'DATASET_COMPLETE.json',complete)
        return {'status':'PUBLICATION_BLOCKED_KEEP5500','consumer_gate':second_gate,'candidate1000_preserved':True}
    write(OUT/'MERGE_TRANSACTION.json',{'status':'ENTRY_CLOSED_LINKS_PREPARED','global_identity':proof['dataset_identity'],'links':links,'old_manifest_byte_prefix_preserved':True,'split_counts':COUNTS})
    for split,p in temporary.items():os.replace(p,ROOT/split/'annotations.jsonl')
    manifests={s:hashlib.sha256((ROOT/s/'annotations.jsonl').read_bytes()).hexdigest() for s in COUNTS}
    write(ROOT/'GEOMETRY_ADMISSION.json',{'candidate_training_allowed':True,'dataset_root':'data/phrase2','manifest_sha256':manifests,
        'label_roots':{s:f'data/phrase2/{s}/geometry' for s in COUNTS},'version':'phrase2_6500_v001','images':6500,'targets':13000,
        'base_release':'data/phrase2/releases/base5500_v001/DATASET_COMPLETE.json','additional_audit':'outputs/coverage1000_v001/FINAL_AUDIT.json','visual_review':'outputs/coverage1000_v001/ROOT_FINAL_VISUAL_REVIEW.json'})
    consumer=json.loads((backup/'consumer_configuration.json').read_text());consumer['expected_counts']={s:{'images':n,'targets':2*n} for s,n in COUNTS.items()};consumer['training_enabled']=False
    consumer['expected_A0_units']={s:2*n for s,n in COUNTS.items()};consumer['dataset_release']='phrase2_6500_v001'
    write(R/'configs/data/action4_phrase2_consumer_v001.json',consumer)
    loader=R/'src/data/action4_phrase2_dataset.py';code=loader.read_text();needle='counts=admission["splits"] if require_complete else {"train":4000,"eval":500,"test":500}'
    if needle in code:
        source_backup=OUT/'source_snapshots/pre6500_loader';source_backup.mkdir(parents=True,exist_ok=True)
        if not(source_backup/loader.name).exists():shutil.copyfile(loader,source_backup/loader.name)
        code=code.replace(needle,'counts=admission["splits"] if require_complete else {split:len(self.records)}');ast.parse(code);loader.write_text(code)
        from data import action4_phrase2_dataset as dataset_module
        importlib.reload(dataset_module)
    # Same production factory, explicit unpublished integration; completion
    # marker remains absent until actual RGB/ROI/mask/XYZ batches pass.
    interface_check('formal',True)
    preserved_after=verify_preserved_artifacts()
    for split in COUNTS:
        if not (ROOT/split/'annotations.jsonl').read_bytes().startswith(oldbytes[split]):raise ValueError('Old annotation bytes changed')
    commit={'status':'FORMAL_DATASET_USABLE','images':6500,'targets':13000,'fixed_id_corners':104000,'splits':COUNTS,'manifest_sha256':manifests,
        'similarity_proof':'outputs/coverage1000_v001/SIMILARITY_COMPLETE.json','similarity_dataset_identity':proof['dataset_identity'],
        'base_release':'data/phrase2/releases/base5500_v001/DATASET_COMPLETE.json','additional_batch':'coverage1000_v001','training_started':False,
        'configuration_sha256':audit['configuration_sha256'],'audit':'outputs/coverage1000_v001/FINAL_AUDIT.json','protocol_id':cfg['protocol_id'],
        'actual_interface_validation':'outputs/coverage1000_v001/FORMAL_ROOT_INTEGRATION_VALIDATION.json'}
    write(complete,commit)
    try:
        from data.action4_phrase2_dataset import Phrase2Dataset
        for split,n in COUNTS.items():
            for role,multiplier in [('A0',2),('B0',1)]:
                data=Phrase2Dataset(split,role=role)
                if len(data)!=n*multiplier:raise ValueError('Formal factory counts differ after commit')
    except BaseException:
        os.rename(complete,OUT/'FAILED_POSTMERGE_COMPLETION.json');raise
    record_release_authority(commit)
    result={'status':'FORMAL6500_DATASET_USABLE','images':6500,'targets':13000,'fixed_id_corners':104000,'splits':COUNTS,
        'new_images':1000,'old5500_annotation_byte_prefix_preserved':True,'preservation':preserved_after,'global_similarity_identity':proof['dataset_identity'],
        'manifest_sha256':manifests,'no_training_started':True,'time_unix':time.time()}
    write(OUT/'MERGE_COMPLETE.json',result);return result

if __name__=='__main__':print(json.dumps(merge(),indent=2))
