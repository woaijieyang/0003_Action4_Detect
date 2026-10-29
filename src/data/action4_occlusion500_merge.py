"""Fail-closed append of separately reviewed500 to the unchanged base5000.

No RGB or mask is rewritten. Preserve the original release and manifests;
publish the new completion marker only after all5500 labels and interfaces pass.
"""
from pathlib import Path
import copy,fcntl,hashlib,json,os,time,shutil
import numpy as np
from data.action4_paths import PROJECT_ROOT as R,contained_path
from data import action4_phrase2_queue as base_queue,action4_occlusion500_queue as batch_queue
from data.action4_occlusion500_similarity import fingerprint,require_proof,accepted,additional_records

OUT=R/'outputs/phrase2_occlusion500_v001';ROOT=R/'data/phrase2';STAGE=ROOT/'.staging_batches/occlusion500_v001'

def write(path,obj):
    t=path.with_name(path.name+'.tmp');t.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');os.replace(t,path)

def record_source_closure():
    pointer=R/'configs/execution/action4_active.json';current=json.loads(pointer.read_text());authority=json.loads((R/current['active_ref']['path']).read_text())
    freeze=json.loads((R/authority['source_freeze']).read_text());freeze=copy.deepcopy(freeze)
    source='src/data/action4_phrase2_dataset.py';freeze['sources'][source]=hashlib.sha256((R/source).read_bytes()).hexdigest()
    freeze['schema']='action4.phrase2_5500_delivered_source_freeze.v1';freeze['parent_execution']=current['active_ref']['path']
    target=OUT/'SOURCE_FREEZE_MERGED_v001.json'
    if not target.exists():target.open('x').write(json.dumps(freeze,indent=2)+'\n')
    else:assert json.loads(target.read_text())['sources'][source]==freeze['sources'][source]
    if authority.get('status')=='PHRASE2_5500_DELIVERED':return
    revision='r'+str(int(authority['revision'][1:])+1).zfill(3)
    authority.update(revision=revision,parent_authority=current['active_ref']['path'],status='PHRASE2_5500_DELIVERED',source_freeze=str(target.relative_to(R)),
        target_images=5500,split_counts={'train':4400,'eval':550,'test':550},formal_dataset_release_complete=True,
        completion='data/phrase2/DATASET_COMPLETE.json',base5000_release='data/phrase2/releases/base5000_v001/DATASET_COMPLETE.json',train_enabled=False)
    path=R/f'configs/execution/action4_execution_{revision}.json';path.open('x').write(json.dumps(authority,indent=2)+'\n')
    assert json.loads(pointer.read_text())==current,'Another authority writer changed the pointer'
    write(pointer,{'schema':'action4.active_execution_pointer.v1','active_ref':{'path':str(path.relative_to(R)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}})

def merge():
    assert Path.cwd().resolve()==R
    lock=(OUT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'MERGE_COMPLETE.json').exists():return json.loads((OUT/'MERGE_COMPLETE.json').read_text())
    assert batch_queue.counts()=={'accepted':500}
    all_records=accepted();similarity=require_proof(all_records)
    audit=json.loads((OUT/'FINAL_AUDIT.json').read_text());interface=json.loads((OUT/'DATA_INTERFACE_VALIDATION.json').read_text())
    assert audit['images']==500 and audit['targets']==1000 and audit['corners']==8000
    assert interface['status']=='PASS_ACTUAL_DATA_BATCHES_NO_TRAINING'
    review=json.loads((OUT/'ROOT_FINAL_VISUAL_REVIEW.json').read_text())
    assert review['decision']=='accept_additional500_and_merge' and review['configuration_sha256']==audit['configuration_sha256']
    assert review['batch_dataset_identity']==fingerprint(additional_records()) and review['global_dataset_identity']==similarity['dataset_identity']
    # Recover a crash after the valid completion marker was committed but before
    # final authority/receipt writing. Never append the same500 a second time.
    if (ROOT/'DATASET_COMPLETE.json').exists():
        done=json.loads((ROOT/'DATASET_COMPLETE.json').read_text())
        if done.get('images')==5500:
            assert done['status']=='FORMAL_DATASET_USABLE' and done['similarity_dataset_identity']==similarity['dataset_identity']
            for split,digest in done['manifest_sha256'].items():assert hashlib.sha256((ROOT/split/'annotations.jsonl').read_bytes()).hexdigest()==digest
            from data.action4_phrase2_dataset import Phrase2Dataset
            for split,n in done['splits'].items():
                for role,m in [('A0',2),('B0',1)]:assert len(Phrase2Dataset(split,role=role))==n*m
            record_source_closure()
            result={'status':'FORMAL5500_DATASET_USABLE','images':5500,'targets':11000,'fixed_id_corners':88000,'splits':done['splits'],
                    'no_training_started':True,'manifest_sha256':done['manifest_sha256'],'recovered_after_completion_commit':True,'time_unix':time.time()}
            write(OUT/'MERGE_COMPLETE.json',result);return result
    backup=ROOT/'releases/base5000_v001'
    if not backup.exists():
        old=json.loads((ROOT/'DATASET_COMPLETE.json').read_text());assert old['status']=='FORMAL_DATASET_USABLE' and old['images']==5000
        backup.mkdir(parents=True,exist_ok=False)
        for name in ['DATASET_COMPLETE.json','GEOMETRY_ADMISSION.json']:shutil.copyfile(ROOT/name,backup/name)
        for split,n in [('train',4000),('eval',500),('test',500)]:
            source=ROOT/split/'annotations.jsonl';assert hashlib.sha256(source.read_bytes()).hexdigest()==old['manifest_sha256'][split]
            shutil.copyfile(source,backup/(split+'_annotations.jsonl'))
        write(backup/'PRESERVATION.json',{'original_images':5000,'files_rewritten':False,'source_release':old,'time_unix':time.time()})
    old=json.loads((backup/'DATASET_COMPLETE.json').read_text());base={}
    for split,n in [('train',4000),('eval',500),('test',500)]:
        path=backup/(split+'_annotations.jsonl');assert hashlib.sha256(path.read_bytes()).hexdigest()==old['manifest_sha256'][split]
        base[split]=[json.loads(line) for line in path.read_text().splitlines()];assert len(base[split])==n
    base_identity=fingerprint([r for records in base.values() for r in records])
    assert base_identity==fingerprint([r for r in all_records if r['slot_id']<5000])
    extra=additional_records();converted=[];moves=[]
    def link(name,split):
        src=contained_path(R,name,exists=True);assert src.is_relative_to(STAGE/split) and not src.is_symlink()
        suffix=src.relative_to(STAGE/split);assert suffix.parts[0] in ['images','masks','geometry']
        dst=contained_path(R,ROOT/split/suffix);dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():assert os.path.samefile(src,dst),'Merge cannot overwrite an existing distinct file'
        else:os.link(src,dst)
        moves.append({'from':name,'to':str(dst.relative_to(R)),'bytes':src.stat().st_size,'kind':'hardlink_same_bytes'})
        return str(dst.relative_to(R))
    for original in extra:
        r=copy.deepcopy(original);split=r['split'];r['image']=link(r['image'],split)
        for obj in r['instances']:
            assert .30<=obj['occlusion_inframe']<=.75 and obj['corner_ids']==list(range(8))
            for key in ['mask_visible','mask_amodal_expanded']:obj[key]=link(obj[key],split)
            obj['surface_supervision']['path']=link(obj['surface_supervision']['path'],split)
        r.update(dataset_root='data/phrase2',batch_admission='additional500_reviewed_and_merged',dataset_release='base5000_plus_occlusion500_v001');converted.append(r)
    assert len(moves)==3500 and fingerprint([r for v in base.values() for r in v]+converted)==similarity['dataset_identity']
    merged={split:base[split]+[r for r in converted if r['split']==split] for split in base}
    counts={'train':4400,'eval':550,'test':550}
    assert {s:len(v) for s,v in merged.items()}==counts
    assert len({r['sample_id'] for rows in merged.values() for r in rows})==5500
    write(OUT/'MERGE_TRANSACTION.json',{'status':'LINKS_PREPARED','base_identity':base_identity,'global_identity':similarity['dataset_identity'],'file_links':moves,'split_counts':counts})
    # Remove the old completion marker before changing any authoritative split.
    # A crash leaves consumers blocked; preserved manifests allow exact recovery.
    complete=ROOT/'DATASET_COMPLETE.json';blocked=OUT/'MERGE_IN_PROGRESS_BASE_COMPLETE.json'
    if complete.exists():
        assert json.loads(complete.read_text())['images']==5000
        if not blocked.exists():os.rename(complete,blocked)
        else:complete.unlink()
    for split,records in merged.items():
        tmp=ROOT/split/'annotations.jsonl.merge_tmp'
        with tmp.open('w') as f:
            for r in records:f.write(json.dumps(r,allow_nan=False)+'\n')
            f.flush();os.fsync(f.fileno())
        os.replace(tmp,ROOT/split/'annotations.jsonl')
    manifests={s:hashlib.sha256((ROOT/s/'annotations.jsonl').read_bytes()).hexdigest() for s in counts}
    write(ROOT/'GEOMETRY_ADMISSION.json',{'candidate_training_allowed':True,'dataset_root':'data/phrase2','manifest_sha256':manifests,
          'label_roots':{s:f'data/phrase2/{s}/geometry' for s in counts},'version':'phrase2_base5000_plus_occlusion500_v001','images':5500,'targets':11000,
          'base_release':'data/phrase2/releases/base5000_v001/DATASET_COMPLETE.json','additional_audit':'outputs/phrase2_occlusion500_v001/FINAL_AUDIT.json','visual_review':'outputs/phrase2_occlusion500_v001/ROOT_FINAL_VISUAL_REVIEW.json'})
    # The formal loader must consume admitted counts, not the old hard-coded map.
    dataset=R/'src/data/action4_phrase2_dataset.py';source=dataset.read_text();needle='counts={"train":4000,"eval":500,"test":500}'
    if needle in source:
        snapshot=OUT/'source_snapshots/pre5500_loader';snapshot.mkdir(parents=True,exist_ok=True)
        if not (snapshot/dataset.name).exists():shutil.copyfile(dataset,snapshot/dataset.name)
        source=source.replace(needle,'counts=admission["splits"] if require_complete else {"train":4000,"eval":500,"test":500}')
        import ast;ast.parse(source);dataset.write_text(source)
    commit={'status':'FORMAL_DATASET_USABLE','images':5500,'targets':11000,'fixed_id_corners':88000,'splits':counts,'manifest_sha256':manifests,
            'similarity_proof':'outputs/phrase2_occlusion500_v001/SIMILARITY_COMPLETE.json','similarity_dataset_identity':similarity['dataset_identity'],
            'base_release':'data/phrase2/releases/base5000_v001/DATASET_COMPLETE.json','additional_batch':'occlusion500_v001','training_started':False,
            'configuration_sha256':audit['configuration_sha256'],'audit':'outputs/phrase2_occlusion500_v001/FINAL_AUDIT.json'}
    write(ROOT/'DATASET_COMPLETE.json',commit)
    try:
        from data.action4_phrase2_dataset import Phrase2Dataset
        for split,n in counts.items():
            for role,m in [('A0',2),('B0',1)]:assert len(Phrase2Dataset(split,role=role))==n*m
    except BaseException:
        os.rename(ROOT/'DATASET_COMPLETE.json',OUT/'FAILED_POSTMERGE_COMPLETION.json');raise
    result={'status':'FORMAL5500_DATASET_USABLE','images':5500,'targets':11000,'fixed_id_corners':88000,'splits':counts,'no_training_started':True,
            'global_similarity_identity':similarity['dataset_identity'],'base5000_record_identity_preserved':True,'manifest_sha256':manifests,'time_unix':time.time()}
    record_source_closure();write(OUT/'MERGE_COMPLETE.json',result);return result

if __name__=='__main__':print(json.dumps(merge(),indent=2))
