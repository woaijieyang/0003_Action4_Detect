"""6500-image duplicate audit reusing both original feature caches and policy."""
from pathlib import Path
import argparse,collections,concurrent.futures,fcntl,hashlib,json,os,time
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from data.action4_paths import PROJECT_ROOT as R,contained_path
from data import action4_coverage1000_queue as queue
from data.action4_occlusion500_similarity import fingerprint,descriptor,geometry_pair,ssim,write

OUT=R/'outputs/coverage1000_v001';ROOT=R/'data/phrase2';STAGE=ROOT/'.staging_batches/coverage1000_v001'
CONFIG=R/'configs/data/action4_phrase2_similarity_v001.json'

def base_records():
    baseline=json.loads((OUT/'BASELINE_IDENTITY.json').read_text());commit=json.loads((ROOT/'DATASET_COMPLETE.json').read_text())
    if commit!=baseline['dataset_complete'] or commit['images']!=5500:raise ValueError('Baseline5500 release changed; never overwrite a newer release')
    records=[]
    for split,count in [('train',4400),('eval',550),('test',550)]:
        path=ROOT/split/'annotations.jsonl'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=commit['manifest_sha256'][split]:raise ValueError('Original manifest changed')
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        if len(rows)!=count or any(r['split']!=split for r in rows):raise ValueError('Baseline split count differs')
        records.extend(rows)
    return records,commit

def new_records():
    db=queue.connect();rows=db.execute("SELECT record FROM slots WHERE status='accepted' ORDER BY slot").fetchall();db.close()
    result=[json.loads(r[0]) for r in rows]
    if any(r.get('batch_id')!='coverage1000_v001' or r['slot_id']<5500 for r in result):raise ValueError('Wrong extension namespace')
    return result

def cache_path(row):
    root=R/'outputs/phrase2_generation_v001' if row['slot_id']<5000 else R/'outputs/phrase2_occlusion500_v001' if row['slot_id']<5500 else OUT
    return root/'similarity/features'/f"{row['sample_id']}_a{row['attempt']}_{row['rgb_sha256'][:16]}.npz"

def features(row):
    target=cache_path(row)
    if target.exists():return str(target)
    if row['slot_id']<5500:raise ValueError('Expected original5500 cached feature missing; diagnose instead of silently recomputing the old library')
    image=contained_path(R,row['image'],exists=True)
    if not image.is_relative_to(STAGE/row['split']) or image.is_symlink():raise ValueError('Unowned feature input')
    with Image.open(image) as im:
        rgb=im.convert('RGB');full=np.asarray(rgb.resize((128,96),Image.Resampling.LANCZOS));local=[]
        for obj in row['instances']:
            box=np.array(obj['bbox_full_xyxy']);pad=(box[2:]-box[:2])*.15
            crop=np.r_[np.maximum(0,box[:2]-pad),np.minimum([row['width'],row['height']],box[2:]+pad)]
            local.append(np.asarray(rgb.crop(tuple(crop)).resize((96,64),Image.Resampling.LANCZOS)))
    target.parent.mkdir(parents=True,exist_ok=True);temp=target.with_name(target.name+f'.{os.getpid()}.tmp')
    with temp.open('xb') as f:np.savez_compressed(f,full=full,local=np.stack(local))
    os.replace(temp,target);return str(target)

def compare(a,b,cfg):
    ok,metrics=geometry_pair(a,b,cfg)
    if a['rgb_sha256']==b['rgb_sha256']:
        if not ok:raise ValueError('Identical RGB contradicts geometry; do not conceal a binding failure by culling')
        return True,dict(metrics,reason='exact_rgb',body_ssim=[1.,1.],full_ssim=1.)
    if not ok:return False,metrics
    with np.load(cache_path(a)) as fa,np.load(cache_path(b)) as fb:
        scores=[ssim(x,y) for x,y in zip(fa['local'],fb['local'])];whole=ssim(fa['full'],fb['full'])
    metrics.update(body_ssim=scores,full_ssim=whole,reason='near_duplicate_geometry_and_both_body_appearance')
    return min(scores)>=cfg['minimum_body_ssim'] and whole>=cfg['minimum_full_ssim'],metrics

def audit(workers=64):
    if not 1<=workers<=64:raise ValueError('Bound feature CPU workers to64')
    if queue.counts()!={'accepted':1000}:raise ValueError('All1000 accepted and generation quiescent required')
    old,commit=base_records();new=new_records();records=old+new;cfg=json.loads(CONFIG.read_text())
    proof=json.loads(contained_path(R,commit['similarity_proof'],exists=True).read_text());config_sha=hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    if proof['status']!='PASS_GLOBAL_SIMILARITY_AUDIT' or proof['dataset_identity']!=fingerprint(old) or proof['configuration_sha256']!=config_sha:raise ValueError('Old5500 proof does not match the same current policy')
    if len(records)!=6500 or len({r['sample_id'] for r in records})!=6500:raise ValueError('Global count/identity differs')
    start=time.time();cached=sum(cache_path(r).exists() for r in records)
    # Original features are read by name, never decoded or rehashed from RGB.
    if any(not cache_path(r).exists() for r in old):raise ValueError('Original5500 feature cache incomplete')
    with concurrent.futures.ProcessPoolExecutor(workers) as pool:list(pool.map(features,new,chunksize=4))
    centers=np.array([descriptor(r)[0] for r in records]);pairs=cKDTree(centers).query_pairs(cfg['max_center_delta_uv'],p=np.inf,output_type='set')
    hashes=collections.defaultdict(list)
    for i,r in enumerate(records):hashes[r['rgb_sha256']].append(i)
    for indices in hashes.values():
        for j in indices[1:]:pairs.add((indices[0],j))
    preview=set(json.loads((OUT/'SLOTS.json').read_text())['preview_slot_ids'])
    order=sorted(range(6500),key=lambda i:(records[i]['slot_id']>=5500,records[i]['slot_id'] not in preview,records[i]['slot_id']))
    rank={i:n for n,i in enumerate(order)};neighbors=collections.defaultdict(list);reused_old_pairs=0
    for a,b in pairs:
        if records[a]['slot_id']<5500 and records[b]['slot_id']<5500:reused_old_pairs+=1;continue
        if rank[a]>rank[b]:a,b=b,a
        neighbors[a].append(b)
    rejected={};tested=geometry=0;examples=[]
    for a in order:
        if a in rejected:continue
        for b in sorted(neighbors[a],key=rank.get):
            if b in rejected:continue
            tested+=1;duplicate,metrics=compare(records[a],records[b],cfg);geometry+=int('body_ssim' in metrics)
            if duplicate:
                if records[b]['slot_id']<5500:raise ValueError('Original5500 may not be replaced')
                rejected[b]={'slot_id':records[b]['slot_id'],'sample_id':records[b]['sample_id'],'attempt':records[b]['attempt'],
                    'kept_slot_id':records[a]['slot_id'],'kept_sample_id':records[a]['sample_id'],'cross_split':records[a]['split']!=records[b]['split'],'metrics':metrics}
            if 'body_ssim' in metrics and len(examples)<40:examples.append({'a':records[a]['sample_id'],'b':records[b]['sample_id'],'duplicate':duplicate,'metrics':metrics})
    result={'status':'RESAMPLE_REQUIRED' if rejected else 'PASS_GLOBAL_SIMILARITY_AUDIT','images':6500,'dataset_identity':fingerprint(records),
        'configuration_sha256':config_sha,'config':cfg,'base5500_proof':commit['similarity_proof'],'base5500_identity':fingerprint(old),
        'base5500_old_old_pairs_reused':reused_old_pairs,'original_features_reused':5500,'features_cached_before':cached,'new_features_computed':6500-cached,
        'candidate_pairs':len(pairs),'tested_pairs':tested,'geometry_pass_pairs':geometry,'exact_duplicate_groups':sum(len(a)>1 for a in hashes.values()),
        'rejected':list(rejected.values()),'rejected_cross_split':sum(r['cross_split'] for r in rejected.values()),'examples':examples,'seconds':time.time()-start,
        'scope':'All6500 within and across splits under the same frozen policy; original5500 pair proof/cache reused; all new-old and new-new indexed candidates checked.',
        'limitation':'A policy-bound near-duplicate check, distinct from distribution coverage. No all-pairs RGB comparison or repeat full-library hashing.'}
    write(OUT/'similarity'/f'AUDIT_{int(time.time())}_{result["dataset_identity"][:12]}.json',result);write(OUT/'SIMILARITY_CURRENT.json',result)
    if not rejected:write(OUT/'SIMILARITY_COMPLETE.json',result)
    return result

def cull(report):
    guard=(OUT/'controller.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    counts=queue.counts()
    if set(counts)-{'accepted','pending','revoking_similarity'}:raise ValueError('Culling requires quiescent owned generation')
    folder=OUT/'similarity'/('replacements_'+report['dataset_identity'][:16]);receipt=folder/'TRANSACTION.json'
    if (folder/'COMPLETE.json').exists():return json.loads((folder/'COMPLETE.json').read_text())
    if not receipt.exists():
        if counts!={'accepted':1000}:raise ValueError('A new cull transaction requires all1000')
        old,_=base_records();new=new_records()
        if fingerprint(old+new)!=report['dataset_identity']:raise ValueError('Stale similarity report')
        byid={r['slot_id']:r for r in new};folder.mkdir(parents=True,exist_ok=False);operations=[]
        for reject in report['rejected']:
            row=byid[reject['slot_id']]
            if row['attempt']!=reject['attempt']:raise ValueError('Replacement attempt changed')
            names=[row['image']]+[name for obj in row['instances'] for name in [obj['mask_visible'],obj['mask_amodal_expanded'],obj['surface_supervision']['path']]]
            if len(set(names))!=7:raise ValueError('Wrong artifact set')
            moves=[]
            for name in names:
                src=contained_path(R,name,exists=True)
                if not src.is_relative_to(STAGE/row['split']) or src.is_symlink():raise ValueError('Only new owned files may be quarantined')
                dst=folder/'files'/src.relative_to(STAGE);moves.append({'from':name,'to':str(dst.relative_to(R)),'bytes':src.stat().st_size})
            operations.append({'record':row,'rejection':reject,'moves':moves})
        write(receipt,{'status':'PREPARED','dataset_identity':report['dataset_identity'],'operations':operations,'old5500_untouched':True})
    transaction=json.loads(receipt.read_text());operations=transaction['operations'];db=queue.connect()
    if transaction['dataset_identity']!=report['dataset_identity']:raise ValueError('Wrong recovery transaction')
    db.execute('BEGIN IMMEDIATE')
    for op in operations:
        row=db.execute('SELECT status,record FROM slots WHERE slot=?',(op['record']['slot_id'],)).fetchone()
        if row['status']=='pending' and row['record'] is None:continue
        if row['status'] not in ['accepted','revoking_similarity'] or json.loads(row['record'])['rgb_sha256']!=op['record']['rgb_sha256']:raise ValueError('Recovery identity differs')
        db.execute("UPDATE slots SET status='revoking_similarity',enabled=0,owner=NULL,updated=? WHERE slot=?",(time.time(),op['record']['slot_id']))
    db.execute('COMMIT');queue.publish_manifests()
    for op in operations:
        for move in op['moves']:
            src=contained_path(R,move['from'],exists=False);dst=contained_path(R,move['to'],exists=False)
            if src.exists():
                if dst.exists() or src.stat().st_size!=move['bytes']:raise FileExistsError('Never overwrite a quarantined identity')
                dst.parent.mkdir(parents=True,exist_ok=True);os.rename(src,dst)
            elif not dst.exists() or dst.stat().st_size!=move['bytes']:raise ValueError('Incomplete quarantine recovery')
        db.execute("UPDATE slots SET status='pending',record=NULL,error=?,enabled=1,updated=? WHERE slot=? AND status='revoking_similarity'",('duplicate_replacement',time.time(),op['record']['slot_id']))
    db.close();queue.publish_manifests();preview=set(json.loads((OUT/'SLOTS.json').read_text())['preview_slot_ids']);replaced=[op['record']['slot_id'] for op in operations if op['record']['slot_id'] in preview]
    if replaced:
        prior=OUT/'PREVIEW_REPLACEMENTS_REQUIRED.json';previous=json.loads(prior.read_text())['slots'] if prior.exists() else []
        write(prior,{'slots':sorted(set(previous+replaced)),'requires':'Actual root review of replaced previews; preserve original preview gate.'})
    write(folder/'COMPLETE.json',{'status':'SAME_SLOTS_REQUEUED','count':len(operations),'quotas_and_attempts_preserved':True,'old5500_untouched':True})
    return {'count':len(operations),'next':'Generate replacement attempts only; never rerender old5500 or other accepted new slots.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=64);p.add_argument('--cull',action='store_true');a=p.parse_args()
    result=cull(json.loads((OUT/'SIMILARITY_CURRENT.json').read_text())) if a.cull else audit(a.workers)
    print(json.dumps({k:v for k,v in result.items() if k not in ['examples','config','rejected']},indent=2))
