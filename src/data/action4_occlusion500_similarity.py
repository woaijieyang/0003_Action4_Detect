"""Cached, foreground-aware split-global duplicate audit and reversible culling.

No model scores, no O(N squared) RGB comparisons, no repeated RGB hashing.
Only terminal owned-generation queues may be culled. A slot's quota and random
attempt counter survive replacement unchanged.
"""
import argparse, collections, concurrent.futures, hashlib, json, os, time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from scipy.ndimage import uniform_filter
from data.action4_paths import PROJECT_ROOT as R, contained_path
from data import action4_occlusion500_queue as queue

OUT=R/'outputs/phrase2_occlusion500_v001'; ROOT=R/'data/phrase2'
STAGING_ROOT=ROOT/'.staging_batches/occlusion500_v001'
CONFIG=R/'configs/data/action4_phrase2_similarity_v001.json'

def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(t,p)

def additional_records():
    db=queue.connect();rows=db.execute("SELECT slot,record FROM slots WHERE status='accepted' ORDER BY slot").fetchall();db.close()
    return [json.loads(r['record']) for r in rows]

def fingerprint(records):
    rows=sorted((r['split'],r['sample_id'],int(r['attempt']),r['rgb_sha256']) for r in records)
    return hashlib.sha256(json.dumps(rows,separators=(',',':')).encode()).hexdigest()

def cache_path(r):
    directory=R/'outputs/phrase2_generation_v001' if r['slot_id']<5000 else OUT
    return directory/'similarity/features'/f"{r['sample_id']}_a{r['attempt']}_{r['rgb_sha256'][:16]}.npz"

def features(r):
    p=cache_path(r)
    if p.exists():return str(p)
    image=contained_path(R,r['image'],exists=True)
    if not (image.is_relative_to(ROOT/r['split']) or (r.get('batch_id')=='occlusion500_v001' and image.is_relative_to(STAGING_ROOT/r['split']))) or image.is_symlink():raise ValueError('Unowned similarity input')
    # Read full image exactly once for a new identity. All later comparisons
    # operate on these small cached arrays and the already bound GT.
    with Image.open(image) as im:
        rgb=im.convert('RGB');full=np.asarray(rgb.resize((128,96),Image.Resampling.LANCZOS))
        local=[]
        for g in r['instances']:
            box=np.array(g['bbox_full_xyxy'],dtype=float);pad=(box[2:]-box[:2])*.15
            crop=np.r_[np.maximum(0,box[:2]-pad),np.minimum([r['width'],r['height']],box[2:]+pad)]
            local.append(np.asarray(rgb.crop(tuple(crop)).resize((96,64),Image.Resampling.LANCZOS)))
    p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    with temp.open('xb') as f:np.savez_compressed(f,full=full,local=np.stack(local))
    os.replace(temp,p);return str(p)

def ssim(a,b):
    """Standard uniform-window SSIM, each RGB channel included in the mean."""
    x=np.asarray(a,dtype=np.float64)/255;y=np.asarray(b,dtype=np.float64)/255
    size=(7,7,1);mx=uniform_filter(x,size);my=uniform_filter(y,size)
    vx=uniform_filter(x*x,size)-mx*mx;vy=uniform_filter(y*y,size)-my*my
    cov=uniform_filter(x*y,size)-mx*my
    result=((2*mx*my+.01**2)*(2*cov+.03**2))/((mx*mx+my*my+.01**2)*(vx+vy+.03**2))
    return float(result[3:-3,3:-3].mean())

def descriptor(r):
    scale=np.array([r['width'],r['height']]);boxes=np.array([o['bbox_full_xyxy'] for o in r['instances']])/np.tile(scale,2)
    centers=(boxes[:,:2]+boxes[:,2:])/2;wh=boxes[:,2:]-boxes[:,:2]
    corners=np.array([o['corners_px'] for o in r['instances']])/scale
    shape=(corners-centers[:,None,:])/wh[:,None,:]
    return centers.ravel(),wh,shape,np.array([o['occlusion_inframe'] for o in r['instances']])

def geometry_pair(a,b,cfg):
    ca,wa,sa,oa=descriptor(a);cb,wb,sb,ob=descriptor(b)
    metrics={'center_linf':float(np.max(np.abs(ca-cb))), 'relative_size_max':float(np.max(np.abs(wa-wb)/np.maximum(wa,wb))),
             'corner_shape_rms':float(np.sqrt(np.mean((sa-sb)**2))), 'occlusion_difference_max':float(np.max(np.abs(oa-ob)))}
    ok=(metrics['center_linf']<=cfg['max_center_delta_uv'] and metrics['relative_size_max']<=cfg['max_relative_size_difference'] and
        metrics['corner_shape_rms']<=cfg['max_normalized_corner_rms'] and metrics['occlusion_difference_max']<=cfg['max_occlusion_difference'])
    return ok,metrics

def compare(a,b,cfg):
    ok,m=geometry_pair(a,b,cfg)
    exact=a['rgb_sha256']==b['rgb_sha256']
    if exact:
        # Identical RGB with contradictory GT is a binding failure, not merely
        # a duplicate to silently discard.
        if not ok:raise ValueError(('Identical RGB but conflicting geometry',a['sample_id'],b['sample_id']))
        return True,dict(m,reason='exact_rgb',full_ssim=1.,body_ssim=[1.,1.])
    if not ok:return False,m
    with np.load(cache_path(a)) as fa,np.load(cache_path(b)) as fb:
        scores=[ssim(x,y) for x,y in zip(fa['local'],fb['local'])]
        whole=ssim(fa['full'],fb['full'])
    m.update(full_ssim=whole,body_ssim=scores,reason='near_duplicate_geometry_and_both_body_appearance')
    return min(scores)>=cfg['minimum_body_ssim'] and whole>=cfg['minimum_full_ssim'],m

def audit(workers=1,partial=False):
    cfg=json.loads(CONFIG.read_text());records=accepted()
    if not partial and len(records)!=5500:raise ValueError(('Wait for base5000 and additional500 accepted slots',len(records)))
    identity=fingerprint(records);started=time.time();cached=sum(cache_path(r).exists() for r in records)
    # Feature extraction is bounded by CPU affinity and scheduled after renders
    # finish for full runs. Partial diagnostics explicitly use one shared core.
    if workers>1:
        with concurrent.futures.ProcessPoolExecutor(workers) as pool:list(pool.map(features,records,chunksize=4))
    else:
        for r in records:features(r)
    centers=np.array([descriptor(r)[0] for r in records]);tree=cKDTree(centers)
    pairs=tree.query_pairs(cfg['max_center_delta_uv'],p=np.inf,output_type='set')
    exact=collections.defaultdict(list)
    for i,r in enumerate(records):exact[r['rgb_sha256']].append(i)
    for indices in exact.values():
        # Add only a star for exact groups; all are removed against the stable
        # keeper. No quadratic expansion even for a broken repeated-image run.
        for j in indices[1:]:pairs.add((indices[0],j))
    plan=json.loads((OUT/'SLOTS.json').read_text());preview=set(plan['preview_slot_ids'])
    order=sorted(range(len(records)),key=lambda i:(records[i]['slot_id']>=5000,records[i]['slot_id'] not in preview,records[i]['slot_id']))
    rank={i:n for n,i in enumerate(order)};neighbors=collections.defaultdict(list)
    for a,b in pairs:
        if rank[a]>rank[b]:a,b=b,a
        neighbors[a].append(b)
    rejected={};kept=[];examples=[];tested=0;geometry_candidates=0
    for a in order:
        if a in rejected:continue
        kept.append(a)
        for b in sorted(neighbors[a],key=rank.get):
            if b in rejected:continue
            tested+=1;ok,metrics=compare(records[a],records[b],cfg)
            geometry_candidates+=int('body_ssim' in metrics)
            if ok:
                rejected[b]={'slot_id':records[b]['slot_id'],'sample_id':records[b]['sample_id'],'attempt':records[b]['attempt'],
                             'kept_slot_id':records[a]['slot_id'],'kept_sample_id':records[a]['sample_id'],'cross_split':records[a]['split']!=records[b]['split'],'metrics':metrics}
            if 'body_ssim' in metrics and len(examples)<32:examples.append({'a':records[a]['sample_id'],'b':records[b]['sample_id'],'rejected':ok,'metrics':metrics})
    if any(records[i]['slot_id']<5000 for i in rejected):raise ValueError('A base5000 duplicate contradicts its frozen global proof; preserve base and diagnose')
    report={'status':'PARTIAL_AUDIT_ONLY' if partial else ('RESAMPLE_REQUIRED' if rejected else 'PASS_GLOBAL_SIMILARITY_AUDIT'),
            'images':len(records),'dataset_identity':identity,'configuration_sha256':hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
            'config':cfg,'candidate_pairs':len(pairs),'tested_pairs':tested,'geometry_pass_pairs':geometry_candidates,'features_cached_before':cached,
            'features_new':len(records)-cached,'seconds':time.time()-started,'rejected':list(rejected.values()),'rejected_cross_split':sum(x['cross_split'] for x in rejected.values()),
            'exact_duplicate_groups':sum(len(v)>1 for v in exact.values()),'examples':examples,'scope':'Within each split and across all three splits; complete RGB identity and indexed geometry+both-body appearance.',
            'limitation':'Near-duplicate is defined by frozen geometric and SSIM tolerances, not proof that every pair differs under every conceivable visual similarity measure.'}
    stamp=f'{int(time.time())}_{identity[:12]}'
    write(OUT/'similarity'/f'AUDIT_{stamp}.json',report);write(OUT/'SIMILARITY_CURRENT.json',report)
    if not partial and not rejected:
        write(OUT/'SIMILARITY_COMPLETE.json',report)
    return report

def cull(report):
    """Quarantine only owned rejected files with crash-recoverable receipts."""
    import fcntl
    guard=(OUT/'controller.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    db=queue.connect();counts=queue.counts()
    if set(counts)-{'accepted','pending','revoking_similarity'}:raise ValueError(('Generation must be quiescent',counts))
    current=accepted()
    # A prior interrupted move is recovered using its immutable batch receipt.
    batch=OUT/'similarity'/f"cull_{report['dataset_identity'][:16]}"
    receipt=batch/'TRANSACTION.json'
    if receipt.exists() and json.loads(receipt.read_text()).get('status')=='QUARANTINED_SAME_SLOTS_PENDING':
        return {'slots_requeued':len(json.loads(receipt.read_text())['operations']),'receipt':str(receipt.relative_to(R)),'already_completed':True}
    if not receipt.exists():
        assert fingerprint(current)==report['dataset_identity']
        batch.mkdir(parents=True,exist_ok=False)
        preview=set(json.loads((OUT/'SLOTS.json').read_text())['preview_slot_ids'])
        by_slot={r['slot_id']:r for r in current};operations=[]
        for rejection in report['rejected']:
            r=by_slot[rejection['slot_id']];assert r['attempt']==rejection['attempt']
            paths=[r['image']]
            for o in r['instances']:paths.extend([o['mask_visible'],o['mask_amodal_expanded'],o['surface_supervision']['path']])
            assert len(paths)==len(set(paths))==7
            moves=[]
            for name in paths:
                src=contained_path(R,name,exists=True)
                assert r['slot_id']>=5000 and src.is_relative_to(STAGING_ROOT/r['split']) and not src.is_symlink()
                dst=contained_path(R,'archive/legacy/phrase2_occlusion500_similarity_v001/'+report['dataset_identity'][:16]+'/'+name)
                assert not dst.exists();moves.append({'from':name,'to':str(dst.relative_to(R)),'bytes':src.stat().st_size})
            operations.append({'record':r,'rejection':rejection,'moves':moves})
        write(receipt,{'dataset_identity':report['dataset_identity'],'operations':operations,'status':'PREPARED'})
    tx=json.loads(receipt.read_text());db.execute('BEGIN IMMEDIATE')
    for op in tx['operations']:
        slot=op['record']['slot_id'];row=db.execute('SELECT * FROM slots WHERE slot=?',(slot,)).fetchone()
        if row['status']=='pending' and row['record'] is None:continue
        assert row['status'] in ('accepted','revoking_similarity') and json.loads(row['record'])['rgb_sha256']==op['record']['rgb_sha256']
        db.execute("UPDATE slots SET status='revoking_similarity',enabled=0,owner=NULL,updated=? WHERE slot=?",(time.time(),slot))
    db.execute('COMMIT');queue.publish_manifests()
    for op in tx['operations']:
        for move in op['moves']:
            src=contained_path(R,move['from']);dst=contained_path(R,move['to'])
            if src.exists():
                assert not dst.exists() and src.stat().st_size==move['bytes'];dst.parent.mkdir(parents=True,exist_ok=True);os.rename(src,dst)
            else:assert dst.exists() and dst.stat().st_size==move['bytes']
        db.execute("UPDATE slots SET status='pending',record=NULL,error=?,owner=NULL,enabled=1,updated=? WHERE slot=? AND status='revoking_similarity'",('similarity_replacement:'+str(receipt.relative_to(R)),time.time(),op['record']['slot_id']))
    db.close();queue.publish_manifests();tx['status']='QUARANTINED_SAME_SLOTS_PENDING';write(receipt,tx)
    preview=set(json.loads((OUT/'SLOTS.json').read_text())['preview_slot_ids'])
    replaced=[op['record']['slot_id'] for op in tx['operations'] if op['record']['slot_id'] in preview]
    if replaced:
        prior=OUT/'PREVIEW_REPLACEMENTS_REQUIRED.json'
        slots=set(json.loads(prior.read_text())['slots']) if prior.exists() else set()
        write(prior,{'slots':sorted(slots|set(replaced)),'requires':'Root review of replacement RGB, ROI, corners and masks; original gate kept immutable.'})
    return {'slots_requeued':len(tx['operations']),'receipt':str(receipt.relative_to(R)),'quota_and_attempt_not_reset':True}

def require_proof(records):
    proof=json.loads((OUT/'SIMILARITY_COMPLETE.json').read_text())
    assert proof['status']=='PASS_GLOBAL_SIMILARITY_AUDIT' and proof['images']==5500 and not proof['rejected']
    assert proof['dataset_identity']==fingerprint(records)
    assert proof['configuration_sha256']==hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    return proof

def accepted():
    from data import action4_phrase2_queue as base_queue
    db=base_queue.connect();rows=db.execute("SELECT record FROM slots WHERE status='accepted' ORDER BY slot").fetchall();db.close()
    base=[json.loads(r[0]) for r in rows]
    if len(base)!=5000:raise ValueError('The base batch must be complete before the extended similarity audit')
    proof=json.loads((R/'outputs/phrase2_generation_v001/SIMILARITY_COMPLETE.json').read_text())
    if proof['status']!='PASS_GLOBAL_SIMILARITY_AUDIT' or proof['dataset_identity']!=fingerprint(base):raise ValueError('Original5000 proof changed')
    extra=additional_records()
    if any(r['slot_id']<5000 or r.get('batch_id')!='occlusion500_v001' for r in extra):raise ValueError('Wrong additional batch')
    return base+extra

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=1);p.add_argument('--partial',action='store_true');p.add_argument('--cull',action='store_true');args=p.parse_args()
    if args.workers<1 or args.workers>64:raise ValueError('Feature workers must be 1..64 within total project CPU budget')
    if args.cull:
        assert not args.partial;result=cull(json.loads((OUT/'SIMILARITY_CURRENT.json').read_text()))
    else:result=audit(args.workers,args.partial)
    print(json.dumps({k:v for k,v in result.items() if k not in ['rejected','examples','config']},indent=2))
