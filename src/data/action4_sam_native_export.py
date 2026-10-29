"""Export frozen native SAM predictions without inference or GT selection.

Historical step0 caches stay development-only. Completed-B0 caches preserve the
entire train/eval candidate domain and original RGB; supervision is separate.
"""
import hashlib,json,os,threading
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from collections import OrderedDict
from pathlib import Path
import numpy as np
from PIL import Image
from .action4_paths import PROJECT_ROOT as ROOT,contained_path
from .action4_sam_roi_inputs import ImageFrame,NativeMaskIdentity,NativeSamPrediction,prediction_metadata,array_digest,verify_native_prediction


def sha(path):
    with contained_path(ROOT,path,exists=True).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def read(path):return json.loads(contained_path(ROOT,path,exists=True).read_text())


def checked(binding):
    if set(binding)!={'path','sha256'} or sha(binding['path'])!=binding['sha256']:raise ValueError('Export dependency changed')
    return read(binding['path'])


def verify_export_receipt(index):
    if index.get('sources',{}).get('upstream_mode')==VARIANT_MODE:return verify_variant_export_receipt(index)
    if index.get('sources',{}).get('upstream_mode')==COMPLETED_MODE:return verify_completed_export_receipt(index)
    receipt=checked(index['producer_receipt'])
    payload={k:v for k,v in index.items() if k!='producer_receipt'}
    if receipt.get('schema')!='action4.sam.native_export_receipt.v1' or receipt.get('index_payload_sha256')!=canonical(payload):
        raise ValueError('Export index differs from producer receipt')
    lock=checked(receipt['native_lock']);completion=checked(receipt['native_completion'])
    if lock['identity']!=canonical({k:v for k,v in lock.items() if k!='identity'}):raise ValueError('Native lock identity mismatch')
    if lock['config']['upstream_role']!='frozen_B0_parent_step0_post_soft_nms':raise ValueError('B0-only provenance violated')
    if completion['status']!='NATIVE_TRANSPORT_COMPLETE_MASK_QUALITY_NOT_SCORED' or completion['completed_jobs']!=1298:
        raise ValueError('Native complete evidence missing')
    if index['sources']!=receipt['sources'] or index['sources']['sam_source_sha256']!=lock['source_snapshot_identity']:
        raise ValueError('Source graph differs from native producer')
    if index['sources']['sam_prompt_sha256']!=lock['manifest_sha256']:raise ValueError('Prompt source mismatch')
    if receipt.get('GT_used') is not False or receipt.get('native_mask_selection_changed') is not False:
        raise ValueError('Exporter must not choose masks by GT')
    return receipt


def export(native_root,output,limit=None):
    native=contained_path(ROOT,native_root,exists=True);out=contained_path(ROOT,output,exists=False)
    if out.exists():raise FileExistsError('Preserve existing export')
    lock=read(native/'RUN_LOCK.json');progress=read(native/'PROGRESS.json');cfg=lock['config']
    if cfg.get('schema')==VARIANT_SCHEMA:return export_completed(native_root,output,limit)
    if cfg.get('schema')=='action4.sam.single_image_native.v2':return export_completed(native_root,output,limit)
    if lock['identity']!=canonical({k:v for k,v in lock.items() if k!='identity'}):raise ValueError('Native lock mismatch')
    if cfg['upstream_role']!='frozen_B0_parent_step0_post_soft_nms' or progress['completed_jobs']!=1298 or progress['status']!='NATIVE_TRANSPORT_COMPLETE_MASK_QUALITY_NOT_SCORED':raise ValueError('Completed B0-only source required')
    if sha(cfg['manifest'])!=cfg['manifest_sha256']:raise ValueError('Prompt manifest changed')
    prompts=[json.loads(x) for x in contained_path(ROOT,cfg['manifest'],exists=True).read_text().splitlines()]
    if len(prompts)!=650:raise ValueError('Expected full650 source')
    if limit is not None:
        if type(limit) is not int or not 1<=limit<=650:raise ValueError('Invalid inspection image count')
        prompts=prompts[:limit]
    load=read(native/'NATIVE_LOAD.json')
    if load['status']!='LOADED_ZERO_FINAL_MISSING_UNEXPECTED':raise ValueError('Native weights not verified')
    first=prompts[0]['source_receipt']
    sources={'upstream_mode':'b0_only_native_receipt_v1','b0_model_sha256':first['parent_sha256'],
        'b0_source_sha256':first['project_code_sha256'],'sam_model_sha256':load['weight_sha256_from_prior_validation'],
        'sam_source_sha256':lock['source_snapshot_identity'],'sam_prompt_sha256':cfg['manifest_sha256'],
        'postprocess_sha256':next(v for k,v in lock['code_sha256'].items() if k.endswith('action4_sam_image_eval.py'))}
    out.mkdir(parents=True);(out/'masks').mkdir();images=[];evidence=[];omitted=[]
    for row in prompts:
        if row['source_receipt']['parent_sha256']!=sources['b0_model_sha256'] or row['source_receipt']['project_code_sha256']!=sources['b0_source_sha256']:raise ValueError('Mixed upstream source')
        if sha(row['image_path'])!=row['image_sha256']:raise ValueError('Native RGB changed')
        rgb=np.asarray(Image.open(contained_path(ROOT,row['image_path'],exists=True)).convert('RGB'))
        frame=ImageFrame(row['image_id'],rgb,row['pixel_space_id'],row['image_sha256'],_variant_id(row),array_digest(rgb))
        bindings=[];seen=set()
        for candidate in row['candidates']:
            grid=candidate['identity_binding']['grid_index']
            if type(grid) is not int or grid<0:raise ValueError('Explicit grid identity missing')
            pid=f"b0:{row['image_id']}:grid:{grid}"
            if pid in seen:raise ValueError('Duplicate grid identity')
            seen.add(pid)
            if not candidate['eligible_for_SAM_diagnostic_prompt']:
                omitted.append({'image_id':row['image_id'],'prediction_id':pid,'source_candidate_id':candidate['candidate_id'],'reason':'upstream_not_prompted'});continue
            key=canonical({'lock':lock['identity'],'image_id':row['image_id'],'image_sha256':row['image_sha256'],'candidate':candidate,'text':cfg['native']['text']})
            marker=native/'jobs'/key/'COMPLETED.json';done=read(marker)
            if done['lock_identity']!=lock['identity'] or done['task_key']!=key:raise ValueError('Wrong completed task')
            rp=contained_path(ROOT,done['result_path'],exists=True)
            if sha(rp)!=done['files_sha256'][str(rp.relative_to(ROOT))]:raise ValueError('Native result changed')
            result=read(rp)
            if (result['image_id'],result['candidate_id'],result['pixel_space_id'])!=(row['image_id'],candidate['candidate_id'],frame.pixel_space_id):raise ValueError('Native instance/pixel mismatch')
            if not result['close_success'] or result['backend_kind']!='sam31_official_native':raise ValueError('Incomplete/non-native source')
            mask=None;binding=None;status='missing';reason=result.get('binding_status','unavailable');overlap=0.
            if result['available']:
                choices=[x for x in result['native_candidates'] if x['obj_id']==result['bound_obj_id']]
                if len(choices)!=1:raise ValueError('Native selection not unique')
                selected=choices[0];storage=selected['mask_storage']
                if sha(storage['npz'])!=storage['raw_sha256']:raise ValueError('Raw mask bytes changed')
                with np.load(contained_path(ROOT,storage['npz'],exists=True),allow_pickle=False) as f:mask=f[storage['array_key']][storage['row_index']].squeeze().astype(bool)
                if mask.shape!=rgb.shape[:2] or int(mask.sum())!=selected['mask_pixels']:raise ValueError('Raw mask shape/count mismatch')
                status='available';reason=None;overlap=float(selected['target_mask_coverage'])
                binding={'session_id':result['session_id'],'obj_id':result['bound_obj_id'],'generation':result['generation'],
                    'chosen_mask_id':result['mask_registration']['mask_identity'],'match_basis':result['binding_status']}
            identity=NativeMaskIdentity(frame.image_id,pid,frame.pixel_space_id)
            meta=prediction_metadata(frame,identity,mask,sources,status=status,assigned_instance_id=pid if mask is not None else None,
                candidate_overlap=overlap,reason=reason,sam_binding=binding)
            verify_native_prediction(NativeSamPrediction(meta,mask),frame,identity,sources)
            dest=out/'masks'/(key+'.npz');arrays={'metadata':np.array(json.dumps(meta,sort_keys=True))}
            if mask is not None:arrays['mask']=mask
            np.savez_compressed(dest,**arrays);bindings.append({'prediction_id':pid,'path':str(dest.relative_to(ROOT)),'sha256':sha(dest)})
            evidence.append({'image_id':frame.image_id,'prediction_id':pid,'source_candidate_id':candidate['candidate_id'],'grid_index':grid,
                'prediction_file_sha256':candidate['identity_binding']['prediction_file_sha256'],'completed_marker':str(marker.relative_to(ROOT)),
                'completed_marker_sha256':sha(marker),'result_sha256':sha(rp),'raw_mask_id':None if binding is None else binding['chosen_mask_id']})
        images.append({'image_id':frame.image_id,'image_path':row['image_path'],'source_sha256':frame.source_sha256,
            'pixel_space_id':frame.pixel_space_id,'variant_id':frame.variant_id,'variant_sha256':frame.variant_sha256,'predictions':bindings})
    index={'schema':'action4.sam_native_v2_index.v1','split':'eval','origin':'predicted_sam','prompt_origin':'frozen_prediction_only',
        'scope':'frozen_B0_parent_step0_development_only','sources':sources,'images':images}
    receipt={'schema':'action4.sam.native_export_receipt.v1','index_payload_sha256':canonical(index),'sources':sources,
        'native_lock':{'path':str((native/'RUN_LOCK.json').relative_to(ROOT)),'sha256':sha(native/'RUN_LOCK.json')},
        'native_completion':{'path':str((native/'PROGRESS.json').relative_to(ROOT)),'sha256':sha(native/'PROGRESS.json')},
        'exporter_sha256':sha(__file__),'GT_used':False,'native_mask_selection_changed':False,'geometry_unchanged':True,
        'mapping':evidence,'upstream_unprompted':omitted,'image_count':len(images),'mask_records':len(evidence),'formal_training_admitted':False}
    rp=out/'PRODUCER_RECEIPT.json';rp.write_text(json.dumps(receipt,indent=2)+'\n')
    index['producer_receipt']={'path':str(rp.relative_to(ROOT)),'sha256':sha(rp)};verify_export_receipt(index)
    ip=out/'INDEX.json';ip.write_text(json.dumps(index,indent=2)+'\n')
    spec={'manifest':{'path':str(ip.relative_to(ROOT)),'sha256':sha(ip)},'expected_sources':sources}
    (out/'CACHE_SPEC.json').write_text(json.dumps(spec,indent=2)+'\n')
    return {'status':'EXPORTED_DEVELOPMENT_CACHE_NOT_QUALITY_ACCEPTED','images':len(images),'mask_records':len(evidence),'spec':str((out/'CACHE_SPEC.json').relative_to(ROOT))}


# Completed-B0 cache mode is deliberately separate from the historical step0 exporter.
COMPLETED_MODE='b0_completed_native_receipt_v2'
VARIANT_MODE='b0_completed_native_rgb_variant_receipt_v1'
VARIANT_SCHEMA='action4.sam.single_image_rgb_variant.v1'
VARIANT_SCOPE='frozen_completed_B0_native_rgb_variant_readonly'

def _is_variant(lock):return lock['config'].get('schema')==VARIANT_SCHEMA
def _variant_id(row):
    return row['variant_id'] if row.get('schema')=='action4.sam.prediction_prompt_image.variant.v1' else 'native_original'
def _source_contract(cfg):
    return _v2_checked(cfg['prepared_variant'])['source_receipt'] if cfg.get('schema')==VARIANT_SCHEMA else cfg['source_contract']
def _receipt_schema(lock):
    return 'action4.sam.native_variant_export_receipt.v1' if _is_variant(lock) else 'action4.sam.native_export_receipt.v2'
_COMPLETED_SCOPE='frozen_completed_B0_native_original'
_VERIFIED_DEPENDENCIES={}
_DEPENDENCY_LOCK=threading.RLock()

def _v2_sha(path):
    import os
    p=contained_path(ROOT,path,exists=True).resolve(strict=True)
    def fingerprint(s):return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    before=fingerprint(p.stat());key=(os.getpid(),str(p),before)
    with _DEPENDENCY_LOCK:
        if key in _VERIFIED_DEPENDENCIES:return _VERIFIED_DEPENDENCIES[key]
    with p.open('rb') as f:
        if fingerprint(os.fstat(f.fileno()))!=before:raise ValueError('Dependency replaced before verification')
        value=hashlib.file_digest(f,'sha256').hexdigest()
        if fingerprint(os.fstat(f.fileno()))!=before:raise ValueError('Dependency modified during verification')
    if fingerprint(p.stat())!=before:raise ValueError('Dependency replaced during verification')
    with _DEPENDENCY_LOCK:_VERIFIED_DEPENDENCIES[key]=value
    return value

def _fingerprint(stat):
    return (stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)

def _v2_read(binding,reader):
    """Bind the decoded data to the verified file, including SHA memo hits."""
    import os
    if not isinstance(binding,dict) or set(binding)!={'path','sha256'}:
        raise ValueError('Explicit completed dependency binding required')
    path=contained_path(ROOT,binding['path'],exists=True).resolve(strict=True)
    before=_fingerprint(path.stat())
    if _v2_sha(path)!=binding['sha256']:raise ValueError('Completed source dependency changed')
    with path.open('rb') as stream:
        if _fingerprint(os.fstat(stream.fileno()))!=before:
            raise ValueError('Dependency replaced between hash and read')
        value=reader(stream)
        if _fingerprint(os.fstat(stream.fileno()))!=before:
            raise ValueError('Dependency changed during read/decode')
    if _fingerprint(path.stat())!=before:
        raise ValueError('Dependency replaced during read/decode')
    return value

def _v2_checked(binding):
    return _v2_read(binding,json.load)

_RGB_DECODE_CACHE=OrderedDict()
_RGB_DECODE_PID=None
_RGB_DECODE_MAX_BYTES=64*1024*1024

def _v2_rgb(path,digest):
    """Reuse exact decoded native bytes, bounded per worker; never cache model outputs."""
    import os
    global _RGB_DECODE_PID
    pid=os.getpid()
    if _RGB_DECODE_PID!=pid:
        _RGB_DECODE_CACHE.clear();_RGB_DECODE_PID=pid
    p=contained_path(ROOT,path,exists=True).resolve(strict=True)
    before=_fingerprint(p.stat())
    if _v2_sha(p)!=digest:raise ValueError('Completed source dependency changed')
    key=(str(p),digest,before)
    if key in _RGB_DECODE_CACHE:
        rgb=_RGB_DECODE_CACHE[key]
        if _fingerprint(p.stat())!=before:raise ValueError('RGB replaced/modified during cached decode')
        _RGB_DECODE_CACHE.move_to_end(key)
        return rgb.copy()
    def decode(stream):
        with Image.open(stream) as image:return np.array(image.convert('RGB'),copy=True)
    rgb=_v2_read({'path':str(p),'sha256':digest},decode)
    if _fingerprint(p.stat())!=before:raise ValueError('RGB changed during decode')
    if rgb.nbytes<=_RGB_DECODE_MAX_BYTES:
        while _RGB_DECODE_CACHE and sum(x.nbytes for x in _RGB_DECODE_CACHE.values())+rgb.nbytes>_RGB_DECODE_MAX_BYTES:
            _RGB_DECODE_CACHE.popitem(last=False)
        stored=rgb.copy();stored.flags.writeable=False;_RGB_DECODE_CACHE[key]=stored
    return rgb

def _v2_payload(binding):
    def decode(stream):
        with np.load(stream,allow_pickle=False) as cache:
            if set(cache.files) not in ({'metadata'},{'metadata','mask'}):raise ValueError('Unexpected completed cache fields')
            meta=cache['metadata']
            if meta.ndim!=0 or meta.dtype.kind!='U':raise ValueError('Cache metadata must be scalar Unicode JSON')
            return NativeSamPrediction(json.loads(str(meta)),None if 'mask' not in cache.files else cache['mask'].copy())
    return _v2_read(binding,decode)

def _pixel_hash(mask):
    return hashlib.sha256(json.dumps({'dtype':'bool','shape':list(mask.shape)},separators=(',',':')).encode()+b'\0'+mask.tobytes(order='C')).hexdigest()

def _check_payload(pred,row,c,entry,sources,status,reason,binding,result,lock):
    from .action4_sam_roi_inputs import cache_key,SCHEMA
    meta=pred.metadata;mask=pred.mask
    required={'schema','native_identity','source_sha256','variant_id','variant_sha256','native_shape','sources',
        'mask_sha256','status','assigned_instance_id','candidate_overlap','reason','origin','sam_binding','cache_key'}
    if (set(meta)!=required or meta['schema']!=SCHEMA or meta['cache_key']!=cache_key(meta)
            or meta['native_identity']!=[row['image_id'],c['candidate_id'],row['pixel_space_id']]
            or (meta['source_sha256'],meta['variant_id'],meta['variant_sha256'],meta['native_shape'])!=
               (row['image_sha256'],_variant_id(row),entry['variant_sha256'],row['native_shape_hw'])
            or meta['sources']!=sources or meta['origin']!='predicted_sam'
            or meta['status']!=status or meta['reason']!=reason or meta['sam_binding']!=binding):
        raise ValueError('Completed cache payload identity/status/source mismatch')
    if status!='available':
        if mask is not None or meta['mask_sha256'] is not None or meta['assigned_instance_id'] is not None or meta['candidate_overlap']!=0 or binding is not None:
            raise ValueError('Unavailable/unprompted cache must not carry mask evidence')
        return
    if mask is None or mask.dtype!=np.bool_ or list(mask.shape)!=row['native_shape_hw'] or meta['mask_sha256']!=array_digest(mask) or meta['assigned_instance_id']!=c['candidate_id']:
        raise ValueError('Completed cache actual mask identity invalid')
    reg=result['mask_registration'];native_binding=reg['binding'];pix=_pixel_hash(mask)
    selected=[v for v in result['native_candidates'] if v['obj_id']==result['bound_obj_id']]
    if len(selected)!=1 or meta['candidate_overlap']!=float(selected[0]['target_mask_coverage']):
        raise ValueError('Completed cache selected overlap mismatch')
    if (pix!=reg['pixel_sha256'] or reg['binding_identity']!=canonical(native_binding)
            or reg['mask_identity']!='sam-mask-sha256:'+canonical({'binding':reg['binding_identity'],'pixels':pix})
            or (native_binding['session_id'],native_binding['obj_id'],native_binding['generation'],native_binding['track_id'],
                native_binding['pixel_identity'],native_binding['dependency_identity'],native_binding['official_commit'])!=
               (result['session_id'],result['bound_obj_id'],result['generation'],c['candidate_id'],row['pixel_space_id'],
                lock['identity'],lock['config']['native']['source_commit'])):
        raise ValueError('Completed cache pixels differ from native selected registration')

# Proofs contain only digest keys, never decoded masks. Capacity covers current
# train10400 + eval1300 candidates while remaining bounded per worker process.
_PAYLOAD_PROOF_LOCK=threading.RLock()
_PAYLOAD_PROOFS=OrderedDict()
_PAYLOAD_PROOF_PID=None
_PAYLOAD_PROOF_MAX_ENTRIES=16384


def _payload_proof_cache():
    import os
    global _PAYLOAD_PROOF_PID
    pid=os.getpid()
    if _PAYLOAD_PROOF_PID!=pid:
        _PAYLOAD_PROOFS.clear()
        _PAYLOAD_PROOF_PID=pid
    return _PAYLOAD_PROOFS


def _verify_payload_with_proof(cache_binding,row,c,entry,sources,status,reason,binding,result,lock,*,proof_context):
    """Reuse only a full successful pixel proof with identical bytes and expectations.

    Outer receipt/source/marker/result validators still execute on every call.
    All objects used by _check_payload are snapshotted and hashed, so caller
    mutation or a resealed index cannot reuse another expectation's proof.
    """
    import copy
    expected=copy.deepcopy({"cache_binding":cache_binding,"row":row,"candidate":c,
        "entry":entry,"sources":sources,"status":status,"reason":reason,
        "sam_binding":binding,"result":result,"lock":lock,"proof_context":proof_context})
    ref=expected["cache_binding"]
    if not isinstance(ref,dict) or set(ref)!={"path","sha256"}:raise ValueError("Explicit payload proof binding required")
    path=contained_path(ROOT,ref["path"],exists=True).resolve(strict=True)
    before=_fingerprint(path.stat())
    if _v2_sha(path)!=ref["sha256"]:raise ValueError("Payload proof cache bytes changed")
    def unchanged():
        current=contained_path(ROOT,ref["path"],exists=True).resolve(strict=True)
        if current!=path or _fingerprint(current.stat())!=before:
            raise ValueError("Payload replaced/modified during proof verification")
    unchanged()
    proof=canonical({"protocol":"action4.completed_native_payload_proof.v1",
        "resolved_path":str(path),"stat":before,"expected":expected})
    with _PAYLOAD_PROOF_LOCK:
        memo=_payload_proof_cache()
        hit=proof in memo
        if hit:memo.move_to_end(proof)
    if hit:
        unchanged()
        return
    payload=_v2_payload(ref)
    _check_payload(payload,expected["row"],expected["candidate"],expected["entry"],expected["sources"],
        expected["status"],expected["reason"],expected["sam_binding"],expected["result"],expected["lock"])
    unchanged()
    with _PAYLOAD_PROOF_LOCK:
        memo=_payload_proof_cache()
        memo[proof]=None
        memo.move_to_end(proof)
        while len(memo)>_PAYLOAD_PROOF_MAX_ENTRIES:memo.popitem(last=False)


def _reset_validation_locks_after_fork():
    global _PAYLOAD_PROOF_LOCK,_DEPENDENCY_LOCK,_PAYLOAD_PROOF_PID
    _PAYLOAD_PROOF_LOCK=threading.RLock()
    _DEPENDENCY_LOCK=threading.RLock()
    _PAYLOAD_PROOFS.clear()
    _VERIFIED_DEPENDENCIES.clear()


if hasattr(os,"register_at_fork"):
    os.register_at_fork(after_in_child=_reset_validation_locks_after_fork)


def _validation_worker_count(workers):
    if type(workers) is not int or not 1<=workers<=8:
        raise ValueError("validation workers must be an integer in [1,8]")
    return workers


def _validate_payload_jobs(jobs,workers):
    """Bound in-flight decoding; return only after every proof and final stat pass."""
    def validate(job):
        args,kwargs=job
        ref=args[0]
        path=contained_path(ROOT,ref["path"],exists=True).resolve(strict=True)
        before=_fingerprint(path.stat())
        _verify_payload_with_proof(*args,**kwargs)
        if _fingerprint(path.stat())!=before:raise ValueError("Payload changed during batch verification")
        return ref["path"],path,before
    verified=[]
    if workers==1:
        verified=[validate(job) for job in jobs]
    else:
        with ThreadPoolExecutor(max_workers=workers,thread_name_prefix="sam-cache-verify") as pool:
            pending=deque()
            for job in jobs:
                pending.append(pool.submit(validate,job))
                if len(pending)>=workers:verified.append(pending.popleft().result())
            verified.extend(future.result() for future in pending)
    for ref,path,before in verified:
        current=contained_path(ROOT,ref,exists=True).resolve(strict=True)
        if current!=path or _fingerprint(current.stat())!=before:
            raise ValueError("Payload changed before full batch verification completed")


def _no_gt_prompt(value):
    if isinstance(value,dict):
        for key,item in value.items():
            if key.lower().startswith(('gt','ground_truth')) and not ((key=='GT_instance_id' and item is None) or (key=='GT_used_for_prompt_selection' and item is False)):
                raise ValueError('GT forbidden in completed prediction prompts')
            _no_gt_prompt(item)
    elif isinstance(value,list):
        for item in value:_no_gt_prompt(item)

def _verify_native_snapshot(lock):
    cfg=lock['config']
    snapshot=contained_path(ROOT,cfg['source_snapshot'],exists=True)
    manifest=read(snapshot/'SNAPSHOT_MANIFEST.json')
    if manifest.get('schema')!='action4.sam.immutable_source.v1' or manifest.get('identity')!=canonical({k:v for k,v in manifest.items() if k!='identity'}) or manifest['identity']!=lock['source_snapshot_identity']:
        raise ValueError('Immutable native source seal mismatch')
    if manifest.get('official_commit')!=cfg['native']['source_commit'] or manifest.get('official_source_relative')!='upstream/sam3':
        raise ValueError('Official native source commit/root mismatch')
    official_root=snapshot/manifest['official_source_relative']
    if not official_root.is_dir() or official_root.is_symlink():raise ValueError('Official source directory invalid')
    sealed_official={relative:digest for relative,digest in manifest['files_sha256'].items() if relative.startswith('upstream/sam3/')}
    actual_official=set()
    for current,dirs,files in __import__('os').walk(official_root,followlinks=False):
        for name in dirs+files:
            if (Path(current)/name).is_symlink():raise ValueError('Official source symlink forbidden')
        actual_official.update(str((Path(current)/name).relative_to(snapshot)) for name in files)
    if not sealed_official or actual_official!=set(sealed_official):raise ValueError('Official source file domain mismatch')
    for relative,digest in sealed_official.items():
        path=snapshot/relative
        if path.stat().st_mode & 0o222 or _v2_sha(path)!=digest:
            raise ValueError('Official sealed source changed or writable')
    config_path=snapshot/manifest['config_file']
    if (_v2_checked({'path':str(config_path),'sha256':manifest['files_sha256'][manifest['config_file']]})!=cfg
            or config_path.stat().st_mode & 0o222):
        raise ValueError('Native sealed config mismatch')
    for path,digest in lock['code_sha256'].items():
        p=contained_path(ROOT,path,exists=True)
        if not p.is_relative_to(snapshot) or manifest['files_sha256'].get(str(p.relative_to(snapshot)))!=digest or _v2_sha(p)!=digest or p.stat().st_mode & 0o222:
            raise ValueError('Immutable native producer code changed')


def _completed_domain(lock,completion):
    if _is_variant(lock):return _variant_domain(lock,completion)
    cfg=lock['config']
    if (lock.get('schema')!='action4.sam.single_image_native.v2'
            or lock['identity']!=canonical({k:v for k,v in lock.items() if k!='identity'})
            or lock['config_identity']!=canonical(cfg)
            or cfg.get('upstream_role')!='frozen_completed_B0_post_soft_nms'
            or cfg.get('split') not in ('train','eval') or cfg.get('rgb_variant')!='native_original'):
        raise ValueError('Completed B0 native source/split contract mismatch')
    if (completion.get('status')!='NATIVE_TRANSPORT_COMPLETE_MASK_QUALITY_NOT_SCORED'
            or completion.get('lock_identity')!=lock['identity']
            or completion.get('backend_kind')!='sam31_official_native'
            or completion.get('completed_jobs')!=cfg['expected_prompts']
            or completion.get('expected_prompt_jobs')!=cfg['expected_prompts']
            or completion.get('image_domain_count')!=cfg['expected_images']
            or completion.get('upstream_saved_candidates')!=cfg['expected_saved_candidates']
            or completion.get('upstream_unprompted_candidates')!=cfg['expected_saved_candidates']-cfg['expected_prompts']):
        raise ValueError('Incomplete completed-B0 native candidate domain; formal export refused')
    if any(type(cfg[k]) is not int or cfg[k]<0 for k in ('expected_images','expected_saved_candidates','expected_prompts')) or cfg['expected_images']<1:
        raise ValueError('Invalid completed domain counts')
    source=cfg['source_contract']
    if set(source)!={'sealed_role','run_spec','b0_bundle','selected_step','config_id','predictions'}:
        raise ValueError('Unknown completed B0 source graph')
    seal=_v2_checked(source['sealed_role']);run=_v2_checked(source['run_spec']);bundle=source['b0_bundle']
    if set(bundle)!={'path','sha256'} or _v2_sha(bundle['path'])!=bundle['sha256']:raise ValueError('B0 bundle changed')
    if (seal.get('schema')!='action4.sealed_independent_role.v2' or seal.get('role')!='B0' or seal.get('strict_load') is not True
            or seal.get('selected_step')!=source['selected_step'] or seal.get('config_id')!=source['config_id']
            or (seal.get('bundle'),seal.get('sha256'))!=(bundle['path'],bundle['sha256'])
            or (seal.get('run_spec'),seal.get('run_spec_sha256'))!=(source['run_spec']['path'],source['run_spec']['sha256'])):
        raise ValueError('Completed B0 release seal mismatch')
    done=seal.get('completion',{})
    if (done.get('status')!='COMPLETION_METADATA_READY' or done.get('role')!='B0'
            or done.get('best_step')!=source['selected_step'] or done.get('evaluations')!=653
            or done.get('config_id')!=source['config_id']):
        raise ValueError('Completed B0 full training/evaluation seal missing')
    if (run.get('schema')!='action4.immutable_run.v1' or run.get('mutable_worktree_not_used_by_children') is not True
            or run.get('code_root')!=seal.get('actual_code_root')):
        raise ValueError('Immutable B0 run mismatch')
    for b in (source['sealed_role'],source['run_spec'],bundle):
        if contained_path(ROOT,b['path'],exists=True).stat().st_mode & 0o222:raise ValueError('B0 sources must remain sealed')
    if _v2_sha(cfg['manifest'])!=cfg['manifest_sha256'] or lock['manifest_sha256']!=cfg['manifest_sha256']:
        raise ValueError('Prompt manifest changed')
    if _v2_sha(cfg['input_contract'])!=lock['input_contract_sha256']:raise ValueError('Native input contract changed')
    _verify_native_snapshot(lock)
    saved_rows=_v2_checked(source['predictions']);_no_gt_prompt(saved_rows)
    saved={r['sample_id']:r['instances'] for r in saved_rows}
    if len(saved)!=len(saved_rows):raise ValueError('Duplicate saved prediction image')
    rows=_v2_read({'path':cfg['manifest'],'sha256':cfg['manifest_sha256']},lambda stream:[json.loads(line) for line in stream if line.strip()])
    if len(rows)!=cfg['expected_images'] or len({r['image_id'] for r in rows})!=len(rows) or set(saved)!={r['image_id'] for r in rows}:
        raise ValueError('Completed image domain mismatch')
    total=eligible=0;ids=set()
    for row in rows:
        _no_gt_prompt(row)
        if (row.get('schema')!='action4.sam.prediction_prompt_image.v2' or row.get('split')!=cfg['split']
                or row.get('rgb_variant')!='native_original' or row['source_receipt']!=source
                or row.get('GT_used_for_prompt_selection') is not False):
            raise ValueError('Completed prompt source/split/RGB mismatch')
        candidates=row['candidates'];raw=saved[row['image_id']]
        if len(candidates)!=len(raw) or row['raw_saved_candidate_count']!=len(raw):raise ValueError('Missing saved candidate')
        ordinals=[];n=0
        for c in candidates:
            b=c['identity_binding'];grid=b['grid_index'];ordinal=b['raw_saved_ordinal']
            if type(grid) is not int or grid<0 or type(ordinal) is not int or not 0<=ordinal<len(raw):
                raise ValueError('Invalid stable candidate grid/ordinal')
            pid=f"b0:{row['image_id']}:grid:{grid}"
            if pid in ids or c['candidate_id']!=pid or b['sample_id']!=row['image_id'] or b['prediction_file_sha256']!=source['predictions']['sha256']:
                raise ValueError('Candidate identity mismatch')
            ids.add(pid);ordinals.append(ordinal);r=raw[ordinal]
            if (c['source']!='frozen_completed_B0_saved_prediction' or c['raw_saved_prediction']!=r or r['grid_index']!=grid
                    or c['score']!=r['score'] or c['bbox_full_native_xyxy']!=r['bbox_full_xyxy']):
                raise ValueError('Saved candidate was changed')
            h,w=row['native_shape_hw'];box=np.asarray(r['bbox_full_xyxy'],float)
            if box.shape!=(4,) or not np.isfinite(box).all() or not np.isfinite(r['score']):raise ValueError('Invalid saved predicted box/score')
            clipped=np.clip(box,[0,0,0,0],[w,h,w,h])
            flag=bool(r['score']>=.3 and (clipped[2:]>clipped[:2]).all())
            if c['published'] is not bool(r['score']>=.3) or c['eligible_for_SAM_diagnostic_prompt'] is not flag:
                raise ValueError('Candidate prompt eligibility changed')
            if flag:
                norm=np.r_[clipped[:2]/[w,h],(clipped[2:]-clipped[:2])/[w,h]]
                if not np.allclose(c['bbox_sam_native_xyxy'],clipped,rtol=0,atol=1e-9) or not np.allclose(c['bbox_sam_normalized_xywh'],norm,rtol=0,atol=1e-9):
                    raise ValueError('Prompt coordinates changed')
            n+=flag
        if sorted(ordinals)!=list(range(len(raw))) or row['SAM_prompt_count']!=n:raise ValueError('Candidate ordinal domain changed')
        total+=len(raw);eligible+=n
    if (total,eligible)!=(cfg['expected_saved_candidates'],cfg['expected_prompts']):raise ValueError('Candidate domain counts differ')
    return rows,run

def _completed_sources(lock,load,run):
    cfg=lock['config'];asset_path=contained_path(ROOT,cfg['native']['asset_receipt'],exists=True)
    snapshot=contained_path(ROOT,cfg['source_snapshot'],exists=True);seal=read(snapshot/'SNAPSHOT_MANIFEST.json')
    if not asset_path.is_relative_to(snapshot) or seal['files_sha256'].get(str(asset_path.relative_to(snapshot)))!=_v2_sha(asset_path):
        raise ValueError('Native asset receipt differs from source seal')
    asset=_v2_checked({'path':str(asset_path),'sha256':seal['files_sha256'][str(asset_path.relative_to(snapshot))]})
    if (load.get('status')!='LOADED_ZERO_FINAL_MISSING_UNEXPECTED' or load.get('source_snapshot_identity')!=lock['source_snapshot_identity']
            or load.get('source_commit')!=cfg['native']['source_commit']
            or load.get('weight_sha256_from_prior_validation')!=asset['sha256']):
        raise ValueError('Native model load source mismatch')
    return {'upstream_mode':VARIANT_MODE if _is_variant(lock) else COMPLETED_MODE,'b0_model_sha256':(_source_contract(cfg)['B0_bundle'] if _is_variant(lock) else cfg['source_contract']['b0_bundle'])['sha256'],
        'b0_source_sha256':run['project_code_sha256'],'sam_model_sha256':asset['sha256'],
        'sam_source_sha256':lock['source_snapshot_identity'],'sam_prompt_sha256':cfg['manifest_sha256'],
        'postprocess_sha256':next(v for k,v in lock['code_sha256'].items() if k.endswith('action4_sam_image_eval.py'))}

def verify_completed_export_receipt(index,*,workers=1):
    import copy
    _validation_worker_count(workers)
    index=copy.deepcopy(index)
    if workers>1 and index.get("sources",{}).get("upstream_mode")!=COMPLETED_MODE:
        raise ValueError("Parallel validation supports completed-original caches only")
    jobs=[]
    def queue_payload(*args,**kwargs):jobs.append((args,kwargs))
    receipt=_v2_checked(index['producer_receipt'])
    if (receipt.get('schema') not in ('action4.sam.native_export_receipt.v2','action4.sam.native_variant_export_receipt.v1')
            or receipt.get('index_payload_sha256')!=canonical({k:v for k,v in index.items() if k!='producer_receipt'})):
        raise ValueError('Completed export producer/index identity mismatch')
    lock=_v2_checked(receipt['native_lock']);progress=_v2_checked(receipt['native_completion'])
    rows,run=_completed_domain(lock,progress);cfg=lock['config']
    if receipt.get('schema')!=_receipt_schema(lock):raise ValueError('Receipt mode differs from native schema')
    sources=_completed_sources(lock,_v2_checked(receipt['native_load']),run)
    if (index.get('split')!=cfg['split'] or index.get('scope')!=(VARIANT_SCOPE if _is_variant(lock) else _COMPLETED_SCOPE)
            or index.get('rgb_variant')!=cfg['rgb_variant'] or index['sources']!=sources
            or receipt.get('sources')!=sources or receipt.get('source_contract')!=_source_contract(cfg)
            or receipt.get('GT_used') is not False or receipt.get('native_mask_selection_changed') is not False
            or receipt.get('training_association_status')!='not_provided'):
        raise ValueError('Completed export source/split/native RGB/association contract mismatch')
    if _is_variant(lock) and receipt.get('formal_training_admitted') is not False:raise ValueError('Variant training not admitted')
    by_image={r['image_id']:r for r in rows}
    if len(index['images'])!=len(rows) or {r['image_id'] for r in index['images']}!=set(by_image):raise ValueError('Export image coverage incomplete')
    expected=set();index_bindings={};candidates={};image_entries={}
    for entry in index['images']:
        row=by_image[entry['image_id']];image_entries[row['image_id']]=entry
        if (entry['image_path'],entry['source_sha256'],entry['pixel_space_id'],entry['variant_id'])!=(row['image_path'],row['image_sha256'],row['pixel_space_id'],_variant_id(row)):
            raise ValueError('Export RGB source mismatch')
        if _is_variant(lock):
            if entry.get('original_source')!=row['original_source'] or entry['variant_sha256']!=row['variant_pixels_sha256']:
                raise ValueError('Variant original source/pixels changed')
        ids=[c['candidate_id'] for c in row['candidates']]
        if len(entry['predictions'])!=len(ids) or {p['prediction_id'] for p in entry['predictions']}!=set(ids):raise ValueError('Export candidate coverage incomplete')
        expected.update((row['image_id'],pid) for pid in ids)
        index_bindings.update({(row['image_id'],p['prediction_id']):p for p in entry['predictions']})
        candidates.update({(row['image_id'],c['candidate_id']):(row,c) for c in row['candidates']})
    mappings=receipt['mapping']
    if len(mappings)!=len(expected) or {(m['image_id'],m['prediction_id']) for m in mappings}!=expected:
        raise ValueError('Producer candidate domain incomplete')
    if receipt.get('image_count')!=len(rows) or receipt.get('candidate_records')!=len(expected):raise ValueError('Producer counts mismatch')
    if receipt.get('native_prompt_records')!=cfg['expected_prompts']:raise ValueError('Producer prompt count mismatch')
    unprompted={(r['image_id'],c['candidate_id']) for r in rows for c in r['candidates'] if not c['eligible_for_SAM_diagnostic_prompt']}
    omitted=receipt['upstream_unprompted']
    if len(omitted)!=len(unprompted) or {(m['image_id'],m['prediction_id']) for m in omitted}!=unprompted:
        raise ValueError('Producer unprompted domain mismatch')
    for m in mappings:
        pair=(m['image_id'],m['prediction_id']);row,c=candidates[pair]
        if (m.get('source_candidate_id')!=c['candidate_id'] or m.get('identity_binding')!=c['identity_binding']
                or m.get('prompted') is not c['eligible_for_SAM_diagnostic_prompt']
                or m.get('cache_sha256')!=index_bindings[pair]['sha256']):
            raise ValueError('Producer candidate/cache binding mismatch')
        cache_binding={"path":index_bindings[pair]["path"],"sha256":index_bindings[pair]["sha256"]}
        proof_context={"mapping":m,"producer_receipt":index["producer_receipt"],
            "index_payload_sha256":receipt["index_payload_sha256"]}
        if not m['prompted']:
            if (m.get('status'),m.get('reason'))!=('missing','upstream_not_prompted'):
                raise ValueError('Unprompted candidate fabricated as native output')
            queue_payload(cache_binding,row,c,image_entries[row['image_id']],sources,'missing','upstream_not_prompted',None,None,lock,proof_context=proof_context)
            continue
        key=canonical({'lock':lock['identity'],'image_id':row['image_id'],'image_sha256':row['image_sha256'],'candidate':c,'text':cfg['native']['text']})
        expected_marker=contained_path(ROOT,cfg['output'],exists=True)/'jobs'/key/'COMPLETED.json'
        if contained_path(ROOT,m['completed_marker'],exists=True)!=expected_marker:raise ValueError('Producer completed task pointer mismatch')
        marker=_v2_checked({'path':m['completed_marker'],'sha256':m['completed_marker_sha256']})
        if marker['lock_identity']!=lock['identity'] or marker['task_key']!=key or marker['files_sha256'].get(marker['result_path'])!=m['result_sha256']:
            raise ValueError('Producer completion evidence mismatch')
        result=_v2_checked({'path':marker['result_path'],'sha256':m['result_sha256']})
        if (result['candidate_id']!=c['candidate_id'] or result['source_receipt']!=_source_contract(cfg)
                or result['status']!=m['native_status'] or result['close_success'] is not True):
            raise ValueError('Producer native result mapping mismatch')
        if _is_variant(lock):_verify_variant_result(lock,row,c,marker,result,expected_marker.parent)
        if result['available']:
            binding={'session_id':result['session_id'],'obj_id':result['bound_obj_id'],'generation':result['generation'],
                'chosen_mask_id':result['mask_registration']['mask_identity'],'match_basis':result['binding_status']}
            if m.get('sam_binding')!=binding or m['status']!='available':raise ValueError('Producer selected native binding mismatch')
            queue_payload(cache_binding,row,c,image_entries[row['image_id']],sources,'available',None,binding,result,lock,proof_context=proof_context)
        else:
            statuses={'ambiguous_identity':'ambiguous','target_visibility_unknown':'target_visibility_unknown',
                'missing_response':'missing_response','session_lost':'session_lost','inference_error':'inference_error','target_absent':'rejected'}
            if result['status'] not in statuses:raise ValueError('Unknown native unavailable status')
            status=statuses[result['status']]
            reason=result['status']+':'+str(result.get('reason') or result.get('binding_status') or 'unavailable')
            if m['status']!=status or m.get('reason')!=reason or m.get('sam_binding') is not None:
                raise ValueError('Unavailable native candidate fabricated as available')
            queue_payload(cache_binding,row,c,image_entries[row['image_id']],sources,status,reason,None,result,lock,proof_context=proof_context)

    _validate_payload_jobs(jobs,workers)
    return receipt

def export_completed(native_root,output,limit=None):
    if limit is not None:raise ValueError('Completed B0 formal cache requires entire domain; no limit export')
    native=contained_path(ROOT,native_root,exists=True);out=contained_path(ROOT,output,exists=False)
    if out.exists():raise FileExistsError('Preserve existing export')
    lock=read(native/'RUN_LOCK.json');progress=read(native/'PROGRESS.json')
    rows,run=_completed_domain(lock,progress);cfg=lock['config'];load=read(native/'NATIVE_LOAD.json')
    sources=_completed_sources(lock,load,run)
    out.mkdir(parents=True);(out/'masks').mkdir()
    images=[];mapping=[];unprompted=[]
    for row in rows:
        rgb=_v2_rgb(row['image_path'],row['image_sha256'])
        if list(rgb.shape[:2])!=row['native_shape_hw']:raise ValueError('Native RGB dimensions changed')
        frame=ImageFrame(row['image_id'],rgb,row['pixel_space_id'],row['image_sha256'],_variant_id(row),array_digest(rgb))
        entries=[]
        for c in row['candidates']:
            pid=c['candidate_id'];key=canonical({'lock':lock['identity'],'image_id':row['image_id'],'image_sha256':row['image_sha256'],'candidate':c,'text':cfg['native']['text']})
            mask=binding=None;overlap=0.;status='missing';reason='upstream_not_prompted'
            evidence={'image_id':row['image_id'],'prediction_id':pid,'source_candidate_id':pid,'identity_binding':c['identity_binding'],'prompted':c['eligible_for_SAM_diagnostic_prompt']}
            if not c['eligible_for_SAM_diagnostic_prompt']:
                unprompted.append({'image_id':row['image_id'],'prediction_id':pid,'reason':reason})
            else:
                marker=native/'jobs'/key/'COMPLETED.json';done=read(marker);jobdir=marker.parent
                if done['lock_identity']!=lock['identity'] or done['task_key']!=key:raise ValueError('Completed task identity mismatch')
                rp=contained_path(ROOT,done['result_path'],exists=True)
                if str(rp.relative_to(ROOT)) not in done['files_sha256']:raise ValueError('Unbound completed result')
                for path,digest in done['files_sha256'].items():
                    p=contained_path(ROOT,path,exists=True)
                    if not p.is_relative_to(jobdir) or _v2_sha(p)!=digest:raise ValueError('Completed job bytes changed/escaped')
                result=_v2_checked({'path':str(rp),'sha256':done['files_sha256'][str(rp.relative_to(ROOT))]})
                if (result.get('schema')!=cfg['schema'] or result.get('task_key')!=key or result.get('lock_identity')!=lock['identity']
                        or result.get('image_id')!=row['image_id'] or result.get('image_sha256')!=row['image_sha256']
                        or result.get('candidate_id')!=pid or result.get('prediction_id')!=pid
                        or result.get('identity_binding')!=c['identity_binding'] or result.get('pixel_space_id')!=frame.pixel_space_id
                        or result.get('source_receipt')!=_source_contract(cfg) or result.get('split')!=cfg['split']
                        or result.get('rgb_variant')!=cfg['rgb_variant'] or result.get('close_success') is not True
                        or result.get('response_valid') is not True or result.get('execution_status')!='COMPLETED'
                        or result.get('backend_kind')!='sam31_official_native'):
                    raise ValueError('Native result source/instance/completion mismatch')
                native_status=result['status']
                reason=native_status+':'+str(result.get('reason') or result.get('binding_status') or 'unavailable')
                statuses={'ambiguous_identity':'ambiguous','target_visibility_unknown':'target_visibility_unknown',
                    'missing_response':'missing_response','session_lost':'session_lost','inference_error':'inference_error',
                    'target_absent':'rejected'}
                if result['available']:
                    if native_status!='target_present':raise ValueError('Available native status conflict')
                    choices=[v for v in result['native_candidates'] if v['obj_id']==result['bound_obj_id']]
                    if len(choices)!=1:raise ValueError('Selected native object not unique')
                    selected=choices[0];storage=selected['mask_storage']
                    rawpath=contained_path(ROOT,storage['npz'],exists=True)
                    if (not rawpath.is_relative_to(jobdir) or done['files_sha256'].get(str(rawpath.relative_to(ROOT)))!=storage['raw_sha256']
                            or _v2_sha(rawpath)!=storage['raw_sha256']):raise ValueError('Selected raw mask changed')
                    if storage['array_key']!='raw__outputs__out_binary_masks' or storage['row_index']!=selected['raw_index']:
                        raise ValueError('Native mask row mapping changed')
                    def selected_raw(stream):
                        with np.load(stream,allow_pickle=False) as raw:
                            if int(raw['raw__outputs__out_obj_ids'][storage['row_index']])!=result['bound_obj_id']:
                                raise ValueError('Native raw object/mask row mismatch')
                            return raw[storage['array_key']][storage['row_index']].squeeze().copy()
                    mask=_v2_read({'path':str(rawpath),'sha256':storage['raw_sha256']},selected_raw)
                    if mask.dtype!=np.bool_ or mask.shape!=rgb.shape[:2] or int(mask.sum())!=selected['mask_pixels']:raise ValueError('Native selected mask dtype/shape/count mismatch')
                    reg=result['mask_registration'];bind=reg['binding']
                    pix=hashlib.sha256(json.dumps({'dtype':'bool','shape':list(mask.shape)},separators=(',',':')).encode()+b'\0'+mask.tobytes(order='C')).hexdigest()
                    if (reg['pixel_sha256']!=pix or bind.get('official_commit')!=cfg['native']['source_commit'] or reg['binding_identity']!=canonical(bind)
                            or reg['mask_identity']!='sam-mask-sha256:'+canonical({'binding':reg['binding_identity'],'pixels':pix})
                            or (bind['session_id'],bind['obj_id'],bind['generation'],bind['track_id'],bind['pixel_identity'],bind['dependency_identity'])!=
                               (result['session_id'],result['bound_obj_id'],result['generation'],pid,frame.pixel_space_id,lock['identity'])):
                        raise ValueError('Native selected mask registration changed')
                    binding={'session_id':result['session_id'],'obj_id':result['bound_obj_id'],'generation':result['generation'],'chosen_mask_id':reg['mask_identity'],'match_basis':result['binding_status']}
                    status='available';reason=None;overlap=float(selected['target_mask_coverage'])
                else:
                    if native_status not in statuses:raise ValueError('Unknown unavailable native status')
                    status=statuses[native_status]
                evidence.update(native_status=native_status,completed_marker=str(marker.relative_to(ROOT)),completed_marker_sha256=sha(marker),result_sha256=sha(rp),sam_binding=binding)
            meta=prediction_metadata(frame,NativeMaskIdentity(frame.image_id,pid,frame.pixel_space_id),mask,sources,status=status,
                assigned_instance_id=pid if mask is not None else None,candidate_overlap=overlap,reason=reason,sam_binding=binding)
            verify_native_prediction(NativeSamPrediction(meta,mask),frame,NativeMaskIdentity(frame.image_id,pid,frame.pixel_space_id),sources)
            dest=out/'masks'/(key+'.npz');arrays={'metadata':np.array(json.dumps(meta,sort_keys=True))}
            if mask is not None:arrays['mask']=mask
            np.savez_compressed(dest,**arrays)
            entries.append({'prediction_id':pid,'path':str(dest.relative_to(ROOT)),'sha256':sha(dest)})
            evidence.update(status=status,reason=reason,cache_sha256=sha(dest));mapping.append(evidence)
        images.append({'image_id':frame.image_id,'image_path':row['image_path'],'source_sha256':frame.source_sha256,'pixel_space_id':frame.pixel_space_id,
            'variant_id':frame.variant_id,'variant_sha256':frame.variant_sha256,'predictions':entries})
    if _is_variant(lock):
        for entry,row in zip(images,rows):entry['original_source']=row['original_source']
    index={'schema':'action4.sam_native_v2_index.v1','split':cfg['split'],'scope':VARIANT_SCOPE if _is_variant(lock) else _COMPLETED_SCOPE,'rgb_variant':cfg['rgb_variant'],
        'origin':'predicted_sam','prompt_origin':'frozen_prediction_only','sources':sources,'images':images}
    receipt={'schema':_receipt_schema(lock),'index_payload_sha256':canonical(index),'sources':sources,'source_contract':_source_contract(cfg),
        'native_lock':{'path':str((native/'RUN_LOCK.json').relative_to(ROOT)),'sha256':sha(native/'RUN_LOCK.json')},
        'native_completion':{'path':str((native/'PROGRESS.json').relative_to(ROOT)),'sha256':sha(native/'PROGRESS.json')},
        'native_load':{'path':str((native/'NATIVE_LOAD.json').relative_to(ROOT)),'sha256':sha(native/'NATIVE_LOAD.json')},
        'exporter_sha256':sha(__file__),'GT_used':False,'native_mask_selection_changed':False,'geometry_unchanged':True,
        'training_association_status':'not_provided','formal_training_admitted':False,'mapping':mapping,'upstream_unprompted':unprompted,
        'image_count':len(images),'candidate_records':len(mapping),'native_prompt_records':cfg['expected_prompts']}
    rp=out/'PRODUCER_RECEIPT.json';rp.write_text(json.dumps(receipt,indent=2)+'\n')
    index['producer_receipt']={'path':str(rp.relative_to(ROOT)),'sha256':sha(rp)}
    verify_completed_export_receipt(index)
    ip=out/'INDEX.json';ip.write_text(json.dumps(index,indent=2)+'\n')
    spec={'manifest':{'path':str(ip.relative_to(ROOT)),'sha256':sha(ip)},'expected_sources':sources}
    (out/'CACHE_SPEC.json').write_text(json.dumps(spec,indent=2)+'\n')
    return {'status':('EXPORTED_COMPLETED_B0_NATIVE_RGB_VARIANT_READONLY_ASSOCIATIONS_NOT_PROVIDED' if _is_variant(lock) else 'EXPORTED_COMPLETED_B0_NATIVE_ORIGINAL_CACHE_ASSOCIATIONS_NOT_PROVIDED'),'images':len(images),'candidate_records':len(mapping),'spec':str((out/'CACHE_SPEC.json').relative_to(ROOT))}


_VARIANT_PIXEL_PROOFS=OrderedDict()

def _verify_variant_pixels(entry):
    """Only reuse a successful full pixel check in this process for unchanged files."""
    import os,copy
    e=copy.deepcopy(entry);before=[]
    for key in ("original_source","actual_input"):
        ref=e[key];p=contained_path(ROOT,ref["path"],exists=True).resolve(strict=True)
        fp=_fingerprint(p.stat())
        if _v2_sha(p)!=ref["sha256"]:raise ValueError("Variant input bytes changed")
        before.append((str(p),fp))
    proof=(os.getpid(),canonical(e),tuple(before))
    def unchanged():
        for key,(path,fp) in zip(("original_source","actual_input"),before):
            p=contained_path(ROOT,e[key]["path"],exists=True).resolve(strict=True)
            if str(p)!=path or _fingerprint(p.stat())!=fp:raise ValueError("Variant inputs changed during verification")
    if proof in _VARIANT_PIXEL_PROOFS:
        unchanged();_VARIANT_PIXEL_PROOFS.move_to_end(proof);return
    _v2_rgb(e["original_source"]["path"],e["original_source"]["sha256"])
    def decode(stream):
        with Image.open(stream) as image:
            if image.format!="PNG" or image.mode!="RGB" or getattr(image,"n_frames",1)!=1:
                raise ValueError("Derived native input must be one RGB PNG")
            return np.array(image,copy=True)
    rgb=_v2_read(e["actual_input"],decode)
    if list(rgb.shape)!=e["native_shape"] or array_digest(rgb)!=e["actual_pixels_sha256"]:
        raise ValueError("Actual variant RGB pixels changed")
    unchanged();_VARIANT_PIXEL_PROOFS[proof]=None
    while len(_VARIANT_PIXEL_PROOFS)>16384:_VARIANT_PIXEL_PROOFS.popitem(last=False)

def _variant_domain(lock,completion):
    """Authenticate a completed prediction-only variant leaf without executing it."""
    cfg=lock['config']
    if (lock.get('schema')!=VARIANT_SCHEMA or lock['identity']!=canonical({k:v for k,v in lock.items() if k!='identity'})
            or lock['config_identity']!=canonical(cfg) or cfg.get('split')!='train'
            or cfg.get('rgb_variant') not in ('light','blur')
            or cfg.get('upstream_role')!='frozen_completed_B0_rgb_variant_predictions'
            or cfg.get('native',{}).get('text')!='camera'
            or cfg.get('K_contract')!={'K':None,'status':'not_supplied_not_consumed_by_native_SAM'}
            or cfg.get('binding')!={'bbox_iou_role':'diagnostic_only','unique_margin':.15,
                'unique_margin_basis':'target_coverage_minus_best_other_candidate_coverage',
                'mask_in_box_min':.8,'multi_target_support_min':.7,'multiple_native_policy':'require_single_eligible_mask'}):
        raise ValueError('Explicit completed variant native contract required')
    for name in ('expected_images','expected_saved_candidates','expected_prompts'):
        if type(cfg[name]) is not int or cfg[name]<0:raise ValueError('Invalid variant counts')
    if cfg['expected_images']<1:raise ValueError('Empty variant domain')
    if (completion.get('status')!='NATIVE_TRANSPORT_COMPLETE_MASK_QUALITY_NOT_SCORED'
            or completion.get('lock_identity')!=lock['identity'] or completion.get('backend_kind')!='sam31_official_native'
            or completion.get('completed_jobs')!=cfg['expected_prompts']
            or completion.get('expected_prompt_jobs')!=cfg['expected_prompts']
            or completion.get('image_domain_count')!=cfg['expected_images']
            or completion.get('upstream_saved_candidates')!=cfg['expected_saved_candidates']
            or completion.get('upstream_unprompted_candidates')!=cfg['expected_saved_candidates']-cfg['expected_prompts']):
        raise ValueError('Incomplete variant native domain')
    prepared=_v2_checked(cfg['prepared_variant']);refs=prepared['source_receipt']
    if (prepared.get('schema')!='action4.sam_rgb_variant_preparation.v1'
            or prepared.get('status')!='EXACT_VARIANT_PROMPTS_PREPARED_NOT_NATIVE_OR_TRAINING_ADMITTED'
            or prepared.get('contains_GT_labels') is not False
            or prepared.get('native_SAM_calls')!=0 or prepared['variant']!=cfg['rgb_variant']
            or prepared['manifest']!={'path':cfg['manifest'],'sha256':cfg['manifest_sha256']}
            or cfg['input_contract']!=cfg['prepared_variant']['path']
            or lock['input_contract_sha256']!=cfg['prepared_variant']['sha256']
            or lock['manifest_sha256']!=cfg['manifest_sha256']):
        raise ValueError('Variant preparation/manifest mismatch')
    if set(refs)!={'variant_B0_completion','variant_B0_config','variant_input','predictions','B0_seal','B0_bundle','B0_run_spec'}:
        raise ValueError('Unknown variant source graph')
    dependencies={k:_v2_checked(v) for k,v in refs.items() if k!='B0_bundle'}
    for ref in refs.values():
        if _v2_sha(ref['path'])!=ref['sha256'] or contained_path(ROOT,ref['path'],exists=True).stat().st_mode & 0o222:
            raise ValueError('Variant producer dependency changed or writable')
    done=dependencies['variant_B0_completion'];bcfg=dependencies['variant_B0_config'];inputs=dependencies['variant_input']
    seal=dependencies['B0_seal'];run=dependencies['B0_run_spec'];saved=dependencies['predictions']
    if (done.get('status')!='COMPLETE_FROZEN_B0_RGB_VARIANT_LEAF' or done.get('engineering_fixture') is not False
            or done.get('GT_passed_to_inference') is not False or done['config']!=refs['variant_B0_config']
            or done['input']!=refs['variant_input'] or done['predictions']!=refs['predictions']
            or bcfg.get('schema')!='action4.rgb_variant_B0_leaf.v1'
            or bcfg.get('mode') not in ('production_cpu_pilot','production_native_variant_leaf')
            or bcfg['input']!=refs['variant_input'] or bcfg['B0_seal']!=refs['B0_seal']):
        raise ValueError('Real completed B0 variant leaf required')
    leaf=contained_path(ROOT,bcfg['output'],exists=True)
    if (contained_path(ROOT,refs['variant_B0_completion']['path'],exists=True)!=leaf/'COMPLETE.json'
            or contained_path(ROOT,refs['predictions']['path'],exists=True)!=leaf/'PREDICTIONS.json'):
        raise ValueError('Cross-run B0 completion splice')
    worker=bcfg['orchestrator']
    if _v2_sha(worker['path'])!=worker['sha256'] or contained_path(ROOT,worker['path'],exists=True).stat().st_mode & 0o222:
        raise ValueError('B0 leaf worker changed')
    if (seal.get('schema')!='action4.sealed_independent_role.v2' or seal.get('role')!='B0' or seal.get('strict_load') is not True
            or seal.get('epochs')!=80 or seal.get('completed_optimizer_steps')!=13040
            or seal.get('completion',{}).get('evaluations')!=653
            or seal['completion'].get('status')!='COMPLETION_METADATA_READY'
            or seal['completion'].get('role')!='B0'
            or seal['completion'].get('best_step')!=seal.get('selected_step')
            or seal['completion'].get('config_id')!=seal.get('config_id')
            or refs['B0_bundle']!={'path':seal['bundle'],'sha256':seal['sha256']}
            or refs['B0_run_spec']!={'path':seal['run_spec'],'sha256':seal['run_spec_sha256']}
            or run.get('schema')!='action4.immutable_run.v1' or run.get('mutable_worktree_not_used_by_children') is not True
            or run.get('code_root')!=seal.get('actual_code_root')):
        raise ValueError('Completed sealed B0 identity required')
    frozen_config=contained_path(ROOT,run['config'],exists=True)
    code=contained_path(ROOT,run['code_root'],exists=True)
    if frozen_config.stat().st_mode & 0o222:raise ValueError('B0 config must remain sealed')
    if not frozen_config.is_relative_to(code):raise ValueError('B0 config escaped sealed code')
    config=_v2_checked({'path':str(frozen_config),'sha256':run['files'][str(frozen_config.relative_to(code))]})
    if bcfg['scoring_policy']!=config['evaluation_scoring_policy'] or bcfg['publication_threshold']!=.3:
        raise ValueError('B0 scoring policy changed')
    _verify_native_snapshot(lock)
    if inputs.get('schema')!='action4.rgb_variant_B0_leaf.v1' or inputs.get('GT_passed_to_inference') is not False or inputs['variant']!=cfg['rgb_variant']:
        raise ValueError('Variant leaf input identity changed')
    manifest=_v2_checked(inputs['manifest']);registration=_v2_checked(inputs['expected_registration'])
    if inputs['expected_registration']!=manifest['expected_registration'] or registration['expected']!=manifest['expected'] or inputs['image_ids']!=manifest['image_ids']:
        raise ValueError('Variant registered image domain changed')
    wanted=[v for v in manifest['rows'] if v['variant']==cfg['rgb_variant']]
    rows=_v2_read(prepared['manifest'],lambda stream:[json.loads(line) for line in stream if line.strip()])
    if (len(rows)!=cfg['expected_images'] or len(saved)!=len(rows) or len(inputs['rows'])!=len(rows) or len(wanted)!=len(rows)
            or [r['image_id'] for r in rows]!=inputs['image_ids'] or len(set(inputs['image_ids']))!=len(rows)):
        raise ValueError('Variant image coverage changed')
    total=eligible=0
    for row,entry,source,pred in zip(rows,inputs['rows'],wanted,saved):
        _no_gt_prompt(row);_no_gt_prompt(pred)
        exact={k:source[k] for k in ('image_id','variant','actual_input','actual_pixels_sha256','native_shape','original_source','recipe')}
        exact['variant_id']=source['recipe']['variant_id']
        if entry!=exact or entry['variant_id'] in ('native_original','original','',None) or entry['recipe']['variant_sha256']!=entry['actual_pixels_sha256']:
            raise ValueError('Variant recipe/source binding mismatch')
        _verify_variant_pixels(entry)
        expected_pred=dict(entry);expected_pred['sample_id']=expected_pred.pop('image_id')
        if any(pred.get(k)!=v for k,v in expected_pred.items()):raise ValueError('B0 actual RGB binding changed')
        if (row.get('schema')!='action4.sam.prediction_prompt_image.variant.v1' or row.get('split')!='train'
                or row.get('rgb_variant')!=entry['variant'] or row.get('variant_id')!=entry['variant_id']
                or row.get('image_path')!=entry['actual_input']['path'] or row.get('image_sha256')!=entry['actual_input']['sha256']
                or row.get('variant_pixels_sha256')!=entry['actual_pixels_sha256'] or row.get('original_source')!=entry['original_source']
                or row.get('native_shape_hw')!=entry['native_shape'][:2] or row.get('pixel_space_id')!='native_integer_centers'
                or row['source_receipt']!=refs or pred['sample_id']!=row['image_id']
                or len(row['candidates'])!=len(pred['instances']) or len(pred['candidates'])!=len(pred['instances'])):
            raise ValueError('Variant prompt/prediction coverage changed')
        seen=set();h,w=row['native_shape_hw']
        for ordinal,(c,raw,wrapped) in enumerate(zip(row['candidates'],pred['instances'],pred['candidates'])):
            grid=raw['grid_index'];score=raw['score'];box=np.asarray(raw['bbox_full_xyxy'],float)
            if type(grid) is not int or grid<0 or grid in seen or not np.isfinite(score) or not 0<=score<=1 or box.shape!=(4,) or not np.isfinite(box).all() or not (box[2:]>box[:2]).all():
                raise ValueError('Invalid variant saved candidate')
            seen.add(grid);pid=f"b0:{row['image_id']}:{entry['variant_id']}:grid:{grid}"
            if wrapped!={'prediction_id':pid,'grid_index':grid,'raw_saved_ordinal':ordinal,'published':score>=.3,'raw_saved_prediction':raw}:
                raise ValueError('B0 wrapped raw candidate changed')
            clipped=np.clip(box,[0,0,0,0],[w,h,w,h]);flag=bool(score>=.3 and (clipped[2:]>clipped[:2]).all())
            binding={'sample_id':row['image_id'],'variant_id':entry['variant_id'],'grid_index':grid,'raw_saved_ordinal':ordinal,'variant_pixels_sha256':entry['actual_pixels_sha256']}
            if (c['candidate_id']!=pid or c['identity_binding']!=binding or c['raw_saved_prediction']!=raw
                    or c['source']!='frozen_completed_B0_rgb_variant_saved_prediction' or c['score']!=score
                    or c['published'] is not bool(score>=.3) or c['eligible_for_SAM_diagnostic_prompt'] is not flag
                    or c['bbox_full_native_xyxy']!=box.tolist() or c['bbox_sam_native_xyxy']!=clipped.tolist()
                    or not np.allclose(c['bbox_sam_normalized_xywh'],np.r_[clipped[:2]/[w,h],(clipped[2:]-clipped[:2])/[w,h]],rtol=0,atol=1e-9)):
                raise ValueError('Variant candidate/prompt identity changed')
            total+=1;eligible+=flag
    if (total,eligible)!=(cfg['expected_saved_candidates'],cfg['expected_prompts']) or (prepared['images'],prepared['saved_candidates'],prepared['eligible_jobs'])!=(len(rows),total,eligible) or (done['images'],done['saved_candidates'],done['published_candidates'])!=(len(rows),total,sum(c['published'] for r in rows for c in r['candidates'])):
        raise ValueError('Variant full domain counts changed')
    return rows,run


def _verify_variant_result(lock,row,c,marker,result,jobdir):
    cfg=lock['config'];key=marker['task_key']
    for path,digest in marker['files_sha256'].items():
        actual=contained_path(ROOT,path,exists=True)
        if not actual.is_relative_to(jobdir) or _v2_sha(actual)!=digest:raise ValueError('Variant job bytes changed/escaped')
    if (result.get('schema')!=VARIANT_SCHEMA or result.get('task_key')!=key or result.get('lock_identity')!=lock['identity']
            or result.get('image_id')!=row['image_id'] or result.get('image_sha256')!=row['image_sha256']
            or result.get('variant_id')!=row['variant_id'] or result.get('variant_pixels_sha256')!=row['variant_pixels_sha256']
            or result.get('original_source')!=row['original_source'] or result.get('rgb_variant')!=cfg['rgb_variant']
            or result.get('prediction_id')!=c['candidate_id'] or result.get('identity_binding')!=c['identity_binding']
            or result.get('pixel_space_id')!=row['pixel_space_id'] or result.get('split')!='train'
            or result.get('close_success') is not True or result.get('response_valid') is not True
            or result.get('execution_status')!='COMPLETED' or result.get('backend_kind')!='sam31_official_native'):
        raise ValueError('Variant native result identity/completion mismatch')
    if result['available']:
        if result['status']!='target_present':raise ValueError('Available status conflict')
        selected=[v for v in result['native_candidates'] if v['obj_id']==result['bound_obj_id']]
        if len(selected)!=1:raise ValueError('Variant raw object not unique')
        st=selected[0]['mask_storage'];raw=contained_path(ROOT,st['npz'],exists=True)
        if (not raw.is_relative_to(jobdir) or marker['files_sha256'].get(str(raw.relative_to(ROOT)))!=st['raw_sha256']
                or st['array_key']!='raw__outputs__out_binary_masks' or st['row_index']!=selected[0]['raw_index']):
            raise ValueError('Variant raw row/file binding changed')
        def decode(stream):
            with np.load(stream,allow_pickle=False) as data:
                if int(data['raw__outputs__out_obj_ids'][st['row_index']])!=result['bound_obj_id']:raise ValueError('Variant raw obj mismatch')
                return data[st['array_key']][st['row_index']].squeeze().copy()
        mask=_v2_read({'path':str(raw),'sha256':st['raw_sha256']},decode)
        if mask.dtype!=np.bool_ or list(mask.shape)!=row['native_shape_hw'] or int(mask.sum())!=selected[0]['mask_pixels'] or _pixel_hash(mask)!=result['mask_registration']['pixel_sha256']:
            raise ValueError('Variant raw mask pixels changed')


def verify_variant_export_receipt(index):
    if index.get('sources',{}).get('upstream_mode')!=VARIANT_MODE:raise ValueError('Explicit variant cache mode required')
    return verify_completed_export_receipt(index)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--native-root',required=True);p.add_argument('--output',required=True);p.add_argument('--limit',type=int);a=p.parse_args()
    print(json.dumps(export(a.native_root,a.output,a.limit)))
