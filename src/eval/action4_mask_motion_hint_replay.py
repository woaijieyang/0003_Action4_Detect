"""Bounded threaded mask-only replay of receipt-bound V3 image evidence; no RGB/LK/model calls."""
from pathlib import Path
from fractions import Fraction
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
import time
import os
from collections import Counter
import numpy as np
from model.action4_mask_motion_hint import evaluate_mask_motion_hint,DEFAULT_POLICY,policy_identity,rational

ROOT=Path('/data/zy/vln/a-test')


def path(name):
    p=(ROOT/name).resolve()
    if not p.is_relative_to(ROOT):raise ValueError('Outside project')
    return p


def binding(name):
    p=path(name)
    with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=str(p.relative_to(ROOT)),sha256=digest)


def verify(ref):
    p=path(ref['path'])
    if binding(p)['sha256']!=ref['sha256']:raise ValueError('Source binding changed: '+ref['path'])
    return p


def read(ref):return json.loads(verify(ref).read_text())


def extract_boundary_rows(predictions_path,frames,time_base,raw_predictions_path):
    compact=[]
    with predictions_path.open() as stream, raw_predictions_path.open() as raw_stream:
        for index,line in enumerate(stream):
            row=json.loads(line);frame=frames[index]
            if row['frame_id']!=index or frame['frame_id']!=index or row['pts']!=frame['pts'] or row['RGB_sha256']!=frame['image_sha256']:
                raise ValueError('V3 frame/PTS/RGB order mismatch')
            raw_line=next(raw_stream,None)
            if raw_line is None:raise ValueError('Raw parent has fewer frames than V3')
            parent=json.loads(raw_line)
            if any(row[k]!=parent[k] for k in ('frame_id','pts','RGB_sha256','track_replay_identity')):
                raise ValueError('V3 row differs from receipt-bound raw parent identity')
            identity=row['prediction']['identity']
            if identity.get('A0_step')!=7300 or identity.get('B0_step')!=9500:raise ValueError('Fixed7300/9500 model identity required')
            evidence={}
            for inst in row['prediction']['instances']:
                e=inst['pose_stabilization']['image_orientation_evidence'];tid=e['track_id']
                if (tid in evidence or inst.get('track_id',tid)!=tid or e['frame_id']!=index or e['pts']!=row['pts'] or
                    e['current_RGB_sha256']!=row['RGB_sha256'] or rational(e['time_base'])!=time_base or
                    type(e.get('available')) is not bool or type(e.get('hold_rotation')) is not bool or type(e.get('current_mask_authenticated')) is not bool):
                    raise ValueError('Saved boundary evidence identity differs')
                evidence[tid]=e
            if set(evidence)!={'track_00','track_01'}:raise ValueError('Complete dual-track boundary map required')
            compact.append(dict(frame_id=index,pts=row['pts'],RGB_sha256=row['RGB_sha256'],evidence=evidence))
        if next(raw_stream,None) is not None:raise ValueError('Raw parent has extra frames')
    if len(compact)!=len(frames):raise ValueError('Incomplete V3 source')
    return compact


def run_replay(config_path):
    import cv2
    from eval.action4_reanchor_roi_provider import ReanchorROIProvider
    from eval.action4_sam_full_review import ReceiptVerifier
    cv2.setNumThreads(1)
    cfg=json.loads(path(config_path).read_text());policy=cfg['policy'];policy_identity(policy)
    workers=cfg.get('workers',4)
    if type(workers) is not int or not 1<=workers<=4:raise ValueError('At most four continuous mask-reader blocks')
    out=path(cfg['output']);out.mkdir(exist_ok=True)
    if any((out/n).exists() for n in ('parts','RUN_SPEC.json','COMPLETE.json','MASK_MOTION_HINTS.jsonl')):raise FileExistsError('Existing replay must not be overwritten')
    receipt_ref=cfg['v3_receipt'];receipt=read(receipt_ref)
    if receipt.get('frames')!=2850 or receipt.get('validated_all_rows') is not True:raise ValueError('Completed2850 V3 receipt required')
    read(receipt['complete']);rawpath=verify(receipt['raw_predictions']);predpath=verify(receipt['processed_predictions'])
    lock=read(receipt['input_lock']);framesref=lock['video']['frames_manifest'];manifest=read(framesref);frames=manifest['frames']
    if len(frames)!=2850:raise ValueError('Expected complete2850 frame manifest')
    time_base=rational(manifest['source']['time_base']);replay=lock['video']['track_replay']
    rows=extract_boundary_rows(predpath,frames,time_base,rawpath)
    refs=dict(v3_receipt=receipt_ref,v3_predictions=receipt['processed_predictions'],raw_predictions=receipt['raw_predictions'],input_lock=receipt['input_lock'],frames_manifest=framesref,replay=replay['predictions_ref'])
    provider=ReanchorROIProvider(**replay,register_project_bodies=True)
    starts=[int(x) for x in np.linspace(0,len(rows),workers+1,dtype=int)]
    (out/'RUN_SPEC.json').write_text(json.dumps(dict(source_refs=refs,policy=policy,policy_id=policy_identity(policy),blocks=[[starts[i],starts[i+1]] for i in range(workers)],CPU=sorted(os.sched_getaffinity(0)),GPU=False,new_model_calls=0,RGB_decodes=0,LK_calls=0,source_boundary_recomputed=False),indent=2))
    (out/'parts').mkdir()
    start=time.monotonic()
    def identity(tid,row,masks,mask_receipt):
        mask=masks.get(tid);authenticated=(mask is not None and mask_receipt.get('status')=='current_native_response' and mask_receipt.get('per_track_status',{}).get(tid)=='current_mask_present')
        return dict(track_id=tid,frame_id=row['frame_id'],pts=row['pts'],time_base=time_base,RGB_sha256=row['RGB_sha256'],pixel_space_id=mask_receipt['pixel_space_id'],session_id=mask_receipt.get('session_id'),generation=mask_receipt.get('generation'),mask_sha256=mask_receipt.get('mask_sha256',{}).get(tid),mask_authenticated=authenticated,source_receipt=mask_receipt)
    def block(i):
        local=copy.copy(provider);local.verifier=ReceiptVerifier();lo,hi=starts[i],starts[i+1]
        target=out/'parts'/f'part_{i:02d}.jsonl';counts=Counter();loaded=0
        def load(fid):
            nonlocal loaded
            masks,mc=local.current_native_masks(fid);r=rows[fid];loaded+=1
            if mc['frame_id']!=fid or mc['pts']!=r['pts'] or mc['RGB_sha256']!=r['RGB_sha256'] or rational(mc['time_base'])!=time_base:
                raise ValueError('Current native masks differ from source frame')
            identities={tid:identity(tid,r,masks,mc) for tid in ('track_00','track_01')}
            for tid,ident in identities.items():
                e=r['evidence'][tid]
                if ident['mask_authenticated']!=e['current_mask_authenticated']:raise ValueError('Saved versus actual current mask authentication differs')
                if ident['mask_authenticated']:
                    source=e['source_refs']['current']
                    if (source['input_lock']!=receipt['input_lock'] or source['replay']!=replay['predictions_ref'] or
                        source['current_track_mask_sha256']!=ident['mask_sha256'] or source['mask_receipt']!=mc):
                        raise ValueError('Saved V3 mask source differs from actual current receipt')
            return masks,identities
        previous=load(lo-1) if lo else ({},{})
        try:
            with target.open('x') as stream:
                for fid in range(lo,hi):
                    masks,ids=load(fid);r=rows[fid];hints={}
                    for tid in ('track_00','track_01'):
                        e=r['evidence'][tid]
                        hint=evaluate_mask_motion_hint(previous[0].get(tid),masks.get(tid),track_id=tid,previous_identity=previous[1].get(tid),current_identity=ids[tid],boundary_evidence=e,policy=policy)
                        hint['source_boundary_evidence']=dict(v3_predictions=receipt['processed_predictions'],frame_id=fid,track_id=tid,policy_id=e['policy_id'],available=e['available'],hold_rotation=e['hold_rotation'],current_mask_authenticated=e['current_mask_authenticated'],reason=e['reason'])
                        if hint['eligible_for_gain'] and (e['available'] or e['hold_rotation'] or not hint['current_mask_authenticated']):raise AssertionError('Boundary-priority gain conflict')
                        hints[tid]=hint;counts['observations']+=1;counts['eligible_for_gain']+=int(hint['eligible_for_gain']);counts['scalar_available']+=int(hint['scalar_available']);counts['reason:'+hint['reason']]+=1
                    record=dict(schema='action4.mask_motion_hint.frame.v1',frame_id=fid,pts=r['pts'],time_base=time_base,RGB_sha256=r['RGB_sha256'],observations=hints,source_refs=refs)
                    stream.write(json.dumps(record,allow_nan=False,separators=(',',':'))+'\n');previous=(masks,ids)
                    if (fid-lo)%100==0:
                        stream.flush();(out/'parts'/f'part_{i:02d}.PROGRESS.json').write_text(json.dumps(dict(frame_id=fid,counts=counts,seconds=time.monotonic()-start)))
            result=dict(block=i,frames=[lo,hi],boundary_extra_frame=lo-1 if lo else None,mask_frames_loaded=loaded,counts=dict(counts),output=binding(target))
            (out/'parts'/f'part_{i:02d}.COMPLETE.json').write_text(json.dumps(result,indent=2));return result
        except Exception as exc:
            (out/'parts'/f'part_{i:02d}.FAILED.json').write_text(json.dumps(dict(error=repr(exc))));raise
    with ThreadPoolExecutor(max_workers=workers) as pool:parts=list(pool.map(block,range(workers)))
    destination=out/'MASK_MOTION_HINTS.jsonl';count=0;counts=Counter()
    with destination.open('x') as final:
        for part in parts:
            with verify(part['output']).open() as stream:
                for line in stream:
                    row=json.loads(line)
                    if row['frame_id']!=count or row['pts']!=rows[count]['pts'] or row['RGB_sha256']!=rows[count]['RGB_sha256']:raise ValueError('Merge order/identity mismatch')
                    final.write(line);count+=1
            counts.update(part['counts'])
    if count!=2850 or counts['observations']!=5700:raise ValueError('Incomplete mask hint coverage')
    # Inputs are immutable receipts; independently rehash V3 after parallel consumption.
    verify(receipt['processed_predictions']);verify(receipt['input_lock'])
    summary=dict(status='COMPLETE2850_MASK_ONLY_WEAK_HINT_NOT_DEPLOYED',frames=count,observations=counts['observations'],counts=dict(counts),seconds=time.monotonic()-start,parts=parts,source_refs=refs,output=binding(destination),policy=policy,policy_id=policy_identity(policy),new_model_calls=0,RGB_decodes=0,LK_calls=0,HOLD_state_run=False,source_boundary_flags_changed=False,production_enabled=False)
    (out/'COMPLETE.json').write_text(json.dumps(summary,indent=2))
    delivery=dict(schema='action4.mask_motion_hint.receipt.v1',status='COMPLETE2850_MASK_MOTION_HINT_READY_FOR_DEV_GAIN_ONLY',input_lock=receipt['input_lock'],
        v3_stabilization_receipt=receipt_ref,v3_predictions=receipt['processed_predictions'],raw_predictions=receipt['raw_predictions'],
        policy_id=policy_identity(policy),frames=2850,observations=5700,hints=binding(destination),complete=binding(out/'COMPLETE.json'),
        source_boundary_flags_changed=False,production_enabled=False,new_model_calls=0,RGB_decodes=0,LK_calls=0)
    (out/'MASK_MOTION_HINT_RECEIPT.json').write_text(json.dumps(delivery,indent=2))
    return summary


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    args=parser.parse_args();print(json.dumps(run_replay(args.config)),flush=True)