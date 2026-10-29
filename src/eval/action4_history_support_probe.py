"""Fixed-window history pose support probe. Hypotheses only; no pose publication."""
from pathlib import Path
from dataclasses import asdict
from fractions import Fraction
from itertools import zip_longest
from collections import Counter
import hashlib,json,os,time,copy,multiprocessing
import numpy as np
from model.action4_pose_support_consistency import pose_support_consistency,SupportPolicy
from model.action4_video_motion_evidence import _pose

ROOT=Path('/data/zy/vln/a-test')
WINDOWS=((100,120),(1180,1210),(2390,2410),(2760,2790))
VISUAL_FRAMES={109,1195,2400,2778}
_CTX=None


class InputIdentityError(ValueError):pass


def path(name):
    p=(ROOT/name).resolve()
    if not p.is_relative_to(ROOT):raise InputIdentityError('Outside project')
    return p


def binding(name):
    p=path(name)
    with p.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    return {'path':str(p.relative_to(ROOT)),'sha256':sha}


def read(ref):
    p=path(ref['path'])
    if binding(p)!=ref:raise InputIdentityError('Bound input changed '+ref['path'])
    return json.loads(p.read_text())


def get_pose(instance):
    p=instance.get('final_pose',{})
    if p.get('coordinate_frame')!='object_to_capture_cv_m':raise InputIdentityError('Selected final_pose coordinate frame differs')
    return _pose(p.get('rotation'),p.get('translation_m'))


def map_translation(translation,K,source_box,current_box):
    t=np.asarray(translation,float);k=np.asarray(K,float);a=np.asarray(source_box,float);b=np.asarray(current_box,float)
    if t.shape!=(3,) or k.shape!=(3,3) or a.shape!=(4,) or b.shape!=(4,) or not all(np.isfinite(x).all() for x in (t,k,a,b)) or t[2]<=0 or (a[2:]<=a[:2]).any() or (b[2:]<=b[:2]).any():raise ValueError('Finite positive-depth translation and full ROI boxes required')
    if not np.array_equal(k[2],[0,0,1]):raise InputIdentityError('Native pinhole K required')
    ds=float(np.linalg.norm(a[2:]-a[:2]));dc=float(np.linalg.norm(b[2:]-b[:2]));cs=(a[:2]+a[2:])/2;cc=(b[:2]+b[2:])/2
    h=k@t;uv=h[:2]/h[2];offset=(uv-cs)/ds;new_uv=cc+offset*dc;z=float(t[2]*ds/dc);mapped=np.linalg.solve(k,np.r_[new_uv,1])*z
    return mapped,dict(source_box_native=a.tolist(),current_box_native=b.tolist(),source_diagonal_native_px=ds,current_diagonal_native_px=dc,source_projected_origin_native=uv.tolist(),ROI_relative_offset=offset.tolist(),current_projected_origin_native=new_uv.tolist(),source_z_m=float(t[2]),mapped_z_m=z,additional_ROI_expansion=False,semantics='history_initialization_hypothesis_not_measurement')


def temporal_eligibility(source_frame,current_frame,source_pts,current_pts,time_base,source_receipt,current_receipt):
    age=current_frame-source_frame;dt=Fraction(current_pts-source_pts)*Fraction(*time_base) if isinstance(time_base,(list,tuple)) else Fraction(current_pts-source_pts)*Fraction(time_base)
    if age<0 or age>5 or dt<0 or dt>Fraction(1,4):return False,'outside_five_frame_or_quarter_second_window',float(dt)
    if any(source_receipt.get(k) is None or source_receipt.get(k)!=current_receipt.get(k) for k in ('session_id','generation','pixel_space_id')):return False,'native_session_generation_or_pixel_space_changed',float(dt)
    if source_receipt.get('time_base')!=current_receipt.get('time_base'):return False,'native_timebase_changed',float(dt)
    return True,'eligible_temporal_domain',float(dt)


def support_check(mask,other,K,canonical,rotation,translation):
    if mask is None:return dict(available=False,reference_compatible=False,reasons=['current_authenticated_mask_missing'])
    try:return pose_support_consistency(mask,K,canonical,rotation,translation,policy=SupportPolicy(),other_instance_mask=other)
    except InputIdentityError:raise
    except Exception as exc:return dict(available=False,reference_compatible=False,reasons=['support_exception:'+type(exc).__name__],error=str(exc))


def angle_degrees(a,b):
    a,_=_pose(a,[0,0,1]);b,_=_pose(b,[0,0,1]);return float(np.rad2deg(np.arccos(np.clip((np.trace(a@b.T)-1)/2,-1,1))))


def choose_candidate(candidates,previous_rotation):
    compatible=[c for c in candidates if c.get('support',{}).get('reference_compatible') is True]
    if not compatible:return None
    for c in compatible:c['angle_to_previous_V4_deg']=angle_degrees(c['rotation'],previous_rotation)
    return min(compatible,key=lambda c:(c['angle_to_previous_V4_deg'],-c['source_frame_id']))


def _worker(index):
    import cv2
    from eval.action4_sam_full_review import ReceiptVerifier
    cv2.setNumThreads(4);cpus=set(range(64+4*index,68+4*index));os.sched_setaffinity(0,cpus)
    ctx=_CTX;lo,hi=WINDOWS[index];out=ctx['out'];provider=copy.copy(ctx['provider']);provider.verifier=ReceiptVerifier();receipts={};records=[];K=ctx['K'];cad=ctx['canonical'];shape=ctx['shape']
    target=out/f'WINDOW_{lo}_{hi}.jsonl';start=time.monotonic()
    try:
        with target.open('x') as stream:
            for f in range(lo-5,hi):
                masks,receipt=provider.current_native_masks(f);raw=ctx['raw'][f]
                if receipt['frame_id']!=f or receipt['pts']!=raw['pts'] or receipt['RGB_sha256']!=raw['RGB_sha256'] or receipt['time_base']!=ctx['time_base']:raise InputIdentityError('Native mask current frame/PTS/RGB/time identity differs')
                receipts[f]=receipt
                if f<lo:continue
                current=ctx['v4'][f];previous=ctx['v4'][f-1]
                for tid in ('track_00','track_01'):
                    current_inst=raw['instances'][tid];v4_inst=current['instances'][tid];prev_inst=previous['instances'][tid]
                    mask=masks.get(tid);other=np.zeros(shape,bool)
                    for key,value in masks.items():
                        if key!=tid:other|=value
                    row=dict(frame_id=f,pts=raw['pts'],RGB_sha256=raw['RGB_sha256'],time_base=ctx['time_base'],track_id=tid,
                        current_mask_authenticated=mask is not None,current_mask_receipt=receipt,other_mask_present=any(key!=tid for key in masks),current_bbox_for_a0=current_inst['bbox_for_a0'],candidates=[],proposal=None,proposal_selection='minimum_angle_to_previous_V4_then_newest_source_frame',geometry_accepted=False,published=False)
                    try:
                        vr,vt=get_pose(v4_inst);v4_support=support_check(mask,other,K,cad,vr,vt);row['V4_pose']={'rotation':vr.tolist(),'translation_m':vt.tolist()}
                    except InputIdentityError:raise
                    except Exception as exc:vr=vt=None;v4_support=dict(available=False,reference_compatible=False,reasons=['V4_pose_exception:'+type(exc).__name__],error=str(exc))
                    row['V4_support']=v4_support
                    try:prevR,_=get_pose(prev_inst);row['previous_V4_frame_id']=f-1;row['previous_V4_rotation']=prevR.tolist()
                    except InputIdentityError:raise
                    except Exception as exc:prevR=None;row['selection_reference_error']=str(exc)
                    for sf in range(f-5,f+1):
                        source=ctx['raw'][sf];inst=source['instances'][tid];src_receipt=receipts[sf]
                        temporal,reason,age_seconds=temporal_eligibility(sf,f,source['pts'],raw['pts'],ctx['time_base'],src_receipt,receipt)
                        candidate=dict(source_frame_id=sf,source_pts=source['pts'],source_prediction_id=inst['prediction_id'],track_id=tid,corner_ids=inst['corner_ids'],age_frames=f-sf,age_seconds=age_seconds,source_session_id=src_receipt.get('session_id'),source_generation=src_receipt.get('generation'),source_RGB_sha256=source['RGB_sha256'],temporal_eligible=temporal,temporal_reason=reason,geometry_accepted=False)
                        if not temporal:candidate['support']=dict(available=False,reference_compatible=False,reasons=[reason])
                        else:
                            try:
                                rr,tt=get_pose(inst);mapped,mapping=map_translation(tt,K,inst['bbox_for_a0'],current_inst['bbox_for_a0'])
                                candidate.update(rotation=rr.tolist(),translation_m=mapped.tolist(),source_translation_m=tt.tolist(),translation_mapping=mapping)
                                candidate['support']=support_check(mask,other,K,cad,rr,mapped)
                                if prevR is not None:candidate['angle_to_previous_V4_deg']=angle_degrees(rr,prevR)
                            except InputIdentityError:raise
                            except Exception as exc:candidate['support']=dict(available=False,reference_compatible=False,reasons=['candidate_exception:'+type(exc).__name__],error=str(exc))
                        row['candidates'].append(candidate)
                    compatible=[c for c in row['candidates'] if c['support'].get('reference_compatible') is True];row['compatible_candidate_count']=len(compatible)
                    if compatible and prevR is not None:
                        proposal=choose_candidate(row['candidates'],prevR);row['proposal']=copy.deepcopy(proposal)
                        if vr is not None:row['proposal']['angle_to_current_V4_deg']=angle_degrees(proposal['rotation'],vr)
                    row['opportunity_V4_failed_or_incompatible_has_candidate']=not v4_support.get('reference_compatible',False) and bool(compatible)
                    row['opportunity_V4_available_incompatible_has_candidate']=v4_support.get('available',False) and row['opportunity_V4_failed_or_incompatible_has_candidate']
                    row['no_compatible_candidate']=not compatible
                    if f in VISUAL_FRAMES:row['overlay']=_overlay(row,ctx)
                    records.append(row);stream.write(json.dumps(row,allow_nan=False,separators=(',',':'))+'\n');stream.flush()
                receipts.pop(f-5,None)
        result=dict(window=[lo,hi],CPU=sorted(cpus),records=len(records),candidates=sum(len(r['candidates']) for r in records),seconds=time.monotonic()-start,output=str(target.relative_to(ROOT)))
        (out/f'WINDOW_{lo}_{hi}_COMPLETE.json').write_text(json.dumps(result,indent=2));return records
    except Exception as exc:
        (out/f'WINDOW_{lo}_{hi}_FAILED.json').write_text(json.dumps(dict(error=repr(exc)),indent=2));raise


def _overlay(row,ctx):
    import cv2
    f=row['frame_id'];ref=ctx['frames'][f];raw=path(ref['image']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=ref['image_sha256']:raise InputIdentityError('Visual RGB changed')
    image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR);panels=[];corners=[];cad=ctx['canonical'];K=ctx['K'];h,w=image.shape[:2]
    for pose in [row.get('V4_pose'),row['proposal']]:
        uv=None
        if pose is not None:
            c=cad@np.asarray(pose['rotation']).T+np.asarray(pose['translation_m']);projected=c@K.T
            if np.isfinite(projected).all() and (c[:,2]>1e-6).all():uv=projected[:,:2]/projected[:,2:]
        corners.append(uv)
    bbox=np.asarray(row['current_bbox_for_a0']);valid=[u for u in corners if u is not None and abs(u).max()<1e6]
    if valid:bbox=np.r_[np.minimum(bbox[:2],np.concatenate(valid).min(0)),np.maximum(bbox[2:],np.concatenate(valid).max(0))]
    l,t=np.maximum(np.floor(bbox[:2]-20).astype(int),0);r,b=np.minimum(np.ceil(bbox[2:]+20).astype(int),[w,h])
    if r<=l or b<=t:l=t=0;r=w;b=h
    edges=[(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for index,uv in enumerate(corners):
        panel=image[t:b,l:r].copy();color=(0,0,255) if index==0 else (0,255,255)
        if uv is not None:
            pts=np.rint(uv-[l,t]).astype(int)
            for a,z in edges:cv2.line(panel,tuple(pts[a]),tuple(pts[z]),color,2,cv2.LINE_AA)
            for cid,point in enumerate(pts):cv2.putText(panel,str(cid),tuple(point),cv2.FONT_HERSHEY_SIMPLEX,.6,color,2,cv2.LINE_AA)
        compatible=row['V4_support'].get('reference_compatible',False) if index==0 else row['proposal'] is not None
        label=f"{f} {row['track_id']} "+('V4 actual final_pose' if index==0 else 'NO COMPATIBLE PROPOSAL' if row['proposal'] is None else 'HISTORY HYPOTHESIS source='+str(row['proposal']['source_frame_id']))+f' compatible={compatible}'
        header=np.zeros((45,panel.shape[1],3),np.uint8);cv2.putText(header,label,(4,28),cv2.FONT_HERSHEY_SIMPLEX,.44,(255,255,255),1,cv2.LINE_AA);panels.append(np.concatenate([header,panel]))
    result=np.concatenate(panels,axis=1);p=ctx['out']/'overlays'/f"FRAME_{f:04d}_{row['track_id']}.jpg";cv2.imwrite(str(p),result,[cv2.IMWRITE_JPEG_QUALITY,94])
    return dict(path=str(p.relative_to(ROOT)),native_crop_xyxy=[int(l),int(t),int(r),int(b)],rescaled=False,visual_margin_only_px=20)


def run_probe(config_path):
    global _CTX
    outcfg=json.loads(path(config_path).read_text());out=path(outcfg['output']);out.mkdir(exist_ok=True)
    if (out/'RUN_SPEC.json').exists():raise FileExistsError('Existing diagnostic may not be restarted')
    (out/'overlays').mkdir()
    receiptref=outcfg['v4_receipt'];receipt=read(receiptref)
    if receipt['frames']!=2850 or not receipt['validated_all_rows']:raise InputIdentityError('Complete2850 V4 required')
    rawref=receipt['raw_predictions'];v4ref=receipt['processed_predictions'];lockref=receipt['input_lock']
    if binding(rawref['path'])!=rawref or binding(v4ref['path'])!=v4ref:raise InputIdentityError('Raw/V4 file SHA mismatch')
    lock=read(lockref);camera=read(lock['video']['K']);K=np.array(camera['K_native'],float);asset=read(lock['CAD']);cad=np.array(asset['corners_3d_mm'],float)/1000
    manifest=read(lock['video']['frames_manifest']);needed={f for lo,hi in WINDOWS for f in range(lo-5,hi)};rawrows={};v4rows={};total=0
    with path(rawref['path']).open() as rawstream,path(v4ref['path']).open() as v4stream:
        for a,b in zip_longest(rawstream,v4stream):
            if a is None or b is None:raise InputIdentityError('Raw/V4 EOF differs')
            raw=json.loads(a);v4=json.loads(b);frame=manifest['frames'][total]
            if raw['frame_id']!=total or any(raw[k]!=v4[k] for k in ('frame_id','pts','RGB_sha256','track_replay_identity')) or raw['pts']!=frame['pts'] or raw['RGB_sha256']!=frame['image_sha256']:raise InputIdentityError('Raw/V4/frame manifest identity mismatch')
            for row in (raw,v4):
                models=row['prediction']['identity']
                if models['A0_step']!=7300 or models['B0_step']!=9500:raise InputIdentityError('Wrong frozen model parent')
                instances={x['track_id']:x for x in row['prediction']['instances']}
                if len(instances)!=len(row['prediction']['instances']) or set(instances)!={'track_00','track_01'}:raise InputIdentityError('Instance domain differs')
                for tid,inst in instances.items():
                    if inst['frame_id']!=total or inst['pts']!=row['pts'] or inst['corner_ids']!=list(range(8)):raise InputIdentityError('Fixed frame/corner IDs differ')
                    if Fraction(*inst['time_base'])!=Fraction(manifest['source']['time_base']):raise InputIdentityError('Instance timebase mismatch')
                if total in needed:
                    compact=dict(frame_id=total,pts=row['pts'],RGB_sha256=row['RGB_sha256'],instances=instances)
                    (rawrows if row is raw else v4rows)[total]=compact
            total+=1
    if total!=2850:raise InputIdentityError('Incomplete source domain')
    from eval.action4_reanchor_roi_provider import ReanchorROIProvider
    provider=ReanchorROIProvider(**lock['video']['track_replay'],register_project_bodies=True)
    _CTX=dict(out=out,raw=rawrows,v4=v4rows,provider=provider,K=K,canonical=cad,frames=manifest['frames'],shape=(manifest['source']['height'],manifest['source']['width']),time_base=[Fraction(manifest['source']['time_base']).numerator,Fraction(manifest['source']['time_base']).denominator])
    refs=dict(raw=rawref,V4=v4ref,V4_receipt=receiptref,input_lock=lockref,K=lock['video']['K'],CAD=lock['CAD'],frames_manifest=lock['video']['frames_manifest'])
    (out/'RUN_SPEC.json').write_text(json.dumps(dict(windows=WINDOWS,current_frames=100,tracks=2,candidate_frames='f-5..f raw selected final_pose',support_policy=asdict(SupportPolicy()),source_refs=refs,model_pose_source='raw.final_pose; V4.final_pose current and f-1 only; top-level V4 rotation is NOT filtered output',historical_translation='ROI-relative hypothesis only, no added ROI expansion',GPU=False,new_model_calls=0),indent=2))
    with multiprocessing.get_context('fork').Pool(4) as pool:groups=pool.map(_worker,range(4))
    rows=[r for group in groups for r in group];assert len(rows)==200
    with (out/'RECORDS.jsonl').open('x') as f:
        for r in rows:f.write(json.dumps(r,allow_nan=False,separators=(',',':'))+'\n')
    candidate_reasons=Counter(reason for r in rows for c in r['candidates'] for reason in c['support'].get('reasons',[]));v4_reasons=Counter(reason for r in rows for reason in r['V4_support'].get('reasons',[]));proposals=[r['proposal'] for r in rows if r['proposal'] is not None]
    summary=dict(status='FIXED100FRAME_HISTORY_SUPPORT_DIAGNOSTIC_COMPLETE_NOT_PUBLISHED',frames=100,records=len(rows),candidate_records=sum(len(r['candidates']) for r in rows),V4_compatible=sum(r['V4_support'].get('reference_compatible',False) for r in rows),has_compatible_candidate=sum(r['compatible_candidate_count']>0 for r in rows),no_compatible_candidate=sum(r['no_compatible_candidate'] for r in rows),opportunity_V4_failed_or_incompatible=sum(r['opportunity_V4_failed_or_incompatible_has_candidate'] for r in rows),opportunity_V4_available_incompatible=sum(r['opportunity_V4_available_incompatible_has_candidate'] for r in rows),proposal_count=len(proposals),proposal_source_ages_frames=dict(Counter(p['age_frames'] for p in proposals)),proposal_angle_to_previous_V4_deg=[p['angle_to_previous_V4_deg'] for p in proposals],candidate_failure_reasons=dict(candidate_reasons),V4_failure_reasons=dict(v4_reasons),input_refs=refs,geometry_accepted=False,published=False,model_calls=0,GPU=False)
    (out/'RESULT.json').write_text(json.dumps(summary,indent=2));_contact(rows,out);return summary


def _contact(rows,out):
    import cv2
    cv2.setNumThreads(1)
    selected=[r for r in rows if r['frame_id'] in VISUAL_FRAMES];assert len(selected)==8
    sheet=np.zeros((4*420,2*1000,3),np.uint8)
    for i,row in enumerate(selected):
        im=cv2.imread(str(path(row['overlay']['path'])));scale=min(990/im.shape[1],385/im.shape[0]);im=cv2.resize(im,(round(im.shape[1]*scale),round(im.shape[0]*scale)),interpolation=cv2.INTER_AREA);y=(i//2)*420;x=(i%2)*1000;sheet[y:y+im.shape[0],x:x+im.shape[1]]=im
        label=f"{row['frame_id']} {row['track_id']} V4compat={row['V4_support'].get('reference_compatible')} candidates={row['compatible_candidate_count']} proposal={None if row['proposal'] is None else row['proposal']['source_frame_id']}"
        cv2.putText(sheet,label,(x+5,y+410),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
    cv2.imwrite(str(out/'FIXED4_FRAME_DUAL_CONTACT.jpg'),sheet,[cv2.IMWRITE_JPEG_QUALITY,94])