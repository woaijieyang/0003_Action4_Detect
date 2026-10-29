"""Read-only current-mask redecoding of saved actual video; no model inference."""
import copy,json,os,time,hashlib
from collections import Counter
from pathlib import Path
import numpy as np
from data.action4_paths import PROJECT_ROOT as ROOT,contained_path
from data.action4_sam_roi_inputs import array_digest
from eval.action4_roi_sweep import binding,read_binding
from eval.action4_corner_threshold_probe import load_module,compare

SCHEMA='action4.corner_video_q02_saved_replay.v1'
def restore_route(record,mask):
    route=copy.deepcopy(record['route'])
    digest=None if mask is None else array_digest(mask)
    if digest!=route['support_mask_sha256'] or (0 if mask is None else int(mask.sum()))!=route['support_mask_pixels']:
        raise ValueError('Saved/current native mask identity differs')
    if route['kind']!='coarse' or route['full_box_quality_passed'] is not False:raise ValueError('Coarse video route required')
    route.pop('support_mask_sha256');route.pop('support_mask_pixels');route['support_mask_native']=mask
    return route

def decode(module,record,mask):
    result=module.decode_single_frame_pose(**{k:record[k] for k in ('canonical_m','K_native','A224_from_native','raw_rotation','raw_translation_m','observations')},
        route=restore_route(record,mask),pixel_space_id=record['observations']['pixel_space_id'],raw_identity=record['roi_identity'],roi_identity=record['roi_identity'],
        quality_ready=False,diagnostic_candidate=mask is not None and bool(mask.any()),config=module.POLICY)
    if mask is None or not mask.any():result['reason']='current_video_mask_unavailable_raw_fallback'
    return result

def verify_projection(instance,K,cad):
    pose=instance['final_pose'];r=np.asarray(pose['rotation'],float);t=np.asarray(pose['translation_m'],float)
    if r.shape!=(3,3) or t.shape!=(3,) or not np.isfinite(r).all() or not np.isfinite(t).all() or not np.allclose(r.T@r,np.eye(3),atol=1e-4,rtol=0) or abs(np.linalg.det(r)-1)>1e-4:
        raise ValueError('Invalid final rigid pose')
    camera=np.asarray(cad)@r.T+t
    if (camera[:,2]<=0).any():
        if instance['corners_final'] is not None or instance['pose_valid']:raise ValueError('Nonpositive final depth')
        return
    h=camera@np.asarray(K).T;uv=h[:,:2]/h[:,2:];saved=np.asarray(instance['corners_final'])
    if saved.shape!=(8,2) or not np.isfinite(saved).all() or not np.allclose(saved,uv,atol=.01,rtol=1e-6):raise ValueError('Final corners not same-pose fixed CAD projection')
    if instance['corners_px']!=instance['corners_final'] or instance['rotation']!=pose['rotation'] or instance['translation_m']!=pose['translation_m']:raise ValueError('Final aliases differ')

def replace_candidate(raw,old,record,decoded):
    result=copy.deepcopy(raw);accepted=decoded['geometric_candidate_accepted']
    if accepted:
        rotation=decoded['candidate_rotation'];translation=decoded['candidate_translation_m'];points=decoded['candidate_corners_native']
        result.update(rotation=rotation,translation_m=translation,corners_px=points,corners_final=points,
            final_pose={'coordinate_frame':'object_to_capture_cv_m','finite':True,'rotation':rotation,'translation_m':translation},
            pose_valid=True,positive_depth_verified=True,post_A0_geometry='corner_q02_research_candidate')
    else:result['post_A0_geometry']='corner_q02_raw_fallback'
    result.update(pose_diagnostic={'method':'corner_q02_candidate_or_raw','reason':decoded['reason'],'geometric_candidate_accepted':accepted,'deployed':False,'valid':result['pose_valid']},
        corner_observations=copy.deepcopy(record['observations']),decoder_diagnostics=decoded,corner_route=copy.deepcopy(record['route']))
    for k in ('prediction_id','track_id','bbox_for_a0','roi_decision_id','raw_network_pose','corners_raw_from_resolved_roi'):
        if result[k]!=old[k]:raise ValueError('Immutable input/raw changed '+k)
    verify_projection(result,record['K_native'],record['canonical_m'])
    return result

def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)

def check_frame(row,provider,masks,receipt,manifest_frame):
    pred=row['prediction'];fid=row['frame_id']
    for k in ('frame_id','pts','RGB_sha256'):
        if row[k]!=receipt[k] or pred[k]!=row[k]:raise ValueError('Native/frame identity differs '+k)
    if row['pts']!=manifest_frame['pts'] or row['RGB_sha256']!=manifest_frame['image_sha256']:raise ValueError('Original frame manifest differs')
    if receipt!=pred['video_roi_consumption']['SAM_feature_receipt']:raise ValueError('Original current native receipt changed')
    if len(pred['instances'])!=2 or {p['track_id'] for p in pred['instances']}!={'track_00','track_01'}:raise ValueError('Full dual instance domain required')
    if len(pred['corner_observations'])!=2 or len(pred['raw_instances'])!=2:raise ValueError('Missing input/raw')
    for o,p,raw in zip(pred['corner_observations'],pred['instances'],pred['raw_instances']):
        if o['prediction_id']!=p['prediction_id'] or raw['prediction_id']!=p['prediction_id'] or p['prediction_id']!=f"video:{p['track_id']}:frame:{fid}":raise ValueError('Saved track instance mismatch')
        if o['frame_id']!=fid or o['pts']!=row['pts'] or o['track_id']!=p['track_id'] or o['bbox_for_a0']!=p['bbox_for_a0'] or o['roi_decision_id']!=p['roi_decision_id']:raise ValueError('Saved ROI/time mismatch')
        restore_route(o,masks.get(o['track_id']))

def contact(out,selected,frames):
    from PIL import Image,ImageDraw,ImageFont
    from eval.action4_video import draw_cuboids
    font=ImageFont.load_default(size=16);width,height=570,432;canvas=Image.new('RGB',(width*3,(height+32)*len(selected)),'black')
    for ri,(fid,(old,new)) in enumerate(sorted(selected.items())):
        frame=frames[fid];p=contained_path(ROOT,frame['image'],exists=True)
        if binding(p)['sha256']!=frame['image_sha256']:raise ValueError('Contact RGB changed')
        with Image.open(p) as im:base=im.convert('RGB');scale=np.array([width/base.width,height/base.height]);base=base.resize((width,height))
        for col,(label,instances) in enumerate([('raw',old['prediction']['raw_instances']),('q0.6',old['prediction']['instances']),('q0.2',new['prediction']['instances'])]):
            draw=[{'corners_px':(np.asarray(x['corners_final'])*scale).tolist()} for x in instances if x['pose_valid']]
            tile=draw_cuboids(base.copy(),draw,line_width=2,debug_numbers=True);canvas.paste(tile,(col*width,ri*(height+32)+32));d=ImageDraw.Draw(canvas)
            accepted=sum(x.get('decoder_diagnostics',{}).get('geometric_candidate_accepted',False) for x in instances)
            d.text((col*width+5,ri*(height+32)+5),f'frame {fid} | {label} | accepted {accepted}/2',font=font,fill='white')
    canvas.save(out/'CONTACT_RAW_Q06_Q02.jpg',quality=94)

def run(output='outputs/action4_corner_video_q02_replay_v001'):
    if set(os.sched_getaffinity(0))!={58,59} or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU58-59/GPU-hidden execution required')
    import torch;torch.set_num_threads(1);torch.set_num_interop_threads(1)
    from eval.action4_reanchor_roi_provider import ReanchorROIProvider
    out=contained_path(ROOT,output,exists=False);out.mkdir(exist_ok=True)
    source_receipt=binding('outputs/action4_corner6375_video_v002/CORNER_TRIAL_RECEIPT.json');seal=read_binding(source_receipt);predpath=contained_path(ROOT,seal['predictions']['path'],exists=True)
    if binding(predpath)!=seal['predictions'] or seal['frames']!=2850 or seal['instances']!=5700:raise ValueError('Full sealed prediction source changed')
    lock=read_binding(seal['input_lock']);frames=read_binding(lock['video']['frames_manifest'])['frames'];provider=ReanchorROIProvider(**lock['video']['track_replay'],register_project_bodies=True)
    if len(frames)!=2850 or [f['frame_id'] for f in frames]!=list(range(2850)):raise ValueError('Full frame manifest required')
    frozen=ROOT/'outputs/action4_corner_threshold_probe_v001/execution_v002';pre=read_binding(binding(frozen/'PREREGISTRATION.json'))
    qpath=contained_path(ROOT,pre['copies']['q0p2']['path'],exists=True)
    if binding(qpath)!=pre['copies']['q0p2']:raise ValueError('Accepted probe decoder copy changed')
    basepath=frozen/'source/decoder_q0p6.py';base=load_module(basepath,'video_probe_base');variant=load_module(qpath,'video_probe_q02')
    if {k for k in base.POLICY if base.POLICY[k]!=variant.POLICY[k]}!={'q_corner_min','schema'} or variant.POLICY['q_corner_min']!=.2:raise ValueError('Unexpected q variant')
    identity={'source_receipt':source_receipt,'source_predictions':seal['predictions'],'input_lock':seal['input_lock'],'qpolicy_module':binding(qpath),'baseline_module':binding(basepath),'accepted_dev_probe':binding(frozen/'COMPLETE.json'),'qpolicy':variant.POLICY,'schema':SCHEMA,'model_calls':0,'GT_used':False,'post_filter':False}
    write(out/'INPUT_BINDINGS.json',identity)
    checks=[];probe_ids=set(range(6))|{1163};contact_ids={4,1163,1195,1289,1950,2400}
    with predpath.open() as stream:
        for line in stream:
            row=json.loads(line);fid=row['frame_id']
            if fid not in probe_ids:continue
            masks,receipt=provider.current_native_masks(fid);check_frame(row,provider,masks,receipt,frames[fid])
            for record in row['prediction']['corner_observations']:
                decoded=decode(base,record,masks.get(record['track_id']));compare(decoded,record['decoder_diagnostics']);checks.append({'frame_id':fid,'track_id':record['track_id'],'reason':decoded['reason'],'matched':True})
            if fid==max(probe_ids):break
    if len(checks)!=14:raise ValueError('Baseline preflight missing records')
    write(out/'BASELINE_REPLAY.json',{'status':'REAL_VIDEO_Q06_REPLAY_MATCH','records':checks,'numeric_atol':1e-8,'numeric_rtol':1e-9})
    started=time.monotonic();reasons=Counter();idcounts=Counter();accepted=0;total=0;selected={};old_reasons=Counter();changes=0
    with predpath.open() as src,(out/'predictions.jsonl').open('x') as dst:
        for expected,line in enumerate(src):
            row=json.loads(line);fid=row['frame_id']
            if fid!=expected or fid>=2850:raise ValueError('Prediction frame order changed')
            masks,receipt=provider.current_native_masks(fid);check_frame(row,provider,masks,receipt,frames[fid]);new=copy.deepcopy(row);pred=new['prediction'];pred['schema']=SCHEMA;pred['qpolicy_binding']=identity['qpolicy_module'];pred['source_schema']=row['prediction']['schema'];new['threshold_replay_binding']=binding(out/'INPUT_BINDINGS.json')
            new_instances=[]
            for i,(record,raw,old) in enumerate(zip(pred['corner_observations'],pred['raw_instances'],row['prediction']['instances'])):
                d=decode(variant,record,masks.get(record['track_id']));old_reasons[record['decoder_diagnostics']['reason']]+=1
                p=replace_candidate(raw,old,record,d);record['source_decoder_diagnostics']=record['decoder_diagnostics'];record['decoder_diagnostics']=d;record['qpolicy_binding']=identity['qpolicy_module'];new_instances.append(p)
                reasons[d['reason']]+=1;idcounts[len(d['used_corner_ids'])]+=1;accepted+=int(d['geometric_candidate_accepted']);total+=1
                changes+=int(p['corners_final']!=old['corners_final'])
            pred['instances']=new_instances;pred['candidate_instances']=copy.deepcopy(new_instances)
            if pred['raw_instances']!=row['prediction']['raw_instances']:raise ValueError('Raw branch changed')
            dst.write(json.dumps(new,allow_nan=False)+'\n')
            if fid in contact_ids:selected[fid]=(row,new)
            if (fid+1)%100==0:
                status={'frames':fid+1,'instances':total,'accepted':accepted,'seconds':time.monotonic()-started};(out/'STATUS.json').write_text(json.dumps(status));print(json.dumps(status),flush=True)
    if expected+1!=2850 or total!=5700:raise ValueError('Incomplete full domain')
    if binding(predpath)!=seal['predictions']:raise ValueError('Source changed during replay')
    write(out/'COMPLETE.json',{'status':'FULL2850_Q02_SAVED_REPLAY_NOT_ACCURACY_ACCEPTANCE','schema':SCHEMA,'frames':2850,'instances':total,'accepted_candidates':accepted,'changed_final_instances':changes,'decoder_reasons':dict(reasons),'baseline_reasons':dict(old_reasons),'used_ID_histogram':dict(idcounts),'inputs':binding(out/'INPUT_BINDINGS.json'),'predictions':binding(out/'predictions.jsonl'),'quality_ready':False,'deployment_admitted':False,'model_calls':0,'GT_used':False,'all_final_rigid_projections_verified':True,'raw_and_ROI_preserved':True,'temporal_filter':False,'seconds':time.monotonic()-started})
    contact(out,selected,frames);write(out/'CONTACT.json',{'frames':sorted(selected),'columns':['raw','q0.6','q0.2'],'image':binding(out/'CONTACT_RAW_Q06_Q02.jpg'),'visual_quality_or_accuracy_acceptance':False})
    print(json.dumps({'status':'COMPLETE','frames':2850,'instances':5700,'accepted':accepted,'changed':changes,'reasons':dict(reasons)}),flush=True)
