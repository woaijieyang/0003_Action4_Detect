"""Same-frame masks, canonical geometry and exact coverage quota validation."""
from pathlib import Path
from copy import deepcopy
import json,hashlib,os
import numpy as np
from PIL import Image
from data import DataAdapter
from data.action4_paths import PROJECT_ROOT as R,contained_path
from data.action4_coverage1000_contract import actual_accept,source_eligible
from data.action4_phrase2_visibility import canonical_visibility


def validate(record,proposal,slot,cfg):
    DataAdapter(R).validate_sample(record)
    expected=slot['sample_id']
    if record.get('output_role')=='dataset_candidate':
        if record['sample_id']!=expected or proposal['entry']['sample_id']!=expected or record['split']!=slot['split']:raise ValueError('Candidate sample/split identity mismatch')
    elif record.get('output_role')!='diagnostic' or record['sample_id'] not in ['arm_probe_'+expected,'diagnostic_'+expected]:raise ValueError('Unexpected output role or diagnostic identity')
    if (record['width'],record['height'])!=(cfg['width'],cfg['height']):raise ValueError('Native dimensions changed')
    files=[(record['image'],record['rgb_sha256'])]
    size=np.array([record['width'],record['height']]);K=np.array(record['camera']['K']);cad=np.array(record['corners_3d_mm'])/1000
    for obj,side in zip(record['instances'],['left','right']):
        g=obj['canonical8_geometry'];T=np.array(g['object_to_capture_cv_m']);uv=np.array(obj['corners_px'])
        cam=cad@T[:3,:3].T+T[:3,3];h=cam@K.T
        if uv.shape!=(8,2) or not np.isfinite(uv).all() or (cam[:,2]<=0).any() or abs(h[:,:2]/h[:,2:]-uv).max()>.005:raise ValueError('Independent canonical8 projection failed')
        if not np.allclose(np.r_[uv.min(0),uv.max(0)],obj['bbox_full_xyxy'],atol=1e-6,rtol=0):raise ValueError('Full body box differs from complete CAD projection')
        with Image.open(contained_path(R,obj['mask_visible'],exists=True)) as image:
            if image.size!=tuple(size):raise ValueError('Visible mask shape differs')
            visible=np.asarray(image)>0
        with Image.open(contained_path(R,obj['mask_amodal_expanded'],exists=True)) as image:
            if image.size!=tuple(size*3):raise ValueError('Amodal shape differs')
            full=np.asarray(image)>0
        amodal=full[size[1]:2*size[1],size[0]:2*size[0]];av=int(visible.sum());ai=int(amodal.sum());af=int(full.sum())
        if min(ai,af)<=0:raise ValueError('Empty body mask')
        if int((visible&~amodal).sum())>max(4,.005*ai):raise ValueError('Mask alignment outside admitted boundary tolerance')
        o=max(0.,1-av/ai);trunc=1-ai/af
        if abs(o-obj['occlusion_inframe'])>1e-12 or abs(trunc-obj['truncation'])>1e-12:raise ValueError('Mask area identity mismatch')
        if not source_eligible(o,record['occlusion_sources'][side]['primary_source_class']):return False,'actual_occlusion_source_'+side,None
        if o>.35 and record['occlusion_sources'][side]['primary_occluder_object']==side+'_forearm_continuation':return False,'upperarm_may_not_replace_wrist_forearm_occlusion',None
        pose=proposal['actor_pose_capture_by_side'][side];B=np.array(pose['brace_B_capture']);A=np.array(proposal['sides'][side]['A_brace_from_CAD'])
        if not np.allclose(B@A,pose['core_C_capture'],atol=1e-9,rtol=0):raise ValueError('Fixed installation failed')
        if not np.allclose(T,pose['core_C_capture'],atol=2e-5,rtol=0):raise ValueError('Rendered pose differs from immutable frame')
        rotation=T[:3,:3]
        if not np.allclose(rotation.T@rotation,np.eye(3),atol=2e-5,rtol=0) or not np.isclose(np.linalg.det(rotation),1.,atol=2e-5):raise ValueError('Non-rigid body transform')
        files.extend((obj[k],obj[k+'_sha256']) for k in ['mask_visible','mask_amodal_expanded'])
    ok,why,stats=actual_accept(slot,record)
    if not ok:return ok,why,stats
    for path,sha in files:
        if hashlib.sha256(contained_path(R,path,exists=True).read_bytes()).hexdigest()!=sha:raise ValueError('Frame file identity differs')
    return True,None,stats


def publish(record,proposal,slot,stats,cfg):
    result=deepcopy(record);root=R/cfg['candidate_root']/slot['split']
    for name in ['images','masks','geometry']:(root/name).mkdir(parents=True,exist_ok=True)
    mappings=[('image',None,root/'images'/(slot['sample_id']+'.png'))]
    for obj in result['instances']:
        for key,suffix in [('mask_visible','visible'),('mask_amodal_expanded','amodal_expanded')]:mappings.append((key,obj,root/'masks'/f"{slot['sample_id']}_i{obj['instance_id']}_{suffix}.png"))
        mappings.append(('surface_supervision',obj,root/'geometry'/f"{slot['sample_id']}_i{obj['instance_id']}.npz"))
    for key,obj,dest in mappings:
        value=result[key] if obj is None else obj[key]['path'] if key=='surface_supervision' else obj[key]
        source=contained_path(R,value,exists=True);dest=contained_path(R,dest,exists=False)
        if dest.exists():
            if not os.path.samefile(source,dest):raise FileExistsError('Never overwrite any accepted sample')
        else:os.link(source,dest)
        if obj is None:result[key]=str(dest.relative_to(R))
        elif key=='surface_supervision':obj[key]['path']=str(dest.relative_to(R))
        else:obj[key]=str(dest.relative_to(R))
    result.update(status='training_accepted',training_accepted=True,dataset_generation_allowed=True,pending_host_png_validation=False,
        dataset_root='data/phrase2',slot_id=slot['slot_id'],attempt=proposal['phrase2']['attempt'],scene_id=slot['scene_id'],
        phrase2=deepcopy(proposal['phrase2']),actual_phrase2_measurements=stats,occlusion_mode='arm_geometry',
        external_occlusion_semantics='actual visible/amodal inframe body area; truncation separate',
        actor_pose_capture_by_side=deepcopy(proposal['actor_pose_capture_by_side']),
        fixed_installation_by_side={s:{k:deepcopy(proposal['sides'][s][k]) for k in ['A_brace_from_CAD','S_skin_relative_to_brace','F_body_from_brace']} for s in ['left','right']},
        environment_controls=deepcopy(proposal['environment_controls']),render_seed=proposal['entry']['seed'],corner_ids=list(range(8)),
        render_profile_version='phrase2_native32_mask1_optix_v001',batch_id=cfg['batch_id'],batch_admission='separately_checked_pending_global_merge',
        occlusion_bin_definition={'batch':cfg['batch_id'],'bins':['[0,.40)','[.40,.50)','[.50,.60)','[.60,.75]'],'joint_statistics_use':'occlusion_inframe'})
    for obj,side in zip(result['instances'],['left','right']):
        note=proposal['phrase2']['by_side'][side];o=obj['occlusion_inframe']
        obj.update(wearing_side=side,corner_ids=list(range(8)),requested_center_uv=note['requested_center_uv'],bbox_center_uv=note['bbox_center_uv'],
            actual_width_fraction=note['actual_width_fraction'],occlusion_bin=0 if o<.4 else 1 if o<.5 else 2 if o<.6 else 3,
            optical_Z_m=note['optical_Z_m'],euclidean_CAD_origin_distance_m=note['euclidean_CAD_origin_distance_m'])
        g=obj['canonical8_geometry'];flags=canonical_visibility(cad:=np.array(result['corners_3d_mm'])/1000,np.array(g['object_to_capture_cv_m']),g['ray_clear_to_virtual_point'],g['point_inframe'])
        obj.update(corner_ray_clear=flags['ray_clear'],corner_self_visible=flags['self_visible'],corner_visible=flags['visible'],corner_visibility_semantics=flags['semantics'])
        g.update(training_accepted=True,training_corner_visible=flags['visible'],training_visibility_semantics=flags['semantics'],claim='Same-frame CAD8 / actual pixels / rigid installation checked')
    result['corner_visibility_semantics']=flags['semantics'];return result
