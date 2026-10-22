"""Bounded deterministic modular candidates; pure CPU, never scene/render/training.

Only complete actor rigid transforms and declared environment controls vary.
The immutable source plan supplies identity/quotas; unsupported family effects
remain explicit pending and can never silently become production data.
"""
from pathlib import Path
from functools import lru_cache
from collections import Counter,defaultdict
from itertools import product
import copy,hashlib,json,math
import numpy as np
from data.action4_global_environment import digest
ROOT=Path('/data/zy/vln/a-test')
CONFIG=ROOT/'configs/data/action4_modular_parameter_bounds_v001_candidate.json'
FAMILIES=('wall_rear','table_fastening','table_turn','near_double','self_forearm_occlusion','cloth_lift','wall_transition_truncation','hard_negative')
ENTRY_KEYS={'sample_id','index','seed','scene_family','target_count','group_id','background_source_id','background_seed','split','target_presence'}
SPLITS={'train':1600,'eval':200,'test':200}
QUOTAS_TEST_DUAL={'train':{'0':48,'1':80,'2':1472},'eval':{'0':6,'1':10,'2':184},'test':{'0':0,'1':0,'2':200}}
QUOTAS_92={'train':{'0':48,'1':80,'2':1472},'eval':{'0':6,'1':10,'2':184},'test':{'0':6,'1':10,'2':184}}
QUOTAS={'train':{'0':80,'1':240,'2':1280},'eval':{'0':10,'1':30,'2':160},'test':{'0':10,'1':30,'2':160}}
CAPS={'translation_capture_each_axis_m':.004,'rotation_brace_local_each_axis_deg':8.,'area_light_gain':1.1,'exposure_offset_ev':.15}

class PendingParametersError(ValueError):
    def __init__(self,reasons):self.reasons=list(reasons);super().__init__(json.dumps({'status':'PENDING_NOT_PRODUCTION','reasons':self.reasons}))

def bounded(value):
    p=Path(value);p=(p if p.is_absolute() else ROOT/p).resolve()
    if not p.is_relative_to(ROOT):raise ValueError('Project boundary')
    return p

@lru_cache(maxsize=64)
def _verified_bytes(path,expected,size,mtime_ns):
    data=Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest()!=expected:raise ValueError('Frozen parameter source changed '+path)
    return data

def verified_bytes(ref):
    if set(ref)!={'path','sha256'} or len(ref['sha256'])!=64:raise ValueError('Explicit source reference required')
    p=bounded(ref['path']);s=p.stat();return _verified_bytes(str(p),ref['sha256'],s.st_size,s.st_mtime_ns)

def _explicit_quota_shape(quotas):
    return isinstance(quotas,dict) and set(quotas)==set(SPLITS) and all(isinstance(row,dict) and set(row)=={'0','1','2'} and all(type(v) is int and v>=0 for v in row.values()) for row in quotas.values())

def audit_plan(plan,expected_quotas=None):
    if plan['count']!=2000 or len(plan['entries'])!=2000:raise ValueError('Original2000 plan only')
    expected_quotas=expected_quotas if expected_quotas is not None else plan.get('intended_instance_quotas_by_split',QUOTAS)
    if not _explicit_quota_shape(expected_quotas) or expected_quotas not in (QUOTAS,QUOTAS_92,QUOTAS_TEST_DUAL):raise ValueError('Explicit supported exact quota contract required')
    splits=Counter();quotas={s:Counter() for s in SPLITS};groups=defaultdict(list);families=Counter();indices=[]
    for e in plan['entries']:
        if set(e)!=ENTRY_KEYS or type(e['index']) is not int or type(e['seed']) is not int or type(e['target_count']) is not int or any(type(v) is not bool for v in e['target_presence']) or len(e['target_presence'])!=2:raise ValueError('Original entry identity/presence schema')
        if e['split'] not in SPLITS or e['scene_family'] not in FAMILIES or sum(e['target_presence'])!=e['target_count'] or (e['scene_family']=='hard_negative')!=(e['target_count']==0):raise ValueError('Original family/count/presence contract')
        splits[e['split']]+=1;quotas[e['split']][str(e['target_count'])]+=1;groups[e['group_id']].append(e);families[e['scene_family']]+=1;indices.append(e['index'])
    actual_quotas={s:{k:v[k] for k in ('0','1','2')} for s,v in quotas.items()}
    if dict(splits)!=SPLITS or actual_quotas!=expected_quotas or sorted(indices)!=list(range(2000)) or len(groups)!=500:raise ValueError('Original split/presence quotas/groups cannot change')
    if plan.get('intended_instance_quotas_by_split',expected_quotas)!=expected_quotas or plan.get('instance_counts_by_split',actual_quotas)!=actual_quotas or plan.get('split_counts',dict(splits))!=dict(splits):raise ValueError('Plan metadata and explicit quota authority disagree')
    if any(len(rows)!=4 or len({r['split'] for r in rows})!=1 or len({r['background_seed'] for r in rows})!=1 for rows in groups.values()):raise ValueError('Four-entry split-leakage groups required')
    return {'count':2000,'split_counts':dict(splits),'instance_counts_by_split':actual_quotas,'groups':500,'group_size':4,'group_semantics':'split leakage prevention only; not four fixed physical views','family_counts':dict(families)}

@lru_cache(maxsize=8)
def _plan_lookup(path,expected,size,mtime_ns,quotas_json):
    plan=json.loads(_verified_bytes(path,expected,size,mtime_ns));audit_plan(plan,json.loads(quotas_json));return {e['index']:e for e in plan['entries']}

def _entries(c):
    ref=c['source_refs']['source_plan'];p=bounded(ref['path']);s=p.stat();return _plan_lookup(str(p),ref['sha256'],s.st_size,s.st_mtime_ns,json.dumps(c['quota_contract']['instance_counts_by_split'],sort_keys=True))

def rigid(value):
    M=np.asarray(value,dtype=float)
    if M.shape!=(4,4) or not np.isfinite(M).all() or not np.array_equal(M[3],[0,0,0,1]) or np.max(abs(M[:3,:3].T@M[:3,:3]-np.eye(3)))>1e-6 or abs(np.linalg.det(M[:3,:3])-1)>1e-6:raise ValueError('Frozen proper metric matrix required; no scale/reflection')
    return M

def quota_counts(c):
    contract=copy.deepcopy(c['quota_contract']);policy_ref=contract.pop('presence_quota_policy',None)
    if contract not in [{'split_counts':SPLITS,'instance_counts_by_split':q,'groups':500,'entries_per_group':4} for q in (QUOTAS,QUOTAS_92,QUOTAS_TEST_DUAL)]:raise ValueError('Explicit split/presence quota contract required')
    quotas=contract['instance_counts_by_split']
    if not _explicit_quota_shape(quotas):raise ValueError('Quota counts must be explicit nonnegative integers, including zeros')
    if quotas in (QUOTAS_92,QUOTAS_TEST_DUAL):
        if policy_ref is None or c['source_refs'].get('presence_quota_policy')!=policy_ref:raise ValueError('Nonlegacy quotas require exact root policy source binding')
        policy=json.loads(verified_bytes(policy_ref));expected={s:{k:v for k,v in row.items() if k!='total'} for s,row in policy['expected_counts_by_split'].items()}
        totals={k:sum(row[k] for row in quotas.values()) for k in ('0','1','2')}
        if not _explicit_quota_shape(expected) or expected!=quotas or any(policy['expected_counts_by_split'][name]['total']!=SPLITS[name] for name in SPLITS) or policy['total_samples']!=2000 or policy['hard_negative_count']!=totals['0'] or policy['expected_counts_total']!=totals or policy['dual_fraction']!=totals['2']/2000:raise ValueError('Root policy and exact split/total quota counts disagree')
    elif policy_ref is not None:raise ValueError('Legacy diagnostic quotas cannot masquerade as current production policy')
    return contract['instance_counts_by_split']

def validate_config(c):
    allowed={'schema','version','status','dataset_generation_allowed','sampling_mode','source_refs','fixed','fixed_sha256','actor_bounds_by_side','bounds','rotation_convention','reference_sampling','augmentation_indices','family_profiles','near_plane_m','quota_contract','evidence_refs','scope_claim','reference_layout'}
    if set(c)!=allowed:raise ValueError('Unknown parameter overrides are forbidden; no scale/A/S/K profile overrides')
    if c['schema']!='action4.modular_parameter_bounds.v1' or c['status']!='candidate_engineering_preview_only' or c['dataset_generation_allowed'] is not False or c['fixed_sha256']!=digest(c['fixed']):raise ValueError('Explicit candidate-only fixed bounds config')
    expected={'translation_capture_each_axis_m':[-.004,.004],'rotation_brace_local_each_axis_deg':[-8.,8.],'area_light_gain':[.9,1.1],'exposure_offset_ev':[-.15,.15]}
    if c['bounds']!=expected or c['rotation_convention']!='B.R @ Rz(rz) @ Ry(ry) @ Rx(rx); degrees; brace-origin pivot; capture-frame translation':raise ValueError('Only declared initial candidate engineering bounds/convention')
    if set(c['family_profiles'])!=set(FAMILIES):raise ValueError('Original family contract')
    quota_counts(c)
    if c['sampling_mode'] not in ['candidate_proposal','baseline_regression']:raise ValueError('Explicit candidate or baseline-regression mode')
    ref=c['reference_sampling']
    if ref['bounds'] is not None:
        if ref['status']!='evidence_checked_candidate' or not ref['evidence_refs']:raise ValueError('Reference bounds require explicit evidence; status alone is insufficient')
        if set(ref['bounds'])!=set(expected) or not ref['primary_control_choices'] or not set(ref['primary_control_choices'])<=set(['whole_actor_translation','whole_actor_rotation','area_light_gain','exposure_offset_ev']):raise ValueError('Reference controls schema')
        for field,bounds in ref['bounds'].items():
            neutral=1. if field=='area_light_gain' else 0.
            if len(bounds)!=2 or not np.isfinite(bounds).all() or not expected[field][0]<=bounds[0]<=neutral<=bounds[1]<=expected[field][1]:raise ValueError('Reference interval must be explicit, within candidate caps and contain neutral')
        if not any(v[0]!=v[1] for v in ref['bounds'].values()):raise ValueError('Zero regression is not a1800-entry reference coverage range')
        for source in ref['evidence_refs']:verified_bytes(source)
    elif ref['status']!='pending_evidence':raise ValueError('Unknown reference bounds are null/pending, not zero intervals')
    indices=c['augmentation_indices']
    if len(indices)!=200 or len(set(indices))!=200 or any(type(i) is not int or not 0<=i<2000 for i in indices):raise ValueError('Exactly10percent single-primary-control augmentation')
    entries=_entries(c)
    if Counter(entries[i]['split'] for i in indices)!=Counter({'train':160,'eval':20,'test':20}):raise ValueError('90/10 within each original split')
    return c

def load_config(path=CONFIG):
    c=validate_config(json.loads(bounded(path).read_text()))
    for ref in c['source_refs'].values():verified_bytes(ref)
    return c

def validate_source_context(c,context):
    fixed=c['fixed']
    if context['scene_config_ref']!=c['source_refs']['scene_config'] or context['W_from_capture']!=fixed['W_from_capture'] or context['K_native']!=fixed['K_native'] or context['baseline_environment']!=fixed['baseline_environment']:raise ValueError('Fixed scene/W/K/environment source context changed')
    for side in ['left','right']:
        actual=context['sides'][side];expected=fixed['sides'][side]
        for field in ['A_brace_from_CAD','S_skin_relative_to_brace','F_body_from_brace','anchor_C_capture_by_frame']:
            if actual[field]!=expected[field]:raise ValueError('Fixed side authority/anchors changed '+side+'/'+field)
        for field in ['A_brace_from_CAD','S_skin_relative_to_brace','F_body_from_brace']:rigid(actual[field])
    digest(context)

def _random(entry,c,label):
    identity={k:entry[k] for k in ['index','seed','sample_id','group_id','split','background_seed']};identity.update(version=c['version'],source_plan_sha256=c['source_refs']['source_plan']['sha256'],stream=label)
    raw=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).digest();return (int.from_bytes(raw[:8],'big')>>11)*2.0**-53

def _choice(entry,c,label,values):return values[min(len(values)-1,int(_random(entry,c,label)*len(values)))]
def _uniform(entry,c,label,bounds):return bounds[0]+(bounds[1]-bounds[0])*_random(entry,c,label)

def rotation_xyz_degrees(angles):
    x,y,z=[math.radians(float(a)) for a in angles];cx,sx=math.cos(x),math.sin(x);cy,sy=math.cos(y),math.sin(y);cz,sz=math.cos(z),math.sin(z)
    return np.array([[cz,-sz,0],[sz,cz,0],[0,0,1.]])@np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])@np.array([[1.,0,0],[0,cx,-sx],[0,sx,cx]])

def project_canonical(C,K,corners,near=.001):
    C=rigid(C);V=np.asarray(corners)@C[:3,:3].T+C[:3,3];positive=bool(np.all(V[:,2]>near))
    if not positive:return {'valid_positive_depth':False,'canonical8_capture_m':V.tolist(),'canonical8_native_px':None,'bbox_native_xyxy':None}
    uv=(V@np.asarray(K).T)[:,:2]/V[:,2,None];lo,hi=uv.min(0),uv.max(0);w,h=3248.,2464.;size=hi-lo;clipped=np.maximum(np.minimum(hi,[w,h])-np.maximum(lo,[0,0]),0);area=float(np.prod(size));center=(lo+hi)/2;local_center=np.mean(np.asarray(corners),axis=0);pc=C[:3,:3]@local_center+C[:3,3]
    return {'valid_positive_depth':True,'canonical8_capture_m':V.tolist(),'canonical8_native_px':uv.tolist(),'bbox_native_xyxy':np.r_[lo,hi].tolist(),'bbox_width_over_W':float(size[0]/w),'bbox_height_over_H':float(size[1]/h),'bbox_area_over_image':area/(w*h),'bbox_center_normalized':(center/[w,h]).tolist(),'bbox_inside_image_fraction':float(np.prod(clipped)/area) if area>0 else None,'canonical_center_distance_to_capture_eye_m':float(np.linalg.norm(pc)),'canonical_center_optical_depth_m':float(pc[2]),'bbox_fraction_is_not_mask_occlusion_or_truncation':True}

def _actor_aabb(B,bounds):
    vertices=np.array(list(product(*zip(bounds['min_m'],bounds['max_m']))));V=vertices@B[:3,:3].T+B[:3,3];return {'min_m':V.min(0).tolist(),'max_m':V.max(0).tolist()}

def reference_layout_constraints(config):
    """Typed raw/demo image audit input; never3D GT or production bounds."""
    c=config;ref=c['reference_layout']['audit_ref']
    if ref is None:return None
    report=json.loads(verified_bytes(ref));clusters=[]
    for row in report['rows']:
        metrics={m['side']:m for m in row['body_envelope_metrics']}
        if set(metrics)!={'L','R'} or type(row['frame']) is not int or any(m['training_GT'] is not False for m in metrics.values()):raise ValueError('Paired image-proxy audit schema')
        for m in metrics.values():
            values=[*m['center_norm'],m['bbox_width_fraction'],m['bbox_height_fraction'],m['bbox_area_fraction']]
            if len(m['center_norm'])!=2 or not np.isfinite(values).all() or min(values[2:])<=0:raise ValueError('Finite nonzero observed image layout')
        clusters.append({'source_frame':row['frame'],'body_envelope_metrics':copy.deepcopy(row['body_envelope_metrics']),'center_horizontal_separation_fraction':row['center_horizontal_separation_fraction'],'raw_sha256':row['raw_sha256'],'demo_sha256':row['demo_sha256'],'training_GT':False})
    return {'source_ref':copy.deepcopy(ref),'clusters':clusters,'observed_envelope':copy.deepcopy(report['observed_envelope']),'diagnostic_proposal_guard':copy.deepcopy(report['usable_reference_bounds_proposal']),'limits':copy.deepcopy(report['limits']),'physical_pose_K_or_anatomy_GT':False,'production_frozen':False}


def reference_layout_request(entry,config):
    constraints=reference_layout_constraints(config)
    if constraints is None:return {'status':'EXACT_IMAGE_LAYOUT_AUDIT_BINDING_PENDING','target_C_or_scale_not_modified':True}
    if not any(entry['target_presence']):return {'status':'NOT_APPLICABLE_NO_TARGET_LAYOUT','source_ref':constraints['source_ref'],'target_C_or_scale_not_modified':True}
    cluster=_choice(entry,config,'paired_image_layout_cluster',constraints['clusters'])
    return {'status':'IMAGE_LAYOUT_CLUSTER_CONDITIONAL_REQUEST_ONLY_POSE_CONTRACT_PENDING','paired_cluster':copy.deepcopy(cluster),'source_ref':constraints['source_ref'],'conditional_capture_pose_bounds':copy.deepcopy(config['reference_layout']['conditional_capture_pose_bounds']),'target_C_or_scale_not_modified':True,'not_a_3D_pose_or_camera_GT':True,'same_real_frame_not_claimed_for_synthetic_pose':True}


def check_reference_layout(parameters,constraints,guard='observed_envelope'):
    """Check actual canonical8-derived image metrics, never overwrite those metrics."""
    if constraints is None:return {'status':'IMAGE_LAYOUT_AUDIT_NOT_BOUND','production_acceptance':False}
    if guard not in ['observed_envelope','diagnostic_proposal_guard']:raise ValueError('Explicit image-only guard type')
    e=constraints[guard];violations=[];metrics={};present=0
    for side,letter in [('left','L'),('right','R')]:
        row=parameters['derived']['by_side'][side]
        if row is None:continue
        present+=1
        if not row['valid_positive_depth']:violations.append(side+':invalid_projection');continue
        values={'width':row['bbox_width_over_W'],'height':row['bbox_height_over_H'],'area':row['bbox_area_over_image'],'cx':row['bbox_center_normalized'][0],'cy':row['bbox_center_normalized'][1]};metrics[side]=values
        ranges={'width':e['bbox_width_fraction'],'height':e['bbox_height_fraction'],'area':e['bbox_area_fraction'],'cx':e[letter+'_center_norm_minmax'][0],'cy':e[letter+'_center_norm_minmax'][1]} if guard=='observed_envelope' else {'width':e['width_fraction'],'height':e['height_fraction'],'cx':e[letter+'_center_x'],'cy':e[letter+'_center_y']}
        for name,bounds in ranges.items():
            if not bounds[0]<=values[name]<=bounds[1]:violations.append({'field':side+'/'+name,'actual':values[name],'bounds':copy.deepcopy(bounds)})
    separation=None
    if set(metrics)=={'left','right'}:
        separation=abs(metrics['right']['cx']-metrics['left']['cx']);bounds=e['horizontal_center_separation_fraction'] if guard=='observed_envelope' else e['horizontal_center_separation']
        if not bounds[0]<=separation<=bounds[1]:violations.append({'field':'paired_center_horizontal_separation_fraction','actual':separation,'bounds':copy.deepcopy(bounds)})
    return {'status':'NOT_APPLICABLE_NO_TARGETS' if not present else ('PASS_DIAGNOSTIC_IMAGE_LAYOUT_ONLY' if not violations else 'OUTSIDE_DIAGNOSTIC_IMAGE_LAYOUT'),'source_ref':constraints['source_ref'],'guard':guard,'actual_canonical8_bbox_metrics':metrics,'paired_center_horizontal_separation_fraction':separation,'violations':violations,'image_proxy_is_not_3D_or_training_GT':True,'production_acceptance':False}


def sample_parameters(entry,config,source_context):
    c=load_config(config) if not isinstance(config,dict) else validate_config(config);validate_source_context(c,source_context);entries=_entries(c)
    if set(entry)!=ENTRY_KEYS or type(entry['seed']) is not int or type(entry['target_count']) is not int or type(entry['index']) is not int or entry['index'] not in entries or entry!=entries[entry['index']] or any(type(v) is not bool for v in entry['target_presence']):raise ValueError('Entry identity/seed/group/split/presence must equal immutable source plan')
    e=copy.deepcopy(entry);presence=e['target_presence'];active=[s for s,on in zip(['left','right'],presence) if on];assigned_augmentation=e['index'] in c['augmentation_indices'];baseline=c['sampling_mode']=='baseline_regression';primary=None;axis=None;chosen_side=None;dt={s:[0.,0.,0.] for s in active};angles={s:[0.,0.,0.] for s in active};gain=1.;exposure=0.
    ref=c['reference_sampling'];reference_pending=not baseline and not assigned_augmentation and ref['bounds'] is None
    status='BASELINE_REGRESSION_ENGINEERING_ONLY' if baseline else ('REFERENCE_BOUNDS_UNKNOWN_PENDING' if reference_pending else ('BOUNDED_AUGMENTATION_CANDIDATE' if assigned_augmentation else 'BOUNDED_REFERENCE_CANDIDATE'))
    if not baseline and not reference_pending:
        bounds=c['bounds'] if assigned_augmentation else ref['bounds'];choices=['whole_actor_translation','whole_actor_rotation','area_light_gain','exposure_offset_ev'] if assigned_augmentation else list(ref['primary_control_choices']);choices=[k for k in choices if active or k not in ['whole_actor_translation','whole_actor_rotation']]
        if not choices:raise PendingParametersError(['reference_controls_not_feasible_for_absent_actors'])
        primary=_choice(e,c,'primary',choices)
        if primary in ['whole_actor_translation','whole_actor_rotation']:
            chosen_side=_choice(e,c,'side',active);axis=_choice(e,c,'axis',[0,1,2]);field='translation_capture_each_axis_m' if primary=='whole_actor_translation' else 'rotation_brace_local_each_axis_deg';value=_uniform(e,c,'value',bounds[field]);(dt if primary=='whole_actor_translation' else angles)[chosen_side][axis]=value
        elif primary=='area_light_gain':gain=_uniform(e,c,'value',bounds['area_light_gain'])
        else:exposure=_uniform(e,c,'value',bounds['exposure_offset_ev'])
    frames={s:None for s in ['left','right']};sides={};derived={};aabbs={}
    for side,on in zip(['left','right'],presence):
        fixed=c['fixed']['sides'][side];A=rigid(fixed['A_brace_from_CAD']);payload={'present':on,'core_C_capture':None,'brace_B_capture':None,'A_brace_from_CAD':copy.deepcopy(fixed['A_brace_from_CAD']),'S_skin_relative_to_brace':copy.deepcopy(fixed['S_skin_relative_to_brace']),'F_body_from_brace':copy.deepcopy(fixed['F_body_from_brace']),'whole_actor_delta_capture':None,'translation_capture_m':[0.,0.,0.],'rotation_brace_local_xyz_deg':[0.,0.,0.]}
        if on:
            frame=0 if baseline or side=='left' or len(active)==2 else _choice(e,c,'right_single_anchor',[0,540,960]);frames[side]=frame;C0=rigid(fixed['anchor_C_capture_by_frame'][str(frame)]);B0=C0@np.linalg.inv(A);B=B0.copy();B[:3,:3]=B0[:3,:3]@rotation_xyz_degrees(angles[side]);B[:3,3]=B0[:3,3]+dt[side];C=B@A;delta=B@np.linalg.inv(B0)
            payload.update(core_C_capture=C.tolist(),brace_B_capture=B.tolist(),whole_actor_delta_capture=delta.tolist(),translation_capture_m=dt[side],rotation_brace_local_xyz_deg=angles[side]);derived[side]=project_canonical(C,c['fixed']['K_native'],c['fixed']['canonical8_m'],c['near_plane_m']);aabbs[side]=_actor_aabb(B,c['actor_bounds_by_side'][side])
        else:
            derived[side]=None;Bbackground=rigid(fixed['anchor_C_capture_by_frame']['0'])@np.linalg.inv(A);aabbs[side]=_actor_aabb(Bbackground,c['actor_bounds_by_side'][side]['background_only'])
        sides[side]=payload
    family=copy.deepcopy(c['family_profiles'][e['scene_family']]);pending=list(family['pending_reasons']);pending += ['reference_pose_bounds_unknown_pending'] if reference_pending else [];missing=sorted(set(family['required_capabilities'])-set(source_context['capabilities']));pending += ['missing_scene_consumer_capability:'+v for v in missing]
    if e['scene_family']=='near_double' and len(active)!=2:pending.append('near_double_requires_two_present_targets_without_changing_original_quota')
    env={'area_light_gain':gain,'exposure_offset_ev':exposure,'wall_visible':family['wall_visible'],'table_visible':family['table_visible']};pair=None
    if len(aabbs)==2:
        low=np.maximum(aabbs['left']['min_m'],aabbs['right']['min_m']);high=np.minimum(aabbs['left']['max_m'],aabbs['right']['max_m']);gap=np.maximum(low-high,0);pair={'core_center_distance_m':float(np.linalg.norm(np.mean(derived['left']['canonical8_capture_m'],axis=0)-np.mean(derived['right']['canonical8_capture_m'],axis=0))) if len(active)==2 else None,'actor_AABB_strictly_disjoint_prior':bool(np.any(gap>0)),'actor_AABB_distance_lower_bound_m':float(np.linalg.norm(gap)),'center_definition':'mean of frozen canonical8 transformed by C','minimum_surface_distance_m':None,'AABB_overlap_requires_actual_BVH':bool(not np.any(gap>0))}
    result={'schema':'action4.modular_parameters.v1','status':status,'estimated':True,'training_accepted':False,'dataset_generation_allowed':False,'entry':e,'target_presence':copy.deepcopy(presence),'sides':sides,'W_from_capture':copy.deepcopy(c['fixed']['W_from_capture']),'K_native':copy.deepcopy(c['fixed']['K_native']),'source_anchor_frames_by_side':frames,'environment_controls':env,'scene_family_controls':{'family':e['scene_family'],'coarse_occluders':copy.deepcopy(family['coarse_occluders']),'required_capabilities':copy.deepcopy(family['required_capabilities']),'pending_reasons':pending,'effect_claim':'candidate_only_not_verified_family_effect'},'augmentation':{'reference_or_augmentation':'baseline_regression' if baseline else ('augmentation' if assigned_augmentation else 'reference'),'counts_as_formal_coverage':False,'formal_partition_assignment':'augmentation' if assigned_augmentation else 'reference','reference_bounds_status':ref['status'],'reference_bounds_known':ref['bounds'] is not None,'reference_bounds':copy.deepcopy(ref['bounds']),'reference_evidence_refs':copy.deepcopy(ref['evidence_refs']),'baseline_regression_does_not_fill1800_reference_quota':baseline,'primary_control':primary,'side':chosen_side,'axis':axis,'one_primary_control_only':True,'distribution':'stable_SHA256_uniform53bit; engineering prior not video frequency'},'derived':{'by_side':derived,'synthetic_present_count':len(active),'same_source_frame_pair_anchor':len(active)==2,'mixed_frame_pair':False,'single_right_pose_cache_is_fixed_f0_Y25_under_other_verified_C_anchors':len(active)==1 and active[0]=='right' and frames['right']!=0,'presence_is_counterfactual_target_and_dedicated_mount_omission_not_measured_video_device_count':len(active)<2},'safety_prior':{'actor_AABB_by_side_capture_m':aabbs,'absent_target_background_policy':source_context.get('absence_policy',{}).get('claim','keep frozen f0 skin/band; omit core/decal and dedicated mount'),'pair':pair,'near_plane_pass':all(row is None or row['valid_positive_depth'] for row in derived.values()),'requires_actual_geometry_contact_containment_and_masks':True,'skin_self_baselines_preserved_not_collision_acceptance':{'left':[99,173],'right':[95,163]}},'provenance':{'config_sha256':digest(c),'source_context_sha256':digest(source_context),'refs':copy.deepcopy(c['source_refs']),'version':c['version'],'F_source_field':'wc.rigid_F_new_body','fixed_size_A_S_F_K_and_pose_cache_not_sampled':True,'group_semantics':'split leakage prevention only; not four physical views'}}
    if 'absence_policy' in source_context:
        result['derived']['absence_policy']=copy.deepcopy(source_context['absence_policy'])
        result['derived']['presence_is_counterfactual_camera_specific_attachment_omission_not_measured_video_device_count']=result['derived'].pop('presence_is_counterfactual_target_and_dedicated_mount_omission_not_measured_video_device_count')
    layout=reference_layout_constraints(c)
    result['derived']['reference_image_layout_request']=reference_layout_request(e,c)
    result['derived']['reference_image_layout_diagnostic']=check_reference_layout(result,layout)
    json.dumps(result,allow_nan=False);return result

def require_consumable_preview(parameters):
    reasons=list(parameters['scene_family_controls']['pending_reasons'])
    if parameters['status'] not in ['BASELINE_REGRESSION_ENGINEERING_ONLY','BOUNDED_AUGMENTATION_CANDIDATE','BOUNDED_REFERENCE_CANDIDATE']:reasons.append('status_not_consumable:'+parameters['status'])
    if parameters['status']=='BOUNDED_REFERENCE_CANDIDATE':
        a=parameters['augmentation']
        if a['reference_bounds_known'] is not True or a['reference_bounds'] is None or a['reference_bounds_status']!='evidence_checked_candidate' or not a['reference_evidence_refs']:reasons.append('reference_evidence_not_bound')
    if not parameters['safety_prior']['near_plane_pass']:reasons.append('canonical8_near_plane_failure')
    if reasons:raise PendingParametersError(reasons)
    return True

def require_production_ready(parameters):
    raise PendingParametersError(['candidate_engineering_bounds_not_frozen_for_production',*parameters['scene_family_controls']['pending_reasons']])


def coverage_deficit(config):
    c=validate_config(config);entries=_entries(c);pending_by_family={family:sum(e['scene_family']==family for e in entries.values()) for family,p in c['family_profiles'].items() if p['pending_reasons']}
    return {'formal_target_count':2000,'split_counts':SPLITS,'instance_counts_by_split':copy.deepcopy(c['quota_contract']['instance_counts_by_split']),'reference_target_count':1800,'augmentation_partition_count':200,'reference_bounds_missing_entries':1800 if c['reference_sampling']['bounds'] is None else 0,'reference_pose_bounds_missing_entries':1800 if c['reference_sampling']['bounds'] is None else 0,'image_layout_support_status':c['reference_layout']['status'],'formal_accepted_count':0,'formal_acceptance_deficit':2000,'pending_family_entry_counts':pending_by_family,'required128_preview_family_set':list(FAMILIES),'group_semantics':'split leakage only','baseline_regression_does_not_count_as_reference_coverage':True,'rendered_mask_joint_distribution_validation_pending':True}
