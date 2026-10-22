"""Pure numerical v2 pose proposals. No file/scene/render/acceptance operations.
Search-domain bounds are engineering proposals, never measured-safe ranges.
"""
from copy import deepcopy
from collections import Counter
import hashlib,json,math,random,re
import numpy as np

SCHEMA='action4.modular_pose_controls.v2'
SIDES=('left','right')
ENTRY_KEYS={'sample_id','index','seed','scene_family','target_count','target_presence','group_id','background_source_id','background_seed','split'}
BASE_DIMENSIONS=tuple(side+'.'+key for side in SIDES for key in ('u','v','depth','rx','ry','rz','occlusion_bin'))+('area_light_gain','exposure_offset_ev')

OCCLUSION_DIMENSIONS=tuple(side+'.'+key for side in SIDES for key in ('occlusion_fraction','occluder_edge_deg','occluder_width_ratio','occluder_height_ratio','occluder_tangent_offset','occluder_depth_fraction'))
DIMENSIONS=BASE_DIMENSIONS+OCCLUSION_DIMENSIONS


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def reference(value):
    if not isinstance(value,dict) or set(value)!={'path','sha256'} or not isinstance(value['path'],str) or not value['path'] or value['path'].startswith('/') or ':' in value['path'] or '\\' in value['path'] or any(part in ('','.','..') for part in value['path'].split('/')) or not re.fullmatch('[0-9a-f]{64}',value.get('sha256','')):
        raise ValueError('Explicit relative source reference required')
    return deepcopy(value)


def array(value,shape):
    a=np.asarray(value,dtype=np.float64)
    if a.shape!=shape or not np.isfinite(a).all():raise ValueError('Finite array shape required: '+str(shape))
    return a


def rigid(value):
    a=array(value,(4,4));r=a[:3,:3]
    if not np.allclose(a[3],[0,0,0,1],rtol=0,atol=1e-12) or not np.allclose(r.T@r,np.eye(3),rtol=0,atol=1e-9) or abs(np.linalg.det(r)-1)>1e-9:
        raise ValueError('Proper rigid transform required; scaling/shear prohibited')
    return a


def default_policy(canonical8_m,canonical_source_ref):
    corners=array(canonical8_m,(8,3))
    return {'schema':'action4.coarse_pose_search_policy.v2','revision':'broad_joint_v005','status':'ENGINEERING_SEARCH_DOMAIN_NOT_ACTUALLY_VALIDATED',
            'canonical_source_ref':reference(canonical_source_ref),'canonical8_ordered_sha256':digest(corners.tolist()),
            'image_size_wh':[3248,2464],'near_plane_m':.001,
            'depth_m':[.40,1.40],'centre_u_fraction':{'left':[.08,.92],'right':[.08,.92]},
            'centre_v_fraction':[.12,.90],'brace_local_rotation_deg':[-20.,20.],
            'brace_local_rotation_xyz_deg':{'rx':[-55.,55.],'ry':[-75.,75.],'rz':[-60.,60.]},
            'occluder_controls':{'edge_deg':[-180.,180.],'width_ratio':[1.2,2.5],'height_ratio':[1.2,2.5],
                'tangent_offset':[-.35,.35],'depth_fraction':[.03,.20]},
            'area_light_gain':[.5,1.8],'exposure_offset_ev':[-.7,.7],'randomize_absent_actors':True,
            'effective_change_thresholds':{'centre_fraction':.01,'depth_m':.02,'orientation_deg':2.,'area_light_gain':.02,'exposure_offset_ev':.03},
            'occlusion_bins':{'clear':[0.,.10],'light':[.10,.30],'medium':[.30,.55],'heavy':[.55,.75]},
            'background_sampling':'fixed_worker_template',
            'background_templates':['wall','table_full_garment','cloth_lift10_full_garment'],
            'backgrounds_by_family':{'wall_rear':['wall'],'near_double':['wall'],'self_forearm_occlusion':['wall'],
                'wall_transition_truncation':['wall'],'hard_negative':['wall'],
                'table_fastening':['table_full_garment'],'table_turn':['table_full_garment'],'cloth_lift':['cloth_lift10_full_garment']},
            'prior':{'minimum_bbox_area_fraction':.0001,'maximum_bbox_area_fraction':.25,'minimum_bbox_inside_fraction':.01,'require_separated_camera_envelopes':True},
            'descriptor':{'translation_step_m':.01,'rotation_matrix_step':.05,'projection_near_threshold_px':24.,'projection_near_mode':'pixels','projection_near_diagonal_fraction':.05},
            'ranges_actual_boundary_validated':False,'training_accepted':False,'dataset_generation_allowed':False}


def interval(value):
    if not isinstance(value,list) or len(value)!=2 or any(type(x) not in (int,float) or not math.isfinite(x) for x in value) or value[0]>value[1]:
        raise ValueError('Finite ordered interval required')
    return value


def validate_policy(policy,corners):
    if policy.get('schema')!='action4.coarse_pose_search_policy.v2' or policy.get('status')!='ENGINEERING_SEARCH_DOMAIN_NOT_ACTUALLY_VALIDATED' or any(policy.get(k) is not False for k in ('ranges_actual_boundary_validated','training_accepted','dataset_generation_allowed')):
        raise ValueError('Unaccepted explicit engineering search policy required')
    reference(policy['canonical_source_ref'])
    if policy['canonical8_ordered_sha256']!=digest(corners.tolist()):raise ValueError('Ordered canonical8 differs from declared source policy')
    if policy['image_size_wh']!=[3248,2464]:raise ValueError('Fixed native resolution required')
    near=policy['near_plane_m']
    if type(near) not in (int,float) or not math.isfinite(near) or near<=0:raise ValueError('Positive near plane required')
    if interval(policy['depth_m'])[0]<=near:raise ValueError('Centre depth domain must exceed near plane')
    if set(policy['centre_u_fraction'])!=set(SIDES):raise ValueError('Both independent centre domains required')
    for bounds in [*policy['centre_u_fraction'].values(),policy['centre_v_fraction']]:
        lo,hi=interval(bounds)
        # Off-image centres are permitted by explicit policy; viewport prior
        # may reject them, and projected corners are never clipped.
        pass
    axes=policy.get('brace_local_rotation_xyz_deg')
    if axes is not None and set(axes)!={'rx','ry','rz'}:raise ValueError('All three local rotation axes required')
    for bounds in (axes.values() if axes is not None else [policy['brace_local_rotation_deg']]):
        lo,hi=interval(bounds)
        if lo<-180 or hi>180:raise ValueError('Explicit local Euler range within +/-180 required')
    occ=policy.get('occluder_controls')
    if occ is not None:
        if set(occ)!={'edge_deg','width_ratio','height_ratio','tangent_offset','depth_fraction'}:raise ValueError('Complete occluder control bounds required')
        for bounds in occ.values():interval(bounds)
        if not 0<occ['depth_fraction'][0]<=occ['depth_fraction'][1]<1 or min(occ['width_ratio'][0],occ['height_ratio'][0])<=0:raise ValueError('Positive occluder size and bounded foreground depth required')
    if interval(policy['area_light_gain'])[0]<=0:raise ValueError('Positive light gain search domain required')
    interval(policy['exposure_offset_ev'])
    if type(policy['randomize_absent_actors']) is not bool:raise ValueError('Explicit absent-actor movement policy required')
    for value in policy['effective_change_thresholds'].values():
        if type(value) not in (int,float) or not math.isfinite(value) or value<=0:raise ValueError('Positive effective-change threshold required')
    if not policy['occlusion_bins']:raise ValueError('Explicit actual-mask target bins required')
    for bounds in policy['occlusion_bins'].values():
        lo,hi=interval(bounds)
        if not 0<=lo<=hi<1:raise ValueError('Occlusion interval must retain some visible support')
    backgrounds=policy['background_templates']
    if not isinstance(backgrounds,list) or not 2<=len(backgrounds)<=3 or len(set(backgrounds))!=len(backgrounds):raise ValueError('Keep two or three explicit existing backgrounds')
    for choices in policy['backgrounds_by_family'].values():
        if not choices or not set(choices)<=set(backgrounds):raise ValueError('Family background must be explicitly declared')
    p=policy['prior']
    if not 0<p['minimum_bbox_area_fraction']<=p['maximum_bbox_area_fraction']<=1 or not 0<p['minimum_bbox_inside_fraction']<=1 or type(p['require_separated_camera_envelopes']) is not bool:
        raise ValueError('Explicit finite occupancy prior required')
    if policy['descriptor']['projection_near_mode'] not in ('pixels','target_diagonal_fraction'):raise ValueError('Explicit projection-nearness metric required')
    for value in (v for k,v in policy['descriptor'].items() if k!='projection_near_mode'):
        if type(value) not in (int,float) or not math.isfinite(value) or value<=0:raise ValueError('Positive descriptor quantization required')


def latin_hypercube(count,seed):
    """Each dimension visits each stratum once; stable independent permutations."""
    if type(count) is not int or not 1<=count<=20000 or type(seed) is not int:raise ValueError('Bounded sample count and integer sampler seed required')
    rng=random.Random(seed);columns=[];bins=[]
    for _ in DIMENSIONS:
        order=list(range(count));rng.shuffle(order)
        columns.append([(j+rng.random())/count for j in order]);bins.append(order)
    return [{'schema':'action4.latin_hypercube_vector.v2','sampler_seed':seed,'index':i,'count':count,
             'dimensions':list(DIMENSIONS),'stratum_ids':[b[i] for b in bins],'values':[c[i] for c in columns]} for i in range(count)]



def sample_uniform(sample_index,master_seed):
    """Production default: independent reproducible stream for each sample index."""
    if type(sample_index) is not int or sample_index<0 or type(master_seed) is not int:
        raise ValueError('Nonnegative integer sample index and integer master seed required')
    seed=int(digest({'sampler':'broad_joint_v005','sample_index':sample_index,'master_seed':master_seed}),16)
    rng=random.Random(seed)
    return {'schema':'action4.uniform_vector.v1','sample_index':sample_index,'master_seed':master_seed,
            'dimensions':list(DIMENSIONS),'values':[rng.random() for _ in DIMENSIONS]}


def unit_vector(vector):
    if vector.get('schema')=='action4.uniform_vector.v1':
        expected=sample_uniform(vector.get('sample_index'),vector.get('master_seed'))
        if vector!=expected:raise ValueError('Uniform vector differs from its declared independent stream')
        values=array(vector['values'],(len(DIMENSIONS),))
        if np.any(values<0) or np.any(values>=1):raise ValueError('Unit interval required')
        return dict(zip(DIMENSIONS,values))
    count=vector.get('count');index=vector.get('index')
    if vector.get('schema')!='action4.latin_hypercube_vector.v2' or vector.get('dimensions')!=list(DIMENSIONS) or type(count) is not int or count<1 or type(index) is not int or not 0<=index<count or type(vector.get('sampler_seed')) is not int:
        raise ValueError('Explicit reproducible stratified vector required')
    values=array(vector['values'],(len(DIMENSIONS),));strata=vector['stratum_ids']
    if len(strata)!=len(DIMENSIONS) or any(type(s) is not int or not 0<=s<count for s in strata) or np.any(values<0) or np.any(values>=1) or any(int(math.floor(float(v)*count))!=s for v,s in zip(values,strata)):
        raise ValueError('Unit sample does not belong to declared strata')
    return dict(zip(DIMENSIONS,values))


def rotation_xyz(angles):
    x,y,z=np.radians(angles);cx,sx=math.cos(x),math.sin(x);cy,sy=math.cos(y),math.sin(y);cz,sz=math.cos(z),math.sin(z)
    return np.array([[cz,-sz,0],[sz,cz,0],[0,0,1.]])@np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])@np.array([[1.,0,0],[0,cx,-sx],[0,sx,cx]])


def project(corners,C,K,size,near):
    xyz=corners@C[:3,:3].T+C[:3,3]
    if not (xyz[:,2]>near).all():return {'valid_positive_depth':False,'canonical8_capture_m':xyz.tolist(),'canonical8_native_px':None}
    homogeneous=xyz@K.T;uv=homogeneous[:,:2]/homogeneous[:,2,None]
    lo,hi=uv.min(0),uv.max(0);span=hi-lo;area=float(np.prod(span));inside=np.maximum(np.minimum(hi,size)-np.maximum(lo,[0,0]),0)
    center=C[:3,:3]@corners.mean(0)+C[:3,3];hc=K@center
    return {'valid_positive_depth':True,'canonical8_capture_m':xyz.tolist(),'canonical8_native_px':uv.tolist(),
            'corner_ids':list(range(8)),'points_clamped':False,'bbox_native_xyxy':np.r_[lo,hi].tolist(),
            'bbox_area_fraction':area/float(np.prod(size)),'bbox_inside_fraction':float(np.prod(inside))/area if area else 0.,
            'projected_body_centre_fraction':(hc[:2]/hc[2]/size).tolist(),
            'mean_projected_corner_fraction':(uv.mean(0)/size).tolist(),'body_centre_capture_m':center.tolist(),
            'depth_m':float(center[2]),'body_centre_range_m':float(np.linalg.norm(center)),'bbox_fraction_is_not_mask_occlusion_or_truncation':True}


def envelope_prior(corners,left,right):
    """15-axis OBB separating-axis prior. Overlap is unresolved, not a surface result."""
    local=(corners.min(0)+corners.max(0))/2;half=(corners.max(0)-corners.min(0))/2
    centers=[C[:3,:3]@local+C[:3,3] for C in (left,right)];rotations=[C[:3,:3] for C in (left,right)]
    axes=[*rotations[0].T,*rotations[1].T]+[np.cross(a,b) for a in rotations[0].T for b in rotations[1].T]
    best=-math.inf
    for axis in axes:
        norm=np.linalg.norm(axis)
        if norm<1e-12:continue
        axis=axis/norm
        gap=abs(float(np.dot(centers[1]-centers[0],axis)))-sum(float(np.dot(np.abs(axis@R),half)) for R in rotations)
        best=max(best,gap)
    return {'method':'conservative_canonical_obb_separating_axes','canonical_envelopes_disjoint':bool(best>1e-9),
            'maximum_separating_axis_gap_m':float(best),'overlap_means_actual_collision':False,
            'actual_surface_pass':False,'full_actor_and_environment_collision_pending':True}



def effective_difference(state,reference,thresholds):
    """Ten physical/control groups; tiny floating differences do not count."""
    changes={}
    for side in SIDES:
        actual,previous=state['actors'][side],reference['actors'][side]
        for i,name in enumerate(('u','v')):
            delta=abs(actual['reference_centre_fraction'][i]-previous['reference_centre_fraction'][i])
            changes[side+'.'+name]={'difference':delta,'threshold':thresholds['centre_fraction'],'changed':delta>=thresholds['centre_fraction']}
        delta=abs(actual['depth_m']-previous['depth_m']);limit=thresholds['depth_m']
        changes[side+'.depth']={'difference':delta,'threshold':limit,'changed':delta>=limit}
        rotation=np.asarray(previous['brace_rotation']).T@np.asarray(actual['brace_rotation'])
        delta=math.degrees(math.acos(float(np.clip((np.trace(rotation)-1)/2,-1,1))));limit=thresholds['orientation_deg']
        changes[side+'.orientation']={'difference':delta,'threshold':limit,'changed':delta>=limit}
    for key in ('area_light_gain','exposure_offset_ev'):
        delta=abs(state[key]-reference[key]);limit=thresholds[key]
        changes[key]={'difference':delta,'threshold':limit,'changed':delta>=limit}
    return {'effective_group_count':len(changes),'changed_group_count':sum(x['changed'] for x in changes.values()),
            'groups':changes,'orientation_is_one_geodesic_group_per_actor':True,
            'comparison_uses_actor_reference_frames_not_absent_target_GT':True}


def actor_reference_state(C,B,corners,K,size):
    centre=C[:3,:3]@corners.mean(0)+C[:3,3];h=K@centre
    if centre[2]<=0:raise ValueError('Positive actor reference centre depth required for coverage')
    return {'reference_centre_fraction':(h[:2]/h[2]/size).tolist(),'depth_m':float(centre[2]),
            'brace_rotation':B[:3,:3].tolist(),'absent_reference_is_not_a_target_label':True}


def propose(anchor,entry,policy,stratified_vector,canonical8_m,*,anchor_ref,previous_proposal=None,include_change_counts=False,diagnostics=False,template_name=None):
    """Derive v2 proposal from a caller-provided anchor; no source file trust is implied."""
    anchor_ref=reference(anchor_ref);corners=array(canonical8_m,(8,3));validate_policy(policy,corners)
    if len(np.unique(corners,axis=0))!=8 or np.any(np.ptp(corners,axis=0)<=0):raise ValueError('Eight distinct nondegenerate ordered body corners required')
    if anchor.get('schema')!='action4.modular_pose_controls.v1':raise ValueError('Explicit existing v1 anchor parameter required')
    if set(entry)!=ENTRY_KEYS or type(entry['target_count']) is not int or type(entry['index']) is not int or entry['index']<0 or type(entry['seed']) is not int or not isinstance(entry['sample_id'],str) or not entry['sample_id'] or entry['split'] not in ('train','eval','test'):
        raise ValueError('Exact original entry required')
    presence=entry['target_presence']
    if not isinstance(presence,list) or len(presence)!=2 or any(type(x) is not bool for x in presence) or sum(presence)!=entry['target_count'] or entry['split']=='test' and entry['target_count']!=2:
        raise ValueError('Original presence/dual-only test required')
    K=array(anchor['K_native'],(3,3));W=rigid(anchor['W_from_capture'])
    if K[0,0]<=0 or K[1,1]<=0 or not np.allclose(K[2],[0,0,1],atol=1e-12,rtol=0):raise ValueError('Fixed calibrated pinhole K required')
    sample=unit_vector(stratified_vector);size=np.asarray(policy['image_size_wh'],float)
    def draw(key,bounds):return float(bounds[0]+sample[key]*(bounds[1]-bounds[0]))
    actors={};sides={};metrics={};requests={};intervals={};descriptors={};Cs={};base_refs={};actor_descriptors={};absent_descriptors={}
    state={'actors':{}};anchor_state={'actors':{}}
    bins=sorted(policy['occlusion_bins'])
    for i,side in enumerate(SIDES):
        fixed=anchor['sides'][side];A=rigid(fixed['A_brace_from_CAD']);rigid(fixed['S_skin_relative_to_brace']);rigid(fixed['F_body_from_brace'])
        source=anchor['actor_pose_capture_by_side'][side];Ba=rigid(source['brace_B_capture']);Da=rigid(source['whole_actor_delta_capture']);Ca=rigid(source['core_C_capture'])
        if not np.allclose(Ba@A,Ca,atol=1e-9,rtol=0):raise ValueError('Anchor B/A/C is inconsistent')
        Bbase=rigid(np.linalg.inv(Da)@Ba);base_refs[side]=Bbase.tolist()
        angles=[draw(side+'.'+axis,policy.get('brace_local_rotation_xyz_deg',{}).get(axis,policy['brace_local_rotation_deg'])) for axis in ('rx','ry','rz')]
        u=draw(side+'.u',policy['centre_u_fraction'][side]);v=draw(side+'.v',policy['centre_v_fraction']);depth=draw(side+'.depth',policy['depth_m'])
        requested_center=np.linalg.solve(K,np.array([u*size[0],v*size[1],1.]));requested_center*=depth/requested_center[2]
        C=np.eye(4);C[:3,:3]=Ba[:3,:3]@rotation_xyz(angles)@A[:3,:3]
        C[:3,3]=requested_center-C[:3,:3]@corners.mean(0)
        B=rigid(C@np.linalg.inv(A));delta=rigid(B@np.linalg.inv(Bbase))
        # Missing cameras still have retained anatomy/attachments. Move their
        # actor frame explicitly, but never emit target points/masks/occlusion.
        if not presence[i] and not policy['randomize_absent_actors']:B,C,delta=Ba,Ca,Da;angles=[0.,0.,0.]
        state['actors'][side]=actor_reference_state(C,B,corners,K,size)
        if include_change_counts:anchor_state['actors'][side]=actor_reference_state(Ca,Ba,corners,K,size)
        q=policy['descriptor']
        actor_descriptors[str(i+1)]=[int(x) for x in np.rint(np.r_[B[:3,3]/q['translation_step_m'],B[:3,:3].ravel()/q['rotation_matrix_step']])]
        if not presence[i]:absent_descriptors[str(i+1)]=actor_descriptors[str(i+1)]
        actors[side]={'brace_B_capture':B.tolist(),'core_C_capture':C.tolist(),'whole_actor_delta_capture':delta.tolist()}
        sides[side]={k:deepcopy(fixed[k]) for k in ('A_brace_from_CAD','S_skin_relative_to_brace','F_body_from_brace')}
        sides[side].update(present=presence[i],brace_B_capture=B.tolist() if presence[i] else None,
                          core_C_capture=C.tolist() if presence[i] else None,whole_actor_delta_capture=delta.tolist() if presence[i] else None,
                          translation_capture_m=(B[:3,3]-Bbase[:3,3]).tolist() if presence[i] else None,
                          rotation_brace_local_xyz_deg=angles if presence[i] else None)
        requests[side]={'centre_fraction':[u,v],'depth_m':depth,'brace_local_rotation_xyz_deg':angles,
                        'installation_angle_offset_deg':[0.,0.,0.],'applied':presence[i] or policy['randomize_absent_actors'],'target_present':presence[i]}
        metrics[side]=project(corners,C,K,size,policy['near_plane_m']) if presence[i] else None
        if presence[i]:
            Cs[side]=C
            bin_name=bins[min(int(sample[side+'.occlusion_bin']*len(bins)),len(bins)-1)]
            intervals[str(i+1)]={'bin':bin_name,'target_interval':deepcopy(policy['occlusion_bins'][bin_name]),'actual_mask_pending':True}
            q=policy['descriptor'];centre=C[:3,:3]@corners.mean(0)+C[:3,3]
            descriptors[str(i+1)]=[int(x) for x in np.rint(np.r_[centre/q['translation_step_m'],C[:3,:3].ravel()/q['rotation_matrix_step']])]
    prior=envelope_prior(corners,Cs['left'],Cs['right']) if diagnostics and len(Cs)==2 else {'method':'not_evaluated' if not diagnostics else 'no_target_pair','canonical_envelopes_disjoint':None,'actual_surface_pass':False,'full_actor_and_environment_collision_pending':True}
    failures=[];bounds=policy['prior']
    if diagnostics:
        for side,metric in metrics.items():
            if metric is None:continue
            if not metric['valid_positive_depth']:failures.append(side+':canonical_near_plane');continue
            if not bounds['minimum_bbox_area_fraction']<=metric['bbox_area_fraction']<=bounds['maximum_bbox_area_fraction']:failures.append(side+':bbox_area_prior')
            if metric['bbox_inside_fraction']<bounds['minimum_bbox_inside_fraction']:failures.append(side+':viewport_prior')
        if len(Cs)==2 and bounds['require_separated_camera_envelopes'] and not prior['canonical_envelopes_disjoint']:failures.append('camera_envelope_overlap_unresolved')
    if policy.get('background_sampling')!='fixed_worker_template':raise ValueError('Fixed worker scene policy required')
    if template_name is None:
        choices=policy['backgrounds_by_family'].get(entry['scene_family'],[])
        if len(choices)!=1:raise ValueError('Explicit fixed worker template required')
        template_name=choices[0]
    if template_name not in policy['background_templates']:raise ValueError('Unknown worker template')
    background=template_name
    relative=None
    if len(Cs)==2:
        L=metrics['left']['body_centre_capture_m'] if metrics['left']['valid_positive_depth'] else (Cs['left'][:3,:3]@corners.mean(0)+Cs['left'][:3,3]).tolist()
        R=metrics['right']['body_centre_capture_m'] if metrics['right']['valid_positive_depth'] else (Cs['right'][:3,:3]@corners.mean(0)+Cs['right'][:3,3]).tolist()
        vector=np.asarray(R)-L
        relative={'right_minus_left_body_centre_capture_m':vector.tolist(),'body_centre_distance_m':float(np.linalg.norm(vector)),
                  'right_core_in_left_core':(np.linalg.inv(Cs['left'])@Cs['right']).tolist()}
    environment=deepcopy(anchor.get('environment_controls',{}))
    for key in ('area_light_gain','exposure_offset_ev'):
        environment[key]=draw(key,policy[key]);state[key]=environment[key]
        anchor_state[key]=float(anchor.get('environment_controls',{}).get(key,1. if key=='area_light_gain' else 0.))
    change_anchor=effective_difference(state,anchor_state,policy['effective_change_thresholds']) if include_change_counts else None
    change_previous=None
    if include_change_counts and previous_proposal is not None:
        if previous_proposal.get('schema')!=SCHEMA or 'effective_state' not in previous_proposal or previous_proposal.get('canonical8_m')!=corners.tolist():raise ValueError('Previous explicit joint proposal with same canonical source required')
        change_previous=effective_difference(state,previous_proposal['effective_state'],policy['effective_change_thresholds'])
    result={'schema':SCHEMA,'status':'NUMERICAL_PROPOSAL_PENDING_ACTUAL_SCENE' if not failures else 'NUMERICAL_PROPOSAL_REJECTED_BY_PRIOR',
            'entry':deepcopy(entry),'target_presence':deepcopy(presence),'W_from_capture':W.tolist(),'K_native':K.tolist(),
            'actor_pose_capture_by_side':actors,'sides':sides,'canonical8_m':corners.tolist(),'corner_ids':list(range(8)),
            'canonical_source_ref':deepcopy(policy['canonical_source_ref']),'controls':{'requested_by_side':requests,'sampling_vector':deepcopy(stratified_vector)},
            'environment_controls':environment,'effective_state':state,
            'effective_changes':{'computed':bool(include_change_counts),'from_anchor':change_anchor,'from_previous':change_previous,'thresholds':deepcopy(policy['effective_change_thresholds']),
                                 'joint_continuous_dimensions':26,'joint_random_dimensions':len(DIMENSIONS),
                                 'rotation_axis_count_from_anchor':{side:sum(abs(x)>=policy['effective_change_thresholds']['orientation_deg'] for x in requests[side]['brace_local_rotation_xyz_deg']) for side in SIDES} if include_change_counts else None},
            'background_request':{'template':background,'existing_geometry_only':True,'actual_application_pending':True,'selection':'fixed_worker_template'},
            'occlusion_condition':{'by_instance_id':intervals,'actual_mask_pending':True,'angles_are_not_occlusion_measurements':True},
            'derived':{'by_side':metrics,'relative_pair':relative,'original8_unclamped':True,'actual_mask_occlusion_by_instance_id':None},
            'safety_prior':{'diagnostics_evaluated':bool(diagnostics),'eligible_for_actual_checks':not failures if diagnostics else None,'failed_priors':failures,'camera_pair':prior,'actual_scene_accepted':False},
            'geometry_descriptor':{'instance_quantized':descriptors,'frame_quantized':[[k,descriptors[k]] for k in sorted(descriptors)],
                                   'actor_quantized':actor_descriptors,'absent_actor_only_quantized':absent_descriptors,
                                   'frame_actor_quantized':[[k,actor_descriptors[k]] for k in sorted(actor_descriptors)],
                                   'quantization_policy':deepcopy(policy['descriptor']),'not_pairwise_distance_proof':True},
            'provenance':{'anchor_ref':anchor_ref,'anchor_parameter_digest':digest(anchor),'policy_digest':digest(policy),
                          'source_context_sha256':anchor.get('provenance',{}).get('source_context_sha256'),
                          'base_B_recovered_from_anchor_delta':base_refs,'source_file_identities_verified':False},
            'ranges_actual_boundary_validated':False,'training_accepted':False,'dataset_generation_allowed':False,
            'formal_entry_adapter':'V2_REQUIRES_EXPLICIT_RENDERER_INTEGRATION'}
    result['occluder_proposals']=make_occluder_proposals(result,policy,sample)
    result['parameter_contract_revision']='broad_joint_v005'
    digest(result)
    return result


def _projection_groups(items,settings):
    """Compare actual ordered8 distances to group representatives, including grid boundaries."""
    groups={};total=0
    for presence,points in items:
        points=np.asarray(points,float).reshape(-1,2);bucket=groups.setdefault(presence,[]);total+=1
        match=None
        for i,(representative,count) in enumerate(bucket):
            distances=np.linalg.norm(points-representative,axis=1)
            if settings['projection_near_mode']=='pixels':
                distance=float(distances.max()) if len(points) else 0.;limit=settings['projection_near_threshold_px']
            else:
                ratios=[]
                for begin in range(0,len(points),8):
                    a,b=points[begin:begin+8],representative[begin:begin+8]
                    scale=max(1e-12,min(float(np.linalg.norm(np.ptp(a,axis=0))),float(np.linalg.norm(np.ptp(b,axis=0)))))
                    ratios.append(float(distances[begin:begin+8].max())/scale)
                distance=max(ratios,default=0.);limit=settings['projection_near_diagonal_fraction']
            if distance<=limit:match=i;break
        if match is None:bucket.append((points.copy(),1))
        else:bucket[match]=(bucket[match][0],bucket[match][1]+1)
    counts=[count for bucket in groups.values() for _,count in bucket]
    return {'count':total,'near_projection_groups':len(counts),'largest_group':max(counts,default=0),
            'multiplicities':sorted(counts,reverse=True),'comparison_mode':settings['projection_near_mode'],'maximum_corner_distance_px':settings['projection_near_threshold_px'],
            'maximum_corner_distance_over_min_target_diagonal':settings['projection_near_diagonal_fraction'],
            'representative_grouping_not_transitive_or_complete_linkage':True}


def diversity_counts(proposals):
    """Per-instance AND full-frame geometry/projection coverage; no seed/photometry claims."""
    frames=Counter();instances=Counter();by_id={};policy=None;canonical=None
    per_projection={};frame_projection=[];coverage={};relative=[];unprojectable=0;threshold=None
    actor_frames=Counter();actor_instances={};previous=None;adjacent_changes=[];anchor_change_counts=[]
    for p in proposals:
        if p.get('schema')!=SCHEMA:raise ValueError('Only explicit v2 proposal descriptors')
        q=digest(p['geometry_descriptor']['quantization_policy']);c=digest(p['canonical8_m'])
        if policy is not None and policy!=q:raise ValueError('One descriptor scale required for comparison')
        if canonical is not None and canonical!=c:raise ValueError('One ordered canonical source required for projection comparison')
        policy=q;canonical=c;threshold=p['geometry_descriptor']['quantization_policy']
        actor_frames[digest(p['geometry_descriptor']['frame_actor_quantized'])]+=1
        for ident,desc in p['geometry_descriptor']['actor_quantized'].items():actor_instances.setdefault(ident,Counter())[digest(desc)]+=1
        anchor_change_counts.append(p['effective_changes']['from_anchor']['changed_group_count'] if p['effective_changes']['from_anchor'] is not None else None)
        if previous is not None:adjacent_changes.append(effective_difference(p['effective_state'],previous['effective_state'],p['effective_changes']['thresholds'])['changed_group_count'])
        previous=p
        frames[digest(p['geometry_descriptor']['frame_quantized'])]+=1
        all_points=[];ids=[];projectable=True
        for ident,desc in p['geometry_descriptor']['instance_quantized'].items():
            key=digest(desc);instances[key]+=1;by_id.setdefault(ident,Counter())[key]+=1
            side=SIDES[int(ident)-1];metric=p['derived']['by_side'][side]
            if not metric['valid_positive_depth']:projectable=False;unprojectable+=1;continue
            points=np.asarray(metric['canonical8_native_px'],float)
            per_projection.setdefault(ident,[]).append((ident,points));all_points.extend(points.tolist());ids.append(ident)
            coverage.setdefault(ident,[]).append([*metric['projected_body_centre_fraction'],metric['depth_m'],metric['bbox_area_fraction'],metric['body_centre_range_m']])
        if projectable:frame_projection.append((tuple(ids),all_points))
        if p['derived']['relative_pair'] is not None:
            pair=p['derived']['relative_pair'];relative.append([*pair['right_minus_left_body_centre_capture_m'],pair['body_centre_distance_m']])
    summary=lambda c:{'count':sum(c.values()),'unique_quantized_groups':len(c),'largest_group':max(c.values(),default=0),'multiplicities':sorted(c.values(),reverse=True)}
    def spans(rows,fields):
        if not rows:return None
        a=np.asarray(rows,float)
        return {name:{'min':float(a[:,i].min()),'max':float(a[:,i].max()),'span':float(np.ptp(a[:,i]))} for i,name in enumerate(fields)}
    return {'schema':'action4.proposal_geometry_diversity.v1','frames':summary(frames),'instances_pooled':summary(instances),
            'instances_by_physical_id':{k:summary(v) for k,v in by_id.items()},'descriptor_policy_digest':policy,
            'near_projection_by_physical_id':{k:_projection_groups(v,threshold) for k,v in per_projection.items()},
            'near_projection_frames':_projection_groups(frame_projection,threshold) if threshold is not None else None,
            'coverage_by_physical_id':{k:spans(v,('centre_u_fraction','centre_v_fraction','depth_m','bbox_area_fraction','body_centre_range_m')) for k,v in coverage.items()},
            'actor_frames_including_absent':summary(actor_frames),'actor_instances_including_absent':{k:summary(v) for k,v in actor_instances.items()},
            'effective_changed_group_counts_from_anchor':anchor_change_counts,'effective_changed_group_counts_from_previous':adjacent_changes,
            'relative_pair_coverage':spans(relative,('delta_x_m','delta_y_m','delta_z_m','distance_m')),
            'unprojectable_instances':unprojectable,'quantization_boundary_can_separate_nearby_3d_poses':True,
            'actual_scene_or_mask_diversity_measured':False,'training_accepted':False}


def make_occluder_proposals(proposal,policy,sample):
    """Position a foreground cloth patch by its projected edge, without rendering.

    Fraction is a geometric request across the target's projected support, NOT
    measured visible/amodal area. Final renderer must use actual object masks.
    Patch vertices are explicit camera/world coordinates for a scene adapter.
    """
    controls=policy.get('occluder_controls')
    if controls is None:return {}
    K=np.asarray(proposal['K_native'],float);W=np.asarray(proposal['W_from_capture'],float)
    result={}
    for i,side in enumerate(SIDES):
        metric=proposal['derived']['by_side'][side]
        if metric is None:continue
        if not metric['valid_positive_depth']:
            result[side]={'status':'UNPROJECTABLE_NO_OCCLUDER'};continue
        def draw(key,bounds):return float(bounds[0]+sample[side+'.'+key]*(bounds[1]-bounds[0]))
        bounds=proposal['occlusion_condition']['by_instance_id'][str(i+1)]['target_interval']
        fraction=draw('occlusion_fraction',bounds)
        angle=draw('occluder_edge_deg',controls['edge_deg'])
        normal=np.array([math.cos(math.radians(angle)),math.sin(math.radians(angle))]);tangent=np.array([-normal[1],normal[0]])
        points=np.asarray(metric['canonical8_native_px']);centre=points.mean(0)
        n=(points-centre)@normal;t=(points-centre)@tangent
        span_n=float(np.ptp(n));span_t=float(np.ptp(t))
        width_ratio=draw('occluder_width_ratio',controls['width_ratio'])
        height_ratio=draw('occluder_height_ratio',controls['height_ratio'])
        offset=draw('occluder_tangent_offset',controls['tangent_offset'])
        gap_fraction=draw('occluder_depth_fraction',controls['depth_fraction'])
        # Edge sweeps across projected target; patch extends to +normal side.
        edge=float(n.max()-fraction*span_n);width=span_t*width_ratio;height=span_n*height_ratio
        origin=centre+normal*edge+tangent*(offset*span_t)
        polygon=np.array([origin-tangent*width/2,origin+tangent*width/2,
                          origin+tangent*width/2+normal*height,origin-tangent*width/2+normal*height])
        front_depth=float(np.asarray(metric['canonical8_capture_m'])[:,2].min())
        depth=front_depth*(1-gap_fraction)
        rays=np.linalg.solve(K,np.column_stack([polygon,np.ones(4)]).T).T
        capture=rays*(depth/rays[:,2,None]);world=capture@W[:3,:3].T+W[:3,3]
        result[side]={'status':'GEOMETRY_PROPOSAL_RENDER_APPLICATION_PENDING','kind':'cloth_patch',
            'target_occlusion_fraction':fraction,'actual_occlusion_fraction':None,
            'edge_angle_deg':angle,'width_over_target_support':width_ratio,'height_over_target_support':height_ratio,
            'tangent_offset_over_target_support':offset,'foreground_gap_fraction':gap_fraction,
            'foreground_gap_m':front_depth-depth,'plane_depth_capture_m':depth,
            'vertices_native_px':polygon.tolist(),'vertices_capture_m':capture.tolist(),'vertices_world_m':world.tolist(),
            'patch_dimensions_m':[float(np.linalg.norm(capture[1]-capture[0])),float(np.linalg.norm(capture[3]-capture[0]))],
            'source_asset':None,'actual_application_pending':True,'requires_existing_cloth_material':True,
            'actual_collision_and_mask_check_pending':True}
    return result
