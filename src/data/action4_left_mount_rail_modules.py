"""LEFT source-derived six-piece mount, single unmeasured sourcebrace180phase candidate; preparation is pure CPU, build requires approval.
Static side adaptation of the SHA-bound reviewed right implementation.
No right source mutation, runtime rewriting, scene/build on import, or mirror.
"""
import hashlib,json,traceback
from pathlib import Path
import numpy as np
from data import action4_global_environment as env
from data import action4_mount_modules as original
ROOT=env.ROOT
OUTPUT=ROOT/'outputs/action4_left_mount_modules_v003'
SOURCE_NAMES={key:'integrated_left_'+key for key in ['quick_release','usb_connector','white_module','ball']}
TEXT='Action4_Left_Mount_Diagnostic.json'
PARENT_SHA='d787812f57ac92615bf3a5f22f0c4da44d60219d63bf5dddd87007499ca1e8e3'
CONFIG_PATH='configs/data/action4_left_mount_modules_v003.json'
# These algorithms have no output/side authority or mutable state.
rigid=original.rigid
transform=original.transform
joint_zones=original.joint_zones
classify_pairs=original.classify_pairs

def path(p):
    p=Path(p);p=(p if p.is_absolute() else ROOT/p).resolve()
    if not p.is_relative_to(ROOT):raise ValueError('Project boundary')
    return p

def sha(p):return hashlib.sha256(path(p).read_bytes()).hexdigest()

def read(p):return json.loads(path(p).read_text())

def ref(p):return {'path':str(path(p).relative_to(ROOT)),'sha256':sha(p)}

def write(p,value):
    p=path(p)
    if not p.is_relative_to(OUTPUT):raise ValueError('Mount diagnostics output only')
    with p.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)

SOURCE_KEYS={'joint_result', 'v001_left_mount_helper', 'CAD_geometry', 'brace_contract', 'source_asset', 'brace_geometry', 'neutral_baseline', 'material_state_helper', 'v002_rejected_receipt', 'neutral_geometry', 'v002_rejected_geometry', 'source_scene_helper', 'left_S_review', 'neutral_full_geometry', 'left_axes', 'registry_validator', 'v002_phase_helper', 'inventory', 'wrist_contract', 'left_band_reopen', 'assembly_helper', 'parent_mount_helper', 'render_tube_helper', 'source_contract', 'wrist_geometry', 'v002_rejected_collision', 'registry'}
ROUTING={'quick_approach_clearance_m': 0.001, 'cable_exit_extra_clearance_m': 0.003, 'nominal_joint_zone_margin_m': 0.001, 'source_reference_shift_m': 0.003, 'route_policy': 'preserve_band_ball_anchor_and_CAD_negativeZ_approach; cable_source_station_magnitudes_on_USB_lateral_side', 'sourcebrace_phase_candidate_degrees': 180, 'phase_claim': 'single_unmeasured_sourcebrace_attachment_phase_candidate_not_measured180_or_realwhite_location', 'rail_clearance_margin_m': 0.001, 'rail_claim': 'single_neutral_slab_support_routing_estimate_not_measured_cable_or_containment'}

def load_config(config_path):
    c=read(config_path)
    if set(c)!={'schema','status','dataset_generation_allowed','sources','routing','output_dir'} or c['schema']!='action4.left_mount_modules.phase_rail.v3' or c['status']!='diagnostic_pending_build_review' or c['dataset_generation_allowed'] is not False:raise ValueError('Exact LEFT diagnostic config required')
    if path(c['output_dir'])!=OUTPUT or path(config_path)!=path(CONFIG_PATH):raise ValueError('Fixed LEFT config/output required')
    required=SOURCE_KEYS
    if set(c['sources'])!=required:raise ValueError('Explicit LEFT source set required')
    for r in c['sources'].values():
        if set(r)!={'path','sha256'} or sha(r['path'])!=r['sha256']:raise ValueError('Frozen source changed')
    if c['sources']['parent_mount_helper']['sha256']!=PARENT_SHA:raise ValueError('Reviewed parent source drift')
    if c['sources']['v001_left_mount_helper']['sha256']!='b18997c84778b9bba3d667ef9b69d1892ddb41106bee3a472def153311110a34':raise ValueError('Immutable v001 LEFT source lineage changed')
    if c['sources']['v002_phase_helper']['sha256']!='cfd6498588a94a9846cb4e688eb011de2ee4b3982858c30016d74fedaa42baee':raise ValueError('Immutable v002 phase source lineage changed')
    if c['routing']!=ROUTING:raise ValueError('Original routing parameters required')
    joint=read(c['sources']['joint_result']['path']);axes=read(c['sources']['left_axes']['path'])
    if joint['side']!='left' or joint['instance_id']!=1 or joint['A_candidate']!='K0p8125_start1' or joint['nonunique_A'] is not True or joint['S_conditioned_candidate_deg']!=3.75 or joint['wearing_accepted'] is not False:raise ValueError('LEFT diagnostic authority required')
    for a,b in [('A_shared','A'),('S_skin_relative_to_brace','S_left'),('K_native','K_native')]:
        if joint[a]!=axes[b]:raise ValueError('LEFT fixed authority differs '+a)
    if joint['C_per_frame']!={'0':axes['C']}:raise ValueError('LEFT f0 C authority')
    rigid(joint['A_shared']);rigid(joint['S_skin_relative_to_brace'])
    wc=read(c['sources']['wrist_contract']['path']);bc=read(c['sources']['brace_contract']['path'])
    if wc['rigid_F_new_body']!=axes['F_left'] or bc['side']!='left' or bc['width_m']!=.034 or bc['inner_radii_m']!=[.02877563688416745,.03260006155662128]:raise ValueError('Frozen LEFT F/radii/width')
    # Existing validator checks the explicit v003 registry (18 files), read-only.
    import importlib.util
    spec=importlib.util.spec_from_file_location('left_mount_frozen_registry',path(c['sources']['registry_validator']['path']));verifier=importlib.util.module_from_spec(spec);spec.loader.exec_module(verifier);verifier.verify(c['sources']['registry']['path'])
    return c

def slab_lateral_support(V,F,ymin,ymax,side):
    """Maximum side*x over ALL neutral triangles clipped to the whole rail Y slab.
    No collision-hit selection, region thinning, shape changes or clearance gate.
    """
    V=np.asarray(V,float);F=np.asarray(F,int)
    if V.ndim!=2 or V.shape[1:]!=(3,) or F.ndim!=2 or F.shape[1:]!=(3,) or not len(V) or not len(F) or not np.isfinite(V).all() or side not in [-1,1] or not np.isfinite([ymin,ymax]).all() or ymin>=ymax:raise ValueError('Finite full triangle slab and lateral side required')
    if F.min()<0 or F.max()>=len(V):raise ValueError('Triangle vertex indices')
    Y=V[F,1];selected=np.flatnonzero((Y.min(1)<=ymax)&(Y.max(1)>=ymin));best=None;witness=None;witness_triangle=None;clipped_points=0
    def clip(poly,bound,lower):
        out=[]
        if not poly:return out
        previous=poly[-1];previous_in=previous[1]>=bound if lower else previous[1]<=bound
        for current in poly:
            current_in=current[1]>=bound if lower else current[1]<=bound
            if previous_in!=current_in:
                t=(bound-previous[1])/(current[1]-previous[1]);point=previous+t*(current-previous);point[1]=bound;out.append(point)
            if current_in:out.append(current)
            previous,previous_in=current,current_in
        return out
    for i in selected:
        polygon=clip([p.copy() for p in V[F[i]]],ymin,True);polygon=clip(polygon,ymax,False);clipped_points+=len(polygon)
        for point in polygon:
            value=float(side*point[0])
            if best is None or value>best:best=value;witness=point.tolist();witness_triangle=int(i)
    if best is None:raise ValueError('Empty full neutral slab support')
    return {'method':'all_triangle_surfaces_clipped_to_closed_Y_slab','full_triangle_count':len(F),'selected_triangle_ids':selected.tolist(),'selected_triangle_count':len(selected),'clipped_polygon_vertex_count':clipped_points,'slab_y_m':[float(ymin),float(ymax)],'lateral_side':int(side),'maximum_side_x_support_m':best,'support_witness_m':witness,'support_witness_triangle_id':witness_triangle,'only_collision_hit_faces_used':False,'measured_cable_or_containment_claim':False}


def routes(c):
    source=read(c['sources']['source_contract']['path']);p=source['config']['initial_priors'];a=source['assembly']['assembly_priors']['left'];A=rigid(read(c['sources']['joint_result']['path'])['A_shared'])
    S=np.diag([-1.,1.,-1.,1.]);S[1,3]=-c['routing']['source_reference_shift_m']
    # Exact Ty(-3mm)@Ry(pi): one root-approved phase candidate, no search/geometry edits.
    old_link=transform(S,np.array(a['mount_link_brace_m']));old_cable=transform(S,np.array(a['cable_points_brace_m']))
    q=np.array(a['quick_release_center_CAD_m']);usb=np.array(a['USB_center_CAD_m']);usb_exit=usb+np.array([0,0,-p['usb_dimensions_m'][2]/2])
    end=transform(A,q);approach=transform(A,q+[0,0,-(p['quick_release_dimensions_m'][2]/2+p['link_radius_m']+c['routing']['quick_approach_clearance_m'])])
    ball=transform(S,np.array(a['ball_center_brace_m']));band=old_link[0]
    elbow=np.array([approach[0],approach[1],ball[2]])
    link=np.array([band,ball,elbow,approach,end])
    start=transform(A,usb_exit);exit_lead=transform(A,usb_exit+[0,0,-(2*p['cable_radius_m']+c['routing']['cable_exit_extra_clearance_m'])])
    sign=-1 if start[0]<0 else 1
    stations=old_cable[1:].copy();stations[:,0]=sign*abs(stations[:,0])
    original_station_abs_x=abs(stations[:,0]).copy()
    if not np.array_equal(original_station_abs_x,np.full(len(stations),original_station_abs_x[0])):raise ValueError('Frozen source has one common straight rail magnitude')
    skinV,skinF,_=collision_skin_context(c)
    receipt=read(c['sources']['v002_rejected_receipt']['path'])
    if receipt['geometry']!=c['sources']['v002_rejected_geometry']:raise ValueError('Actual v002 full neutral geometry receipt binding')
    actual=np.load(path(c['sources']['v002_rejected_geometry']['path']),allow_pickle=False)
    actualV,actualF=actual['skin_vertices_m'],actual['skin_triangles']
    if actualV.shape!=skinV.shape or not np.array_equal(actualF,skinF) or np.max(abs(actualV-skinV))>1e-14:raise ValueError('Actual full neutral skin vs unchanged LEFT context')
    skinV,skinF=actualV,actualF
    rail_y=np.r_[stations[:,1],exit_lead[1]]
    support=slab_lateral_support(skinV,skinF,float(rail_y.min()),float(rail_y.max()),sign)
    rail_magnitude=max(float(original_station_abs_x[0]),support['maximum_side_x_support_m']+p['cable_radius_m']+c['routing']['rail_clearance_margin_m'])
    stations[:,0]=sign*rail_magnitude
    support.update(original_station_abs_x_m=original_station_abs_x.tolist(),derived_common_rail_abs_x_m=rail_magnitude,original_cable_radius_m=p['cable_radius_m'],engineering_margin_m=c['routing']['rail_clearance_margin_m'],rail_x_shift_m=float(rail_magnitude-original_station_abs_x[0]),derived_safe_routing_estimate_not_measured=True)
    transition=np.array([stations[0,0],stations[0,1],max(exit_lead[2],stations[0,2])])
    # Root-approved exact LEFT degeneration only: retain every original rail station.
    exact_transition_duplicate=np.array_equal(transition,stations[0])
    cable=np.vstack([start,exit_lead,stations]) if exact_transition_duplicate else np.vstack([start,exit_lead,transition,stations])
    for curve in [link,cable]:
        if not np.isfinite(curve).all() or np.any(np.linalg.norm(np.diff(curve,axis=0),axis=1)<1e-7):raise ValueError('Finite nonduplicate route stations required')
    if not np.array_equal(link[0],old_link[0]) or not np.array_equal(link[1],ball) or not np.array_equal(link[-1],end) or not np.array_equal(cable[0],start):raise ValueError('Nominal endpoint binding failed')
    return {'A_new_brace_from_CAD':A.tolist(),'old_brace_to_new_brace':S.tolist(),'sourcebrace_phase_candidate_degrees':180,'sourcebrace_phase_measured':False,'white_location_known':False,'source_link_path_reexpressed_m':old_link.tolist(),'source_cable_path_reexpressed_m':old_cable.tolist(),'derived_link_path_m':link.tolist(),'derived_cable_path_m':cable.tolist(),'link_radius_m':p['link_radius_m'],'cable_radius_m':p['cable_radius_m'],'ball_radius_m':p['ball_radius_m'],'quick_release_center_CAD_m':q.tolist(),'USB_exit_CAD_m':usb_exit.tolist(),'quick_release_center_newbrace_m':end.tolist(),'USB_exit_newbrace_m':start.tolist(),'ball_center_newbrace_m':ball.tolist(),'band_anchor_newbrace_m':band.tolist(),'cable_lateral_side':sign,'rail_slab_support_estimate':support,'exact_degenerate_transition_omitted':bool(exact_transition_duplicate),'original_link_endpoint_gap_to_new_anchor_m':float(np.linalg.norm(old_link[-1]-end)),'original_cable_start_gap_to_new_anchor_m':float(np.linalg.norm(old_cable[0]-start)),'anchor_claim':'Unmeasured prior quickrelease center and USBbox -Z exit; not actual product port/socket measurement','routing_claim':'Single fixed route, no search/optimization. AUTO Bezier may overshoot; actual full derived mesh must be checked before any assembly acceptance.','source_dimensions_m':{k:p[k] for k in ['quick_release_dimensions_m','usb_dimensions_m','white_module_dimensions_m']}}

def collision_skin_context(c,supplied=None):
    """Exact LEFT neutral+S3.75 cached authority; no Y25, actual-fit or wearing claim."""
    keys=['joint_result','left_axes','left_S_review','neutral_baseline','neutral_geometry','neutral_full_geometry','wrist_geometry','wrist_contract','brace_geometry','brace_contract']
    for key in keys:
        r=c['sources'][key]
        if sha(r['path'])!=r['sha256']:raise ValueError('LEFT skin context identity changed')
    j=read(c['sources']['joint_result']['path']);axes=read(c['sources']['left_axes']['path']);review=read(c['sources']['left_S_review']['path']);row=read(c['sources']['neutral_baseline']['path'])
    S=rigid(j['S_skin_relative_to_brace']);A=rigid(j['A_shared']);C=rigid(j['C_per_frame']['0']);B=C@np.linalg.inv(A)
    candidates=[x for x in review['independent_checks'] if x['angle_deg']==3.75]
    if review['status']!='PASS_BOUNDED_LEFT_CACHE_RELATION_SPEC_ONLY' or len(candidates)!=1 or candidates[0]['S']!=S.tolist() or axes['S_left']!=S.tolist():raise ValueError('Reviewed LEFT S375 required')
    if row['status']!='LEFT_NEUTRAL_CACHE_ONLY_BVH_BASELINE' or row['side']!='left' or row['instance_id']!=1 or row['pose']!={'left_hand':[0,0,0]} or row['self_intersections']['pair_count']!=99 or row['self_triangle_pair_count']!=173 or row['core_skin_pairs'] or row['band_skin_pairs'] or row['actual_mesh_extractions']!=0:raise ValueError('Independent LEFT neutral cache evidence required')
    skin=np.load(path(c['sources']['wrist_geometry']['path']),allow_pickle=False);full=np.load(path(c['sources']['neutral_full_geometry']['path']),allow_pickle=False);cache=np.load(path(c['sources']['neutral_geometry']['path']),allow_pickle=False);F=np.array(read(c['sources']['wrist_contract']['path'])['rigid_F_new_body']);Fi=np.linalg.inv(F)
    ids=skin['source_vertex_ids'];tids=skin['source_triangle_ids']
    if not np.array_equal(full['vertices_world_m'][ids],skin['source_vertices_m']) or not np.array_equal(full['triangles'][tids],ids[skin['faces']]) or not np.array_equal(cache['source_body_vertices'],skin['source_vertices_m']) or not np.array_equal(cache['skin_faces'],skin['faces']):raise ValueError('Exact LEFT neutral source mapping')
    tolerance=review['numeric_tolerance_audit']['new_consistency_assert_m']
    capture_error=float(np.max(abs(cache['skin_capture_vertices']-transform(B@S@Fi,skin['source_vertices_m']))))
    if tolerance!=1e-14 or capture_error>tolerance:raise ValueError('LEFT baseline capture transform exceeds existing analytic roundoff bound')
    V=transform(S,skin['brace_vertices_m']);T=skin['faces']
    tolerance=review['numeric_tolerance_audit']['new_consistency_assert_m']
    error=float(np.max(abs(transform(B,V)-cache['skin_capture_vertices'])))
    if tolerance!=1e-14 or error>tolerance:raise ValueError('Existing LEFT analytic roundoff consistency')
    if supplied is not None and (not np.array_equal(supplied[0],V) or not np.array_equal(supplied[1],T)):raise ValueError('Exact LEFT neutral+S375 collision context required')
    return V,T,{'frame':'LEFT_new34brace','side':'left','instance_id':1,'pose':{'left_hand':[0,0,0]},'S_skin_relative_to_brace':S.tolist(),'sources':{k:c['sources'][k] for k in keys},'baseline_polygon_pairs':99,'baseline_triangle_pairs':173,'maximum_cached_analytic_roundoff_m':error,'maximum_direct_cache_capture_roundoff_m':capture_error,'existing_analytic_roundoff_tolerance_m':tolerance,'identity_substitution_max_difference_m':float(np.max(abs(V-skin['brace_vertices_m']))),'geometry_vertex_sha256':hashlib.sha256(np.asarray(V,dtype=np.float64).tobytes()).hexdigest(),'triangles_sha256':hashlib.sha256(np.asarray(T,dtype=np.int64).tobytes()).hexdigest(),'A_changed':False,'S_adjusted_by_this_preparation':False,'wearing_alignment_accepted':False,'claim':'LEFT neutral existing cache+reviewed S375 only; Y25 is rejected by root and unused; no posed/fullscene/realwearing acceptance'}

def source_plan(c):
    inv=read(c['sources']['inventory']['path']);rows={r['name']:r for r in inv['parts']};result={}
    for key,name in SOURCE_NAMES.items():
        r=rows[name]
        if r['side']!='left' or r['pass_index']!=0 or r['modifiers'] or r['canonical8_owner'] is not False:raise ValueError('Original LEFT source inventory required')
        if key=='ball' and r['object_scale']!=[.006000000052154064]*3:raise ValueError('Original LEFT ball .006 basis required')
        result[key]=r
    return result

def verify_source_objects(c,originals,assembly):
    plan=source_plan(c)
    for key,o in originals.items():
        r=plan[key]
        if o.name!=r['name'] or not np.array_equal(np.array(o.matrix_basis),r['basis_matrix']) or not np.array_equal(np.array(o.matrix_parent_inverse),r['parent_inverse']) or list(o.scale)!=r['object_scale']:raise ValueError('LEFT original basis/parent inverse/scale differs')
        digest=hashlib.sha256(np.array([v.co[:] for v in o.data.vertices],np.float32).tobytes()).hexdigest()
        if digest!=r['mesh_data_vertices_sha256'] or env.digest([env._material_state(m) for m in o.data.materials])!=r['material_state_sha256']:raise ValueError('LEFT original data/material differs')
        if [m.name for m in o.data.materials]!=r['materials']:raise ValueError('LEFT source material slots changed')

def guard_source_names(bpy):
    names={'mount_left_newbrace_root','mount_left_sourcebrace_subframe','mount_left_CAD_subframe'}|{'mount_left_'+k for k in ['quick_release','usb_connector','white_module','ball','link','cable']}|{'mount_left_'+k+'_curve' for k in ['link','cable']}
    if any(n in bpy.data.objects for n in names) or 'left_mount_components' in bpy.data.collections or TEXT in bpy.data.texts or 'LeftMountDiagnosticOnly' in bpy.data.scenes:raise ValueError('LEFT output name collision; do not suffix or reuse source names')
    return sorted(names)


def prepare(config_path):
    c=load_config(config_path);OUTPUT.mkdir(parents=True,exist_ok=True);r=routes(c);_,_,context=collision_skin_context(c);plan=source_plan(c)
    result={'status':'PREPARED_ONLY_NO_BUILD','side':'left','instance_id':1,'config':ref(config_path),'source':ref(__file__),'parent_algorithm_source':c['sources']['parent_mount_helper'],'routes':r,'collision_skin_context':context,'source_fixed_piece_inventory':plan,'joint_candidate_zones':joint_zones(c,r),'fixed_piece_policy':'Copy original integrated_left objects mesh/material/basis; ball .006 and3mm shift fixed; no mirror/rename-right','target_geometry':False,'canonical8':None,'dataset_generation_allowed':False,'max_authorized_future_builds':1,'build_authorized':False,'GPU':False}
    write(OUTPUT/'preparation.json',result);return result

def _approval(c,approval_path):
    if not path(approval_path).is_relative_to(OUTPUT):raise ValueError('New LEFT approval path required')
    a=read(approval_path);hp=OUTPUT/'preparation_handoff.json'
    if a.get('status')!='APPROVED_SINGLE_CPU_LEFT_PHASE_RAIL_MOUNT_BUILD' or a.get('preparation_handoff_sha256')!=sha(hp):raise ValueError('New bound LEFT approval required')
    for key in ['spec_review','root_review']:
        p=path(a[key+'_path'])
        if not p.is_relative_to(OUTPUT) or a.get(key+'_pass') is not True or sha(p)!=a[key+'_sha256']:raise ValueError('Independent SPEC/root binding')
        review=read(p)
        if review.get('status')!='PASS' or review.get('preparation_handoff_sha256')!=sha(hp):raise ValueError('Explicit PASS current LEFT handoff required')
    h=read(hp)
    for p,digest in h['files'].items():
        if sha(p)!=digest:raise ValueError('Frozen preparation changed')
    if c!=load_config(h['config']['path']):raise ValueError('Requested config differs from approved preparation')
    return a

def build(config_path,approval_path):
    # Exactly one fresh attempt; no retry or CPU/GPU job dispatch here.
    c=load_config(config_path);_approval(c,approval_path);out=OUTPUT/'actual_build_v001';out.mkdir(exist_ok=False)
    write(out/'started.json',{'source':ref(__file__),'config':ref(config_path),'approval':ref(approval_path),'single_attempt':True,'CPU_threads':2,'GPU':False})
    try:return _build_once(c,config_path,out)
    except Exception as exc:
        write(out/'failure.json',{'status':'FAILED_RETAINED_NO_AUTOMATIC_RETRY','error':str(exc),'traceback':traceback.format_exc(),'production_accepted':False});raise

def _build_once(c,config_path,out):
    import bpy
    from mathutils import Matrix
    from mathutils.bvhtree import BVHTree
    from data import action4_global_scene as source
    from data import action4_modular_assembly as assembly
    from data.action4_render import _tube
    r=routes(c);A=np.array(r['A_new_brace_from_CAD']);S=np.array(r['old_brace_to_new_brace'])
    bpy.ops.wm.open_mainfile(filepath=str(path(c['sources']['source_asset']['path'])),load_ui=False,use_scripts=False)
    source_scene=bpy.context.scene;source.verify(source_scene,read(c['sources']['source_contract']['path']))
    guard_source_names(bpy)
    originals={key:source_scene.objects[name] for key,name in SOURCE_NAMES.items()};verify_source_objects(c,originals,assembly);original_states={k:assembly.mesh_state(o) for k,o in originals.items()}
    if any(o.type!='MESH' or len(o.modifiers) or len(o.constraints) or o.animation_data is not None for o in originals.values()):raise ValueError('Static reviewed source parts required')
    scene=bpy.data.scenes.new('LeftMountDiagnosticOnly');bpy.context.window.scene=scene;scene.unit_settings.system='METRIC';scene.unit_settings.scale_length=1
    collection=bpy.data.collections.new('left_mount_components');scene.collection.children.link(collection)
    def empty(name,parent,M):
        o=bpy.data.objects.new(name,None);collection.objects.link(o);o.parent=parent;o.matrix_parent_inverse=Matrix.Identity(4);assembly.set_rigid_basis(o,M);return o
    root=empty('mount_left_newbrace_root',None,np.eye(4));oldbrace=empty('mount_left_sourcebrace_subframe',root,S);cad=empty('mount_left_CAD_subframe',root,A)
    objects={}
    for key,o in originals.items():
        copy=o.copy();copy.name='mount_left_'+key;copy.parent=cad if key in ['quick_release','usb_connector'] else oldbrace;copy.matrix_parent_inverse=o.matrix_parent_inverse.copy();copy.matrix_basis=o.matrix_basis.copy();collection.objects.link(copy);objects[key]=copy
        if env.digest(assembly.mesh_state(copy))!=env.digest(original_states[key]):raise ValueError('Fixed source data/material changed')
        if key=='ball' and not np.allclose(copy.scale,(.006,.006,.006),rtol=0,atol=1e-9):raise ValueError('Original metric ball scale must not be reset')
    black=originals['quick_release'].data.materials[0]
    route_objects=[]
    for key,points,radius in [('link',r['derived_link_path_m'],r['link_radius_m']),('cable',r['derived_cable_path_m'],r['cable_radius_m'])]:
        curve=_tube('mount_left_'+key+'_curve',points,radius,black,root);bpy.context.view_layer.update();ev=curve.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=bpy.data.meshes.new_from_object(ev,depsgraph=bpy.context.evaluated_depsgraph_get())
        o=bpy.data.objects.new('mount_left_'+key,mesh);collection.objects.link(o);o.parent=root;o.matrix_parent_inverse=Matrix.Identity(4);o.matrix_basis=Matrix.Identity(4);objects[key]=o
        route_objects.append({'name':key,'helper':'action4_render._tube','source_radius_m':radius,'bezier_handle_type':'AUTO','control_points_m':points,'baked':True})
        bpy.data.objects.remove(curve,do_unlink=True)
    for o in objects.values():o.pass_index=0;o['target_geometry']=False;o['canonical8_owner']=False;o['dataset_generation_allowed']=False
    scene['dataset_generation_allowed']=False;bpy.context.view_layer.update()
    # Independent source assets provide collision context only; not copied into library.
    geometry={k:assembly.world_geometry(o) for k,o in objects.items()}
    cadmesh=np.load(path(c['sources']['CAD_geometry']['path']),allow_pickle=False);geometry['core']=(transform(A,cadmesh['vertices_mm']*.001),cadmesh['faces'])
    skinV,skinF,skin_context=collision_skin_context(c);geometry['skin']=(skinV,skinF)
    band=np.load(path(c['sources']['brace_geometry']['path']),allow_pickle=False);faces=band['faces'];geometry['band']=(band['vertices_module_m'],np.r_[faces[:,[0,1,2]],faces[:,[0,2,3]]])
    trees={k:BVHTree.FromPolygons(v.tolist(),f.tolist(),all_triangles=True,epsilon=0) for k,(v,f) in geometry.items()};zones=joint_zones(c,r);collisions={};keys=sorted(geometry)
    for i,a in enumerate(keys):
        for b in keys[i+1:]:
            key=a+'__'+b;zone=zones.get(key) or zones.get(b+'__'+a);pairs=trees[a].overlap(trees[b]);collisions[key]=classify_pairs(*geometry[a],*geometry[b],pairs,zone)
    state={}
    for key,o in objects.items():
        v,f=geometry[key];edges=np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]);_,counts=np.unique(np.sort(edges,axis=1),axis=0,return_counts=True)
        state[key]={'name':o.name,'parent':o.parent.name,'basis':np.array(o.matrix_basis).tolist(),'parent_inverse':np.array(o.matrix_parent_inverse).tolist(),'scale':list(o.scale),'world_matrix':np.array(o.matrix_world).tolist(),'state':assembly.mesh_state(o),'module_dimensions_m':np.ptp(v,axis=0).tolist(),'boundary_edges':int((counts==1).sum()),'nonmanifold_edges':int((counts>2).sum()),'source_unchanged':key in originals}
    # No source data mutation through shared data pointers is allowed.
    for key,o in originals.items():
        if env.digest(assembly.mesh_state(o))!=env.digest(original_states[key]):raise ValueError('Original datablock changed')
    contract={'schema':'action4.mount_modules.diagnostic.v1','status':'UNMEASURED_PRIOR_DERIVED_MOUNT_NOT_CALIBRATED','config':ref(config_path),'source':ref(__file__),'sources':c['sources'],'A_source_role':'LEFT K0p8125_start1 fixed diagnosticA unchanged; sourcebrace180phase is a single unmeasured engineering candidate','routes':r,'collision_skin_context':skin_context,'objects':state,'source_fixed_states':original_states,'joint_candidate_zones':zones,'root_objects':{'newbrace':root.name,'oldbrace_subframe':oldbrace.name,'CAD_subframe':cad.name},'canonical8':None,'target_geometry':False,'dataset_generation_allowed':False,'production_accepted':False,'limitations':['Actual mechanical ports not measured; nominal center/boxface anchors only','Source fixed ball childscale retained; no free scaling','Curve AUTO handles may overshoot; complete actual intersection report retained','Within joint zone means candidate design join only; outside/unresolved must be inspected, no blanket exceptions','All surface intersection zeros do not establish containment; open tubes/core have limited solid semantics','No RGB/occlusion image rendered; visible appearance still requires later controlled assembly preview','Sourcebrace180phase is unmeasured; white actual location unknown; not realwearing/mount acceptance','Cable rail uses one full neutral-Yslab support plus original cable radius+1mm engineering margin; safe-routing estimate, not measured cable or proof of actual AUTOtube clearance']}
    text=bpy.data.texts.new(TEXT);text.write(json.dumps(contract,sort_keys=True));asset=out/'left_mount_components.blend';bpy.data.libraries.write(str(asset),{collection,text},path_remap='NONE',fake_user=True,compress=True)
    write(out/'module_contract.json',{**contract,'saved_asset':ref(asset)})
    write(out/'collision_diagnostics.json',{'pairs':collisions,'collision_skin_context':skin_context,'design_candidates_are_not_mechanical_acceptance':True,'production_accepted':False})
    np.savez_compressed(out/'actual_geometry.npz',**{key+'_'+tag:x for key,values in geometry.items() for tag,x in zip(['vertices_m','triangles'],values)})
    # Same authorized CPU process performs a clean library reload; no second build.
    bpy.ops.wm.read_factory_settings(use_empty=True)
    with bpy.data.libraries.load(str(asset),link=False) as (available,loaded):loaded.collections=available.collections;loaded.texts=available.texts
    if len(loaded.collections)!=1:raise ValueError('One independent collection required')
    bpy.context.scene.collection.children.link(loaded.collections[0]);bpy.context.view_layer.update()
    if env.digest(json.loads(bpy.data.texts[TEXT].as_string()))!=env.digest(contract):raise ValueError('Embedded contract mismatch')
    if set(o.name for o in bpy.context.scene.objects)!={rootname for rootname in contract['root_objects'].values()}|{x['name'] for x in state.values()}:raise ValueError('Unexpected dependency/fullbody in module')
    for key,expected in state.items():
        o=bpy.data.objects[expected['name']]
        if env.digest(assembly.mesh_state(o))!=env.digest(expected['state']) or not np.array_equal(np.array(o.matrix_world),np.array(expected['world_matrix'])):raise ValueError('Reload changed fixed geometry/material/frame')
        v,f=assembly.world_geometry(o)
        if not np.array_equal(v,geometry[key][0]) or not np.array_equal(f,geometry[key][1]):raise ValueError('Reload geometry differs')
    load_config(config_path)
    result={'status':'CPU_MODULE_BUILD_RELOAD_DIAGNOSTIC_ONLY','asset':ref(asset),'contract':ref(out/'module_contract.json'),'collisions':ref(out/'collision_diagnostics.json'),'collision_skin_context':skin_context,'geometry':ref(out/'actual_geometry.npz'),'all_fixed_source_bytes_unchanged':True,'reloaded_geometry_exact':True,'new_target_geometry':False,'canonical8':None,'production_accepted':False,'GPU':False,'RGB_rendered':False}
    write(out/'build_receipt.json',result);return result
