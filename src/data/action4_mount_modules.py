"""Prepare/reuse a diagnostic right mount; only an approved CPU build writes a library.

Fixed source pieces keep original data/basis (including the ball's .006 scale).
Link/cable endpoints are declared engineering anchors, never measured ports.
"""
import hashlib
import json
import traceback
from pathlib import Path
import numpy as np
from data import action4_global_environment as env
ROOT=env.ROOT
OUTPUT=ROOT/'outputs/action4_mount_modules_v001'
SOURCE_NAMES={'quick_release':'integrated_right_quick_release','usb_connector':'integrated_right_usb_connector','white_module':'integrated_right_white_module','ball':'integrated_right_ball'}
TEXT='Action4_Right_Mount_Diagnostic.json'


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


def rigid(M):
    M=np.asarray(M,float)
    if M.shape!=(4,4) or not np.isfinite(M).all() or not np.array_equal(M[3],[0,0,0,1]) or np.max(abs(M[:3,:3].T@M[:3,:3]-np.eye(3)))>1e-6 or abs(np.linalg.det(M[:3,:3])-1)>1e-6:
        raise ValueError('Proper fixed metric transform required')
    return M


def transform(M,V):return np.asarray(V)@np.asarray(M)[:3,:3].T+np.asarray(M)[:3,3]


def load_config(config_path):
    c=read(config_path)
    if set(c)!={'schema','status','dataset_generation_allowed','sources','routing','output_dir'} or c['schema']!='action4.mount_modules.v1' or c['status']!='diagnostic_pending_build_review' or c['dataset_generation_allowed'] is not False:raise ValueError('Exact diagnostic config required')
    if path(c['output_dir'])!=OUTPUT:raise ValueError('Fixed output directory required')
    required={'source_asset','source_contract','source_scene_helper','render_tube_helper','material_state_helper','assembly_helper','joint_result','registry','registry_validator','wrist_geometry','wrist_contract','brace_geometry','brace_contract','CAD_geometry','inventory','right_relation_config','right_relation_roll_report','right_relation_root_quality','right_relation_actual_frame','right_relation_actual_geometry','right_relation_actual_result'}
    if set(c['sources'])!=required:raise ValueError('Explicit source set required')
    for r in c['sources'].values():
        if set(r)!={'path','sha256'} or sha(r['path'])!=r['sha256']:raise ValueError('Frozen source changed')
    if c['routing']!={'quick_approach_clearance_m':.001,'cable_exit_extra_clearance_m':.003,'nominal_joint_zone_margin_m':.001,'source_reference_shift_m':.003,'route_policy':'preserve_band_ball_anchor_and_CAD_negativeZ_approach; cable_source_station_magnitudes_on_USB_lateral_side'}:raise ValueError('Unapproved routing parameters')
    result=read(c['sources']['joint_result']['path'])
    if result.get('production_accepted') is not False or result.get('body_and_brace_used_in_fit_not_independent_validation') is not True:raise ValueError('Diagnostic joint source required')
    rigid(result['A_shared'])
    import importlib.util
    spec=importlib.util.spec_from_file_location('mount_frozen_registry',path(c['sources']['registry_validator']['path']));verifier=importlib.util.module_from_spec(spec);spec.loader.exec_module(verifier)
    verifier.verify(c['sources']['registry']['path'])
    return c


def routes(c):
    source=read(c['sources']['source_contract']['path']);p=source['config']['initial_priors'];a=source['assembly']['assembly_priors']['right'];A=rigid(read(c['sources']['joint_result']['path'])['A_shared'])
    S=np.eye(4);S[1,3]=-c['routing']['source_reference_shift_m']
    old_link=transform(S,np.array(a['mount_link_brace_m']));old_cable=transform(S,np.array(a['cable_points_brace_m']))
    q=np.array(a['quick_release_center_CAD_m']);usb=np.array(a['USB_center_CAD_m']);usb_exit=usb+np.array([0,0,-p['usb_dimensions_m'][2]/2])
    end=transform(A,q);approach=transform(A,q+[0,0,-(p['quick_release_dimensions_m'][2]/2+p['link_radius_m']+c['routing']['quick_approach_clearance_m'])])
    ball=transform(S,np.array(a['ball_center_brace_m']));band=old_link[0]
    elbow=np.array([approach[0],approach[1],ball[2]])
    link=np.array([band,ball,elbow,approach,end])
    start=transform(A,usb_exit);exit_lead=transform(A,usb_exit+[0,0,-(2*p['cable_radius_m']+c['routing']['cable_exit_extra_clearance_m'])])
    sign=-1 if start[0]<0 else 1
    stations=old_cable[1:].copy();stations[:,0]=sign*abs(stations[:,0])
    transition=np.array([stations[0,0],stations[0,1],max(exit_lead[2],stations[0,2])])
    cable=np.vstack([start,exit_lead,transition,stations])
    for curve in [link,cable]:
        if not np.isfinite(curve).all() or np.any(np.linalg.norm(np.diff(curve,axis=0),axis=1)<1e-7):raise ValueError('Finite nonduplicate route stations required')
    if not np.array_equal(link[0],old_link[0]) or not np.array_equal(link[1],ball) or not np.array_equal(link[-1],end) or not np.array_equal(cable[0],start):raise ValueError('Nominal endpoint binding failed')
    return {'A_new_brace_from_CAD':A.tolist(),'old_brace_to_new_brace':S.tolist(),'source_link_path_reexpressed_m':old_link.tolist(),'source_cable_path_reexpressed_m':old_cable.tolist(),'derived_link_path_m':link.tolist(),'derived_cable_path_m':cable.tolist(),'link_radius_m':p['link_radius_m'],'cable_radius_m':p['cable_radius_m'],'ball_radius_m':p['ball_radius_m'],'quick_release_center_CAD_m':q.tolist(),'USB_exit_CAD_m':usb_exit.tolist(),'quick_release_center_newbrace_m':end.tolist(),'USB_exit_newbrace_m':start.tolist(),'ball_center_newbrace_m':ball.tolist(),'band_anchor_newbrace_m':band.tolist(),'cable_lateral_side':sign,'original_link_endpoint_gap_to_new_anchor_m':float(np.linalg.norm(old_link[-1]-end)),'original_cable_start_gap_to_new_anchor_m':float(np.linalg.norm(old_cable[0]-start)),'anchor_claim':'Unmeasured prior quickrelease center and USBbox -Z exit; not actual product port/socket measurement','routing_claim':'Single fixed route, no search/optimization. AUTO Bezier may overshoot; actual full derived mesh must be checked before any assembly acceptance.','source_dimensions_m':{k:p[k] for k in ['quick_release_dimensions_m','usb_dimensions_m','white_module_dimensions_m']}}


def joint_zones(c,r):
    """Narrow candidate contact regions; never grant mechanical permission."""
    A=np.array(r['A_new_brace_from_CAD']);q=np.array(r['quick_release_center_CAD_m']);u=np.array(r['USB_exit_CAD_m']);margin=c['routing']['nominal_joint_zone_margin_m'];dims=r['source_dimensions_m'];lr=r['link_radius_m'];cr=r['cable_radius_m'];ball=np.array(r['ball_center_newbrace_m']);band=np.array(r['band_anchor_newbrace_m'])
    def zone(frame,center,half,claim):return {'module_to_zone':frame.tolist(),'min_m':(center-half).tolist(),'max_m':(center+half).tolist(),'claim':claim}
    I=np.eye(4);inv=np.linalg.inv(A);zmin=q[2]+dims['quick_release_dimensions_m'][2]/2
    return {'core__quick_release':zone(inv,np.array([0,0,zmin]),np.array([dims['quick_release_dimensions_m'][0]/2+margin,dims['quick_release_dimensions_m'][1]/2+margin,margin]),'Source nominal CAD -Z support contact slab; not a verified real mounting port'),
            'link__quick_release':zone(inv,q+np.array([0,0,-dims['quick_release_dimensions_m'][2]/4]),np.array([lr+margin,lr+margin,dims['quick_release_dimensions_m'][2]/4+margin]),'Narrow center approach into quickrelease bottom half'),
            'ball__link':zone(I,ball,np.full(3,r['ball_radius_m']+margin),'Tube/sphere design joint candidate, bounded by source sphere'),
            'ball__band':zone(I,band,np.array([r['ball_radius_m']+margin,r['ball_radius_m']+margin,2*margin]),'Nominal sphere seat near outer band surface'),
            'band__link':zone(I,band,np.array([lr+margin,lr+margin,2*margin]),'Nominal tube seat near outer band surface'),
            'cable__usb_connector':zone(inv,u,np.array([cr+margin,cr+margin,cr+margin]),'Nominal USBbox -Z cable exit, not measured socket')}


def classify_pairs(Va,Fa,Vb,Fb,pairs,zone=None):
    """Conservative overlap-AABB containment in a nominated joint zone.

    The overlap box contains the actual triangle intersection. Outside/unproved
    cases remain unresolved, and even inside cases are only design candidates.
    """
    inside=[];unresolved=[]
    if zone is not None:
        a=transform(zone['module_to_zone'],Va);b=transform(zone['module_to_zone'],Vb);lo=np.array(zone['min_m']);hi=np.array(zone['max_m'])
    for i,j in pairs:
        ok=False
        if zone is not None:
            lower=np.maximum(a[Fa[i]].min(0),b[Fb[j]].min(0));upper=np.minimum(a[Fa[i]].max(0),b[Fb[j]].max(0))
            ok=bool(np.all(lower<=upper+1e-12) and np.all(lower>=lo) and np.all(upper<=hi))
        (inside if ok else unresolved).append([int(i),int(j)])
    return {'surface_pairs':len(pairs),'within_nominal_joint_zone_candidates':inside,'outside_or_unresolved_pairs':unresolved,'mechanical_acceptance':False,'zero_pairs_not_containment_proof':True}


def collision_skin_context(c, supplied=None):
    """Use the reviewed rightS and verify its actual-assembly frame provenance.

    This changes only collision context, never skin mesh data or A/routes.
    The previous actual scene has float32 transform roundoff; its existing
    shared-local-geometry1um tolerance is retained, not a wearing-fit gate.
    """
    keys=['right_relation_config','right_relation_roll_report','right_relation_root_quality','right_relation_actual_frame','right_relation_actual_geometry','right_relation_actual_result','wrist_geometry']
    for key in keys:
        r=c['sources'][key]
        if sha(r['path'])!=r['sha256']:raise ValueError('Skin context source identity changed')
    rel=read(c['sources']['right_relation_config']['path']);roll=read(c['sources']['right_relation_roll_report']['path']);quality=read(c['sources']['right_relation_root_quality']['path'])
    frame=read(c['sources']['right_relation_actual_frame']['path']);actual=read(c['sources']['right_relation_actual_result']['path']);joint=read(c['sources']['joint_result']['path'])
    S=rigid(rel['S_skin_relative_to_brace'])
    if rel['joint_result_sha256']!=c['sources']['joint_result']['sha256'] or not np.array_equal(rel['A_shared'],joint['A_shared']):raise ValueError('Skin relation and fixedA come from different joint result')
    if len(roll['checks'])!=1 or not roll['checks'][0]['bounded_contact_checks_pass'] or not np.array_equal(S,roll['checks'][0]['same_S_all_frames_diagnostic_only']):raise ValueError('S differs from reviewed roll candidate')
    if quality['status']!='PASS_BOUNDED_CACHED_RELATION_CANDIDATE_ONLY' or quality['candidate']['roll_deg']!=roll['checks'][0]['roll_Y_deg'] or not np.allclose(S[:3,3],quality['candidate']['translation_m'],rtol=0,atol=1e-14):raise ValueError('RightS root-quality provenance mismatch')
    if actual['joint_result_sha256']!=c['sources']['joint_result']['sha256'] or actual['A_and_S_shared'] is not True or frame['same_shared_A_S'] is not True:raise ValueError('Actual relation evidence mismatch')
    fid=str(frame['frame']);A=rigid(joint['A_shared']);C=rigid(joint['C_per_frame'][fid]);W=rigid(rel['W_from_capture_engineering_embedding']);WB=W@C@np.linalg.inv(A)
    if not np.allclose(frame['expected_WB'],WB,rtol=0,atol=1e-12) or not np.allclose(frame['expected_WBS'],WB@S,rtol=0,atol=1e-12):raise ValueError('Actual relation frame equations differ')
    skin=np.load(path(c['sources']['wrist_geometry']['path']),allow_pickle=False);V=transform(S,skin['brace_vertices_m']);F=skin['faces']
    if supplied is not None:
        givenV,givenF=supplied
        if not np.array_equal(givenV,V) or not np.array_equal(givenF,F):raise ValueError('Collision skin must use the exact reviewedS, not identity or another relation')
    cache=np.load(path(c['sources']['right_relation_actual_geometry']['path']),allow_pickle=False)
    actualV=cache['right_forearm_natural_hand_vertices_m'];actualF=cache['right_forearm_natural_hand_triangles'];T=rigid(frame['actual_object_world']['brace_module_root'])
    local=transform(np.linalg.inv(T),actualV);error=float(np.max(abs(local-V)))
    tolerance=rel['tolerances']['shared_local_geometry_m']
    if tolerance!=1e-6 or error>tolerance or not np.array_equal(actualF,F):raise ValueError('Collision context does not match previous actual assembly local skin')
    meta={'frame':'new34brace','S_skin_relative_to_brace':S.tolist(),'sources':{k:c['sources'][k] for k in keys},'previous_actual_frame':frame['frame'],'previous_actual_world_to_brace':np.linalg.inv(T).tolist(),'maximum_previous_actual_local_difference_m':error,'existing_shared_local_geometry_tolerance_m':tolerance,'geometry_vertex_sha256':hashlib.sha256(np.asarray(V,dtype=np.float64).tobytes()).hexdigest(),'triangles_sha256':hashlib.sha256(np.asarray(F,dtype=np.int64).tobytes()).hexdigest(),'identity_substitution_max_difference_m':float(np.max(abs(V-skin['brace_vertices_m']))),'skin_wrist_proxy_m':transform(S,np.array([0,.032,0])).tolist(),'A_changed':False,'S_adjusted_by_this_fix':False,'wearing_alignment_accepted':False,'claim':'Exact reviewed rightS collision context, consistent with existing actual assembly within its original1um float32 tolerance; not real wearing validation'}
    return V,F,meta


def prepare(config_path):
    c=load_config(config_path);OUTPUT.mkdir(parents=True,exist_ok=True);r=routes(c);_,_,skin_context=collision_skin_context(c)
    result={'status':'PREPARED_ONLY_NO_BUILD','config':ref(config_path),'source':ref(__file__),'routes':r,'collision_skin_context':skin_context,'joint_candidate_zones':joint_zones(c,r),'fixed_piece_policy':'Copy original mesh/material/basis; original ball .006 childscale fixed, no arbitraryscale or mirror','target_geometry':False,'canonical8':None,'dataset_generation_allowed':False,'max_authorized_future_builds':1,'build_authorized':False,'GPU':False}
    write(OUTPUT/'preparation_v002.json',result);return result


def _approval(c,approval_path):
    ap=read(approval_path)
    if ap.get('status')!='APPROVED_SINGLE_CPU_RIGHT_MOUNT_BUILD' or ap.get('preparation_handoff_sha256')!=sha(OUTPUT/'preparation_handoff_v002.json'):raise ValueError('New bound root approval required')
    for key in ['spec_review','root_review']:
        if ap.get(key+'_pass') is not True or sha(ap[key+'_path'])!=ap[key+'_sha256']:raise ValueError('Review approval binding required')
    h=read(OUTPUT/'preparation_handoff_v002.json')
    for p,digest in h['files'].items():
        if sha(p)!=digest:raise ValueError('Frozen preparation changed')
    if c!=load_config(h['config']['path']):raise ValueError('Requested config differs from approved preparation')
    return ap


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
    originals={key:source_scene.objects[name] for key,name in SOURCE_NAMES.items()};original_states={k:assembly.mesh_state(o) for k,o in originals.items()}
    if any(o.type!='MESH' or len(o.modifiers) or len(o.constraints) or o.animation_data is not None for o in originals.values()):raise ValueError('Static reviewed source parts required')
    scene=bpy.data.scenes.new('MountDiagnosticOnly');bpy.context.window.scene=scene;scene.unit_settings.system='METRIC';scene.unit_settings.scale_length=1
    collection=bpy.data.collections.new('right_mount_components');scene.collection.children.link(collection)
    def empty(name,parent,M):
        o=bpy.data.objects.new(name,None);collection.objects.link(o);o.parent=parent;o.matrix_parent_inverse=Matrix.Identity(4);assembly.set_rigid_basis(o,M);return o
    root=empty('mount_right_newbrace_root',None,np.eye(4));oldbrace=empty('mount_right_sourcebrace_subframe',root,S);cad=empty('mount_right_CAD_subframe',root,A)
    objects={}
    for key,o in originals.items():
        copy=o.copy();copy.name='mount_right_'+key;copy.parent=cad if key in ['quick_release','usb_connector'] else oldbrace;copy.matrix_parent_inverse=o.matrix_parent_inverse.copy();copy.matrix_basis=o.matrix_basis.copy();collection.objects.link(copy);objects[key]=copy
        if env.digest(assembly.mesh_state(copy))!=env.digest(original_states[key]):raise ValueError('Fixed source data/material changed')
        if key=='ball' and not np.allclose(copy.scale,(.006,.006,.006),rtol=0,atol=1e-9):raise ValueError('Original metric ball scale must not be reset')
    black=originals['quick_release'].data.materials[0]
    route_objects=[]
    for key,points,radius in [('link',r['derived_link_path_m'],r['link_radius_m']),('cable',r['derived_cable_path_m'],r['cable_radius_m'])]:
        curve=_tube('mount_right_'+key+'_curve',points,radius,black,root);bpy.context.view_layer.update();ev=curve.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=bpy.data.meshes.new_from_object(ev,depsgraph=bpy.context.evaluated_depsgraph_get())
        o=bpy.data.objects.new('mount_right_'+key,mesh);collection.objects.link(o);o.parent=root;o.matrix_parent_inverse=Matrix.Identity(4);o.matrix_basis=Matrix.Identity(4);objects[key]=o
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
    contract={'schema':'action4.mount_modules.diagnostic.v1','status':'UNMEASURED_PRIOR_DERIVED_MOUNT_NOT_CALIBRATED','config':ref(config_path),'source':ref(__file__),'sources':c['sources'],'A_source_role':'joint_v003 fixed diagnosticA, not measured/production','routes':r,'collision_skin_context':skin_context,'objects':state,'source_fixed_states':original_states,'joint_candidate_zones':zones,'root_objects':{'newbrace':root.name,'oldbrace_subframe':oldbrace.name,'CAD_subframe':cad.name},'canonical8':None,'target_geometry':False,'dataset_generation_allowed':False,'production_accepted':False,'limitations':['Actual mechanical ports not measured; nominal center/boxface anchors only','Source fixed ball childscale retained; no free scaling','Curve AUTO handles may overshoot; complete actual intersection report retained','Within joint zone means candidate design join only; outside/unresolved must be inspected, no blanket exceptions','All surface intersection zeros do not establish containment; open tubes/core have limited solid semantics','No RGB/occlusion image rendered; visible appearance still requires later controlled assembly preview']}
    text=bpy.data.texts.new(TEXT);text.write(json.dumps(contract,sort_keys=True));asset=out/'right_mount_components.blend';bpy.data.libraries.write(str(asset),{collection,text},path_remap='NONE',fake_user=True,compress=True)
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
