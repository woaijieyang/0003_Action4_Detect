"""Source-preserving LEFT neutral + RIGHT Y25 common-f0 scene composition.

Import/prepare are CPU/cache only. Explicit create_scene owns Blender assembly;
this module contains no pixel renderer, job dispatch, or save/build controller.
"""
from pathlib import Path
import hashlib,json
import numpy as np
from data import action4_modular_composed_render as composed
from data import action4_modular_assembly as assembly
from data import action4_global_environment as env
from data import action4_global_scene as source_scene
from data import action4_wrist_modules as wrist
from data import action4_mount_modules as mount_algorithms
from data import action4_canonical_export as canonical
ROOT=Path('/data/zy/vln/a-test')
OWN=ROOT/'outputs/action4_modular_bilateral_prepare_v003'
STATUS='BILATERAL_F0_LEFT_NEUTRAL_RIGHT_Y25_ESTIMATED_NOT_TRAINING'
LEFT_TARGETS=('left_Action4_core','left_Action4_core_top_id_decal')
RIGHT_TARGETS=tuple(composed.TARGET_NAMES)
BAND_RENAMES={'brace_module_root':'left_brace_module_root','brace_band':'left_brace_band','wrist_reference':'left_wrist_reference'}
MOUNT_KEYS=composed.MOUNT_KEYS
SOURCE_KEYS={'composed_config','bilateral_module','left_assembly','left_axes','left_neutral_row','left_neutral_geometry','left_wrist_asset','left_wrist_contract','left_wrist_geometry','left_band_asset','left_band_contract','left_band_geometry','left_mount_asset','left_mount_contract','left_mount_geometry','left_mount_receipt','left_mount_collision','source_asset','source_contract','registry_v003','assembly_helper','wrist_helper','source_scene_helper','environment_helper','mount_algorithms'}
bounded,sha,read=composed.bounded,composed.sha,composed.read
transform=assembly.transform

def frozen_sources(c):
    if set(c['sources'])!=SOURCE_KEYS:raise ValueError('Complete bilateral sources including actual LEFT mount required; no placeholder SHA')
    for name,r in c['sources'].items():
        if set(r)!={'path','sha256'} or len(r['sha256'])!=64 or sha(r['path'])!=r['sha256']:raise ValueError('Bilateral frozen source changed '+name)
    if bounded(c['sources']['bilateral_module']['path'])!=Path(__file__).resolve():raise ValueError('Loaded bilateral module identity')
    for role,module in [('assembly_helper',assembly),('wrist_helper',wrist),('source_scene_helper',source_scene),('environment_helper',env),('mount_algorithms',mount_algorithms)]:
        if bounded(c['sources'][role]['path'])!=Path(module.__file__).resolve():raise ValueError('Loaded helper identity '+role)

def validate_mount(c,joint):
    """A build/reload receipt alone never grants contact acceptance."""
    mc=read(c['sources']['left_mount_contract']['path']);receipt=read(c['sources']['left_mount_receipt']['path']);collision=read(c['sources']['left_mount_collision']['path'])
    if receipt['status']!='CPU_MODULE_BUILD_RELOAD_DIAGNOSTIC_ONLY' or receipt['reloaded_geometry_exact'] is not True or receipt['all_fixed_source_bytes_unchanged'] is not True:raise ValueError('Actual LEFT build/reload provenance required')
    for role,key in [('asset','left_mount_asset'),('contract','left_mount_contract'),('geometry','left_mount_geometry'),('collisions','left_mount_collision')]:
        if receipt[role]!=c['sources'][key]:raise ValueError('Actual LEFT mount receipt binding '+role)
    keys=sorted(['core','skin','band',*MOUNT_KEYS]);expected_pairs={a+'__'+b for i,a in enumerate(keys) for b in keys[i+1:]}
    if set(mc['objects'])!=set(MOUNT_KEYS) or set(collision['pairs'])!=expected_pairs:raise ValueError('Complete six-piece LEFT mount and exact36 pair identities required')
    if any(not row['name'].startswith('mount_left_') for row in mc['objects'].values()):raise ValueError('Original LEFT mount identities required')
    if any(row['outside_or_unresolved_pairs'] for row in collision['pairs'].values()):raise ValueError('LEFT mount unresolved contacts block bilateral scene; build success is insufficient')
    if not np.array_equal(mc['routes']['A_new_brace_from_CAD'],joint['A_shared']) or not np.array_equal(mc['collision_skin_context']['S_skin_relative_to_brace'],joint['S_skin_relative_to_brace']):raise ValueError('LEFT mount A/S differs')
    if mc['collision_skin_context']['pose']!={'left_hand':[0,0,0]} or mc['collision_skin_context']['side']!='left':raise ValueError('LEFT mount requires neutral context; no Y25 substitution')
    if mc['objects']['ball']['scale']!=[.006000000052154064]*3:raise ValueError('Fixed LEFT ball childscale')
    return mc

def left_geometry_inputs(c,right_ctx):
    j=read(c['sources']['left_assembly']['path']);axes=read(c['sources']['left_axes']['path']);wc=read(c['sources']['left_wrist_contract']['path']);bc=read(c['sources']['left_band_contract']['path']);row=read(c['sources']['left_neutral_row']['path'])
    if j['side']!='left' or j['instance_id']!=1 or j['A_candidate']!='K0p8125_start1' or j['nonunique_A'] is not True or j['S_conditioned_candidate_deg']!=3.75:raise ValueError('Fixed nonunique LEFT f0 diagnostic authority')
    for a,b in [('A_shared','A'),('K_native','K_native'),('S_skin_relative_to_brace','S_left')]:
        if j[a]!=axes[b]:raise ValueError('LEFT fixed authority differs '+a)
    if j['C_per_frame']!={'0':axes['C']} or wc['rigid_F_new_body']!=axes['F_left'] or wc['side']!='left' or bc['side']!='left' or bc['width_m']!=.034 or bc['inner_radii_m']!=[.02877563688416745,.03260006155662128]:raise ValueError('LEFT C/F/source radii/width')
    A=assembly.rigid(j['A_shared']);C=assembly.rigid(j['C_per_frame']['0']);S=assembly.rigid(j['S_skin_relative_to_brace']);W=assembly.rigid(j['W_from_capture_engineering_embedding']);B=C@np.linalg.inv(A);Fi=np.linalg.inv(wc['rigid_F_new_body']);K=np.array(j['K_native'])
    if not np.array_equal(W,right_ctx['geometry']['W']) or not np.array_equal(K,right_ctx['rc']['K_native']):raise ValueError('Same f0/common W/K required')
    if row['status']!='LEFT_NEUTRAL_CACHE_ONLY_BVH_BASELINE' or row['pose']!={'left_hand':[0,0,0]} or row['instance_id']!=1 or row['self_intersections']['pair_count']!=99 or row['self_triangle_pair_count']!=173 or row['core_skin_pairs'] or row['band_skin_pairs']:raise ValueError('Independent LEFT neutral99/173 baseline required')
    for reference,role in [(wc['saved_asset'],'left_wrist_asset'),(wc['geometry'],'left_wrist_geometry'),(bc['saved_asset'],'left_band_asset'),(wc['config']['sources']['asset'],'source_asset'),(wc['config']['sources']['contract'],'source_contract')]:
        if reference!=c['sources'][role]:raise ValueError('Original LEFT asset/source reference differs '+role)
    mod=np.load(bounded(c['sources']['left_wrist_geometry']['path']),allow_pickle=False);cache=np.load(bounded(c['sources']['left_neutral_geometry']['path']),allow_pickle=False);band=np.load(bounded(c['sources']['left_band_geometry']['path']),allow_pickle=False)
    if mod['source_vertices_m'].shape!=(8663,3) or mod['faces'].shape!=(17280,3) or not np.array_equal(cache['source_body_vertices'],mod['source_vertices_m']) or not np.array_equal(cache['skin_faces'],mod['faces']):raise ValueError('Exact neutral LEFT source module')
    if np.max(abs(cache['skin_capture_vertices']-transform(B@S@Fi,mod['source_vertices_m'])))>1e-14:raise ValueError('Existing LEFT analytic roundoff consistency bound')
    if band['vertices_module_m'].shape!=(432,3) or band['faces'].shape!=(432,4):raise ValueError('Original LEFT34 band topology')
    return {'j':j,'wc':wc,'bc':bc,'row':row,'mod':mod,'cache':cache,'band':band,'A':A,'C':C,'S':S,'W':W,'B':B,'Fi':Fi,'K':K}

def prepare(config):
    if Path.cwd().resolve()!=ROOT:raise ValueError('Physical project root required')
    c=read(config) if not isinstance(config,dict) else config
    if set(c)!={'schema','status','dataset_generation_allowed','frame_id','instance_ids','resolution_wh','sources','output_scope'} or c['schema']!='action4.modular_bilateral_scene.v1' or c['status']!='diagnostic_pending_scene_review' or c['dataset_generation_allowed'] is not False:raise ValueError('Explicit diagnostic bilateral config')
    if (c['frame_id'],c['instance_ids'],c['resolution_wh'])!=(0,[1,2],[3248,2464]) or bounded(c['output_scope'])!=OWN/'runs':raise ValueError('One common-f0 bilateral scene scope')
    frozen_sources(c)
    rc,rctx=composed.prepare(c['sources']['composed_config']['path'])
    if rc['sources']['registry_v003']!=c['sources']['registry_v003']:raise ValueError('Common frozen18 registry binding')
    left=left_geometry_inputs(c,rctx);mc=validate_mount(c,left['j'])
    return c,{'right_config':rc,'right_context':rctx,'left':left,'left_mount_contract':mc}

def _append_objects(scene,asset,names,renames=None):
    import bpy
    source_names=tuple(names)
    renames=renames or {n:n for n in source_names}
    if set(renames)!=set(source_names) or any(n in scene.objects for n in renames.values()):raise ValueError('Explicit fresh LEFT instance names required')
    with bpy.data.libraries.load(str(bounded(asset)),link=False) as (available,loaded):
        if not set(source_names)<=set(available.objects):raise ValueError('Exact frozen LEFT objects missing')
        loaded.objects=list(source_names)
    result={name:o for name,o in zip(source_names,loaded.objects)}
    for name,o in result.items():
        if renames[name]!=o.name and renames[name] in bpy.data.objects:raise ValueError('Reserved LEFT instance name already in data')
        o.name=renames[name];scene.collection.objects.link(o)
    return result

def _preserve_materials(o,expected,actions):
    assembly.share_exact_materials(o,expected,actions)
    if [env.digest(env._material_state(m)) for m in o.data.materials]!=[env.digest(s) for s in expected]:raise ValueError('Exact LEFT source material graph required')

def _group_state(o):
    return env.digest({'names':[g.name for g in o.vertex_groups],'weights':[(v.index,g.group,g.weight) for v in o.data.vertices for g in v.groups]})

def _right_state(scene,names):
    state=composed.neutral.new_scene_state(scene,None)
    return {'objects':{n:state['objects'][n] for n in names},'common':{k:v for k,v in state.items() if k not in ['objects','modular_preview_status']}}

def targets(scene):
    result={};allparts=[]
    for instance_id,names in [(1,LEFT_TARGETS),(2,RIGHT_TARGETS)]:
        core=scene.objects[names[0]];parts=[core]+list(core.children_recursive)
        if set(o.name for o in parts)!=set(names) or any(o.type!='MESH' or o.pass_index!=instance_id for o in parts):raise ValueError('Original core+own decal target ownership required')
        result[instance_id]={'core':core,'parts':parts};allparts+=parts
    if any(o.pass_index!=0 for o in scene.objects if o not in allparts):raise ValueError('All non-target objects must have pass0')
    return result

def create_scene(c,ctx):
    """RIGHT verifier runs before any LEFT import; final verifier owns bilateral scope."""
    import bpy
    from mathutils import Matrix
    scene,right_contract=composed.create_scene(ctx['right_config'],ctx['right_context'])
    composed.verify_composed_scene(scene,ctx['right_config'],ctx['right_context'],right_contract)
    right_names=set(scene.objects.keys());right_state=_right_state(scene,right_names);g=ctx['left'];wc=g['wc'];bc=g['bc'];mc=ctx['left_mount_contract'];actions=[]
    objects=_append_objects(scene,c['sources']['left_wrist_asset']['path'],[wc['root_name'],wc['object_name']])
    wr,skin=objects[wc['root_name']],objects[wc['object_name']]
    _preserve_materials(skin,[wc['material_state']],actions);wrist._verify_mesh(skin,wc,g['mod']);skin_source_state=assembly.mesh_state(skin);skin_groups=_group_state(skin)
    assembly.set_rigid_basis(wr,g['W']@g['B']@g['S'])
    band_objects=_append_objects(scene,c['sources']['left_band_asset']['path'],list(BAND_RENAMES),BAND_RENAMES)
    br,band,reference=(band_objects[n] for n in ['brace_module_root','brace_band','wrist_reference'])
    _preserve_materials(band,bc['material_states'],actions)
    if not np.array_equal(np.array([v.co[:] for v in band.data.vertices]),g['band']['vertices_module_m']) or not np.array_equal(np.array([p.vertices[:] for p in band.data.polygons]),g['band']['faces']) or not np.array_equal(np.array([p.material_index for p in band.data.polygons]),g['band']['material_indices']):raise ValueError('Imported LEFT band differs from frozen geometry/material indices')
    band_state=assembly.mesh_state(band);assembly.set_rigid_basis(br,g['W']@g['B'])
    source=read(c['sources']['source_contract']['path']);source_rows=source['original_invariants']['objects'];camera=_append_objects(scene,c['sources']['source_asset']['path'],list(LEFT_TARGETS));camera_source={};relative={}
    for name,o in camera.items():
        row=source_rows[name]
        if o.type!='MESH' or source_scene._geometry(o)!=row['mesh'] or not np.array_equal(o.matrix_basis,row['basis']) or not np.array_equal(o.matrix_parent_inverse,row['parent_inverse']) or len(o.modifiers) or len(o.constraints) or o.animation_data is not None or o.pass_index!=1:raise ValueError('Original LEFT source target data/local matrix/pass differs')
        # Keep the LEFT object's own slots, including ID1 decal image; never substitute RIGHT ID2.
        own=[env._material_state(m) for m in o.data.materials];_preserve_materials(o,own,actions);camera_source[name]=assembly.mesh_state(o);relative[name]=np.array(o.matrix_parent_inverse)@np.array(o.matrix_basis)
    core=camera[LEFT_TARGETS[0]];decal=camera[LEFT_TARGETS[1]]
    if decal.parent!=core:raise ValueError('Original LEFT decal ownership')
    core.parent=br;core.matrix_parent_inverse=Matrix.Identity(4);assembly.set_rigid_basis(core,g['A'])
    decal.parent=core;decal.matrix_parent_inverse=Matrix.Identity(4);decal.matrix_basis=Matrix(relative[decal.name])
    with bpy.data.libraries.load(str(bounded(c['sources']['left_mount_asset']['path'])),link=False) as (available,loaded):
        if len(available.collections)!=1:raise ValueError('One explicit independent LEFT mount collection')
        loaded.collections=available.collections;loaded.texts=available.texts
    collection=loaded.collections[0];scene.collection.children.link(collection);bpy.context.view_layer.update()
    embedded=dict(mc);embedded.pop('saved_asset')
    # The explicit asset/contract refs can point to a future LEFT mount version.
    matching_texts=[]
    for text in loaded.texts:
        try:matches=env.digest(json.loads(text.as_string()))==env.digest(embedded)
        except (ValueError,TypeError):matches=False
        if matches:matching_texts.append(text.name)
    if len(matching_texts)!=1:raise ValueError('Exactly one LEFT mount embedded/source contract match')
    if set(o.name for o in collection.all_objects)!=set(mc['root_objects'].values())|{r['name'] for r in mc['objects'].values()}:raise ValueError('LEFT mount extra dependency')
    for key,row in mc['objects'].items():
        o=scene.objects[row['name']];_preserve_materials(o,row['state']['material_states'],actions)
        if env.digest(assembly.mesh_state(o))!=env.digest(row['state']) or not np.array_equal(o.matrix_world,row['world_matrix']) or not np.array_equal(o.matrix_basis,row['basis']) or not np.array_equal(o.matrix_parent_inverse,row['parent_inverse']):raise ValueError('Original local LEFT mount state')
    assembly.set_rigid_basis(scene.objects[mc['root_objects']['newbrace']],g['W']@g['B'])
    for o in [wr,skin,br,band,reference]+list(collection.all_objects):o.pass_index=0;o['dataset_generation_allowed']=False
    scene['modular_preview_status']=STATUS;scene['dataset_generation_allowed']=False;bpy.context.view_layer.update();targets(scene)
    left_names={wc['root_name'],wc['object_name']}|set(BAND_RENAMES.values())|set(LEFT_TARGETS)|set(mc['root_objects'].values())|{r['name'] for r in mc['objects'].values()}
    if set(scene.objects.keys())-right_names!=left_names:raise ValueError('Exact LEFT instance inventory; no loaded source ancestors/fullbody')
    state=composed.neutral.new_scene_state(scene,None)
    contract={'schema':'action4.modular_bilateral_scene.contract.v1','status':STATUS,'dataset_generation_allowed':False,'sources':c['sources'],'right_verified_before_left_import':True,'right_contract':right_contract,'right_names':sorted(right_names),'right_state_before_left':right_state,'left_names':sorted(left_names),'left_source_mesh_states':{'skin':skin_source_state,'band':band_state,'camera':camera_source},'left_skin_vertex_groups_sha256':skin_groups,'left_original_decal_relative_to_core':relative[decal.name].tolist(),'left_source_target_invariants':{n:source_rows[n] for n in LEFT_TARGETS},'material_collision_actions':actions,'state':state,'state_sha256':env.digest(state),'canonical_definition':ctx['right_context']['oldcontract']['canonical_definition'],'left_pose':{'left_hand':[0,0,0]},'right_pose':{'right_hand':[0,25,0]},'target_names':{'1':list(LEFT_TARGETS),'2':list(RIGHT_TARGETS)},'wearing_alignment_accepted':False,'training_accepted':False,'mechanical_acceptance':False}
    verify_bilateral_scene(scene,c,ctx,contract)
    return scene,contract

def _check_static(o):
    if len(o.modifiers) or len(o.constraints) or o.animation_data is not None:raise ValueError('Static source object required '+o.name)

def verify_bilateral_scene(scene,c,ctx,contract):
    import bpy
    from bpy_extras.object_utils import world_to_camera_view
    from mathutils import Vector
    g=ctx['left'];mc=ctx['left_mount_contract'];states=contract['left_source_mesh_states']
    if set(scene.objects.keys())!=set(contract['right_names'])|set(contract['left_names']):raise ValueError('Exact independent bilateral inventory')
    if _right_state(scene,contract['right_names'])!=contract['right_state_before_left']:raise ValueError('RIGHT source objects/camera/environment changed after LEFT import')
    if scene.get('modular_preview_status')!=STATUS or scene.get('dataset_generation_allowed') is not False:raise ValueError('Diagnostic bilateral boundary')
    skin=scene.objects[g['wc']['object_name']];band=scene.objects['left_brace_band'];br=scene.objects['left_brace_module_root'];wr=scene.objects[g['wc']['root_name']]
    for o,M in [(wr,g['W']@g['B']@g['S']),(skin,g['W']@g['B']@g['S']@g['Fi']),(br,g['W']@g['B'])]:
        _check_static(o)
        if np.max(abs(np.array(o.matrix_world)-M))>2e-6:raise ValueError('LEFT fixed source frame '+o.name)
    if skin.parent!=wr or band.parent!=br or not np.array_equal(band.matrix_basis,np.eye(4)) or not np.array_equal(band.matrix_parent_inverse,np.eye(4)):raise ValueError('LEFT skin/band hierarchy')
    reference=scene.objects['left_wrist_reference'];expected_reference=np.eye(4);expected_reference[:3,3]=g['bc']['wrist_reference_m']
    if wr.parent is not None or br.parent is not None or reference.parent!=br or reference.type!='EMPTY' or not np.array_equal(reference.matrix_parent_inverse,np.eye(4)) or np.max(abs(np.array(reference.matrix_basis)-expected_reference))>2e-9:raise ValueError('LEFT original root/reference matrices')
    for o in [wr,skin,br,band,reference]:
        _check_static(o)
        if tuple(o.scale)!=(1.,1.,1.) or not np.array_equal(o.matrix_parent_inverse,np.eye(4)):raise ValueError('LEFT static unit-scale/source parent inverse')
    mountroot=scene.objects[mc['root_objects']['newbrace']]
    if mountroot.parent is not None or mountroot.type!='EMPTY' or np.max(abs(np.array(mountroot.matrix_basis)-g['W']@g['B']))>2e-6 or not np.array_equal(mountroot.matrix_parent_inverse,np.eye(4)):raise ValueError('LEFT mount newbrace root')
    for key,M in [('oldbrace_subframe',mc['routes']['old_brace_to_new_brace']),('CAD_subframe',mc['routes']['A_new_brace_from_CAD'])]:
        o=scene.objects[mc['root_objects'][key]];_check_static(o)
        if o.parent!=mountroot or o.type!='EMPTY' or tuple(o.scale)!=(1.,1.,1.) or not np.array_equal(o.matrix_parent_inverse,np.eye(4)) or np.max(abs(np.array(o.matrix_basis)-np.array(M)))>2e-6:raise ValueError('LEFT original mount subframe')
    if env.digest(assembly.mesh_state(skin))!=env.digest(states['skin']) or _group_state(skin)!=contract['left_skin_vertex_groups_sha256'] or env.digest(assembly.mesh_state(band))!=env.digest(states['band']):raise ValueError('LEFT frozen skin/band state/weights changed')
    V,F=assembly.world_geometry(skin)
    if np.max(abs(V-transform(g['W'],g['cache']['skin_capture_vertices'])))>1e-6 or not np.array_equal(F,g['mod']['faces']):raise ValueError('LEFT neutral cache/source world chain')
    core=scene.objects[LEFT_TARGETS[0]];decal=scene.objects[LEFT_TARGETS[1]]
    if core.parent!=br or decal.parent!=core or not np.array_equal(core.matrix_parent_inverse,np.eye(4)) or not np.array_equal(decal.matrix_parent_inverse,np.eye(4)) or np.max(abs(np.array(core.matrix_world)-g['W']@g['C']))>2e-6 or not np.array_equal(np.array(decal.matrix_basis),contract['left_original_decal_relative_to_core']):raise ValueError('LEFT core/decal source relative matrices')
    for name in LEFT_TARGETS:
        o=scene.objects[name];_check_static(o)
        if env.digest(assembly.mesh_state(o))!=env.digest(states['camera'][name]) or source_scene._geometry(o)!=contract['left_source_target_invariants'][name]['mesh']:raise ValueError('LEFT original target mesh/material changed')
    cache=np.load(bounded(c['sources']['left_mount_geometry']['path']),allow_pickle=False)
    for key,row in mc['objects'].items():
        o=scene.objects[row['name']];_check_static(o)
        if (o.parent.name if o.parent else None)!=row['parent'] or not np.array_equal(o.matrix_basis,row['basis']) or not np.array_equal(o.matrix_parent_inverse,row['parent_inverse']) or not np.array_equal(o.scale,row['scale']) or env.digest(assembly.mesh_state(o))!=env.digest(row['state']):raise ValueError('LEFT mount original local state/ballscale')
        V,F=assembly.world_geometry(o)
        if np.max(abs(np.array(o.matrix_world)-g['W']@g['B']@row['world_matrix']))>2e-6 or not np.array_equal(F,cache[key+'_triangles']) or np.max(abs(V-transform(g['W']@g['B'],cache[key+'_vertices_m'])))>1e-6:raise ValueError('LEFT mount complete cached geometry/source chain')
    if not np.allclose(composed.actual_K(scene),g['K'],rtol=0,atol=1e-6):raise ValueError('Common native K')
    exports={}
    for instance_id,target in targets(scene).items():
        M=np.array(target['core'].matrix_world);definition=contract['canonical_definition'];export=canonical.build_export(canonical=definition,instance_id=instance_id,object_to_world_m=M.tolist(),camera_to_world_blender_m=np.array(scene.camera.matrix_world).tolist(),K=composed.actual_K(scene),image_size_wh=[3248,2464],near_plane_m=scene.camera.data.clip_start)
        baseline=g['row']['canonical8'] if instance_id==1 else ctx['right_context']['Y25_row']['canonical8']
        if export['corner_ids']!=list(range(8)) or export['corner_order']!=baseline['corner_order'] or np.max(abs(np.array(export['points_native_px'])-np.array(baseline['points_native_px'])))>.003:raise ValueError('Fixed original instance8 reprojection changed')
        ideal=assembly.project(transform(g['C'] if instance_id==1 else ctx['right_context']['geometry']['C'],definition['corners_3d_m']),g['K']);actual=np.array(export['points_native_px'])
        if np.max(abs(actual-ideal))>.002:raise ValueError('Original fixedC8 projection')
        native=[]
        for point in transform(M,definition['corners_3d_m']):
            q=world_to_camera_view(scene,scene.camera,Vector(point));native.append([q.x*3248-.5,(1-q.y)*2464-.5])
        if np.max(abs(np.array(native)-actual))>.003:raise ValueError('Common actual Blender camera original8 projection')
        exports[str(instance_id)]=export
    if env.digest(composed.neutral.new_scene_state(scene,None))!=contract['state_sha256']:raise ValueError('Bilateral scene contract state changed')
    return {'status':'PASS_BILATERAL_SOURCE_TRANSFORM_CONTRACT_ONLY','canonical8_by_instance':exports,'right_unchanged':True,'wearing_accepted':False,'containment_accepted':False,'training_accepted':False}

def aabb_separation(Va,Vb):
    a,b=np.asarray(Va),np.asarray(Vb)
    if a.ndim!=2 or b.ndim!=2 or a.shape[1:]!=(3,) or b.shape[1:]!=(3,) or not len(a) or not len(b) or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Finite nonempty3D geometry required')
    gap=np.maximum(np.maximum(b.min(0)-a.max(0),a.min(0)-b.max(0)),0)
    return {'strictly_disjoint':bool(np.any(gap>0)),'axis_gaps_m':gap.tolist(),'distance_lower_bound_m':float(np.linalg.norm(gap))}

def filter_separated_bvh_candidates(Va,Fa,Vb,Fb,pairs,zone=None):
    """Keep all possible contacts; prove only AABB-empty conservative BVH candidates.
    The existing classifier1e-12 comparison is unchanged and reused literally.
    """
    if zone is None:a,b=np.asarray(Va,dtype=np.float64),np.asarray(Vb,dtype=np.float64);frame='world'
    else:a,b=transform(zone['module_to_zone'],Va),transform(zone['module_to_zone'],Vb);frame='existing_joint_zone'
    retained=[];separated=[]
    for i,j in pairs:
        amin,amax=a[Fa[i]].min(0),a[Fa[i]].max(0);bmin,bmax=b[Fb[j]].min(0),b[Fb[j]].max(0)
        lower=np.maximum(amin,bmin);upper=np.minimum(amax,bmax)
        if np.any(lower>upper+1e-12):
            separated.append({'classification':'separated_BVH_candidate','pair':[int(i),int(j)],'frame':frame,'triangle_A_AABB_min_m':amin.tolist(),'triangle_A_AABB_max_m':amax.tolist(),'triangle_B_AABB_min_m':bmin.tolist(),'triangle_B_AABB_max_m':bmax.tolist(),'intersection_lower_m':lower.tolist(),'intersection_upper_m':upper.tolist(),'axis_gaps_m':(lower-upper).tolist(),'separating_axes':np.flatnonzero(lower>upper+1e-12).tolist(),'existing_numeric_margin_m':1e-12,'not_surface_intersection':True,'world_to_test_frame':np.eye(4).tolist() if zone is None else np.asarray(zone['module_to_zone']).tolist()})
        else:retained.append((int(i),int(j)))
    return retained,separated


def geometry_diagnostics(scene,c,ctx,contract):
    """Actor surfaces only: source candidate joints stay unresolved outside old zones.
    Cross-instance AABB separation first; BVH epsilon0 only when boxes overlap.
    No skin selfbaseline relabel, environment/containment/mechanical acceptance.
    """
    from mathutils.bvhtree import BVHTree
    verify_bilateral_scene(scene,c,ctx,contract)
    sides={};geometries={};trees={}
    for side in ['left','right']:
        mc=ctx['left_mount_contract'] if side=='left' else ctx['right_context']['mount_contract'];g=ctx['left'] if side=='left' else ctx['right_context']['geometry']
        names={'core':side+'_Action4_core','skin':g['wc']['object_name'],'band':'left_brace_band' if side=='left' else 'brace_band'};names.update({k:row['name'] for k,row in mc['objects'].items()});sides[side]=names
        for key,name in names.items():geometries[side+'/'+key]=assembly.world_geometry(scene.objects[name])
    def surface_pair(a,b,zone=None):
        V,F=geometries[a];U,G=geometries[b];sep=aabb_separation(V,U)
        if sep['strictly_disjoint']:return {**sep,'surface_pairs':0,'BVH_evaluated':False,'BVH_candidate_pair_count':0,'separated_BVH_candidate_count':0,'separated_BVH_candidates':[],'outside_or_unresolved_pairs':[],'within_nominal_joint_zone_candidates':[],'mechanical_acceptance':False,'zero_pairs_not_containment_proof':True}
        for key,(vertices,faces) in [(a,(V,F)),(b,(U,G))]:
            if key not in trees:trees[key]=BVHTree.FromPolygons(vertices.tolist(),faces.tolist(),all_triangles=True,epsilon=0)
        pairs=trees[a].overlap(trees[b]);retained,separated=filter_separated_bvh_candidates(V,F,U,G,pairs,zone)
        return {**sep,**mount_algorithms.classify_pairs(V,F,U,G,retained,zone),'BVH_evaluated':True,'BVH_candidate_pair_count':len(pairs),'separated_BVH_candidate_count':len(separated),'separated_BVH_candidates':separated}
    cross={a+'__'+b:surface_pair('left/'+a,'right/'+b) for a in sides['left'] for b in sides['right']}
    internal={}
    for side,names in sides.items():
        mc=ctx['left_mount_contract'] if side=='left' else ctx['right_context']['mount_contract'];g=ctx['left'] if side=='left' else ctx['right_context']['geometry'];inv=np.linalg.inv(g['W']@g['B']);keys=sorted(names);rows={}
        for i,a in enumerate(keys):
            for b in keys[i+1:]:
                zone=mc['joint_candidate_zones'].get(a+'__'+b) or mc['joint_candidate_zones'].get(b+'__'+a)
                if zone is not None:zone={**zone,'module_to_zone':(np.array(zone['module_to_zone'])@inv).tolist()}
                rows[a+'__'+b]=surface_pair(side+'/'+a,side+'/'+b,zone)
        internal[side]=rows
    unresolved=sum(len(r['outside_or_unresolved_pairs']) for r in cross.values())+sum(len(r['outside_or_unresolved_pairs']) for rows in internal.values() for r in rows.values())
    return {'status':'BILATERAL_ACTOR_SURFACE_DIAGNOSTICS_REQUIRE_REVIEW','cross_pairs':cross,'cross_pair_count':81,'internal_pairs':internal,'internal_pair_count_each':36,'outside_or_unresolved_surface_pairs':unresolved,'BVH_epsilon':0,'candidate_joint_zones_unchanged':True,'mechanical_acceptance':False,'containment_accepted':False,'training_accepted':False},geometries
