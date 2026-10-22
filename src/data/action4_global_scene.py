"""Integrated pending-fit scene: reuse frozen finger rig, source photos and G1B."""
import json,math,hashlib
from pathlib import Path
from data import action4_global_environment as env
ROOT=env.ROOT
PREFIX='integrated_'
TEXT='Action4_Integrated_Initialization.json'
ASSETS=ROOT/'data/action4_scene_assets/global_scene_init_v001'
OUTPUTS=ROOT/'outputs/action4_realism_reassessment_v001/global_scene_init_v001'
SIDES=('left','right')
SOURCE_KEYS={'finger_asset','finger_contract','environment_config','environment_asset','environment_module','finger_module','rig_module','synthesis_module','foundation_module','render_module','photo_manifest','skin_forearm','wristband_weave','canonical_manifest'}
PRIOR_KEYS={'claim','skin_linear_rgb','skin_photo_tile_m','skin_photo_strength','brace_linear_rgb','brace_photo_tile_m','green_linear_rgb','green_edge_radius_m','white_linear_rgb','white_module_dimensions_m','white_module_surface_gap_m','black_linear_rgb','ball_radius_m','link_radius_m','quick_release_dimensions_m','usb_dimensions_m','usb_cad_yz_m','cable_radius_m','cable_proximal_offsets_m','collision_claim'}


def validate_config(cfg):
    base_keys={'schema','status','dataset_generation_allowed','sources','initial_priors'}
    if cfg.get('schema') not in ('action4.global_scene_init.v1','action4.global_scene_init.v2'):
        raise ValueError('Supported integrated config schema required')
    expected=base_keys|({'derived_skin'} if cfg['schema']=='action4.global_scene_init.v2' and 'derived_skin' in cfg else set())
    if set(cfg)!=expected or cfg['status']!='initialization_pending_fit' or cfg['dataset_generation_allowed'] is not False:
        raise ValueError('Initialization-only integrated config required')
    if set(cfg['sources'])!=SOURCE_KEYS or set(cfg['initial_priors'])!=PRIOR_KEYS:raise ValueError('Unexpected input or per-scene override')
    for ref in cfg['sources'].values():
        if set(ref)!={'path','sha256'} or env.sha(env.bounded(ref['path'],True))!=ref['sha256']:raise ValueError('Frozen integrated input changed')
    p=cfg['initial_priors']
    for key,val in p.items():
        if key in ('claim','collision_claim'):
            if not isinstance(val,str) or not val:raise ValueError('Explicit prior claim required')
            continue
        values=val if isinstance(val,list) else [val]
        if any(type(x) not in (int,float) or not math.isfinite(x) or abs(x)>3.4028234663852886e38 for x in values):raise ValueError('Finite numeric prior required')
        if key.endswith('_rgb') and (len(values)!=3 or any(not 0<=x<=1 for x in values)):raise ValueError('RGB prior invalid')
        elif key.endswith('dimensions_m') and (len(values)!=3 or min(values)<=0):raise ValueError('Metric dimensions invalid')
        elif key=='usb_cad_yz_m' and len(values)!=2:raise ValueError('CAD y/z pair required')
        elif key not in ('usb_cad_yz_m',) and not key.endswith('_rgb') and min(values)<=0:raise ValueError('Positive design dimensions required')
    if not 0<p['skin_photo_strength']<=1:raise ValueError('Texture mix invalid')
    if len(p['cable_proximal_offsets_m'])!=4 or p['cable_proximal_offsets_m']!=sorted(p['cable_proximal_offsets_m']):raise ValueError('Four ordered proximal cable stations required')
    photo=json.loads(env.bounded(cfg['sources']['photo_manifest']['path'],True).read_text())
    for key in ('skin_forearm','wristband_weave'):
        if any(photo['patches'][key][x]!=cfg['sources'][key][x] for x in ('path','sha256')):raise ValueError('Photo manifest/input mismatch')
    finger=json.loads(env.bounded(cfg['sources']['finger_contract']['path'],True).read_text())
    if finger['saved_asset']!=cfg['sources']['finger_asset'] or finger['bone_count']!=35:raise ValueError('Reviewed neutral finger input required')
    ecfg=json.loads(env.bounded(cfg['sources']['environment_config']['path'],True).read_text())
    ancestor=json.loads(env.bounded(ecfg['sources']['rig_contract']['path'],True).read_text())
    if cfg['sources']['canonical_manifest']!=ancestor['source_camera']['asset_manifest']:
        raise ValueError('Canonical label manifest must be the original G1 camera source')
    canonical_definition(cfg)
    skin_input_spec(cfg)
    return cfg


def _verified_ref(ref):
    if not isinstance(ref,dict) or set(ref)!={'path','sha256'}:
        raise ValueError('Exact path/SHA reference required')
    path=env.bounded(ref['path'],True)
    if env.sha(path)!=ref['sha256']:raise ValueError('Derived skin provenance bytes changed: '+ref['path'])
    return path


def skin_input_spec(cfg):
    """v1/raw stays sRGB; only an explicit v2 derived binding permits Linear EXR."""
    original=cfg['sources']['skin_forearm']
    raw={'mode':'original_photo','source_photo':original,'active_image':original,'color_space':'sRGB'}
    if 'derived_skin' not in cfg:return raw
    d=cfg['derived_skin']
    if cfg['schema']!='action4.global_scene_init.v2' or not isinstance(d,dict) or set(d)!={'manifest','image','color_space'} or d['color_space']!='Linear':
        raise ValueError('Derived skin requires explicit v2 manifest/image/Linear binding')
    path=_verified_ref(d['image']);manifest_path=_verified_ref(d['manifest'])
    if not path.is_relative_to(ROOT/'data/action4_scene_assets/skin_texture_periodic_v001') or not manifest_path.is_relative_to(path.parent):
        raise ValueError('Derived skin must use its immutable data asset directory')
    manifest=json.loads(manifest_path.read_text())
    if manifest.get('schema')!='action4.derived_skin_texture.v1' or manifest.get('material_accepted') is not False or manifest.get('production_generation_allowed') is not False:
        raise ValueError('Numerically reviewed candidate scope required, not production acceptance')
    image=manifest['image']
    if {k:image[k] for k in ('path','sha256')}!=d['image'] or image['color_space_required']!='Linear' or image['format']!='OPEN_EXR' or image['channel_type']!='FLOAT32' or image['compression']!='ZIP' or image['resolution_wh']!=[160,160]:
        raise ValueError('Derived image/precision/colour-space manifest mismatch')
    if manifest['original_source_png']!=original or manifest['original_source_photo_manifest']!=cfg['sources']['photo_manifest']:
        raise ValueError('Derived texture must retain original photo lineage')
    for ref in manifest['derivation'].values():
        if isinstance(ref,dict) and set(ref)=={'path','sha256'}:_verified_ref(ref)
    for ref in manifest['encoding_evidence'].values():_verified_ref(ref)
    review=manifest['limited_reviews'];spec=json.loads(_verified_ref(review['spec']).read_text());quality=json.loads(_verified_ref(review['root_quality']).read_text())
    if spec['status']!='PASS_NUMERIC_EXR_SPEC_ONLY' or spec['reviewed_EXR_sha256']!=d['image']['sha256'] or quality['status']!='PASS_NUMERICAL_LINEAR_TEXTURE_CANDIDATE_QUALITY_ONLY' or quality['spec_sha256']!=review['spec']['sha256']:
        raise ValueError('Derived skin lacks matching limited numerical reviews')
    return {'mode':'derived_periodic_source_photo','source_photo':original,'active_image':d['image'],
            'color_space':'Linear','derivation_manifest':d['manifest'],
            'float_reference':manifest['derivation']['float64_decomposition_NPZ'],
            'claim':'Source-photo periodicization only; not albedo/BRDF or generated skin detail'}


def _replace_derived_skin_image(material,cfg):
    import bpy,hashlib
    spec=skin_input_spec(cfg)
    if spec['mode']=='original_photo':return
    original=[node for node in material.node_tree.nodes if node.type=='TEX_IMAGE' and node.image and node.image.packed_file and hashlib.sha256(bytes(node.image.packed_file.data)).hexdigest()==spec['source_photo']['sha256']]
    if len(original)!=1:raise ValueError('Unique original skin image node required before replacement')
    node=original[0];old_image=node.image
    # Retain the original packed PNG as inactive provenance; it is not active-input evidence.
    old_image.use_fake_user=True
    derived=bpy.data.images.load(str(_verified_ref(spec['active_image'])),check_existing=False)
    derived.colorspace_settings.name='Linear';derived.alpha_mode=old_image.alpha_mode;derived.pack()
    if not derived.is_float or derived.colorspace_settings.name!='Linear' or hashlib.sha256(bytes(derived.packed_file.data)).hexdigest()!=spec['active_image']['sha256']:
        raise ValueError('Derived active image is not the exact packed Linear EXR')
    node.image=derived
    material['active_skin_texture_source']=spec['active_image']['path']
    material['active_skin_texture_color_space']='Linear'
    material['derived_skin_manifest']=spec['derivation_manifest']['path']
    material['source_photo_provenance']=spec['source_photo']['path']
    material['derived_skin_claim']=spec['claim']


def skin_input_state(scene,cfg):
    import bpy,numpy as np,hashlib
    spec=skin_input_spec(cfg);body=scene.objects['global_source_anatomy']
    if len(body.data.materials)!=1:raise ValueError('One body skin material required')
    material=body.data.materials[0];tree=material.node_tree;shader=tree.nodes.get('Principled BSDF')
    if not material.use_nodes or shader is None or len(shader.inputs['Base Color'].links)!=1:raise ValueError('Original skin colour route missing')
    mix=shader.inputs['Base Color'].links[0].from_node
    if mix.type!='MIX_RGB' or mix.blend_type!='MIX' or mix.inputs[0].is_linked or abs(mix.inputs[0].default_value-cfg['initial_priors']['skin_photo_strength'])>1e-7 or len(mix.inputs[2].links)!=1:
        raise ValueError('Original skin mix or active image connection changed')
    link=mix.inputs[2].links[0];node=link.from_node
    if node.type!='TEX_IMAGE' or link.from_socket!=node.outputs['Color'] or node.image is None:
        raise ValueError('Skin Color2 must use the explicitly bound image Color output')
    image=node.image
    if image.colorspace_settings.name!=spec['color_space']:raise ValueError('Active skin image colour space mismatch')
    if image.packed_file is None or hashlib.sha256(bytes(image.packed_file.data)).hexdigest()!=spec['active_image']['sha256']:
        raise ValueError('Active packed skin image SHA mismatch')
    if Path(bpy.path.abspath(image.filepath)).resolve()!=_verified_ref(spec['active_image']):
        raise ValueError('Active image path differs from declared source')
    if list(image.size)!=[160,160] or node.projection!='BOX' or abs(node.projection_blend-.25)>1e-7 or node.extension!='REPEAT' or len(node.inputs['Vector'].links)!=1:
        raise ValueError('Original skin mapping changed')
    scale=node.inputs['Vector'].links[0].from_node
    if scale.type!='VECT_MATH' or scale.operation!='SCALE' or len(scale.inputs[0].links)!=1 or scale.inputs[0].links[0].from_node.type!='TEX_COORD' or scale.inputs[0].links[0].from_socket.name!='Object' or abs(scale.inputs[3].default_value-1/cfg['initial_priors']['skin_photo_tile_m'])>2e-6:
        raise ValueError('Original Object/tile coordinate route changed')
    if spec['mode']!='original_photo':
        if not image.is_float:raise ValueError('Derived image must retain float precision')
        reference=np.load(_verified_ref(spec['float_reference']))['periodic_linear'].astype(np.float32)
        pixels=np.empty(len(image.pixels),np.float32);image.pixels.foreach_get(pixels)
        actual=pixels.reshape(160,160,-1)[::-1,:,:3]
        if not np.array_equal(actual,reference):raise ValueError('Active Linear EXR pixels changed or were colour-transformed')
    return {**spec,'material':material.name,'image_node':node.name,'mix_node':mix.name,
            'packed_sha256':hashlib.sha256(bytes(image.packed_file.data)).hexdigest(),'actual_color_space':image.colorspace_settings.name,
            'actual_is_float':image.is_float,'resolution_wh':list(image.size),'mix_factor_actual':float(mix.inputs[0].default_value),
            'Object_scale_actual':float(scale.inputs[3].default_value),'projection':node.projection,'projection_blend':float(node.projection_blend),
            'extension':node.extension,'interpolation':node.interpolation}


def verify_skin_input(scene,contract):
    actual=skin_input_state(scene,contract['config'])
    declared=contract.get('skin_color_input')
    if declared is None:
        if contract['config']['schema']!='action4.global_scene_init.v1' or 'derived_skin' in contract['config']:
            raise ValueError('New derived skin requires an explicit active input contract')
    elif actual!=declared:raise ValueError('Active skin input differs from saved contract')
    return actual


def canonical_definition(cfg):
    ref=cfg['sources']['canonical_manifest'];path=env.bounded(ref['path'],True)
    if env.sha(path)!=ref['sha256']:raise ValueError('Canonical source bytes changed')
    manifest=json.loads(path.read_text())
    points=manifest['corners_3d_mm'];order=manifest['corner_order']
    if manifest['units']!='mm' or len(points)!=8 or len(order)!=8 or len(set(order))!=8 or any(len(p)!=3 or any(type(x) not in (int,float) or not math.isfinite(x) for x in p) for p in points):
        raise ValueError('Invalid original ordered canonical definition')
    return {'source_manifest':dict(ref),'coordinate_frame':'Action4 CAD local',
            'corner_order':order,'corners_3d_mm':points,
            'corners_3d_m':[[x*.001 for x in p] for p in points],
            'source_units':'mm','scene_units':'m','conversion':'metres = millimetres * 0.001; order unchanged'}


def verify_canonical_definition(cfg,definition):
    if definition!=canonical_definition(cfg):
        raise ValueError('Ordered canonical8 differs from original manifest; permutations forbidden')
    return True


def _geometry(obj):
    return env.digest({'vertices':[list(v.co) for v in obj.data.vertices],
                       'faces':[list(p.vertices) for p in obj.data.polygons],
                       'uv':{u.name:[list(x.uv) for x in u.data] for u in obj.data.uv_layers}})


def _original_invariants(scene, names):
    """Explicit geometry, weights, bone paths and local attachments, excluding appearance."""
    body=scene.objects['global_source_anatomy'];rig=scene.objects['global_metric_armature']
    return {'objects':{name:{'type':scene.objects[name].type,'parent':scene.objects[name].parent.name if scene.objects[name].parent else None,
        'parent_type':scene.objects[name].parent_type,'parent_bone':scene.objects[name].parent_bone,
        'basis':[list(r) for r in scene.objects[name].matrix_basis],'parent_inverse':[list(r) for r in scene.objects[name].matrix_parent_inverse],
        'scale':list(scene.objects[name].scale),'mesh':_geometry(scene.objects[name]) if scene.objects[name].type=='MESH' else None}
        for name in sorted(names)},
        'bones':{b.name:{'parent':b.parent.name if b.parent else None,'matrix_local':[list(r) for r in b.matrix_local],'length':b.length,'inherit_scale':b.inherit_scale} for b in rig.data.bones},
        'weights':env.digest({'groups':[g.name for g in body.vertex_groups],'weights':[(v.index,g.group,g.weight) for v in body.data.vertices for g in v.groups]}),
        'body_modifiers':[(m.name,m.type,getattr(m,'levels',None),getattr(m,'render_levels',None),m.show_viewport,m.show_render,getattr(getattr(m,'object',None),'name',None)) for m in body.modifiers],
        'fixed_A':{side:[list(r) for r in scene.objects[side+'_Action4_core'].matrix_basis] for side in SIDES},
        'diagnostic_envelope_purpose':'Unordered geometric extent invariant only; NOT training labels',
        'diagnostic_unordered_envelope':{side:[[x,y,z] for x in (min(v.co.x for v in scene.objects[side+'_Action4_core'].data.vertices),max(v.co.x for v in scene.objects[side+'_Action4_core'].data.vertices)) for y in (min(v.co.y for v in scene.objects[side+'_Action4_core'].data.vertices),max(v.co.y for v in scene.objects[side+'_Action4_core'].data.vertices)) for z in (min(v.co.z for v in scene.objects[side+'_Action4_core'].data.vertices),max(v.co.z for v in scene.objects[side+'_Action4_core'].data.vertices))] for side in SIDES}}



def _mesh_shading_state(mesh):
    smooth = sum(poly.use_smooth for poly in mesh.polygons)
    return {'vertices': len(mesh.vertices), 'polygons': len(mesh.polygons),
            'loops': len(mesh.loops), 'smooth_polygons': smooth,
            'flat_polygons': len(mesh.polygons) - smooth,
            'use_auto_smooth': mesh.use_auto_smooth,
            'auto_smooth_angle_rad': float(mesh.auto_smooth_angle),
            'has_custom_normals': mesh.has_custom_normals,
            'edges': len(mesh.edges), 'sharp_edges': sum(edge.use_edge_sharp for edge in mesh.edges)}


def body_shading_state(scene):
    """Actual control and evaluated shading state, separate from geometry hashes."""
    import bpy
    body = scene.objects['global_source_anatomy']
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = body.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        return {'control': _mesh_shading_state(body.data),
                'evaluated': _mesh_shading_state(mesh), 'depsgraph_mode': depsgraph.mode}
    finally:
        evaluated.to_mesh_clear()


def validate_body_shading_contract(shading):
    """Only all-body smooth flags may differ from the frozen source appearance."""
    if not isinstance(shading, dict) or set(shading) != {'schema', 'object', 'policy', 'source', 'current'}:
        raise ValueError('Explicit source/current body shading contract required')
    if shading['schema'] != 'action4.body_shading.v1' or shading['object'] != 'global_source_anatomy' or shading['policy'] != 'all_body_polygons_smooth_shading_only':
        raise ValueError('Unexpected body shading policy')
    source, current = shading['source'], shading['current']
    if set(source) != {'control', 'evaluated', 'depsgraph_mode'} or set(current) != set(source) or source['depsgraph_mode'] != current['depsgraph_mode']:
        raise ValueError('Body shading evaluation scope changed')
    for label in ('control', 'evaluated'):
        old, new = source[label], current[label]
        for row in (old, new):
            if any(type(row[k]) is not int or row[k] < 0 for k in ('polygons', 'smooth_polygons', 'flat_polygons')) or row['polygons'] <= 0 or row['smooth_polygons'] + row['flat_polygons'] != row['polygons']:
                raise ValueError('Invalid body shading polygon counts')
        if new['smooth_polygons'] != new['polygons'] or new['flat_polygons'] != 0:
            raise ValueError('All control and evaluated body polygons must be smooth')
        omit = {'smooth_polygons', 'flat_polygons'}
        if {k:v for k,v in old.items() if k not in omit} != {k:v for k,v in new.items() if k not in omit}:
            raise ValueError('Shading-only change altered geometry or auto/custom/sharp normal state')
    return True


def verify_body_shading(scene, contract):
    shading = contract.get('body_shading')
    validate_body_shading_contract(shading)
    if body_shading_state(scene) != shading['current']:
        raise ValueError('Actual control/evaluated body shading differs from explicit contract')
    return True


def _attachment_state(scene):
    return {o.name:{'parent':o.parent.name if o.parent else None,'basis':[list(r) for r in o.matrix_basis],
            'pass_index':o.pass_index,'owner':o.get('attachment_owner'),'source_frame':o.get('source_frame'),'claim':o.get('claim')}
            for o in scene.objects if o.name.startswith(PREFIX)}


def contributing_objects(scene):
    """Shared integration consumer list for rendering/collision: excludes hidden scaffolds."""
    return [o for o in scene.objects if o.type=='MESH' and not o.hide_render and not o.get('non_render_scaffold',False)]


def _assemble(scene,cfg):
    import bpy
    from mathutils import Matrix,Vector
    from data.action4_synthesis import _skin,_rounded_box
    from data.action4_blender_foundation import apply_photo_material
    from data import action4_render as base
    p=cfg['initial_priors'];before=set(scene.objects.keys())
    skin=_skin(PREFIX+'skin',p['skin_linear_rgb'])
    apply_photo_material(skin,cfg['sources']['skin_forearm']['path'],p['skin_photo_tile_m'],strength=p['skin_photo_strength'])
    _replace_derived_skin_image(skin,cfg)
    body = scene.objects['global_source_anatomy']
    body.data.materials.append(skin)
    # Appearance-only edit in this derived scene; source topology/weights stay intact.
    for polygon in body.data.polygons:
        polygon.use_smooth = True
    body.data.update()
    bpy.context.view_layer.update()
    gray=base._material(PREFIX+'brace_weave',p['brace_linear_rgb'],.87)
    apply_photo_material(gray,cfg['sources']['wristband_weave']['path'],p['brace_photo_tile_m'],strength=1)
    green=base._material(PREFIX+'green_edge',p['green_linear_rgb'],.80)
    pale=base._material(PREFIX+'white_module',p['white_linear_rgb'],.65)
    black=base._material(PREFIX+'black_mount',p['black_linear_rgb'],.62)
    ecfg=json.loads(env.bounded(cfg['sources']['environment_config']['path'],True).read_text())
    rigcontract=json.loads(env.bounded(ecfg['sources']['rig_contract']['path'],True).read_text())
    assembly={};converted=[]
    for side in SIDES:
        brace=scene.objects[side+'_brace_frame'];core=scene.objects[side+'_Action4_core'];band=scene.objects[side+'_brace_initial_geometry']
        band.data.materials.clear();band.data.materials.append(gray)
        scaffold=scene.objects[side+'_mount_initial_support'];scaffold.hide_render=True;scaffold.hide_viewport=True
        scaffold['non_render_scaffold']=True;scaffold['excluded_from_integration_collision']=True
        for key in ('visible_camera','visible_diffuse','visible_glossy','visible_transmission','visible_volume_scatter','visible_shadow'):setattr(scaffold,key,False)
        root=bpy.data.objects.new(PREFIX+side+'_brace_accessories',None);scene.collection.objects.link(root);root.parent=brace;root.matrix_parent_inverse=Matrix.Identity(4)
        cad=bpy.data.objects.new(PREFIX+side+'_cad_accessories',None);scene.collection.objects.link(cad);cad.parent=brace;cad.matrix_parent_inverse=Matrix.Identity(4);cad.matrix_basis=core.matrix_basis.copy()
        rx,rz=rigcontract['fixed_brace_and_camera_attachment'][side]['brace_inner_radii_m']
        halfwidth=max(abs(v.co.y) for v in band.data.vertices);thickness=band.modifiers['metric_brace_thickness'].thickness
        outer=rz+thickness
        for sign in (-1,1):
            points=[((rx+thickness)*math.cos(i*math.pi/24),sign*halfwidth,(rz+thickness)*math.sin(i*math.pi/24)) for i in range(49)]
            base._tube(PREFIX+side+'_green_edge',points,p['green_edge_radius_m'],green,root)
        dims=p['white_module_dimensions_m'];white_z=-(outer+p['white_module_surface_gap_m']+dims[2]/2)
        _rounded_box(PREFIX+side+'_white_module',(0,0,white_z),dims,pale,root,.004)
        # Physical mount is the frozen CAD -Z bottom, never the CAD -Y lens face.
        q=p['quick_release_dimensions_m'];cad_z_min=min(v.co.z for v in core.data.vertices)
        quick_cad=Vector((0,0,cad_z_min-q[2]/2))
        quick_brace=core.matrix_basis@quick_cad
        _rounded_box(PREFIX+side+'_quick_release',quick_cad,q,black,cad,.0015)
        # Route below the camera then along its bottom side under the fixed A.
        # Constant design priors, not a fitted support or calibrated joint.
        low_z=outer+p['link_radius_m']+.001
        link_points=[(0,0,outer),(0,0,low_z),(0,quick_brace.y,low_z),tuple(quick_brace)]
        midpoint=low_z
        base._ellipsoid(PREFIX+side+'_ball',(0,0,midpoint),(p['ball_radius_m'],)*3,black,root)
        base._tube(PREFIX+side+'_mount_link',link_points,p['link_radius_m'],black,root)
        u=p['usb_dimensions_m'];xmax=max(v.co.x for v in core.data.vertices);uy,uz=p['usb_cad_yz_m']
        connector=Vector((xmax+u[0]/2,uy,uz))
        _rounded_box(PREFIX+side+'_usb_connector',connector,u,black,cad,.001)
        start=core.matrix_basis@Vector((connector.x,uy,uz-u[2]/2))
        # Source cable thickness/curved helper reused; stations re-expressed in new brace.
        points=[tuple(start)]
        points += [(rx+.012,-distance,outer+.006) for distance in p['cable_proximal_offsets_m']]
        base._tube(PREFIX+side+'_cable',points,p['cable_radius_m'],black,root)
        assembly[side]={'brace_inner_radii_m':[rx,rz],'brace_halfwidth_m':halfwidth,
            'camera_radial_side':'+Z','white_module_radial_side':'-Z','white_center_brace_m':[0,0,white_z],
            'physical_mount_face':'CAD -Z','quick_release_center_CAD_m':list(quick_cad),'mount_link_brace_m':link_points,'ball_center_brace_m':[0,0,midpoint],
            'USB_center_CAD_m':list(connector),'cable_points_brace_m':points,
            'claim':'Rebased unmeasured priors; no collision-free or reference-fit claim',
            'hidden_proxy':scaffold.name}
    bpy.context.view_layer.update()
    new=[o for o in scene.objects if o.name not in before]
    # Bake only new accessory modifiers/curves into derived meshes; never touch source body/core.
    deps=bpy.context.evaluated_depsgraph_get()
    for obj in new:
        if obj.type in ('MESH','CURVE'):
            evaluated=obj.evaluated_get(deps);mesh=bpy.data.meshes.new_from_object(evaluated,depsgraph=deps)
            old_type=obj.type
            if old_type=='CURVE':
                replacement=bpy.data.objects.new(obj.name+'_baked',mesh);scene.collection.objects.link(replacement)
                replacement.parent=obj.parent;replacement.matrix_parent_inverse=obj.matrix_parent_inverse.copy();replacement.matrix_basis=obj.matrix_basis.copy()
                name=obj.name;bpy.data.objects.remove(obj,do_unlink=True);replacement.name=name;obj=replacement
            else:
                obj.data=mesh;obj.modifiers.clear()
            converted.append({'object':obj.name,'from':old_type,'to':'MESH'})
    for obj in scene.objects:
        if obj.name.startswith(PREFIX):
            obj.pass_index=0;obj['attachment_owner']='left' if obj.name.startswith(PREFIX+'left_') else 'right'
            obj['source_frame']='CAD' if 'cad_accessories' in obj.name or (obj.parent and 'cad_accessories' in obj.parent.name) else 'brace'
            obj['claim']='unmeasured_accessory_initial_prior_non_target'
    bpy.context.view_layer.update()
    from mathutils.bvhtree import BVHTree
    bounds={}
    for obj in scene.objects:
        if obj.name.startswith(PREFIX) and obj.type=='MESH':bounds[obj.name]=env._bounds(obj)
    for side in SIDES:
        core=scene.objects[side+'_Action4_core']
        # Compare in world coordinates; FromObject gives local coordinates.
        def world_tree(obj):
            return BVHTree.FromPolygons([obj.matrix_world@v.co for v in obj.data.vertices],[list(p.vertices) for p in obj.data.polygons])
        target=world_tree(core)
        assembly[side]['mount_body_intersections']={}
        for suffix in ('_ball','_mount_link'):
            obj=scene.objects[PREFIX+side+suffix];n=len(target.overlap(world_tree(obj)))
            assembly[side]['mount_body_intersections'][obj.name]=n
            if n:raise ValueError('Mount routing intersects frozen camera body: '+obj.name)
    return {'physical_bounds_world_m':bounds,'assembly_priors':assembly,'derived_mesh_conversions':converted,'new_objects':sorted(o.name for o in scene.objects if o.name not in before)}


def verify(scene,contract):
    import numpy as np
    cfg=validate_config(contract['config'])
    verify_canonical_definition(cfg,contract['canonical8'])
    for side in SIDES:
        brace=scene.objects[side+'_brace_frame'];core=scene.objects[side+'_Action4_core']
        actual=np.asarray(brace.matrix_world.inverted()@core.matrix_world)
        if not np.allclose(actual,contract['original_invariants']['fixed_A'][side],rtol=0,atol=2e-6) or not np.allclose(core.scale,(1,1,1),rtol=0,atol=1e-7):raise ValueError('Actual fixed A or target scale changed')
    if scene.get('integrated_generation_allowed') is not False:raise ValueError('Integrated scope changed')
    if env.digest(_original_invariants(scene,contract['original_names']))!=contract['original_invariants_sha256']:raise ValueError('Original geometry/bones/weights/fixed attachment changed')
    if _attachment_state(scene)!=contract['attachment_state']:raise ValueError('Accessory ownership or local attachment changed')
    for side in SIDES:
        scaffold=scene.objects[side+'_mount_initial_support']
        if not scaffold.hide_render or not scaffold.hide_viewport or not scaffold.get('non_render_scaffold') or scaffold in contributing_objects(scene):raise ValueError('Proxy scaffold contributed')
    verify_skin_input(scene,contract)
    env.verify_environment(scene,contract['environment'])
    verify_body_shading(scene,contract)
    for key in ('skin_forearm','wristband_weave'):
        found=[im for im in __import__('bpy').data.images if im.packed_file and hashlib.sha256(bytes(im.packed_file.data)).hexdigest()==cfg['sources'][key]['sha256']]
        if not found:raise ValueError('Source photo not packed')
    return True


def build(config_path,output_dir,report_dir):
    import bpy
    cfg=validate_config(json.loads(env.bounded(config_path,True).read_text()))
    out,report=env.bounded(output_dir),env.bounded(report_dir)
    if not out.is_relative_to(ASSETS) or not report.is_relative_to(OUTPUTS):raise ValueError('Assigned integration directories required')
    if out.exists() or report.exists():raise FileExistsError('Use fresh version; no overwrite')
    bpy.ops.wm.open_mainfile(filepath=str(env.bounded(cfg['sources']['finger_asset']['path'],True)),load_ui=False,use_scripts=False)
    scene=bpy.context.scene;bpy.context.view_layer.update()
    if len(scene.objects['global_metric_armature'].data.bones)!=35 or scene.world or scene.camera or TEXT in bpy.data.texts:raise ValueError('Neutral reviewed finger scene required')
    if any(o.name.startswith(PREFIX) for o in scene.objects):raise ValueError('Integrated namespace already occupied')
    names=sorted(scene.objects.keys());original=_original_invariants(scene,names)
    source_shading = body_shading_state(scene)
    appearance_before={name:env.object_record(scene.objects[name]) for name in names}
    added=_assemble(scene,cfg)
    shading = {'schema':'action4.body_shading.v1','object':'global_source_anatomy',
               'policy':'all_body_polygons_smooth_shading_only',
               'source':source_shading,'current':body_shading_state(scene)}
    validate_body_shading_contract(shading)
    if original!=_original_invariants(scene,names):raise ValueError('Assembly altered protected original rig geometry')
    allowed={'global_source_anatomy',*(s+'_brace_initial_geometry' for s in SIDES),*(s+'_mount_initial_support' for s in SIDES)}
    changed=[name for name in names if appearance_before[name]!=env.object_record(scene.objects[name])]
    if set(changed)!=allowed:raise ValueError('Unexpected appearance changes: '+str(changed))
    ecfg=json.loads(env.bounded(cfg['sources']['environment_config']['path'],True).read_text())
    environment=env.add_environment(scene,ecfg)
    scene['integrated_generation_allowed']=False;scene['integrated_status']='initialization_pending_fit'
    contract={'schema':'action4.integrated_scene_contract.v1','status':'initialization_pending_fit','dataset_generation_allowed':False,
        'canonical8':canonical_definition(cfg),'config':cfg,'actual_loaded_input':cfg['sources']['finger_asset'],'environment_G1_ref_role':'ancestral control-mesh verification only, not actual loaded asset',
        'body_shading':shading,'skin_color_input':skin_input_state(scene,cfg),
        'original_names':names,'original_invariants':original,'original_invariants_sha256':env.digest(original),
        'intentional_original_changes':{'body_and_brace_materials':['global_source_anatomy',*(s+'_brace_initial_geometry' for s in SIDES)],'hidden_non_render_scaffolds':[s+'_mount_initial_support' for s in SIDES],
            'body_polygon_shading':{'object':'global_source_anatomy','policy':'all_body_polygons_smooth_shading_only'},
            'derived_skin_color_input':skin_input_spec(cfg) if 'derived_skin' in cfg else None},
        'assembly':added,'attachment_state':_attachment_state(scene),'environment':environment,
        'source_module_sha256':env.sha(Path(__file__)),'limits':['Unmeasured initialization; no production or calibrated sampling ranges','Skin photo contains capture illumination/compression, not albedo/BRDF','Proxy supports are hidden and excluded by contributing_objects API; downstream collision consumers must use this list','No full collision-free, anatomical fit, grasp/contact or rendered-realism acceptance']}
    verify(scene,contract)
    for pose in ecfg['engineering_capture_poses']:env.apply_capture_pose(scene,pose,environment);verify(scene,contract)
    c=ecfg['initial_priors']['capture'];env.apply_capture_pose(scene,{'location_m':c['initial_location_m'],'target_m':c['initial_target_m'],'roll_deg':c['initial_roll_deg']},environment)
    bpy.data.texts.new(TEXT).write(json.dumps(contract,indent=2,allow_nan=False))
    out.mkdir(parents=True);report.mkdir(parents=True)
    asset=out/'initial.blend';bpy.ops.wm.save_as_mainfile(filepath=str(asset),check_existing=False)
    contract['saved_asset']={'path':str(asset.relative_to(ROOT)),'sha256':env.sha(asset)}
    (out/'integrated_contract.json').write_text(json.dumps(contract,indent=2,allow_nan=False))
    (report/'build_report.json').write_text(json.dumps({'status':'BUILT_PENDING_REOPEN','asset':contract['saved_asset'],'intentional_changes':contract['intentional_original_changes'],'new_accessories':len(added['new_objects']),'dataset_generation_allowed':False},indent=2))
    validate_config(cfg)
    return contract['saved_asset']


def reopen(asset,contract_path,report_path):
    import bpy
    c=json.loads(env.bounded(contract_path,True).read_text());p=env.bounded(asset,True);out=env.bounded(report_path)
    if not out.is_relative_to(OUTPUTS) or out.exists():raise ValueError('Fresh assigned evidence path required')
    if env.sha(p)!=c['saved_asset']['sha256'] or env.sha(Path(__file__))!=c['source_module_sha256']:raise ValueError('Integrated asset/source bytes changed')
    bpy.ops.wm.open_mainfile(filepath=str(p),load_ui=False,use_scripts=False);scene=bpy.context.scene;bpy.context.view_layer.update();verify(scene,c)
    for pose in c['environment']['config']['engineering_capture_poses']:env.apply_capture_pose(scene,pose,c['environment']);verify(scene,c)
    result={'status':'PASS_INTEGRATED_INITIALIZATION_REOPEN','asset':c['saved_asset'],'bone_count':len(scene.objects['global_metric_armature'].data.bones),'dataset_generation_allowed':False,'capture_pose_cases':2,'render_calls':0,'gpu_used':False,'new_accessory_count':len(c['assembly']['new_objects']),'original_input_geometry_weights_bones_fixed_A_preserved':True,'body_shading':c['body_shading']['current']}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2));return result
