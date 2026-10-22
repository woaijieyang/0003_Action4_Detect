"""Shared G1B world initialization. No render, sampling, fitting, or acceptance.

add_environment adds only new environment objects to a neutral G1 scene.
apply_capture_pose accepts extrinsics only; the saved global world and K are
verified before and after every change. Importing this module performs no IO.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path

ROOT = Path('/data/zy/vln/a-test')
SCHEMA = 'action4.global_environment_init.v1'
STATUS = 'initial_prior_fit_pending'
PREFIX = 'g1b_env_'
TEXT = 'Action4_Global_Environment_Contract.json'
TOP = {'schema_version','status','dataset_generation_allowed','units','world_frame',
       'sources','native_image','initial_priors','engineering_capture_poses'}
GROUPS = {
 'geometry': set('table_size_xy_m table_center_xy_m table_top_z_m table_thickness_m table_leg_width_m table_leg_inset_m floor_size_xy_m floor_center_xy_m floor_top_z_m floor_thickness_m wall_center_y_m wall_width_m wall_height_m wall_thickness_m body_table_clearance_m table_wall_clearance_m'.split()),
 'garment': set('rest_outline_size_xy_m phase_radians support_clearance_m orientation_z_deg source_nominal_outline_xy_m source_outline_center_xy_m claim'.split()),
 'materials': set('wall_linear_rgb floor_linear_rgb table_leg_linear_rgb roughness grid_cell_linear_rgb grid_line_linear_rgb grid_pitch_m grid_line_fraction claim'.split()),
 'lighting': set('area_location_m area_target_m area_energy_w area_size_m area_color_linear_rgb world_linear_rgb world_strength exposure_ev gamma view_transform look claim'.split()),
 'capture': set('focal_mm sensor_width_mm clip_start_m clip_end_m initial_location_m initial_target_m initial_roll_deg distortion_initial claim'.split())}
SOURCES = {'architecture','rig_asset','rig_contract','synthesis_helper','render_helper','material_helper','denim_manifest'}


def bounded(value, exists=False):
    # Check lexical path before resolving any link.
    import os
    p = Path(os.path.abspath(str(ROOT / value)))
    if not p.is_relative_to(ROOT):
        raise ValueError('Environment path leaves project')
    p = p.resolve(strict=exists)
    if not p.is_relative_to(ROOT):
        raise ValueError('Environment symlink leaves project')
    return p


def sha(path):
    h = hashlib.sha256()
    with bounded(path, True).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _scalar(x):
    if type(x) not in (int, float) or not math.isfinite(x) or abs(x)>3.4028234663852886e38:
        raise ValueError('Finite float32-representable engineering value required; not a calibrated range')
    return x


def _vector(x, n):
    if not isinstance(x, list) or len(x) != n:
        raise ValueError('Initial prior vector has wrong dimensions')
    return [_scalar(v) for v in x]


def validate_pose(pose):
    if not isinstance(pose, dict) or set(pose) != {'location_m','target_m','roll_deg'}:
        raise ValueError('Capture pose accepts extrinsics only; focal, size and scene overrides forbidden')
    a, b = _vector(pose['location_m'], 3), _vector(pose['target_m'], 3)
    _scalar(pose['roll_deg'])
    if sum((x-y)**2 for x,y in zip(a,b)) < 1e-12:
        raise ValueError('Capture location and target must differ')
    return pose


def validate_config(cfg, check_files=True):
    if not isinstance(cfg, dict) or set(cfg) != TOP or cfg.get('schema_version') != SCHEMA:
        raise ValueError('Exact shared global schema required; no family/per-scene overrides')
    if cfg['status'] != STATUS or cfg['dataset_generation_allowed'] is not False or cfg['units'] != 'metres':
        raise ValueError('Environment is an initial prior, never fitted or generation accepted')
    if cfg['world_frame'] != 'G1 source body front -Y, up +Z; no per-scenario world rescaling':
        raise ValueError('Global G1 coordinate convention changed')
    image = cfg['native_image']
    if image != {'resolution_wh':[3248,2464], 'evidence_status':'measured_input_video_dimensions', 'pixel_convention':'integer_centres'}:
        raise ValueError('Native input image contract changed')
    priors = cfg['initial_priors']
    if set(priors) != set(GROUPS) | {'claim'} or 'unmeasured initial prior' not in priors['claim']:
        raise ValueError('Every unmeasured value must remain inside explicit initial_priors')
    for group, keys in GROUPS.items():
        if not isinstance(priors[group], dict) or set(priors[group]) != keys:
            raise ValueError('Unknown global parameter or per-scene override: '+group)
        for key, value in priors[group].items():
            if key in ('claim','view_transform','look'):
                if not isinstance(value,str) or not value: raise ValueError('Missing parameter evidence label')
            elif isinstance(value,list):
                _vector(value, 2 if key.endswith('_xy_m') else 5 if key=='distortion_initial' else 3)
            else: _scalar(value)
    g, cloth, mat, light, camera = [priors[k] for k in ('geometry','garment','materials','lighting','capture')]
    for key in ('table_size_xy_m','floor_size_xy_m'):
        if min(g[key]) <= 0: raise ValueError('Positive fixed world extent required')
    for key in ('table_thickness_m','table_leg_width_m','table_leg_inset_m','floor_thickness_m','wall_width_m','wall_height_m','wall_thickness_m','body_table_clearance_m','table_wall_clearance_m'):
        if g[key] <= 0: raise ValueError('Positive shared geometry prior required')
    fx,fy=g['floor_center_xy_m'];fw,fd=g['floor_size_xy_m']
    if abs(g['table_center_xy_m'][0]-fx)+g['table_size_xy_m'][0]/2>fw/2 or abs(g['table_center_xy_m'][1]-fy)+g['table_size_xy_m'][1]/2>fd/2 or abs(g['wall_center_y_m']-fy)+g['wall_thickness_m']/2>fd/2 or abs(fx)+g['wall_width_m']/2>fw/2:
        raise ValueError('Shared floor must cover table and front wall')
    if g['table_top_z_m'] - g['table_thickness_m'] <= g['floor_top_z_m']:
        raise ValueError('Table support must stand above floor')
    if not g['table_leg_width_m']/2 <= g['table_leg_inset_m'] < min(g['table_size_xy_m'])/2:
        raise ValueError('Table legs must fit under tabletop')
    tx,ty = g['table_center_xy_m'];tw,td = g['table_size_xy_m']
    if ty+td/2 >= 0 or ty-td/2-(g['wall_center_y_m']+g['wall_thickness_m']/2) < g['table_wall_clearance_m']:
        raise ValueError('Table must be in front of body (-Y) with positive wall clearance')
    if cloth['source_nominal_outline_xy_m'] != [.9,.5] or cloth['source_outline_center_xy_m'] != [0,.15]:
        raise ValueError('Existing garment source geometry contract changed')
    if min(cloth['rest_outline_size_xy_m']) <= 0 or any(a>=b for a,b in zip(cloth['rest_outline_size_xy_m'],g['table_size_xy_m'])) or cloth['support_clearance_m'] <= 0:
        raise ValueError('Rest garment must fit on table with positive support clearance')
    if cloth['orientation_z_deg'] != 180:
        raise ValueError('Shared garment collar initially points away from body toward -Y')
    for key,value in mat.items():
        if key.endswith('_rgb') and any(not 0<=x<=1 for x in value):raise ValueError('Invalid linear material RGB')
    if not 0<mat['roughness']<=1 or mat['grid_pitch_m']<=0 or not 0<mat['grid_line_fraction']<.5:
        raise ValueError('Invalid fixed material/grid prior')
    if camera['focal_mm']<1 or camera['sensor_width_mm']<1 or not 0<camera['clip_start_m']<camera['clip_end_m'] or camera['distortion_initial'] != [0]*5:
        raise ValueError('Positive shared pinhole K required; distortion unsupported, not fitted')
    if light['area_energy_w']<=0 or light['area_size_m']<=0 or light['world_strength']<0 or light['gamma']<=0:
        raise ValueError('Invalid initial illumination')
    for key in ('area_color_linear_rgb','world_linear_rgb'):
        if any(x<0 for x in light[key]):raise ValueError('Negative illumination RGB')
    validate_pose({'location_m':camera['initial_location_m'],'target_m':camera['initial_target_m'],'roll_deg':camera['initial_roll_deg']})
    validate_pose({'location_m':light['area_location_m'],'target_m':light['area_target_m'],'roll_deg':0})
    if not isinstance(cfg['engineering_capture_poses'], list):raise ValueError('Engineering poses must be explicit list, not sampling ranges')
    for pose in cfg['engineering_capture_poses']:validate_pose(pose)
    if set(cfg['sources']) != SOURCES:raise ValueError('Missing immutable source references')
    if check_files:
        for ref in cfg['sources'].values():
            if set(ref) != {'path','sha256'} or sha(ref['path'])!=ref['sha256']:raise ValueError('Source input changed')
        contract=json.loads(bounded(cfg['sources']['rig_contract']['path'],True).read_text())
        if contract['asset'] != cfg['sources']['rig_asset'] or contract['dataset_generation_allowed'] is not False:
            raise ValueError('Use the validated G1 asset matching its pending-fit contract')
        denim=json.loads(bounded(cfg['sources']['denim_manifest']['path'],True).read_text())
        for spec in denim['files'].values():
            if sha(spec['path'])!=spec['sha256']:raise ValueError('Denim provenance bytes changed')
    import sys
    if 'bpy' in sys.modules:_blender_config_preflight(cfg)
    return cfg


def expected_K(cfg):
    c=cfg['initial_priors']['capture'];w,h=cfg['native_image']['resolution_wh'];f=c['focal_mm']/c['sensor_width_mm']*w
    return [[f,0,(w-1)/2],[0,f,(h-1)/2],[0,0,1]]


def _plain(value):
    if type(value) in (str,int,float,bool) or value is None:return value
    try:return [float(x) for x in value]
    except (TypeError,ValueError):return str(type(value))


SUPPORTED_SHADER_NODES = {
    'ShaderNodeBsdfPrincipled','ShaderNodeOutputMaterial','ShaderNodeTexImage',
    'ShaderNodeTexCoord','ShaderNodeVectorMath','ShaderNodeMixRGB','ShaderNodeTexNoise',
    'ShaderNodeValToRGB','ShaderNodeTexWave','ShaderNodeBump','ShaderNodeNormalMap',
    'ShaderNodeSeparateXYZ','ShaderNodeMath','ShaderNodeBackground','ShaderNodeOutputWorld'}


def _rna_state(value, *, node=False, depth=0):
    """Supported shader/light RNA values, not a generic Blender serializer.

    Socket defaults/links and packed image bytes are recorded separately. UI
    layout is excluded. Nested texture/color mapping, ramps and image-user
    state are included; unknown active pointer/collection types fail closed.
    """
    import bpy
    if depth>5: raise ValueError('Unsupported nested shader state')
    id_metadata={p.identifier for p in bpy.types.ID.bl_rna.properties}
    skip={'rna_type','id_data','inputs','outputs','internal_links'}
    if isinstance(value,bpy.types.ID):skip |= id_metadata
    if node:skip |= {'name','label','location','width','width_hidden','height','dimensions',
                     'select','show_options','show_preview','show_texture','hide','parent',
                     'use_custom_color','color','warning_propagation'}
    result={}
    for prop in value.bl_rna.properties:
        key=prop.identifier
        if key in skip or key.startswith('bl_'):continue
        item=getattr(value,key)
        if prop.type in {'BOOLEAN','INT','FLOAT','STRING','ENUM'}:
            result[key]=_plain(item)
        elif prop.type=='POINTER':
            if item is None:result[key]=None
            elif isinstance(item,bpy.types.ID):
                result[key]={'id_type':item.bl_rna.identifier,'name':item.name_full}
            elif key in {'color_ramp','texture_mapping','color_mapping','image_user'}:
                result[key]=_rna_state(item,depth=depth+1)
            else:raise ValueError('Unsupported active shader/light pointer: '+key)
        elif prop.type=='COLLECTION':
            if key=='elements':result[key]=[_rna_state(x,depth=depth+1) for x in item]
            elif len(item):raise ValueError('Unsupported active shader/light collection: '+key)
    return result


def _nodes(tree):
    if tree.animation_data is not None:
        raise ValueError('Animated shared shader state is unsupported by initialization contract')
    nodes=[]
    for node in tree.nodes:
        if node.bl_idname not in SUPPORTED_SHADER_NODES:
            raise ValueError('Shader node type needs an explicit serializer review: '+node.bl_idname)
        row={'name':node.name,'type':node.bl_idname,'rna':_rna_state(node,node=True),
             'inputs':[{'index':i,'identifier':socket.identifier,'name':socket.name,
                        'value':_plain(socket.default_value)} for i,socket in enumerate(node.inputs)
                       if hasattr(socket,'default_value')]}
        if getattr(node,'image',None):
            image=node.image
            if image.packed_file is None:raise ValueError('Environment texture must be packed')
            row['image']={'name':image.name,'sha256':hashlib.sha256(bytes(image.packed_file.data)).hexdigest(),
                          'color_space':image.colorspace_settings.name,'source':image.source,
                          'alpha_mode':image.alpha_mode}
        nodes.append(row)
    links=sorted((link.from_node.name,list(link.from_node.outputs).index(link.from_socket),
                  link.to_node.name,list(link.to_node.inputs).index(link.to_socket)) for link in tree.links)
    return {'nodes':sorted(nodes,key=lambda x:x['name']),'links':links}


def _material_state(material):
    # Cycles initialization only; no generic LineArt/Freestyle serialization.
    keys=('use_nodes','use_backface_culling','blend_method','shadow_method','alpha_threshold','diffuse_color')
    return {'properties':{k:_plain(getattr(material,k)) for k in keys if hasattr(material,k)},
            'nodes':_nodes(material.node_tree) if material.use_nodes else None}


def _light_state(light):
    keys=('type','energy','color','shape','size','size_y','spread','shadow_soft_size',
          'diffuse_factor','specular_factor','volume_factor','use_shadow','use_nodes',
          'use_custom_distance','cutoff_distance')
    return {'properties':{k:_plain(getattr(light,k)) for k in keys if hasattr(light,k)},
            'nodes':_nodes(light.node_tree) if light.use_nodes else None}



def object_record(obj):
    import numpy as np
    row={'name':obj.name,'type':obj.type,'hide_render':obj.hide_render,'hide_viewport':obj.hide_viewport,'parent':obj.parent.name if obj.parent else None,'matrix_world':[list(v) for v in obj.matrix_world],'scale':list(obj.scale),'pass_index':obj.pass_index}
    row['cycles_visibility']={key:getattr(obj,key) for key in ('visible_camera','visible_diffuse','visible_glossy','visible_transmission','visible_volume_scatter','visible_shadow','is_holdout','is_shadow_catcher')}
    if obj.type=='MESH':
        row['mesh_sha256']=digest({'vertices':[list(v.co) for v in obj.data.vertices],'faces':[list(p.vertices) for p in obj.data.polygons],'material_indices':[p.material_index for p in obj.data.polygons],'uv':{u.name:[list(x.uv) for x in u.data] for u in obj.data.uv_layers}})
        row['materials']={m.name:_material_state(m) for m in obj.data.materials if m}
    elif obj.type=='CURVE':
        row['curve_sha256']=digest({'bevel_depth':obj.data.bevel_depth,'splines':[[list(p.co) for p in sp.bezier_points] for sp in obj.data.splines]})
        row['materials']={m.name:_material_state(m) for m in obj.data.materials if m}
    elif obj.type=='ARMATURE':
        row['bones']={b.name:{'length':b.bone.length,'matrix_basis':[list(x) for x in b.matrix_basis]} for b in obj.pose.bones}
    elif obj.type=='LIGHT':
        row['light']=_light_state(obj.data)
    return row


def _bounds(obj):
    import bpy
    from mathutils import Vector
    evaluated=obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh=evaluated.to_mesh()
    try:
        pts=[evaluated.matrix_world@v.co for v in mesh.vertices]
        return [[min(v[i] for v in pts) for i in range(3)],[max(v[i] for v in pts) for i in range(3)]]
    finally:evaluated.to_mesh_clear()


def _collection_visibility(scene):
    """Current render-layer contribution hierarchy; no generic RNA traversal."""
    def layer_record(layer, hidden=False, excluded=False):
        collection=layer.collection
        hidden=hidden or collection.hide_render
        excluded=excluded or layer.exclude
        return {'collection':collection.name,
                'objects':sorted(obj.name for obj in collection.objects),
                'hide_render':collection.hide_render,'hide_viewport':collection.hide_viewport,
                'exclude':layer.exclude,'holdout':layer.holdout,'indirect_only':layer.indirect_only,
                'layer_hide_viewport':layer.hide_viewport,
                'effective_render_enabled':not(hidden or excluded),
                'children':[layer_record(child,hidden,excluded) for child in sorted(layer.children,key=lambda x:x.name)]}
    return {layer.name:{'use':layer.use,'collections':layer_record(layer.layer_collection)}
            for layer in sorted(scene.view_layers,key=lambda x:x.name)}


def _world_state(world):
    return {'name':world.name,'use_nodes':world.use_nodes,'fallback_color':list(world.color),
            'cycles_visibility':{key:getattr(world.cycles_visibility,key)
                                 for key in ('camera','diffuse','glossy','transmission','shadow','scatter')},
            'nodes':_nodes(world.node_tree) if world.node_tree else None}


def fixed_snapshot(scene):
    observer=scene.objects[PREFIX+'observer'];c=observer.data
    objects={o.name:object_record(o) for o in scene.objects if o.get('g1b_static')}
    for name,row in objects.items():
        row['object_id']=scene.objects[name]['global_environment_object_id']
        if scene.objects[name].type in ('MESH','CURVE'):
            import bpy
            obj=scene.objects[name].evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=obj.to_mesh()
            try:
                pts=[list(obj.matrix_world@v.co) for v in mesh.vertices]
                row['evaluated_geometry_sha256']=digest({'world_vertices':pts,'faces':[list(p.vertices) for p in mesh.polygons]})
                row['evaluated_bounds_m']=[[min(p[i] for p in pts) for i in range(3)],[max(p[i] for p in pts) for i in range(3)]]
                row['evaluated_dimensions_m']=[row['evaluated_bounds_m'][1][i]-row['evaluated_bounds_m'][0][i] for i in range(3)]
            finally:obj.to_mesh_clear()
    inventory={obj.name:{'type':obj.type,'data':obj.data.name if obj.data else None,
                         'hide_render':obj.hide_render,'parent':obj.parent.name if obj.parent else None}
               for obj in scene.objects}
    dynamic_rig={}
    for obj in scene.objects:
        if obj.get('g1b_static') or obj==observer:continue
        if obj.type not in {'MESH','ARMATURE','EMPTY'}:raise ValueError('Unregistered scene contributor outside G1 dynamic objects')
        row=object_record(obj);row.pop('matrix_world',None)
        if 'bones' in row:
            for bone in row['bones'].values():bone.pop('matrix_basis',None)
        dynamic_rig[obj.name]=row
    return {'objects':objects,'scene_inventory':inventory,'collection_visibility':_collection_visibility(scene),'dynamic_G1_geometry_and_materials':dynamic_rig,'camera_intrinsics':{k:_plain(getattr(c,k)) for k in ('type','lens','sensor_width','sensor_height','sensor_fit','shift_x','shift_y','clip_start','clip_end')},
        'render':{'engine':scene.render.engine,'use_freestyle':scene.render.use_freestyle,'resolution_x':scene.render.resolution_x,'resolution_y':scene.render.resolution_y,'resolution_percentage':scene.render.resolution_percentage,'pixel_aspect_x':scene.render.pixel_aspect_x,'pixel_aspect_y':scene.render.pixel_aspect_y},
        'world':_world_state(scene.world),'view':{k:getattr(scene.view_settings,k) for k in ('view_transform','look','exposure','gamma')}}


def garment_geometry_checks(scene,cfg,source_primary_span_m):
    """Source-scale and physical support checks; diagnostic geometry, not fit claims."""
    import numpy as np
    g=cfg['initial_priors']['geometry'];cloth=cfg['initial_priors']['garment']
    garments=[o for o in scene.objects if o.name.startswith(PREFIX+'garment_')]
    buttons=[o for o in garments if 'jacket_metal_button' in o.name]
    primary=[o for o in garments if o.type=='MESH' and 'jacket_connected_front_sleeves' in o.name]
    if len(buttons)!=6 or len(primary)!=1:raise ValueError('Six source buttons and one primary garment required')
    bounds={o.name:_bounds(o) for o in garments}
    # Current shared source rest dimensions preserve physical 14x14x3.2 mm buttons.
    sx,sy=cloth['rest_outline_size_xy_m'][0]/.9,cloth['rest_outline_size_xy_m'][1]/.5
    angle=math.radians(cloth['orientation_z_deg']);ca,sa=abs(math.cos(angle)),abs(math.sin(angle))
    expected=[.014*(ca*sx+sa*sy),.014*(sa*sx+ca*sy),.0032]
    button_dimensions={}
    for obj in buttons:
        lo,hi=bounds[obj.name];dims=np.subtract(hi,lo);button_dimensions[obj.name]=dims.tolist()
        if not np.allclose(dims,expected,rtol=0,atol=2e-6):raise ValueError('Source button scale mismatch: '+obj.name)
    tx,ty=g['table_center_xy_m'];tw,td=g['table_size_xy_m']
    xy_min=np.min([b[0][:2] for b in bounds.values()],axis=0);xy_max=np.max([b[1][:2] for b in bounds.values()],axis=0)
    if np.any(xy_min<[tx-tw/2-2e-6,ty-td/2-2e-6]) or np.any(xy_max>[tx+tw/2+2e-6,ty+td/2+2e-6]):
        raise ValueError('Actual garment XY leaves shared tabletop')
    lo,hi=bounds[primary[0].name];expected_min=g['table_top_z_m']+cloth['support_clearance_m']
    if abs(lo[2]-expected_min)>2e-6 or abs((hi[2]-lo[2])-source_primary_span_m)>2e-6:
        raise ValueError('Primary cloth support/height mismatch independently of buttons')
    return {'button_dimensions_m':button_dimensions,'source_button_dimensions_m':[.014,.014,.0032],
            'actual_garment_xy_min_m':xy_min.tolist(),'actual_garment_xy_max_m':xy_max.tolist(),
            'primary_bounds_m':[lo,hi],'primary_table_clearance_m':lo[2]-g['table_top_z_m'],
            'source_primary_height_span_m':source_primary_span_m}


def verify_environment(scene, contract):
    if scene.camera != scene.objects.get(PREFIX+'observer') or len([o for o in scene.objects if o.type=='CAMERA'])!=1:
        raise ValueError('Exactly one shared observing camera must remain active')
    if scene.get('g1b_status') != STATUS or scene.get('g1b_dataset_generation_allowed') is not False:
        raise ValueError('Environment scope changed')
    garment_geometry_checks(scene,contract['config'],contract['garment_geometry_checks']['source_primary_height_span_m'])
    _assert_numeric(actual_K(scene),contract['K_prior'],'observing K')
    if digest(fixed_snapshot(scene)) != contract['fixed_snapshot_sha256']:
        raise ValueError('Shared world geometry/material/light or observing K changed')
    return True


def _assert_numeric(actual, requested, label):
    import numpy as np
    a,b=np.asarray(actual,dtype=float),np.asarray(requested,dtype=float)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.allclose(a,b,rtol=2e-7,atol=1e-6):
        raise ValueError('Blender assignment differs from requested '+label)


def _assign_checked(owner, key, requested):
    prop=owner.bl_rna.properties[key]
    if prop.type in {'FLOAT','INT'}:
        values=requested if isinstance(requested,(tuple,list)) else [requested]
        if any(not math.isfinite(v) or v<prop.hard_min or v>prop.hard_max for v in values):
            raise ValueError('Value outside Blender RNA representation range: '+key)
    setattr(owner,key,requested)
    actual=getattr(owner,key)
    if prop.type in {'FLOAT','INT'}:_assert_numeric(_plain(actual),requested,key)
    elif actual!=requested:raise ValueError('Blender enum/value assignment differs: '+key)


def _blender_config_preflight(cfg):
    """Check actual dynamic enums/ranges on isolated, promptly removed probes."""
    import bpy
    scene=camera=light=None
    try:
        scene=bpy.data.scenes.new('__g1b_rna_probe_scene')
        camera=bpy.data.cameras.new('__g1b_rna_probe_camera')
        light=bpy.data.lights.new('__g1b_rna_probe_light','AREA')
        c=cfg['initial_priors']['capture'];l=cfg['initial_priors']['lighting']
        for key,value in {'lens':c['focal_mm'],'sensor_width':c['sensor_width_mm'],
                          'clip_start':c['clip_start_m'],'clip_end':c['clip_end_m']}.items():_assign_checked(camera,key,value)
        for key,value in {'energy':l['area_energy_w'],'size':l['area_size_m'],'color':l['area_color_linear_rgb']}.items():_assign_checked(light,key,value)
        for key,value in {'view_transform':l['view_transform'],'look':l['look'],'exposure':l['exposure_ev'],'gamma':l['gamma']}.items():_assign_checked(scene.view_settings,key,value)
    except (TypeError,ValueError,RuntimeError) as exc:
        raise ValueError('Blender configuration preflight failed: '+str(exc)) from exc
    finally:
        if light is not None:bpy.data.lights.remove(light)
        if camera is not None:bpy.data.cameras.remove(camera)
        if scene is not None:bpy.data.scenes.remove(scene)


def actual_K(scene):
    c=scene.objects[PREFIX+'observer'].data
    w=scene.render.resolution_x*scene.render.resolution_percentage/100
    h=scene.render.resolution_y*scene.render.resolution_percentage/100
    f=c.lens/c.sensor_width*w
    return [[f,0,(w-1)/2],[0,f*scene.render.pixel_aspect_x/scene.render.pixel_aspect_y,(h-1)/2],[0,0,1]]


def apply_capture_pose(scene, pose, contract):
    import bpy,numpy as np
    from mathutils import Vector,Matrix
    validate_pose(pose)
    verify_environment(scene,contract)
    observer=scene.objects[PREFIX+'observer']
    old_channels={key:getattr(observer,key).copy() if hasattr(getattr(observer,key),'copy') else tuple(getattr(observer,key)) for key in ('location','rotation_euler','rotation_quaternion','rotation_axis_angle','scale')}
    try:
        location=np.asarray(pose['location_m'],dtype=float)
        direction=np.asarray(pose['target_m'],dtype=float)-location
        direction/=np.linalg.norm(direction)  # float64 normalization before float32 Blender values
        rotation=Vector(direction.tolist()).to_track_quat('-Z','Y').to_matrix()@Matrix.Rotation(math.radians(pose['roll_deg']),3,'Z')
        expected=rotation.to_4x4();expected.translation=Vector(location.tolist())
        observer.matrix_world=expected
        bpy.context.view_layer.update()
        _assert_numeric(list(observer.matrix_world.translation),pose['location_m'],'capture location')
        _assert_numeric([list(row) for row in observer.matrix_world],[list(row) for row in expected],'capture matrix')
        basis=np.asarray(observer.matrix_world,dtype=float)[:3,:3]
        if not np.isfinite(basis).all() or not np.allclose(basis.T@basis,np.eye(3),atol=2e-6) or abs(np.linalg.det(basis)-1)>2e-6:
            raise ValueError('Actual capture rotation is not finite proper SE3')
        verify_environment(scene,contract)
    except BaseException:
        for key,value in old_channels.items():setattr(observer,key,value)
        bpy.context.view_layer.update()
        raise
    return [list(x) for x in observer.matrix_world]


def _creation_state(scene):
    import bpy
    collections=('objects','meshes','curves','materials','images','lights','cameras','worlds','texts','collections','node_groups','actions','scenes')
    return {
        'datablocks':{key:{x.as_pointer() for x in getattr(bpy.data,key)} for key in collections},
        'world':scene.world,'camera':scene.camera,'properties':{key:(scene[key].to_dict() if hasattr(scene[key],'to_dict') else scene[key]) for key in scene.keys()},
        'render':{key:getattr(scene.render,key) for key in ('engine','resolution_x','resolution_y','resolution_percentage','pixel_aspect_x','pixel_aspect_y','threads_mode','threads','use_freestyle')},
        'view':{key:getattr(scene.view_settings,key) for key in ('view_transform','look','exposure','gamma')},
        'active':bpy.context.view_layer.objects.active,'selected':tuple(bpy.context.selected_objects)}


def _restore_creation(scene, before):
    import bpy
    scene.world=before['world'];scene.camera=before['camera']
    for key,value in before['render'].items():setattr(scene.render,key,value)
    # Look enums depend on the view transform: restore in this explicit order.
    for key in ('view_transform','look','exposure','gamma'):setattr(scene.view_settings,key,before['view'][key])
    for key,pointers in before['datablocks'].items():
        collection=getattr(bpy.data,key)
        for value in list(collection):
            if value.as_pointer() not in pointers:collection.remove(value,do_unlink=True)
    for obj in scene.objects:obj.select_set(obj in before['selected'])
    bpy.context.view_layer.objects.active=before['active']
    bpy.context.view_layer.update()
    # RNA access can lazily create the cycles ID-property group; restore exact presence last.
    for key in list(scene.keys()):
        if key not in before['properties']:del scene[key]
    for key,value in before['properties'].items():scene[key]=value


def add_environment(scene, cfg):
    """Transactional addition to G1: preflight before changes, rollback on error."""
    import bpy
    validate_config(cfg)  # Includes isolated RNA setter/readback preflight in Blender.
    if scene!=bpy.context.scene or bpy.context.mode!='OBJECT':
        raise ValueError('Initialize the active G1 scene in object mode')
    before=_creation_state(scene)
    try:
        return _add_environment_impl(scene,cfg)
    except BaseException:
        _restore_creation(scene,before)
        raise


def _add_environment_impl(scene, cfg):
    """Add a new shared environment; never reset scene or alter existing rig assets."""
    import bpy
    from mathutils import Matrix,Vector
    from data import action4_render as base
    from data.action4_synthesis import _connected_garment
    if scene!=bpy.context.scene or scene.camera is not None or scene.world is not None:
        raise ValueError('Initialize neutral G1 scene without pre-existing observing camera/world')
    if scene.unit_settings.scale_length!=1 or scene.unit_settings.system!='METRIC':raise ValueError('Metric G1 scene required')
    required={'global_source_anatomy','global_metric_armature','left_Action4_core','right_Action4_core'}
    if not required.issubset(scene.objects.keys()) or 'GLOBAL_RIG_INITIALIZATION.json' not in bpy.data.texts:raise ValueError('Validated neutral G1 scene required')
    import numpy as np
    rig_contract=json.loads(bounded(cfg['sources']['rig_contract']['path'],True).read_text())
    body=scene.objects['global_source_anatomy'];verts=np.array([list(v.co) for v in body.data.vertices]);faces=[tuple(p.vertices) for p in body.data.polygons]
    if hashlib.sha256(verts.tobytes()+json.dumps(faces).encode()).hexdigest()!=rig_contract['source_geometry_sha256']:raise ValueError('G1 source body geometry differs')
    if any(abs(b.matrix_basis[i][j]-(1 if i==j else 0))>1e-7 for b in scene.objects['global_metric_armature'].pose.bones for i in range(4) for j in range(4)):raise ValueError('Initialize environment from neutral G1 pose only')
    reserved=('g1b_env_','jacket_','chest_pocket_','collar_','connected_jacket')
    if TEXT in bpy.data.texts or any(x.name.startswith(reserved) for collection in (bpy.data.objects,bpy.data.materials,bpy.data.meshes,bpy.data.curves,bpy.data.worlds,bpy.data.cameras,bpy.data.lights) for x in collection):
        raise ValueError('Environment would overwrite or suffix an existing reserved object/datablock')
    denim=json.loads(bounded(cfg['sources']['denim_manifest']['path'],True).read_text())
    for image in bpy.data.images:
        if image.name in {Path(s['path']).name for s in denim['files'].values()} or (image.packed_file and hashlib.sha256(bytes(image.packed_file.data)).hexdigest() in {s['sha256'] for s in denim['files'].values()}):
            raise ValueError('Existing denim image datablock must not be mutated by source helper')
    old={obj.name:object_record(obj) for obj in scene.objects}
    before_images={im.name:hashlib.sha256(bytes(im.packed_file.data)).hexdigest() for im in bpy.data.images if im.packed_file}
    initial_names=set(scene.objects.keys())
    p=cfg['initial_priors'];g=p['geometry'];mat=p['materials'];cloth=p['garment'];light=p['lighting'];capture=p['capture']
    body_bounds=_bounds(scene.objects['global_source_anatomy'])
    near_y=g['table_center_xy_m'][1]+g['table_size_xy_m'][1]/2
    clearance=body_bounds[0][1]-near_y
    if clearance<g['body_table_clearance_m'] or g['floor_top_z_m']>body_bounds[0][2]:
        raise ValueError('Initial table/floor intersects neutral G1 body envelope')
    def material(name,color):return base._material(PREFIX+name,color,mat['roughness'])
    def box(name,centre,size,material):
        return base._cube(PREFIX+name,centre,size,material)
    wallmat=material('wall_material',mat['wall_linear_rgb']);floormat=material('floor_material',mat['floor_linear_rgb']);legmat=material('table_leg_material',mat['table_leg_linear_rgb'])
    grid=base._grid_material();grid.name=PREFIX+'table_grid_material'
    for node in grid.node_tree.nodes:
        if node.type=='MATH' and node.operation=='MULTIPLY':node.inputs[1].default_value=1/mat['grid_pitch_m']
        if node.type=='MATH' and node.operation=='LESS_THAN':node.inputs[1].default_value=mat['grid_line_fraction']
        if node.type=='MIX_RGB':node.inputs[1].default_value=(*mat['grid_cell_linear_rgb'],1);node.inputs[2].default_value=(*mat['grid_line_linear_rgb'],1)
    grid.node_tree.nodes['Principled BSDF'].inputs['Roughness'].default_value=mat['roughness']
    tx,ty=g['table_center_xy_m'];tw,td=g['table_size_xy_m'];floor=g['floor_top_z_m'];top=g['table_top_z_m']
    box('table_surface',(tx,ty,top-g['table_thickness_m']/2),(tw,td,g['table_thickness_m']),grid)
    box('floor',(*g['floor_center_xy_m'],floor-g['floor_thickness_m']/2),(*g['floor_size_xy_m'],g['floor_thickness_m']),floormat)
    box('front_wall',(0,g['wall_center_y_m'],floor+g['wall_height_m']/2),(g['wall_width_m'],g['wall_thickness_m'],g['wall_height_m']),wallmat)
    height=top-g['table_thickness_m']-floor
    for i,(sx,sy) in enumerate(((-1,-1),(-1,1),(1,-1),(1,1))):
        box('table_leg_'+str(i),(tx+sx*(tw/2-g['table_leg_inset_m']),ty+sy*(td/2-g['table_leg_inset_m']),floor+height/2),(g['table_leg_width_m'],g['table_leg_width_m'],height),legmat)
    prior=set(scene.objects.keys());_connected_garment({'material_evidence':{'denim_manifest':cfg['sources']['denim_manifest']['path']}},cloth['phase_radians'])
    # Newly assigned object.scale is not yet guaranteed in cached matrix_world.
    bpy.context.view_layer.update()
    garments=[obj for obj in scene.objects if obj.name not in prior]
    source_world={obj:obj.matrix_world.copy() for obj in garments}
    def control_points(obj):
        return [v.co for v in obj.data.vertices] if obj.type=='MESH' else [p.co for sp in obj.data.splines for p in sp.bezier_points]
    source_vertices={obj:[v.copy() for v in control_points(obj)] for obj in garments}
    source_buttons=[obj for obj in garments if 'jacket_metal_button' in obj.name]
    if len(source_buttons)!=6:raise ValueError('Six source buttons required')
    for obj in source_buttons:
        lo,hi=_bounds(obj)
        if not np.allclose(np.subtract(hi,lo),[.014,.014,.0032],rtol=0,atol=2e-6):
            raise ValueError('Source button dimensions invalid before rebase')
    primary=next(obj for obj in garments if obj.type=='MESH' and 'jacket_connected_front_sleeves' in obj.name)
    source_bounds=_bounds(primary);source_primary_span=source_bounds[1][2]-source_bounds[0][2]
    scale=Matrix.Diagonal((cloth['rest_outline_size_xy_m'][0]/.9,cloth['rest_outline_size_xy_m'][1]/.5,1,1))
    local=scale@Matrix.Translation((-cloth['source_outline_center_xy_m'][0],-cloth['source_outline_center_xy_m'][1],0))
    rigid=Matrix.Translation((tx,ty,0))@Matrix.Rotation(math.radians(cloth['orientation_z_deg']),4,'Z')
    for obj in garments:
        obj.data.transform(local@source_world[obj]);obj.matrix_world=rigid;obj.name=PREFIX+'garment_'+obj.name
    bpy.context.view_layer.update()
    rebase_errors={}
    for obj in garments:
        expected=np.asarray([list(rigid@local@source_world[obj]@v) for v in source_vertices[obj]])
        actual=np.asarray([list(obj.matrix_world@v) for v in control_points(obj)])
        error=float(np.max(np.abs(expected-actual)));rebase_errors[obj.name]=error
        if error>2e-6:raise ValueError('Garment rebase position/size mismatch: '+obj.name)
    minimum=min(_bounds(obj)[0][2] for obj in garments)
    shift=top+cloth['support_clearance_m']-minimum
    for obj in garments:obj.location.z+=shift
    bpy.context.view_layer.update()
    garment_checks=garment_geometry_checks(scene,cfg,source_primary_span)
    garment_checks['rebase_max_vertex_error_m']=rebase_errors
    world=bpy.data.worlds.new(PREFIX+'world');world.use_nodes=True;scene.world=world
    world.node_tree.nodes['Background'].inputs['Color'].default_value=(*light['world_linear_rgb'],1)
    world.node_tree.nodes['Background'].inputs['Strength'].default_value=light['world_strength']
    ld=bpy.data.lights.new(PREFIX+'area_data','AREA');lamp=bpy.data.objects.new(PREFIX+'area',ld);scene.collection.objects.link(lamp)
    lamp.location=light['area_location_m'];lamp.rotation_euler=(Vector(light['area_target_m'])-lamp.location).to_track_quat('-Z','Y').to_euler();ld.energy=light['area_energy_w'];ld.size=light['area_size_m'];ld.color=light['area_color_linear_rgb']
    cd=bpy.data.cameras.new(PREFIX+'observer_data');observer=bpy.data.objects.new(PREFIX+'observer',cd);scene.collection.objects.link(observer);scene.camera=observer
    cd.type='PERSP';cd.lens=capture['focal_mm'];cd.sensor_width=capture['sensor_width_mm'];cd.sensor_fit='HORIZONTAL';cd.shift_x=cd.shift_y=0;cd.clip_start=capture['clip_start_m'];cd.clip_end=capture['clip_end_m']
    scene.render.resolution_x,scene.render.resolution_y=cfg['native_image']['resolution_wh'];scene.render.resolution_percentage=100;scene.render.pixel_aspect_x=scene.render.pixel_aspect_y=1
    scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.render.threads_mode='FIXED';scene.render.threads=2
    for key in ('view_transform','look','exposure','gamma'):
        setattr(scene.view_settings,key,light['exposure_ev'] if key=='exposure' else light[key])
    for obj in scene.objects:
        if obj.name not in initial_names and obj!=observer:
            obj['g1b_static']=True;obj['global_environment_object_id']=obj.name.removeprefix(PREFIX);obj['evidence_status']=STATUS;obj.pass_index=0
    scene['g1b_status']=STATUS;scene['g1b_dataset_generation_allowed']=False
    bpy.context.view_layer.update()
    if old!={name:object_record(scene.objects[name]) for name in old}:raise ValueError('Existing G1 object/geometry/material changed')
    if before_images!={im.name:hashlib.sha256(bytes(im.packed_file.data)).hexdigest() for im in bpy.data.images if im.name in before_images}:raise ValueError('Existing packed image changed')
    _assert_numeric(actual_K(scene),expected_K(cfg),'initial observing K')
    fixed=fixed_snapshot(scene)
    baseline_hashes={'K_sha256':digest(expected_K(cfg)), 'materials_sha256':digest({name:row.get('materials',{}) for name,row in fixed['objects'].items()}), 'lights_and_world_sha256':digest({'lights':{name:row for name,row in fixed['objects'].items() if row['type']=='LIGHT'},'world':fixed['world'],'view':fixed['view']}), 'static_geometry_sha256':digest({name:{k:v for k,v in row.items() if k!='materials'} for name,row in fixed['objects'].items()})}
    contract={'schema_version':'action4.global_environment_contract.v1','status':STATUS,'dataset_generation_allowed':False,'fit_status':'pending_G2_multiframe_fit','units':'metres','config':cfg,'config_sha256':digest(cfg),'K_prior':expected_K(cfg),'baseline_hashes':baseline_hashes,'distortion_prior':capture['distortion_initial'],'fixed_snapshot':fixed,'fixed_snapshot_sha256':digest(fixed),'existing_G1_objects_sha256':digest(old),'neutral_body_table_y_clearance_m':clearance,'garment_support_z_shift_m':shift,'garment_geometry_checks':garment_checks,'existing_packed_images_preserved':before_images,'limits':['No render or appearance acceptance','All unmeasured geometry/capture/material/light values are initial_priors','Garment uses inherited analytic rest shape, not equilibrium/subject reconstruction','Engineering capture poses are tests, not calibrated sampling ranges']}
    pose={'location_m':capture['initial_location_m'],'target_m':capture['initial_target_m'],'roll_deg':capture['initial_roll_deg']}
    apply_capture_pose(scene,pose,contract)
    bpy.data.texts.new(TEXT).write(json.dumps(contract,indent=2,allow_nan=False))
    return contract


def build(config_path, output_dir, report_dir):
    import bpy
    cfg=validate_config(json.loads(bounded(config_path,True).read_text()))
    out,report=bounded(output_dir),bounded(report_dir)
    if not out.is_relative_to(ROOT/'data/action4_scene_assets/global_environment_init_v001') or not report.is_relative_to(ROOT/'outputs/action4_realism_reassessment_v001/global_environment_init'):
        raise ValueError('Use assigned environment asset/evidence directories')
    if out.exists() or (report/'build_report.json').exists():raise FileExistsError('Use fresh environment version/evidence; never overwrite')
    report.mkdir(parents=True,exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=str(bounded(cfg['sources']['rig_asset']['path'],True)),load_ui=False,use_scripts=False)
    contract=add_environment(bpy.context.scene,cfg)
    for pose in cfg['engineering_capture_poses']:apply_capture_pose(bpy.context.scene,pose,contract)
    c=cfg['initial_priors']['capture'];apply_capture_pose(bpy.context.scene,{'location_m':c['initial_location_m'],'target_m':c['initial_target_m'],'roll_deg':c['initial_roll_deg']},contract)
    out.mkdir(parents=True);asset=out/'initial.blend';bpy.ops.wm.save_as_mainfile(filepath=str(asset),check_existing=False)
    validate_config(cfg)
    contract['saved_asset']={'path':str(asset.relative_to(ROOT)),'sha256':sha(asset)}
    for target,value in ((out/'environment_contract.json',contract),(report/'build_report.json',{'status':'INITIALIZATION_ONLY_REOPEN_PENDING','asset':contract['saved_asset'],'fixed_snapshot_sha256':contract['fixed_snapshot_sha256'],'existing_G1_preserved':True,'dataset_generation_allowed':False,'source_hashes':cfg['sources']})):
        with target.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False)
    return contract['saved_asset']


def reopen(asset, contract_path, report_path):
    import bpy
    contract=json.loads(bounded(contract_path,True).read_text());path=bounded(asset,True)
    if contract['saved_asset']['sha256']!=sha(path):raise ValueError('Saved initialization bytes changed')
    validate_config(contract['config'])
    bpy.ops.wm.open_mainfile(filepath=str(path),load_ui=False,use_scripts=False)
    verify_environment(bpy.context.scene,contract)
    for pose in contract['config']['engineering_capture_poses']:apply_capture_pose(bpy.context.scene,pose,contract)
    value={'status':'PASS_ENGINEERING_REOPEN_ONLY','dataset_generation_allowed':False,'fitted':False,'asset_sha256':sha(path),'fixed_snapshot_sha256':contract['fixed_snapshot_sha256'],'capture_pose_cases':len(contract['config']['engineering_capture_poses']),'source_inputs_unchanged':True,'limits':'No GPU/render/appearance fit or generation acceptance'}
    target=bounded(report_path)
    if not target.is_relative_to(ROOT/'outputs/action4_realism_reassessment_v001/global_environment_init'):raise ValueError('Report outside assigned outputs')
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('x') as stream:json.dump(value,stream,indent=2)
    return value
