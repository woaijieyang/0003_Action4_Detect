"""Immutable neutral wrist libraries extracted from the reviewed source surface.

Mesh data remain in source-body coordinates. A proper rigid child transform
expresses them in the brace-root frame; callers must not treat data vertices as
brace coordinates. This library is not an assembled/accepted training scene.
"""
import json
import hashlib
from pathlib import Path
import numpy as np
from data import action4_global_environment as env

ROOT = env.ROOT
ASSETS = ROOT / 'data/action4_scene_assets/modular_wrist_v001'
OUTPUTS = ROOT / 'outputs/action4_realism_reassessment_v001/modular_wrist_build_v001'
TEXT = 'Action4_Wrist_Module.json'
SIDES = ('left', 'right')
REGIONS = [44,45,46,47,48,49,50,51,52,53,77,78,79,88,89,90,91,92,93,102]


def checked_ref(ref):
    if not isinstance(ref, dict) or set(ref) != {'path', 'sha256'}:
        raise ValueError('Exact source path and SHA required')
    path = env.bounded(ref['path'], True)
    if env.sha(path) != ref['sha256']:
        raise ValueError('Source identity changed: ' + ref['path'])
    return path


def ref(path):
    path = env.bounded(path, True)
    return {'path': str(path.relative_to(ROOT)), 'sha256': env.sha(path)}


def write_json(path, value):
    path = env.bounded(path)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def rigid_frame(raw):
    """Remove only float32 frame roundoff; never accept intentional scale/mirror."""
    raw = np.asarray(raw, dtype=float)
    if raw.shape != (4,4) or not np.isfinite(raw).all() or not np.array_equal(raw[3], [0,0,0,1]):
        raise ValueError('Finite homogeneous frame required')
    u, s, vt = np.linalg.svd(raw[:3,:3])
    if np.linalg.det(raw[:3,:3]) <= 0 or np.max(abs(s-1)) > 1e-6:
        raise ValueError('No mirrored/scaled module frame')
    result = raw.copy(); result[:3,:3] = u @ vt
    return result


def transform(matrix, points):
    return np.asarray(points) @ np.asarray(matrix)[:3,:3].T + np.asarray(matrix)[:3,3]


def selection(vertices, triangles, regions, side):
    if side not in SIDES:
        raise ValueError('Explicit source anatomical side required')
    sign = 1 if side == 'left' else -1
    return np.flatnonzero(np.isin(regions, REGIONS) & (sign*vertices[triangles,0] > 0).all(axis=1))


def boundary_edges(faces):
    edges = np.sort(np.concatenate([faces[:,[0,1]], faces[:,[1,2]], faces[:,[2,0]]]), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    if np.any(counts > 2):
        raise ValueError('Non-manifold edge')
    return unique[counts == 1]


def boundary_loop(faces):
    edges = boundary_edges(faces); adj = {}
    for a,b in edges:
        adj.setdefault(int(a), []).append(int(b)); adj.setdefault(int(b), []).append(int(a))
    if not adj or any(len(v) != 2 for v in adj.values()):
        raise ValueError('One two-regular elbow opening required')
    start = min(adj); loop = [start]; prev = None; cur = start
    while True:
        nxt = min(v for v in adj[cur] if v != prev)
        if nxt == start: break
        if nxt in loop: raise ValueError('Boundary cycle invalid')
        loop.append(nxt); prev,cur = cur,nxt
    if len(loop) != len(adj): raise ValueError('More than one boundary loop')
    return np.asarray(loop, dtype=np.int32)


def validate_config(cfg):
    if set(cfg) != {'schema','sources','dataset_generation_allowed','source_object','selection','frame_policy'}:
        raise ValueError('Unexpected module configuration')
    if cfg['schema'] != 'action4.wrist_modules.v1' or cfg['dataset_generation_allowed'] is not False:
        raise ValueError('Diagnostic library only')
    if cfg['source_object'] != 'global_source_anatomy' or cfg['selection'] != 'frozen_forearm_and_natural_hand' or cfg['frame_policy'] != 'body_data_rigid_child_in_brace_root':
        raise ValueError('Unsupported extraction policy')
    if set(cfg['sources']) != {'design','design_handoff','asset','contract','snapshot','neutral_cache','scene_module','environment_module'}:
        raise ValueError('Complete source chain required')
    for item in cfg['sources'].values(): checked_ref(item)
    handoff = json.loads(checked_ref(cfg['sources']['design_handoff']).read_text())
    for name in ('design','asset','contract','snapshot','neutral_cache','scene_module'):
        r = cfg['sources'][name]
        declared = {**handoff['sources'], **handoff['artifacts']}
        if declared.get(r['path']) != r['sha256']: raise ValueError('Design ancestry mismatch')
    return cfg


def _array(collection, prop, width, dtype=float):
    a = np.empty(len(collection)*width, dtype=dtype); collection.foreach_get(prop,a)
    return a.reshape((-1,width)) if width != 1 else a


def _attribute(mesh, name, domain, values):
    a = mesh.attributes.new(name=name, type='INT', domain=domain)
    a.data.foreach_set('value', np.asarray(values,dtype=np.int32).ravel())


def _normals(mesh):
    mesh.calc_normals_split()
    return _array(mesh.loops, 'normal', 3)


def _texture_space(mesh):
    return {'location': list(mesh.texspace_location), 'size': list(mesh.texspace_size)}


def _verify_mesh(obj, contract, archive):
    from mathutils import Matrix
    mesh = obj.data
    vertices = _array(mesh.vertices,'co',3)
    faces = np.asarray([p.vertices[:] for p in mesh.polygons],dtype=np.int32)
    if not np.array_equal(vertices,archive['source_vertices_m']) or not np.array_equal(faces,archive['faces']):
        raise ValueError('Source surface geometry changed')
    for name,key in [('source_evaluated_vertex_id','source_vertex_ids'),('source_evaluated_triangle_id','source_triangle_ids'),('source_evaluated_polygon_id','source_polygon_ids'),('.sculpt_face_set','source_regions'),('source_evaluated_loop_id','source_loop_ids')]:
        if name not in mesh.attributes or not np.array_equal(_array(mesh.attributes[name].data,'value',1,np.int32),archive[key].ravel()):
            raise ValueError('Source provenance attribute changed: '+name)
    if len(boundary_loop(faces)) != 44 or not np.array_equal(np.sort(archive['source_vertex_ids'][boundary_loop(faces)]), np.sort(archive['source_boundary_vertex_ids'])):
        raise ValueError('Elbow boundary changed')
    if not all(p.use_smooth for p in mesh.polygons) or not mesh.has_custom_normals or not mesh.use_auto_smooth:
        raise ValueError('Source smooth-loop normal representation missing')
    normals = _normals(mesh); dots = np.einsum('ij,ij->i', normals, archive['source_loop_normals'])
    # Blender custom normal storage is quantized. This is angular agreement,
    # not an assertion of bitwise normal or rendered pixel equivalence.
    maximum_angle = float(np.max(np.arccos(np.clip(dots,-1,1))))
    if maximum_angle > .002: raise ValueError('Source loop normal preservation failed')
    if list(mesh.uv_layers.keys()) != contract['uv_layers']:
        raise ValueError('UV layer names changed')
    for i,uv in enumerate(mesh.uv_layers):
        if not np.array_equal(_array(uv.data,'uv',2),archive['uv_'+str(i)]):
            raise ValueError('Frozen extraction UV changed')
    if mesh.use_auto_texspace or _texture_space(mesh) != contract['source_texture_space']:
        raise ValueError('Generated coordinate texture-space changed')
    if len(obj.modifiers) or obj.pass_index != 0 or obj.parent is None or obj.parent.name != contract['root_name']:
        raise ValueError('Library hierarchy/non-target state changed')
    if list(obj.scale) != [1,1,1] or np.linalg.det(np.asarray(obj.matrix_basis)[:3,:3]) <= 0:
        raise ValueError('No scale or mirror')
    if not np.array_equal(np.asarray(obj.parent.matrix_world), np.eye(4)) or not np.array_equal(np.asarray(obj.matrix_parent_inverse),np.eye(4)):
        raise ValueError('Library root must be identity brace frame')
    if not np.array_equal(np.asarray(obj.matrix_basis),np.asarray(contract['actual_child_matrix'])):
        raise ValueError('Child frame changed')
    local = transform(obj.matrix_world,vertices)
    err = float(np.max(abs(local - archive['brace_vertices_m'])))
    roundtrip = float(np.max(abs(transform(contract['rigid_F_new_body'],local)-vertices)))
    if max(err,roundtrip) > 1e-6: raise ValueError('Rigid source/brace roundtrip failed')
    if len(mesh.materials)!=1 or env.digest(env._material_state(mesh.materials[0])) != env.digest(contract['material_state']):
        raise ValueError('Active source shader graph/image changed')
    return {'vertices':len(vertices),'triangles':len(faces),'open_elbow_edges':44,'max_root_local_error_m':err,'max_roundtrip_error_m':roundtrip,'maximum_source_normal_angle_rad':maximum_angle,'all_smooth':True,'source_provenance_exact':True,'frozen_extraction_UV_exact':True,'active_shader_and_packed_image_unchanged':True}


def build(config_path, output_dir, report_dir):
    import bpy
    from mathutils import Matrix
    from data import action4_global_scene as integrated
    cfg = validate_config(json.loads(env.bounded(config_path,True).read_text()))
    out, report = env.bounded(output_dir), env.bounded(report_dir)
    if not out.is_relative_to(ASSETS) or not report.is_relative_to(OUTPUTS):
        raise ValueError('Assigned asset/output boundaries required')
    if out.exists() or report.exists(): raise FileExistsError('Fresh immutable output directories required')
    design = json.loads(checked_ref(cfg['sources']['design']).read_text())
    source_contract = json.loads(checked_ref(cfg['sources']['contract']).read_text())
    bpy.ops.wm.open_mainfile(filepath=str(checked_ref(cfg['sources']['asset'])),load_ui=False,use_scripts=False)
    scene=bpy.context.scene; bpy.context.view_layer.update(); integrated.verify(scene,source_contract)
    body=scene.objects[cfg['source_object']]
    if not np.array_equal(np.asarray(body.matrix_world),np.eye(4)) or len(body.data.materials)!=1:
        raise ValueError('Reviewed identity-world body/material required')
    evaluated=body.evaluated_get(bpy.context.evaluated_depsgraph_get())
    source=evaluated.to_mesh(preserve_all_data_layers=True,depsgraph=bpy.context.evaluated_depsgraph_get())
    try:
        source.calc_loop_triangles(); source.calc_normals_split()
        V=_array(source.vertices,'co',3); T=np.asarray([t.vertices[:] for t in source.loop_triangles],dtype=np.int32)
        P=np.asarray([t.polygon_index for t in source.loop_triangles],dtype=np.int32)
        L=np.asarray([t.loops[:] for t in source.loop_triangles],dtype=np.int32)
        regions=_array(source.attributes['.sculpt_face_set'].data,'value',1,np.int32)[P]
        cache=np.load(checked_ref(cfg['sources']['neutral_cache']),allow_pickle=False)
        if not np.array_equal(V,cache['vertices_world_m']) or not np.array_equal(T,cache['triangles']) or not np.array_equal(P,cache['triangle_polygon_ids']) or not np.array_equal(regions,cache['triangle_source_regions']):
            raise ValueError('Actual v007 evaluated geometry differs from frozen design neutral')
        material=body.data.materials[0]; material_state=env._material_state(material)
        source_space=_texture_space(source)
        normals=_normals(source)
        out.mkdir(parents=True);report.mkdir(parents=True)
        rows=[]
        for side in SIDES:
            frozen=next(x for x in design['face_inventory'] if x['side']==side and x['part']=='recommended_forearm_with_natural_hand')
            tids=selection(V,T,regions,side); vids=np.unique(T[tids]); faces=np.searchsorted(vids,T[tids]); loops=L[tids].ravel()
            if tids.tolist()!=frozen['source_triangle_ids'] or vids.tolist()!=frozen['source_vertex_ids'] or len(tids)!=17280 or len(vids)!=8663:
                raise ValueError('Frozen per-side selection changed')
            boundary=boundary_loop(faces)
            if len(boundary)!=44: raise ValueError('Expected original open elbow44')
            F=rigid_frame(frozen['common_F_new']); inverse=np.linalg.inv(F)
            root=bpy.data.objects.new(side+'_wrist_module_root',None)
            collection=bpy.data.collections.new(side+'_wrist_module_library'); scene.collection.children.link(collection); collection.objects.link(root)
            mesh=bpy.data.meshes.new(side+'_source_body_surface');mesh.from_pydata(V[vids].tolist(),[],faces.tolist());mesh.update()
            mesh.use_auto_texspace=False;mesh.texspace_location=source_space['location'];mesh.texspace_size=source_space['size']
            for p in mesh.polygons:p.use_smooth=True
            mesh.use_auto_smooth=True;mesh.auto_smooth_angle=3.141592653589793
            mesh.normals_split_custom_set(normals[loops].tolist())
            for uv in source.uv_layers:
                layer=mesh.uv_layers.new(name=uv.name);layer.data.foreach_set('uv',_array(uv.data,'uv',2)[loops].ravel())
            mesh.materials.append(material)
            obj=bpy.data.objects.new(side+'_forearm_natural_hand',mesh);collection.objects.link(obj);obj.parent=root
            obj.rotation_mode='QUATERNION';obj.location=inverse[:3,3];obj.rotation_quaternion=Matrix(inverse.tolist()).to_quaternion();obj.scale=(1,1,1);obj.pass_index=0
            obj['dataset_generation_allowed']=False;obj['mesh_data_frame']='source_body';root['frame']='rigid_new_brace';root['side']=side
            for name,domain,data in [('source_evaluated_vertex_id','POINT',vids),('source_evaluated_triangle_id','FACE',tids),('source_evaluated_polygon_id','FACE',P[tids]),('.sculpt_face_set','FACE',regions[tids]),('source_evaluated_loop_id','CORNER',loops)]:_attribute(mesh,name,domain,data)
            bpy.context.view_layer.update()
            arrays={'source_vertices_m':V[vids],'faces':faces,'source_vertex_ids':vids,'source_triangle_ids':tids,'source_polygon_ids':P[tids],'source_regions':regions[tids],'source_loop_ids':loops,'source_loop_normals':normals[loops],'source_boundary_vertex_ids':vids[boundary],'brace_vertices_m':transform(inverse,V[vids])}
            for i,uv in enumerate(source.uv_layers):arrays['uv_'+str(i)]=_array(uv.data,'uv',2)[loops]
            geometry_path=out/(side+'_geometry.npz');np.savez_compressed(geometry_path,**arrays)
            contract={'schema':'action4.wrist_module_contract.v1','side':side,'status':'neutral_open_library_not_assembled','dataset_generation_allowed':False,'config':cfg,'source_module':ref(Path(__file__)),'geometry':ref(geometry_path),'root_name':root.name,'object_name':obj.name,'collection_name':collection.name,'mesh_data_frame':'original_evaluated_source_body_metres','effective_root_local_frame':'rigid_F_new_inverse_times_source_body','source_F_new_body':frozen['common_F_new'],'rigid_F_new_body':F.tolist(),'frame_rotation_roundoff_cleanup_max':float(np.max(abs(F-np.asarray(frozen['common_F_new'])))),'actual_child_matrix':np.asarray(obj.matrix_basis).tolist(),'source_texture_space':source_space,'material_state':material_state,'uv_layers':list(source.uv_layers.keys()),'source_skin_input':source_contract['skin_color_input'],'elbow_source_vertex_loop':vids[boundary].tolist(),'source_evaluated_polygons':len(np.unique(P[tids])),'source_rig_reuse':'neutral evaluated 35-bone source; no mutable rig in this static module library','camera_reuse':'Original R3/R4 module and canonical8 only; not embedded or remodelled here','band_reference':'Actual source28mm and analytic34mm are distinct; neither modified or assembled','limits':['One open elbow44 must be proven outside views or occluded before scene acceptance; no cap added.','Natural source hand is coarse occluder, no fine finger fit.','Separate side libraries are not an assembled scene and cannot be counted as colliding overlapping instances.','No mounting A, collision acceptance, RGB, or training labels produced.','Copied frozen extraction UV is preserved; independently evaluated sourceUV has known small unexplained differences. Current skin shader uses Object, not UV.','Custom split loop normals retain source smooth directions within0.002rad storage tolerance; no bitwise shading/RGB equality claim.','Source fullbody Generated texture space is copied explicitly; Object/BOX mesh coordinates remain originalbody.']}
            check=_verify_mesh(obj,contract,arrays)
            text=bpy.data.texts.new(TEXT+'_'+side);text.write(json.dumps(contract,indent=2,allow_nan=False))
            asset=out/(side+'.blend')
            # Packed source PNG lineage is retained as well as the active EXR dependency.
            packed={im for im in bpy.data.images if im.packed_file and hashlib.sha256(bytes(im.packed_file.data)).hexdigest()==source_contract['skin_color_input']['source_photo']['sha256']}
            bpy.data.libraries.write(str(asset),{collection,text}|packed,path_remap='NONE',fake_user=True,compress=True)
            external={**contract,'saved_asset':ref(asset)};write_json(out/(side+'_contract.json'),external)
            rows.append({'side':side,'asset':ref(asset),'contract':ref(out/(side+'_contract.json')),'geometry':ref(geometry_path),'build_checks':check})
            scene.collection.children.unlink(collection)
            bpy.context.view_layer.update()
        # Construction only added own collections; source/body and all source bytes stay authoritative.
        integrated.verify(scene,source_contract)
        validate_config(cfg)
        result={'schema':'action4.wrist_module_libraries.v1','modules':rows,'source_module':ref(Path(__file__)),'config':ref(config_path),'dataset_generation_allowed':False,'scope':'two independent neutral surface libraries; not an assembly; no rendering'}
        write_json(out/'modules_manifest.json',result);write_json(report/'build_report.json',result)
        return result
    finally:
        evaluated.to_mesh_clear()


def reopen(asset_path, contract_path, report_path):
    import bpy
    contract=json.loads(env.bounded(contract_path,True).read_text());asset=env.bounded(asset_path,True);report=env.bounded(report_path)
    if not asset.is_relative_to(ASSETS) or not report.is_relative_to(OUTPUTS) or report.exists():
        raise ValueError('Fresh assigned reload report required')
    if ref(asset)!=contract['saved_asset'] or ref(Path(__file__))!=contract['source_module']:raise ValueError('Asset/module identity mismatch')
    validate_config(contract['config']); archive=np.load(checked_ref(contract['geometry']),allow_pickle=False)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    with bpy.data.libraries.load(str(asset),link=False) as (available,loaded):
        if available.collections != [contract['collection_name']]: raise ValueError('Unexpected library collections')
        loaded.collections=available.collections;loaded.texts=available.texts
    collection=loaded.collections[0];bpy.context.scene.collection.children.link(collection);bpy.context.view_layer.update()
    if sorted(o.name for o in bpy.context.scene.objects)!=sorted([contract['root_name'],contract['object_name']]):raise ValueError('Not a standalone two-object module library')
    embedded=json.loads(bpy.data.texts[TEXT+'_'+contract['side']].as_string())
    if embedded!={k:v for k,v in contract.items() if k!='saved_asset'}:raise ValueError('Embedded/external contract diverged')
    checks=_verify_mesh(bpy.context.scene.objects[contract['object_name']],contract,archive)
    result={'status':'PASS_CPU_LIBRARY_REOPEN_ONLY','asset':ref(asset),'contract':ref(contract_path),'checks':checks,'training_accepted':False,'RGB_rendered':False}
    report.parent.mkdir(parents=True,exist_ok=True);write_json(report,result);return result
