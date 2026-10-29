"""Vectorized projection with the original isolated-mask rendering and crop checks."""
from pathlib import Path
import numpy as np
from data.action4_amodal import (_dimensions,MARGIN_PX,ROOT,crop_bounds,validate_geometry_coverage,pad_mask,write_binary_png,temporary_border,validate_settings)
from data.action4_modular_render import actual_K

def project_world(points,scene):
    capture=np.diag([1.,-1.,-1.,1.])@np.linalg.inv(np.array(scene.camera.matrix_world))
    p=np.asarray(points)@capture[:3,:3].T+capture[:3,3];h=p@np.asarray(actual_K(scene)).T
    uv=h[:,:2]/h[:,2:]
    return np.c_[uv[:,0]+.5,scene.render.resolution_y-uv[:,1]-.5,p[:,2]]

def geometry_preflight(scene,core,parts,corners):
    import bpy
    width,height=scene.render.resolution_x,scene.render.resolution_y
    matrix=np.array(core.matrix_world)
    bounds=crop_bounds(project_world(np.asarray(corners)@matrix[:3,:3].T+matrix[:3,3],scene),width,height)
    counts={}
    for part in parts:
        evaluated=part.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=evaluated.to_mesh()
        try:
            values=np.empty(len(mesh.vertices)*3,dtype=np.float64);mesh.vertices.foreach_get("co",values);values=values.reshape(-1,3);matrix=np.array(evaluated.matrix_world)
            validate_geometry_coverage(project_world(values@matrix[:3,:3].T+matrix[:3,3],scene),bounds,width,height);counts[part.name]=len(values)
        finally:evaluated.to_mesh_clear()
    return {"expanded_crop_bottomleft_xyxy":list(bounds),"evaluated_target_vertices":counts}

def render_crop(scene, camera, core, parts, corners_m, nodes, instance_id, output_path, scratch_dir):
    """Render the current isolated target, validate geometry, return full bool mask."""
    import bpy
    import numpy as np
    from mathutils import Vector
    from bpy_extras.object_utils import world_to_camera_view
    from data.action4_render import _mask_array
    width, height = scene.render.resolution_x, scene.render.resolution_y
    _dimensions(width, height)
    if scene.render.resolution_percentage != 100 or scene.render.pixel_aspect_x != 1 or scene.render.pixel_aspect_y != 1:
        raise ValueError('Amodal crop requires unscaled square-pixel full resolution')
    if camera.data.type != 'PERSP' or camera.data.shift_x != 0 or camera.data.shift_y != 0:
        raise ValueError('Amodal expanded-origin convention requires an unshifted perspective camera')
    if len(nodes) != 2 or instance_id not in (1, 2):
        raise ValueError('Two target-ID output nodes and a valid instance ID are required')
    output_path, scratch_dir = Path(output_path).resolve(), Path(scratch_dir).resolve()
    if not output_path.is_relative_to(ROOT) or not scratch_dir.is_relative_to(ROOT / 'outputs'):
        raise ValueError('Amodal outputs/scratch escape the project or outputs root')
    if output_path.exists() or scratch_dir.exists():
        raise FileExistsError('Amodal output and scratch must be new')
    matrix=np.array(core.matrix_world)
    projected=project_world(np.asarray(corners_m)@matrix[:3,:3].T+matrix[:3,3],scene)
    bounds = crop_bounds(projected, width, height)
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    counts = {}
    for part in parts:
        if part.type not in ('MESH', 'CURVE', 'SURFACE', 'FONT', 'META'):
            continue
        evaluated = part.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        try:
            values=np.empty(len(mesh.vertices)*3,dtype=np.float64)
            mesh.vertices.foreach_get("co",values);values=values.reshape(-1,3)
            matrix=np.array(evaluated.matrix_world)
            vertices=project_world(values@matrix[:3,:3].T+matrix[:3,3],scene)
            validate_geometry_coverage(vertices, bounds, width, height)
            counts[part.name] = len(vertices)
        finally:
            evaluated.to_mesh_clear()
    if not counts or core.name not in counts:
        raise ValueError('No evaluated camera body geometry was validated')
    scratch_dir.mkdir(parents=True, exist_ok=False)
    with temporary_border(scene, nodes, scratch_dir, bounds, width, height):
        bpy.ops.render.render(write_still=False)
        crop_path = scratch_dir / f'id{instance_id}_{scene.frame_current:04d}.png'
        crop = _mask_array(crop_path)
        full = pad_mask(crop, bounds, width, height)
        write_binary_png(output_path, full)
    return full, {'mode': 'crop', 'margin_px': MARGIN_PX, 'crop_bottomleft_xyxy': list(bounds),
                  'crop_shape_hw': list(crop.shape), 'full_shape_hw': [height, width],
                  'evaluated_vertex_counts': counts, 'geometry_coverage': 'all_evaluated_vertices_inside_crop_and_canvas',
                  'pixel_equivalence_policy': 'exact_binary_mask', 'samples': int(scene.cycles.samples)}