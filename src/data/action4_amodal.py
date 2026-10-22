"""Lossless optional amodal border rendering; imports never initialize Blender.

The full expanded camera canvas is unchanged. Pixel bounds use bottom-left
integer edges, while returned NumPy masks use top-left row order.
"""
from contextlib import contextmanager
import math
from pathlib import Path
import struct
import zlib

ROOT = Path(__file__).resolve().parents[2]
MARGIN_PX = 16


def validate_settings(settings):
    mode = settings.get('amodal_mode', 'full')
    if mode not in ('full', 'crop'):
        raise ValueError('amodal_mode must be full or crop')
    if 'amodal_margin_px' in settings and (type(settings['amodal_margin_px']) is not int or settings['amodal_margin_px'] != MARGIN_PX):
        raise ValueError('Only the validated fixed amodal margin of 16 pixels is supported')
    return mode


def _dimensions(width, height):
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise ValueError('Positive integer full-canvas dimensions required')


def crop_bounds(projected_corners_xyz, width, height):
    """Finite expanded-canvas x/y edge coordinates and positive camera depth."""
    import numpy as np
    _dimensions(width, height)
    points = np.asarray(projected_corners_xyz, dtype=float)
    if points.shape != (8, 3) or not np.isfinite(points).all():
        raise ValueError('Exactly eight finite projected corners required')
    if (points[:, 2] <= 0).any():
        raise ValueError('Target corner is behind the camera')
    if (points[:, :2] < 0).any() or (points[:, :2] > [width, height]).any():
        raise ValueError('Expanded canvas does not contain target corners')
    low = np.maximum(np.floor(points[:, :2].min(axis=0)).astype(int) - MARGIN_PX, 0)
    high = np.minimum(np.ceil(points[:, :2].max(axis=0)).astype(int) + MARGIN_PX, [width, height])
    bounds = tuple(map(int, (*low, *high)))
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        raise ValueError('Empty amodal crop')
    return bounds


def validate_geometry_coverage(projected_vertices_xyz, bounds, width, height):
    """Fail for detached parts outside the crop, even if its visible border is empty."""
    import numpy as np
    _dimensions(width, height)
    points = np.asarray(projected_vertices_xyz, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError('Nonempty finite evaluated target geometry required')
    if (points[:, 2] <= 0).any():
        raise ValueError('Evaluated target geometry is behind the camera')
    if (points[:, :2] < 0).any() or (points[:, :2] > [width, height]).any():
        raise ValueError('Expanded canvas does not contain evaluated target geometry')
    x0, y0, x1, y1 = bounds
    # One pixel of interior coverage avoids accepting a silhouette on the crop edge.
    if (points[:, :2] < [x0 + 1, y0 + 1]).any() or (points[:, :2] > [x1 - 1, y1 - 1]).any():
        raise ValueError('Evaluated target geometry extends outside the safe crop interior')


def pad_mask(crop, bounds, width, height):
    import numpy as np
    _dimensions(width, height)
    x0, y0, x1, y1 = bounds
    if any(type(v) is not int for v in bounds) or not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError('Invalid integer crop bounds')
    crop = np.asarray(crop)
    if crop.dtype != np.bool_ or crop.shape != (y1 - y0, x1 - x0):
        raise ValueError('Rendered crop dimensions or binary type differ from integer bounds')
    if crop[:1].any() or crop[-1:].any() or crop[:, :1].any() or crop[:, -1:].any():
        raise ValueError('Rendered target touches crop boundary')
    full = np.zeros((height, width), dtype=bool)
    full[height-y1:height-y0, x0:x1] = crop
    return full


def write_binary_png(path, mask):
    """Stream full-size 8-bit grayscale PNG, without any RGBA float canvas."""
    import numpy as np
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Amodal mask output escapes project root')
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError('PNG input must be a two-dimensional binary mask')
    height, width = mask.shape
    _dimensions(width, height)
    def chunk(stream, kind, data):
        stream.write(struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data) & 0xffffffff))
    with path.open('xb') as stream:
        stream.write(b'\x89PNG\r\n\x1a\n')
        chunk(stream, b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 0, 0, 0, 0))
        compressor = zlib.compressobj(6)
        for row in mask:
            data = compressor.compress(b'\0' + (row.astype('uint8') * 255).tobytes())
            if data:
                chunk(stream, b'IDAT', data)
        chunk(stream, b'IDAT', compressor.flush())
        chunk(stream, b'IEND', b'')


@contextmanager
def temporary_border(scene, nodes, scratch_dir, bounds, width, height):
    """Restore every altered render/node property even when render/decode raises."""
    render = scene.render
    names = ('use_border', 'use_crop_to_border', 'border_min_x', 'border_max_x', 'border_min_y', 'border_max_y')
    previous = {name: getattr(render, name) for name in names}
    node_previous = [(node, node.base_path, node.file_slots[0].path) for node in nodes]
    x0, y0, x1, y1 = bounds
    try:
        render.use_border = True
        render.use_crop_to_border = True
        # Blender stores float32 and floors pixel edges; 0.1 px avoids rounding
        # a requested integer N to N-1. At the canvas edge max is exactly 1.
        render.border_min_x = (x0 + .1) / width
        render.border_max_x = min(1., (x1 + .1) / width)
        render.border_min_y = (y0 + .1) / height
        render.border_max_y = min(1., (y1 + .1) / height)
        for index, node in enumerate(nodes, 1):
            node.base_path = str(scratch_dir)
            node.file_slots[0].path = f'id{index}_'
        yield
    finally:
        for name, value in previous.items():
            setattr(render, name, value)
        for node, base_path, slot_path in node_previous:
            node.base_path = base_path
            node.file_slots[0].path = slot_path


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
    def project(point):
        v = world_to_camera_view(scene, camera, point)
        return (float(v.x * width), float(v.y * height), float(v.z))
    projected = [project(core.matrix_world @ Vector(corner)) for corner in corners_m]
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
            vertices = [project(evaluated.matrix_world @ vertex.co) for vertex in mesh.vertices]
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