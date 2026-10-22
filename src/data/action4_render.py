"""Conservative Action 4 asset preparation and an explicit Blender CPU pilot.

Importing this module performs no I/O and does not import bpy.  Geometry keeps the
source STL's axes (X width, +Y rear, +Z top); only a documented translation is
applied.  The pilot is a rendering/data-contract check, not a validated training
corpus or a claim of metrically reconstructed scene appearance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys


def _project_path(path, *, exists=False):
    root = Path(__file__).resolve().parents[2]
    value = Path(path)
    value = (value if value.is_absolute() else root / value).resolve(strict=False)
    if not value.is_relative_to(root):
        raise ValueError("Render input/output escapes the project root")
    return value.resolve(strict=True) if exists else value


def _json(path, value):
    _project_path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha(path):
    return hashlib.sha256(_project_path(path, exists=True).read_bytes()).hexdigest()


def _read_stl(path):
    import numpy as np
    raw = _project_path(path, exists=True).read_bytes()
    if len(raw) < 84:
        raise ValueError("Binary STL header is incomplete")
    count = struct.unpack_from("<I", raw, 80)[0]
    if len(raw) != 84 + count * 50:
        raise ValueError("Only exact binary STL input is supported")
    dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
    records = np.frombuffer(raw, dtype=dtype, offset=84, count=count)
    if not np.isfinite(records["vertices"]).all():
        raise ValueError("Non-finite STL geometry")
    return records


def prepare_asset(source, asset_dir):
    """Write a new centered STL; preserve source and audit every removed face."""
    import numpy as np
    source = _project_path(source, exists=True)
    out = Path(asset_dir).resolve()
    if not out.is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError("Asset output must be inside this project")
    out.mkdir(parents=True, exist_ok=False)
    records = _read_stl(source)
    triangles = records["vertices"].astype(np.float64)
    lower, upper = triangles.min(axis=(0, 1)), triangles.max(axis=(0, 1))
    center = (lower + upper) / 2
    area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1) / 2
    vertices, inverse = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    canonical_faces = np.sort(faces, axis=1)
    _, unique_indices = np.unique(canonical_faces, axis=0, return_index=True)
    keep = area >= 1e-8
    unique_mask = np.zeros(len(faces), dtype=bool)
    unique_mask[unique_indices] = True
    keep &= unique_mask
    parent = np.arange(len(vertices))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return int(i)

    for face in faces:
        a, b, c = (find(int(x)) for x in face)
        root = min(a, b, c)
        parent[a] = parent[b] = parent[c] = root
    roots = np.array([find(i) for i in range(len(vertices))])
    _, component = np.unique(roots, return_inverse=True)
    face_components = component[faces[:, 0]][keep]
    normalized = vertices - center
    repaired_triangles = normalized[faces[keep]]
    new_records = np.zeros(keep.sum(), dtype=records.dtype)
    new_records["vertices"] = repaired_triangles
    normals = np.cross(repaired_triangles[:, 1] - repaired_triangles[:, 0], repaired_triangles[:, 2] - repaired_triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    new_records["normal"] = normals
    repaired = out / "Osmo_Action_4_repaired_v001.stl"
    header = b"Action4 v001: centered mm; near-zero-area/duplicate removal only".ljust(80, b" ")
    repaired.write_bytes(header + struct.pack("<I", int(keep.sum())) + new_records.tobytes())
    np.savez_compressed(out / "mesh_v001.npz", vertices_mm=normalized, faces=faces[keep], face_components=face_components)
    bounds = np.array([lower - center, upper - center])
    corners = np.array([[bounds[x, 0], bounds[y, 1], bounds[z, 2]] for x, y, z in [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]])
    transform = np.eye(4)
    transform[:3, 3] = -center
    manifest = {
        "asset_version": "action4_asset_v001", "status": "conservative_geometry_pilot",
        "source_path": str(source), "source_sha256": _sha(source), "source_bytes": source.stat().st_size,
        "source_triangle_count": len(faces), "triangle_count": int(keep.sum()), "unique_vertices": len(vertices),
        "component_count_before_cleanup": int(component.max() + 1), "units": "mm",
        "source_bounds_mm": [lower.tolist(), upper.tolist()], "bounds_mm": bounds.tolist(),
        "dimensions_mm": (upper - lower).tolist(), "source_to_object_mm": transform.tolist(),
        "corner_order": ["xmin_ymin_zmin", "xmax_ymin_zmin", "xmax_ymax_zmin", "xmin_ymax_zmin", "xmin_ymin_zmax", "xmax_ymin_zmax", "xmax_ymax_zmax", "xmin_ymax_zmax"],
        "corners_3d_mm": corners.tolist(), "axis_interpretation": {"x": "width", "+y": "rear screen", "+z": "top buttons"},
        "operations": [
            {"op": "translate", "vector_mm": (-center).tolist(), "reason": "Center coordinates; preserve dimensions and orientation"},
            {"op": "remove_numerically_degenerate_faces", "threshold_area_mm2": 1e-8, "source_face_indices": np.flatnonzero(area < 1e-8).tolist(), "removed_area_sum_mm2": float(area[area < 1e-8].sum()), "reason": "Faces have negligible area; no hole filling or inferred solid reconstruction"},
            {"op": "remove_duplicate_unordered_faces", "count": int((~unique_mask).sum())},
            {"op": "recompute_stl_face_normals", "reason": "Normals derived from preserved winding"},
        ],
        "geometry_changes_not_performed": ["hole filling", "nonuniform scaling", "smoothing geometry", "remeshing", "unverified buttons or structural details"],
        "limitations": ["Original non-watertight multi-part structure retained", "Material parameters are pilot approximations", "Capture camera intrinsics unknown"],
        "repaired_stl": repaired.name, "repaired_sha256": _sha(repaired),
        "mesh_data": "mesh_v001.npz", "mesh_data_sha256": _sha(out / "mesh_v001.npz"),
    }
    _json(out / "asset_manifest.json", manifest)
    print(json.dumps({"asset_dir": str(out), "removed_faces": int((~keep).sum()), "triangles": int(keep.sum())}))
    return manifest


def extract_reference_textures(source_frame, asset_dir):
    """Rectify explicitly recorded frame-0 appearance patches; these are not pose labels."""
    import numpy as np
    from PIL import Image
    source, asset = _project_path(source_frame, exists=True), _project_path(asset_dir, exists=True)
    if not asset.is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError("Texture output must be inside this project")
    im = Image.open(source).convert("RGB")
    if im.size != (1624, 1232):
        raise ValueError("Recorded evidence quadrilaterals require the exact half-resolution first frame")
    specs = {
        "rear_screen_reference": ((630, 372), [(1060, 577), (1352, 598), (1320, 696), (1060, 676)]),
        "rear_screen_left_reference": ((630, 372), [(612, 542), (820, 598), (776, 695), (583, 644)]),
        "top_id_reference": ((450, 120), [(1115, 525), (1239, 527), (1244, 550), (1111, 548)]),
    }
    evidence = {"source_frame": str(source), "source_frame_sha256": _sha(source), "frame_id": 0, "source_video_sha256": "016628eca2887220080d823f65d4f9c520ded71be363dbbbee95e175df7bd3e8", "patches": {},
        "source_frame_size_wh": [1624, 1232], "native_frame_size_wh": [3248, 2464],
        "quad_pixel_convention": "integer pixel centres; source quads refer to the exact half-resolution evidence frame",
        "source_to_native_pixel_transform": [[2, 0, 0.5], [0, 2, 0.5], [0, 0, 1]],
        "native_quad_status": "derived display reference via inverse half-size resize; not an independent native-frame annotation",
        "status": "Manually selected appearance-only quadrilaterals, not pose ground truth or capture-camera calibration"}
    for name, (size, quad) in specs.items():
        w, h = size; a, b = [], []
        for (x, y), (u, v) in zip([(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)], quad):
            a.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.append(u)
            a.append([0, 0, 0, x, y, 1, -v * x, -v * y]); b.append(v)
        coefficients = np.linalg.solve(a, b)
        result = im.transform(size, Image.Transform.PERSPECTIVE, coefficients, Image.Resampling.BICUBIC)
        destination = asset / (name + ".png")
        if destination.exists():
            if not np.array_equal(np.array(Image.open(destination)), np.array(result)):
                raise FileExistsError("Existing texture differs: " + str(destination))
        else:
            result.save(destination)
        evidence["patches"][name] = {"output_size": size, "source_quad_tl_tr_br_bl": quad, "native_quad_tl_tr_br_bl": [[2 * x + 0.5, 2 * y + 0.5] for x, y in quad], "output_to_source_homography8": coefficients.tolist(), "sha256": _sha(destination)}
    destination = asset / "appearance_evidence_v002.json"
    if destination.exists():
        raise FileExistsError("Evidence record exists")
    _json(destination, evidence)
    return evidence


def _bpy():
    import bpy
    return bpy


def _material(name, color, roughness=0.5, metallic=0.0):
    bpy = _bpy()
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    node = mat.node_tree.nodes.get("Principled BSDF")
    node.inputs["Base Color"].default_value = (*color, 1)
    node.inputs["Roughness"].default_value = roughness
    node.inputs["Metallic"].default_value = metallic
    node.inputs["Specular"].default_value = 0.24
    return mat


def _image_material(name, path, roughness=0.3, emission=0.05):
    bpy = _bpy()
    mat = _material(name, (0.1, 0.1, 0.1), roughness)
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    tex = mat.node_tree.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(str(path), check_existing=True)
    tex.image.pack()
    mat.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    mat.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Emission"])
    bsdf.inputs["Emission Strength"].default_value = emission
    return mat


def _core(asset_dir, name="Action4_core", pass_index=1):
    import numpy as np
    bpy = _bpy()
    data = np.load(Path(asset_dir) / "mesh_v001.npz", allow_pickle=False)
    vertices = data["vertices_mm"] * 0.001
    mesh = bpy.data.meshes.new(name + "_mesh")
    mesh.from_pydata(vertices.tolist(), [], data["faces"].tolist())
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.pass_index = pass_index
    mats = [_material("body_charcoal", (0.018, 0.021, 0.023), 0.58), _material("display_black", (0.007, 0.011, 0.014), 0.30), _material("lens_glass", (0.008, 0.012, 0.016), 0.14), _material("button_dark", (0.012, 0.014, 0.015), 0.50), _material("record_ring_muted_red", (0.10, 0.008, 0.005), 0.50)]
    rear_texture = Path(asset_dir) / ("rear_screen_left_reference.png" if pass_index == 2 else "rear_screen_reference.png")
    if rear_texture.exists():
        mats[1] = _image_material("rear_screen_video_f000000", rear_texture)
    for mat in mats:
        mesh.materials.append(mat)
    for polygon, comp in zip(mesh.polygons, data["face_components"]):
        if comp == 6:
            polygon.material_index = 1
        elif comp in (13, 19, 20, 24, 26, 28):
            polygon.material_index = 2
        elif comp == 10:
            polygon.material_index = 3
            midpoint = np.mean(vertices[data["faces"][polygon.index]], axis=0) * 1000
            radial = math.hypot(midpoint[0] + 18.09654, midpoint[1] - 2.56242)
            if 4.0 < radial < 5.2 and midpoint[2] > 21.1:
                polygon.material_index = 4
        polygon.use_smooth = True
    mesh.use_auto_smooth = True
    mesh.auto_smooth_angle = math.radians(30)
    uv = mesh.uv_layers.new(name="source_surface_uv")
    source_center = np.array(json.loads((Path(asset_dir) / "asset_manifest.json").read_text())["source_to_object_mm"])[:3, 3] * -1
    rear_min = np.array([93.29044, 121.11955, 2.1292]) - source_center
    rear_max = np.array([156.14404, 121.25043, 39.34881]) - source_center
    for polygon in mesh.polygons:
        for loop in polygon.loop_indices:
            vertex = vertices[mesh.loops[loop].vertex_index] * 1000
            uv.data[loop].uv = (1 - (vertex[0] - rear_min[0]) / (rear_max[0] - rear_min[0]), (vertex[2] - rear_min[2]) / (rear_max[2] - rear_min[2]))
    obj["asset_manifest"] = str(Path(asset_dir) / "asset_manifest.json")
    obj["geometry_units"] = "meters; derived from canonical mm by 0.001"
    obj["target_object_id"] = 1
    _top_decal(asset_dir, obj, pass_index)
    return obj


def _reset_scene(width, height, samples=12):
    bpy = _bpy()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 4
    scene.cycles.diffuse_bounces = 2
    scene.cycles.glossy_bounces = 2
    scene.render.threads_mode = "FIXED"
    blender_arguments = sys.argv[:sys.argv.index("--")] if "--" in sys.argv else []
    threads = 4
    for flag in ("--threads", "-t"):
        if flag in blender_arguments:
            threads = int(blender_arguments[blender_arguments.index(flag) + 1])
            if not 1 <= threads <= 1024:
                raise ValueError("The explicit CPU pilot thread budget must be positive (automatic thread count is not used)")
            break
    scene.render.threads = threads
    scene.render.resolution_x, scene.render.resolution_y = width, height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1
    scene.world = bpy.data.worlds.new("neutral_indoor_world")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs[0].default_value = (0.45, 0.46, 0.47, 1)
    scene.world.node_tree.nodes["Background"].inputs[1].default_value = 1.0
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    return scene


def _look_at(obj, point):
    from mathutils import Vector
    obj.rotation_euler = (Vector(point) - obj.location).to_track_quat("-Z", "Y").to_euler()


def _camera(location, target, orthographic=None):
    bpy = _bpy()
    data = bpy.data.cameras.new("capture_camera")
    obj = bpy.data.objects.new("capture_camera", data)
    bpy.context.collection.objects.link(obj)
    obj.location = location
    _look_at(obj, target)
    data.lens = 26
    data.sensor_width = 32
    data.sensor_fit = "HORIZONTAL"
    data.clip_start, data.clip_end = 0.001, 100
    if orthographic is not None:
        data.type = "ORTHO"
        data.ortho_scale = orthographic
    bpy.context.scene.camera = obj
    return obj


def _light(location, energy=40, size=0.8):
    bpy = _bpy()
    data = bpy.data.lights.new("large_neutral_area", "AREA")
    data.energy, data.size = energy, size
    obj = bpy.data.objects.new("large_neutral_area", data)
    bpy.context.collection.objects.link(obj)
    obj.location = location
    _look_at(obj, (0, 0, 0))
    return obj


def diagnose(asset_dir, output_dir, blend_name="Osmo_Action_4_appearance_v003.blend"):
    bpy = _bpy()
    out = Path(output_dir).resolve()
    project = Path(__file__).resolve().parents[2]
    if not out.is_relative_to(project) or not Path(asset_dir).resolve().is_relative_to(project):
        raise ValueError("Diagnostics and asset output must be inside this project")
    out.mkdir(parents=True, exist_ok=False)
    scene = _reset_scene(512, 512, 8)
    _core(asset_dir)
    camera = _camera((0, 0.3, 0.12), (0, 0, 0), 0.095)
    light = _light((0.05, 0.12, 0.18), 0.5, 0.2)
    views = {"front_minus_y": (0, -0.3, 0), "rear_plus_y": (0, 0.3, 0), "left_minus_x": (-0.3, 0, 0), "right_plus_x": (0.3, 0, 0), "top_plus_z": (0, 0, 0.3), "bottom_minus_z": (0, 0, -0.3)}
    for name, loc in views.items():
        camera.location = loc
        _look_at(camera, (0, 0, 0))
        light.location = (loc[0] + 0.1, loc[1] + 0.1, loc[2] + 0.15)
        _look_at(light, (0, 0, 0))
        scene.render.filepath = str(out / (name + ".png"))
        bpy.ops.render.render(write_still=True)
    camera.location = (0.10, 0.20, 0.10)
    _look_at(camera, (0, 0, 0))
    for filename in ("asset_manifest.json", "appearance_evidence_v002.json"):
        source = Path(asset_dir) / filename
        if source.exists():
            bpy.data.texts.new(filename).write(source.read_text())
    alternate = Path(asset_dir) / "rear_screen_left_reference.png"
    if alternate.exists():
        image = bpy.data.images.load(str(alternate), check_existing=True)
        image.pack(); image.use_fake_user = True
    bpy.ops.wm.save_as_mainfile(filepath=str(Path(asset_dir).resolve() / blend_name))
    _json(out / "diagnostic_manifest.json", {"views": views, "asset_manifest": str(Path(asset_dir).resolve() / "asset_manifest.json"), "renderer": bpy.app.version_string, "purpose": "Inspect original mesh geometry and axis assignments; appearance remains provisional"})


def _cube(name, center, dimensions, material, parent=None):
    bpy = _bpy()
    bpy.ops.mesh.primitive_cube_add(size=1, location=center)
    obj = bpy.context.object
    obj.name, obj.dimensions, obj.parent = name, dimensions, parent
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(material)
    return obj


def _ellipsoid(name, center, scale, material, parent=None):
    bpy = _bpy()
    bpy.ops.mesh.primitive_uv_sphere_add(segments=20, ring_count=12, radius=1, location=center)
    obj = bpy.context.object
    obj.name, obj.scale, obj.parent = name, scale, parent
    obj.data.materials.append(material)
    for face in obj.data.polygons:
        face.use_smooth = True
    return obj


def _tube(name, points, radius, material, parent=None):
    bpy = _bpy()
    curve = bpy.data.curves.new(name, "CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth, curve.bevel_resolution = radius, 3
    curve.use_fill_caps = True
    spline = curve.splines.new("BEZIER")
    spline.bezier_points.add(len(points) - 1)
    for point, coordinate in zip(spline.bezier_points, points):
        point.co = coordinate
        point.handle_left_type = point.handle_right_type = "AUTO"
    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    obj.parent = parent
    curve.materials.append(material)
    return obj


def _rig(asset_dir, name, position, rotation_degrees, skin_color, target_index=1):
    """Camera, mount, strap, cable, and approximate arm share one rigid wrist frame."""
    bpy = _bpy()
    rig = bpy.data.objects.new(name, None)
    bpy.context.collection.objects.link(rig)
    core = _core(asset_dir, name + "_core", target_index)
    core.parent = rig
    skin = _material(name + "_skin_proxy", skin_color, 0.78)
    black = _material(name + "_mount_black", (0.008, 0.009, 0.009), 0.68)
    strap = _material(name + "_strap_gray", (0.10, 0.13, 0.135), 0.90)
    green = _material(name + "_strap_edge_green", (0.18, 0.24, 0.015), 0.82)
    _ellipsoid(name + "_forearm_proxy", (0, -0.045, -0.21), (0.040, 0.032, 0.17), skin, rig)
    _ellipsoid(name + "_wrist_proxy", (0, -0.044, -0.055), (0.033, 0.026, 0.045), skin, rig)
    _ellipsoid(name + "_palm_proxy", (0, -0.047, 0.016), (0.038, 0.020, 0.053), skin, rig)
    for i, x in enumerate((-0.026, -0.009, 0.009, 0.025)):
        _tube(name + "_finger_proxy_" + str(i), [(x, -0.047, 0.055), (x, -0.039, 0.080), (x, -0.019, 0.085)], 0.0080, skin, rig)
    _tube(name + "_thumb_proxy", [(-0.027, -0.048, 0.016), (-0.050, -0.023, 0.029), (-0.047, -0.008, 0.047)], 0.010, skin, rig)
    bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=1, depth=1, location=(0, -0.044, -0.070))
    band = bpy.context.object
    band.name, band.scale, band.parent = name + "_strap_proxy", (0.037, 0.030, 0.034), rig
    band.data.materials.append(strap)
    for z in (-0.087, -0.053):
        points = [(0.037 * math.cos(a), -0.044 + 0.030 * math.sin(a), z) for a in [i * 2 * math.pi / 16 for i in range(17)]]
        _tube(name + "_strap_edge", points, 0.0012, green, rig)
    _cube(name + "_quick_release_proxy", (0, -0.003, -0.028), (0.045, 0.025, 0.009), black, rig)
    _ellipsoid(name + "_mount_ball_proxy", (0, -0.010, -0.042), (0.014, 0.014, 0.014), black, rig)
    _cube(name + "_mount_base_proxy", (0, -0.033, -0.056), (0.060, 0.022, 0.024), black, rig)
    _cube(name + "_cable_plug_proxy", (0.039, 0.003, -0.004), (0.008, 0.012, 0.026), black, rig)
    _tube(name + "_cable_proxy", [(0.040, 0.006, -0.010), (0.045, 0.013, -0.070), (0.044, -0.005, -0.15), (0.05, -0.016, -0.34)], 0.0017, black, rig)
    target_parts = [core, *core.children]
    rig.location = position
    rig.rotation_euler = tuple(math.radians(a) for a in rotation_degrees)
    rig["status"] = "coarse wrist/mount/cable proxy; not measured human or mount reconstruction"
    bpy.context.view_layer.update()
    return rig, core, target_parts


def _top_decal(asset_dir, parent, target_index):
    bpy = _bpy()
    name = parent.name
    label_path = Path(asset_dir) / "top_id_reference.png"
    if label_path.exists():
        label_mat = _image_material(name + "_top_id_video_f000000", label_path, 0.75, 0)
        mesh = bpy.data.meshes.new(name + "_top_id_mesh")
        mesh.from_pydata([(-0.003, -0.001, 0.02139), (0.026, -0.001, 0.02139), (0.026, 0.007, 0.02139), (-0.003, 0.007, 0.02139)], [], [(0, 1, 2, 3)])
        mesh.update()
        label = bpy.data.objects.new(name + "_top_id_decal", mesh)
        bpy.context.collection.objects.link(label)
        label.parent, label.pass_index = parent, target_index
        mesh.materials.append(label_mat)
        uv = mesh.uv_layers.new(name="decal_uv")
        for i, pair in enumerate(((1, 1), (0, 1), (0, 0), (1, 0))):
            uv.data[i].uv = pair
        return label
    return None


def _grid_material():
    bpy = _bpy()
    mat = _material("white_black_grid_tablecloth_proxy", (0.47, 0.47, 0.45), 0.90)
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    tex = nodes.new("ShaderNodeTexCoord")
    separate = nodes.new("ShaderNodeSeparateXYZ")
    links.new(tex.outputs["Object"], separate.inputs[0])
    edge_outputs = []
    for axis in ("X", "Y"):
        scale = nodes.new("ShaderNodeMath"); scale.operation = "MULTIPLY"; scale.inputs[1].default_value = 40
        fraction = nodes.new("ShaderNodeMath"); fraction.operation = "FRACT"
        edge = nodes.new("ShaderNodeMath"); edge.operation = "LESS_THAN"; edge.inputs[1].default_value = 0.04
        links.new(separate.outputs[axis], scale.inputs[0]); links.new(scale.outputs[0], fraction.inputs[0]); links.new(fraction.outputs[0], edge.inputs[0])
        edge_outputs.append(edge.outputs[0])
    maximum = nodes.new("ShaderNodeMath"); maximum.operation = "MAXIMUM"
    links.new(edge_outputs[0], maximum.inputs[0]); links.new(edge_outputs[1], maximum.inputs[1])
    mix = nodes.new("ShaderNodeMixRGB"); mix.inputs[1].default_value = (0.40, 0.42, 0.43, 1); mix.inputs[2].default_value = (0.008, 0.012, 0.015, 1)
    links.new(maximum.outputs[0], mix.inputs[0]); links.new(mix.outputs[0], nodes.get("Principled BSDF").inputs["Base Color"])
    return mat


def _denim_material():
    bpy = _bpy()
    mat = _material("denim_proxy", (0.045, 0.070, 0.11), 0.94)
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    noise = nodes.new("ShaderNodeTexNoise"); noise.inputs["Scale"].default_value = 160; noise.inputs["Detail"].default_value = 2
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.022, 0.039, 0.061, 1)
    ramp.color_ramp.elements[1].color = (0.09, 0.13, 0.18, 1)
    links.new(noise.outputs["Fac"], ramp.inputs[0]); links.new(ramp.outputs[0], nodes.get("Principled BSDF").inputs["Base Color"])
    return mat


def _cloth(name, center, size, material, phase=0):
    bpy = _bpy()
    nx, ny = 20, 24
    vertices, faces = [], []
    for j in range(ny):
        for i in range(nx):
            x, y = (i / (nx - 1) - 0.5) * size[0], (j / (ny - 1) - 0.5) * size[1]
            z = 0.005 * math.sin(40 * x + phase) * math.cos(23 * y) + 0.003 * math.sin(48 * y + 3 * phase)
            vertices.append((center[0] + x, center[1] + y, center[2] + z))
    for j in range(ny - 1):
        for i in range(nx - 1):
            k = j * nx + i
            faces.append((k, k + 1, k + 1 + nx, k + nx))
    mesh = bpy.data.meshes.new(name + "_mesh"); mesh.from_pydata(vertices, [], faces); mesh.update()
    obj = bpy.data.objects.new(name, mesh); bpy.context.collection.objects.link(obj)
    mesh.materials.append(material)
    for face in mesh.polygons:
        face.use_smooth = True
    return obj


def _environment(family, phase=0):
    wall = _material("gray_wall", (0.26, 0.275, 0.29), 0.98)
    floor = _material("dark_near_clothing", (0.005, 0.006, 0.008), 0.98)
    if family == "wall_rear":
        _cube("gray_wall", (0, -0.55, 0.4), (2.0, 0.01, 3.0), wall)
        return
    _cube("floor_proxy", (0, 0, -0.09), (3.0, 3.0, 0.01), _material("floor_gray_brown_proxy", (0.12, 0.10, 0.07), 0.96))
    _cube("grid_table", (0, 0.16, -0.006), (1.05, 0.78, 0.012), _grid_material())
    _cube("gray_wall", (0, 0.63, 0.7), (1.8, 0.01, 1.4), wall)
    _cloth("near_dark_clothing_proxy", (0, -0.47, -0.020), (0.65, 0.35), floor, phase)
    denim = _denim_material()
    _cloth("jacket_body_proxy", (0, 0.16, 0.010), (0.35, 0.45), denim, phase)
    for side in (-1, 1):
        _cloth("jacket_sleeve_proxy", (side * 0.245, 0.21, 0.013), (0.16, 0.20), denim, phase + side)
    button = _material("jacket_buttons_proxy", (0.27, 0.29, 0.28), 0.67)
    for i in range(6):
        _ellipsoid("jacket_button", (0.012, -0.015 + i * 0.063, 0.019), (0.007, 0.007, 0.002), button)


def _mask_nodes(out_dir):
    bpy = _bpy()
    scene = bpy.context.scene
    scene.view_layers[0].use_pass_object_index = True
    scene.use_nodes = True
    nodes = scene.node_tree.nodes; nodes.clear()
    layers = nodes.new("CompositorNodeRLayers")
    composite = nodes.new("CompositorNodeComposite")
    scene.node_tree.links.new(layers.outputs["Image"], composite.inputs[0])
    outputs = []
    for index in (1, 2):
        mask = nodes.new("CompositorNodeIDMask"); mask.index = index; mask.use_antialiasing = False
        scene.node_tree.links.new(layers.outputs["IndexOB"], mask.inputs[0])
        output = nodes.new("CompositorNodeOutputFile")
        output.base_path = str(out_dir)
        output.format.file_format = "PNG"; output.format.color_mode = "BW"; output.format.color_depth = "8"
        scene.node_tree.links.new(mask.outputs[0], output.inputs[0])
        outputs.append(output)
    return outputs


def _mask_array(path):
    import numpy as np
    bpy = _bpy()
    image = bpy.data.images.load(str(path), check_existing=False)
    width, height = image.size
    pixels = np.empty(width * height * 4, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    array = pixels.reshape(height, width, 4)[::-1, :, 0] > 0.5
    bpy.data.images.remove(image)
    return array


def render_pilot(asset_dir, output_dir, project_root, count=32, seed=20261018, start=0, stop=None):
    """Render a bounded unsplit pilot and exact synthetic projections/masks."""
    import numpy as np
    from mathutils import Vector
    from bpy_extras.object_utils import world_to_camera_view
    bpy = _bpy()
    if count != 32:
        raise ValueError("This explicit pilot recipe requires count=32 (5 wall / 24 tabletop / 3 transition); define a new recipe for another budget")
    root = Path(project_root).resolve(strict=True)
    out = Path(output_dir).resolve()
    if not out.is_relative_to(root):
        raise ValueError("Pilot output must be inside project root")
    out.mkdir(parents=True, exist_ok=True)
    (out / "rgb").mkdir(exist_ok=True); (out / "masks").mkdir(exist_ok=True); (out / "scenes").mkdir(exist_ok=True)
    manifest_path = out / "manifest.jsonl"
    if start == 0 and manifest_path.exists():
        raise FileExistsError("Use a new output directory; existing pilot is immutable")
    asset = json.loads((Path(asset_dir) / "asset_manifest.json").read_text())
    source_sha = _sha(__file__)
    source_snapshot = out / "render_source_snapshot.py"
    if source_snapshot.exists() and _sha(source_snapshot) != source_sha:
        raise ValueError("An append must use the same renderer source identity")
    if not source_snapshot.exists():
        source_snapshot.write_bytes(Path(__file__).read_bytes())
    corners = np.array(asset["corners_3d_mm"]) * 0.001
    width, height = 812, 616
    stop = min(count, stop if stop is not None else count)
    existing = {json.loads(line)["sample_id"] for line in manifest_path.read_text().splitlines()} if manifest_path.exists() else set()
    records = []
    for index in range(start, stop):
        rng = np.random.default_rng(seed + index)
        family = "wall_rear" if index < 5 else ("transition_edge" if index >= count - 3 else "tabletop_small")
        n_targets = 0 if index in (4, 19) else (1 if index in (3, 9, 15, 23, 29) else 2)
        scene = _reset_scene(width, height, 12)
        scene.cycles.seed = seed + index
        scene.view_settings.exposure = 0.20 if family == "wall_rear" else -0.90
        _environment(family, float(rng.uniform(-math.pi, math.pi)))
        if family == "wall_rear":
            camera = _camera((0, float(rng.uniform(0.33, 0.40)), 0.12), (0, 0, 0.01))
            _light((-0.35, 0.45, 0.65), 8.0, 1.0)
        else:
            camera = _camera((0, -0.33, 0.90), (0, 0.13, 0.06))
            _light((-0.28, -0.2, 1.05), 18.0, 1.1)
        targets = []
        target_objects = []
        scene_parameters = []
        for j in range(n_targets):
            side = -1 if j == 0 else 1
            if family == "wall_rear":
                position = (side * 0.095, float(rng.uniform(-0.020, 0.02)), float(rng.uniform(-0.015, 0.015)))
                rotation = (float(rng.uniform(-10, 10)), float(rng.uniform(-15, 15)), float(rng.uniform(-20, 20)))
            else:
                separation = 0.066 if index % 7 == 0 else 0.115
                position = (side * separation, float(rng.uniform(-0.17, -0.10)), float(rng.uniform(0.17, 0.23)))
                rotation = (float(rng.uniform(-108, -72)), side * float(rng.uniform(45, 92)), side * float(rng.uniform(10, 28)))
                if family == "transition_edge":
                    position = (side * 0.095, -0.240 if j == 0 else -0.285, 0.400 if j == 0 else 0.380)
                    rotation = (-55 + 20 * j, side * 60, side * 25)
            skin = (float(rng.uniform(0.18, 0.23)), float(rng.uniform(0.105, 0.145)), float(rng.uniform(0.065, 0.085)))
            rig, core, parts = _rig(asset_dir, "wrist_" + str(j + 1), position, rotation, skin, j + 1)
            targets.append((rig, core, parts)); target_objects.extend(parts)
            scene_parameters.append({"instance_id": j + 1, "wrist_position_m": position, "wrist_euler_xyz_degrees": rotation, "skin_proxy_linear_rgb": skin})
        bpy.context.view_layer.update()
        nodes = _mask_nodes(out / "masks")
        sample_id = out.name + "_" + str(index).zfill(6)
        if sample_id in existing:
            raise FileExistsError("Sample already exists: " + sample_id)
        for k, node in enumerate(nodes, 1):
            node.file_slots[0].path = sample_id + "_i" + str(k) + "_visible_"
        scene.render.filepath = str(out / "rgb" / (sample_id + ".png"))
        if index in (0, 5, count - 1):
            bpy.ops.wm.save_as_mainfile(filepath=str(out / "scenes" / (sample_id + ".blend")))
        bpy.ops.render.render(write_still=True)
        instances, excluded_instances = [], []
        projected_all = []
        for j, (rig, core, parts) in enumerate(targets, 1):
            projected, visible_corners = [], []
            depsgraph = bpy.context.evaluated_depsgraph_get()
            for point in corners:
                world = core.matrix_world @ Vector(point)
                uv = world_to_camera_view(scene, camera, world)
                pixel = [float(uv.x * width - 0.5), float((1 - uv.y) * height - 0.5)]
                if uv.z <= 0:
                    raise ValueError("Pilot pose places a corner behind the camera")
                projected.append(pixel)
                direction = world - camera.location
                distance = direction.length
                hit, location, *_ = scene.ray_cast(depsgraph, camera.location, direction.normalized(), distance=distance + 1e-6)
                los = not hit or (location - camera.location).length >= distance - 1e-5
                visible_corners.append(bool(los and -0.5 <= pixel[0] < width - 0.5 and -0.5 <= pixel[1] < height - 0.5))
            projected = np.array(projected)
            bbox = [float(projected[:, 0].min()), float(projected[:, 1].min()), float(projected[:, 0].max()), float(projected[:, 1].max())]
            projected_all.append((projected, bbox, visible_corners))
        original_hidden = {obj.name: obj.hide_render for obj in scene.objects}
        for j, ((rig, core, parts), (projected, bbox, visible_corners)) in enumerate(zip(targets, projected_all), 1):
            for obj in scene.objects:
                if obj.type not in ("CAMERA", "LIGHT"):
                    obj.hide_render = obj not in parts
            for k, node in enumerate(nodes, 1):
                node.file_slots[0].path = sample_id + "_i" + str(k) + "_amodal_for_" + str(j) + "_"
            scene.render.resolution_x, scene.render.resolution_y = 3 * width, 3 * height
            camera.data.sensor_width = 96
            scene.cycles.samples, scene.cycles.use_denoising = 1, False
            bpy.ops.render.render(write_still=False)
            visible_path = out / "masks" / (sample_id + "_i" + str(j) + "_visible_0001.png")
            full_path = out / "masks" / (sample_id + "_i" + str(j) + "_amodal_for_" + str(j) + "_0001.png")
            visible_mask, full_mask = _mask_array(visible_path), _mask_array(full_path)
            amodal_mask = full_mask[height:2 * height, width:2 * width]
            area_visible, area_amodal, area_full = int(visible_mask.sum()), int(amodal_mask.sum()), int(full_mask.sum())
            if not area_full:
                raise ValueError("No amodal pixels: invalid synthetic pose")
            if np.any(projected[:, 0] < -width - 0.5) or np.any(projected[:, 0] > 2 * width - 0.5) or np.any(projected[:, 1] < -height - 0.5) or np.any(projected[:, 1] > 2 * height - 0.5):
                raise ValueError("Expanded amodal canvas does not contain full cuboid")
            nonnested = int((visible_mask & ~amodal_mask).sum())
            raw_occlusion = float(1 - area_visible / area_amodal) if area_amodal else None
            raw_truncation = float(1 - area_amodal / area_full)
            instance = {"instance_id": j, "object_id": 1, "corners_px": projected.tolist(), "bbox_full_xyxy": bbox, "corner_visible": visible_corners,
                "mask_visible": str(visible_path.relative_to(root)), "mask_amodal_expanded": str(full_path.relative_to(root)),
                "amodal_expanded_origin_px": [-width, -height], "amodal_expanded_shape_hw": [3 * height, 3 * width],
                "area_visible_px": area_visible, "area_inframe_amodal_px": area_amodal, "area_full_amodal_px": area_full,
                "occlusion_inframe": float(max(0, min(1, raw_occlusion))) if raw_occlusion is not None else None,
                "occlusion_inframe_valid": bool(area_amodal), "occlusion_inframe_raw": raw_occlusion,
                "truncation": float(max(0, min(1, raw_truncation))), "truncation_raw": raw_truncation,
                "mask_nonnested_boundary_pixels": nonnested, "corner_visibility_semantics": "line of sight to virtual cuboid corner; not a physical surface landmark",
                "object_to_world_m": [list(row) for row in core.matrix_world]}
            yy, xx = np.nonzero(visible_mask)
            if len(xx):
                instance["bbox_visible_xyxy"] = [float(xx.min() - 0.5), float(yy.min() - 0.5), float(xx.max() + 0.5), float(yy.max() + 0.5)]
            if area_amodal:
                instances.append(instance)
            else:
                instance["exclusion_reason"] = "full amodal object is outside the image; not a valid image target"
                excluded_instances.append(instance)
        for obj in scene.objects:
            obj.hide_render = original_hidden[obj.name]
        scene.render.resolution_x, scene.render.resolution_y = width, height
        camera.data.sensor_width = 32
        record = {"schema_version": "action4.v1", "sample_id": sample_id, "image": str((out / "rgb" / (sample_id + ".png")).relative_to(root)), "width": width, "height": height,
            "group_id": sample_id, "background_source_id": "procedural_" + family + "_v001", "scene_family": family, "instances": instances,
            "excluded_instances": excluded_instances, "rendered_target_count": n_targets, "target_count": len(instances),
            "status": "diagnostic_pilot_not_training_accepted", "seed": seed + index, "renderer": "Blender " + bpy.app.version_string + " Cycles CPU",
            "render_code_sha256": source_sha,
            "asset_version": asset["asset_version"], "source_stl_sha256": asset["source_sha256"],
            "render_settings": {"rgb_samples": 12, "amodal_samples": 1, "cpu_threads": int(scene.render.threads), "exposure_stops": 0.20 if family == "wall_rear" else -0.90, "area_light_energy_W": 8 if family == "wall_rear" else 18, "area_light_size_m": 1.0 if family == "wall_rear" else 1.1, "world_strength": 1.0, "world_linear_rgb": [0.45, 0.46, 0.47], "photometry_status": "composition-level pilot matching; not recovered physical illumination"},
            "camera": {"calibration_status": "synthetic design only; not calibrated capture camera", "pixel_convention": "integer centers", "K": [[26 / 32 * width, 0, (width - 1) / 2], [0, 26 / 32 * width, (height - 1) / 2], [0, 0, 1]], "camera_to_world_m": [list(row) for row in camera.matrix_world]},
            "scene_parameters": scene_parameters, "evidence_regime": {"wall_rear": "source video 0–3s", "tabletop_small": "source video ~5–92s including 40.040s and 66.066s", "transition_edge": "source video ~93–95.062s"}[family],
            "limitations": ["Coarse procedural arm, strap, cable, garment and room proxies", "Appearance and camera model require further matching", "No fabric occluder inserted over camera in this initial pilot", "Unsplit development pilot", "Different sample counts for RGB and amodal mask rendering; boundary mismatch recorded"]}
        with manifest_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        records.append(record)
        print("PILOT_SAMPLE_COMPLETE", sample_id, "instances", len(instances), flush=True)
    _json(out / ("render_batch_" + str(start).zfill(6) + ".json"), {"range": [start, stop], "seed": seed, "count": len(records), "status": "rendered_pending_visual_and_adapter_validation", "manifest": str(manifest_path), "code_sha256": source_sha})
    return records


def main():
    arguments = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--source", required=True)
    prep.add_argument("--asset-dir", required=True)
    textures = sub.add_parser("textures")
    textures.add_argument("--source-frame", required=True)
    textures.add_argument("--asset-dir", required=True)
    diag = sub.add_parser("diagnose")
    diag.add_argument("--asset-dir", required=True)
    diag.add_argument("--output-dir", required=True)
    pilot = sub.add_parser("pilot")
    pilot.add_argument("--asset-dir", required=True)
    pilot.add_argument("--output-dir", required=True)
    pilot.add_argument("--project-root", required=True)
    pilot.add_argument("--count", type=int, default=32)
    pilot.add_argument("--seed", type=int, default=20261018)
    pilot.add_argument("--start", type=int, default=0)
    pilot.add_argument("--stop", type=int)
    args = parser.parse_args(arguments)
    if args.command == "prepare":
        prepare_asset(args.source, args.asset_dir)
    elif args.command == "textures":
        extract_reference_textures(args.source_frame, args.asset_dir)
    elif args.command == "diagnose":
        diagnose(args.asset_dir, args.output_dir)
    elif args.command == "pilot":
        render_pilot(args.asset_dir, args.output_dir, args.project_root, args.count, args.seed, args.start, args.stop)


if __name__ == "__main__":
    main()
