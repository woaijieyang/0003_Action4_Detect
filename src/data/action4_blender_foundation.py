"""Load the supplied editable R4 body into the existing R3 scene generator."""
from pathlib import Path
import hashlib
import json

from .action4_paths import PROJECT_ROOT, contained_path


def load_r4_core(asset_dir, foundation, name, pass_index):
    import bpy
    import numpy as np
    from mathutils import Matrix
    asset_dir = contained_path(PROJECT_ROOT, asset_dir, exists=True)
    foundation = contained_path(PROJECT_ROOT, foundation, exists=True)
    requested = ["Action4_core", "Action4_core_top_id_decal"]
    with bpy.data.libraries.load(str(foundation), link=False) as (source, loaded):
        if any(n not in source.objects for n in requested):
            raise ValueError("R4 lacks the canonical body or attached top decal")
        loaded.objects = requested
        loaded.images = [n for n in source.images if n != "Render Result"]
        loaded.texts = ["Action4_canonical_contract.json"]
    contract = json.loads(loaded.texts[0].as_string())
    expected = json.loads((asset_dir / "asset_manifest.json").read_text())
    if contract["corner_order"] != expected["corner_order"] or not np.allclose(
            contract["corners_3d_mm"], expected["corners_3d_mm"], atol=1e-5, rtol=0):
        raise ValueError("R4 corner contract differs from existing canonical labels")
    images = {}
    for image in loaded.images:
        # Appended datablocks may have Blender numeric suffixes; packed bytes
        # bind identities without guessing by a display name.
        if image.packed_file is None:
            raise ValueError("R4 references an unpacked image")
        digest = hashlib.sha256(bytes(image.packed_file.data)).hexdigest()
        matches = [n for n,v in contract["packed_images"].items() if v["sha256"] == digest]
        if len(matches) != 1:
            raise ValueError("R4 packed image differs from its source contract")
        images[matches[0]] = image
    if set(images) != set(contract["packed_images"]):
        raise ValueError("R4 packed image coverage is incomplete")
    core, decal = loaded.objects
    world = core.matrix_world.copy()
    decal_world = decal.matrix_world.copy()
    core.parent = None
    core.matrix_world = world
    bpy.context.collection.objects.link(core)
    bpy.context.collection.objects.link(decal)
    decal.parent = core
    decal.matrix_world = decal_world
    core.name, decal.name = name, name + "_top_id_decal"
    core.pass_index = decal.pass_index = pass_index
    bpy.context.view_layer.update()
    evaluated = core.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        vertices = np.array([list(core.matrix_world @ v.co) for v in mesh.vertices])
        faces = np.array([list(p.vertices) for p in mesh.polygons])
    finally:
        evaluated.to_mesh_clear()
    with np.load(asset_dir / "mesh_v001.npz", allow_pickle=False) as canonical:
        if vertices.shape != canonical["vertices_mm"].shape or not np.allclose(
                vertices, canonical["vertices_mm"] * .001, atol=1e-8, rtol=0):
            raise ValueError("R4 evaluated vertices differ from canonical R3 geometry")
        if not np.array_equal(faces, canonical["faces"]):
            raise ValueError("R4 topology differs from canonical R3 geometry")
    # Keep the original per-instance rear appearance selection, while retaining
    # R4 material parameters/UVs. Copy material nodes before editing instance 2.
    if pass_index == 2:
        material = core.data.materials[1].copy()
        core.data.materials[1] = material
        for node in material.node_tree.nodes:
            if node.type == "TEX_IMAGE":
                node.image = images["rear_screen_left_reference.png"]
    core["asset_backend"] = "blend_r4"
    core["foundation_path"] = str(foundation.relative_to(PROJECT_ROOT))
    core["foundation_sha256"] = hashlib.sha256(foundation.read_bytes()).hexdigest()
    core["target_object_id"] = 1
    core["geometry_units"] = "meters; canonical mm * 0.001"
    return core

def apply_photo_material(material, texture_path, tile_m, *, strength=1.0):
    """Use a real source-photo patch as color, without inventing BRDF/normal data."""
    import bpy
    path = contained_path(PROJECT_ROOT, texture_path, exists=True)
    nodes, links = material.node_tree.nodes, material.node_tree.links
    shader = nodes.get("Principled BSDF")
    coords = nodes.new("ShaderNodeTexCoord")
    scale = nodes.new("ShaderNodeVectorMath"); scale.operation = "SCALE"
    scale.inputs[3].default_value = 1.0 / tile_m
    links.new(coords.outputs["Object"], scale.inputs[0])
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(str(path), check_existing=True)
    tex.image.colorspace_settings.name = "sRGB"; tex.image.pack()
    tex.projection = "BOX"; tex.projection_blend = .25
    links.new(scale.outputs["Vector"], tex.inputs["Vector"])
    mix = nodes.new("ShaderNodeMixRGB"); mix.blend_type = "MIX"
    mix.inputs[0].default_value = strength
    mix.inputs[1].default_value = shader.inputs["Base Color"].default_value
    links.new(tex.outputs["Color"], mix.inputs[2])
    links.new(mix.outputs[0], shader.inputs["Base Color"])
    material["photo_texture_source"] = str(path.relative_to(PROJECT_ROOT))
    material["tile_m_design_prior"] = tile_m
    material["appearance_only_not_measured_brdf"] = True


def apply_denim_material(material, manifest_path):
    """Official CC0 color/roughness/OpenGL normal maps on explicit metric UVs."""
    import bpy
    manifest_path = contained_path(PROJECT_ROOT, manifest_path, exists=True)
    manifest = json.loads(manifest_path.read_text())
    nodes, links = material.node_tree.nodes, material.node_tree.links
    shader = nodes.get("Principled BSDF")
    textures = {}
    for key in ("diff", "rough", "nor_gl"):
        spec = manifest["files"][key]
        path = contained_path(PROJECT_ROOT, spec["path"], exists=True)
        if hashlib.sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
            raise ValueError("Texture bytes changed after provenance freeze")
        node = nodes.new("ShaderNodeTexImage")
        node.image = bpy.data.images.load(str(path), check_existing=True)
        node.image.colorspace_settings.name = spec["color_space"]
        node.image.pack()
        textures[key] = node
    links.new(textures["diff"].outputs["Color"], shader.inputs["Base Color"])
    links.new(textures["rough"].outputs["Color"], shader.inputs["Roughness"])
    normal = nodes.new("ShaderNodeNormalMap"); normal.space = "TANGENT"
    normal.inputs["Strength"].default_value = .5
    links.new(textures["nor_gl"].outputs["Color"], normal.inputs["Color"])
    links.new(normal.outputs["Normal"], shader.inputs["Normal"])
    material["texture_manifest"] = str(manifest_path.relative_to(PROJECT_ROOT))
    material["normal_convention"] = "OpenGL +Y tangent; UV x/y in meters / .1"
    material["displacement_disabled"] = True

def join_wrist_skin(hand, forearm):
    """Union only generic skin; keep target body/corners/material IDs unchanged."""
    import bpy
    bpy.ops.object.select_all(action="DESELECT")
    hand.select_set(True); bpy.context.view_layer.objects.active = hand
    for modifier in list(hand.modifiers):
        bpy.ops.object.modifier_apply(modifier=modifier.name)
    union = hand.modifiers.new("continuous_wrist_surface", "BOOLEAN")
    union.operation = "UNION"; union.solver = "EXACT"; union.object = forearm
    bpy.ops.object.modifier_apply(modifier=union.name)
    group = hand.vertex_groups.new(name="wrist_transition_only")
    for vertex in hand.data.vertices:
        weight = max(0.0, 1.0 - abs(vertex.co.z + .113) / .035)
        if weight:
            group.add([vertex.index], weight, "REPLACE")
    smooth = hand.modifiers.new("wrist_transition_smoothing", "SMOOTH")
    smooth.factor = .8; smooth.iterations = 20; smooth.vertex_group = group.name
    bpy.ops.object.modifier_apply(modifier=smooth.name)
    forearm.hide_render = True
    hand["wrist_join"] = "Boolean union and local smoothing z=-0.148..-0.078m"
    hand["target_geometry_changed"] = False

def apply_camera_head_pose(rig, core, name, rotation_xyz_deg):
    """Rotate the existing camera/quick-release assembly about its ball joint."""
    import bpy
    from math import radians, isfinite
    from mathutils import Vector
    angles = [float(x) for x in rotation_xyz_deg]
    if len(angles) != 3 or any(not isfinite(x) or abs(x) > 120 for x in angles):
        raise ValueError("Camera head design-prior angle exceeds bounded joint range")
    if not any(angles):
        return
    group = bpy.data.objects.new(name + "_camera_head_pose", None)
    bpy.context.collection.objects.link(group); group.parent = rig
    group.location = (0, -.010, -.041)
    bpy.context.view_layer.update()
    selected = [] if core is None else [core]
    selected.extend(o for o in list(rig.children)
                    if o.name.startswith(name + "_quick_release") or o.name.startswith(name + "_cable_connector"))
    for obj in selected:
        world = obj.matrix_world.copy(); obj.parent = group; obj.matrix_world = world
    group.rotation_euler = tuple(radians(x) for x in angles)
    bpy.context.view_layer.update()
    # Cable root follows its real connector; the remaining route stays attached
    # to this wrist/forearm instead of rotating the entire cable into free space.
    cable = next((o for o in rig.children if o.name == name + "_attached_cable"), None)
    if cable and core is not None:
        endpoint = rig.matrix_world.inverted() @ (core.matrix_world @ Vector((.040, .003, -.009)))
        points = cable.data.splines[0].bezier_points
        points[0].co = endpoint
        points[1].co = endpoint + Vector((.012, -.025, -.035))
    group["pose_claim"] = "bounded appearance prior about existing mount ball; not measured real pose"

def fold_existing_garment(garments, state):
    """Finite folded variants of the existing garment; no cloth simulation claim."""
    import bpy, math
    if state not in ("laid_open", "left_sleeve_fold", "right_sleeve_fold", "folded_lift"):
        raise ValueError("Unsupported finite garment appearance state")
    if state == "laid_open": return
    def folded(point):
        x,y,z=point
        use_left=state in ("left_sleeve_fold","folded_lift")
        use_right=state in ("right_sleeve_fold","folded_lift")
        if (x < -.18 and use_left) or (x > .18 and use_right):
            sign=-1 if x<0 else 1; distance=abs(x)-.18;angle=math.radians(178 if state=="folded_lift" else 155)
            x=sign*(.18+math.cos(angle)*distance)
            z+=math.sin(angle)*distance+.006*min(1.,distance/.08)
        if state == "folded_lift" and y>.18:
            distance=y-.18;angle=math.radians(180)
            y=.18+math.cos(angle)*distance
            z+=math.sin(angle)*distance+.015*min(1.,distance/.05)
        return (x,y,z)
    for garment in garments:
        garment["contact_rest_coordinates"]=[c for v in garment.data.vertices for c in v.co]
        for vertex in garment.data.vertices:vertex.co=folded(vertex.co)
        garment["garment_state"]=state
    for obj in bpy.context.scene.objects:
        if obj.type=="CURVE" and "boundary_stitch" in obj.name:
            original=[]
            for spline in obj.data.splines:
                for point in spline.bezier_points:
                    original.extend(point.co);point.co=folded(point.co)
            obj["contact_rest_coordinates"]=original
        elif obj.name.startswith("jacket_metal_button"):
            obj["contact_rest_location"]=list(obj.location);obj.location=folded(obj.location)
    bpy.context.view_layer.update()
