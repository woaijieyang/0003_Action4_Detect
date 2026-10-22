"""G1 fixed metric anatomy/brace/camera foundation. INITIALIZATION, never accepted data."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT=Path("/data/zy/vln/a-test")
SCHEMA="action4.global_rig_init.v1"
JOINTS=tuple(f"{side}_{part}" for side in ("left","right") for part in ("upper_arm","forearm","hand"))
TOP_KEYS={"schema_version","status","dataset_generation_allowed","architecture","body","camera","brace","attachment","pose_limit_degrees","validation_poses"}


def bounded(path,exists=False):
    p=(ROOT/path).resolve(strict=exists)
    if not p.is_relative_to(ROOT):raise ValueError("Path leaves project")
    return p


def sha(path):
    h=hashlib.sha256()
    with bounded(path,True).open("rb") as f:
        for chunk in iter(lambda:f.read(1048576),b""):h.update(chunk)
    return h.hexdigest()


def validate_config(cfg,check_files=True):
    if set(cfg)!=TOP_KEYS or cfg.get("schema_version")!=SCHEMA:
        raise ValueError("Exact global rig schema required; per-scenario overrides/scales forbidden")
    if cfg.get("status")!="initialization_pending_fit" or cfg.get("dataset_generation_allowed") is not False:
        raise ValueError("G1 cannot be marked fitted or generation-accepted")
    if set(cfg["body"])!={"path","sha256","object","collection","source_scale","region_ids","side_sign"}:
        raise ValueError("Unexpected anatomy parameters")
    if type(cfg["body"]["source_scale"]) not in (int,float) or cfg["body"]["source_scale"]!=1.0 or cfg["body"]["side_sign"]!={"left":1,"right":-1}:
        raise ValueError("Preserve source anatomy scale and explicit side map")
    if set(cfg["camera"])!={"asset_dir","asset_manifest","foundation"}:
        raise ValueError("Camera scale or per-scenario camera overrides forbidden")
    if set(cfg["body"]["region_ids"])!={"shoulder","upper_arm","forearm","palm","digits"}:
        raise ValueError("Unexpected source region mapping")
    if set(cfg["brace"])!={"proximal_offset_m","width_m","clearance_m","thickness_m","radial_reference_world"}:
        raise ValueError("Unexpected brace overrides")
    if set(cfg["attachment"])!={"camera_rotation_xyz_deg","support_height_m","support_width_m","support_depth_m","claim"}:
        raise ValueError("Unexpected attachment overrides")
    for key in ("proximal_offset_m","width_m","clearance_m","thickness_m"):
        x=cfg["brace"][key]
        if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or not 0<x<.1:raise ValueError("Invalid brace metric")
    radial=cfg["brace"]["radial_reference_world"]
    if not isinstance(radial,list) or len(radial)!=3 or any(type(v) not in (int,float) or not math.isfinite(v) for v in radial) or sum(v*v for v in radial)<1e-8:
        raise ValueError("Brace radial reference must be a nonzero finite3-vector")
    angles=cfg["attachment"]["camera_rotation_xyz_deg"]
    if not isinstance(angles,list) or len(angles)!=3 or any(type(v) not in (int,float) or not math.isfinite(v) for v in angles):
        raise ValueError("One finite fixed attachment rotation required")
    for key in ("support_height_m","support_width_m","support_depth_m"):
        value=cfg["attachment"][key]
        if type(value) not in (int,float) or not math.isfinite(value) or not 0<value<.1:
            raise ValueError("Initial mount dimensions must be finite positive metric values")
    if cfg["attachment"]["claim"]!="source_geometry_initialization_not_video_fit":
        raise ValueError("Attachment evidence scope invalid")
    if cfg["pose_limit_degrees"]!=25:
        raise ValueError("G1 uses a fixed bounded engineering test envelope, not anatomical range")
    if check_files:
        for ref in (cfg["architecture"],cfg["body"],cfg["camera"]["foundation"],cfg["camera"]["asset_manifest"]):
            if sha(ref["path"])!=ref["sha256"]:raise ValueError("Input identity changed")
        architecture=json.loads(bounded(cfg["architecture"]["path"],True).read_text())
        if architecture["units"]!="metres" or not architecture["metric_anchor"]["fixed_across_scenarios"]:
            raise ValueError("Architecture metric anchor differs")
    for pose in cfg["validation_poses"]:validate_pose(pose["joints"],cfg["pose_limit_degrees"])
    return cfg


def validate_pose(pose,limit=25):
    if not isinstance(pose,dict) or set(pose)-set(JOINTS):
        raise ValueError("Pose accepts joint rotations only, never scale/length/mount/family overrides")
    for angles in pose.values():
        if not isinstance(angles,list) or len(angles)!=3 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or abs(v)>limit for v in angles):
            raise ValueError("Pose outside bounded G1 engineering envelope")


def invariant_contract(report):
    """Immutable G1 input for later fit/sampling; no calibrated ranges implied."""
    if report["status"]!="initialization_pending_fit" or report["dataset_generation_allowed"] is not False:
        raise ValueError("G1 report cannot claim fitted/accepted scene")
    value={"schema_version":"action4.global_rig_contract.v1","status":"initialization_pending_fit",
           "dataset_generation_allowed":False,"units":"metres","asset":report["saved_blend"],
           "source_body":report["source_inputs"]["body"],"source_camera":report["source_inputs"]["camera"],
           "source_geometry_sha256":report["source_geometry_sha256"],
           "fixed_camera_mesh_xyz_m":report["camera_current_mesh_xyz_m"],
           "source_body_scale":1.0,"camera_object_scale":[1,1,1],
           "rest_bone_lengths_m":report["rest_lengths_m"],"joint_proxy_definitions":report["joint_proxy_definitions"],
           "fixed_brace_and_camera_attachment":report["brace_and_fixed_attachment"],
           "pose_input":{"joint_names":list(JOINTS),"unit":"degrees","representation":"localXYZEuler",
                         "engineering_validation_limit_degrees":25,"calibrated_sampling_ranges":None,
                         "allow_scale_mount_or_bone_length_override":False},
           "observer_intrinsics":None,"body_and_mount_fit_status":"pending_multiframe_video_fit"}
    value["invariants_sha256"]=hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    return value


def _write(path,value):
    p=bounded(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("x") as f:json.dump(value,f,indent=2,allow_nan=False)


def _mesh(name,verts,faces,parent,material):
    import bpy
    mesh=bpy.data.meshes.new(name);mesh.from_pydata(verts,[],faces);mesh.update()
    obj=bpy.data.objects.new(name,mesh);bpy.context.collection.objects.link(obj);obj.parent=parent
    if material:mesh.materials.append(material)
    return obj


def _box(name,center,size,parent,mat):
    import itertools
    verts=[tuple(center[a]+sign[a]*size[a]/2 for a in range(3)) for sign in itertools.product((-1,1),repeat=3)]
    return _mesh(name,verts,[(0,1,3,2),(4,6,7,5),(0,4,5,1),(2,3,7,6),(0,2,6,4),(1,5,7,3)],parent,mat)


def _smooth(x):
    x=max(0.,min(1.,float(x)));return x*x*(3-2*x)


def apply_pose(armature,pose,config):
    import bpy
    validate_pose(pose,config["pose_limit_degrees"])
    for name in JOINTS:
        bone=armature.pose.bones[name];bone.rotation_mode="XYZ"
        bone.rotation_euler=tuple(math.radians(v) for v in pose.get(name,[0,0,0]))
        if tuple(bone.scale)!=(1.,1.,1.):raise ValueError("Bone scaling forbidden")
    bpy.context.view_layer.update()


def _evaluated_vertices(obj):
    import bpy,numpy as np
    ob=obj.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=ob.to_mesh()
    try:return np.array([list(obj.matrix_world@v.co) for v in mesh.vertices]),[tuple(p.vertices) for p in mesh.polygons]
    finally:ob.to_mesh_clear()


def _bvh(obj):
    from mathutils.bvhtree import BVHTree
    points,faces=_evaluated_vertices(obj)
    return BVHTree.FromPolygons(points.tolist(),faces,all_triangles=False)


def _inside(tree,point):
    from mathutils import Vector
    votes=[]
    for direction in [(1,.371,.193),(.213,1,.417),(.197,.329,1)]:
        d=Vector(direction).normalized();p=Vector(point);count=0
        for _ in range(128):
            hit,normal,index,distance=tree.ray_cast(p,d)
            if hit is None:break
            count+=1;p=hit+d*1e-6
        else:raise ValueError("Containment ray did not terminate")
        votes.append(count%2)
    if len(set(votes))!=1:raise ValueError("Ambiguous containment test")
    return bool(votes[0])


def build(config_path,output_dir,report_dir):
    import bpy,numpy as np
    from mathutils import Matrix,Vector,Euler
    from data.action4_blender_foundation import load_r4_core
    from data.action4_render import _material
    if Path.cwd().resolve()!=ROOT:raise ValueError("Use exact project root")
    cfg=validate_config(json.loads(bounded(config_path,True).read_text()))
    out,report=bounded(output_dir),bounded(report_dir)
    if out.exists():raise FileExistsError("Global rig asset versions are immutable")
    report.mkdir(parents=True,exist_ok=True)
    if (report/"report.json").exists():raise FileExistsError("Use fresh evidence directory")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene=bpy.context.scene;scene.unit_settings.system="METRIC";scene.unit_settings.scale_length=1
    scene.render.threads_mode="FIXED";scene.render.threads=2
    scene["status"]="initialization_pending_fit";scene["dataset_generation_allowed"]=False
    with bpy.data.libraries.load(str(bounded(cfg["body"]["path"],True)),link=False) as (src,dst):
        dst.objects=[cfg["body"]["object"]]
    body=dst.objects[0];bpy.context.collection.objects.link(body)
    gallery=[list(x) for x in body.matrix_world]
    body.parent=None;body.matrix_world=Matrix.Identity(4);body.name="global_source_anatomy"
    bpy.context.view_layer.update()
    verts=np.array([list(v.co) for v in body.data.vertices]);faces=[list(p.vertices) for p in body.data.polygons]
    geometry_digest=hashlib.sha256(verts.tobytes()+json.dumps(faces).encode()).hexdigest()
    sets={}
    attr=body.data.attributes.get(".sculpt_face_set")
    if attr is None:raise ValueError("Expected source surface region IDs")
    for polygon,datum in zip(body.data.polygons,attr.data):sets.setdefault(datum.value,set()).update(polygon.vertices)
    regions=cfg["body"]["region_ids"];joints={};proxy={}
    for side,sign in cfg["body"]["side_sign"].items():
        centers=[]
        for a,b in [(regions["shoulder"],regions["upper_arm"]),(regions["upper_arm"],regions["forearm"]),(regions["forearm"],regions["palm"])]:
            ids=sorted(i for i in sets[a]&sets[b] if verts[i,0]*sign>0)
            if len(ids)<3:raise ValueError("Missing source anatomical proxy boundary")
            center=verts[ids].mean(0);centers.append(center)
            proxy[f"{side}_{a}_{b}"]={"vertex_ids":ids,"center_m":center.tolist(),"claim":"surface_boundary_centroid_initial_proxy_not_validated_skeletal_joint"}
        palm_ids=[i for i in sets[regions["palm"]] if verts[i,0]*sign>0]
        palm=verts[palm_ids].mean(0)
        joints[side]=[*centers,palm]
    rigdata=bpy.data.armatures.new("global_metric_armature")
    rig=bpy.data.objects.new("global_metric_armature",rigdata);bpy.context.collection.objects.link(rig)
    bpy.context.view_layer.objects.active=rig;rig.select_set(True);bpy.ops.object.mode_set(mode="EDIT")
    root=rigdata.edit_bones.new("torso_root");root.head=(0,0,1);root.tail=(0,0,1.1)
    for side,points in joints.items():
        parent=root
        for i,part in enumerate(("upper_arm","forearm","hand")):
            bone=rigdata.edit_bones.new(f"{side}_{part}");bone.head=points[i];bone.tail=points[i+1]
            bone.parent=parent;bone.use_connect=i>0;bone.align_roll(Vector((0,-1,0)));parent=bone
    bpy.ops.object.mode_set(mode="OBJECT")
    for bone in rigdata.bones:bone.inherit_scale="NONE"
    body.parent=rig
    # The bundle object has no named groups but retains orphan deform indices.
    # Clear those explicitly in this NEW copy before assigning the new rig.
    source_group_names=[g.name for g in body.vertex_groups]
    source_weights=[(v.index,g.group,float(g.weight)) for v in body.data.vertices for g in v.groups]
    max_group=max((g for _,g,_ in source_weights),default=-1)
    if max_group>1024:raise ValueError("Unexpected source deform layer size")
    while len(body.vertex_groups)<=max_group:body.vertex_groups.new(name="unused_source_deform_index")
    indices=list(range(len(body.data.vertices)))
    for group in body.vertex_groups:group.remove(indices)
    body.vertex_groups.clear()
    if any(len(v.groups) for v in body.data.vertices):raise ValueError("Orphan source deform weights remain")
    discarded_drivers=[]
    if body.animation_data:
        if body.animation_data.action is not None:raise ValueError("Unexpected source body animation action")
        for driver in list(body.animation_data.drivers):
            if driver.data_path not in ('modifiers["mask_legs"].show_viewport','modifiers["mask_arms"].show_viewport'):
                raise ValueError("Unexpected source driver; requires explicit review")
            if any(m.name in ("mask_legs","mask_arms") for m in body.modifiers):
                raise ValueError("Source visibility driver has a real target; do not discard")
            discarded_drivers.append({"path":driver.data_path,"array_index":driver.array_index})
        body.animation_data_clear()
    source_deform_cleanup={"source_named_groups":source_group_names,"orphan_weight_records_removed":len(source_weights),
                           "source_deform_state_sha256":hashlib.sha256(json.dumps(source_weights).encode()).hexdigest(),
                           "removed_dangling_visibility_drivers":discarded_drivers,
                           "scope":"derived copy only; original bundle unchanged"}
    groups={name:body.vertex_groups.new(name=name) for name in ("torso_root",*JOINTS)}
    arm_regions=set([regions["shoulder"],regions["upper_arm"],regions["forearm"],regions["palm"],*regions["digits"]])
    arm_ids=set().union(*(sets.get(k,set()) for k in arm_regions))
    weight_totals={name:0. for name in groups}
    for i,v in enumerate(verts):
        weights={"torso_root":1.}
        if i in arm_ids:
            side="left" if v[0]>0 else "right";shoulder,elbow,wrist,palm=joints[side]
            a=(elbow-shoulder);a/=np.linalg.norm(a);b=(wrist-elbow);b/=np.linalg.norm(b);c=(palm-wrist);c/=np.linalg.norm(c)
            upper=_smooth((np.dot(v-shoulder,a)+.03)/.06)
            fore=_smooth((np.dot(v-elbow,(a+b)/np.linalg.norm(a+b))+.025)/.05)
            hand=_smooth((np.dot(v-wrist,c)+.018)/.036)
            weights={"torso_root":1-upper,f"{side}_upper_arm":upper*(1-fore)*(1-hand),
                     f"{side}_forearm":upper*fore*(1-hand),f"{side}_hand":upper*hand}
        if abs(sum(weights.values())-1)>1e-8:raise ValueError("Skin weight sum mismatch")
        for name,value in weights.items():
            if value>0:groups[name].add([i],float(value),"REPLACE");weight_totals[name]+=float(value)
    modifier=body.modifiers.new("bounded_source_anatomy_arm_deformation","ARMATURE");modifier.object=rig;modifier.use_deform_preserve_volume=True
    body["source_geometry_sha256"]=geometry_digest;body["skin_weights_scope"]="initial surface-region/axial blending, not fitted production skinning"
    body["fingers"]="source natural rest; move rigidly with hand bone; no old scalar curl or .78 scaling"
    gray=_material("G1_brace_initial_gray",(.12,.13,.14),.8)
    black=_material("G1_mount_initial_black",(.01,.01,.01),.6)
    braces={};cores={};attachments={};brace_info={}
    camera_manifest=json.loads(bounded(cfg["camera"]["asset_manifest"]["path"],True).read_text())
    dims=np.array(camera_manifest["dimensions_mm"])*.001
    for side,sign in cfg["body"]["side_sign"].items():
        shoulder,elbow,wrist,palm=joints[side];y=(wrist-elbow)/np.linalg.norm(wrist-elbow)
        z=np.array(cfg["brace"]["radial_reference_world"],float);z-=y*np.dot(z,y);z/=np.linalg.norm(z)
        x=np.cross(y,z);x/=np.linalg.norm(x);basis=np.column_stack([x,y,z])
        if abs(np.linalg.det(basis)-1)>1e-8:raise ValueError("Improper brace basis")
        origin=wrist-y*cfg["brace"]["proximal_offset_m"]
        local=(verts-origin)@basis;points=[]
        for face,datum in zip(faces,attr.data):
            if datum.value!=regions["forearm"]:continue
            if not all(verts[i,0]*sign>0 for i in face):continue
            for a,b in zip(face,face[1:]+face[:1]):
                va,vb=local[a],local[b]
                if va[1]*vb[1]<=0 and abs(vb[1]-va[1])>1e-12:
                    t=-va[1]/(vb[1]-va[1]);points.append(va+t*(vb-va))
        if len(points)<8:raise ValueError("Brace plane lacks actual forearm contour")
        points=np.array(points);mid=(points.min(0)+points.max(0))/2;mid[1]=0
        origin+=basis@mid;points-=mid
        rx=max(abs(points[:,0]))+cfg["brace"]["clearance_m"]
        rz=max(abs(points[:,2]))+cfg["brace"]["clearance_m"]
        transform=Matrix.Identity(4)
        for row in range(3):
            for col in range(3):transform[row][col]=float(basis[row,col])
            transform[row][3]=float(origin[row])
        brace=bpy.data.objects.new(side+"_brace_frame",None);bpy.context.collection.objects.link(brace)
        brace.parent=rig;brace.parent_type="BONE";brace.parent_bone=side+"_forearm"
        bpy.context.view_layer.update();brace.matrix_world=transform;bpy.context.view_layer.update()
        if np.max(abs(np.array(brace.matrix_world)-np.array(transform)))>1e-6:raise ValueError("Bone parent brace frame mismatch")
        braces[side]=brace
        A=Euler(tuple(math.radians(v) for v in cfg["attachment"]["camera_rotation_xyz_deg"])).to_matrix().to_4x4()
        outer=rz+cfg["brace"]["thickness_m"]
        A.translation=Vector((0,0,outer+cfg["attachment"]["support_height_m"]+float(dims[1])/2))
        core=load_r4_core(bounded(cfg["camera"]["asset_dir"],True),bounded(cfg["camera"]["foundation"]["path"],True),side+"_Action4_core",1 if side=="left" else 2)
        core.parent=brace;core.matrix_parent_inverse=Matrix.Identity(4);core.matrix_basis=A
        core.lock_scale=(True,True,True);core["dataset_generation_allowed"]=False
        core["attachment_claim"]="constant source-geometry initialization, not fitted real mount"
        cores[side]=core;attachments[side]=np.array(A)
        brace["A_brace_camera_rowmajor"]=[float(a) for row in A for a in row]
        brace["frame_definition"]="Y distal forearm axis; Z projected world radial reference; X=YcrossZ; origin actual forearm cross-section centre"
        bandverts=[];bandfaces=[];n=48;width=cfg["brace"]["width_m"]
        for yy in (-width/2,width/2):
            for i in range(n):
                angle=2*math.pi*i/n;bandverts.append((rx*math.cos(angle),yy,rz*math.sin(angle)))
        for i in range(n):bandfaces.append((i,(i+1)%n,(i+1)%n+n,i+n))
        band=_mesh(side+"_brace_initial_geometry",bandverts,bandfaces,brace,gray)
        thick=band.modifiers.new("metric_brace_thickness","SOLIDIFY");thick.thickness=cfg["brace"]["thickness_m"];thick.offset=1
        _box(side+"_mount_initial_support",(0,0,outer+cfg["attachment"]["support_height_m"]/2),
             (cfg["attachment"]["support_width_m"],cfg["attachment"]["support_depth_m"],cfg["attachment"]["support_height_m"]),brace,black)
        brace_info[side]={"rest_world_frame":[list(r) for r in transform],"A_brace_camera":attachments[side].tolist(),
                         "actual_forearm_section_points":len(points),"section_half_extents_m":[float(rx-cfg["brace"]["clearance_m"]),float(rz-cfg["brace"]["clearance_m"])],
                         "brace_inner_radii_m":[float(rx),float(rz)],"proximal_offset_m":cfg["brace"]["proximal_offset_m"],
                         "source_prior_only":True}
    bpy.context.view_layer.update()
    rest_lengths={name:float(rig.data.bones[name].length) for name in JOINTS}
    neutral_vertices,_=_evaluated_vertices(body)
    neutral_camera={side:np.array(core.matrix_world) for side,core in cores.items()}
    tests=[]
    for case in [{"name":"neutral","joints":{}},*cfg["validation_poses"]]:
        apply_pose(rig,case["joints"],cfg)
        snapshot={"name":case["name"],"bone_lengths_m":{},"cameras":{}}
        for name in JOINTS:
            posebone=rig.pose.bones[name];length=float((posebone.tail-posebone.head).length)
            if abs(length-rest_lengths[name])>1e-6:raise ValueError("Bone length changed")
            snapshot["bone_lengths_m"][name]=length
        actual_vertices,_=_evaluated_vertices(body)
        movement=float(np.linalg.norm(actual_vertices-neutral_vertices,axis=1).max())
        snapshot["actual_evaluated_skin_max_displacement_m"]=movement
        if case["joints"] and movement<1e-4:raise ValueError("Actual skin did not move")
        bodytree=_bvh(body)
        for side,core in cores.items():
            world=np.array(core.matrix_world);A=np.array(braces[side].matrix_world.inverted()@core.matrix_world)
            det=float(np.linalg.det(world[:3,:3]));scales=np.linalg.norm(world[:3,:3],axis=0)
            if abs(det-1)>2e-6 or max(abs(scales-1))>2e-6 or np.max(abs(A-attachments[side]))>2e-6:raise ValueError("Camera scale/proper rotation/fixed A violation")
            actual_local=np.array([list(v.co) for v in core.data.vertices])
            actual_dims=np.ptp(actual_local,axis=0)
            if np.max(abs(actual_dims-dims))>1e-7:raise ValueError("Camera physical mesh dimensions changed")
            tree=_bvh(core)
            pairs=len(tree.overlap(bodytree));contained=_inside(bodytree,world[:3,3])
            if pairs or contained:raise ValueError(f"Camera/body collision in {case['name']}/{side}: {pairs}, containment{contained}")
            if case["name"]=="hand_only" and np.max(abs(world-neutral_camera[side]))>2e-6:raise ValueError("Hand-only motion moved forearm camera")
            snapshot["cameras"][side]={"scale":scales.tolist(),"rotation_determinant":det,"fixed_A_max_abs_error":float(np.max(abs(A-attachments[side]))),"world_matrix":world.tolist(),"body_surface_intersection_pairs":pairs,"camera_center_inside_body":contained,"actual_mesh_xyz_m":actual_dims.tolist()}
        if _bvh(cores["left"]).overlap(_bvh(cores["right"])):raise ValueError("Camera/camera intersection")
        snapshot["camera_center_distance_m"]=float((cores["left"].matrix_world.translation-cores["right"].matrix_world.translation).length)
        tests.append(snapshot)
    apply_pose(rig,{},cfg)
    restored,_=_evaluated_vertices(body)
    if np.max(abs(restored-neutral_vertices))>1e-7:raise ValueError("Neutral skin was not restored")
    if hashlib.sha256(np.array([list(v.co) for v in body.data.vertices]).tobytes()+json.dumps(faces).encode()).hexdigest()!=geometry_digest:
        raise ValueError("Source control geometry altered")
    for ref in (cfg["architecture"],cfg["body"],cfg["camera"]["foundation"],cfg["camera"]["asset_manifest"]):
        if sha(ref["path"])!=ref["sha256"]:raise ValueError("Original input changed during build")
    result={"schema_version":SCHEMA,"status":"initialization_pending_fit","dataset_generation_allowed":False,
            "source_inputs":cfg,"source_control_vertices":len(verts),"source_control_faces":len(faces),
            "source_geometry_sha256":geometry_digest,"source_deform_cleanup":source_deform_cleanup,"removed_gallery_matrix":gallery,"source_body_dimensions_m":np.ptp(verts,axis=0).tolist(),
            "joint_proxy_definitions":proxy,"rest_lengths_m":rest_lengths,"skin_weight_totals":weight_totals,
            "skin_weight_definition":"fixed shared surface-region membership plus smooth axial blending near shoulder/elbow/wrist; deformation initialized, not visually calibrated",
            "camera_current_mesh_xyz_m":dims.tolist(),"official_camera_nominal_xzy_m":[.0705,.0442,.0328],
            "body_geometry_normalized_to_official":False,"brace_and_fixed_attachment":brace_info,
            "actual_cpu_blender_fk_checks":tests,"neutral_restored":True,"observer_camera":None,
            "limitations":["Source surface boundaries are proxy joint centres, not fitted subject joints",
                          "Initial skin weights only checked on two bounded poses; no claim of production anatomy across full joint range",
                          "Fingers retain source rest shape and move with hand bone; independent finger articulation deferred to G2",
                          "Brace radial axis, proximal offset and camera attachment are explicit geometry priors requiring joint video fit",
                          "No photoreal preview, observer calibration, texture/garment fit or dataset acceptance",
                          "Body/camera surface and center-containment checks cover these3 poses only, not arbitrary poses"]}
    scene["global_rig_config_sha256"]=sha(config_path)
    text=bpy.data.texts.new("GLOBAL_RIG_INITIALIZATION.json");text.write(json.dumps(result,indent=2))
    out.mkdir(parents=True)
    target=out/"Action4_Global_Rig_Initialization_v001.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(target),check_existing=False)
    result["saved_blend"]={"path":str(target.relative_to(ROOT)),"sha256":sha(target)}
    _write(out/"asset_manifest.json",result);_write(report/"report.json",result)
    _write(out/"rig_contract.json",invariant_contract(result))
    return result


def main():
    import sys,traceback
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True);parser.add_argument("--output",required=True);parser.add_argument("--report",required=True)
    argv=sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else None
    args=parser.parse_args(argv)
    try:
        result=build(args.config,args.output,args.report)
        print(json.dumps({"status":result["status"],"dataset_generation_allowed":False,"saved_blend":result["saved_blend"]}))
    except Exception:
        path=bounded(args.report)/"failure.json"
        if not path.exists():_write(path,{"status":"FAILED_NOT_ACCEPTED","traceback":traceback.format_exc()})
        raise
