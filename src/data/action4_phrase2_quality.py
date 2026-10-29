"""Host-side same-frame checks and no-overwrite phrase2 slot publication."""
import json,hashlib,os
from pathlib import Path
from copy import deepcopy
import numpy as np
from PIL import Image
from data import DataAdapter
from data.action4_paths import PROJECT_ROOT as R,contained_path
from data.action4_phrase2_sampling import occlusion_bin
from data.action4_phrase2_plan import category

def validate(record,proposal,slot,cfg):
    DataAdapter(R).validate_sample(record)
    if record["target_count"]!=2 or [x["instance_id"] for x in record["instances"]]!=[1,2] or record["excluded_instances"]:raise ValueError("Two complete physical actors required")
    if record["width"]!=cfg["width"] or record["height"]!=cfg["height"]:raise ValueError("Native dimensions changed")
    files=[(record["image"],record["rgb_sha256"])]
    boxes=[];stats=[]
    cad=np.array(record["corners_3d_mm"])/1000;K=np.array(record["camera"]["K"]);size=np.array([record["width"],record["height"]])
    for obj,side,keys in zip(record["instances"],["left","right"],["AB","CD"]):
        uv=np.array(obj["corners_px"]);g=obj["canonical8_geometry"];T=np.array(g["object_to_capture_cv_m"]);camera=cad@T[:3,:3].T+T[:3,3];h=camera@K.T
        if uv.shape!=(8,2) or not np.isfinite(uv).all() or np.any(camera[:,2]<=0) or np.max(np.abs(h[:,:2]/h[:,2:]-uv))>.005:raise ValueError("Canonical8 independent reprojection failure")
        box=np.r_[uv.min(0),uv.max(0)]
        if not np.allclose(box,obj["bbox_full_xyxy"],atol=1e-6,rtol=0):raise ValueError("Full box differs from complete body")
        if (box[:2]<-.5).any() or (box[2:]>size-.5).any():return False,"body_outside",None
        center=(box[:2]+box[2:])/(2*size)
        for i,key in enumerate(keys):
            expected="outer" if key in slot["outer_coordinates"] else "core"
            if category(center[i],cfg["position_ranges"][key])!=expected:return False,"actual_position_class",None
        with Image.open(contained_path(R,obj["mask_visible"],exists=True)) as image:
            if image.size!=tuple(size):raise ValueError("Visible dimensions differ")
            visible=np.asarray(image)>0
        with Image.open(contained_path(R,obj["mask_amodal_expanded"],exists=True)) as image:
            if image.size!=tuple(size*3):raise ValueError("Amodal expanded dimensions differ")
            full=np.asarray(image)>0
        amodal=full[size[1]:2*size[1],size[0]:2*size[0]]
        av=int(visible.sum());ai=int(amodal.sum());af=int(full.sum())
        if ai<=0 or af<=0:raise ValueError("Empty actual amodal mask")
        bad=int((visible&~amodal).sum())
        if bad>max(4,.005*ai):raise ValueError("Mask spatial alignment outside existing boundary tolerance")
        o=max(0.,1-av/ai);trunc=1-ai/af
        if abs(o-obj["occlusion_inframe"])>1e-12 or abs(trunc-obj["truncation"])>1e-12:raise ValueError("Mask measurement identity mismatch")
        if occlusion_bin(o)!=slot["occlusion_bins"][side]:return False,"actual_occlusion_bin",{"side":side,"occlusion":o}
        if trunc>1e-5:return False,"actual_truncation",None
        pose=proposal["actor_pose_capture_by_side"][side];B=np.array(pose["brace_B_capture"]);A=np.array(proposal["sides"][side]["A_brace_from_CAD"])
        if not np.allclose(B@A,pose["core_C_capture"],atol=1e-9,rtol=0):raise ValueError("Fixed installation broken")
        actual_R=T[:3,:3]
        if not np.allclose(actual_R.T@actual_R,np.eye(3),atol=2e-5,rtol=0) or not np.isclose(np.linalg.det(actual_R),1,atol=2e-5):raise ValueError("Scale/shear in target pose")
        for key in ["mask_visible","mask_amodal_expanded"]:files.append((obj[key],obj[key+"_sha256"]))
        boxes.append(box);stats.append({"side":side,"occlusion":o,"visible_pixels":av,"amodal_pixels":ai,"bbox_center_uv":center.tolist(),"width_fraction":float((box[2]-box[0])/size[0])})
    if (np.minimum(boxes[0][2:],boxes[1][2:])-np.maximum(boxes[0][:2],boxes[1][:2])>0).all():return False,"actual_boxes_overlap",None
    for name,digest in files:
        if hashlib.sha256(contained_path(R,name,exists=True).read_bytes()).hexdigest()!=digest:raise ValueError("Rendered image identity changed")
    return True,None,stats

def publish(record,proposal,slot,stats):
    result=deepcopy(record);root=R/"data/phrase2"/slot["split"]
    for name in ["images","masks","geometry"]:(root/name).mkdir(parents=True,exist_ok=True)
    mappings=[("image",None,root/"images"/(slot["sample_id"]+".png"))]
    for obj in result["instances"]:
        for key,suffix in [("mask_visible","visible"),("mask_amodal_expanded","amodal_expanded")]:
            mappings.append((key,obj,root/"masks"/f"{slot['sample_id']}_i{obj['instance_id']}_{suffix}.png"))
        if obj.get("surface_supervision"):
            mappings.append(("surface_supervision",obj,root/"geometry"/f"{slot['sample_id']}_i{obj['instance_id']}.npz"))
    for key,obj,dest in mappings:
        source_value=result[key] if obj is None else obj[key]["path"] if key=="surface_supervision" else obj[key]
        source=contained_path(R,source_value,exists=True);dest=contained_path(R,dest,exists=False)
        if dest.exists():
            if not os.path.samefile(source,dest):raise FileExistsError("Publication may not overwrite")
        else:os.link(source,dest)
        if obj is None:result[key]=str(dest.relative_to(R))
        elif key=="surface_supervision":obj[key]["path"]=str(dest.relative_to(R))
        else:obj[key]=str(dest.relative_to(R))
    result.update(status="training_accepted",training_accepted=True,dataset_generation_allowed=True,pending_host_png_validation=False,
        dataset_root="data/phrase2",slot_id=slot["slot_id"],attempt=proposal["phrase2"]["attempt"],scene_id=slot["scene_id"],
        asset_version="phrase2_scene_layout_v002_original_arms_v003",phrase2=proposal["phrase2"],actual_phrase2_measurements=stats,
        corner_visibility_semantics="ray clear to fixed virtual cuboid vertex AND in-frame; not visible physical mesh vertex",
        occlusion_mode="arm_geometry",external_occlusion_semantics="actual visible vs isolated inframe body mask; truncation separate")
    result["actor_pose_capture_by_side"]=deepcopy(proposal["actor_pose_capture_by_side"])
    result["fixed_installation_by_side"]={side:{key:deepcopy(proposal["sides"][side][key]) for key in ["A_brace_from_CAD","S_skin_relative_to_brace","F_body_from_brace"]} for side in ["left","right"]}
    result["environment_controls"]=deepcopy(proposal["environment_controls"])
    result["render_seed"]=proposal["entry"]["seed"]
    result["corner_ids"]=list(range(8))
    for obj,side in zip(result["instances"],["left","right"]):
        note=proposal["phrase2"]["by_side"][side]
        obj.update(wearing_side=side,corner_ids=list(range(8)),requested_center_uv=note["requested_center_uv"],bbox_center_uv=note["bbox_center_uv"],actual_width_fraction=note["actual_width_fraction"],occlusion_bin=slot["occlusion_bins"][side])
    from data.action4_phrase2_visibility import canonical_visibility
    for obj in result["instances"]:
        g=obj["canonical8_geometry"]
        flags=canonical_visibility(np.array(result["corners_3d_mm"])/1000,np.array(g["object_to_capture_cv_m"]),g["ray_clear_to_virtual_point"],g["point_inframe"])
        obj.update(corner_ray_clear=flags["ray_clear"],corner_self_visible=flags["self_visible"],corner_visible=flags["visible"],corner_visibility_semantics=flags["semantics"])
        g.update(training_accepted=True,training_corner_visible=flags["visible"],training_visibility_semantics=flags["semantics"],claim="Same-frame fixed canonical8; host projection/mask/rigid installation checked")
    result["corner_visibility_semantics"]=flags["semantics"]
    result["render_profile_version"]="phrase2_native32_mask1_optix_v001"
    return result
