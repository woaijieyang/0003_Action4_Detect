"""Deterministic coupled sampling for exact position/arm-occlusion slot classes."""
import json,math,random
from copy import deepcopy
import numpy as np
from data.action4_phrase2_plan import seed_for,draw_interval,category
from data.action4_phrase2_geometry import rotation_for_arm,pose_for_box,projected
from data.action4_parameter_sampling import project,rigid

def occlusion_bin(o):
    if not np.isfinite(o) or o<0 or o>.70:return None
    return 0 if o<=.35 else 1 if o<=.50 else 2

def propose(runtime,cfg,slot,attempt,catalog,local_visibility=None):
    rng=random.Random(seed_for(cfg["master_seed"],slot["split"],slot["slot_id"],attempt))
    common_height=rng.gauss(0,1)
    common=rng.random();tail=rng.random()>=.85
    desired={};widths=[None,None]
    size=np.array([runtime.scene.render.resolution_x,runtime.scene.render.resolution_y]);corners=np.array(runtime.metadata["canonical_definition"]["corners_3d_m"])
    poses={};sides={};metrics={};notes={};boxes=[]
    for i,side in enumerate(["left","right"]):
        chosen=None;last_reason="bounded_side_orientation"
        for _ in range(cfg["max_orientation_proposals_per_side"]):
            horizontal,vertical=("A","B") if side=="left" else ("C","D")
            z=.80*common_height+.60*rng.gauss(0,1)
            unit=.5*(1+math.erf(z/math.sqrt(2)))
            desired[side]=[draw_interval(rng,cfg["position_ranges"][horizontal],horizontal in slot["outer_coordinates"]),draw_interval(rng,cfg["position_ranges"][vertical],vertical in slot["outer_coordinates"],unit)]
            q=.75*common+.25*rng.random()
            if not tail:widths[i]=.08+.06*q
            elif common<.5:widths[i]=.06+.02*rng.random()
            else:widths[i]=.14+.09*rng.random()
            values,error=_propose_side(runtime,cfg,slot,rng,catalog,side,i,desired,widths,size,corners)
            if error:last_reason=error;continue
            pose,side_fixed,metric,note,box=values
            if local_visibility is not None:
                m=local_visibility[side].estimate(np.array(pose["brace_B_capture"]),np.array(pose["core_C_capture"]),28,return_sources=True)
                if m["occlusion_estimate"] is None or occlusion_bin(m["occlusion_estimate"])!=slot["occlusion_bins"][side] or (slot["occlusion_bins"][side]>0 and m["primary_class"]!="own_wrist_forearm"):
                    last_reason="estimated_occlusion_or_source";continue
            chosen=values;break
        if chosen is None:return None,last_reason
        poses[side],sides[side],metrics[side],notes[side],box=chosen;boxes.append(box)
    if abs(notes["right"]["bbox_center_uv"][1]-notes["left"]["bbox_center_uv"][1])>.42:return None,"pair_height"
    if not np.isfinite(boxes).all():raise ValueError("Nonfinite box")
    overlap=np.minimum(boxes[0][2:],boxes[1][2:])-np.maximum(boxes[0][:2],boxes[1][:2])
    if (overlap>0).all():return None,"body_boxes_overlap"
    # Camera rays for disjoint projected body boxes are disjoint, so bodies cannot intersect.
    entry={"sample_id":slot["sample_id"],"index":slot["slot_id"],"seed":seed_for(cfg["master_seed"],slot["split"],slot["slot_id"],attempt,"render")%(2**31-1),"scene_family":{"wall":"wall_rear","table_full_garment":"table_turn","cloth_lift10_full_garment":"cloth_lift"}[slot["scene_id"]],"target_count":2,"group_id":slot["group_id"],"background_source_id":"phrase2_"+slot["scene_id"],"background_seed":cfg["master_seed"],"split":slot["split"],"target_presence":[True,True]}
    result={"schema":"action4.phrase2_arm_proposal.v1","parameter_contract_revision":"phrase2_arm_v001","occlusion_mode":"arm_geometry","occluder_proposals":{},"entry":entry,"target_presence":[True,True],"W_from_capture":runtime.W.tolist(),"K_native":runtime.K.tolist(),"canonical8_m":corners.tolist(),"corner_ids":list(range(8)),"actor_pose_capture_by_side":poses,"sides":sides,"derived":{"by_side":metrics},"background_request":{"template":runtime.name},"environment_controls":{"area_light_gain":rng.uniform(.85,1.15),"exposure_offset_ev":0.},"provenance":{"anchor_ref":runtime.metadata["baseline_parameters_ref"]},"phrase2":{"slot_id":slot["slot_id"],"attempt":attempt,"outer_coordinates":slot["outer_coordinates"],"occlusion_bins":slot["occlusion_bins"],"by_side":notes,"master_seed":cfg["master_seed"],"source_RNG":"master+split+slot+attempt"}}
    return result,None

def _propose_side(runtime,cfg,slot,rng,catalog,side,i,desired,widths,size,corners):
    poses={};sides={};metrics={};notes={};boxes=[]
    kind=slot["occlusion_bins"][side]
    cache=catalog.setdefault("_orientation_cache",{})
    key=side+str(kind)
    if key not in cache:
        ready=[]
        for a in catalog["rows"][side]["anchors"]:
            values=[(x["roll_deg"],x["direction_x"],x["direction_z"]) for x in a["samples"] if x["occlusion_estimate"] is not None and occlusion_bin(x["occlusion_estimate"])==kind and (kind==0 or x["primary_class"]=="own_wrist_forearm")]
            if values:ready.append((a["center_uv"],values))
        if not ready:raise ValueError("Own-skin occlusion class not reachable in initialized catalog")
        cache[key]=ready
    ready=cache[key]
    weights=[math.exp(-float(np.sum((np.array(center)-desired[side])**2))/(2*.22**2))+.01 for center,values in ready]
    _,choices=rng.choices(ready,weights=weights,k=1)[0]
    roll_center,direction_x,direction_z=rng.choice(choices)
    roll=roll_center+rng.uniform(-1.,1.)
    direction=[direction_x+rng.uniform(-.15,.15),1,direction_z+rng.uniform(-.12,.12)]
    RB=rotation_for_arm(runtime,side,roll,direction);fixed=runtime.anchor["sides"][side];A=np.array(fixed["A_brace_from_CAD"])
    try:B,C=pose_for_box(RB,A,corners,runtime.K,size,desired[side],widths[i])
    except ValueError:return None,"projection_depth"
    if not cfg["depth_m"][0]<=C[2,3]<=cfg["depth_m"][1]:return None,"depth_range"
    # Test actual vertices inside the finite tabletop cuboid, not its infinite plane.
    if runtime.support_solid is not None:
        inverse,low,high=runtime.support_solid
        actor=runtime.local_actor[side]@B[:3,:3].T+B[:3,3]
        local=actor@inverse[:3,:3].T+inverse[:3,3]
        if ((local>low+1e-6)&(local<high-1e-6)).all(1).any():return None,"inside_table_solid"
    uv,box,cam=projected(corners,C,runtime.K)
    if np.any(cam[:,2]<=runtime.metadata["near_plane_m"]) or np.any(box[:2]<.5) or np.any(box[2:]>size-.5):return None,"body_outside"
    center=(box[:2]+box[2:])/(2*size)
    for axis,key in enumerate(["A","B"] if side=="left" else ["C","D"]):
        expected="outer" if key in slot["outer_coordinates"] else "core"
        if category(center[axis],cfg["position_ranges"][key])!=expected:return None,"position_class"
    end=runtime.arm_terminal_points_B[side]@B[:3,:3].T+B[:3,3]
    # Behind-camera arm ends are legal only after the continuous arm has
    # already left the image: clip each wrist-to-terminal segment at z=.02.
    start=B[:3,3]
    if start[2]<=.02:return None,"wrist_near_camera"
    behind=end[:,2]<.02
    if behind.any():
        fraction=(start[2]-.02)/(start[2]-end[behind,2])
        end=end.copy();end[behind]=start+(end[behind]-start)*fraction[:,None]
    eh=end@runtime.K.T;eu=eh[:,:2]/eh[:,2:]/size
    if not any([(eu[:,0]<-.01).all(),(eu[:,0]>1.01).all(),(eu[:,1]<-.01).all(),(eu[:,1]>1.01).all()]):return None,"arm_terminal_in_frame"
    from data.action4_phrase2_geometry import upperarm_reenters,junction_outside_frame
    if not junction_outside_frame(runtime,side,B,size):return None,"elbow_junction_in_frame"
    if upperarm_reenters(runtime,side,B,size):return None,"upperarm_reentry"
    source=runtime.anchor["actor_pose_capture_by_side"][side];Bbase=np.linalg.inv(np.array(source["whole_actor_delta_capture"]))@np.array(source["brace_B_capture"])
    poses[side]={"brace_B_capture":B.tolist(),"core_C_capture":C.tolist(),"whole_actor_delta_capture":(B@np.linalg.inv(Bbase)).tolist()}
    sides[side]={k:deepcopy(fixed[k]) for k in ["A_brace_from_CAD","S_skin_relative_to_brace","F_body_from_brace"]};sides[side].update(present=True,**poses[side])
    metrics[side]=project(corners,C,runtime.K,size,runtime.metadata["near_plane_m"])
    notes[side]={"requested_center_uv":desired[side],"requested_center_semantics":"projected full-body bbox center","bbox_center_uv":center.tolist(),"requested_width_fraction":widths[i],"actual_width_fraction":float((box[2]-box[0])/size[0]),"forearm_roll_deg":roll,"forearm_direction_capture":direction,"target_occlusion_bin":kind,"arm_terminal_uv":eu.tolist()}
    boxes.append(box)
    return (poses[side],sides[side],metrics[side],notes[side],box),None
