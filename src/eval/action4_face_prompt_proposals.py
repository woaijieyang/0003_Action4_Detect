"""Pilot face point hypotheses from A0 XYZ and current authenticated body SAM.

No SAM execution, face-mask production, semantic GT, nearest-face assignment,
corner probabilities or model training. Patch bounds are only a candidate
range, never proof of a physical face or an exterior surface.
"""
from dataclasses import dataclass,asdict,field
import copy
import json
import numpy as np
from scipy.ndimage import distance_transform_edt

from eval.action4_face_inputs import FaceInputs
from eval.action4_sam_full_review import ReceiptVerifier,ROOT,checked_path
from data.action4_roi_contract import RoiSpec
from data.action4_sam_roi_inputs import ImageFrame,verify_image_frame,array_digest
from data.action4_video_roi_context import VideoROIContext,_digest,validate_project_body_registration
from data.action4_video_sam_snapshot import VideoMaskBinding
from model.action4_face_geometry import FaceGeometryContract,project_six_faces


@dataclass(frozen=True)
class FacePromptThresholds:
    plane_distance_mm: float
    bounds_tolerance_mm: float
    minimum_support_cells: int
    minimum_point_distance_px: float
    positive_points: int
    adjacent_negative_points: int
    background_negative_points: int

    def __post_init__(self):
        for value in (self.plane_distance_mm,self.bounds_tolerance_mm,self.minimum_point_distance_px):
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or value<0:
                raise ValueError("Finite explicit nonnegative pilot thresholds required")
        if self.plane_distance_mm<=0 or self.minimum_point_distance_px<=0:
            raise ValueError("Positive plane tolerance and native point spacing required")
        for value in (self.minimum_support_cells,self.positive_points,self.adjacent_negative_points,self.background_negative_points):
            if type(value) is not int or value<0:raise ValueError("Explicit integer point/cell counts required")
        if (self.minimum_support_cells<1 or not 2<=self.positive_points<=4
                or not 2<=self.adjacent_negative_points+self.background_negative_points<=4):
            raise ValueError("Native point prompting needs2-4 positives and2-4 explicit negatives")


_PATCH_VERIFIED=object()


@dataclass(frozen=True)
class FacePatchCandidates:
    reference_json: str
    patches_json: str
    mesh_sha256: str
    contract_sha256: str
    _verified: object=field(default=None,init=False,repr=False,compare=False)

    @classmethod
    def from_ref(cls,reference,contract):
        if not isinstance(contract,FaceGeometryContract):raise TypeError("Fixed face contract required")
        verifier=ReceiptVerifier();doc=verifier.bound_json(reference)
        parent=verifier.bound_json({"path":contract.parent_asset_path,"sha256":contract.parent_asset_sha256})
        expected_mesh=checked_path(contract.parent_asset_path).parent/parent["mesh_data"]
        if (doc.get("schema")!="action4.physical_patch_candidates.v1"
                or doc.get("status")!="CANDIDATES_NOT_REGISTERED_NOT_SEMANTIC_GT"
                or doc.get("thresholds_calibrated") is not False or doc.get("training_labels_modified") is not False
                or doc["mesh_sha256"]!=parent["mesh_data_sha256"]
                or checked_path(doc["source_evidence"]["mesh_path"])!=expected_mesh):
            raise ValueError("Patch candidates do not match the fixed parent asset")
        verifier.verify(expected_mesh,doc["mesh_sha256"])
        evidence=doc["source_evidence"]
        verifier.verify(evidence["audit_path"],evidence["audit_sha256"])
        if len(evidence["first_hit_refs"])!=6:raise ValueError("Six first-hit source refs required")
        for ref in evidence["first_hit_refs"]:verifier.verify(ref["path"],ref["sha256"])
        ids=set()
        for patch in doc["patches"]:
            pid=patch["physical_patch_id"]
            if not isinstance(pid,str) or not pid or pid in ids:raise ValueError("Unique physical patch IDs required")
            ids.add(pid)
            if patch.get("semantic_gt") is not False or patch.get("exterior_verified") is not False:
                raise ValueError("Only explicitly unverified patch candidates are supported")
            face_ids=patch["logical_face_candidates"]
            if not isinstance(face_ids,list) or len(set(face_ids))!=len(face_ids) or any(type(i) is not int or not 0<=i<6 for i in face_ids):
                raise ValueError("Fixed logical FaceID candidate set required")
            plane=patch["plane_equation_n_dot_x_plus_d_equals_0"];normal=np.asarray(plane["normal_object"],float)
            bounds=np.asarray(patch["bounds_mm"],float)
            if (normal.shape!=(3,) or bounds.shape!=(2,3) or not np.isfinite(normal).all()
                    or not np.isfinite(bounds).all() or not np.isfinite(plane["d_mm"])
                    or not np.isclose(np.linalg.norm(normal),1.,rtol=0,atol=1e-6)
                    or (bounds[1]<bounds[0]).any()):
                raise ValueError("Finite unit-normal plane and ordered mm bounds required")
        result=cls(json.dumps(reference,sort_keys=True),json.dumps(doc["patches"],allow_nan=False),
                   doc["mesh_sha256"],contract.source_sha256)
        object.__setattr__(result,"_verified",_PATCH_VERIFIED)
        return result


def _checked(face,frame,context,binding,mask,contract,models):
    if not isinstance(face,FaceInputs) or not isinstance(frame,ImageFrame) or not isinstance(context,VideoROIContext):
        raise TypeError("Actual FaceInputs, ImageFrame and authenticated VideoROIContext required")
    if not isinstance(binding,VideoMaskBinding):raise TypeError("Current native VideoMaskBinding required")
    verify_image_frame(frame);binding.validate()
    m=face.metadata;g=face.geometry_prior;body=m.get("body_identity")
    if (m.get("models")!=models or m.get("image_id")!=frame.image_id or m.get("pixel_space_id")!=frame.pixel_space_id
            or m.get("source_kind")!="A0_current_image_geometry_prior" or m.get("independent_observation") is not False
            or m.get("semantic_face_labels_available") is not False or not body
            or body.get("persistent_identity_ready") is not True):
        raise ValueError("Authenticated current-image model/body identity required")
    K=np.asarray(g["K_native"],float)
    rows=context.consume(np.moveaxis(frame.pixels,-1,0),K,b0_sha256=models["B0"]["sha256"])
    selected=[r for r in context.select_for_a0(rows) if r["prediction_id"]==m["prediction_id"]]
    if len(selected)!=1 or selected[0]["track_id"]!=body["track_id"]:
        raise ValueError("Face inputs do not belong to current body/frame ROI")
    ledger=json.loads(context.source_ledger_json)
    event=ledger.get("project_body_registrations",{}).get(body["track_id"])
    owner=(ledger.get("native_owners") or {}).get(body["track_id"])
    validate_project_body_registration(event,ledger.get("project_body_scope"),track_id=body["track_id"],
        frame_id=context.frame_id,pts=ledger.get("pts"),time_base=ledger.get("time_base"),
        pixel_space_id=frame.pixel_space_id,replay_identity=context.replay_identity,native_owner=owner)
    if (body["generation"]!=event["project_generation"] or body["sequence_identity"]!=event["sequence_identity"]
            or body["scoped_body_id"]!=event["scoped_body_id"] or body["frame_id"]!=context.frame_id
            or binding.frame_id!=context.frame_id or binding.pts!=ledger["pts"]
            or binding.time_base!=tuple(ledger["time_base"]) or binding.pixel_space_id!=frame.pixel_space_id
            or binding.track_id!=body["track_id"] or binding.track_generation!=body["generation"]
            or binding.session_id!=owner["session_id"] or binding.native_obj_id!=owner["native_obj_id"]
            or binding.mask_generation!=owner["generation"] or binding.replay_identity!=context.replay_identity
            or binding.a0_selection_sha256!=models["A0"]["sha256"]):
        raise ValueError("Stale/swapped native mask, project generation, frame or model")
    native_status=ledger.get("native_status")
    if (not isinstance(native_status,dict) or native_status.get(body["track_id"])!="target_present"
            or selected[0].get("current_support_status",{}).get("SAM") is not True):
        raise ValueError("Current native body support is unavailable or rejected")
    if selected[0].get("temporal_status")=="sam_supported":
        current=selected[0]["sam_support_evidence"]["receipt"]
        if current["mask_sha256"]!=binding.mask_sha256:
            raise ValueError("Body mask differs from authenticated current SAM support receipt")
    if not isinstance(mask,np.ndarray) or mask.dtype!=np.bool_ or mask.shape!=frame.pixels.shape[:2] or array_digest(mask)!=binding.mask_sha256:
        raise ValueError("Current actual native body mask bytes differ")
    spec=m["roi_spec"]
    roi=RoiSpec.from_full_box(spec["full_body_bbox"],spec["expansion"],spec["output_size"])
    if roi.record()!=spec or list(roi.full_body_bbox)!=selected[0]["bbox_for_a0"]:
        raise ValueError("Actual body ROI differs")
    from data.action4_surface_snapshot import roi_identity
    if m["roi_id"]!=roi_identity(frame.image_id,m["prediction_id"],list(roi.full_body_bbox),roi.affine()):
        raise ValueError("Face ROI identity differs")
    regenerated=project_six_faces(np.asarray(contract.canonical8_m),contract,g["R"],g["t_m"],K,
        units="m",pixel_space_id=frame.pixel_space_id,image_size_wh=(frame.pixels.shape[1],frame.pixels.shape[0]),
        min_depth_m=g["thresholds"]["min_depth_m"],grazing_cos_threshold=g["thresholds"]["grazing_cos_threshold"])
    if (g["source_contract_sha256"]!=contract.source_sha256 or g["faces"]!=regenerated["faces"]):
        raise ValueError("FaceID/pose/geometry prior changed")
    if (face.xyz_object_m.shape!=(3,112,112) or not np.isfinite(face.xyz_object_m).all()
            or face.valid_image_support112.shape!=(1,112,112)
            or not np.isfinite(face.valid_image_support112).all()
            or not np.isin(face.valid_image_support112,(0,1)).all()):
        raise ValueError("Finite112-grid predicted XYZ and image support required")
    return roi,body


def _pick(cells,native_xy,count,spacing,excluded=()):
    """Existing predicted-instance strategy: interior-distance then separation."""
    if count==0:return []
    score=distance_transform_edt(np.pad(cells,1))[1:-1,1:-1]
    selected=[];blocked={tuple(p) for p in excluded}
    for point in blocked:
        score[np.linalg.norm(native_xy-np.asarray(point),axis=-1)<spacing]=0
    for _ in range(count):
        while True:
            iy,ix=np.unravel_index(np.argmax(score),score.shape)
            if score[iy,ix]<=0:return None
            point=tuple(int(v) for v in native_xy[iy,ix])
            if point in blocked:score[iy,ix]=0;continue
            break
        selected.append(list(point));blocked.add(point)
        distance=np.linalg.norm(native_xy-np.asarray(point),axis=-1)
        score[distance<spacing]=0
        score[iy,ix]=0
    return selected


def propose_face_prompts(face_inputs,*,frame,video_context,body_binding,body_mask,
                         patch_candidates,contract,expected_models,thresholds,support_policy=None):
    logical_only=False;probe_disagreement=False
    if support_policy is not None:
        if not isinstance(support_policy,dict):raise ValueError("Explicit face support policy required")
        version=support_policy.get("schema")
        allowed={"schema","mode"} if version=="action4.face_prompt_support_policy.v1" else {"schema","mode","allow_pose_prior_disagreement"}
        if (version not in ("action4.face_prompt_support_policy.v1","action4.face_prompt_support_policy.v2")
                or set(support_policy)!=allowed
                or support_policy["mode"] not in ("exclusive_physical_patch","logical_face_consensus")):
            raise ValueError("Explicit registered face prompt support policy required")
        logical_only=support_policy["mode"]=="logical_face_consensus"
        if version=="action4.face_prompt_support_policy.v2":
            if type(support_policy["allow_pose_prior_disagreement"]) is not bool or not logical_only:
                raise ValueError("Pose-prior probe requires explicit logical-face policy")
            probe_disagreement=support_policy["allow_pose_prior_disagreement"]
    if not isinstance(thresholds,FacePromptThresholds) or not isinstance(patch_candidates,FacePatchCandidates):
        raise TypeError("Explicit pilot thresholds and bound patch candidate collection required")
    if patch_candidates._verified is not _PATCH_VERIFIED:raise ValueError("Patch candidates require local source verification")
    if patch_candidates.contract_sha256!=contract.source_sha256:
        raise ValueError("Patch collection belongs to another face contract")
    roi,body=_checked(face_inputs,frame,video_context,body_binding,body_mask,contract,expected_models)
    yy,xx=np.indices((112,112));grid=np.stack((xx,yy,np.ones_like(xx)),-1)
    native=np.rint((grid@np.linalg.inv(roi.affine(112)).T)[...,:2]).astype(int)
    h,w=body_mask.shape
    in_image=(native[...,0]>=0)&(native[...,0]<w)&(native[...,1]>=0)&(native[...,1]<h)
    valid=in_image&(face_inputs.valid_image_support112[0]>0)
    supported=np.zeros((112,112),bool)
    supported[valid]=body_mask[native[valid,1],native[valid,0]]
    xyz=np.moveaxis(face_inputs.xyz_object_m,0,-1)*1000.
    if not np.isfinite(xyz).all():raise ValueError("Object XYZ unit conversion overflow")
    matches=[];records=[];front={f["face_id"]:f["geometric_front_facing"] for f in face_inputs.geometry_prior["faces"]}
    for patch in json.loads(patch_candidates.patches_json):
        logical=patch["logical_face_candidates"]
        record={"physical_patch_id":patch["physical_patch_id"],"logical_face_candidates":logical,
                "unknown_or_exclusion_reasons":patch["unknown_or_exclusion_reasons"]}
        if patch["geometric_plane_candidate"] is not True or not logical:
            record.update(status="ineligible_patch_candidate",matched_cells=0);records.append(record);continue
        plane=patch["plane_equation_n_dot_x_plus_d_equals_0"];bounds=np.asarray(patch["bounds_mm"])
        on_plane=np.abs(xyz@np.asarray(plane["normal_object"])+plane["d_mm"])<=thresholds.plane_distance_mm
        in_bounds=((xyz>=bounds[0]-thresholds.bounds_tolerance_mm)&(xyz<=bounds[1]+thresholds.bounds_tolerance_mm)).all(-1)
        match=supported&on_plane&in_bounds
        record.update(status="pilot_geometry_candidate",matched_cells=int(match.sum()))
        records.append(record);matches.append((patch,match))
    original_matches=matches
    physical_count=np.zeros((112,112),int)
    for patch,mask in original_matches:physical_count+=mask*len(patch["logical_face_candidates"])
    if logical_only:
        # This union is for LOGICAL prompt identity, not a fitted plane. Keep
        # all contributing physical patch IDs/depth uncertainty. No hull,
        # interpolation, hole filling, nearest-patch or depth winner selection.
        matches=[]
        for face_id in range(6):
            members=[(p,m) for p,m in original_matches if face_id in p["logical_face_candidates"]]
            if not members:continue
            union=np.zeros((112,112),bool);multiplicity=np.zeros((112,112),int)
            for _,mask in members:union|=mask;multiplicity+=mask
            patch={"physical_patch_id":"logical_face_"+str(face_id),
                "physical_patch_ids":[p["physical_patch_id"] for p,m in members if m.any()],
                "logical_face_candidates":[face_id],
                "depth_ambiguous":bool((multiplicity>1).any()),
                "ambiguous_physical_cells":int((multiplicity>1).sum()),
                "unknown_or_exclusion_reasons":sorted(set(
                    reason for p,_ in members for reason in p["unknown_or_exclusion_reasons"])
                    |{"logical_identity_consensus_only_physical_patch_and_depth_unresolved"})}
            matches.append((patch,union))
    count=np.zeros((112,112),int)
    for patch,mask in matches:count+=mask*len(patch["logical_face_candidates"])
    ambiguous=supported&(count>1);unknown=supported&(count==0)
    # Each accepted cell has exactly one patch and one face candidate. Native
    # rounding may collide across112 cells; such multi-label pixels also reject.
    owners={}
    for patch,mask in matches:
        for y,x in np.argwhere(mask&(count==1)):
            owners.setdefault(tuple(native[y,x]),set()).add((patch["physical_patch_id"],patch["logical_face_candidates"][0]))
    collision=np.zeros((112,112),bool)
    for y,x in np.argwhere(supported):
        collision[y,x]=len(owners.get(tuple(native[y,x]),()))>1
    supports=[]
    for patch,mask in matches:
        face_id=patch["logical_face_candidates"][0] if len(patch["logical_face_candidates"])==1 else None
        supports.append((patch,face_id,mask&(count==1)&~collision))
    quads={f["face_id"]:set(f["corner_ids"]) for f in face_inputs.geometry_prior["faces"]}
    proposals=[]
    for patch,face_id,cells in supports:
        pid=patch["physical_patch_id"]
        base={"face_id":face_id,"physical_patch_ids":patch.get("physical_patch_ids",[pid]),"logical_face_candidates":patch["logical_face_candidates"],
              "support_cells":int(cells.sum()),"status":"no_proposal",
              "patch_unknown_reasons":patch["unknown_or_exclusion_reasons"]}
        if logical_only:
            base.update(logical_face_only=True,depth_unresolved=True,depth_ambiguous=patch["depth_ambiguous"],
                ambiguous_physical_cells=patch["ambiguous_physical_cells"],physical_plane_registered=False,
                XYZ_binding_available=False,geometry_fit_eligible=False,homography_eligible=False,PnP_eligible=False)
        if probe_disagreement:
            base.update(pose_visibility_prior_only=True,pose_prior_disagrees=face_id is not None and not front[face_id],
                        independent_image_face_confirmation_required=True)
        if face_id is None:base["reason"]="multiple_face_candidates"
        elif not front[face_id] and not probe_disagreement:base["reason"]="not_geometric_front_facing"
        elif cells.sum()<thresholds.minimum_support_cells:base["reason"]="insufficient_exclusive_patch_support"
        else:
            positives=_pick(cells,native,thresholds.positive_points,thresholds.minimum_point_distance_px)
            adjacent=np.zeros_like(cells)
            for other,other_face,other_cells in supports:
                if other_face is not None and other_face!=face_id and (front[other_face] or probe_disagreement) and len(quads[face_id]&quads[other_face])==2:
                    adjacent|=other_cells
            neighbors=None if positives is None else _pick(adjacent,native,thresholds.adjacent_negative_points,
                thresholds.minimum_point_distance_px,positives)
            background=None if neighbors is None else _pick(valid&~supported,native,thresholds.background_negative_points,
                thresholds.minimum_point_distance_px,positives+neighbors)
            if positives is None or neighbors is None or background is None:
                base["reason"]="insufficient_real_positive_adjacent_or_background_points"
            else:
                points=positives+neighbors+background;labels=[1]*len(positives)+[0]*(len(neighbors)+len(background))
                base.update(status="prompt_hypothesis",reason="logical_face_consensus_prompt_only" if logical_only else "exclusive_predicted_XYZ_patch_support_pilot",
                    points_native=points,points_xy01=(np.asarray(points)/[w,h]).tolist(),labels=labels,
                    point_sources=["logical_face_consensus_inside_current_body" if logical_only else "candidate_patch_inside_current_body"]*len(positives)
                        +["adjacent_face_candidate_inside_current_body"]*len(neighbors)
                        +["outside_current_body_within_valid_ROI_not_background_GT"]*len(background),
                    support_grid_sha256=array_digest(cells),native_face_mask_produced=False)
        proposals.append(base)
    dependence={"surface_receipt":face_inputs.metadata["surface_receipt"],"models":copy.deepcopy(expected_models),
        "body":copy.deepcopy(body),"body_mask_binding":asdict(body_binding),"body_mask_sha256":array_digest(body_mask),
        "project_replay_identity":video_context.replay_identity,"roi_id":face_inputs.metadata["roi_id"],
        "roi_spec":roi.record(),"K_native":face_inputs.geometry_prior["K_native"],
        "pose":{"R":face_inputs.geometry_prior["R"],"t_m":face_inputs.geometry_prior["t_m"]},
        "face_contract_sha256":contract.source_sha256,"patch_candidates_ref":json.loads(patch_candidates.reference_json),
        "predicted_XYZ_sha256":array_digest(face_inputs.xyz_object_m),"frame_id":body_binding.frame_id,
        "pts":body_binding.pts,"time_base":list(body_binding.time_base),"pixel_space_id":body_binding.pixel_space_id}
    if support_policy is not None:dependence["support_policy"]=copy.deepcopy(support_policy)
    for proposal in proposals:
        proposal["parent_dependency_identity"]=_digest(dependence)
        proposal["hypothesis_identity"]=_digest(proposal)
    face_diagnostics=[]
    for face_id in range(6):
        selected=[p for p in proposals if p["face_id"]==face_id and p["status"]=="prompt_hypothesis"]
        face_diagnostics.append({"face_id":face_id,"geometric_front_facing":front[face_id],
            "status":"prompt_hypotheses_available" if selected else "unknown_or_no_proposal",
            "hypothesis_ids":[p["hypothesis_identity"] for p in selected],
            "physical_patch_candidates":[p["physical_patch_id"] for p,_ in original_matches
                                         if face_id in p["logical_face_candidates"]],
            "observed_face_certified":False})
    result={"schema":"action4.face_prompt_hypotheses.v1","status":"PILOT_UNCALIBRATED_NOT_FACE_ACCEPTANCE",
        "dependencies":dependence,"thresholds":asdict(thresholds),"thresholds_calibrated":False,
        "sampling":"nearest native pixel of each actual112-grid centre; no interpolated/filled face mask",
        "candidate_policy":("same logical FaceID union for SAM prompt only; physical depth remains unresolved; different FaceIDs withheld"
                            if logical_only else "all plane-and-range matches; multi-patch/depth/face matches withheld, never nearest-face assigned"),
        "support_summary":{"body_supported_cells":int(supported.sum()),"unknown_cells":int(unknown.sum()),
            "ambiguous_patch_or_face_cells":int(ambiguous.sum()),"native_rounding_conflict_cells":int(collision.sum())},
        "patch_diagnostics":records,"face_diagnostics":face_diagnostics,"hypotheses":proposals,"semantic_gt":False,"face_tracking_accepted":False,
        "GT_used":False,"corner_probabilities_available":False,"native_inference_executed":False}

    if logical_only:
        result.update(support_policy=copy.deepcopy(support_policy),logical_face_only=True,depth_unresolved=True,
            geometry_fit_eligible=False,XYZ_binding_available=False,homography_eligible=False,PnP_eligible=False)
        result["support_summary"].update(physical_ambiguous_cells=int((supported&(physical_count>1)).sum()),
            same_face_multi_patch_consensus_cells=int((supported&(physical_count>1)&(count==1)).sum()),
            distinct_face_ambiguous_cells=int(ambiguous.sum()))
    return result


@dataclass(frozen=True)
class ProjectedFacePromptThresholds:
    inner_margin_px: float
    minimum_support_points: int
    minimum_point_distance_px: float
    adjacent_negative_points: int
    background_negative_points: int

    def __post_init__(self):
        if (isinstance(self.inner_margin_px,bool) or not np.isfinite(self.inner_margin_px)
                or self.inner_margin_px<=0 or isinstance(self.minimum_point_distance_px,bool)
                or not np.isfinite(self.minimum_point_distance_px) or self.minimum_point_distance_px<=0):
            raise ValueError("Explicit positive native-pixel interior margin/spacing required")
        if type(self.minimum_support_points) is not int or self.minimum_support_points<4:
            raise ValueError("At least four distinct projected support points required")
        if (type(self.adjacent_negative_points) is not int or type(self.background_negative_points) is not int
                or self.adjacent_negative_points<1 or self.background_negative_points<1
                or self.adjacent_negative_points+self.background_negative_points>4):
            raise ValueError("Both adjacent-face and outside-body negatives required,2..4 total")

def _polygon_interior_distance(native_xy,quad):
    """Exact convex-quad membership and segment distance; no mask filling output."""
    q=np.asarray(quad,float)
    if q.shape!=(4,2) or not np.isfinite(q).all():raise ValueError("Finite native projected face quad required")
    p=np.asarray(native_xy,float);cross=[];distance=[]
    for a,b in zip(q,np.roll(q,-1,axis=0)):
        e=b-a;length=float(e@e)
        if length<=1e-12:return np.zeros(p.shape[:-1],bool),np.zeros(p.shape[:-1])
        rel=p-a;cross.append(rel[...,0]*e[1]-rel[...,1]*e[0])
        alpha=np.clip(np.sum(rel*e,axis=-1)/length,0,1)
        distance.append(np.linalg.norm(rel-alpha[...,None]*e,axis=-1))
    cross=np.stack(cross);inside=(cross>=-1e-9).all(0)|(cross<=1e-9).all(0)
    return inside,np.min(np.stack(distance),axis=0)

def _spread_points(cells,native_xy,interior,count,spacing,excluded=()):
    """Deduplicate native pixels; deepest first, then maximin spatial spread."""
    candidates=native_xy[cells].astype(int);scores=np.asarray(interior)[cells]
    if not len(candidates):return None
    unique,index=np.unique(candidates,axis=0,return_index=True);scores=scores[index]
    previous=[np.asarray(p,float) for p in excluded];chosen=[]
    if previous:
        separation=np.min(np.linalg.norm(unique[:,None]-np.stack(previous)[None],axis=2),axis=1)
    else:separation=np.full(len(unique),np.inf)
    available=separation>=spacing
    for _ in range(count):
        if not available.any():return None
        criterion=scores if not previous and not chosen else separation
        idx=int(np.argmax(np.where(available,criterion,-np.inf)))
        point=unique[idx];chosen.append(point.tolist())
        separation=np.minimum(separation,np.linalg.norm(unique-point,axis=1))
        available&=separation>=spacing;available[idx]=False
    return chosen

def propose_projected_face_region_prompts(face_inputs,*,frame,video_context,body_binding,body_mask,
                                         patch_candidates,contract,expected_models,thresholds):
    """Explicit virtual-face hypothesis fallback. Does NOT use XYZ plane support.

    Geometry/prior, RGB/body/native owner authentication remains identical to
    surface strategy. Only ephemeral candidate point sets are formed; no face
    mask is returned or represented as a native SAM result.
    """
    if not isinstance(thresholds,ProjectedFacePromptThresholds) or not isinstance(patch_candidates,FacePatchCandidates):
        raise TypeError("Explicit projected-face thresholds and bound provenance required")
    if patch_candidates._verified is not _PATCH_VERIFIED or patch_candidates.contract_sha256!=contract.source_sha256:
        raise ValueError("Patch provenance must be verified against the same contract")
    roi,body=_checked(face_inputs,frame,video_context,body_binding,body_mask,contract,expected_models)
    yy,xx=np.indices((112,112));grid=np.stack((xx,yy,np.ones_like(xx)),-1)
    native=np.rint((grid@np.linalg.inv(roi.affine(112)).T)[...,:2]).astype(int)
    h,w=body_mask.shape
    in_image=(native[...,0]>=0)&(native[...,0]<w)&(native[...,1]>=0)&(native[...,1]<h)
    valid=in_image&(face_inputs.valid_image_support112[0]>0)
    body_cells=np.zeros((112,112),bool);body_cells[valid]=body_mask[native[valid,1],native[valid,0]]
    # Exact native body-interior distance on a bounded foreground ROI.
    body_distance=np.zeros((112,112),float)
    ys,xs=np.nonzero(body_mask)
    if len(xs):
        l,r=int(xs.min()),int(xs.max())+1;t,b=int(ys.min()),int(ys.max())+1
        dist=distance_transform_edt(np.pad(body_mask[t:b,l:r],1))[1:-1,1:-1]
        selected=valid&(native[...,0]>=l)&(native[...,0]<r)&(native[...,1]>=t)&(native[...,1]<b)
        body_distance[selected]=dist[native[selected,1]-t,native[selected,0]-l]
    geometry=face_inputs.geometry_prior["faces"]
    supports={};depth_scores={};raw_polygons={}
    for face in geometry:
        fid=face["face_id"];cells=np.zeros_like(valid);score=np.zeros((112,112),float)
        if face["geometric_front_facing"] and face["positive_depth"] and face["intersects_image"]:
            inside,poly_distance=_polygon_interior_distance(native,face["quad_native_uv"])
            score=np.minimum(poly_distance,body_distance)
            cells=valid&body_cells&inside&(score>=thresholds.inner_margin_px)
        supports[fid]=cells;depth_scores[fid]=score
        raw_polygons[fid]=face["quad_native_uv"]
    multiplicity=sum(supports.values(),np.zeros((112,112),int))
    supports={fid:mask&(multiplicity==1) for fid,mask in supports.items()}
    quads={f["face_id"]:set(f["corner_ids"]) for f in geometry}
    dependence={"surface_receipt":face_inputs.metadata["surface_receipt"],"models":copy.deepcopy(expected_models),
        "body":copy.deepcopy(body),"body_mask_binding":asdict(body_binding),"body_mask_sha256":array_digest(body_mask),
        "project_replay_identity":video_context.replay_identity,"roi_id":face_inputs.metadata["roi_id"],
        "roi_spec":roi.record(),"K_native":face_inputs.geometry_prior["K_native"],
        "pose":{"R":face_inputs.geometry_prior["R"],"t_m":face_inputs.geometry_prior["t_m"]},
        "face_contract_sha256":contract.source_sha256,"patch_candidates_ref":json.loads(patch_candidates.reference_json),
        "predicted_XYZ_sha256":array_digest(face_inputs.xyz_object_m),"frame_id":body_binding.frame_id,
        "pts":body_binding.pts,"time_base":list(body_binding.time_base),"pixel_space_id":body_binding.pixel_space_id,
        "source_strategy":"projected_cuboid_face_prior","patch_ref_role":"provenance_only_not_used_for_point_selection",
        "dense_XYZ_used_for_point_selection":False}
    hypotheses=[];diagnostics=[]
    for face in geometry:
        fid=face["face_id"];cells=supports[fid]
        count=len(np.unique(native[cells],axis=0))
        item={"face_id":fid,"physical_patch_ids":[],"logical_face_candidates":[fid],
              "source_strategy":"projected_cuboid_face_prior","physical_surface_verified":False,
              "geometry_fit_eligible":False,"homography_eligible":False,"PnP_eligible":False,
              "native_face_mask_produced":False,"support_cells":int(cells.sum()),
              "distinct_native_support_points":count,"projected_quad_native_uv":raw_polygons[fid],"status":"no_proposal"}
        if not face["geometric_front_facing"]:item["reason"]="not_geometric_front_facing"
        elif count<thresholds.minimum_support_points:item["reason"]="insufficient_projected_face_body_intersection"
        else:
            positive=_spread_points(cells,native,depth_scores[fid],4,thresholds.minimum_point_distance_px)
            adjacent=np.zeros_like(valid);adjscore=np.zeros((112,112))
            for other in geometry:
                oid=other["face_id"]
                if oid!=fid and other["geometric_front_facing"] and len(quads[fid]&quads[oid])==2:
                    adjacent|=supports[oid];adjscore=np.maximum(adjscore,np.where(supports[oid],depth_scores[oid],0))
            neighbor=None if positive is None else _spread_points(adjacent,native,adjscore,
                thresholds.adjacent_negative_points,thresholds.minimum_point_distance_px,positive)
            background=None if neighbor is None else _spread_points(valid&~body_cells,native,np.ones((112,112)),
                thresholds.background_negative_points,thresholds.minimum_point_distance_px,positive+neighbor)
            if positive is None or neighbor is None or background is None:item["reason"]="insufficient_distinct_positive_adjacent_or_background_points"
            else:
                pts=positive+neighbor+background
                item.update(status="prompt_hypothesis",reason="projected_virtual_face_intersection_only",
                    points_native=pts,points_xy01=(np.asarray(pts)/[w,h]).tolist(),
                    labels=[1]*4+[0]*(len(neighbor)+len(background)),
                    point_sources=["projected_cuboid_face_inside_current_body"]*4+
                        ["adjacent_visible_projected_face_inside_current_body"]*len(neighbor)+
                        ["outside_current_body_within_valid_ROI_not_background_GT"]*len(background),
                    support_grid_sha256=array_digest(cells))
        item["parent_dependency_identity"]=_digest(dependence);item["hypothesis_identity"]=_digest(item)
        hypotheses.append(item)
        diagnostics.append({"face_id":fid,"geometric_front_facing":face["geometric_front_facing"],
            "status":"prompt_hypotheses_available" if item["status"]=="prompt_hypothesis" else "unknown_or_no_proposal",
            "hypothesis_ids":[item["hypothesis_identity"]] if item["status"]=="prompt_hypothesis" else [],
            "physical_patch_candidates":[],"observed_face_certified":False})
    return {"schema":"action4.face_prompt_hypotheses.v1","status":"PILOT_UNCALIBRATED_NOT_FACE_ACCEPTANCE",
        "source_strategy":"projected_cuboid_face_prior","dependencies":dependence,"thresholds":asdict(thresholds),
        "physical_patch_ids":[],"patch_ref_role":"provenance_only_not_used_for_point_selection",
        "thresholds_calibrated":False,"physical_surface_verified":False,"geometry_fit_eligible":False,
        "homography_eligible":False,"PnP_eligible":False,"semantic_gt":False,"face_tracking_accepted":False,
        "native_inference_executed":False,"native_face_mask_produced":False,"GT_used":False,
        "corner_probabilities_available":False,"face_diagnostics":diagnostics,"hypotheses":hypotheses,
        "patch_diagnostics":[],"support_summary":{"ambiguous_projected_face_cells":int((multiplicity>1).sum()),
            "body_supported_cells":int(body_cells.sum()),"valid_image_cells":int(valid.sum())},
        "candidate_policy":"4 spread points from interior projected virtual-face/body/validROI intersection; not a physical face boundary or a SAM mask"}
