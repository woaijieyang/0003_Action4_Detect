"""Opt-in, prediction-only surface snapshots. Not material/CAD correspondence acceptance."""
from dataclasses import dataclass
import hashlib,json,os
from pathlib import Path
import numpy as np
from data.action4_paths import PROJECT_ROOT as ROOT,contained_path
from data.action4_sam_roi_inputs import verify_image_frame,array_digest,_digest
from data import DataAdapter

SCHEMA="action4.surface_snapshot.v1"
VIDEO_SCHEMA="action4.surface_snapshot.video_roi.v1"
SHAPES={"xyz_normalized":(3,112,112),"mask_logits":(2,112,112),"region_logits":(65,112,112),
        "valid_image_support112":(1,112,112),"A112":(3,3),"A448":(3,3),"K_native":(3,3),"K112":(3,3),"canonical_m":(8,3),
        "low_m":(3,),"extent_m":(3,),"rotation":(3,3),"translation_m":(3,),"bbox_native":(4,)}
META_KEYS={"schema","image_id","prediction_id","grid_index","source_sha256","variant_id","variant_sha256",
           "pixel_space_id","native_shape","roi_id","models","sam_input","semantics","array_sha256","receipt"}
SEMANTICS={"xyz":"estimated_unclamped_normalized_object_XYZ; X_m=low_m+xyz_normalized*extent_m",
           "masks":["visible_logit","amodal_logit"],"regions":"background0_plus_64_region_logits",
           "pose":"raw_network_object_to_camera_metres","material_status":"unconfirmed_no_material_binding",
           "identity":"frame_local_prediction_not_persistent_track","precision":"not_measured"}

VIDEO_SEMANTICS=dict(SEMANTICS,identity="frame_and_project_track_ROI; actual_B0_grid_optional; not_material_confirmation")

def _video_context_identity(context,frame,K,prediction,models,sam_input=None):
    from data.action4_video_roi_context import VideoROIContext
    if not isinstance(context,VideoROIContext):raise ValueError("Typed video ROI context required for video surface")
    rows=context.consume(np.moveaxis(frame.pixels,-1,0),K,b0_sha256=models["B0"]["sha256"])
    if prediction not in context.select_for_a0(rows):raise ValueError("Surface prediction not owned by current video context")
    ledger=json.loads(context.source_ledger_json) if context.source_ledger_json is not None else None
    if not isinstance(ledger,dict) or ledger.get("frame_id")!=context.frame_id:raise ValueError("Video source ledger required")
    value={"frame_id":context.frame_id,"rgb_identity":context.rgb_identity,"replay_identity":context.replay_identity,
        "B0_sha256":context.b0_sha256,"K_native":list(context.K_native),"candidate":json.loads(json.dumps(prediction)),
        "source_ledger":ledger,"SAM_mask_features_enabled":sam_input is not None}
    value["identity"]=_canonical(value)
    return value

def _validate_video_identity(m,a,frame):
    v=m["video_context"];p=v.get("candidate",{});ledger=v.get("source_ledger",{})
    if (set(v)!={"frame_id","rgb_identity","replay_identity","B0_sha256","K_native","candidate","source_ledger","SAM_mask_features_enabled","identity"}
            or v["identity"]!=_canonical({k:x for k,x in v.items() if k!="identity"})
            or type(v["frame_id"]) is not int or v["frame_id"]<0 or ledger.get("frame_id")!=v["frame_id"]
            or type(v["SAM_mask_features_enabled"]) is not bool):
        raise ValueError("Video surface context integrity differs")
    if v["SAM_mask_features_enabled"]:
        source=m["sam_input"]
        if not isinstance(source,dict) or source.get("protocol")!="action4.video_sam_features.v1":
            raise ValueError("Enhanced video surface requires current native feature provenance")
        upstream=source.get("sources",{});receipt=source.get("current_native_receipt",{})
        if (source.get("frame_id")!=v["frame_id"] or source.get("track_id")!=p.get("track_id")
                or source.get("roi_decision_id")!=p.get("roi_decision_id")
                or source.get("bbox_for_a0")!=p.get("bbox_for_a0")
                or upstream.get("b0_model_sha256")!=v["B0_sha256"]
                or upstream.get("replay_identity")!=v["replay_identity"]
                or receipt.get("frame_id")!=v["frame_id"] or receipt.get("RGB_sha256")!=m["source_sha256"]
                or receipt.get("pixel_space_id")!=m["pixel_space_id"]
                or source.get("still_image_cache_used") is not False):
            raise ValueError("Enhanced surface frame/instance/ROI/source mismatch")
        from data.action4_roi_contract import RoiSpec
        roi=source.get("roi_spec",{})
        spec=RoiSpec.from_full_box(a["bbox_native"],roi.get("expansion"),roi.get("output_size"))
        if (roi!=spec.record() or not np.array_equal(spec.affine(),a["A448"])
                or not np.array_equal(spec.affine(112),a["A112"])
                or receipt.get("pts")!=ledger.get("pts")
                or receipt.get("time_base")!=ledger.get("time_base")):
            raise ValueError("Enhanced surface ROI sampling or native timing differs")
        _digest(upstream.get("sam_model_sha256"))
        reason=source.get("availability_reason")
        if reason not in ("current_video_native_mask","current_video_mask_unavailable","current_video_mask_empty_after_resolved_ROI"):
            raise ValueError("Explicit current mask availability reason required")
        if reason=="current_video_native_mask":
            _digest(receipt.get("mask_sha256",{}).get(p["track_id"]))
    elif m["sam_input"] is not None:
        raise ValueError("Baseline video surface cannot claim unused SAM features")
    for key in ("rgb_identity","replay_identity","B0_sha256"):_digest(v[key])
    if not isinstance(p.get("track_id"),str) or not p["track_id"]:raise ValueError("Explicit video track ID required")
    if (m["prediction_id"]!=f"video:{p['track_id']}:frame:{v['frame_id']}" or p.get("prediction_id")!=m["prediction_id"]
            or p.get("grid_index")!=m["grid_index"] or not np.array_equal(a["bbox_native"],p.get("bbox_full_xyxy"))
            or v["B0_sha256"]!=m["models"]["B0"]["sha256"]
            or not np.array_equal(np.asarray(v["K_native"]).reshape(3,3),a["K_native"])):
        raise ValueError("Video surface frame/track/ROI/model differs")
    if "bbox_for_a0" in p and not np.array_equal(a["bbox_native"],p["bbox_for_a0"]):
        raise ValueError("Surface differs from actually selected A0 ROI")
    if p.get("temporal_status")=="observed":
        grid=m["grid_index"]
        if type(grid) is not int or grid<0 or p.get("source_detection_ids")!=[f"b0:frame{v['frame_id']}:grid:{grid}"]:
            raise ValueError("Observed surface must retain actual current B0 source")
    elif p.get("temporal_status")=="sam_supported":
        receipt=p.get("sam_support_evidence",{}).get("receipt",{})
        owners=ledger.get("native_owners")
        if owners is not None:
            owner=owners.get(p["track_id"]) if isinstance(owners,dict) else None
            if not isinstance(owner,dict):
                raise ValueError("SAM-maintained surface lacks per-track native owner")
            _digest(owner.get("ledger_sha"))
            if (receipt.get("session_id")!=owner.get("session_id")
                    or receipt.get("native_obj_id")!=owner.get("native_obj_id")
                    or receipt.get("mask_generation")!=owner.get("generation")
                    or receipt.get("track_generation")!=owner.get("generation")
                    or receipt.get("replay_identity")!=v["replay_identity"]
                    or receipt.get("ledger_sha256")!=_canonical(owners)):

                raise ValueError("SAM-maintained surface per-track native binding differs")
            expected_session=owner["session_id"];expected_generation=owner["generation"]
        else:
            expected_session=ledger.get("native_session_id");expected_generation=ledger.get("mask_generation")
            if not isinstance(expected_session,str) or type(expected_generation) is not int:
                raise ValueError("Legacy SAM-maintained surface requires explicit session/generation")
        if (m["grid_index"] is not None or p.get("score") is not None or p.get("source_detection_ids")!=[]
                or p.get("B0_current_observation") is not False
                or p.get("sam_support_admission_id")!=_canonical(p.get("sam_support_evidence"))
                or receipt.get("session_id")!=expected_session
                or receipt.get("mask_generation")!=expected_generation
                or receipt.get("frame_id")!=v["frame_id"] or receipt.get("track_id")!=p["track_id"]):
            raise ValueError("SAM-maintained surface cannot invent B0 identity or change generation")
    else:raise ValueError("Unobserved/unadmitted video surface refused")
    if frame is not None:
        from data.action4_video_roi_context import rgb_digest
        if rgb_digest(np.moveaxis(frame.pixels,-1,0))!=v["rgb_identity"]:raise ValueError("Video surface RGB differs")

def _canonical(v):
    return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def _stat(p):
    s=p.stat()
    return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)

def bind_model_file(path):
    p=contained_path(ROOT,path,exists=True).resolve(strict=True)
    return {"path":str(p.relative_to(ROOT)),"stat":_stat(p)}

def check_model_file(binding):
    p=contained_path(ROOT,binding["path"],exists=True)
    if _stat(p)!=binding["stat"]:raise ValueError("Loaded model file changed; reload before surface export")
    return p

def model_identities(bindings):
    if not bindings or not {"A0","B0"}<=set(bindings):raise ValueError("Loaded model provenance unavailable")
    result={}
    for role,binding in bindings.items():
        p=check_model_file(binding)
        if "sha256" not in binding:
            with p.open("rb") as f:
                s=os.fstat(f.fileno())
                if (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)!=binding["stat"]:
                    raise ValueError("Model replaced before hashing")
                digest=hashlib.file_digest(f,"sha256").hexdigest()
            check_model_file(binding)
            binding["sha256"]=digest
        result[role]={"path":binding["path"],"sha256":binding["sha256"]}
    return result

def checked_frame(frame,rgb):
    f=verify_image_frame(frame)
    if not isinstance(rgb,np.ndarray) or rgb.dtype!=np.uint8 or rgb.shape!=f.pixels.shape or not np.array_equal(rgb,f.pixels):
        raise ValueError("Surface export RGB differs from explicit native ImageFrame")
    if array_digest(rgb)!=f.variant_sha256:raise ValueError("Surface input RGB digest mismatch")
    return f

def roi_identity(image_id,prediction_id,bbox,A448):
    return _canonical({"image_id":image_id,"prediction_id":prediction_id,
                       "bbox_native":np.asarray(bbox).tolist(),"A448":np.asarray(A448).tolist()})

@dataclass(frozen=True)
class SurfaceSnapshot:
    metadata:dict
    arrays:dict

def _validate_coordinate_contract(a):
    for key in ("A448","A112"):
        matrix=a[key]
        if (matrix.shape!=(3,3) or not np.isfinite(matrix).all()
                or not np.array_equal(matrix[2],[0.,0.,1.])
                or matrix[0,0]<=0 or matrix[1,1]<=0
                or matrix[0,1]!=0 or matrix[1,0]!=0):
            raise ValueError("Axis-aligned positive-scale affine required: "+key)
    for key in ("K_native","K112"):
        matrix=a[key]
        if (matrix.shape!=(3,3) or not np.isfinite(matrix).all()
                or not np.array_equal(matrix[2],[0.,0.,1.])
                or matrix[0,0]<=0 or matrix[1,1]<=0):
            raise ValueError("Pinhole intrinsics with positive focal lengths required: "+key)
    bbox=a["bbox_native"]
    if bbox.shape!=(4,) or not np.isfinite(bbox).all() or bbox[2]<=bbox[0] or bbox[3]<=bbox[1]:
        raise ValueError("Native xyxy bbox must have positive width and height")

def validate_snapshot(snapshot,*,frame=None,expected_models=None,expected_prediction_id=None,expected_roi_id=None):
    if not isinstance(snapshot,SurfaceSnapshot):raise ValueError("Typed SurfaceSnapshot required")
    m,a=snapshot.metadata,snapshot.arrays
    video=m.get("schema")==VIDEO_SCHEMA
    if set(m)!=(META_KEYS|{"video_context"} if video else META_KEYS) or m["schema"]!=(VIDEO_SCHEMA if video else SCHEMA) or m["semantics"]!=(VIDEO_SEMANTICS if video else SEMANTICS):raise ValueError("Unknown snapshot metadata")
    if set(a)!=set(SHAPES):raise ValueError("Unexpected snapshot arrays")
    for key,shape in SHAPES.items():
        x=a[key]
        if not isinstance(x,np.ndarray) or x.shape!=shape or x.dtype.kind!="f" or not np.isfinite(x).all():
            raise ValueError("Invalid finite surface array: "+key)
    if not np.isin(a["valid_image_support112"],[0.,1.]).all():raise ValueError("Invalid image support")
    for key in ("source_sha256","variant_sha256"):_digest(m[key])
    for key in ("image_id","variant_id","pixel_space_id"):
        if not isinstance(m[key],str) or not m[key]:raise ValueError("Explicit source identity required")
    if not video and (type(m["grid_index"]) is not int or m["grid_index"]<0 or m["prediction_id"]!=f'b0:{m["image_id"]}:grid:{m["grid_index"]}'):
        raise ValueError("Surface prediction must bind actual B0 grid identity")
    if not isinstance(m["native_shape"],list) or len(m["native_shape"])!=2 or any(type(x) is not int or x<1 for x in m["native_shape"]):
        raise ValueError("Invalid native shape")
    if set(m["models"]) not in ({"A0","B0"},{"A0","B0","A0_backbone"}):raise ValueError("Explicit loaded model identities required")
    for b in m["models"].values():
        if set(b)!={"path","sha256"} or not isinstance(b["path"],str) or not b["path"]:raise ValueError("Model binding malformed")
        _digest(b["sha256"])
        contained_path(ROOT,b["path"],exists=False)
    if expected_models is not None and m["models"]!=expected_models:raise ValueError("Snapshot model identity mismatch")
    if m["sam_input"] is not None and not isinstance(m["sam_input"],dict):raise ValueError("Invalid optional SAM provenance")
    if video:_validate_video_identity(m,a,frame)
    _validate_coordinate_contract(a)
    for k in ("A448","A112","K_native","K112"):
        if abs(np.linalg.det(a[k]))<1e-12:raise ValueError("Singular coordinate map")
    down,_=DataAdapter.letterbox_affine((448,448),(112,112))
    if not np.allclose(a["A112"],down@a["A448"],rtol=0,atol=1e-8):raise ValueError("ROI scale mismatch")
    if not np.allclose(a["K112"],a["A112"]@a["K_native"],rtol=2e-6,atol=1e-5):raise ValueError("Crop intrinsics mismatch")
    if not np.array_equal(a["low_m"],a["canonical_m"].min(0)) or not np.array_equal(a["extent_m"],np.ptp(a["canonical_m"],axis=0)) or (a["extent_m"]<=0).any():
        raise ValueError("XYZ normalization contract mismatch")
    if m["roi_id"]!=roi_identity(m["image_id"],m["prediction_id"],a["bbox_native"],a["A448"]):raise ValueError("Crop identity mismatch")
    if m["array_sha256"]!={k:array_digest(v) for k,v in a.items()}:raise ValueError("Snapshot array integrity mismatch")
    if m["receipt"]!=_canonical({k:v for k,v in m.items() if k!="receipt"}):raise ValueError("Snapshot metadata integrity mismatch")
    if expected_prediction_id is not None and m["prediction_id"]!=expected_prediction_id:raise ValueError("Snapshot prediction identity mismatch")
    if expected_roi_id is not None and m["roi_id"]!=expected_roi_id:raise ValueError("Snapshot ROI identity mismatch")
    if frame is not None:
        f=verify_image_frame(frame)
        if (m["image_id"],m["pixel_space_id"],m["source_sha256"],m["variant_id"],m["variant_sha256"],m["native_shape"])!=(f.image_id,f.pixel_space_id,f.source_sha256,f.variant_id,f.variant_sha256,list(f.pixels.shape[:2])):
            raise ValueError("Snapshot source frame identity mismatch")
    return snapshot

def make_snapshot(*,frame,prediction,output,index,A112,A448,K_native,K112,canonical,models,sam_input=None,video_context=None):
    def cpu(v):
        return v.detach().cpu().numpy() if hasattr(v,"detach") else np.asarray(v)
    cad=cpu(canonical)
    _validate_coordinate_contract({"A448":np.asarray(A448),"A112":np.asarray(A112),
        "K_native":np.asarray(K_native),"K112":cpu(K112),
        "bbox_native":np.asarray(prediction["bbox_full_xyxy"],dtype=float)})
    yy,xx=np.meshgrid(np.arange(112),np.arange(112),indexing="ij")
    native=np.stack((xx,yy,np.ones_like(xx)),-1)@np.linalg.inv(A112).T
    nearest=np.rint(native[...,:2]).astype(np.int64)
    support=((nearest[...,0]>=0)&(nearest[...,0]<frame.pixels.shape[1])&
             (nearest[...,1]>=0)&(nearest[...,1]<frame.pixels.shape[0]))[None].astype(np.float32)
    a={"valid_image_support112":support,"xyz_normalized":cpu(output["surface"]["xyz"][index]),"mask_logits":cpu(output["surface"]["masks"][index]),
       "region_logits":cpu(output["surface"]["regions"][index]),"A112":np.asarray(A112),"A448":np.asarray(A448),
       "K_native":np.asarray(K_native),"K112":cpu(K112),"canonical_m":cad,"low_m":cad.min(0),"extent_m":np.ptp(cad,axis=0),
       "rotation":cpu(output["rotation"][index]),"translation_m":cpu(output["translation"][index]),
       "bbox_native":np.asarray(prediction["bbox_full_xyxy"],dtype=float)}
    a={k:np.frombuffer(np.ascontiguousarray(v).tobytes(),dtype=v.dtype).reshape(v.shape) for k,v in a.items()}
    grid=prediction.get("grid_index") if video_context is not None else int(prediction["grid_index"]);pid=prediction["prediction_id"]
    m={"schema":SCHEMA,"image_id":frame.image_id,"prediction_id":pid,"grid_index":grid,
       "source_sha256":frame.source_sha256,"variant_id":frame.variant_id,"variant_sha256":frame.variant_sha256,
       "pixel_space_id":frame.pixel_space_id,"native_shape":list(frame.pixels.shape[:2]),"roi_id":roi_identity(frame.image_id,pid,a["bbox_native"],a["A448"]),
       "models":json.loads(json.dumps(models)),"sam_input":sam_input,"semantics":SEMANTICS,
       "array_sha256":{k:array_digest(v) for k,v in a.items()}}
    if video_context is not None:
        m.update(schema=VIDEO_SCHEMA,semantics=VIDEO_SEMANTICS,video_context=_video_context_identity(video_context,frame,K_native,prediction,models,sam_input=sam_input))
    m["receipt"]=_canonical(m)
    return validate_snapshot(SurfaceSnapshot(m,a),frame=frame,expected_models=models)

def save_snapshot(path,snapshot):
    validate_snapshot(snapshot)
    p=contained_path(ROOT,path,exists=False)
    # Exclusive creation preserves existing evidence; caller creates its owned directory.
    with p.open("xb") as f:
        np.savez_compressed(f,metadata=np.array(json.dumps(snapshot.metadata,sort_keys=True,allow_nan=False)),**snapshot.arrays)
    return p

def load_snapshot(path,*,frame=None,expected_models=None,expected_prediction_id=None,expected_roi_id=None):
    with np.load(contained_path(ROOT,path,exists=True),allow_pickle=False) as f:
        if set(f.files)!=set(SHAPES)|{"metadata"}:raise ValueError("Unexpected NPZ fields")
        md=f["metadata"]
        if md.ndim or md.dtype.kind!="U":raise ValueError("Scalar Unicode JSON metadata required")
        s=SurfaceSnapshot(json.loads(str(md)),{k:f[k].copy() for k in SHAPES})
    return validate_snapshot(s,frame=frame,expected_models=expected_models,expected_prediction_id=expected_prediction_id,expected_roi_id=expected_roi_id)
