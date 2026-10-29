"""Authenticated current video support for provisional material input only."""
from dataclasses import dataclass,field
from fractions import Fraction
from pathlib import Path
import hashlib,json,os,io
import numpy as np
from .action4_paths import PROJECT_ROOT as ROOT,contained_path
from .action4_sam_roi_inputs import verify_image_frame,array_digest
from .action4_video_frame_binding import validate_video_binding
from .action4_video_roi_context import VideoROIContext,validate_project_body_registration,rgb_digest

_TOKEN=object()
def _json(x):return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False)
def _bound(ref,evidence):
    if not isinstance(ref,dict) or set(ref)!={"path","sha256"}:raise ValueError("Bound video evidence required")
    p=contained_path(ROOT,ref["path"],exists=True);raw=p.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=ref["sha256"]:raise ValueError("Video evidence SHA mismatch")
    evidence.append((str(p),ref["sha256"]))
    return p,raw
def _read(ref,evidence):return json.loads(_bound(ref,evidence)[1])
def _raw(receipt,folder,evidence):
    arr,arr_bytes=_bound({"path":receipt["arrays"],"sha256":receipt["sha256"]},evidence)
    header,header_bytes=_bound({"path":receipt["header"],"sha256":receipt["header_sha256"]},evidence)
    if arr!=folder/"raw.npz" or header!=folder/"raw.json":raise ValueError("Video raw response directory differs")
    # Decode the bytes whose identity was checked, not a second path read.
    with np.load(io.BytesIO(arr_bytes),allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
    used=set()
    def restore(v):
        if isinstance(v,dict) and "array_key" in v:
            k=v["array_key"]
            if k not in arrays or k in used or set(v)!={"array_key","shape","dtype"}:raise ValueError("Bad raw array descriptor")
            a=arrays[k];used.add(k)
            if list(a.shape)!=v["shape"] or str(a.dtype)!=v["dtype"]:raise ValueError("Raw descriptor differs")
            return a
        if isinstance(v,dict):return {k:restore(x) for k,x in v.items()}
        if isinstance(v,list):return [restore(x) for x in v]
        return v
    raw=restore(json.loads(header_bytes))
    if used!=set(arrays):raise ValueError("Unreferenced raw arrays")
    return raw

@dataclass(frozen=True)
class VerifiedVideoMaterialSupport:
    payload_json:str
    mask_shape:tuple
    mask_bytes:bytes
    source_frame:object
    frame_binding:object
    evidence:tuple
    _token:object=field(default=None,init=False,repr=False,compare=False)
    _pid:object=field(default=None,init=False,repr=False,compare=False)

    def __reduce_ex__(self,protocol):raise TypeError("Revalidate native video support in each process")
    @property
    def sources(self):return json.loads(self.payload_json)["sources"]

    @classmethod
    def from_current_response(cls,*,frame,source_frame,frame_binding,surface,context,
                              expected_models,track_id,track_generation,native_status_ref,owner_ledger_ref):
        if not isinstance(context,VideoROIContext):raise ValueError("Typed current VideoROIContext required")
        f=verify_image_frame(frame);source=verify_image_frame(source_frame)
        video=surface.metadata.get("video_context")
        if not isinstance(video,dict):raise ValueError("Actual video surface snapshot required")
        fid=context.frame_id;ledger=json.loads(context.source_ledger_json)
        pts=ledger.get("pts");tb=ledger.get("time_base")
        if tb is None:raise ValueError("Video project registration/time domain required")
        timing=validate_video_binding(frame_binding,source,fid,pts,str(Fraction(*tb)))
        if f.source_sha256!=source.source_sha256 or f.variant_sha256!=source.variant_sha256 or not np.array_equal(f.pixels,source.pixels):
            raise ValueError("Video namespace alias changed actual RGB pixels")
        from .action4_surface_snapshot import validate_snapshot
        validate_snapshot(surface,frame=f,expected_models=expected_models)
        K=surface.arrays["K_native"];chw=np.moveaxis(f.pixels,-1,0)
        rows=context.select_for_a0(context.consume(chw,K,b0_sha256=expected_models["B0"]["sha256"]))
        selected=[r for r in rows if r["track_id"]==track_id and r["prediction_id"]==surface.metadata["prediction_id"]]
        if len(selected)!=1:raise ValueError("Surface/current project track differs")
        if (video["frame_id"]!=fid or video["replay_identity"]!=context.replay_identity
                or video["rgb_identity"]!=rgb_digest(chw)
                or video["source_ledger"]!=ledger):
            raise ValueError("Surface was not captured from this exact current video ledger")
        owner=(ledger.get("native_owners") or {}).get(track_id)
        event=(ledger.get("project_body_registrations") or {}).get(track_id)
        validate_project_body_registration(event,ledger.get("project_body_scope"),track_id=track_id,frame_id=fid,
            pts=pts,time_base=tb,pixel_space_id=f.pixel_space_id,replay_identity=context.replay_identity,native_owner=owner)
        if type(track_generation) is not int or event["project_generation"]!=track_generation:
            raise ValueError("Project body generation differs")
        evidence=[]
        owner_doc=_read(owner_ledger_ref,evidence)
        if owner_ledger_ref["sha256"]!=owner["ledger_sha"]:raise ValueError("Native owner ledger SHA differs")
        oid=owner["native_obj_id"];gen=owner["generation"];sid=owner["session_id"]
        mapping={"track_00":0,"track_01":1}
        if gen==0:
            if owner_doc.get("session_id")!=sid or owner_doc.get("joint_obj_map")!={"0":"track_00","1":"track_01"}:
                raise ValueError("Initial joint native owner differs")
        elif (owner_doc.get("new_session")!=sid or owner_doc.get("proposed_generation")!=gen
              or owner_doc.get("native_obj_map")!=mapping or owner_doc.get("outcome")!="native_bound"):
            raise ValueError("Reanchor current native owner differs")
        if mapping.get(track_id)!=oid:raise ValueError("Project/native instance mapping differs")
        status=_read(native_status_ref,evidence)
        for k,v in {"global_frame_id":fid,"pts":pts,"time_base":tb,"RGB_sha256":f.source_sha256,
                    "pixel_space_id":f.pixel_space_id,"session_id":sid,"generation":gen}.items():
            if status.get(k)!=v:raise ValueError("Current native status binding differs: "+k)
        if status.get("raw") is None:raise ValueError("Missing current native response")
        status_path=contained_path(ROOT,native_status_ref["path"],exists=True)
        owner_path=contained_path(ROOT,owner_ledger_ref["path"],exists=True)
        expected_status=(owner_path.parent/("frame_%06d"%fid)/"STATUS.json" if gen==0 else
                         owner_path.parents[2]/("generation_%03d"%gen)/("frame_%06d"%fid)/"STATUS.json")
        if status_path!=expected_status:raise ValueError("Status is outside authenticated owner/frame directory")
        if gen and status.get("native_obj_map")!=mapping:raise ValueError("Reanchor instance map differs")
        folder=status_path.parent
        raw=_raw(status["raw"],folder,evidence);out=raw["outputs"]
        ids=np.asarray(out["out_obj_ids"]);probs=np.asarray(out["out_probs"]);masks=np.asarray(out["out_binary_masks"])
        if (raw["frame_index"]!=fid or ids.ndim!=1 or ids.dtype.kind not in "iu"
            or len(set(ids.tolist()))!=len(ids) or ids.tolist().count(oid)!=1
            or probs.shape!=(len(ids),) or not np.isfinite(probs).all()
            or masks.dtype!=np.bool_ or masks.shape!=(len(ids),*f.pixels.shape[:2])):
            raise ValueError("Actual current raw frame/object/mask malformed")
        i=ids.tolist().index(oid);mask=masks[i]
        if not 0<=probs[i]<=1 or not mask.any() or ledger.get("native_status",{}).get(track_id)!="target_present":
            raise ValueError("Current predicted visible support is unavailable")
        admitted=selected[0].get("sam_support_evidence",{}).get("receipt")
        if admitted and admitted["mask_sha256"]!=array_digest(mask):raise ValueError("SAM admission/raw mask differs")
        payload={"schema":"action4.video_material_support.v1","track_id":track_id,"track_generation":track_generation,
            "frame_id":fid,"pts":pts,"time_base":str(Fraction(*tb)),"image_id":f.image_id,
            "pixel_space_id":f.pixel_space_id,"RGB_sha256":f.source_sha256,"RGB_pixels_sha256":f.variant_sha256,
            "surface_receipt":surface.metadata["receipt"],"models":expected_models,"mask_sha256":array_digest(mask),
            "native_session_id":sid,"native_obj_id":oid,"native_generation":gen,"timing":timing,
            "sources":{"schema":"action4.video_material_support_sources.v1","b0_model_sha256":expected_models["B0"]["sha256"],
                "replay_identity":context.replay_identity,"native_owner_ledger_sha256":owner["ledger_sha"],
                "native_status_sha256":native_status_ref["sha256"]},
            "project_body_registration":event,"scoped_body_id":event["scoped_body_id"],
            "physical_or_semantic_identity_verified":False,
            "support_semantics":"current_predicted_visible_body_not_full_body_GT","geometry_accepted":False}
        result=cls(_json(payload),mask.shape,mask.tobytes(),source,frame_binding,tuple(evidence))
        object.__setattr__(result,"_token",_TOKEN);object.__setattr__(result,"_pid",os.getpid())
        return result

    def consume(self,snapshot,frame,*,expected_models,expected_sources,track_id,track_generation,frame_id,pts,time_base):
        if self._token is not _TOKEN or self._pid!=os.getpid():raise ValueError("Validated current video material input required")
        p=json.loads(self.payload_json);f=verify_image_frame(frame)
        validate_video_binding(self.frame_binding,self.source_frame,frame_id,pts,time_base)
        actual=(track_id,track_generation,frame_id,pts,str(Fraction(time_base)),f.image_id,f.pixel_space_id,
                f.source_sha256,f.variant_sha256,snapshot.metadata["receipt"])
        expected=tuple(p[k] for k in ("track_id","track_generation","frame_id","pts","time_base","image_id",
                                    "pixel_space_id","RGB_sha256","RGB_pixels_sha256","surface_receipt"))
        if actual!=expected or p["models"]!=expected_models or p["sources"]!=expected_sources:
            raise ValueError("Changed video frame/PTS/RGB/model/instance/generation/support contract")
        for path,sha in self.evidence:
            if hashlib.sha256(contained_path(ROOT,path,exists=True).read_bytes()).hexdigest()!=sha:
                raise ValueError("Current video evidence changed")
        mask=np.frombuffer(self.mask_bytes,dtype=np.bool_).reshape(self.mask_shape)
        if array_digest(mask)!=p["mask_sha256"]:raise ValueError("Video support pixels changed")
        return mask.copy(),p
