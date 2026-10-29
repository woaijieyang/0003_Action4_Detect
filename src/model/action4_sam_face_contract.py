"""Pilot A0-face point seeds and native mask diagnostics; never semantic GT."""
from dataclasses import dataclass, asdict
import hashlib,json,math
import numpy as np
from model.action4_sam_response_adapter import validate_raw_response, MissingOutput

def _hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def _ref(value):
    if not isinstance(value,dict) or set(value)!={"path","sha256"} or not isinstance(value["path"],str) or not value["path"]:
        raise ValueError("Explicit bound receipt required")
    _sha(value["sha256"])
def _sha(value):
    if not isinstance(value,str) or len(value)!=64 or any(x not in "0123456789abcdef" for x in value):
        raise ValueError("SHA256 required")
def _integer(v):
    if type(v) is not int or v<0:raise ValueError("Nonnegative integer identity required")

def persistent_face_identity(body_track_id,body_generation,face_id):
    if not isinstance(body_track_id,str) or not body_track_id.strip():
        raise ValueError("Established body track identity required")
    _integer(body_generation);_integer(face_id)
    if face_id>=6:raise ValueError("Canonical face0..5 required")
    # Excludes frame, screen position, SAM session and face-generation IDs.
    return "face_entity:"+_hash([body_track_id,body_generation,face_id])


def persistent_corner_identity(body_track_id,body_generation,corner_id):
    if not isinstance(body_track_id,str) or not body_track_id.strip():
        raise ValueError("Established body track identity required")
    _integer(body_generation);_integer(corner_id)
    if corner_id>=8:raise ValueError("Canonical corner0..7 required")
    return "corner_entity:"+_hash([body_track_id,body_generation,corner_id])


@dataclass(frozen=True)
class FaceSeedBinding:
    body_track_id: str
    body_generation: int
    face_id: int
    face_generation: int
    region_identity: str
    local_index: int
    global_frame_id: int
    pts: int
    time_base: tuple
    pixel_space_id: str
    a0_model_sha256: str
    rgb_sha256: str
    K_native: tuple
    pose_receipt: dict
    patch_ref: dict
    prompt_ref: dict
    box_xywh01: tuple
    source: str = "A0_face_hypothesis"
    semantic_face_verified: bool = False

    def __post_init__(self):
        if self.source!="A0_face_hypothesis" or self.semantic_face_verified is not False:
            raise ValueError("Only unverified A0 face hypothesis source allowed")
        for v in (self.body_generation,self.face_id,self.face_generation,self.local_index,self.global_frame_id):
            _integer(v)
        if self.face_id>=6 or not isinstance(self.body_track_id,str) or not self.body_track_id.strip():
            raise ValueError("Explicit body identity and face0..5 required")
        if type(self.pts) is not int or len(self.time_base)!=2 or any(type(v) is not int or v<=0 for v in self.time_base):
            raise ValueError("Exact PTS/time base required")
        if not self.region_identity or not self.pixel_space_id:raise ValueError("Native region/pixel identity required")
        _sha(self.a0_model_sha256);_sha(self.rgb_sha256)
        for ref in (self.pose_receipt,self.patch_ref,self.prompt_ref):_ref(ref)
        K=np.asarray(self.K_native,float)
        if K.size!=9 or not np.isfinite(K).all() or np.linalg.det(K.reshape(3,3))==0:
            raise ValueError("Finite invertible native K required")
        object.__setattr__(self,"K_native",tuple(K.reshape(-1)))
        box=tuple(self.box_xywh01)
        if len(box)!=4 or not all(math.isfinite(v) for v in box):
            raise ValueError("Finite normalized support box required")
        x,y,w,h=box
        if not (0<=x<1 and 0<=y<1 and w>0 and h>0 and x+w<=1 and y+h<=1):
            raise ValueError("Face support bounds outside frame")
        object.__setattr__(self,"box_xywh01",box)

    @property
    def persistent_face_id(self):
        """Stable face entity within an established body identity, not a semantic-accuracy claim."""
        return persistent_face_identity(self.body_track_id,self.body_generation,self.face_id)

    @property
    def session_track_id(self):
        """Existing native-session lineage, intentionally changes on face-session restart."""
        return "face:"+_hash([self.body_track_id,self.body_generation,self.face_id,self.face_generation])

    @property
    def track_id(self):
        # Preserve old ledger/session fingerprints and references.
        return self.session_track_id
    @property
    def candidate_id(self):return "A0_face:"+self.prompt_ref["sha256"]
    @property
    def evidence_ids(self):return (self.a0_model_sha256,self.rgb_sha256,self.pose_receipt["sha256"],self.patch_ref["sha256"],self.prompt_ref["sha256"])
    @property
    def fingerprint(self):return _hash(asdict(self))
    def validate_region(self,region):
        if (region.identity!=self.region_identity or region.frame(self.local_index)!=(self.global_frame_id,self.pts)
                or tuple(region.time_base)!=tuple(self.time_base) or region.pixel_identity!=self.pixel_space_id):
            raise ValueError("Face seed region/frame/PTS/pixel domain mismatch")

def face_point_request(ledger,record_id,seed,*,points_xy01,point_labels):
    if not isinstance(seed,FaceSeedBinding):raise TypeError("Typed A0 face seed required")
    record=ledger.get(record_id)
    if record.seed_fingerprint!=seed.fingerprint or record.track_id!=seed.track_id:
        raise ValueError("Face generation/source differs from ledger")
    _unique_points(points_xy01)
    return ledger.seed_request(record_id,points_xy01=points_xy01,point_labels=point_labels,
                               obj_id=0,point_evidence_id=seed.prompt_ref["sha256"])

def _unique_points(points):
    p=np.asarray(points,float)
    if p.ndim!=2 or p.shape[1:]!=(2,) or len(np.unique(p,axis=0))!=len(p):
        raise ValueError("Distinct positive/negative point locations required")

@dataclass(frozen=True)
class FaceMaskThresholds:
    min_positive_coverage: float
    max_negative_coverage: float
    min_body_containment: float
    max_whole_body_iou: float
    max_neighbor_coverage: float
    max_other_body_fraction: float
    max_duplicate_iou: float
    def __post_init__(self):
        if any(isinstance(v,bool) or not math.isfinite(v) or not 0<=v<=1 for v in asdict(self).values()):
            raise ValueError("Explicit finite pilot fractions required")

def validate_face_raw(raw,*,seed,record,region,envelope,points_xy01,point_labels,
                      body_support,neighbor_supports,other_body_supports,existing_face_masks,
                      thresholds,raw_receipt_ref):
    """Preserve raw; return only row binding/diagnostics, never a clipped mask.

    Receipt refs are identity declarations: runner must persist/hash raw and
    authenticate all prompt/support inputs before calling. Unknown support is
    not a negative annotation.
    """
    if not isinstance(seed,FaceSeedBinding) or not isinstance(thresholds,FaceMaskThresholds):
        raise TypeError("Typed face seed and explicit pilot thresholds required")
    if envelope.operation!="seed" or envelope.expected_local_index!=seed.local_index:
        raise ValueError("Seed-frame face gate requires seed response; forward needs current-frame evidence")
    seed.validate_region(region);_ref(raw_receipt_ref)
    if record.seed_fingerprint!=seed.fingerprint or record.track_id!=seed.track_id:
        raise ValueError("Face body/generation identity differs")
    points=np.asarray(points_xy01,float);labels=tuple(point_labels)
    if (points.ndim!=2 or points.shape[1:]!=(2,) or len(points)!=len(labels) or not np.isfinite(points).all()
        or np.any((points<0)|(points>1)) or any(type(x) is not int or x not in (0,1) for x in labels)
        or not 2<=labels.count(1)<=4 or not 2<=labels.count(0)<=4):
        raise ValueError("Face needs2-4 positive and2-4 negative normalized points")
    _unique_points(points)
    request={"type":"add_prompt","session_id":record.session_id,"frame_index":seed.local_index,
             "obj_id":0,"points":points.tolist(),"point_labels":list(labels),
             "rel_coordinates":True,"clear_old_points":True}
    initial=_hash({"seed":seed.fingerprint,"corrections":None})
    if record.prompt_log_identity!=_hash({"previous":initial,"request":request,"evidence":seed.prompt_ref["sha256"]}):
        raise ValueError("Face points differ from actual ledger seed prompt")
    masks=[body_support]+list(neighbor_supports)+list(other_body_supports)+list(existing_face_masks)
    for m in masks:
        if not isinstance(m,np.ndarray) or m.dtype!=np.bool_ or m.shape!=(region.height,region.width):
            raise ValueError("Actual native boolean support masks required")
    output=validate_raw_response(raw,record=record,region=region,envelope=envelope)
    result={"source":"A0_face_hypothesis","face_key":[seed.body_track_id,seed.body_generation,seed.face_id,seed.face_generation],
            "status":"unknown","selected_raw_index":None,"native_obj_id":None,"reasons":[],
            "thresholds":asdict(thresholds),"thresholds_calibrated":False,"semantic_face_verified":False,
            "pose_accuracy_verified":False,"raw_receipt_ref":dict(raw_receipt_ref),"raw_modified":False}
    if isinstance(output,MissingOutput):
        result["reasons"]=["missing_native_output"];return result
    if output.object_ids!=(0,) or 0 in output.removed_object_ids:
        result["reasons"]=["unexpected_ambiguous_or_removed_native_object"];return result
    mask=output.masks[0];size=int(mask.sum());body_area=int(body_support.sum())
    if not size or not body_area:
        result["reasons"]=["empty_native_mask_or_unknown_body_support"];return result
    # Native normalized coordinates map to pixel indices; 1 is clamped at border.
    xy=np.minimum((points*np.array([region.width,region.height])).astype(int),
                  [region.width-1,region.height-1])
    samples=mask[xy[:,1],xy[:,0]];lab=np.array(labels)
    overlap=int((mask&body_support).sum())
    iou=lambda other: int((mask&other).sum())/max(1,int((mask|other).sum()))
    values={"positive_coverage":float(samples[lab==1].mean()),"negative_coverage":float(samples[lab==0].mean()),
            "body_containment":overlap/size,"whole_body_iou":overlap/(size+body_area-overlap),
            "neighbor_coverage":max([int((mask&m).sum())/max(1,int(m.sum())) for m in neighbor_supports] or [0.]),
            "other_body_fraction":max([int((mask&m).sum())/size for m in other_body_supports] or [0.]),
            "duplicate_iou":max([iou(m) for m in existing_face_masks] or [0.])}
    checks=[(values["positive_coverage"]<thresholds.min_positive_coverage,"positive_points_not_supported"),
            (values["negative_coverage"]>thresholds.max_negative_coverage,"negative_points_contaminated"),
            (values["body_containment"]<thresholds.min_body_containment,"cross_instance_or_outside_body"),
            (values["whole_body_iou"]>=thresholds.max_whole_body_iou,"whole_body_swallowed_not_face"),
            (values["neighbor_coverage"]>thresholds.max_neighbor_coverage,"multiple_faces_swallowed"),
            (values["other_body_fraction"]>thresholds.max_other_body_fraction,"cross_body_support"),
            (values["duplicate_iou"]>=thresholds.max_duplicate_iou,"duplicate_shared_face_mask")]
    result["metrics"]=values;result["reasons"]=[reason for bad,reason in checks if bad]
    if not result["reasons"]:
        result.update(status="predicted_face_support_only",selected_raw_index=0,native_obj_id=0)
    return result
