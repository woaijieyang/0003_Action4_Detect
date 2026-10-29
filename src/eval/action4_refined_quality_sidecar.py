"""Separate, uncalibrated refined-B0 quality capture. Never rewrites B0/SAM caches."""
import copy,hashlib,json,os,threading
from collections import OrderedDict
from pathlib import Path
from types import MappingProxyType
import numpy as np
from data.action4_paths import PROJECT_ROOT as R,CODE_ROOT,contained_path
from data.action4_sam_native_export import _v2_checked,_v2_sha,_v2_read,_fingerprint,canonical
from data.action4_sam_roi_inputs import array_digest
from eval.action4_b0_scoring import predict,REFINED_QUALITY_PROTOCOL

SCHEMA="action4.refined_quality_sidecar.v1"
FIELDS={"refined_quality_score","refined_quality_protocol"}
_SIDECAR_TOKEN=object()
_RGB_PROOF_PID=None
_RGB_PROOFS=OrderedDict()
_RGB_PROOF_LOCK=threading.Lock()
_RGB_PROOF_LIMIT=4096

def _rgb_proofs():
    global _RGB_PROOF_PID,_RGB_PROOFS,_RGB_PROOF_LOCK
    if _RGB_PROOF_PID!=os.getpid():
        _RGB_PROOFS=OrderedDict();_RGB_PROOF_LOCK=threading.Lock();_RGB_PROOF_PID=os.getpid()
    return _RGB_PROOFS,_RGB_PROOF_LOCK



def _read_manifest(binding):
    """Authenticated JSONL original annotations, or explicit JSON-array fixtures."""
    suffix=Path(binding["path"]).suffix.lower()
    if suffix==".jsonl":
        rows=_v2_read(binding,lambda stream:[json.loads(line) for line in stream if line.strip()])
    elif suffix==".json":
        rows=_v2_checked(binding)
    else:raise ValueError("Explicit JSONL/JSON image manifest required")
    if not isinstance(rows,list):raise ValueError("Image manifest must contain records")
    return rows


def _image(row):
    if (not isinstance(row["sample_id"],str) or not row["sample_id"]
        or type(row["height"]) is not int or type(row["width"]) is not int
        or min(row["height"],row["width"])<1 or not isinstance(row["image"],str)
        or not isinstance(row["rgb_sha256"],str) or len(row["rgb_sha256"])!=64
        or any(c not in "0123456789abcdef" for c in row["rgb_sha256"])):
        raise ValueError("Explicit native image identity/dimensions required")
    contained_path(R,row["image"],exists=True)
    K=np.asarray(row["camera"]["K"],float)
    if K.shape!=(3,3) or not np.isfinite(K).all():raise ValueError("Finite native K required")
    return {"sample_id":row["sample_id"],"image":row["image"],"RGB_file_sha256":row["rgb_sha256"],
        "shape_hw":[row["height"],row["width"]],"K":K.tolist()}

def _domain(base,manifest,scope):
    images=[_image(row) for row in manifest]
    for row in base:
        seen=set()
        if not isinstance(row["instances"],list):raise ValueError("Explicit base candidate list required")
        for candidate in row["instances"]:
            if any(k.lower().startswith(("gt","ground_truth")) for k in candidate):
                raise ValueError("GT fields cannot become detector inputs")
            grid=candidate.get("grid_index");box=np.asarray(candidate.get("bbox_full_xyxy"),float)
            if (type(grid) is not int or grid<0 or grid in seen or box.shape!=(4,)
                or not np.isfinite(box).all() or not (box[2:]>box[:2]).all()
                or type(candidate.get("score")) not in (int,float)
                or not np.isfinite(candidate["score"]) or not 0<=candidate["score"]<=1
                or candidate.get("prediction_id",f"b0:{row['sample_id']}:grid:{grid}")!=f"b0:{row['sample_id']}:grid:{grid}"):
                raise ValueError("Base candidate identity/box/score invalid")
            canonical(candidate);seen.add(grid)
    ids=[row["sample_id"] for row in base]
    if ids!=[r["sample_id"] for r in images] or len(ids)!=len(set(ids)):
        raise ValueError("Base predictions and original image manifest domain differ")
    if scope in ("full650","pilot32") and any(r.get("split")!="eval" for r in manifest):
        raise ValueError("Production sidecar requires eval-only image manifest")
    if scope=="full650":
        if len(ids)!=650:raise ValueError("Full sidecar requires650 images")
        return base,images
    if scope=="pilot32":
        if len(ids)!=650:raise ValueError("Pilot32 must be first32 of actual full650 source")
        return base[:32],images[:32]
    if scope=="cpu_fixture":
        if not 1<=len(ids)<=4:raise ValueError("Bounded CPU fixture domain required")
        return base,images
    raise ValueError("Unknown sidecar scope")

def quality_rows(sample_id,base_instances,enriched):
    """Full original dictionaries compare exactly; no tolerance or score substitution."""
    if len(base_instances)!=len(enriched):raise ValueError("Quality candidate count differs")
    result=[];seen=set()
    for base,row in zip(base_instances,enriched):
        if FIELDS & set(base):raise ValueError("Baseline already contains sidecar quality fields")
        stripped={k:v for k,v in row.items() if k not in FIELDS}
        if canonical(stripped)!=canonical(base):raise ValueError("Original B0 fields differ exactly")
        grid=base.get("grid_index")
        if type(grid) is not int or grid<0 or grid in seen:raise ValueError("Unique stable grid required")
        seen.add(grid);pid=f"b0:{sample_id}:grid:{grid}"
        if base.get("prediction_id",pid)!=pid:raise ValueError("Base prediction ID differs from sample/grid")
        q=row.get("refined_quality_score")
        if (set(row)!=set(base)|FIELDS or row.get("refined_quality_protocol")!=REFINED_QUALITY_PROTOCOL
            or type(q) not in (int,float) or not np.isfinite(q) or not 0<=q<=1):
            raise ValueError("Actual refined quality field/protocol required")
        result.append({"prediction_id":pid,"grid_index":grid,"base_row_sha256":canonical(base),
            "refined_quality_score":q,"refined_quality_protocol":copy.deepcopy(REFINED_QUALITY_PROTOCOL)})
    return result


def capture_one(model,model_cfg,rgb,image,base_instances,*,device,scoring_policy):
    """Prediction-only call boundary, independently testable with tiny fixtures."""
    a=np.asarray(rgb)
    if a.dtype!=np.uint8 or list(a.shape)!=image["shape_hw"]+[3]:
        raise ValueError("Native uint8 RGB shape differs")
    enriched=predict(model,a,model_cfg,device,scoring_policy,include_refined_quality=True)
    try:quality_rows(image["sample_id"],base_instances,enriched)
    except BaseException as error:
        error.capture_evidence={"sample_id":image["sample_id"],"image":copy.deepcopy(image),
            "RGB_pixel_sha256":array_digest(a),"expected_base":copy.deepcopy(base_instances),"actual_enriched":enriched}
        raise
    return {"sample_id":image["sample_id"],"image":copy.deepcopy(image),
        "RGB_pixel_sha256":array_digest(a),"enriched":enriched}


def make_sidecar(bindings,base,manifest,captures,*,scope,model_config_id,selected_step):
    """Finish only an exact declared domain. Capture failures cannot become complete."""
    expected,images=_domain(base,manifest,scope)
    if len(captures)!=len(expected):raise ValueError("Incomplete quality capture")
    rows=[]
    for source,image,capture in zip(expected,images,captures):
        if capture["sample_id"]!=image["sample_id"] or capture["image"]!=image:
            raise ValueError("Captured sample/K/RGB source differs")
        digest=capture["RGB_pixel_sha256"]
        if not isinstance(digest,str) or len(digest)!=64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Decoded native RGB digest required")
        rows.append({"image":image,"RGB_pixel_sha256":digest,
            "qualities":quality_rows(image["sample_id"],source["instances"],capture["enriched"])})
    return {"schema":SCHEMA,"status":"CAPTURE_COMPLETE_UNCALIBRATED","scope":scope,"complete":True,
        "bindings":copy.deepcopy(bindings),"model_config_id":model_config_id,"selected_step":selected_step,
        "score_protocol":copy.deepcopy(REFINED_QUALITY_PROTOCOL),"rows":rows,"images":len(rows),
        "candidate_count":sum(len(r["qualities"]) for r in rows),"GT_used_for_capture":False,
        "calibrated":False,"threshold":None,"admitted":False}

class RefinedQualitySidecar:
    """Once-authenticated immutable rows; current pixels/K/base/model still checked."""
    __slots__=("_files","_rows","_model","_scope","_pid","_token","_sealed")
    def __setattr__(self,name,value):
        if getattr(self,"_sealed",False):raise AttributeError("Authenticated sidecar is immutable")
        object.__setattr__(self,name,value)

    def __init__(self,binding,*,bindings,model_config_id,selected_step,expected_scope="full650"):
        if set(bindings)!={"B0","base_predictions","data_manifest","source"}:raise ValueError("Exact model/base/image/source bindings required")
        self._files=[]
        refs=[binding]+list(bindings.values())
        for ref in refs:
            if not isinstance(ref,dict) or set(ref)!={"path","sha256"}:raise ValueError("Exact path/SHA binding required")
            path=contained_path(R,ref["path"],exists=True);before=_fingerprint(path.stat())
            if _v2_sha(path)!=ref["sha256"] or _fingerprint(path.stat())!=before:raise ValueError("Sidecar dependency changed")
            self._files.append((path,before,ref["sha256"]))
        doc=_v2_checked(binding);base=_v2_checked(bindings["base_predictions"]);manifest=_read_manifest(bindings["data_manifest"])
        expected,images=_domain(base,manifest,expected_scope)
        if (doc.get("schema")!=SCHEMA or doc.get("status")!="CAPTURE_COMPLETE_UNCALIBRATED"
            or doc.get("scope")!=expected_scope or doc.get("complete") is not True or doc.get("bindings")!=bindings
            or doc.get("model_config_id")!=model_config_id or doc.get("selected_step")!=selected_step
            or doc.get("score_protocol")!=REFINED_QUALITY_PROTOCOL or doc.get("GT_used_for_capture") is not False
            or doc.get("calibrated") is not False or doc.get("threshold") is not None or doc.get("admitted") is not False
            or doc.get("images")!=len(expected) or len(doc.get("rows",[]))!=len(expected)):
            raise ValueError("Sidecar source/domain/uncalibrated contract differs")
        encoded={}
        for row,base_row,image in zip(doc["rows"],expected,images):
            if row.get("image")!=image:raise ValueError("Sidecar sample/K/image metadata differs")
            qs=row.get("qualities")
            if not isinstance(qs,list) or len(qs)!=len(base_row["instances"]):raise ValueError("Incomplete sidecar candidate domain")
            enriched=[]
            for b,q in zip(base_row["instances"],qs):
                if q.get("base_row_sha256")!=canonical(b) or q.get("grid_index")!=b["grid_index"] or q.get("prediction_id")!=f"b0:{image['sample_id']}:grid:{b['grid_index']}":
                    raise ValueError("Sidecar base candidate identity differs")
                enriched.append(dict(b,refined_quality_score=q.get("refined_quality_score"),refined_quality_protocol=q.get("refined_quality_protocol")))
            if quality_rows(image["sample_id"],base_row["instances"],enriched)!=qs:raise ValueError("Sidecar quality records malformed")
            pixel=row.get("RGB_pixel_sha256")
            if not isinstance(pixel,str) or len(pixel)!=64 or any(c not in "0123456789abcdef" for c in pixel):raise ValueError("Invalid sidecar pixel identity")
            encoded[image["sample_id"]]=json.dumps({"sidecar":row,"base_instances":base_row["instances"]},allow_nan=False)
        if len(encoded)!=len(expected) or doc.get("candidate_count")!=sum(len(x["instances"]) for x in expected):raise ValueError("Duplicate/incomplete sidecar domain")
        self._rows=MappingProxyType(encoded);self._model=bindings["B0"]["sha256"];self._scope=expected_scope;self._pid=os.getpid()
        self._files=tuple(self._files);self._token=_SIDECAR_TOKEN
        self._check_files();self._sealed=True

    def _check_files(self):
        if getattr(self,"_token",None) is not _SIDECAR_TOKEN or self._pid!=os.getpid():raise ValueError("Sidecar must be authenticated in consumer process")
        for path,before,sha in self._files:
            if _fingerprint(path.stat())!=before or _v2_sha(path)!=sha or _fingerprint(path.stat())!=before:
                raise ValueError("Authenticated sidecar/source file changed")

    def _current_row(self,sample_id,K,base_instances,b0_sha256):
        self._check_files()
        if b0_sha256!=self._model or sample_id not in self._rows:raise ValueError("Sidecar model/sample differs")
        value=json.loads(self._rows[sample_id]);row=value["sidecar"];camera=np.asarray(K,float)
        path=contained_path(R,row["image"]["image"],exists=True);before=_fingerprint(path.stat())
        if _v2_sha(path)!=row["image"]["RGB_file_sha256"] or _fingerprint(path.stat())!=before:
            raise ValueError("Original sidecar RGB file changed")
        if (camera.shape!=(3,3) or not np.array_equal(camera,np.asarray(row["image"]["K"]))
            or canonical(base_instances)!=canonical(value["base_instances"])):
            raise ValueError("Current RGB/K/base predictions differ from sidecar")
        return row,path,before

    def _proof_key(self,sample_id):
        return (self._pid,self._files,sample_id)

    def _quality_value(self,row):
        return {"qualities":row["qualities"],"scope":self._scope,"calibrated":False,"threshold":None,"admitted":False}

    def consume(self,*,sample_id,rgb,K,base_instances,b0_sha256):
        row,path,before=self._current_row(sample_id,K,base_instances,b0_sha256)
        memo,lock=_rgb_proofs();key=self._proof_key(sample_id)
        with lock:memo.pop(key,None)
        a=np.asarray(rgb)
        if (a.dtype!=np.uint8 or list(a.shape)!=row["image"]["shape_hw"]+[3]
            or array_digest(a)!=row["RGB_pixel_sha256"]):
            raise ValueError("Current RGB/K/base predictions differ from sidecar")
        self._check_files()
        if _fingerprint(path.stat())!=before:raise ValueError("RGB changed during pixel verification")
        with lock:
            memo[key]=(str(path),before,row["RGB_pixel_sha256"],_SIDECAR_TOKEN)
            memo.move_to_end(key)
            while len(memo)>_RGB_PROOF_LIMIT:memo.popitem(last=False)
        return self._quality_value(row)

    def consume_previously_verified_original(self,*,sample_id,K,base_instances,b0_sha256):
        """Scoring-only fast path. No pixel/model output cache; misses require normal decode."""
        row,path,before=self._current_row(sample_id,K,base_instances,b0_sha256)
        memo,lock=_rgb_proofs();key=self._proof_key(sample_id)
        with lock:
            proof=memo.get(key)
            if proof is not None:memo.move_to_end(key)
        if proof!=(str(path),before,row["RGB_pixel_sha256"],_SIDECAR_TOKEN):return None
        self._check_files()
        if _fingerprint(path.stat())!=before:raise ValueError("RGB changed while checking prior pixel proof")
        return self._quality_value(row)

def produce(config_path,output):
    """Explicit GPU entry, to be launched only after separate resource authorization."""
    import torch
    from PIL import Image
    from model.action4_release_bundle import load_bundle
    cfg=_v2_checked(config_path);out=contained_path(R,output,exists=False);out.mkdir(parents=True,exist_ok=False)
    captures=[]
    try:
        if cfg.get("schema")!="action4.refined_quality_capture_config.v1" or cfg.get("scope") not in ("pilot32","full650"):
            raise ValueError("Registered pilot32/full650 capture required")
        if cfg.get("threshold") is not None or cfg.get("calibrated") is not False:raise ValueError("Capture cannot invent calibration")
        if cfg.get("selected_step")!=9500 or cfg.get("scoring_policy")!="center":
            raise ValueError("This sidecar requires frozenB09500 center-policy identity")
        if sorted(os.sched_getaffinity(0))!=cfg["cpu_affinity"] or os.environ.get("CUDA_VISIBLE_DEVICES")!=cfg["gpu_uuid"]:
            raise ValueError("Explicit allocated CPU/GPU differs")
        if cfg.get("physical_gpu")!=5:raise ValueError("This prepared probe requires physicalGPU5 authorization")
        import subprocess
        live=subprocess.check_output(["nvidia-smi","--id=5","--query-gpu=uuid,memory.free","--format=csv,noheader,nounits"],text=True).strip().split(",")
        if live[0].strip()!=cfg["gpu_uuid"] or int(live[1])<1536+16384:
            raise ValueError("PhysicalGPU5 identity/headroom insufficient")
        bindings=cfg["bindings"]
        if set(bindings)!={"B0","base_predictions","data_manifest","source"}:raise ValueError("Exact input bindings required")
        input_stats=[]
        for ref in bindings.values():
            path=contained_path(R,ref["path"],exists=True)
            before=_fingerprint(path.stat())
            if _v2_sha(path)!=ref["sha256"] or _fingerprint(path.stat())!=before:raise ValueError("Input changed")
            input_stats.append((path,before,ref["sha256"]))
        source=_v2_checked(bindings["source"])
        if CODE_ROOT==R or contained_path(R,source["source_root"],exists=True)!=CODE_ROOT:raise ValueError("Actual frozen source required")
        for name,sha in source["files"].items():
            if _v2_sha(CODE_ROOT/name)!=sha:raise ValueError("Frozen code changed")
        base=_v2_checked(bindings["base_predictions"]);manifest=_read_manifest(bindings["data_manifest"])
        rows,images=_domain(base,manifest,cfg["scope"])
        for image in images:
            if _v2_sha(image["image"])!=image["RGB_file_sha256"]:
                raise ValueError("Pilot RGB dependency changed before CUDA/model loading")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
        torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
        torch.cuda.set_per_process_memory_fraction(1024**3/torch.cuda.get_device_properties(0).total_memory,0)
        model,_,state=load_bundle(bindings["B0"]["path"],"B0","cuda:0",expected_sha256=bindings["B0"]["sha256"])
        if state["selected_step"]!=cfg["selected_step"] or state["config_id"]!=cfg["model_config_id"]:raise ValueError("Loaded B0 identity differs")
        policy=state.get("inference_scoring_policy",state["config"].get("evaluation_scoring_policy","learned"))
        if policy!=cfg["scoring_policy"]:raise ValueError("Frozen scoring policy differs")
        for row,image in zip(rows,images):
            path=contained_path(R,image["image"],exists=True)
            from data.action4_sam_native_export import _v2_read
            def decode(stream):
                with Image.open(stream) as im:return np.asarray(im.convert("RGB")).copy()
            rgb=_v2_read({"path":str(path),"sha256":image["RGB_file_sha256"]},decode)
            if list(rgb.shape)!=image["shape_hw"]+[3]:raise ValueError("Original RGB shape changed")
            with torch.inference_mode():capture=capture_one(model,state["config"],rgb,image,row["instances"],device="cuda:0",scoring_policy=policy)
            memory=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,used_gpu_memory","--format=csv,noheader,nounits"],text=True)
            own=[int(line.split(",")[1]) for line in memory.splitlines() if int(line.split(",")[0])==os.getpid()]
            if own and max(own)>1536:raise RuntimeError("Quality capture process exceeds1.5GiB")
            captures.append(capture)
            (out/"PROGRESS.json").write_text(json.dumps({"completed":len(captures),"expected":len(rows),"scope":cfg["scope"],"admitted":False}))
        for path,before,sha in input_stats:
            if _fingerprint(path.stat())!=before or _v2_sha(path)!=sha:raise ValueError("Capture input changed before completion")
        doc=make_sidecar(bindings,base,manifest,captures,scope=cfg["scope"],model_config_id=state["config_id"],selected_step=state["selected_step"])
        (out/"SIDECAR.json").write_text(json.dumps(doc,indent=2,allow_nan=False))
        return doc
    except BaseException as error:
        import traceback
        def diagnostic(value):
            if isinstance(value,float) and not np.isfinite(value):return {"nonfinite":repr(value)}
            if isinstance(value,dict):return {k:diagnostic(v) for k,v in value.items()}
            if isinstance(value,(tuple,list)):return [diagnostic(v) for v in value]
            return value
        if hasattr(error,"capture_evidence"):
            (out/"FAILED_FRAME.json").write_text(json.dumps(diagnostic(error.capture_evidence),indent=2,allow_nan=False))
        (out/"PARTIAL_CAPTURE.json").write_text(json.dumps({"captures":captures,"complete":False,"admitted":False},indent=2,allow_nan=False))
        (out/"FAILURE.json").write_text(json.dumps({"traceback":traceback.format_exc(),"completed_images":len(captures),"complete":False,"admitted":False},indent=2))
        raise

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--config-sha256",required=True);p.add_argument("--output",required=True)
    a=p.parse_args();produce({"path":a.config,"sha256":a.config_sha256},a.output)
