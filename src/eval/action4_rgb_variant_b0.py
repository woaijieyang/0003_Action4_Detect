"""Frozen completed-B0 native RGB variant leaf producer. No SAM or GT inference.

prepare() runs in the new project namespace. The copied standalone worker runs
only the sealed B0 source namespace. Production offers no predictor injection.
"""
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path("/data/zy/vln/a-test")
SCHEMA = "action4.rgb_variant_B0_leaf.v1"

def path(value, exists=True):
    p = Path(value)
    p = p if p.is_absolute() else ROOT/p
    p = p.resolve(strict=exists)
    if not p.is_relative_to(ROOT):
        raise ValueError("Path outside project")
    return p

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def canonical(value):
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())

def read(binding):
    if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
        raise ValueError("Explicit file binding required")
    raw = path(binding["path"]).read_bytes()
    if digest(raw) != binding["sha256"]:
        raise ValueError("Bound bytes changed")
    return raw

def binding(p):
    p = path(p)
    return {"path": str(p.relative_to(ROOT)), "sha256": digest(p.read_bytes())}

def write(p, value, readonly=True):
    p = path(p, False)
    temp = p.with_name(p.name+".partial."+uuid.uuid4().hex)
    with temp.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n");stream.flush();os.fsync(stream.fileno())
    if p.exists():
        raise FileExistsError(p)
    temp.replace(p)
    if readonly:p.chmod(0o444)
    return binding(p)

def pixels_hash(rgb):
    h = hashlib.sha256()
    h.update(str(rgb.dtype).encode());h.update(json.dumps(list(rgb.shape)).encode());h.update(rgb.tobytes())
    return h.hexdigest()

def decode(entry):
    import numpy as np
    from PIL import Image
    raw = read(entry["actual_input"])
    with Image.open(io.BytesIO(raw)) as image:
        if image.format != "PNG" or image.mode != "RGB" or getattr(image, "n_frames", 1) != 1:
            raise ValueError("Derived native RGB input must be single RGB PNG")
        rgb = np.array(image, dtype=np.uint8, copy=True)
    if list(rgb.shape) != entry["native_shape"] or pixels_hash(rgb) != entry["actual_pixels_sha256"]:
        raise ValueError("Actual input pixels/shape mismatch")
    return rgb

def leaf_inputs(manifest_binding, variant):
    if variant not in ("light", "blur"):
        raise ValueError("Explicit light or blur leaf only; original remains existing artifact")
    from data.action4_rgb_variant_manifest import verify_manifest
    manifest = verify_manifest(manifest_binding)
    rows = []
    for row in manifest["rows"]:
        if row["variant"] == variant:
            rows.append({"image_id": row["image_id"], "variant": variant,
                "variant_id": row["recipe"]["variant_id"], "actual_input": row["actual_input"],
                "actual_pixels_sha256": row["actual_pixels_sha256"], "native_shape": row["native_shape"],
                "original_source": row["original_source"], "recipe": row["recipe"]})
    if [r["image_id"] for r in rows] != manifest["image_ids"]:
        raise ValueError("Leaf differs from predeclared image domain")
    value = {"schema": SCHEMA, "variant": variant, "manifest": manifest_binding,
        "expected_registration": manifest["expected_registration"], "image_ids": manifest["image_ids"],
        "rows": rows, "GT_passed_to_inference": False}
    validate_inputs(value)
    return value

def validate_inputs(value):
    if value["schema"] != SCHEMA or value["variant"] not in ("light", "blur") or value["GT_passed_to_inference"] is not False:
        raise ValueError("Invalid prediction-only leaf inputs")
    manifest = json.loads(read(value["manifest"]))
    expected = json.loads(read(value["expected_registration"]))
    if value["expected_registration"] != manifest["expected_registration"] or expected["expected"] != manifest["expected"]:
        raise ValueError("Changed predeclared variant domain")
    wanted = [r for r in manifest["rows"] if r["variant"] == value["variant"]]
    if value["image_ids"] != manifest["image_ids"] or len(set(value["image_ids"])) != len(value["image_ids"]) or not wanted:
        raise ValueError("Invalid leaf image domain")
    if len(value["rows"]) != len(wanted):
        raise ValueError("Missing leaf rows")
    for row, source in zip(value["rows"], wanted):
        exact = {"image_id":source["image_id"],"variant":source["variant"],"variant_id":source["recipe"]["variant_id"],
            "actual_input":source["actual_input"],"actual_pixels_sha256":source["actual_pixels_sha256"],
            "native_shape":source["native_shape"],"original_source":source["original_source"],"recipe":source["recipe"]}
        if row != exact or row["recipe"]["variant_sha256"] != row["actual_pixels_sha256"]:
            raise ValueError("Leaf row differs from verified manifest")
        read(row["original_source"])
        decode(row)
    if [r["image_id"] for r in value["rows"]] != value["image_ids"]:
        raise ValueError("Leaf order differs")
    return value

def verify_source_before_import(spec_binding, spec):
    """Standard-library trust bootstrap; never execute a snapshot to verify itself."""
    if spec.get("project_root") != str(ROOT) or spec.get("mutable_worktree_not_used_by_children") is not True:
        raise ValueError("Sealed source project/root contract differs")
    code = path(spec["code_root"])
    if code == ROOT or not code.is_dir():
        raise ValueError("Explicit contained frozen code root required")
    files = spec.get("files")
    if not isinstance(files,dict) or not files:
        raise ValueError("Complete sealed source file map required")
    # A namespace package can be replaced by an added __init__, or a native/
    # bytecode module can shadow bound Python. Close the actual import inventory.
    for relative_root in ("src", "runtime/surface_deps"):
        import_root = code/relative_root
        if import_root.is_symlink():
            raise ValueError("Sealed import root symlink forbidden")
        if not import_root.exists():
            continue
        if not import_root.is_dir() or not import_root.resolve().is_relative_to(code):
            raise ValueError("Sealed import root escaped code root")
        for current, directories, names in os.walk(import_root, followlinks=False):
            for name in directories + names:
                candidate = Path(current)/name
                if candidate.is_symlink():
                    raise ValueError("Sealed import inventory symlink forbidden")
            for name in names:
                candidate = Path(current)/name
                # CPython ABI-tagged native modules still have these suffixes;
                # include versioned shared libraries and platform dylib/pyo too.
                lower = name.lower()
                importable = (lower.endswith((".py", ".pyc", ".pyo", ".so", ".pyd", ".dll", ".dylib"))
                              or ".so." in lower)
                if importable and str(candidate.relative_to(code)) not in files:
                    raise ValueError("Unlisted sealed importable file: " + str(candidate.relative_to(code)))
    config_path = path(spec["config"])
    if not config_path.is_relative_to(code):
        raise ValueError("Frozen configuration outside code root")
    config_relative = str(config_path.relative_to(code))
    if config_relative not in files:
        raise ValueError("Frozen configuration absent from source file map")
    config_bytes = None
    fingerprint = lambda st:(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
    for name, expected in files.items():
        relative = Path(name)
        if not isinstance(name,str) or relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Sealed source entry is not relative")
        actual = path(code/relative)
        if not actual.is_relative_to(code) or not actual.is_file() or (code/relative).is_symlink():
            raise ValueError("Sealed source entry escaped code root")
        before = actual.stat()
        if before.st_mode & 0o222:
            raise ValueError("Sealed source file is writable")
        with actual.open("rb") as stream:
            opened = os.fstat(stream.fileno())
            if fingerprint(opened) != fingerprint(before):
                raise ValueError("Sealed source changed before read")
            if name == config_relative:
                payload = stream.read();measured = digest(payload);config_bytes = payload
            else:
                measured = hashlib.file_digest(stream,"sha256").hexdigest()
            after = os.fstat(stream.fileno())
        if measured != expected or fingerprint(after) != fingerprint(before) or fingerprint(actual.stat()) != fingerprint(before):
            raise ValueError("Sealed source SHA/stat mismatch before project import: " + name)
    cfg = json.loads(config_bytes)
    if cfg.get("run_spec") != spec_binding["path"]:
        raise ValueError("Frozen cfg.run_spec differs from bound sealed RunSpec")
    return cfg

def sealed_source(seal_binding):
    seal = json.loads(read(seal_binding))
    if (seal.get("role") != "B0" or seal.get("strict_load") is not True or
            seal.get("epochs") != 80 or seal.get("completed_optimizer_steps") != 13040 or
            seal.get("completion",{}).get("evaluations") != 653):
        raise ValueError("Completed sealed B0 required")
    spec_binding = {"path":seal["run_spec"],"sha256":seal["run_spec_sha256"]}
    spec = json.loads(read(spec_binding))
    bundle = {"path":seal["bundle"],"sha256":seal["sha256"]}
    read(bundle)
    for b in (seal_binding, spec_binding, bundle):
        if path(b["path"]).stat().st_mode & 0o222:
            raise ValueError("B0 source must be readonly")
    cfg = verify_source_before_import(spec_binding, spec)
    return seal, spec, cfg

def resources(spec):
    """Live authority before importing/initializing CUDA; no default physical GPU."""
    active = json.loads(path("configs/execution/action4_active.json").read_bytes())
    authority = json.loads(read(active["active_ref"]))
    cpu = spec["cpu_affinity"]
    if (not isinstance(cpu,list) or not cpu or len(set(cpu)) != len(cpu) or
            any(type(c) is not int for c in cpu) or set(os.sched_getaffinity(0)) != set(cpu) or
            not set(cpu) <= set(authority["cpu_policy"]["allowed_logical_cpu_ids"]) & set(range(24,96))):
        raise ValueError("Explicit current CPU affinity mismatch")
    if (type(spec["threads"]) is not int or not 1 <= spec["threads"] <= min(2,len(cpu)) or
            type(spec["project_total_reserved_cpu"]) is not int or
            not len(cpu) <= spec["project_total_reserved_cpu"] <= min(72,authority["cpu_policy"]["max_logical_cpus"])):
        raise ValueError("CPU/thread total budget mismatch")
    device = spec["device"]
    if device == "cpu":
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise ValueError("CPU pilot requires empty CUDA visibility")
    elif device == "cuda:0":
        gpu = spec.get("physical_gpu")
        visible = spec.get("gpu_uuid")
        if (type(gpu) is not int or gpu not in (4,5,6,7) or
                gpu not in authority["authorized_evaluation_gpu"] or gpu in authority["forbidden_gpu_ids"] or
                not isinstance(visible,str) or not visible.startswith("GPU-") or
                os.environ.get("CUDA_VISIBLE_DEVICES") != visible or "," in visible):
            raise ValueError("Explicit authorized physical GPU/UUID required")
        budget = spec.get("allocator_mib")
        if type(budget) is not int or not 0 < budget <= 14336:
            raise ValueError("Bounded GPU allocator budget required")
        info = subprocess.check_output(["nvidia-smi","-i",str(gpu),"--query-gpu=uuid,memory.free","--format=csv,noheader,nounits"],text=True).strip().split(",")
        if len(info) != 2 or info[0].strip() != visible or int(info[1]) < budget+8192:
            raise ValueError("GPU mapping/headroom mismatch")
    else:
        raise ValueError("Only explicit CPU or authorized single-GPU device supported")
    return active["active_ref"]

def prepare(manifest_binding, variant, B0_seal, output, resource_spec):
    """Freeze input-only staging and this standalone worker; does not run inference."""
    inputs = leaf_inputs(manifest_binding,variant)
    seal,spec,cfg = sealed_source(B0_seal)
    out = path(output,False);out.mkdir(parents=True,exist_ok=False)
    source = out/"source";source.mkdir()
    worker = source/"worker.py";worker.write_bytes(Path(__file__).read_bytes());worker.chmod(0o444)
    inp = write(source/"INPUT.json",inputs)
    config = {"schema":SCHEMA,"mode":"production_cpu_pilot" if resource_spec["device"]=="cpu" else "production_native_variant_leaf",
        "input":inp,"B0_seal":B0_seal,"orchestrator":binding(worker),
        "resources":copy.deepcopy(resource_spec),"output":str((out/"run").relative_to(ROOT)),
        "scoring_policy":cfg["evaluation_scoring_policy"],"publication_threshold":0.3,
        "validator_sources":{n:digest(path(n).read_bytes()) for n in (
            "src/data/action4_rgb_variant_manifest.py","src/data/action4_rgb_variants.py")}}
    config_binding = write(source/"CONFIG.json",config)
    return write(out/"PREPARED.json",{"schema":SCHEMA,"worker":binding(worker),"config":config_binding,
        "variant":variant,"images":len(inputs["rows"]),"inference_started":False})

def pack_row(entry, raw):
    if not isinstance(raw,list):
        raise ValueError("Native predict must return complete candidate list")
    raw = copy.deepcopy(raw);candidates=[];seen=set()
    for ordinal,prediction in enumerate(raw):
        grid=prediction.get("grid_index");score=prediction.get("score")
        if (type(grid) is not int or grid < 0 or grid in seen or
                type(score) not in (int,float) or not math.isfinite(score)):
            raise ValueError("Invalid/duplicate native grid or score")
        # Preserve every raw field exactly; metadata is in a separate wrapper.
        canonical(prediction)
        seen.add(grid)
        candidates.append({"prediction_id":"b0:"+entry["image_id"]+":"+entry["variant_id"]+":grid:"+str(grid),
            "grid_index":grid,"raw_saved_ordinal":ordinal,"published":score>=0.3,
            "raw_saved_prediction":copy.deepcopy(prediction)})
    return {"sample_id":entry["image_id"],"variant":entry["variant"],"variant_id":entry["variant_id"],
        "actual_input":entry["actual_input"],"actual_pixels_sha256":entry["actual_pixels_sha256"],
        "native_shape":entry["native_shape"],"original_source":entry["original_source"],
        "recipe":entry["recipe"],"instances":raw,"candidates":candidates}

def _run(config_binding, config, inputs, factory, *, engineering, resume=False):
    out=path(config["output"],False)
    if out.exists() and not resume:raise FileExistsError("Explicit verified resume required")
    if not out.exists():out.mkdir(parents=True)
    import fcntl
    with (out/"producer.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/"COMPLETE.json").exists():raise FileExistsError("Already completed; do not republish")
        identity={"schema":SCHEMA,"config":config_binding,"input":config["input"],"engineering_fixture":engineering}
        if (out/"IDENTITY.json").exists():
            if json.loads((out/"IDENTITY.json").read_bytes()) != identity:raise ValueError("Resume configuration differs")
        else:
            if resume and any(p.name != "producer.lock" for p in out.iterdir()):raise ValueError("Unknown existing output")
            write(out/"IDENTITY.json",identity)
        rows_dir=out/"rows";rows_dir.mkdir(exist_ok=True)
        files=list(rows_dir.iterdir())
        if any(p.suffix not in (".json",) or not p.name.endswith((".row.json",".done.json")) for p in files):
            raise ValueError("Partial/unknown row artifact; preserve for inspection")
        rows=[];start=0
        for i,entry in enumerate(inputs["rows"]):
            rowpath=rows_dir/(str(i).zfill(6)+".row.json");done=rows_dir/(str(i).zfill(6)+".done.json")
            if not rowpath.exists() and not done.exists():break
            if not rowpath.exists() or not done.exists():raise ValueError("Partial row commit; cannot resume unknown state")
            marker=json.loads(done.read_bytes())
            if set(marker)!={"row","input_identity","run_identity"} or marker["run_identity"]!=canonical(identity) or marker["input_identity"]!=canonical(entry) or path(marker["row"]["path"])!=rowpath:
                raise ValueError("Resume row marker identity mismatch")
            row=json.loads(read(marker["row"]))
            if row != pack_row(entry,row["instances"]):raise ValueError("Resume row/domain/variant mismatch")
            if rowpath.stat().st_mode & 0o222 or done.stat().st_mode & 0o222:raise ValueError("Resume rows must be readonly")
            rows.append(row);start+=1
        if len(files)!=2*start:raise ValueError("Resume rows are not exact ordered prefix")
        try:
            predict = factory() if start < len(inputs["rows"]) else None
            for i in range(start,len(inputs["rows"])):
                entry=inputs["rows"][i];rgb=decode(entry)
                raw=predict(rgb)
                row=pack_row(entry,raw)
                b=write(rows_dir/(str(i).zfill(6)+".row.json"),row)
                write(rows_dir/(str(i).zfill(6)+".done.json"),{"row":b,"input_identity":canonical(entry),"run_identity":canonical(identity)})
                rows.append(row)
            validate_inputs(inputs)
            read(config_binding)
            if [r["sample_id"] for r in rows]!=inputs["image_ids"]:raise ValueError("Incomplete output image domain")
            predictions=write(out/"PREDICTIONS.json",rows)
            status="ENGINEERING_FIXTURE_COMPLETE_NOT_PRODUCTION" if engineering else "COMPLETE_FROZEN_B0_RGB_VARIANT_LEAF"
            return write(out/"COMPLETE.json",{"status":status,"config":config_binding,"input":config["input"],
                "predictions":predictions,"images":len(rows),"saved_candidates":sum(len(r["instances"]) for r in rows),
                "published_candidates":sum(c["published"] for r in rows for c in r["candidates"]),
                "engineering_fixture":engineering,"SAM_calls":0,"GT_passed_to_inference":False,
                "formal_training_admitted":False,"device":config["resources"]["device"],
                "numerical_scope":"CPU_pilot_not_substitute_for_existing_GPU_predictions" if config["resources"]["device"]=="cpu" else "explicit_frozen_single_GPU_variant_leaf"})
        except BaseException as exc:
            write(out/("FAILED_"+uuid.uuid4().hex+".json"),{"status":"FAILED_NO_COMPLETE","error":str(exc),
                "completed_images":len(rows),"engineering_fixture":engineering})
            raise

def engineering_fixture(manifest_binding,variant,output,predictor,*,resume=False):
    """Explicit CPU-only fixture; its receipt can never claim production execution."""
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise ValueError("Fixture requires empty CUDA")
    inputs=leaf_inputs(manifest_binding,variant)
    output=path(output,False)
    config={"schema":SCHEMA,"mode":"engineering_fixture","input":None,"output":str((output/"run").relative_to(ROOT)),
        "resources":{"device":"cpu"}}
    if not output.exists():
        output.mkdir(parents=True);config["input"]=write(output/"INPUT.json",inputs)
        cb=write(output/"CONFIG.json",config)
    else:
        cb=binding(output/"CONFIG.json");old=json.loads(read(cb))
        if json.loads(read(old["input"]))!=inputs:raise ValueError("Fixture resume domain differs")
        config=old
    return _run(cb,config,inputs,lambda:predictor,engineering=True,resume=resume)

def production_worker(config_binding, *, resume=False):
    if Path.cwd().resolve()!=ROOT:raise ValueError("Worker must start in authorized root")
    config=json.loads(read(config_binding))
    if config["mode"] not in ("production_cpu_pilot","production_native_variant_leaf"):
        raise ValueError("Production worker refuses engineering configuration")
    own=Path(__file__).resolve()
    if path(config["orchestrator"]["path"])!=own or digest(own.read_bytes())!=config["orchestrator"]["sha256"]:
        raise ValueError("Worker must execute frozen bound orchestration source")
    for p in (own,path(config_binding["path"]),path(config["input"]["path"])):
        if p.stat().st_mode & 0o222:raise ValueError("Worker source/config/input must be readonly")
    resources(config["resources"])
    inputs=validate_inputs(json.loads(read(config["input"])))
    seal,spec,cfg=sealed_source(config["B0_seal"])
    if config["scoring_policy"]!=cfg["evaluation_scoring_policy"] or config["publication_threshold"]!=0.3:
        raise ValueError("Frozen native scoring policy differs")
    code=path(spec["code_root"])
    if any(n.split(".")[0] in ("data","model","eval") for n in sys.modules):
        raise ValueError("Use fresh standalone worker, not mixed project namespace")
    os.environ["ACTION4_PROJECT_ROOT"]=str(ROOT);os.environ["PYTHONDONTWRITEBYTECODE"]="1"
    sys.dont_write_bytecode=True
    sys.path[:]=[str(code/"runtime/surface_deps"),str(code/"src")]+[
        p for p in sys.path if p and (not Path(p).resolve().is_relative_to(ROOT) or Path(p).resolve().is_relative_to(ROOT/".venv"))]
    from model.training.action4_run_snapshot import verify_running_snapshot
    verify_running_snapshot(cfg)
    from model.action4_release_bundle import load_bundle
    from eval.action4_b0_scoring import predict
    import torch
    def origins():
        for name,module in list(sys.modules.items()):
            filename=getattr(module,"__file__",None)
            if filename and name.split(".")[0] in ("data","model","eval") and not Path(filename).resolve().is_relative_to(code):
                raise ValueError("Mixed B0 module origins")
    origins()
    device=config["resources"]["device"]
    torch.set_num_threads(config["resources"]["threads"]);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    def factory():
        resources(config["resources"])
        if device=="cuda:0":
            properties=torch.cuda.get_device_properties(0)
            actual_uuid=getattr(properties,"uuid",None)
            if isinstance(actual_uuid,bytes) and len(actual_uuid)==16:actual_uuid=str(uuid.UUID(bytes=actual_uuid))
            actual_uuid=str(actual_uuid).removeprefix("GPU-").lower()
            expected_uuid=config["resources"]["gpu_uuid"].removeprefix("GPU-").lower()
            if actual_uuid!=expected_uuid or torch.cuda.current_device()!=0:raise ValueError("Initialized logical GPU mapping differs")
            total=properties.total_memory
            torch.cuda.set_per_process_memory_fraction(config["resources"]["allocator_mib"]*1024**2/total,0)
        modelpath=path(seal["bundle"]);before=modelpath.stat()
        model,_,state=load_bundle(str(modelpath),"B0",device)
        after=modelpath.stat()
        fingerprint=lambda st:(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
        if fingerprint(before)!=fingerprint(after) or digest(modelpath.read_bytes())!=seal["sha256"]:
            raise ValueError("Loaded B0 file changed")
        if state["selected_step"]!=seal["selected_step"] or state["config_id"]!=seal["config_id"] or state["config"]!=cfg:
            raise ValueError("Actual B0 state/config differs from sealed source")
        origins()
        def forward(rgb):
            with torch.inference_mode():
                result=predict(model,torch.from_numpy(rgb).permute(2,0,1),state["config"],device,config["scoring_policy"])
            origins()
            return result
        return forward
    return _run(config_binding,config,inputs,factory,engineering=False,resume=resume)

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",required=True);parser.add_argument("--config-sha256",required=True)
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    print(json.dumps(production_worker({"path":args.config,"sha256":args.config_sha256},resume=args.resume)))
