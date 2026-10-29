"""Pinned per-instance SAM input-cache boundary, with no import-time model work.

During one configured init call this temporarily replaces a module-global loader
name under an RLock. Only the calling thread AND exact registered resource route
to cache; all other calls use the original loader. The namespace is restored in
finally. No native class or source file is modified. Existing instance-level
offload_state_to_cpu=False compatibility remains the inner callable.
"""
import functools
import hashlib
import inspect
import json
import math
from pathlib import Path
import threading

ROOT=Path("/data/zy/vln/a-test")
OFFICIAL_COMMIT="7f473cfd983d5eb43040ff608f66bff0d7bdcb53"
_LOCK=threading.RLock()
_LOADER="load_resource_as_video_frames"


def _inside(value):
    p=Path(value);p=p if p.is_absolute() else ROOT/p
    if p.is_symlink() or ".." in p.parts or not p.resolve(strict=True).is_relative_to(ROOT):
        raise ValueError("Cached input dependency escapes project")
    return p.resolve(strict=True)


def _sha(path):
    with _inside(path).open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


def _json(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()


def _stat(path):
    st=path.stat();return (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)


def _matches(value,resource):
    if not isinstance(value,(str,Path)):return False
    p=Path(value);p=p if p.is_absolute() else ROOT/p
    return ".." not in p.parts and p==resource


def _half_triplet(value):
    return (isinstance(value,(tuple,list)) and len(value)==3
            and all(type(x) in (int,float) and math.isfinite(x) and x==.5 for x in value))


def _validate_loader(arguments):
    if type(arguments.get("image_size")) is not int or arguments["image_size"]!=1008:
        raise ValueError("Cache requires exact image_size1008")
    if arguments.get("offload_video_to_cpu") is not True or arguments.get("async_loading_frames") is not False:
        raise ValueError("Cache requires CPU video offload and synchronous loading")
    if not _half_triplet(arguments.get("img_mean")) or not _half_triplet(arguments.get("img_std")):
        raise ValueError("Cached mean/std must each be(.5,.5,.5)")
    if arguments.get("video_loader_type","cv2")!="cv2":
        raise ValueError("Cached folder loader requires original cv2 branch")


def _make_init_wrapper(original,namespace,loader,reader,resource,stable=lambda:None):
    """Internal CPU-testable routing primitive; installer supplies pinned inputs."""
    signature=inspect.signature(loader)
    usage={"configured_calls":0,"cache_boundary_hits":0,"default_calls":0,"last_call":None}
    @functools.wraps(original)
    def wrapped(*args,**kwargs):
        requested=kwargs.get("resource_path",args[0] if args else None)
        if not _matches(requested,resource):
            usage["default_calls"]+=1
            return original(*args,**kwargs)
        # Do NOT signature-bind init_state here: the existing inner compat must
        # see and strip its explicitly False offload_state_to_cpu argument.
        with _LOCK:
            stable()
            if namespace.get(_LOADER) is not loader:
                raise RuntimeError("Native loader namespace already changed; no nested/foreign override")
            owner=threading.get_ident();hits=0
            def scoped_loader(*loader_args,**loader_kwargs):
                nonlocal hits
                requested_loader=loader_kwargs.get("resource_path",loader_args[0] if loader_args else None)
                if threading.get_ident()!=owner or not _matches(requested_loader,resource):
                    return loader(*loader_args,**loader_kwargs)
                bound=signature.bind(*loader_args,**loader_kwargs);bound.apply_defaults()
                _validate_loader(bound.arguments)
                if hits:raise RuntimeError("Pinned init unexpectedly requested loader more than once")
                hits+=1
                return reader.as_video_frames()
            usage["configured_calls"]+=1
            namespace[_LOADER]=scoped_loader
            success=False
            try:
                result=original(*args,**kwargs)
                if hits!=1:raise RuntimeError("Configured native init did not consume the cache boundary")
                success=True
                return result
            finally:
                interference=namespace.get(_LOADER) is not scoped_loader
                namespace[_LOADER]=loader
                usage["cache_boundary_hits"]+=hits
                usage["last_call"]={"thread_id":owner,"cache_boundary_hits":hits,"init_returned":success and not interference,
                    "module_loader_restored":namespace.get(_LOADER) is loader,
                    "namespace_interference_detected":interference,"native_model_accuracy_verified":False}
                if interference:raise RuntimeError("Native loader namespace changed during owned cache scope")
    return wrapped,usage


def _verify_native(model,official_root,expected_source_sha256,snapshot_manifest,expected_commit):
    if expected_commit!=OFFICIAL_COMMIT:raise ValueError("Unreviewed native commit")
    root=_inside(official_root);manifest_path=_inside(snapshot_manifest)
    manifest=json.loads(manifest_path.read_text())
    payload={k:v for k,v in manifest.items() if k!="identity"}
    if hashlib.sha256(_json(payload)).hexdigest()!=manifest["identity"] or manifest["official_commit"]!=expected_commit:
        raise ValueError("Native snapshot identity/commit differs")
    if root!=_inside(manifest_path.parent/manifest["official_source_relative"]):
        raise ValueError("Native root not bound to snapshot")
    if type(model).__name__!="Sam3MultiplexTrackingWithInteractivity":
        raise ValueError("Only the pinned native multiplex model instance is supported")
    if getattr(model,"_action4_cached_input_bridge",None) is not None:
        raise ValueError("Cache bridge already installed on this instance")
    compat=getattr(model,"_action4_init_state_compat",None)
    if not isinstance(compat,dict) or compat.get("official_method_source_sha256")!=expected_source_sha256:
        raise ValueError("Install and preserve existing verified init-state compatibility first")
    original=model.init_state;unwrapped=inspect.unwrap(original)
    actual=_inside(inspect.getsourcefile(unwrapped))
    expected=root/"sam3/model/sam3_multiplex_tracking.py"
    relative=str(expected.relative_to(manifest_path.parent))
    if actual!=expected or _sha(expected)!=expected_source_sha256 or manifest["files_sha256"].get(relative)!=expected_source_sha256:
        raise ValueError("Actual init-state source path/hash differs")
    namespace=unwrapped.__globals__;loader=namespace.get(_LOADER)
    if not callable(loader):raise ValueError("Pinned loader global absent")
    loader_source=_inside(inspect.getsourcefile(inspect.unwrap(loader)))
    io_source=root/"sam3/model/io_utils.py"
    io_relative=str(io_source.relative_to(manifest_path.parent))
    io_sha=_sha(io_source)
    if loader_source!=io_source or manifest["files_sha256"].get(io_relative)!=io_sha:
        raise ValueError("Actual global video loader is not pinned native IO")
    return original,namespace,loader,io_sha,[manifest_path,expected,io_source]


def _resource_order(resource,contract):
    if not resource.is_dir():raise ValueError("Prepared resource must be a frame directory")
    rows=contract["frames"]
    files=sorted(resource.iterdir())
    if [p.name for p in files]!=[f"{i:06d}.png" for i in range(len(rows))]:
        raise ValueError("Prepared resource frame domain/order differs")
    for i,(p,row) in enumerate(zip(files,rows)):
        source=_inside(row["image"]);actual=_inside(p)
        # Reuse the existing exact prepared hardlink directory; no new links,
        # re-encoding, resizing, or metadata changes are performed here.
        if p.is_symlink() or not actual.samefile(source):
            raise ValueError("Prepared frame is not the original source frame")
        if type(row["local_index"]) is not int or row["local_index"]!=i:
            raise ValueError("Cache-local native frame order differs")


def install_pinned_cached_input(predictor,*,cache_spec,official_root,
                               expected_source_sha256,snapshot_manifest,
                               expected_commit=OFFICIAL_COMMIT):
    """Install on one already-loaded model instance; never constructs a model.

    cache_spec contains exact resource_path, manifest{path,sha256}, and an
    independently caller-bound contract{path,sha256}. Cache producer is untouched.
    """
    if not isinstance(cache_spec,dict) or set(cache_spec)!={"resource_path","manifest","contract"}:
        raise ValueError("Explicit resource/manifest/contract bindings required")
    model=predictor.model
    original,namespace,loader,io_sha,native_files=_verify_native(
        model,official_root,expected_source_sha256,snapshot_manifest,expected_commit)
    def bound_json(ref):
        if not isinstance(ref,dict) or set(ref)!={"path","sha256"}:raise ValueError("Exact cache path/SHA reference required")
        p=_inside(ref["path"])
        if _sha(p)!=ref["sha256"]:raise ValueError("Cached input reference SHA differs")
        return json.loads(p.read_text()),p
    contract,contract_path=bound_json(cache_spec["contract"])
    preprocess=contract.get("preprocess",{})
    if (preprocess.get("image_size"),preprocess.get("layout"),preprocess.get("dtype"),
        preprocess.get("offload_video_to_cpu"))!=(1008,"CHW","float16",True):
        raise ValueError("Cache preprocessing contract differs")
    if not _half_triplet(preprocess.get("mean")) or not _half_triplet(preprocess.get("std")):
        raise ValueError("Cache normalization differs")
    if contract["official_io_sha256"]!=io_sha:raise ValueError("Cache and actual native IO code differ")
    manifest_ref=cache_spec["manifest"]
    if not isinstance(manifest_ref,dict) or set(manifest_ref)!={"path","sha256"}:raise ValueError("Cache manifest binding required")
    # Reader validates its own unchanged producer code, library versions, original
    # frame/PTS contract, source stats and every fetched payload.
    from data.action4_sam_input_cache import SamInputTensorCache
    reader=SamInputTensorCache(manifest_ref["path"],expected_sha256=manifest_ref["sha256"],expected_contract=contract)
    resource=_inside(cache_spec["resource_path"]);_resource_order(resource,contract)
    watched=native_files+[contract_path,_inside(manifest_ref["path"])]
    stamps={p:_stat(p) for p in watched}
    def stable():
        if any(_stat(p)!=s for p,s in stamps.items()):raise ValueError("Pinned bridge dependency changed after installation")
        _resource_order(resource,contract)
    wrapper,usage=_make_init_wrapper(original,namespace,loader,reader,resource,stable)
    receipt={"schema":"action4.sam_cached_input_bridge.v1","official_commit":expected_commit,
        "actual_native_source_sha256":expected_source_sha256,"actual_native_io_sha256":io_sha,
        "resource_path":str(resource.relative_to(ROOT)),"cache_manifest":dict(manifest_ref),
        "cache_contract":dict(cache_spec["contract"]),"frames":len(reader),"native_hw":list(reader.native_hw),
        "cache_contract_id":reader.contract_id,"preprocess_applied_again":False,
        "scope":"persistent instance init wrapper; temporary module-global loader name swap under RLock; only owner thread/exact resource routed",
        "global_namespace_temporarily_modified":True,"global_loader_restored_finally":True,
        "native_class_or_file_modified":False,"compat_wrapper_preserved":True,
        "GPU_model_execution_performed_by_installer":False}
    model.init_state=wrapper
    model._action4_cached_input_bridge=receipt
    model._action4_cached_input_usage=usage
    return receipt
