"""Enable the pinned native tracker's existing CPU state storage option."""
import functools
import inspect
import hashlib
from pathlib import Path

def state_offload_bridge(original):
    signature=inspect.signature(original)
    if "offload_state_to_cpu" not in signature.parameters:
        raise ValueError("Native tracker has no CPU state-storage option")
    @functools.wraps(original)
    def bridge(*args,**kwargs):
        bound=signature.bind(*args,**kwargs)
        bound.arguments["offload_state_to_cpu"]=True
        return original(*bound.args,**bound.kwargs)
    return bridge

def install_state_offload(predictor,official_root):
    model=predictor.model
    if type(model).__name__!="Sam3MultiplexTrackingWithInteractivity":
        raise ValueError("Unexpected official tracker model")
    tracker=model.tracker
    if getattr(tracker,"_action4_cpu_state_storage",None) is not None:
        raise ValueError("CPU state-storage bridge already installed")
    original=tracker.init_state
    source=Path(inspect.getsourcefile(inspect.unwrap(original))).resolve(strict=True)
    expected=Path(official_root).resolve(strict=True)/"sam3/model/video_tracking_multiplex_demo.py"
    if source!=expected or hashlib.sha256(source.read_bytes()).hexdigest()!="d57ac540b400d5cb37377fc84810f75c2721f1b6e6ee9c8f119b5d6af7a6a80c":
        raise ValueError("Pinned native state-storage implementation changed")
    receipt={"kind":"official_core_offload_state_to_cpu_true","official_file_modified":False,
        "native_source_sha256":"d57ac540b400d5cb37377fc84810f75c2721f1b6e6ee9c8f119b5d6af7a6a80c",
        "network_or_threshold_changed":False,"whole_pipeline_memory_bound_proven":False}
    tracker.init_state=state_offload_bridge(original)
    tracker._action4_cpu_state_storage=receipt
    return receipt
