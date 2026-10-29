"""Bound ordinary SAM history; conditioning anchors and disk records are separate."""
def prune_history(state,frame_idx,*,reverse=False,window=15,after_frame=False,max_anchors=2):
    if type(frame_idx) is not int or frame_idx<0 or type(window) is not int or window<1:
        raise ValueError("Explicit frame index and positive history window required")
    if type(reverse) is not bool or type(after_frame) is not bool:raise ValueError("Explicit direction/phase required")
    cores=state.get("sam2_inference_states",[])
    anchors=set()
    for core in cores:
        anchors.update(core["output_dict"].get("cond_frame_outputs",{}))
    if len(anchors)>max_anchors:
        raise ValueError("Conditioning anchor budget exceeded; explicit rebind required")
    if reverse:
        keep=set(range(frame_idx if after_frame else frame_idx+1,frame_idx+window+1))
        if after_frame:keep.discard(frame_idx+window)
    else:
        keep=set(range(max(0,frame_idx-window),frame_idx+(1 if after_frame else 0)))
        if after_frame:keep.discard(frame_idx-window)
    # Existing current-frame data may be needed by explicit native refinement.
    if not after_frame:keep.add(frame_idx)
    removed=0
    def trim(mapping,protected):
        nonlocal removed
        for key in list(mapping):
            if type(key) is int and key not in keep and key not in protected:
                del mapping[key];removed+=1
    for core in cores:
        # A correction anchor belongs only to this native core. Another object
        # at the same frame still has ordinary history and must age out.
        core_anchors=set(core["output_dict"].get("cond_frame_outputs",{}))
        trim(core["output_dict"].get("non_cond_frame_outputs",{}),core_anchors)
        for field in ("output_dict_per_obj","temp_output_dict_per_obj"):
            for outputs in core.get(field,{}).values():
                trim(outputs.get("non_cond_frame_outputs",{}),core_anchors)
        consolidated=core.get("consolidated_frame_inds",{}).get("non_cond_frame_outputs")
        if consolidated is not None:consolidated.intersection_update(keep|core_anchors)
    # The public cache is shared, so it retains the union of genuine anchors.
    trim(state.get("cached_frame_outputs",{}),anchors)
    return {"window":window,"phase":"after" if after_frame else "before","reverse":reverse,
            "anchors":sorted(anchors),"retained_frame_domain":sorted(keep),"references_removed":removed,
            "archive_modified":False}

def install_history_window(predictor,official_root,*,window=15,max_anchors=2):
    """Apply retention before native memory reads; backend trims after each yielded frame."""
    import functools,inspect,hashlib
    from pathlib import Path
    model=predictor.model
    if type(model).__name__!="Sam3MultiplexTrackingWithInteractivity":
        raise ValueError("Unexpected native model for history policy")
    if getattr(model,"_action4_history_window",None) is not None:
        raise ValueError("History policy already installed")
    original=model._prepare_backbone_feats
    source=Path(inspect.getsourcefile(inspect.unwrap(original))).resolve(strict=True)
    expected=Path(official_root).resolve(strict=True)/"sam3/model/sam3_multiplex_tracking.py"
    if source!=expected or hashlib.sha256(source.read_bytes()).hexdigest()!="65e1cea27ee243dad228fb05967a8be92c90ef304665ada76930b899aa034d24":
        raise ValueError("Pinned memory-read boundary changed")
    @functools.wraps(original)
    def before(state,frame_idx,reverse):
        prune_history(state,frame_idx,reverse=reverse,window=window,max_anchors=max_anchors)
        state["_action4_history_frame"]=frame_idx
        state["_action4_history_reverse"]=reverse
        return original(state,frame_idx,reverse)
    receipt={"window":window,"max_conditioning_anchors":max_anchors,
             "before_read_hook":"_prepare_backbone_feats","after_yield_trim_required":True,
             "changes_memory_selection_domain":True,"upstream_file_modified":False}
    model._prepare_backbone_feats=before
    model._action4_history_window=receipt
    return receipt
