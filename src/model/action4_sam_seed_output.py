"""Candidate output assembly for explicit instance point seeds."""
def merge_seed_output(cache, frame_idx, refined):
    result = cache.get(frame_idx, {}).copy()
    for oid, mask in (refined or {}).items():
        if mask is None:
            raise ValueError("Computed instance mask is missing")
        result[oid] = mask
    return result

def make_seed_output_bridge(original):
    """Preserve original behavior except fresh computed instance output."""
    import functools
    @functools.wraps(original)
    def bridge(state, frame_idx, refined_obj_id_to_mask=None):
        if frame_idx not in state["cached_frame_outputs"] and refined_obj_id_to_mask:
            return merge_seed_output(state["cached_frame_outputs"], frame_idx, refined_obj_id_to_mask)
        return original(state, frame_idx, refined_obj_id_to_mask)
    return bridge


def install_seed_output_bridge(predictor, official_root):
    """Instance-local, exact-source guard; never edit pinned upstream files."""
    import hashlib
    import inspect
    from pathlib import Path
    model = predictor.model
    if type(model).__name__ != "Sam3MultiplexTrackingWithInteractivity":
        raise ValueError("Unexpected SAM model for instance output bridge")
    if getattr(model, "_action4_seed_output_bridge", None) is not None:
        raise ValueError("Instance output bridge already installed")
    original = model._build_sam2_output
    source = Path(inspect.getsourcefile(inspect.unwrap(original))).resolve(strict=True)
    expected = Path(official_root).resolve(strict=True) / "sam3/model/sam3_multiplex_tracking.py"
    if source != expected:
        raise ValueError("Instance output method escaped verified native source")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != "65e1cea27ee243dad228fb05967a8be92c90ef304665ada76930b899aa034d24":
        raise ValueError("Pinned SAM output implementation changed; re-review required")
    receipt = {"kind": "fresh_frame_computed_instance_output_merge",
               "official_source_sha256": digest, "upstream_file_modified": False,
               "segmentation_or_threshold_modified": False}
    model._build_sam2_output = make_seed_output_bridge(original)
    model._action4_seed_output_bridge = receipt
    return receipt
