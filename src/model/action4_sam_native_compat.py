"""Narrow instance-only compatibility for one pinned SAM3.1 init_state mismatch.

Official base predictor passes offload_state_to_cpu=False unconditionally, while
the pinned multiplex model does not accept it. No SAM or torch import occurs here.
"""
import functools
import hashlib
import inspect
import json
from pathlib import Path

OFFICIAL_COMMIT = "7f473cfd983d5eb43040ff608f66bff0d7bdcb53"


def make_init_state_bridge(original):
    """Remove only an unsupported, explicitly False state-offload argument.

    Other kwargs are passed unchanged and validated against the original signature.
    An original implementation supporting this argument keeps its own semantics.
    """
    signature = inspect.signature(original)
    accepts_state_offload = "offload_state_to_cpu" in signature.parameters

    @functools.wraps(original)
    def bridge(*args, **kwargs):
        if not accepts_state_offload and "offload_state_to_cpu" in kwargs:
            if kwargs["offload_state_to_cpu"] is not False:
                raise ValueError("Pinned multiplex init_state does not support state offload; only explicit False is compatible")
            kwargs = dict(kwargs)
            del kwargs["offload_state_to_cpu"]
        signature.bind(*args, **kwargs)  # Do not hide unrelated invalid arguments.
        return original(*args, **kwargs)

    return bridge


def install_pinned_init_state_compat(predictor, *, official_root, expected_source_sha256,
                                    expected_commit=OFFICIAL_COMMIT, snapshot_manifest=None):
    """Patch only predictor.model.init_state after exact local source verification.

    The caller already verified Git HEAD. This additionally pins the actual method
    source file SHA and its path. It never edits the official file or model class.
    """
    root = Path(official_root).resolve(strict=True)
    allowed = Path("/data/zy/vln/a-test/upstream/sam3")
    if expected_commit != OFFICIAL_COMMIT:
        raise ValueError("compat is restricted to the pinned SAM3.1 commit")
    if snapshot_manifest is None:
        if root != allowed:
            raise ValueError("non-live official root requires an explicit sealed snapshot manifest")
    else:
        manifest_path = Path(snapshot_manifest).resolve(strict=True)
        project = Path("/data/zy/vln/a-test")
        if not manifest_path.is_relative_to(project):
            raise ValueError("snapshot manifest escaped project")
        manifest = json.loads(manifest_path.read_text())
        payload = {k: v for k, v in manifest.items() if k != "identity"}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        expected_root = manifest_path.parent / manifest["official_source_relative"]
        if not expected_root.resolve(strict=True).is_relative_to(manifest_path.parent) or not root.is_relative_to(manifest_path.parent):
            raise ValueError("official snapshot source escaped sealed tree")
        if digest != manifest["identity"] or manifest.get("official_commit") != expected_commit or root != expected_root.resolve(strict=True):
            raise ValueError("official snapshot root/identity mismatch")
        relative = str((root / "sam3/model/sam3_multiplex_tracking.py").relative_to(manifest_path.parent))
        if manifest["files_sha256"].get(relative) != expected_source_sha256:
            raise ValueError("snapshot did not seal the expected official init_state source")
    model = predictor.model
    if type(model).__name__ != "Sam3MultiplexTrackingWithInteractivity":
        raise ValueError("compat targets the pinned multiplex model class only")
    if getattr(model, "_action4_init_state_compat", None) is not None:
        raise ValueError("compat already installed on this model instance")
    original = model.init_state
    # torch.inference_mode wraps the method in torch/_contextlib.py; verify the
    # unwrapped official function, not that runtime decorator implementation.
    source = Path(inspect.getsourcefile(inspect.unwrap(original))).resolve(strict=True)
    expected_path = root / "sam3/model/sam3_multiplex_tracking.py"
    if source != expected_path or not source.is_relative_to(root):
        raise ValueError("actual init_state came from a different source")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != expected_source_sha256:
        raise ValueError("official init_state source hash changed")
    signature = inspect.signature(original)
    has_video_loader_type = hasattr(predictor, "video_loader_type")
    if has_video_loader_type and "video_loader_type" not in signature.parameters:
        raise ValueError("predictor would pass unsupported video_loader_type; this compat does not remove it")
    if "offload_state_to_cpu" in signature.parameters:
        raise ValueError("pinned mismatch no longer present; re-review instead of applying stale compat")
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        raise ValueError("pinned concrete init_state signature changed")
    receipt = {"kind": "instance_only_init_state_false_offload_bridge",
               "official_commit": OFFICIAL_COMMIT, "official_method_source": str(source),
               "official_method_source_sha256": digest, "original_signature": str(signature),
               "video_loader_type_attribute_present": has_video_loader_type,
               "video_loader_type_supported_by_model": "video_loader_type" in signature.parameters,
               "allowed_removed_argument": "offload_state_to_cpu=False",
               "state_offload_true": "rejected", "other_unknown_kwargs": "rejected_by_original_signature",
               "compat_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    model.init_state = make_init_state_bridge(original)
    model._action4_init_state_compat = receipt
    return receipt
