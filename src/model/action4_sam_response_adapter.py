"""CPU-only bridge for pinned SAM3.1 raw dictionaries, not model inference.

Schema inspected at facebookresearch/sam3 commit
7f473cfd983d5eb43040ff608f66bff0d7bdcb53:
sam3_base_predictor.py:228,351 and sam3_multiplex_tracking.py:678-792.
clear_old_points replaces THIS object's THIS frame's points (video_tracking_
multiplex_demo.py:1283-1303); it is not session/cache reset.
"""
from dataclasses import dataclass
import hashlib
import json
from collections.abc import Mapping
import numpy as np
from model.action4_sam_session_contracts import FrameResponse, AbsenceDecision

OFFICIAL_COMMIT = "7f473cfd983d5eb43040ff608f66bff0d7bdcb53"


class ResponseSchemaError(ValueError): pass
class ResponseIdentityError(ValueError): pass


def _integer(value, label):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ResponseSchemaError(f"{label} requires a nonnegative integer, not float/bool")
    return int(value)


def _name(value, label):
    if not isinstance(value, str) or not value: raise ResponseSchemaError(f"nonempty {label} required")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _readonly(array):
    a = np.ascontiguousarray(array)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


@dataclass(frozen=True)
class ResponseEnvelope:
    """Trusted worker request context; raw SAM response does not carry these IDs."""
    session_id: str
    region_identity: str
    generation: int
    attempt: int
    operation: str
    response_id: str
    expected_local_index: int | None = None

    def __post_init__(self):
        for label in ("session_id", "region_identity", "operation", "response_id"):
            _name(getattr(self, label), label)
        _integer(self.generation, "generation"); _integer(self.attempt, "attempt")
        if self.operation not in {"seed", "propagate_both"} and not self.operation.startswith("correction:"):
            raise ResponseSchemaError("unsupported operation")
        if self.operation != "propagate_both" and self.expected_local_index is None:
            raise ResponseSchemaError("prompt response requires requested local frame index")
        if self.expected_local_index is not None: _integer(self.expected_local_index, "expected_local_index")


@dataclass(frozen=True)
class ValidatedOutput:
    local_index: int
    object_ids: tuple
    probabilities: np.ndarray
    boxes_xywh01: np.ndarray
    masks: np.ndarray
    removed_object_ids: tuple = ()


@dataclass(frozen=True)
class MissingOutput:
    local_index: int
    status: str = "missing_response"
    reason: str = "official_outputs_none_incomplete_acknowledgement"
    acknowledged: bool = False


def validate_raw_response(raw, *, record, region, envelope):
    """Validate all raw rows before selecting a target. Does not import SAM.

    Missing transport messages remain absent from the collector. A returned
    outputs=None is a MissingOutput diagnostic, never a target_absent frame.
    """
    if not isinstance(envelope, ResponseEnvelope): raise ResponseSchemaError("typed request envelope required")
    if (envelope.session_id, envelope.region_identity, envelope.generation, envelope.attempt) != (
            record.session_id, record.region_identity, record.generation, record.attempt) or region.identity != record.region_identity:
        raise ResponseIdentityError("request/record/region identity mismatch")
    if envelope.operation != record.active_operation:
        raise ResponseIdentityError("response operation differs from pending official request")
    if not isinstance(raw, Mapping) or "frame_index" not in raw or "outputs" not in raw:
        raise ResponseSchemaError("official response requires frame_index and outputs")
    # Official response has no such envelope fields; reject conflicting fields
    # if a transport wrapper supplies them rather than silently overriding them.
    for key, expected in (("session_id", envelope.session_id), ("region_identity", envelope.region_identity),
                          ("generation", envelope.generation), ("attempt", envelope.attempt)):
        if key in raw:
            if key in {"generation", "attempt"}: _integer(raw[key], key)
            if raw[key] != expected: raise ResponseIdentityError(f"raw {key} conflicts with request context")
    local = _integer(raw["frame_index"], "frame_index")
    try: region.frame(local)
    except ValueError as exc: raise ResponseIdentityError("response frame outside region") from exc
    if envelope.expected_local_index is not None and local != envelope.expected_local_index:
        raise ResponseIdentityError("prompt returned a different frame")
    outputs = raw["outputs"]
    if outputs is None: return MissingOutput(local)
    keys = ("out_obj_ids", "out_probs", "out_boxes_xywh", "out_binary_masks")
    if not isinstance(outputs, Mapping) or any(k not in outputs for k in keys):
        raise ResponseSchemaError("official outputs dictionary is incomplete or dummy")
    ids, probs, boxes, masks = (outputs[k] for k in keys)
    if not isinstance(ids, np.ndarray) or ids.ndim != 1 or ids.dtype.kind not in "iu":
        raise ResponseSchemaError("out_obj_ids must be a one-dimensional NumPy integer array")
    object_ids = tuple(_integer(i, "obj_id") for i in ids)
    if len(set(object_ids)) != len(object_ids): raise ResponseSchemaError("duplicate raw object IDs")
    n = len(object_ids)
    if not isinstance(probs, np.ndarray) or probs.shape != (n,) or probs.dtype.kind != "f" or not np.isfinite(probs).all() or np.any(((probs < 0) | (probs > 1)) & (probs != -1e4)):
        raise ResponseSchemaError("invalid aligned probability array")
    if not isinstance(boxes, np.ndarray) or boxes.shape != (n, 4) or boxes.dtype.kind != "f" or not np.isfinite(boxes).all():
        raise ResponseSchemaError("invalid aligned normalized XYWH boxes")
    if np.any(boxes < 0) or np.any(boxes > 1) or np.any(boxes[:, :2] + boxes[:, 2:] > 1 + 1e-6):
        raise ResponseSchemaError("normalized XYWH box outside image")
    # Official masks_to_boxes can yield width/height zero for a one-pixel mask.
    if not isinstance(masks, np.ndarray) or masks.dtype != np.bool_ or masks.shape != (n, region.height, region.width):
        raise ResponseSchemaError("native boolean masks must align to object IDs and region dimensions")
    return ValidatedOutput(local, object_ids, _readonly(probs), _readonly(boxes), _readonly(masks),
                           tuple(obj for obj, value in zip(object_ids, probs) if value == -1e4))


@dataclass(frozen=True)
class MaskRegistrationRequest:
    mask_identity: str
    pixel_sha256: str
    binding_identity: str
    binding_json: str
    mask: np.ndarray


@dataclass(frozen=True)
class MaskRegistrationReceipt:
    mask_identity: str
    pixel_sha256: str
    binding_identity: str
    storage_reference: str


@dataclass(frozen=True)
class ParsedTarget:
    frame_response: FrameResponse | None
    acknowledged: bool
    status: str
    reason: str | None
    local_index: int
    response_id: str
    registration: MaskRegistrationReceipt | None = None
    public_output_contains_bound_id: bool | None = None
    available: bool = False
    use_as_training_negative: bool = False
    public_mask_nonempty: bool | None = None


def parse_target_response(raw, *, record, region, envelope, register_mask, absence_decision=None):
    """Normalize one bound target, registering the actual selected mask bytes.

    register_mask must persist/register the supplied readonly mask with its exact
    identity, then return a receipt. The receipt attests caller registration, not
    visual correctness; no mask precision or object-association claim is made.
    """
    output = validate_raw_response(raw, record=record, region=region, envelope=envelope)
    if isinstance(output, MissingOutput):
        return ParsedTarget(None, False, output.status, output.reason, output.local_index, envelope.response_id)
    if record.obj_id is None: raise ResponseIdentityError("target response requires a unique bound object; validate seed candidates first")
    target = _integer(record.obj_id, "bound obj_id")
    if absence_decision is not None:
        if not isinstance(absence_decision, AbsenceDecision): raise ResponseSchemaError("typed versioned absence decision required")
        try: absence_decision.validate(record, region, output.local_index)
        except ValueError as exc: raise ResponseIdentityError(str(exc)) from exc
    if target in output.removed_object_ids:
        # Pinned sam3_multiplex_base sets obj_id_to_score=-1e4 on removal,
        # and its public postprocessor can still expose a residual mask.
        # This is neither a probability nor proof of physical target absence.
        status = "target_absent" if absence_decision is not None else "target_visibility_unknown"
        response = FrameResponse(record.region_identity, record.session_id, record.generation,
                                 output.local_index, status, target,
                                 reason="official_removed_object_score_sentinel_residual_mask_not_usable",
                                 absence_decision=absence_decision)
        return ParsedTarget(response, True, status, response.reason, output.local_index, envelope.response_id,
                            public_output_contains_bound_id=True, available=False,
                            public_mask_nonempty=bool(output.masks[output.object_ids.index(target)].any()))
    if target not in output.object_ids:
        status = "target_absent" if absence_decision is not None else "target_visibility_unknown"
        reason = "explicit_versioned_absence_decision" if absence_decision is not None else "native_public_bound_id_missing_without_absence_decision"
        response = FrameResponse(record.region_identity, record.session_id, record.generation,
                                 output.local_index, status, target, reason=reason, absence_decision=absence_decision)
        return ParsedTarget(response, True, status, reason, output.local_index, envelope.response_id,
                            public_output_contains_bound_id=False)
    if absence_decision is not None: raise ResponseIdentityError("absence decision conflicts with a publicly present bound target")
    index = output.object_ids.index(target)  # Preserve raw row-to-mask association.
    mask = output.masks[index]
    if not mask.any():
        # Official postprocessing applies non-overlap constraints AFTER its
        # zero-area filter, so a retained public ID can still have zero pixels.
        reason = "native_bound_id_present_but_empty_mask"
        response = FrameResponse(record.region_identity, record.session_id, record.generation,
                                 output.local_index, "target_visibility_unknown", target, reason=reason)
        return ParsedTarget(response, True, "target_visibility_unknown", reason,
                            output.local_index, envelope.response_id,
                            public_output_contains_bound_id=True, public_mask_nonempty=False)
    header = json.dumps({"dtype": "bool", "shape": list(mask.shape)}, separators=(",", ":")).encode()
    pixel_sha = hashlib.sha256(header + b"\0" + mask.tobytes(order="C")).hexdigest()
    frame, pts = region.frame(output.local_index)
    binding = {"official_commit": OFFICIAL_COMMIT, "session_id": record.session_id,
               "region_identity": record.region_identity, "track_id": record.track_id,
               "seed_fingerprint": record.seed_fingerprint, "obj_id": target, "generation": record.generation,
               "attempt": record.attempt, "operation": envelope.operation,
               "propagation_round": record.completed_rounds, "local_index": output.local_index,
               "global_frame_id": frame, "pts": pts, "dependency_identity": record.dependency_identity,
               "pixel_identity": region.pixel_identity, "prompt_log_identity": record.prompt_log_identity}
    binding_id = _digest(binding)
    identity = "sam-mask-sha256:" + _digest({"binding": binding_id, "pixels": pixel_sha})
    request = MaskRegistrationRequest(identity, pixel_sha, binding_id,
                                      json.dumps(binding, sort_keys=True, separators=(",", ":")), mask)
    if not callable(register_mask): raise ResponseSchemaError("real input mask registration callback required")
    receipt = register_mask(request)
    if not isinstance(receipt, MaskRegistrationReceipt) or (receipt.mask_identity, receipt.pixel_sha256, receipt.binding_identity) != (identity, pixel_sha, binding_id):
        raise ResponseSchemaError("mask registration receipt does not match actual selected pixels and binding")
    _name(receipt.storage_reference, "mask storage reference")
    response = FrameResponse(record.region_identity, record.session_id, record.generation,
                             output.local_index, "target_present", target, identity)
    return ParsedTarget(response, True, "target_present", None, output.local_index, envelope.response_id, receipt,
                        public_output_contains_bound_id=True, available=True, public_mask_nonempty=True)


def ingest_response(collector, raw, *, envelope, register_mask, absence_decision=None):
    """Bridge propagation replies to RoundCollector; rejection poisons its round.

    Prompt replies are parsed against their own pending request record using
    parse_target_response; retain its frame_response as a prompt diagnostic for
    a later collector. Never pretend a prompt was sent during propagation.
    """
    try:
        if envelope.operation != "propagate_both":
            raise ResponseIdentityError("stream collector ingestion requires a propagation request envelope")
        parsed = parse_target_response(raw, record=collector.record, region=collector.region,
                                       envelope=envelope, register_mask=register_mask, absence_decision=absence_decision)
        if parsed.frame_response is not None:
            collector.add(parsed.frame_response, source="propagation")
        return parsed
    except Exception:
        collector.invalidate("raw_response_adapter_rejected")
        raise
