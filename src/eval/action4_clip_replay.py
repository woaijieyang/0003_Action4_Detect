"""Read-only cold-start clip adapter for actual native_tracks artifacts.

No native inference is run. evidence_refs maps project-relative metadata paths
to SHA256 values sealed by the caller AFTER the producer has terminated.
native_tracks does not itself hash SEEDS/CLOSE/RAW_RECEIPT, so these explicit
refs are required, not inferred from filenames. Missing evidence fails closed.

Only generation0 independent per-track sessions are supported. Explicit forward
mode uses seed..end responsibility; both mode remains offline/noncausal. Seed failure
and native system failure retain every core frame as diagnostic rows; system
failure cannot feed A0. Geometry is provisional, never full-bound calibration
acceptance. No whole-video state or mask-tight box substitutes a body box.
"""
import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from eval.action4_sam_full_review import ReceiptVerifier, checked_path, ROOT, candidates_for_frame
from eval.action4_sam_b0_sequence import (select_seeds, load_instance_seed_points,
    sequence_direction, forward_responsibility, forward_response)
from eval.action4_sam_image_eval import validate_source_snapshot, bind_native_output
from model.action4_sam_session_contracts import RegionSpec, SeedBinding, SessionLedger
from model.action4_sam_response_adapter import (
    ResponseEnvelope, MissingOutput, validate_raw_response, ingest_response, MaskRegistrationReceipt)
from model.action4_sam_supported_boxes import SamSupportedBoxes
from data.action4_video_roi_context import (
    VideoROIContext, VideoSAMSupportReceipt, VideoSAMSupportAdmission, rgb_digest)
from data.action4_sam_roi_inputs import array_digest


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


class ClipReplay:
    def __init__(self, *, registry_ref, clip_id, config_ref, tracker_config_ref, producer_exit_ref, evidence_refs):
        self.verifier = ReceiptVerifier()
        self.refs = dict(evidence_refs)
        self.tracker_config = self.verifier.bound_json(tracker_config_ref)
        policy = self.tracker_config
        keys = {"geometry_mode", "window"}
        if policy.get("geometry_mode") == "sam_b0_fallback_v2":
            keys.add("fallback_policy")
        if set(policy) != keys or type(policy.get("window")) is not int:
            raise ValueError("Explicit exact tracker configuration required")
        self._make_tracker([])  # Validate policy even for failed/zero-seed clips.
        self.registry = self.verifier.bound_json(registry_ref)
        matches = [c for c in self.registry["clips"] if c["clip_id"] == clip_id]
        if len(matches) != 1:
            raise ValueError("One registered clip required")
        self.clip = matches[0]
        c = self.clip
        if (c.get("session_policy") != "independent_clip_cold_start"
                or c.get("reuse_whole_video_track_state") is not False
                or c.get("core_only_scoring") is not True):
            raise ValueError("Independent registered core-only clip required")
        self.cfg = self.verifier.bound_json(config_ref)
        self.producer_exit_ref = copy.deepcopy(producer_exit_ref)
        self.producer_exit = _validated_exit(self.verifier, producer_exit_ref, config_ref)
        cfg = self.cfg
        self.direction = sequence_direction(cfg)
        self.causal = self.direction == "forward"
        if cfg.get("schema") != "action4.sam.b0_video_sequence.v1" or cfg.get("discovery_seed_only", False):
            raise ValueError("Actual complete sequence configuration required")
        self.out = checked_path(cfg["output"])
        manifest_path = checked_path(cfg["source_snapshot"]) / "SNAPSHOT_MANIFEST.json"
        manifest = self._json(manifest_path)
        if validate_source_snapshot(cfg) != manifest:
            raise ValueError("Immutable source identity mismatch")
        self.source_identity = manifest["identity"]
        self.doc = self.verifier.bound_json(cfg["region"])
        d = self.doc
        if (d.get("region_id") != clip_id or d.get("core") != c["core"]
                or d.get("session_policy") != "independent_clip_cold_start"
                or d.get("inherited_session") is not None or d.get("core_only_scoring") is not True
                or d["source_manifest"] != self.registry["source_manifest"]):
            raise ValueError("Clip region is not the registered independent domain")
        frames = d["frames"]
        if ([f["global_frame_id"] for f in frames] != c["input_frame_ids"]
                or c["input_frame_ids"] != list(range(*c["context"]))
                or c["evaluation_frame_ids"] != list(range(*c["core"]))
                or len(c["evaluation_frame_ids"]) != c["expected_core_frames"]
                or cfg["sequence"]["frame_count"] != len(frames)):
            raise ValueError("Clip frame/core coverage mismatch")
        source = self.verifier.bound_json(d["source_manifest"])
        original = {f["frame_id"]: f for f in source["frames"]}
        self.by_global = {}
        for i, f in enumerate(frames):
            g = original[f["global_frame_id"]]
            if (f["local_index"] != i or (f["pts"], f["image"], f["sha256"])
                    != (g["pts"], g["image"], g["image_sha256"])):
                raise ValueError("Local/global PTS/RGB mapping differs")
            self.by_global[f["global_frame_id"]] = i
        bindings = [{"frame_id": f["global_frame_id"], "pts": f["pts"], "RGB_sha256": f["sha256"]}
                    for f in frames if f["global_frame_id"] in c["evaluation_frame_ids"]]
        if bindings != c["core_frame_bindings"]:
            raise ValueError("Registered core RGB/PTS bindings differ")
        self.region = RegionSpec(clip_id, tuple(f["global_frame_id"] for f in frames),
            tuple(f["pts"] for f in frames), tuple(d["time_base"]), cfg["region"]["sha256"],
            d["pixel_space_id"], d["width"], d["height"])
        # Validate the actual isolated loader resource, not merely the mapping.
        staged = checked_path(d["staged_frames"])
        names = {f"{i:06d}.png" for i in range(len(frames))}
        if {p.name for p in staged.iterdir()} != names:
            raise ValueError("Isolated frame folder has missing/extra entries")
        for i, f in enumerate(frames):
            self.verifier.verify(f["image"], f["sha256"])
            self.verifier.verify(staged / f"{i:06d}.png", f["sha256"])
        b0ref = cfg.get("b0_evidence")
        b0 = self.verifier.bound_json(b0ref) if b0ref else self._json(self.out / "B0_COMPLETE.json")
        if (b0["b0"] != cfg["b0"] or b0["region"] != cfg["region"]
                or b0["frames"] != len(frames) or b0.get("status") != "B0_REGION_EVIDENCE_COMPLETE"):
            raise ValueError("B0 model/region/coverage differs")
        if b0ref:
            predecessor = self.verifier.bound_json(cfg["b0_source_manifest"])
            if (_digest({k: v for k, v in predecessor.items() if k != "identity"}) != predecessor["identity"]
                    or b0["source_identity"] != predecessor["identity"]):
                raise ValueError("B0 immutable predecessor identity differs")
        elif b0["source_identity"] != self.source_identity:
            raise ValueError("B0 producer source differs")
        poolpath = self.verifier.verify(b0["pools"]["path"], b0["pools"]["sha256"])
        self.b0_pool_identity = b0["pools"]["sha256"]
        self.b0_pool_ref = copy.deepcopy(b0["pools"])
        self.pools = [json.loads(line) for line in poolpath.read_text().splitlines()]
        if [r["frame"] for r in self.pools] != frames:
            raise ValueError("B0 rows are reordered or incomplete")
        for row, frame in zip(self.pools, frames):
            candidates_for_frame(row, frame)
        selection = select_seeds(self.pools, self.region,
            {"model_sha256": cfg["b0"]["sha256"], "source_sha256": b0["source_identity"]},
            cfg["seed_policy"])
        if self._json(self.out / "SEEDS.json") != selection:
            raise ValueError("Saved seed selection differs from frozen B0 selection")
        self.seeds = selection["seeds"]
        if self.causal and any(s["local_index"] != cfg["seed_policy"]["confirmations"]-1 for s in self.seeds):
            raise ValueError("Causal seed must use declared common prefix confirmation boundary")
        self.owners = {}
        self.mask_refs = {}
        self.native_status = {}
        self.failure_evidence = []
        _terminal_state(self.cfg, self.producer_exit, self._json)
        complete_path = self.out / "NATIVE_COMPLETE.json"
        failures = list(self.out.glob("*/FAILED.json")) + list(self.out.glob("FAILED.json"))
        if failures:
            if complete_path.exists():
                raise ValueError("Completion contradicts retained system failure")
            self.failure_evidence = [
                {"path": str(p.relative_to(ROOT)), "sha256": self.refs[str(p.relative_to(ROOT))],
                 "failure": self._json(p), "error": self._json(p).get("error"),
                 "traceback": self._json(p).get("traceback")} for p in failures]
            self.status = "system_failure"
        elif not self.seeds:
            if complete_path.exists():
                raise ValueError("Zero-seed native completion is not an actual producer format")
            self.status = "seed_failure"
        else:
            if not complete_path.exists():
                raise ValueError("Nonterminal or unsupported producer: missing completion/failure")
            self._authenticate_complete(self._json(complete_path), selection)
            self.status = "native_complete_not_accuracy"
        self.implementation_identity = implementation_identity()
        self.replay_identity = _digest({"config": config_ref, "registry": registry_ref,
            "clip_id": clip_id, "tracker_config": tracker_config_ref,
            "producer_exit_ref": producer_exit_ref, "implementation": self.implementation_identity,
            "evidence_refs": self.refs, "status": self.status})
        self._register_project_bodies()
        self._build_rows()

    def _json(self, path):
        path = checked_path(path)
        key = str(path.relative_to(ROOT))
        if key not in self.refs:
            raise ValueError("Unsealed metadata reference: " + key)
        return self.verifier.bound_json({"path": key, "sha256": self.refs[key]})

    def _authenticate_complete(self, done, selection):
        cfg, region = self.cfg, self.region
        if (done.get("status") != ("CAUSAL_FORWARD_NATIVE_RESPONSIBILITY_COMPLETE_NOT_ACCURACY" if self.causal else "CONTINUOUS_NATIVE_SAM_RESPONSES_COMPLETE_NOT_ASSOCIATION_EFFECT_ACCEPTANCE")
                or done.get("source_identity") != self.source_identity
                or done.get("region_identity") != region.identity
                or done.get("accuracy_accepted") is not False):
            raise ValueError("Native completion domain/status differs")
        if [x["track_id"] for x in done["tracks"]] != [s["track_id"] for s in self.seeds]:
            raise ValueError("Completed tracks differ from selected seed count/order")
        ledger = SessionLedger()
        pairs = set()
        candidates = [{"candidate_id": s["candidate_id"], "bbox_sam_native_xyxy": s["bbox_native"],
                       "eligible_for_SAM_diagnostic_prompt": True} for s in self.seeds]
        for info, result in zip(self.seeds, done["tracks"]):
            tid = info["track_id"]
            folder = self.out / tid
            if not folder.resolve().is_relative_to(self.out.resolve()):
                raise ValueError("Track folder escapes clip")
            if checked_path(result["snapshot"]) != folder / "SNAPSHOT.json":
                raise ValueError("Snapshot belongs to another clip/track")
            saved = self.verifier.bound_json({"path": result["snapshot"], "sha256": result["snapshot_sha256"]})
            snap = saved["snapshot"]
            if snap["generation"] != 0 or snap["attempt"] != 0 or snap["propagation_round"] != 0:
                raise ValueError("Only native_tracks cold-start generation0 is supported")
            sid, oid = snap["session_id"], snap["obj_id"]
            if (sid, oid) in pairs:
                raise ValueError("Duplicate native session/object binding")
            pairs.add((sid, oid))
            evidence = tuple(info["evidence_ids"])
            points = None
            if cfg.get("seed_mode", "box_text") == "instance_points":
                points = load_instance_seed_points(cfg, info, self.seeds)
                if self._json(folder / "INSTANCE_POINTS.json") != points[2]:
                    raise ValueError("Point seed evidence differs")
                evidence += ("predicted_mask_points:" + points[2]["identity"],)
            elif cfg.get("seed_mode", "box_text") != "box_text":
                raise ValueError("Unsupported seed mode")
            responsibility, proof = forward_responsibility(region, info) if self.causal else (region, None)
            offset = info["local_index"] if self.causal else 0
            if self.causal and self._json(folder / "RESPONSIBILITY.json") != proof:
                raise ValueError("Causal responsibility mapping differs")
            seed = SeedBinding(tid, responsibility.identity, info["candidate_id"], info["local_index"]-offset,
                info["global_frame_id"], tuple(info["box_xywh01"]), evidence, "frozen_b0")
            record = ledger.plan(responsibility, seed, worker_id="sam_sequence_worker",
                device_identity=cfg["resources"]["gpu_uuid"], dependency_identity=self.source_identity)
            ledger.begin_create(record.record_id)
            ledger.created(record.record_id, sid)
            if points is None:
                request = ledger.seed_request(record.record_id)
                request["text"] = cfg["native"]["text"]
            else:
                request = ledger.seed_request(record.record_id, points_xy01=points[0],
                    point_labels=points[1], obj_id=0, point_evidence_id=points[2]["identity"])
            if self.causal:
                collector_request = dict(request)
                request = dict(request, frame_index=info["local_index"])
                if self._json(folder / "SEED_REQUEST_MAPPING.json") != {
                        "collector_request":collector_request,"native_request":request,"responsibility":proof}:
                    raise ValueError("Causal native/collector seed request mapping differs")
            binding = self._json(folder / "SEED_BINDING.json")
            if binding["request"] != request:
                raise ValueError("Actual native seed request differs")
            raw = self.verifier.raw(self._json(folder / "RAW_RECEIPT.json"), folder)
            normalized = forward_response(raw, region, responsibility, offset, offset) if self.causal else raw
            parsed = validate_raw_response(normalized, record=ledger.get(record.record_id), region=responsibility,
                envelope=ResponseEnvelope(sid, responsibility.identity, 0, 0, "seed", "seed", seed.local_index))
            if isinstance(parsed, MissingOutput):
                raise ValueError("Completed run has missing seed response")
            candidate = next(c for c in candidates if c["candidate_id"] == seed.candidate_id)
            actual_oid, reason, native = bind_native_output(parsed,
                {"native_shape_hw": [region.height, region.width], "candidates": candidates},
                candidate, cfg["binding"])
            if binding["native_candidates"] != native:
                from eval.action4_sam_binding_replay import replay_native_diagnostics
                native=replay_native_diagnostics(native,parsed.boxes_xywh01,width=region.width,height=region.height,
                    target_box=candidate["bbox_sam_native_xyxy"],producer_load=self._json(self.out/"NATIVE_LOAD.json"),
                    source_identity=self.source_identity)
            if (actual_oid != oid or binding["obj_id"] != oid or binding["reason"] != reason
                    or binding["native_candidates"] != native or oid is None or points is not None and oid != 0):
                raise ValueError("Actual unique seed/native object binding differs")
            ledger.bind_seed(record.record_id, [oid])
            collector = ledger.begin_propagation(record.record_id)
            if self.causal and self._json(folder / "PROPAGATION_REQUEST.json") != {
                    "type":"propagate_in_video","session_id":sid,"propagation_direction":"forward",
                    "start_frame_index":offset}:
                raise ValueError("Actual causal native propagation request differs")
            response_dirs = sorted(folder.glob("response_*"))
            if [p.name for p in response_dirs] != [f"response_{i:04d}" for i in range(len(response_dirs))]:
                raise ValueError("Native response ordinal gap")
            used = set()
            def register(req):
                entry = saved["registrations"].get(req.mask_identity)
                if entry is None or (entry["pixel_sha256"], entry["binding_identity"], entry["binding"]) != (
                        req.pixel_sha256, req.binding_identity, json.loads(req.binding_json)):
                    raise ValueError("Mask registration binding differs from actual native response")
                path = self.verifier.verify(entry["path"], entry["sha256"])
                if path != response_dir / "selected_mask.npz":
                    raise ValueError("Selected mask outside actual response directory")
                with np.load(path, allow_pickle=False) as z:
                    if (set(z.files) != {"mask"} or z["mask"].dtype != np.bool_
                            or not np.array_equal(z["mask"], req.mask)):
                        raise ValueError("Selected mask differs from native response")
                local = entry["binding"]["local_index"] + offset
                self.mask_refs[(tid, local)] = copy.deepcopy(entry)
                used.add(req.mask_identity)
                return MaskRegistrationReceipt(req.mask_identity, req.pixel_sha256,
                                               req.binding_identity, entry["path"])
            for ordinal, response_dir in enumerate(response_dirs):
                raw = self.verifier.raw(self._json(response_dir / "RAW_RECEIPT.json"), response_dir)
                normalized = forward_response(raw, region, responsibility, offset, offset+ordinal) if self.causal else raw
                parsed = ingest_response(collector, normalized, envelope=ResponseEnvelope(
                    sid, responsibility.identity, 0, 0, "propagate_both", f"response_{ordinal:04d}"), register_mask=register)
                self.native_status[(tid, parsed.local_index+offset)] = parsed.status
            snapshot = ledger.finish_propagation(record.record_id, collector)
            if (not snapshot.responses_complete or snapshot.identity != saved["identity"]
                    or json.loads(json.dumps(asdict(snapshot))) != snap
                    or used != set(saved["registrations"])
                    or result["frames"] != len(snapshot.frames) or result["available"] != snapshot.mask_available_count):
                raise ValueError("Replayed native responses differ from completed snapshot")
            ledger.publish(record.record_id, identity_confirmed=True)
            close = self._json(folder / "CLOSE.json")
            if close.get("released") is not True or close.get("response", {}).get("is_success") is not True:
                raise ValueError("Native close not acknowledged")
            ledger.close(record.record_id, released=True)
            self.owners[tid] = {"session_id": sid, "native_obj_id": oid, "generation": 0,
                "ledger_sha": _digest({"snapshot": result, "close": close}),
                "project_seed":{"seed_fingerprint":seed.fingerprint,"creation_global_frame":info["global_frame_id"],
                    "creation_pts":self.doc["frames"][info["local_index"]]["pts"],
                    "seed_binding_ref":{"path":str((folder/"SEED_BINDING.json").relative_to(ROOT)),
                                        "sha256":self.refs[str((folder/"SEED_BINDING.json").relative_to(ROOT))]},
                    "seeds_ref":{"path":str((self.out/"SEEDS.json").relative_to(ROOT)),
                                 "sha256":self.refs[str((self.out/"SEEDS.json").relative_to(ROOT))]}}}
        if json.loads(json.dumps(ledger.audit_log)) != done["ledger"]:
            raise ValueError("Native completion ledger differs from reconstructed operations")

    def _mask(self, tid, local):
        entry = self.mask_refs.get((tid, local))
        if entry is None:
            return None
        with np.load(self.verifier.verify(entry["path"], entry["sha256"]), allow_pickle=False) as z:
            return z["mask"].copy()

    def _make_tracker(self, seeds):
        policy = self.tracker_config
        tracker = SamSupportedBoxes(seeds, window=policy["window"],
                                    geometry_mode=policy["geometry_mode"])
        if policy["geometry_mode"] == "sam_b0_fallback_v2":
            if tracker.fallback_policy != policy["fallback_policy"]:
                raise ValueError("Consumed fallback policy differs from bound configuration")
            for seed in seeds:
                appearance = np.asarray(seed.get("initial_appearance128", []), float)
                if (appearance.shape != (128,) or not np.isfinite(appearance).all()
                        or np.linalg.norm(appearance) == 0):
                    raise ValueError("Fallback needs actual frozen B0 seed appearance")
        return tracker

    def _register_project_bodies(self):
        # This is an explicit project creation policy for authenticated fixed
        # initial seeds. Native session/mask generation never sets body generation.
        scope={"clip_id":self.clip["clip_id"],"region_identity":self.region.identity,
               "producer_identity":self.source_identity}
        self.body_scope=dict(scope,schema="action4.project_body_scope.v1",sequence_identity=_digest(scope))
        self.body_events={}
        if self.status!="native_complete_not_accuracy":return
        for tid,owner in self.owners.items():
            event={"schema":"action4.project_body_registration.v1","event_type":"project_body_created",
                "sequence_identity":self.body_scope["sequence_identity"],"producer_identity":self.source_identity,
                "replay_identity":self.replay_identity,"track_id":tid,
                "scoped_body_id":_digest({"sequence_identity":self.body_scope["sequence_identity"],"track_id":tid}),
                "project_generation":0,**copy.deepcopy(owner["project_seed"]),
                "time_base":list(self.region.time_base),"pixel_space_id":self.region.pixel_identity,
                "semantics":"provisional_seed_association_not_semantic_ground_truth","semantic_gt":False,
                "physical_left_right_verified":False}
            event["event_identity"]=_digest(event)
            self.body_events[tid]=event

    def _build_rows(self):
        self.rows = {}
        tracker = self._make_tracker(self.seeds) if self.status == "native_complete_not_accuracy" else None
        for local, frame in enumerate(self.doc["frames"]):
            initializing = self.causal and bool(self.seeds) and local < self.seeds[0]["local_index"]
            masks = {s["track_id"]: self._mask(s["track_id"], local) for s in self.seeds} if tracker else {}
            fused = (tracker.update(frame["global_frame_id"], frame["pts"] * self.region.time_base[0] / self.region.time_base[1],
                         candidates_for_frame(self.pools[local], frame), masks) if tracker and not initializing else
                     {"mode": "video_inference", "frame_id": frame["global_frame_id"], "tracks": [
                         {"track_id":s["track_id"],"status":"initializing","bbox_for_roi":None,
                          "raw_detection":None,"candidate_id":None,"sam_current_support":False}
                         for s in self.seeds] if initializing else []})
            self.rows[frame["global_frame_id"]] = {"frame": copy.deepcopy(frame), "fused": fused,
                "clip_status": self.status, "seed_count": len(self.seeds), "time_base": list(self.region.time_base),
                "pixel_space_id": self.region.pixel_identity,
                "native_unavailable": [tid for tid, mask in masks.items() if mask is None or not mask.any()],
                "B0_raw": copy.deepcopy(self.pools[local]["pool"]),
                "full_box_accuracy_verified": False, "calibration_scope": "predicted_B0_provisional_not_trusted_full",
                "tracker_config": copy.deepcopy(self.tracker_config),
                "native_owners": copy.deepcopy(self.owners),
                "project_body_scope":copy.deepcopy(self.body_scope),
                "project_body_registrations":{tid:copy.deepcopy(event) for tid,event in self.body_events.items()
                    if frame["global_frame_id"]>=event["creation_global_frame"] and frame["pts"]>=event["creation_pts"]},
                "native_status": {s["track_id"]: self.native_status.get(
                    (s["track_id"], local), "initializing_no_native_response" if initializing else "not_authenticated_due_to_" + self.status) for s in self.seeds},
                "causal": self.causal, "propagation_direction": self.direction,
                "temporal_mode": "causal_forward" if self.causal else "offline_bidirectional_clip", "not_approved_for_causal_B0_track": not self.causal,
                "seed_frame_by_track": {s["track_id"]: s["global_frame_id"] for s in self.seeds},
                "preseed_replayed": not self.causal and self.status == "native_complete_not_accuracy" and any(
                    frame["global_frame_id"] < s["global_frame_id"] for s in self.seeds),
                "preseed_by_track": {s["track_id"]: not self.causal and self.status == "native_complete_not_accuracy"
                    and frame["global_frame_id"] < s["global_frame_id"] for s in self.seeds}}


    def core_rows(self):
        """All registered core frames, including seed/system failure diagnostics."""
        return [copy.deepcopy(self.rows[f]) for f in self.clip["evaluation_frame_ids"]]

    def public_result(self):
        """Complete JSON-safe audit result; failures/preseed frames are retained."""
        from collections import Counter
        rows = self.core_rows()
        statuses = Counter(status for row in rows for status in row["native_status"].values())
        coverage = {"expected_core_frames": self.clip["expected_core_frames"],
                    "retained_core_frames": len(rows), "seed_count": len(self.seeds),
                    "registered_track_core_slots": len(rows)*len(self.seeds),
                    "native_status_counts": dict(statuses),
                    "scope": "selected-track slots only; zero/one seed never fabricates a missing second ID"}
        return {"schema": "action4.clip_replay_result.v1", "clip_id": self.clip["clip_id"],
                "status": self.status, "replay_identity": self.replay_identity,
                "producer_exit_ref": copy.deepcopy(self.producer_exit_ref),
                "producer_exit": copy.deepcopy(self.producer_exit),
                "implementation_identity": copy.deepcopy(self.implementation_identity),
                "failure_evidence": copy.deepcopy(self.failure_evidence),
                "b0_pool_identity": self.b0_pool_identity, "b0_pool_ref": copy.deepcopy(self.b0_pool_ref),
                "seed_count": len(self.seeds), "causal": self.causal, "propagation_direction": self.direction,
                "temporal_mode": "causal_forward" if self.causal else "offline_bidirectional_clip", "not_approved_for_causal_B0_track": not self.causal,
                "scope": ("independent clip causal forward responsibility; accuracy not accepted" if self.causal
                          else "independent clip offline bidirectional replay; not online cold-start"),
                "accuracy_accepted": False, "core_coverage": coverage, "core_rows": rows}

    def load(self, frame_id, K):
        """Return original CHW RGB and typed context; never certify geometry."""
        if self.status == "system_failure":
            raise ValueError("System failure is diagnostic-only and cannot feed A0")
        if frame_id not in self.rows:
            raise ValueError("Frame is outside registered clip")
        from PIL import Image
        import torch
        row = copy.deepcopy(self.rows[frame_id])
        local = self.by_global[frame_id]
        f = row["frame"]
        with Image.open(self.verifier.verify(f["image"], f["sha256"])) as im:
            rgb = torch.from_numpy(np.array(im.convert("RGB"), copy=True)).permute(2, 0, 1)
        if rgb.shape != (3, self.region.height, self.region.width):
            raise ValueError("RGB dimensions differ")
        expected, actual = [], []
        visible = {}
        for track in row["fused"]["tracks"]:
            tid = track["track_id"]
            mask = self._mask(tid, local)
            if mask is not None and mask.any():
                ys, xs = np.nonzero(mask)
                visible[tid] = [float(xs.min()), float(ys.min()), float(xs.max()+1), float(ys.max()+1)]
            if track["status"] != "sam_supported":
                continue
            if mask is None:
                raise ValueError("SAM-supported geometry lacks actual mask")
            owner = self.owners[tid]
            receipt = VideoSAMSupportReceipt(frame_id, f["pts"], self.region.time_base, rgb_digest(rgb),
                self.region.pixel_identity, tid, owner["native_obj_id"], owner["session_id"], 0, 0,
                self.replay_identity, self.source_identity, _digest(self.owners), array_digest(mask),
                tuple(track["bbox_for_roi"]))
            expected.append(receipt)
            actual.append((receipt, mask))
        row["_roi_visible_boxes"] = visible
        # Calibration frames from heuristic predictions are deliberately NOT
        # promoted into _roi_last_trusted_full.
        support = (VideoSAMSupportAdmission(self.region.pixel_identity, self.region.time_base,
                   self.source_identity, _digest(self.owners), tuple(expected), tuple(actual)) if expected else None)
        return rgb, VideoROIContext.from_replay(row, rgb, K, b0_sha256=self.cfg["b0"]["sha256"],
                                                replay_identity=self.replay_identity, sam_support=support)


def freeze_metadata_refs(config_ref, output, *, producer_exit_ref):
    """Index existing terminal producer metadata, never infer masks or create completion."""
    verifier=ReceiptVerifier();cfg=verifier.bound_json(config_ref);root=checked_path(cfg["output"])
    if cfg.get("schema")!="action4.sam.b0_video_sequence.v1":
        raise ValueError("Explicit native sequence config required")
    exit_record = _validated_exit(verifier, producer_exit_ref, config_ref)
    _terminal_state(cfg, exit_record, read_json_file)
    source=checked_path(cfg["source_snapshot"])/"SNAPSHOT_MANIFEST.json"
    refs={}
    for current,dirs,files in __import__("os").walk(root,followlinks=False):
        current=Path(current)
        if any((current/name).is_symlink() for name in dirs+files):
            raise ValueError("Metadata index refuses symlink entries")
        for name in files:
            if name.endswith(".json"):
                path=checked_path(current/name)
                refs[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    refs[str(source.relative_to(ROOT))]=hashlib.sha256(source.read_bytes()).hexdigest()
    target=checked_path(output)
    if target.is_relative_to(root):raise ValueError("Consumer metadata index must not mutate producer output")
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        existing=read_json_file(target)
        if (existing.get("schema")!="action4.clip_metadata_refs.v1" or existing.get("config_ref")!=config_ref
                or existing.get("evidence_refs")!=refs or existing.get("producer_exit_ref")!=producer_exit_ref):
            raise ValueError("Existing metadata index has different source identity")
        return refs
    with target.open("x") as stream:
        json.dump({"schema":"action4.clip_metadata_refs.v1","config_ref":config_ref,
            "producer_exit_ref":producer_exit_ref,"evidence_refs":refs,"scope":"metadata identity index; ClipReplay must authenticate responses/closure",
            "new_native_inference":False},stream,indent=2)
    return refs


def read_json_file(path):
    return json.loads(path.read_text())


def implementation_identity():
    """Hash actual adapter dependencies, not a guessed checkout identity.

    This is a consumer-code identity. Native inference still requires its
    complete immutable producer source snapshot, already checked separately.
    """
    import inspect
    import importlib
    modules = [__name__, "eval.action4_sam_binding_replay", "model.action4_sam_supported_boxes",
               "model.action4_sam_session_contracts", "model.action4_sam_response_adapter",
               "data.action4_video_roi_context", "eval.action4_sam_b0_sequence",
               "eval.action4_sam_full_review", "eval.action4_sam_image_eval"]
    files = {}
    for name in modules:
        module = importlib.import_module(name)
        path = checked_path(inspect.getfile(module))
        files[name] = {"path": str(path.relative_to(ROOT)),
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return {"schema": "action4.clip_consumer_implementation.v1", "modules": files,
            "scope": "consumer implementation files; full native inference source requires producer snapshot"}


def record_process_exit(process, config_ref, output):
    """Record an actual owned Popen exit; never treat FAILED as process exit.

    The launcher calls this after its native producer has terminated. A real
    Popen handle is required, preventing a caller from supplying a guessed PID
    or return code. This function does not launch, wait for, or stop a process.
    """
    import subprocess
    if not isinstance(process, subprocess.Popen):
        raise TypeError("Actual owned subprocess.Popen handle required")
    returncode = process.poll()
    if returncode is None:
        raise ValueError("Producer still running; exit cannot be recorded")
    verifier = ReceiptVerifier()
    cfg = verifier.bound_json(config_ref)
    command = process.args
    if (not isinstance(command, (list, tuple)) or not command
            or any(not isinstance(arg, str) for arg in command)):
        raise ValueError("Explicit argv list required for producer exit provenance")
    _validate_producer_argv(command, config_ref)
    value = {"schema": "action4.clip_producer_exit.v1", "evidence": "owned_Popen_poll",
             "pid": process.pid, "command": list(command), "returncode": returncode,
             "config_ref": copy.deepcopy(config_ref)}
    target = checked_path(output)
    if target.is_relative_to(checked_path(cfg["output"])):
        raise ValueError("Exit receipt must not mutate native producer output")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
    return {"path": str(target.relative_to(ROOT)),
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}



def _validate_producer_argv(command, config_ref):
    """Bind native_tracks CLI mode and config to the actual recorded argv.

    Config paths follow this project's CLI convention: relative paths resolve
    under ROOT. Equals-form flags and end-of-options are rejected rather than
    ambiguously interpreted; exactly one explicit pair of each flag is required.
    """
    if "--" in command:
        raise ValueError("Producer argv cannot hide options after --")
    for flag in ("--config", "--mode"):
        if command.count(flag) != 1 or any(arg.startswith(flag + "=") for arg in command):
            raise ValueError("Producer argv requires exactly one " + flag + " value")
        index = command.index(flag)
        if index + 1 >= len(command) or command[index + 1].startswith("--"):
            raise ValueError("Missing producer argv value for " + flag)
    actual = checked_path(command[command.index("--config") + 1])
    expected = checked_path(config_ref["path"])
    if actual.resolve(strict=True) != expected.resolve(strict=True):
        raise ValueError("Producer argv config differs from bound config_ref")
    if command[command.index("--mode") + 1] != "sam":
        raise ValueError("Producer argv must select --mode sam")

def _validated_exit(verifier, ref, config_ref):
    value = verifier.bound_json(ref)
    if (value.get("schema") != "action4.clip_producer_exit.v1"
            or value.get("evidence") != "owned_Popen_poll" or value.get("config_ref") != config_ref
            or type(value.get("pid")) is not int or value["pid"] <= 0
            or type(value.get("returncode")) is not int
            or not isinstance(value.get("command"), list) or not value["command"]
            or any(not isinstance(arg, str) for arg in value["command"])):
        raise ValueError("Bound actual producer exit evidence required")
    _validate_producer_argv(value["command"], config_ref)
    return value


def _terminal_state(cfg, exit_record, read):
    root = checked_path(cfg["output"])
    complete, seeds = root / "NATIVE_COMPLETE.json", root / "SEEDS.json"
    failed = list(root.glob("*/FAILED.json")) + list(root.glob("FAILED.json"))
    zero_seed = seeds.exists() and read(seeds).get("seeds") == []
    if complete.exists() and failed:
        raise ValueError("Mixed terminal evidence")
    if complete.exists():
        if exit_record["returncode"] != 0:
            raise ValueError("Native completion contradicts nonzero producer exit")
    else:
        if not failed and not zero_seed:
            raise ValueError("Nonterminal producer: missing completion/failure")
        if exit_record["returncode"] == 0:
            raise ValueError("Failure/zero-seed outcome requires nonzero producer exit")
    # Failure files can precede finally/CLOSE. Check after observed exit, never
    # equate FAILED or process exit alone with released native resources.
    for directory in root.iterdir():
        if not directory.is_dir():
            continue
        checked_path(directory)
        started = any((directory / name).exists() for name in
                      ("FAILED.json", "RAW_RECEIPT.json", "SEED_BINDING.json", "SNAPSHOT.json"))
        started = started or any(directory.glob("response_*"))
        if not started:
            continue
        close = directory / "CLOSE.json"
        if not close.exists():
            raise ValueError("cleanup_unconfirmed: started/failed track lacks CLOSE")
        receipt = read(close)
        if receipt.get("released") is not True or receipt.get("response", {}).get("is_success") is not True:
            raise ValueError("cleanup_unconfirmed: native release not acknowledged")
    return "complete" if complete.exists() else "seed_failure" if zero_seed and not failed else "system_failure"
