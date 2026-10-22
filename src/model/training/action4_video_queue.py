"""Durable video handoff to the canonical Dispatcher, never another scheduler.

The ledger tracks immutable epoch events; human reviews are supplied separately.
No helper in this module manufactures coverage, GT, reviews or successful jobs.
"""
from __future__ import annotations

import contextlib
import copy
import csv
import fcntl
import json
import hashlib
from datetime import datetime, timezone
import os
from pathlib import Path

from data.action4_paths import checked_root
from .action4_runtime import (atomic_json, canonical_hash, input_path, output_path,
                              sha256_file, write_json_new, validate_run_id, load_checkpoint)
from .action4_dispatch import Dispatcher, ACTIVE, descendants, is_live, ALLOWED_GPU_UUIDS


def execution_mode(config):
    mode = config.get("video", {}).get("execution", "inline")
    if mode not in ("inline", "queued"):
        raise ValueError("video.execution must be inline or queued")
    return mode


def validate_queue_config(config):
    if execution_mode(config) != "queued":
        raise ValueError("Queued configuration must declare queued execution")
    video = config.get("video", {})
    dispatch = video.get("dispatch")
    if not video.get("frame_manifest") or not isinstance(dispatch, dict):
        raise ValueError("Queued video requires a frame manifest and dispatch configuration")
    allowed = dispatch.get("allowed_gpu_uuids")
    if (not isinstance(allowed, list) or not allowed or len(set(allowed)) != len(allowed)
            or any(device not in ALLOWED_GPU_UUIDS for device in allowed)):
        raise ValueError("Queued video candidates must use unique authorized physical GPU UUIDs")
    if type(dispatch.get("reservation_mib")) is not int or dispatch["reservation_mib"] <= 0:
        raise ValueError("Queued video needs an explicit positive memory reservation")
    threads = dispatch.get("cpu_threads", 9)
    if type(threads) is not int or not 9 <= threads <= 32:
        raise ValueError("Queued video needs 9..32 reserved CPU processing threads")
    if not isinstance(dispatch.get("profile_key"), str) or not dispatch["profile_key"].strip():
        raise ValueError("Queued video requires an explicit profile_key")
    if dispatch.get("profile_id"):
        validate_run_id(dispatch["profile_id"])
    elif dispatch.get("profile_first") is not True:
        raise ValueError("Queued video needs profile_id or explicit profile_first")
    remaining = dispatch.get("estimated_remaining_bytes", 0)
    if type(remaining) is not int or remaining < 0:
        raise ValueError("Queued video estimated bytes must be a nonnegative integer")


def video_due(stage, epoch, total_epochs, config, *, component=False, completed=True):
    return (completed and stage in ("B0", "C1") and bool(config.get("video", {}).get("frame_manifest"))
            and (not component or config["video"].get("run_during_component", False))
            and (epoch % config["video"].get("every_epochs", 5) == 0 or epoch == total_epochs
                 or stage == "C1" and epoch % config["stages"]["C1"].get("review_interval_epochs", 5) == 0))


def epoch_pause_required(stage, epoch, total_epochs, outstanding, *, complete_epoch, component=False,
                         review_interval=5):
    if not complete_epoch or component or epoch >= total_epochs:
        return False
    return (stage == "B0" and outstanding >= 2
            or stage == "C1" and epoch % review_interval == 0)


def require_dispatch_owner(root, run_id):
    """A directly launched trainer cannot silently create unserviced GPU jobs."""
    root = checked_root(root)
    dispatcher = Dispatcher(root)
    expected = str(root / "models/action4" / run_id)
    with dispatcher.lock():
        jobs = dispatcher.state()["jobs"]
        owners = [job for job in jobs.values()
                  if job["status"] in ACTIVE and job["spec"].get("kind") == "train"
                  and expected in job["spec"].get("result_dirs", [])]
    if len(owners) != 1 or os.getpid() not in descendants(owners[0].get("worker")):
        raise ValueError("Queued training must be a live child of its canonical Dispatcher train job")


class VideoQueue:
    """Small idempotent handoff adapter; Dispatcher owns all jobs/resources."""
    def __init__(self, root, run_id, config):
        self.root = checked_root(root)
        validate_run_id(run_id)
        self.run_id = run_id
        self.config = copy.deepcopy(config)
        validate_queue_config(config)
        self.directory = output_path(self.root, Path("outputs/action4") / run_id / "video_queue")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.directory / "ledger.json"
        self.config_path = self.directory / "video_config.json"
        self.dispatcher = Dispatcher(self.root)
        with self.lock():
            if self.config_path.exists():
                if json.loads(self.config_path.read_text()) != self.config:
                    raise ValueError("Frozen video configuration changed within the training run")
            else:
                write_json_new(self.config_path, self.config)
            if not self.ledger_path.exists():
                write_json_new(self.ledger_path, {
                    "schema_version": "action4.video_queue.v1", "run_id": run_id,
                    "config_sha256": sha256_file(self.config_path), "events": []})
            self._ledger()

    @contextlib.contextmanager
    def lock(self):
        with (self.directory / "handoff.lock").open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            yield

    def _ledger(self):
        ledger = json.loads(self.ledger_path.read_text())
        if (ledger.get("schema_version") != "action4.video_queue.v1" or ledger.get("run_id") != self.run_id
                or ledger.get("config_sha256") != sha256_file(self.config_path)):
            raise ValueError("Video ledger/config identity changed")
        return ledger

    def _code(self):
        paths = sorted((self.root / "src").rglob("*.py"))
        paths += [self.root / "scripts/action4_video.py", self.root / "scripts/action4_dispatch.py"]
        return {str(input_path(self.root, p).relative_to(self.root)): sha256_file(p) for p in paths}

    def _verify_event(self, event):
        if event["event_sha256"] != canonical_hash(event["identity"]):
            raise ValueError("Video event identity changed")
        identity = event["identity"]
        if identity["code"] != self._code() or identity["config_sha256"] != sha256_file(self.config_path):
            raise ValueError("Video source/config changed; queued jobs cannot reuse stale profiles")
        for key in ("checkpoint", "frame_manifest"):
            if sha256_file(input_path(self.root, identity[key])) != identity[key + "_sha256"]:
                raise ValueError("Immutable video " + key + " identity changed")

    def ensure_event(self, stage, epoch, checkpoint, *, component=False):
        if stage not in ("B0", "C1") or type(epoch) is not int or epoch < 1:
            raise ValueError("Queued video requires a completed B0/C1 epoch")
        with self.lock():
            ledger = self._ledger()
            existing = next((e for e in ledger["events"] if e["identity"]["epoch"] == epoch), None)
            if existing:
                self._verify_event(existing)
                if existing["identity"]["stage"] != stage:
                    raise ValueError("Queued epoch stage changed")
                original = output_path(self.root, checkpoint)
                if str(original.relative_to(self.root)) != existing["identity"]["source_checkpoint"]:
                    raise ValueError("Queued epoch original checkpoint identity changed")
                if original.exists() and sha256_file(original) != existing["identity"]["checkpoint_sha256"]:
                    raise ValueError("Queued epoch original checkpoint bytes changed")
                return copy.deepcopy(existing)
            source = input_path(self.root, checkpoint)
            state = load_checkpoint(source)
            if (state.get("stage") != stage or state.get("epoch_boundary") is not True
                    or type(state.get("global_step")) is not int or state["global_step"] <= 0
                    or state.get("run_id") != self.run_id
                    or type(state.get("next_epoch")) is not int or not 0 < state["next_epoch"] <= epoch):
                raise ValueError("Video snapshot must be this run's qualified completed-epoch checkpoint")
            if state.get("config") != self.config:
                raise ValueError("Video snapshot configuration differs from the frozen merged configuration")
            if not component and state.get("claim_scope") != "accepted_data_model_training":
                raise ValueError("Component checkpoint cannot supply a formal queued video")
            snapshot = output_path(self.root, Path("models/action4") / self.run_id
                                   / "video_snapshots" / f"epoch_{epoch:03d}.pt")
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            if not snapshot.exists():
                os.link(source, snapshot)  # immutable inode survives top3 alias retirement
            if sha256_file(snapshot) != sha256_file(source):
                raise ValueError("Existing video snapshot differs from the original epoch checkpoint")
            frame_manifest = input_path(self.root, self.config["video"]["frame_manifest"])
            identity = {
                "stage": stage, "epoch": epoch, "component": bool(component),
                "source_checkpoint": str(source.relative_to(self.root)),
                "checkpoint": str(snapshot.relative_to(self.root)), "checkpoint_sha256": sha256_file(snapshot),
                "frame_manifest": str(frame_manifest.relative_to(self.root)),
                "frame_manifest_sha256": sha256_file(frame_manifest),
                "config_sha256": sha256_file(self.config_path), "code": self._code(),
            }
            event_sha = canonical_hash(identity)
            job_id = f"video_{self.run_id[:40]}_{canonical_hash(self.run_id)[:8]}_e{epoch:03d}"
            event = {"identity": identity, "event_sha256": event_sha, "job_id": job_id,
                     "result_dir": f"outputs/action4/{job_id}", "status": "prepared"}
            event["attempts"] = [{"number": 1, "job_id": job_id, "result_dir": event["result_dir"],
                                  "status": "prepared"}]
            ledger["events"].append(event)
            atomic_json(self.ledger_path, ledger)  # durable intent before enqueue
            return copy.deepcopy(event)

    def _spec(self, event, *, profile_id=None, profile=False):
        dispatch = self.config["video"].get("dispatch", {})
        if not isinstance(dispatch, dict):
            raise ValueError("video.dispatch must declare resource/profile parameters")
        for key in ("allowed_gpu_uuids", "reservation_mib", "profile_key"):
            if not dispatch.get(key):
                raise ValueError("video.dispatch missing " + key)
        cpu_threads = dispatch.get("cpu_threads", 9)
        if type(cpu_threads) is not int or cpu_threads < 9:
            raise ValueError("Queued video needs at least nine reserved CPU processing threads")
        identity = event["identity"]
        config_rel = str(self.config_path.relative_to(self.root))
        argv = [str(self.root / ".venv/bin/python"), str(self.root / "scripts/action4_video.py"), "--config", config_rel,
                "--checkpoint", identity["checkpoint"], "--frame-manifest", identity["frame_manifest"],
                "--run-id", event["job_id"], "--device", "cuda:0"]
        if identity["component"]:
            argv.append("--allow-component")
        spec = {
            "job_id": event["job_id"], "kind": "video", "cwd": str(self.root), "argv": argv,
            "output_dir": f"outputs/action4_dispatch_runs/{event['job_id']}",
            "result_dirs": [event["result_dir"]], "gpu_count": 1,
            "allowed_gpu_uuids": dispatch["allowed_gpu_uuids"],
            "reservation_mib": dispatch["reservation_mib"], "cpu_threads": cpu_threads,
            "profile_key": dispatch["profile_key"], "code": identity["code"],
            "configs": {config_rel: identity["config_sha256"]},
            "inputs": {identity[key]: identity[key + "_sha256"] for key in ("checkpoint", "frame_manifest")},
            "estimated_remaining_bytes": dispatch.get("estimated_remaining_bytes", 0),
        }
        if not profile:
            spec["profile_id"] = profile_id
        return spec

    def _attempt(self, event):
        # Additive compatibility for a prepared v1 ledger; never promote or
        # replace its original job. New events already include this history.
        if "attempts" not in event:
            event["attempts"] = [{key: event[key] for key in ("job_id", "result_dir", "status")}]
            event["attempts"][0]["number"] = 1
        attempts = event["attempts"]
        if (not 1 <= len(attempts) <= 2 or [a["number"] for a in attempts] != list(range(1, len(attempts) + 1))
                or event["job_id"] != attempts[-1]["job_id"] or event["result_dir"] != attempts[-1]["result_dir"]):
            raise ValueError("Invalid video attempt history/current binding")
        return attempts[-1]

    def _check_job_intent(self, event, job, profile_id, profile):
        expected = self._spec(event, profile_id=profile_id, profile=profile)
        expected["output_dir"] = str(output_path(self.root, expected["output_dir"]))
        expected["result_dirs"] = [str(output_path(self.root, p)) for p in expected["result_dirs"]]
        expected["profile"] = bool(profile)
        for key in ("argv", "code", "configs", "inputs", "profile_key", "profile_id", "profile",
                    "kind", "cwd", "output_dir", "result_dirs", "gpu_count", "allowed_gpu_uuids",
                    "reservation_mib", "cpu_threads", "estimated_remaining_bytes"):
            if job["spec"].get(key) != expected.get(key):
                raise ValueError("Existing dispatcher job differs from immutable video intent")

    def pump(self):
        """Replay intents, never automatically retry a failed/live attempt."""
        with self.lock():
            ledger = self._ledger()
            state = self.dispatcher.status()
            dispatch = self.config["video"].get("dispatch", {})
            profile_id = dispatch.get("profile_id")
            if not profile_id and ledger["events"] and dispatch.get("profile_first") is True:
                profile_id = ledger["events"][0]["job_id"]
            for index, event in enumerate(ledger["events"]):
                self._verify_event(event)
                attempt = self._attempt(event)
                job_id = event["job_id"]
                profile = dispatch.get("profile_first") is True and not dispatch.get("profile_id") and index == 0
                if job_id in state["jobs"]:
                    job = state["jobs"][job_id]
                    # A crash after enqueue adopts only that exact attempt.
                    bound_profile = attempt.get("profile_id", profile_id)
                    self._check_job_intent(event, job, bound_profile, profile)
                    event["status"] = attempt["status"] = job["status"]
                    continue
                if not profile:
                    if not profile_id:
                        raise ValueError("Queued video requires a measured profile or explicit profile_first")
                    profile_path = self.dispatcher.directory / "profiles" / (profile_id + ".json")
                    if not profile_path.exists():
                        event["status"] = attempt["status"] = "waiting_for_profile"
                        event["profile_status"] = "not_available"
                        continue
                    profile_record = json.loads(input_path(self.root, profile_path).read_text())
                    if profile_record.get("status") != "conclusive":
                        # Remain diagnosable and allow an explicitly requested
                        # isolated retry of the failed profile event.
                        event["status"] = attempt["status"] = "waiting_for_profile"
                        event["profile_status"] = "inconclusive"
                        continue
                attempt.update(profile=profile, profile_id=None if profile else profile_id)
                atomic_json(self.ledger_path, ledger)  # persist chosen profile before enqueue
                spec = self._spec(event, profile_id=profile_id, profile=profile)
                self.dispatcher.enqueue(spec, profile=profile)
                event["status"] = attempt["status"] = "queued"
                event.pop("profile_status", None)
                state = self.dispatcher.state()
            atomic_json(self.ledger_path, ledger)
            return copy.deepcopy(ledger)

    def retry_failed(self, epoch, *, controller, reason_category, reason):
        """Explicit controller-only one retry for evidenced transient IO/device/OOM.

        Requires terminal nonzero exit receipt and absent known owners/ranks.
        Method/input changes require another experiment, never this retry.
        """
        categories = {
            "oom": ("out of memory",),
            "transient_io": ("input/output error", "errno 5", "temporarily unavailable",
                             "connectionreseterror", "connection reset by peer"),
            "transient_device": ("cuda-capable device(s) is/are busy or unavailable",
                                 "cuda error: system not yet initialized", "driver shutting down"),
        }
        if (reason_category not in categories
                or not all(isinstance(value, str) and value.strip() for value in (controller, reason))):
            raise ValueError("Retry requires controller, detailed reason and a supported transient category")
        with self.lock():
            ledger = self._ledger()
            event = next((e for e in ledger["events"] if e["identity"]["epoch"] == epoch), None)
            if event is None:
                raise ValueError("Unknown video epoch")
            self._verify_event(event)
            attempt = self._attempt(event)
            if len(event["attempts"]) >= 2:
                raise ValueError("Exactly one isolated retry is allowed per video event")
            state = self.dispatcher.status()
            job = state["jobs"].get(event["job_id"])
            if not job or job["status"] != "failed" or type(job.get("exitcode")) is not int or job["exitcode"] == 0:
                raise ValueError("Retry requires terminal failed attempt with known nonzero exit")
            profile = self.config["video"]["dispatch"].get("profile_first") is True and not self.config["video"]["dispatch"].get("profile_id") and event is ledger["events"][0]
            self._check_job_intent(event, job, attempt.get("profile_id"), profile)
            out = input_path(self.root, job["spec"]["output_dir"])
            exit_path = input_path(self.root, out / "dispatch_exit.json")
            exit_receipt = json.loads(exit_path.read_text())
            if (not isinstance(job.get("token"), str) or not job["token"]
                    or exit_receipt.get("token") != job["token"] or exit_receipt.get("exitcode") != job["exitcode"]):
                raise ValueError("Retry exit receipt identity is missing or inconsistent")
            worker_path = input_path(self.root, out / "dispatch_worker.json")
            worker_receipt = json.loads(worker_path.read_text())
            if (worker_receipt.get("token") != job["token"] or worker_receipt.get("worker") != job.get("worker")
                    or worker_receipt.get("child") != job.get("child")):
                raise ValueError("Retry worker receipt token/owner mismatch")
            identities = [job.get("worker"), job.get("child"), worker_receipt.get("worker"),
                          worker_receipt.get("child")] + job.get("tracked_processes", [])
            if any(not isinstance(value, dict) or not {"pid", "start", "command_sha256"} <= value.keys()
                   for value in identities):
                raise ValueError("Retry owner/rank identity is unknown; observe before retrying")
            # Query failures propagate without recording/submitting a retry.
            if any(is_live(identity) or descendants(identity) for identity in identities):
                raise ValueError("Retry cannot start while a known worker/rank/descendant is live")
            log_path = input_path(self.root, out / "stdout.log")
            log_bytes = log_path.read_bytes()
            log_text = log_bytes.decode("utf-8", errors="replace").lower()
            deterministic = ("valueerror:", "assertionerror:", "typeerror:", "keyerror:", "identity changed",
                             "size mismatch", "shape mismatch", "device-side assert", "illegal memory access")
            if any(marker in log_text for marker in deterministic) or not any(marker in log_text for marker in categories[reason_category]):
                raise ValueError("Failure evidence is deterministic/unknown or does not support the retry category")
            attempt["status"] = "failed"
            attempt["failure_evidence"] = {
                "exitcode": job["exitcode"], "exit_receipt_sha256": sha256_file(exit_path),
                "worker_receipt_sha256": sha256_file(worker_path),
                "stdout_sha256": hashlib.sha256(log_bytes).hexdigest(),
                "known_processes_confirmed_absent": identities,
            }
            job_id = event["attempts"][0]["job_id"] + "_retry1"
            validate_run_id(job_id)
            new = {"number": 2, "job_id": job_id, "result_dir": f"outputs/action4/{job_id}",
                   "status": "prepared", "retry_of": event["attempts"][0]["job_id"],
                   "controller": controller, "reason_category": reason_category, "reason": reason,
                   "requested_at_utc": datetime.now(timezone.utc).isoformat(),
                   "event_sha256": event["event_sha256"]}
            event["attempts"].append(new)
            event.update(job_id=job_id, result_dir=new["result_dir"], status="prepared")
            event.pop("profile_status", None)
            atomic_json(self.ledger_path, ledger)
        # Failure/crash here leaves durable new intent; pump() replays it once.
        self.pump()
        with self.lock():
            return copy.deepcopy(next(e for e in self._ledger()["events"] if e["identity"]["epoch"] == epoch))

    def _reviewed(self, event):
        """A signed decision is data, checked against complete real output coverage."""
        self._verify_event(event)
        number = self._attempt(event)["number"]
        suffix = "" if number == 1 else f"_attempt_{number:02d}"
        review_path = self.directory / "reviews" / f"epoch_{event['identity']['epoch']:03d}{suffix}.json"
        if not review_path.exists():
            return False
        review = json.loads(input_path(self.root, review_path).read_text())
        required = {"schema_version": "action4.video_review.v1", "job_id": event["job_id"],
                    "event_sha256": event["event_sha256"], "status": "reviewed"}
        if any(review.get(k) != value for k, value in required.items()):
            raise ValueError("Review receipt identity/schema mismatch")
        if not all(isinstance(review.get(k), str) and review[k].strip()
                   for k in ("reviewer", "evidence", "observation")):
            raise ValueError("Review requires a real reviewer, observation and evidence")
        if review.get("decision") not in ("continue", "stop"):
            raise ValueError("Review decision must explicitly continue or stop")
        if event["status"] != "succeeded":
            raise ValueError("Review cannot promote an unfinished/failed video job")
        result_dir = input_path(self.root, event["result_dir"])
        files = {"run_manifest": "run_manifest.json", "completion": "completion.json",
                 "predictions": "predictions_raw.jsonl", "coverage": "review_coverage.csv",
                 "video": "full_red_raw.mp4"}
        for key, name in files.items():
            if review.get(key + "_sha256") != sha256_file(input_path(self.root, result_dir / name)):
                raise ValueError("Review " + key + " identity mismatch")
        run = json.loads((result_dir / "run_manifest.json").read_text())
        complete = json.loads((result_dir / "completion.json").read_text())
        if (run.get("checkpoint_sha256") != event["identity"]["checkpoint_sha256"]
                or run.get("frame_manifest_sha256") != event["identity"]["frame_manifest_sha256"]
                or complete.get("status") != "complete_io_pending_accuracy_review"):
            raise ValueError("Video completion/checkpoint/frame identity mismatch")
        frames = json.loads(input_path(self.root, event["identity"]["frame_manifest"]).read_text())
        with (result_dir / "review_coverage.csv").open(newline="") as f:
            coverage = list(csv.DictReader(f))
        expected = frames["frames"]
        if len(coverage) != len(expected) or len(coverage) != run.get("frame_count") or len(coverage) != complete.get("frame_count"):
            raise ValueError("Review coverage must include every source frame exactly once")
        for row, frame in zip(coverage, expected):
            if (row.get("frame_id") != str(frame["frame_id"]) or row.get("pts") != str(frame["pts"])
                    or row.get("time_base") != frames["source"]["time_base"]
                    or row.get("status") != "reviewed" or not row.get("reviewer", "").strip()
                    or not row.get("notes", "").strip()):
                raise ValueError("Incomplete or mismatched frame review coverage")
        if event["identity"]["stage"] == "C1":
            current = self.root / "outputs/action4" / self.run_id
            epoch = event["identity"]["epoch"]
            candidates = list(current.glob(f"epoch_{epoch:03d}_step_*/eval_metrics.json"))
            candidates += list(current.glob(f"resume_*/epoch_{epoch:03d}_step_*/eval_metrics.json"))
            comparison = review.get("c1_comparison", {})
            if len(candidates) != 1:
                raise ValueError("C1 current-epoch eval evidence is missing or ambiguous")
            for key, path in (("baseline_eval_sha256", current / "initial_eval/eval_metrics.json"),
                              ("current_eval_sha256", candidates[0])):
                if comparison.get(key) != sha256_file(input_path(self.root, path)):
                    raise ValueError("C1 comparison eval identity mismatch")
            if comparison.get("improved") is not True or not comparison.get("ordinary_scene_review"):
                raise ValueError("C1 continuation requires evidenced improvement and ordinary-scene review")
        if review["decision"] == "stop":
            raise ValueError("Controller review stops continuation; preserve the current run")
        return True

    def state(self):
        ledger = self.pump()
        outstanding = []
        for event in ledger["events"]:
            if not self._reviewed(event):
                outstanding.append(event["job_id"])
        return {"outstanding": outstanding, "events": ledger["events"],
                "failed": [e["job_id"] for e in ledger["events"] if e["status"] == "failed"],
                "all_jobs_succeeded": bool(ledger["events"]) and all(e["status"] == "succeeded" for e in ledger["events"])}

    def require_resume(self, stage, next_epoch):
        with self.lock():
            ledger = self._ledger()
            expected = {epoch for epoch in range(1, next_epoch + 1)
                        if video_due(stage, epoch, self.config["stages"][stage]["epochs"], self.config)}
            recorded = {e["identity"]["epoch"] for e in ledger["events"]}
            if (recorded != expected or any(e["identity"]["stage"] != stage for e in ledger["events"])):
                raise ValueError("Missing or inconsistent historical video event; never reconstruct old best from later weights")
        state = self.state()
        if state["failed"] or stage == "C1" and state["outstanding"] or stage == "B0" and len(state["outstanding"]) >= 2:
            raise ValueError("Queued video completion and truthful identity-bound review required before resume")
        return state
