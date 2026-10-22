"""Durable project-local GPU task queue. No shell, retries or external kills.

Independent tasks and fixed 1/2/4-GPU training share one atomic reservation ledger.
Only the trusted training entry is wrapped by a fixed torch.distributed.run launcher.
Specs bind project files by SHA256 and each attempt owns a fresh output directory.
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import uuid as uuidlib

from data.action4_devices import ALLOWED_GPU_UUIDS, worker_environment
from data.action4_quality_device import gpu_identity, QUALITY_GPU_UUID
from data.action4_paths import PROJECT_ROOT, checked_root

HEADROOM_MIB = 3072  # Additional reserve; each task still reserves ceil(1.25 * measured peak).
DISK_FLOOR = 100 * 1024 ** 3
ACTIVE = {"launching", "running"}
TERMINAL = {"succeeded", "failed"}
# Exact long options only: argparse abbreviation and repeated last-wins options
# must not bypass the registered root/output/device contracts.
TRAIN_OPTIONS = set("root config local-config stage run-id init-weights resume d0-gate device allow-diagnostic component-smoke max-updates".split())
EVAL_OPTIONS = set("root config local-config checkpoint split run-id gt-roi selection freeze-selection tune-grid device allow-diagnostic".split())
EVAL_SWITCHES = {"gt-roi", "allow-diagnostic"}
OPTIONS = TRAIN_OPTIONS | set("root queue help config local-config checkpoint frame-manifest run-id device allow-component debug-numbers manifest roi-count updates max-seconds max-memory-gib probe-only recipe output count seed requests max-count source-asset-dir hand-bundle output-asset-dir output-hand-dir max-trials plan shard-index num-shards limit gpu-uuid entry-start entry-count blender threads shards acceptance output-dir output-root output-json destination dest out".split())
WRITE_OPTIONS = {"output", "output-dir", "output-root", "output-json", "output-asset-dir", "output-hand-dir", "destination", "dest", "out"}
RANK_KEYS = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK",
             "ROLE_RANK", "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def bounded(root, value, *, exists=False):
    path = Path(value)
    if ".." in path.parts:
        raise ValueError("Parent path traversal is forbidden")
    path = (path if path.is_absolute() else root / path).resolve(strict=exists)
    if not path.is_relative_to(root):
        raise ValueError("Path escapes project root: " + str(value))
    return path


def atomic(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuidlib.uuid4().hex + ".tmp")
    with open(tmp, "x", encoding="utf-8") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def process_identity(pid):
    """Query only a known PID using a system command; never inspect /proc files."""
    result = subprocess.run(["ps", "-p", str(int(pid)), "-o", "pid=,lstart=,stat=,args="],
                            text=True, capture_output=True, check=False, timeout=5)
    fields = result.stdout.strip().split()
    if result.returncode == 1 and not fields:
        return None
    if result.returncode != 0 or (fields and len(fields) < 8):
        raise RuntimeError("Process identity query failed; observation is inconclusive")
    if not fields or fields[6].startswith("Z"):
        return None
    return {"pid": int(fields[0]), "start": " ".join(fields[1:6]),
            "command_sha256": hashlib.sha256(" ".join(fields[7:]).encode()).hexdigest()}


def is_live(identity):
    return bool(identity and process_identity(identity["pid"]) == identity)


def descendants(identity):
    if not is_live(identity):
        return set()
    found, todo = {identity["pid"]}, [identity["pid"]]
    while todo:
        result = subprocess.run(["pgrep", "-P", str(todo.pop())], capture_output=True, text=True)
        if result.returncode not in (0, 1):
            raise RuntimeError("Cannot inspect own child process identities")
        for entry in result.stdout.split():
            child = int(entry)
            if child not in found:
                found.add(child)
                todo.append(child)
    return found


def gpu_process_memory(device):
    gpu_identity(device)  # recheck physical mapping before every telemetry query
    output = subprocess.check_output(["nvidia-smi", "-i", device,
        "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"], text=True)
    values = {}
    for line in output.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 2 or not fields[0].isdigit() or not fields[1].isdigit():
            raise RuntimeError("GPU process memory telemetry unavailable; no admission")
        values[int(fields[0])] = int(fields[1])
    return values


def assigned_devices(job):
    return job.get("devices", [job["device"]] if job.get("device") else [])


def reserved_mib(spec, device):
    return spec.get("reservation_mib_by_gpu", {}).get(device, spec.get("reservation_mib", 0))


def measured_mib(job, device):
    return job.get("current_mib_by_gpu", {}).get(device, job.get("current_mib", 0) if job.get("device") == device else 0)


def workload_argv(root, spec, token):
    """Only our fixed launcher can create a multi-rank process group."""
    if spec["kind"] != "train":
        return spec["argv"]
    count = spec["gpu_count"]
    return [str(root / ".venv/bin/python"), "-m", "torch.distributed.run", "--nnodes=1",
            "--nproc_per_node=" + str(count), "--max_restarts=0", "--rdzv_backend=c10d",
            "--rdzv_endpoint=127.0.0.1:0", "--rdzv_id=" + token,
            "--log_dir=" + str(bounded(root, Path(spec["output_dir"]) / "torchrun")),
            "--redirects=3", "--tee=0"] + spec["argv"][1:]


def admission(free_mib, own_current_mib, reservations_mib, requested_mib):
    # free already subtracts external usage AND actual own usage. Add back only
    # measured owned processes before subtracting their full booked reservations.
    return free_mib + own_current_mib - reservations_mib - HEADROOM_MIB >= requested_mib


class Dispatcher:
    def __init__(self, root, queue="outputs/action4_dispatch"):
        self.root = checked_root(root)
        self.directory = bounded(self.root, queue)
        if not self.directory.is_relative_to(self.root / "outputs"):
            raise ValueError("Dispatcher state must be under outputs")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.state_file = self.directory / "queue.json"

    @contextlib.contextmanager
    def lock(self, name="queue", blocking=True):
        path = bounded(self.root, self.directory / (name + ".lock"))
        with open(path, "a+") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
            except BlockingIOError as exc:
                raise RuntimeError("Another scheduler is active") from exc
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def state(self):
        return read(bounded(self.root, self.state_file)) if self.state_file.exists() else {"schema": "action4.dispatch.v1", "jobs": {}}

    def save(self, state):
        atomic(bounded(self.root, self.state_file), state)

    def _effective_workload(self, spec, script, flags, count):
        """Derive resources from actual entry inputs, not caller profile labels."""
        from model.training.action4_runtime import canonical_hash, load_config
        role = {"entry": str(script.relative_to(self.root))}
        cpu = {"minimum": 1, "rule": "declared independent worker pool"}
        if spec["kind"]=="quality":
            cfg=read(bounded(self.root,flags["config"],exists=True))
            if cfg.get("schema")!="action4.quality_job.v1" or cfg.get("mode") not in ("smoke","stream") or cfg.get("threads_per_worker")!=2 or cfg.get("workers") not in (1,4):raise ValueError("Explicit quality workload contract required")
            cpu={"minimum":cfg["workers"]*2,"rule":"quality spawned workers times two CPU threads"}
            role.update(quality_config=cfg)
        elif script.name == "action4_train.py":
            config = load_config(bounded(self.root, flags["config"], exists=True),
                                 bounded(self.root, flags["local-config"], exists=True) if flags.get("local-config") else None)
            runtime = config.get("runtime", {})
            compute = runtime.get("cpu_threads", 2)
            workers = runtime.get("stage_workers", {}).get(flags["stage"], runtime.get("workers", 0))
            validation_workers = runtime.get("validation_workers", 1)
            if type(validation_workers) is not int or validation_workers < 1:
                raise ValueError("Validation worker configuration is invalid")
            if type(compute) is not int or compute < 1 or type(workers) is not int or workers < 0:
                raise ValueError("Training CPU thread/worker configuration is invalid")
            total = count * (compute + max(workers, validation_workers if validation_workers > 1 else 0))
            cpu = {"minimum": total, "compute_per_rank": compute, "loader_workers_per_rank": workers,
                   "validation_workers_per_rank": validation_workers, "ranks": count, "rule": "ranks * (compute threads + max(loader workers, validation workers)); phases do not overlap"}
            video = config.get("video", {})
            if flags["stage"] in ("B0", "C1") and video.get("frame_manifest") and ("component-smoke" not in flags or video.get("run_during_component", False)):
                cpu["inline_video_phase"] = self._encoding_cpu_budget()
                cpu["minimum"] = max(total, cpu["inline_video_phase"]["minimum"])
            role.update(training_stage=flags["stage"], merged_config_sha256=canonical_hash(config))
        elif script == self.root / "scripts/action4_eval.py":
            # GT-ROI diagnostics, full-image inference and threshold grids have
            # different workloads; their measured GPU profiles cannot substitute.
            mode = ("freeze" if "freeze-selection" in flags else "tune" if "tune-grid" in flags
                    else "gt_roi" if "gt-roi" in flags else "full_rgb")
            role.update(evaluation_mode=mode, evaluation_split=flags.get("split", "eval"))
        elif script.name == "action4_diagnostic.py":
            cpu = {"minimum": 2, "compute_threads": 2, "rule": "diagnose explicitly calls torch.set_num_threads(2)"}
        elif script.name == "action4_video.py":
            cpu = self._encoding_cpu_budget()
        elif script in (self.root / "scripts/action4_synthesis.py", self.root / "scripts/action4_render_cached.py", self.root / "scripts/action4_render_fast.py") and spec["argv"][2:3] == ["render-shard"]:
            plan = read(bounded(self.root, flags["plan"], exists=True))
            recipe, assets = plan.get("recipe"), plan.get("asset_identities")
            if not isinstance(recipe, dict) or not isinstance(assets, dict) or not assets:
                raise ValueError("Render plan needs its actual embedded recipe and frozen asset identities")
            resolution = recipe.get("resolution_wh")
            if not isinstance(resolution, list) or len(resolution) != 2 or any(type(v) is not int or v < 1 for v in resolution):
                raise ValueError("Invalid effective render resolution")
            if not isinstance(recipe.get("render"), dict):
                raise ValueError("Effective render settings are missing")
            for name, digest in assets.items():
                path = bounded(self.root, name, exists=True)
                if not isinstance(digest, str) or sha256(path) != digest:
                    raise ValueError("Effective render asset identity changed: " + name)
            cpu = {"minimum": int(flags["threads"]), "rule": "explicit Blender --threads"}
            # Full embedded recipe binds resolution, backend, renderer settings,
            # geometry/material/randomization regime. Top-level seed/entries,
            # shard index/range, plan path and result paths are intentionally not
            # resource identities, so same-recipe shards can share a profile.
            role.update(render_recipe=recipe, asset_identities=assets, render_threads=int(flags["threads"]))
            if flags.get("blender"):
                binary = bounded(self.root, flags["blender"], exists=True)
                role["blender_sha256"] = sha256(binary)
        role["cpu_contract"] = cpu
        return role, canonical_hash(role), cpu["minimum"]

    @staticmethod
    def _encoding_cpu_budget():
        import inspect
        from data.action4_adapter import DataAdapter
        threads = inspect.signature(DataAdapter.encode_video).parameters["threads"].default
        if type(threads) is not int or threads < 1:
            raise ValueError("Video encoder thread default needs an explicit supported contract")
        return {"minimum": 2 * threads + 1, "decode_threads": threads, "encode_threads": threads,
                "filter_threads": 1, "rule": "conservative concurrent decoder + encoder + filter pools; inference runs in a separate phase"}

    def _validate(self, supplied, *, profile=False):
        spec = json.loads(json.dumps(supplied))
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,95}", spec.get("job_id", "")):
            raise ValueError("Invalid job_id")
        if spec.get("kind") not in {"render", "video", "inference", "component", "cpu", "train", "quality"}:
            raise ValueError("Invalid job kind")
        cwd = bounded(self.root, spec.get("cwd", "."), exists=True)
        if cwd != self.root:
            raise ValueError("Jobs must use project root cwd")
        spec["cwd"] = str(cwd)
        out = bounded(self.root, spec["output_dir"])
        if not out.is_relative_to(self.root / "outputs") or out == self.root / "outputs":
            raise ValueError("Runtime outputs must use a unique outputs subdirectory")
        spec["output_dir"] = str(out)
        argv = spec.get("argv")
        if not isinstance(argv, list) or len(argv) < 2 or not all(isinstance(x, str) and x for x in argv):
            raise ValueError("argv must be a nonempty string array with a project script")
        # Only the project Python interpreter and a hashed Python script are
        # supported. No -c/-m, shells, executables selected via PATH, or eval.
        interpreter = Path(argv[0])
        if not interpreter.is_absolute():
            interpreter = self.root / interpreter
        if str(interpreter) != str(self.root / ".venv/bin/python"):
            raise ValueError("Use the project .venv/bin/python executable")
        script = bounded(self.root, argv[1], exists=True)
        if script.suffix != ".py" or not script.is_relative_to(self.root / "scripts"):
            raise ValueError("Entrypoint must be a project scripts/*.py file")
        if (spec["kind"] == "train") != (script == self.root / "scripts/action4_train.py"):
            raise ValueError("Training jobs must exclusively use scripts/action4_train.py")
        if (spec["kind"]=="quality") != (script==self.root/"scripts/action4_quality.py"):
            raise ValueError("Quality GPU3 requires the exclusive trusted quality entry")
        argv[0], argv[1] = str(interpreter), str(script)
        for arg in argv[2:]:
            if any(c in arg for c in ("\x00", "\n", "\r")) or arg in {";", "&&", "||", "|", ">", "<"}:
                raise ValueError("Invalid command argument")
            value = arg.split("=", 1)[1] if arg.startswith("--") and "=" in arg else arg
            if Path(value).is_absolute() or ".." in Path(value).parts:
                bounded(self.root, value)
        flags = {}
        for i, arg in enumerate(argv[2:], 2):
            if arg.startswith("--"):
                option = arg[2:].split("=", 1)[0]
                if option not in (EVAL_OPTIONS if script == self.root / "scripts/action4_eval.py" else TRAIN_OPTIONS if spec["kind"] == "train" else OPTIONS) or option in flags:
                    raise ValueError("Unknown, abbreviated or duplicate option: " + arg)
                value = arg.split("=", 1)[1] if "=" in arg else (argv[i + 1] if i + 1 < len(argv) and not argv[i + 1].startswith("--") else None)
                flags[option] = value
            elif arg.startswith("-") and not re.fullmatch(r"-\d+(?:\.\d+)?", arg):
                raise ValueError("Short options are forbidden; use exact long options")
        if "root" in flags and (flags["root"] is None or bounded(self.root, flags["root"], exists=True) != self.root):
            raise ValueError("--root must equal registered project cwd")
        if spec["kind"] == "train":
            if flags.get("stage") not in ("D0", "A0", "B0", "C1") or not flags.get("config"):
                raise ValueError("Training needs a known stage and explicit frozen config")
            spec["training_stage"] = flags["stage"]
            if "init-weights" in flags and "resume" in flags:
                raise ValueError("Training initialization and resume are mutually exclusive")
            if bool(flags.get("resume")) != bool(spec.get("resume_from_job")):
                raise ValueError("Training resume requires a recorded previous job")
        elif spec.get("resume_from_job"):
            raise ValueError("Only training continuation can reuse owned run directories")
        eval_entry = script == self.root / "scripts/action4_eval.py"
        eval_freeze = eval_entry and "freeze-selection" in flags
        if eval_entry:
            if spec["kind"] not in {"cpu", "inference", "component"}:
                raise ValueError("Evaluation requires cpu/inference/component kind")
            # Keep the existing CLI's valued options and Boolean switches exact.
            # Generic worker arguments must not acquire evaluation-only flags.
            for option, value in flags.items():
                if (option in EVAL_SWITCHES and value is not None
                        or option not in EVAL_SWITCHES and not value):
                    raise ValueError("Invalid evaluation option value: --" + option)
            for i, arg in enumerate(argv[2:], 2):
                if not arg.startswith("--"):
                    previous = argv[i - 1]
                    if (not previous.startswith("--") or "=" in previous
                            or previous[2:] in EVAL_SWITCHES):
                        raise ValueError("Unexpected positional evaluation argument")
            if not flags.get("config") or not flags.get("checkpoint"):
                raise ValueError("Evaluation requires explicit config and checkpoint")
            split = flags.get("split", "eval")
            if split not in {"eval", "test"}:
                raise ValueError("Evaluation split must be eval or test")
            if "tune-grid" in flags:
                if split != "eval" or any(k in flags for k in ("gt-roi", "selection", "freeze-selection")):
                    raise ValueError("Threshold tuning requires eval without GT-ROI, selection or freeze")
            elif eval_freeze:
                if split != "eval" or any(k in flags for k in ("gt-roi", "allow-diagnostic", "selection", "run-id")):
                    raise ValueError("Selection freeze is a standalone mode without evaluation/diagnostic options")
                if spec["kind"] != "cpu" or spec.get("allowed_gpu_uuids") or flags.get("device", "cpu") != "cpu":
                    raise ValueError("Selection freeze is CPU-only and must not reserve GPUs")
            elif split == "test":
                if not flags.get("selection") or any(k in flags for k in ("gt-roi", "allow-diagnostic")):
                    raise ValueError("Formal test requires frozen selection and forbids GT-ROI/diagnostic modes")
            elif "selection" in flags:
                raise ValueError("Frozen selection belongs only to formal test")
        result_dirs = []
        for value in spec.get("result_dirs", []):
            result = bounded(self.root, value)
            legal = result.is_relative_to(self.root / "outputs") or (spec["kind"] == "render" and result.is_relative_to(self.root / "data")) or (spec["kind"] in ("component", "train") and result.is_relative_to(self.root / "models"))
            if not legal or result in (self.root / "outputs", self.root / "data", self.root / "models"):
                raise ValueError("Invalid registered result directory")
            if result == out or result in out.parents or out in result.parents:
                raise ValueError("Result and dispatcher-log directories must be disjoint")
            if result.exists() and not spec.get("resume_from_job"):
                raise ValueError("Registered result directory already exists; never overwrite")
            if any(result == p or result in p.parents or p in result.parents for p in result_dirs):
                raise ValueError("Result directories overlap")
            result_dirs.append(result)
        spec["result_dirs"] = [str(p) for p in result_dirs]
        write_options = WRITE_OPTIONS | ({"freeze-selection"} if eval_entry else set())
        for option in write_options.intersection(flags):
            if flags[option] is None:
                raise ValueError("Output flag has no value")
            target = bounded(self.root, flags[option])
            if not any(target == p or target.is_relative_to(p) for p in result_dirs):
                raise ValueError("Output flag is outside registered fresh result directories")
            if option == "freeze-selection" and (target.exists() or target in result_dirs or not target.is_relative_to(self.root / "outputs")):
                raise ValueError("Selection freeze requires a new file inside a registered outputs directory")
        if script in (self.root / "scripts/action4_render_cached.py", self.root / "scripts/action4_render_fast.py") and argv[2:3] != ["render-shard"]:
            raise ValueError("Cached renderer requires the public render-shard command")
        if script == self.root / "scripts/action4_synthesis.py" and argv[2:3] == ["_render-shard"]:
            raise ValueError("Internal Blender render entry must not be enqueued directly")
        if script in (self.root / "scripts/action4_synthesis.py", self.root / "scripts/action4_render_cached.py", self.root / "scripts/action4_render_fast.py") and argv[2:3] == ["render-shard"]:
            if spec["kind"] != "render" or not flags.get("output"):
                raise ValueError("render-shard needs kind=render and an explicit output")
            if not flags.get("threads"):
                raise ValueError("render-shard requires explicit --threads; implicit defaults cannot bypass CPU reservation")
            # Reuse the renderer's authoritative mapping, never duplicate its
            # data -> runtime-evidence path convention in the scheduler.
            from data.action4_synthesis import _run_output_dir
            runtime_output = bounded(self.root, _run_output_dir(bounded(self.root, flags["output"])))
            if runtime_output not in result_dirs:
                raise ValueError("Derived render runtime output must be registered in result_dirs: " + str(runtime_output))
        if script.name in {"action4_video.py", "action4_diagnostic.py", "action4_train.py"} or (eval_entry and not eval_freeze):
            run_id = flags.get("run-id")
            if not run_id or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,95}", run_id):
                raise ValueError("Workload requires an exact safe --run-id")
            expected = self.root / "outputs/action4" / run_id
            if expected not in result_dirs:
                raise ValueError("Workload default output must be registered in result_dirs")
            if script.name in {"action4_diagnostic.py", "action4_train.py"} and self.root / "models/action4" / run_id not in result_dirs:
                raise ValueError("Training/diagnostic model output must also be registered in result_dirs")
        if "device" in flags and flags["device"] not in ("cuda", "cuda:0", "cpu", "auto"):
            raise ValueError("Independent worker device must be cuda:0 after UUID masking")
        for group in ("code", "configs", "inputs"):
            bindings = spec.setdefault(group, {})
            if not isinstance(bindings, dict):
                raise ValueError("Identity maps must be objects")
            for name, digest in bindings.items():
                path = bounded(self.root, name, exists=True)
                if not path.is_file() or not re.fullmatch(r"[0-9a-f]{64}", digest) or sha256(path) != digest:
                    raise ValueError("Stale or invalid " + group + " identity: " + name)
        if str(script.relative_to(self.root)) not in spec["code"]:
            raise ValueError("Entrypoint source SHA256 is required")
        # Freeze imported implementation too, not only the thin CLI wrapper.
        # New/deleted/changed source files invalidate a queued spec or profile.
        for source in sorted((self.root / "src").rglob("*.py")) + [self.root / "scripts/action4_dispatch.py"]:
            path = bounded(self.root, source, exists=True)
            name, digest = str(path.relative_to(self.root)), sha256(path)
            if name in spec["code"] and spec["code"][name] != digest:
                raise ValueError("Imported source identity changed: " + name)
            spec["code"][name] = digest
        if not spec["configs"]:
            raise ValueError("At least one frozen configuration hash is required")
        # Bind explicitly referenced inputs/configs, including immutable video
        # snapshots; positional image lists are caller-declared in inputs.
        for flag, group in (("--selection", "inputs"), ("--tune-grid", "configs"),
                            ("--checkpoint", "inputs"), ("--frame-manifest", "inputs"), ("--manifest", "inputs"),
                            ("--plan", "inputs"), ("--recipe", "configs"), ("--requests", "inputs"),
                            ("--config", "configs"), ("--local-config", "configs"), ("--resume", "inputs"),
                            ("--init-weights", "inputs"), ("--d0-gate", "inputs")):
            for i, arg in enumerate(argv):
                if arg == flag or arg.startswith(flag + "="):
                    value = arg.split("=", 1)[1] if "=" in arg else argv[i + 1]
                    rel = str(bounded(self.root, value, exists=True).relative_to(self.root))
                    if rel not in spec[group]:
                        raise ValueError(flag + " must have an immutable hash binding")
        if spec["kind"] == "video" and not any(a == "--checkpoint" or a.startswith("--checkpoint=") for a in argv):
            raise ValueError("Video jobs require a hashed checkpoint")
        allowed = spec.get("allowed_gpu_uuids", [])
        if not isinstance(allowed, list) or len(allowed) != len(set(allowed)) or any(x not in ((QUALITY_GPU_UUID,) if spec["kind"]=="quality" else ALLOWED_GPU_UUIDS) for x in allowed):
            raise ValueError("Only authorized physical GPU UUIDs may be requested")
        if spec["kind"]=="quality" and allowed!=[QUALITY_GPU_UUID]:raise ValueError("Quality work requires only physical GPU3")
        if "gpu-uuid" in flags and allowed != [flags["gpu-uuid"]]:
            raise ValueError("Explicit --gpu-uuid must match a single registered allowed UUID")
        if allowed and flags.get("device") == "cpu":
            raise ValueError("CPU device conflicts with GPU reservation")
        if not allowed and flags.get("device") in ("cuda", "cuda:0"):
            raise ValueError("CUDA device requires an allowed UUID reservation")
        count = spec.get("gpu_count", 1 if allowed else 0)
        if not isinstance(count, int) or isinstance(count, bool) or count not in (0, 1, 2, 4, 6):
            raise ValueError("gpu_count must be 0/1/2/4/6")
        if bool(allowed) != bool(count) or count > len(allowed):
            raise ValueError("gpu_count does not match allowed GPUs")
        if count > 1 and spec["kind"] != "train":
            raise ValueError("Only the trusted training entry can use multiple GPUs")
        if spec["kind"] == "train" and (count == 0 or len(allowed) != count):
            raise ValueError("Training requires a fixed ordered UUID group of gpu_count devices")
        reservation = spec.get("reservation_mib", 0)
        if not isinstance(reservation, int) or isinstance(reservation, bool) or reservation < 0:
            raise ValueError("Invalid memory reservation")
        by_gpu = spec.get("reservation_mib_by_gpu")
        if by_gpu is None:
            if count > 1:
                raise ValueError("Multi-GPU jobs require explicit per-GPU reservations")
            by_gpu = {device: reservation for device in allowed}
        if not isinstance(by_gpu, dict) or set(by_gpu) != set(allowed) or any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in by_gpu.values()):
            raise ValueError("Per-GPU reservation map must cover exactly the allowed UUIDs")
        if not allowed and reservation:
            raise ValueError("CPU jobs need zero reservation")
        if count == 1 and reservation and any(v != reservation for v in by_gpu.values()):
            raise ValueError("Single-GPU scalar/map reservations disagree")
        reservation = max(by_gpu.values(), default=0)  # legacy display only; admission uses the map
        threads = spec.get("cpu_threads", 2)
        if not isinstance(threads, int) or isinstance(threads, bool) or not 1 <= threads <= 72:
            raise ValueError("cpu_threads must be 1..72")
        if spec["kind"] == "train" and threads < count:
            raise ValueError("Training thread budget must provide at least one thread per rank")
        if "threads" in flags and (flags["threads"] is None or not flags["threads"].isdigit() or not 1 <= int(flags["threads"]) <= threads):
            raise ValueError("Workload --threads exceeds registered CPU reservation")
        effective, effective_hash, minimum_cpu = self._effective_workload(spec, script, flags, count)
        if threads < minimum_cpu:
            raise ValueError(f"CPU reservation {threads} is below actual entry requirement {minimum_cpu}")
        spec["effective_workload"] = effective
        spec["effective_workload_sha256"] = effective_hash
        spec.update(cpu_threads=threads, allowed_gpu_uuids=allowed, gpu_count=count,
                    reservation_mib=reservation, reservation_mib_by_gpu=by_gpu, profile=bool(profile))
        if spec.get("resume_from_job"):
            self._validate_resume(spec, flags)
        estimate = spec.get("estimated_remaining_bytes", 0)
        if not isinstance(estimate, int) or estimate < 0:
            raise ValueError("Invalid disk production estimate")
        if allowed:
            if not spec.get("profile_key") or not isinstance(spec["profile_key"], str):
                raise ValueError("GPU job requires a workload profile_key")
            if not profile:
                profile_path = bounded(self.root, self.directory / "profiles" / (str(spec.get("profile_id", "")) + ".json"), exists=True)
                record = read(profile_path)
                if record.get("status") != "conclusive" or record.get("profile_key") != spec["profile_key"]:
                    raise ValueError("No matching measured profile")
                if spec["kind"] == "train" and record.get("training_stage") != spec["training_stage"]:
                    raise ValueError("Training stage differs from measured profile")
                if record.get("gpu_count", 1) != count:
                    raise ValueError("Profile gpu_count/topology mismatch")
                if spec["kind"] == "train":
                    if record.get("devices") != allowed:
                        raise ValueError("Training profile ordered GPU topology mismatch")
                    recommended = record.get("recommended_reservation_mib_by_gpu", {})
                    if set(recommended) != set(allowed):
                        raise ValueError("Training profile lacks per-GPU measured peaks")
                else:
                    recommended = {d: record["recommended_reservation_mib"] for d in allowed}
                if any(by_gpu[d] < recommended[d] for d in allowed):
                    raise ValueError("Reservation is below ceil(1.25 * measured peak)")
                # Workload source/config changes invalidate a profile. Input
                # identities remain per-job; dimensions belong in frozen config.
                if record["code"] != spec["code"] or record["configs"] != spec["configs"]:
                    raise ValueError("Profile source/config identities changed")
                if record.get("effective_workload_sha256") != spec["effective_workload_sha256"]:
                    raise ValueError("Measured profile effective workload differs from the actual entry inputs")
                spec["profile_sha256"] = sha256(profile_path)
            if self.root != PROJECT_ROOT or self.directory != PROJECT_ROOT / "outputs/action4_dispatch":
                raise ValueError("GPU jobs must use the one canonical outputs/action4_dispatch queue")
        return spec

    def _validate_resume(self, spec, flags):
        previous = self.state()["jobs"].get(spec["resume_from_job"])
        if not previous or previous["status"] not in TERMINAL or previous["spec"]["kind"] != "train":
            raise ValueError("Resume source must be a terminal registered training job")
        if spec["gpu_count"] != previous["spec"].get("gpu_count") or spec["allowed_gpu_uuids"] != assigned_devices(previous):
            raise ValueError("Cannot change ordered GPU topology within a training run")
        if spec["result_dirs"] != previous["spec"].get("result_dirs"):
            raise ValueError("Resume must reuse exactly its original owned result directories")
        checkpoint = bounded(self.root, flags["resume"], exists=True)
        models = self.root / "models/action4" / flags["run-id"]
        if not checkpoint.is_relative_to(models) or not all(Path(p).is_dir() for p in spec["result_dirs"]):
            raise ValueError("Resume checkpoint/directories do not belong to this run")
        if any(is_live(previous.get(key)) for key in ("worker", "child")) or any(is_live(p) for p in previous.get("tracked_processes", [])):
            raise ValueError("Previous training processes are still live")
        for job_id, job in self.state()["jobs"].items():
            if job_id == spec["job_id"]:
                continue
            if job["status"] not in TERMINAL and set(spec["result_dirs"]).intersection(job["spec"].get("result_dirs", [])):
                raise ValueError("Another active or queued continuation owns this run")

    def enqueue(self, spec, *, profile=False):
        # Canonical execution requires one explicit active phase. Test/private
        # queues retain their existing CPU control-plane behavior.
        if self.directory == self.root / "outputs/action4_dispatch" and spec.get("kind") in {"render", "train"}:
            from data.action4_execution_authority import spec_references
            spec = json.loads(json.dumps(spec))
            for reference in spec_references(self.root, spec):
                inputs = spec.setdefault("inputs", {})
                if reference["path"] in inputs and inputs[reference["path"]] != reference["sha256"]:
                    raise ValueError("Stale execution authority binding")
                inputs[reference["path"]] = reference["sha256"]
        spec = self._validate(spec, profile=profile)
        with self.lock():
            state = self.state()
            if spec.get("resume_from_job"):
                spec = self._validate(spec, profile=profile)
            if spec["job_id"] in state["jobs"]:
                raise ValueError("job_id already exists; use a new attempt ID")
            out = bounded(self.root, spec["output_dir"])
            proposed = [out] + [Path(p) for p in spec["result_dirs"]]
            existing = [Path(p) for j in state["jobs"].values() for p in [j["spec"]["output_dir"]] +
                        ([] if spec.get("resume_from_job") and j["status"] in TERMINAL and j["spec"].get("result_dirs") == spec["result_dirs"] else j["spec"].get("result_dirs", []))]
            if out.exists() or any(a == b or a in b.parents or b in a.parents for a in proposed for b in existing):
                raise ValueError("Output directory already exists or overlaps another attempt")
            out.mkdir(parents=True)
            state["jobs"][spec["job_id"]] = {"spec": spec, "status": "queued", "enqueued_at": time.time(),
                "token": uuidlib.uuid4().hex, "peak_mib": 0, "memory_samples": 0}
            self.save(state)
        return state["jobs"][spec["job_id"]]

    def _refresh(self, state):
        for job in state["jobs"].values():
            if job["status"] not in ACTIVE:
                continue
            out = bounded(self.root, job["spec"]["output_dir"])
            exit_path, worker_path = out / "dispatch_exit.json", out / "dispatch_worker.json"
            if worker_path.exists():
                worker = read(worker_path)
                if worker.get("token") != job["token"]:
                    raise RuntimeError("Worker receipt identity mismatch")
                job.update(worker=worker["worker"], child=worker.get("child"))
            if exit_path.exists():
                result = read(exit_path)
                if result.get("token") != job["token"]:
                    raise RuntimeError("Exit receipt identity mismatch")
                if any(is_live(identity) for identity in [job.get("worker"), job.get("child")] + job.get("tracked_processes", [])):
                    job["status"] = "running"
                    continue  # a receipt does not release surviving rank reservations
                job.update(status="succeeded" if result["exitcode"] == 0 else "failed", exitcode=result["exitcode"],
                           finished_at=result["finished_at"], failure=result.get("failure"))
                self._finish_profile(job)
                continue
            if is_live(job.get("worker")) or is_live(job.get("child")) or any(is_live(identity) for identity in job.get("tracked_processes", [])):
                job["status"] = "running"
            elif time.time() - job["launched_at"] > 10:
                job.update(status="failed", exitcode=None, finished_at=time.time(),
                           failure="Recorded process identity is absent; no exit receipt; never auto-relaunch")
                self._finish_profile(job)

    def _finish_profile(self, job):
        if not job["spec"]["profile"]:
            return
        duration = job["finished_at"] - job["launched_at"]
        devices = assigned_devices(job)
        peaks = job.get("peak_mib_by_gpu", {d: job.get("peak_mib", 0) for d in devices})
        samples = job.get("samples_by_gpu", {d: job.get("memory_samples", 0) for d in devices})
        count = job["spec"].get("gpu_count", 1 if devices else 0)
        good = (job["status"] == "succeeded" and duration >= 2 and len(devices) == count and count > 0
                and all(samples.get(d, 0) >= 3 and peaks.get(d, 0) > 0 for d in devices))
        recommended = {d: math.ceil(1.25 * peaks.get(d, 0)) for d in devices}
        record = {"status": "conclusive" if good else "inconclusive", "job_id": job["spec"]["job_id"],
                  "profile_key": job["spec"].get("profile_key"), "gpu_count": count, "devices": devices,
                  "training_stage": job["spec"].get("training_stage"),
                  "effective_workload": job["spec"].get("effective_workload"),
                  "effective_workload_sha256": job["spec"].get("effective_workload_sha256"),
                  "peak_mib_by_gpu": peaks, "samples_by_gpu": samples,
                  "peak_mib": max(peaks.values(), default=job.get("peak_mib", 0)),
                  "duration_seconds": duration, "memory_samples": job.get("memory_samples", 0),
                  "recommended_reservation_mib_by_gpu": recommended,
                  "recommended_reservation_mib": max(recommended.values(), default=math.ceil(1.25 * job.get("peak_mib", 0))),
                  "code": job["spec"]["code"], "configs": job["spec"]["configs"], "device": job.get("device"),
                  "sampling_note": "Observed full process-tree peak on each assigned GPU; future unseen inputs may peak higher"}
        atomic(bounded(self.root, self.directory / "profiles" / (job["spec"]["job_id"] + ".json")), record)

    def _telemetry(self, state):
        active = [j for j in state["jobs"].values() if j["status"] in ACTIVE and assigned_devices(j)]
        devices = {d for job in active for d in assigned_devices(job)}
        memory = {device: gpu_process_memory(device) for device in devices}
        for job in active:
            # All ranks and their descendants, not only the torchrun parent.
            # Persist known child identities so an orphan cannot release a lease
            # merely because an intermediate launcher disappeared.
            roots = [job.get("worker"), job.get("child")] + job.get("tracked_processes", [])
            pids = set()
            for identity in roots:
                pids.update(descendants(identity))
            tracked = []
            for pid in sorted(pids):
                identity = process_identity(pid)
                if identity is not None:
                    tracked.append(identity)
            job["tracked_processes"] = tracked
            job.setdefault("current_mib_by_gpu", {})
            job.setdefault("peak_mib_by_gpu", {})
            job.setdefault("samples_by_gpu", {})
            for device in assigned_devices(job):
                current = sum(memory[device].get(pid, 0) for pid in pids)
                job["current_mib_by_gpu"][device] = current
                job["peak_mib_by_gpu"][device] = max(job["peak_mib_by_gpu"].get(device, 0), current)
                job["samples_by_gpu"][device] = job["samples_by_gpu"].get(device, 0) + 1
                if current > reserved_mib(job["spec"], device):
                    job["reservation_overrun"] = True
            job["current_mib"] = max(job["current_mib_by_gpu"].values(), default=0)
            job["peak_mib"] = max(job["peak_mib_by_gpu"].values(), default=0)
            job["memory_samples"] += 1
            job["last_sample_at"] = time.time()

    def _choose_device(self, spec, state):
        active = [j for j in state["jobs"].values() if j["status"] in ACTIVE]
        if sum(j["spec"]["cpu_threads"] for j in active) + spec["cpu_threads"] > 72:
            return False
        free_disk = shutil.disk_usage(self.root).free
        estimates = sum(j["spec"].get("estimated_remaining_bytes", 0) for j in active)
        if free_disk < DISK_FLOOR + math.ceil(1.5 * (estimates + spec.get("estimated_remaining_bytes", 0))):
            return False
        if not spec["allowed_gpu_uuids"]:
            return None
        count = spec.get("gpu_count", 1)
        # Training uses a fixed ordered UUID group. Independent single-GPU work
        # may choose one allowed UUID. No partial group gets booked or launched.
        groups = [spec["allowed_gpu_uuids"]] if spec["kind"] == "train" else [[d] for d in spec["allowed_gpu_uuids"]]
        for group in groups:
            if len(group) != count:
                raise ValueError("Invalid fixed GPU group")
            eligible = True
            for device in group:
                same = [j for j in active if device in assigned_devices(j)]
                if (spec["profile"] and same) or any(j["spec"]["profile"] or j.get("reservation_overrun") for j in same):
                    eligible = False
                    break
                info = gpu_identity(device)
                own = sum(measured_mib(j, device) for j in same)
                reservations = sum(reserved_mib(j["spec"], device) for j in same)
                if not admission(info["free_mib"], own, reservations, reserved_mib(spec, device)):
                    eligible = False
                    break
            if eligible:
                return group[0] if count == 1 else list(group)
        return False

    def _launch(self, state, job, device):
        spec = self._validate(job["spec"], profile=job["spec"]["profile"])
        if spec != job["spec"]:
            raise ValueError("Job specification identity changed")
        devices = [] if device is None else ([device] if isinstance(device, str) else list(device))
        if len(devices) != spec["gpu_count"] or any(d not in spec["allowed_gpu_uuids"] for d in devices):
            raise ValueError("Assigned GPU group differs from reservation")
        if spec["kind"] == "train" and devices != spec["allowed_gpu_uuids"]:
            raise ValueError("Training GPU rank mapping changed")
        job.update(status="launching", devices=devices, device=devices[0] if len(devices) == 1 else None,
                   launched_at=time.time(), current_mib_by_gpu={d: 0 for d in devices},
                   peak_mib_by_gpu={d: 0 for d in devices}, samples_by_gpu={d: 0 for d in devices})
        self.save(state)  # durable intent before spawn; worker verifies token under lock
        env = worker_environment(None)
        for key in tuple(env):
            if key in RANK_KEYS or key in ("NODE_RANK", "GROUP_WORLD_SIZE") or key.startswith("TORCHELASTIC_"):
                env.pop(key, None)
        for assigned in devices:
            gpu_identity(assigned)
        env["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
        env["PYTHONPATH"] = str(self.root / "src")
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONNOUSERSITE"] = "1"
        for key, subdir in {"TMPDIR": ".cache/tmp", "XDG_CACHE_HOME": ".cache/xdg", "XDG_CONFIG_HOME": "runtime/config",
                            "TORCH_HOME": ".cache/torch", "HF_HOME": ".cache/huggingface", "MPLCONFIGDIR": ".cache/matplotlib",
                            "CUDA_CACHE_PATH": ".cache/cuda", "TRITON_CACHE_DIR": ".cache/triton", "TORCH_EXTENSIONS_DIR": ".cache/torch_extensions"}.items():
            env[key] = str(bounded(self.root, subdir))
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            # Trainer sets its checked torch compute pool explicitly. Loader
            # workers must not multiply an inherited BLAS pool per worker.
            env[key] = "1" if spec["kind"] == "train" else str(spec["cpu_threads"])
        out = bounded(self.root, spec["output_dir"])
        with open(out / "dispatch_wrapper.log", "ab", buffering=0) as log:
            proc = subprocess.Popen([str(self.root / ".venv/bin/python"), str(self.root / "scripts/action4_dispatch.py"),
                "--root", str(self.root), "--queue", str(self.directory), "_worker", "--job-id", spec["job_id"], "--token", job["token"]],
                cwd=self.root, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        job["worker"] = process_identity(proc.pid)
        self.save(state)
        return proc

    def status(self):
        with self.lock():
            state = self.state()
            self._refresh(state)
            self.save(state)
            return state

    def run(self, *, poll_seconds=1., max_seconds=None):
        if poll_seconds <= 0 or (max_seconds is not None and max_seconds <= 0):
            raise ValueError("Polling and bounded observation must be positive")
        started, children = time.monotonic(), []
        with self.lock("scheduler", blocking=False):
            while True:
                with self.lock():
                    state = self.state()
                    self._refresh(state)
                    self._telemetry(state)
                    for job in state["jobs"].values():
                        if job["status"] != "queued":
                            continue
                        try:
                            device = self._choose_device(job["spec"], state)
                            if device is not False:
                                children.append(self._launch(state, job, device))
                        except (ValueError, FileNotFoundError) as exc:
                            job.update(status="failed", failure=str(exc), exitcode=None, finished_at=time.time())
                    self.save(state)
                    pending = any(j["status"] == "queued" or j["status"] in ACTIVE for j in state["jobs"].values())
                children = [proc for proc in children if proc.poll() is None]
                if not pending or (max_seconds is not None and time.monotonic() - started >= max_seconds):
                    return state  # observation timeout never kills/restarts live work
                time.sleep(poll_seconds)

    def worker(self, job_id, token):
        out, proc = None, None
        try:
            with self.lock():
                state = self.state()
                job = state["jobs"][job_id]
                if job["token"] != token or job["status"] not in ACTIVE:
                    raise RuntimeError("Launch no longer belongs to this active attempt")
                identity = process_identity(os.getpid())
                if job.get("worker") and job["worker"] != identity:
                    raise RuntimeError("Another worker identity already owns this attempt")
                job["worker"] = identity
                self.save(state)
                out = bounded(self.root, job["spec"]["output_dir"])
                if os.environ.get("CUDA_VISIBLE_DEVICES", "") != ",".join(assigned_devices(job)):
                    raise RuntimeError("Worker CUDA mask differs from its atomic device-group reservation")
                for assigned in assigned_devices(job):
                    gpu_identity(assigned)
                spec = self._validate(job["spec"], profile=job["spec"]["profile"])
                if spec != job["spec"]:
                    raise ValueError("Worker specification changed after reservation")
                out = bounded(self.root, spec["output_dir"])
                atomic(out / "dispatch_worker.json", {"token": token, "worker": identity, "child": None})
                with open(out / "stdout.log", "ab", buffering=0) as log:
                    actual_argv = workload_argv(self.root, spec, token)
                    atomic(out / "dispatch_launch.json", {"token": token, "argv": actual_argv,
                        "devices": assigned_devices(job), "gpu_count": spec["gpu_count"],
                        "reservation_mib_by_gpu": spec["reservation_mib_by_gpu"]})
                    proc = subprocess.Popen(actual_argv, cwd=spec["cwd"], stdin=subprocess.DEVNULL,
                                            stdout=log, stderr=subprocess.STDOUT)
                atomic(out / "dispatch_worker.json", {"token": token, "worker": identity, "child": process_identity(proc.pid)})
            code = proc.wait()
            atomic(out / "dispatch_exit.json", {"token": token, "exitcode": code, "finished_at": time.time()})
            return code
        except Exception as exc:
            if out is not None and proc is None:
                atomic(out / "dispatch_exit.json", {"token": token, "exitcode": 125,
                    "finished_at": time.time(), "failure": type(exc).__name__ + ": " + str(exc)})
            raise
