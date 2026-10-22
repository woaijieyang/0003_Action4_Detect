"""D0/A0/B0/C1 training with image-weighted DDP, epoch resume and replay.

Real accuracy is established by measured eval/video evidence, never by this
module merely finishing. Diagnostic runs propagate their scope to checkpoints.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import random
import re
import time
from pathlib import Path

from data.action4_paths import checked_root
from .action4_stage_policy import configure_stage_trainability, frozen_state_digest, prepare_checkpoint_topology
from .action4_initialization import validate_weights_initialization
from data.action4_split_protocol import binding_identity, check_checkpoint_binding

import numpy as np

from data import DataAdapter
from data.action4_dataset import is_ignored
from .action4_gate import build_gate_binding, make_d0_gate, validate_d0_gate
from .action4_timing import timing_enabled, synchronized_update_boundary, measured_update_window
from .action4_checkpoint import (INTEGRITY_SCHEMA, validate_resume_checkpoint, verify_retained,
                                 bind_previous_top, recover_epoch20, synchronize_checkpoint_recovery,
                                 synchronized_rank0_action, cleanup_owned_process_group)
from .action4_video_queue import (VideoQueue, execution_mode, video_due, epoch_pause_required,
                                  require_dispatch_owner, validate_queue_config)
from .action4_runtime import (atomic_json, canonical_hash, capture_rng, environment_identity, input_path,
                              jsonable, load_checkpoint, output_path, restore_rng, run_directories,
                              save_checkpoint_atomic, link_checkpoint_atomic, seed_all, select_device, sha256_file, source_identity,
                              write_json_new, validate_run_id)


def amp_initial_scale(config):
    value = config.get("runtime", {}).get("amp_init_scale", 65536.)
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError("runtime.amp_init_scale must be finite and positive")
    return float(value)


def learning_rate_factor(step: int, total_updates: int, warmup_updates: int, final_ratio: float = .01) -> float:
    if total_updates < 1 or not 0 <= warmup_updates < total_updates or not 0 < final_ratio <= 1:
        raise ValueError("Scheduler requires 0 <= warmup < total and 0 < final_ratio <= 1")
    if warmup_updates and step < warmup_updates:
        return .1 + .9 * step / warmup_updates
    progress = min(max((step - warmup_updates) / (total_updates - warmup_updates), 0), 1)
    return final_ratio + (1 - final_ratio) * (1 + math.cos(math.pi * progress)) / 2


def build_optimizer(model, stage_config: dict):
    import torch
    groups = {(backbone, decay): [] for backbone in (True, False) for decay in (True, False)}
    for name, parameter in model.named_parameters():
        groups[(name.startswith("backbone."), parameter.ndim > 1 and not name.endswith(".bias"))].append(parameter)
    parameter_groups = []
    for (backbone, decay), parameters in groups.items():
        if parameters:
            parameter_groups.append({"params": parameters,
                                     "lr": stage_config["lr_backbone"] if backbone else stage_config["lr_new"],
                                     "weight_decay": stage_config.get("weight_decay", 1e-4) if decay else 0.,
                                     "group_name": ("backbone" if backbone else "new") + ("_decay" if decay else "_no_decay")})
    return torch.optim.AdamW(parameter_groups)


def epoch_indices(base_count: int, *, epoch: int, seed: int, s1_count: int = 0,
                  replay_base: int = 6000, replay_new: int = 2000) -> list[int]:
    if base_count < 1 or epoch < 0:
        raise ValueError("Nonempty training data and nonnegative epoch required")
    if not s1_count:
        indices = list(range(base_count))
    else:
        if base_count < replay_base or s1_count < replay_new or min(replay_base, replay_new) < 1:
            raise ValueError("C1 needs enough distinct S0 and S1 records for the declared replay quota")
        base_order, new_order = list(range(base_count)), list(range(s1_count))
        random.Random(seed).shuffle(base_order)
        random.Random(seed + 1).shuffle(new_order)
        indices = [base_order[(epoch * replay_base + i) % base_count] for i in range(replay_base)]
        indices += [base_count + new_order[(epoch * replay_new + i) % s1_count] for i in range(replay_new)]
    random.Random(seed + 1000003 * epoch).shuffle(indices)
    return indices


class EpochWindowSampler:
    """Global effective-batch windows with zero-weight rank padding only.

    The final small window is weighted by its actual global count. No real
    sample is duplicated to fill a distributed batch, or silently dropped.
    """
    def __init__(self, indices: list[int], global_batch: int, micro_batch: int, rank: int, world_size: int):
        if not indices or min(global_batch, micro_batch, world_size) < 1 or not 0 <= rank < world_size:
            raise ValueError("Invalid distributed batch plan")
        self.indices, self.global_batch, self.micro_batch = indices, global_batch, micro_batch
        self.rank, self.world_size = rank, world_size
        self.accumulation = math.ceil(global_batch / (micro_batch * world_size))
        self.window_counts = [len(indices[start:start + global_batch]) for start in range(0, len(indices), global_batch)]

    def __iter__(self):
        for start in range(0, len(self.indices), self.global_batch):
            window = self.indices[start:start + self.global_batch]
            for micro in range(self.accumulation):
                begin = (micro * self.world_size + self.rank) * self.micro_batch
                batch = window[begin:begin + self.micro_batch]
                yield [(index, True) for index in batch] if batch else [(self.indices[0], False)]

    def __len__(self):
        return len(self.window_counts) * self.accumulation


class _TrainingView:
    def __init__(self, base, s1=None, *, roi_only: bool = False, d0_limit: int | None = None):
        self.base, self.s1, self.roi_only = base, s1, roi_only
        self.units = []
        if roi_only:
            for si, record in enumerate(base.records):
                for gi, instance in enumerate(record["instances"]):
                    if not is_ignored(instance):
                        self.units.append((si, gi))
            if d0_limit is not None:
                if len(self.units) < d0_limit:
                    raise ValueError(f"D0 requires {d0_limit} distinct non-ignored ROIs")
                self.units = self.units[:d0_limit]
        self.base_count = len(self.units) if roi_only else len(base)
        self.s1_count = len(s1) if s1 is not None else 0

    def __len__(self):
        return self.base_count + self.s1_count

    def __getitem__(self, key):
        index, real = key
        if self.roi_only:
            si, gi = self.units[index]
            result = dict(self.base[si])
            result["roi_instance_index"] = gi
        else:
            result = dict(self.base[index] if index < self.base_count else self.s1[index - self.base_count])
        result["training_real"] = real
        return result


def _worker_seed(worker_id):
    import torch
    value = torch.initial_seed() % (2**32)
    np.random.seed(value)
    random.seed(value)


def _jitter_box(box, rng, *, enabled=True, center_fraction=.05, scale_range=(.9, 1.1)):
    box = np.asarray(box, dtype=float)
    if not enabled:
        return box.tolist()
    if (type(center_fraction) not in (int, float) or not math.isfinite(center_fraction) or not 0 <= center_fraction <= .5
            or not isinstance(scale_range, (list, tuple)) or len(scale_range) != 2
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in scale_range)
            or not 0 < scale_range[0] <= scale_range[1]):
        raise ValueError("Invalid declared ROI jitter parameters")
    size = box[2:] - box[:2]
    center = (box[:2] + box[2:]) / 2 + rng.uniform(-center_fraction, center_fraction, size=2) * size
    size = size * rng.uniform(*scale_range)
    return np.r_[center - size / 2, center + size / 2].tolist()


def _roi_builder(images, records, config, device, rng, *, predictions=None, selected_instances=None,
                 jitter=True, predicted_probability=.5, jitter_parameters=None):
    import torch
    from data.action4_dataset import prepare_rois, build_corner_targets
    from eval.action4_metrics import bbox_iou, maximum_matching
    if jitter_parameters is not None and (not isinstance(jitter_parameters, dict) or set(jitter_parameters)-{'center_fraction','scale_range'}):
        raise ValueError("Explicit ROI jitter parameter object required")
    image_ids, instance_ids, boxes = [], [], []
    attempted = used = 0
    for bi, record in enumerate(records):
        valid = [gi for gi, gt in enumerate(record["instances"])
                 if not is_ignored(gt)
                 and (selected_instances is None or gi == selected_instances[bi])]
        matches = {}
        if predictions is not None:
            edges = [[j for j, gi in enumerate(valid)
                      if bbox_iou(p["bbox_full_xyxy"], record["instances"][gi]["bbox_full_xyxy"]) >= .5]
                     for p in predictions[bi]]
            matches = {valid[g]: p for p, g in maximum_matching(edges, len(valid))}
        for gi in valid:
            box = _jitter_box(record["instances"][gi]["bbox_full_xyxy"], rng, enabled=jitter, **(jitter_parameters or {}))
            if predictions is not None and rng.random() < predicted_probability:
                attempted += 1
                if gi in matches:
                    box = predictions[bi][matches[gi]]["bbox_full_xyxy"]
                    used += 1
            image_ids.append(bi)
            instance_ids.append(gi)
            boxes.append(box)
    roi_size = config["preprocess"]["roi_size"]
    rois, affines = prepare_rois(images, boxes, image_ids, roi_size=roi_size,
                                expansion=config["preprocess"]["roi_expansion"], device=device)
    maps, valid = build_corner_targets(records, image_ids, instance_ids, affines, roi_size=roi_size,
                                       heatmap_size=(roi_size // 4, roi_size // 4), device=device)
    return rois, {"targets": maps, "valid": valid,
                  "roi_batch_index": torch.as_tensor(image_ids, dtype=torch.long, device=device),
                  "batch_size": len(images), "predicted_roi_attempted": attempted, "predicted_roi_used": used}


def _all_ranks_finite(value, device, distributed):
    import torch
    indicator = torch.tensor(int(bool(value)), device=device, dtype=torch.int32)
    if distributed:
        torch.distributed.all_reduce(indicator, op=torch.distributed.ReduceOp.MIN)
    return bool(indicator.item())


def _validate_split_isolation(base, eval_data, test_records, s1=None, *, config=None, root=None):
    collections = {"train": base.records + (s1.records if s1 is not None else []),
                   "eval": eval_data.records, "test": test_records}
    # Legacy configs retain the exact strict background-disjoint checks below.
    if config is not None and config.get("data", {}).get("split_protocol_ref") is not None:
        if s1 is not None:
            raise ValueError("Shared-template pilot release does not authorize an S1/C1 extension")
        binding = binding_identity(root, config, records_by_split=collections)
        if binding is None:
            raise ValueError("Explicit shared-template binding required")
        return binding
    if config is not None and any(config.get("data", {}).get(key) is not None
                                  for key in ("dataset_release_ref", "dataset_commit_ref")):
        raise ValueError("Dataset release requires its explicit split protocol")
    for field in ("sample_id", "group_id", "background_source_id"):
        owners = {}
        for split, records in collections.items():
            for record in records:
                value = record[field]
                if value in owners and owners[value] != split:
                    raise ValueError(f"Split leakage via {field}: {value}")
                owners[value] = split
    ids = [r["sample_id"] for r in collections["train"]]
    if len(set(ids)) != len(ids):
        raise ValueError("S0 and S1 train sample IDs overlap")


def _run_epoch_evaluation(model, eval_data, config, device, stage):
    from eval.action4_runner import evaluate_gt_rois, evaluate_records
    batch = config.get("eval", {}).get("batch_size", 1)
    roi = evaluate_gt_rois(model, eval_data, config, device, batch_size=batch)
    report, predictions = (None, None) if stage in ("D0", "A0") else evaluate_records(model, eval_data, config, device, batch_size=batch)
    return roi, report, predictions


def d0_geometry_checks(model, base, units, config, device) -> dict:
    """Executable geometry/resampling and actual-network empty-image checks."""
    import torch
    from data.action4_dataset import prepare_full_images, prepare_rois, build_detection_targets
    from model.action4_losses import detection_losses, corner_heatmap_loss
    round_trip_errors = []
    for sample_index, instance_index in units:
        item = base[sample_index]
        gt = item["record"]["instances"][instance_index]
        _, matrices = prepare_rois([item["image"]], [gt["bbox_full_xyxy"]], [0],
                                    roi_size=config["preprocess"]["roi_size"],
                                    expansion=config["preprocess"]["roi_expansion"], device=device)
        points = np.asarray(gt["corners_px"], dtype=float)
        mapped = DataAdapter.transform_points(points, matrices[0])
        recovered = DataAdapter.transform_points(mapped, matrices[0], inverse=True)
        round_trip_errors.extend(np.linalg.norm(points - recovered, axis=1).tolist())
    yy, xx = torch.meshgrid(torch.arange(80), torch.arange(96), indexing="ij")
    ramp = torch.stack([xx, yy, torch.zeros_like(xx)]).to(torch.uint8)
    sampled, matrices = prepare_rois([ramp], [[16., 12., 72., 68.]], [0], roi_size=32, expansion=1.0, device=device)
    channel_coordinates = (sampled[0, :2].cpu() * torch.tensor([.229, .224])[:, None, None]
                           + torch.tensor([.485, .456])[:, None, None]) * 255
    target_y, target_x = np.mgrid[:32, :32]
    expected = DataAdapter.transform_points(np.stack([target_x, target_y], axis=-1).reshape(-1, 2), matrices[0], inverse=True)
    sampled_points = channel_coordinates.permute(1, 2, 0).numpy().reshape(-1, 2)
    ramp_error = float(np.linalg.norm(sampled_points - expected, axis=1).max())
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        empty_image = torch.zeros((3, 64, 64), dtype=torch.uint8)
        full, matrices = prepare_full_images([empty_image], size=tuple(config["preprocess"]["full_size"]), device=device)
        outputs = model(full)
        targets = build_detection_targets([{"width": 64, "height": 64, "instances": []}], matrices,
                                           full_size=tuple(config["preprocess"]["full_size"]), device=device)
        losses = detection_losses(outputs, targets)
        logits = outputs["corner_logits"]
        losses["heatmap"] = corner_heatmap_loss(logits, torch.zeros_like(logits),
                                                torch.zeros((0, 8), dtype=torch.bool, device=device),
                                                torch.zeros((0,), dtype=torch.long, device=device), 1)
        empty_finite = all(torch.isfinite(value).item() for value in losses.values())
    model.train(was_training)
    maximum = max(round_trip_errors) if round_trip_errors else None
    return {"affine_round_trip_max_original_px": maximum, "roi_coordinate_ramp_max_native_px": ramp_error,
            "actual_network_empty_image_losses_finite": empty_finite,
            "gate_pass": maximum is not None and maximum <= .001 and ramp_error <= .001 and empty_finite,
            "scope": "native-coordinate ramp and actual selected ROI transforms; not real-video accuracy"}


@cleanup_owned_process_group
def train(root: str | Path, config: dict, *, stage: str, run_id: str,
          init_weights: str | Path | None = None, resume: str | Path | None = None,
          d0_gate: str | Path | None = None, allow_diagnostic: bool = False,
          component_smoke: bool = False, max_updates: int | None = None) -> dict:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if config.get("runtime", {}).get("nccl_p2p_disable", False):
        os.environ["NCCL_P2P_DISABLE"] = "1"
    import torch
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader
    from data.action4_dataset import Action4Dataset, prepare_full_images, build_detection_targets
    from model.action4_network import Action4Net
    from model.action4_losses import detection_losses, corner_heatmap_loss
    from eval.action4_runner import decode_full_boxes, write_predictions
    from eval.action4_metrics import selection_key
    if stage not in ("D0", "A0", "B0", "C1"):
        raise ValueError("Supported stages are D0, A0, B0 and C1")
    validate_run_id(run_id)
    if resume is not None and init_weights is not None:
        raise ValueError("resume and init_weights are mutually exclusive")
    if max_updates is not None and not component_smoke:
        raise ValueError("Shortened update budgets require --component-smoke; they do not fulfill the training stage")
    root = checked_root(root)
    initial_scale = amp_initial_scale(config)
    synchronized_timing = timing_enabled(config)
    queued = (execution_mode(config) == "queued" and stage in ("B0", "C1")
              and (not component_smoke or config.get("video", {}).get("run_during_component", False)))
    video_queue = None
    # Reject unsafe resume before allocating a model/GPU or taking optimizer
    # steps. A crashed enqueue is replayed from the exact saved top checkpoint.
    if queued:
        validate_queue_config(config)
        require_dispatch_owner(root, run_id)
        if resume is not None:
            resume_path = input_path(root, resume)
            previous = load_checkpoint(resume_path)
            validate_resume_checkpoint(previous)
            latest = input_path(root, Path("models/action4") / run_id / "last.pt")
            previous_identity = previous.get("identity", {})
            if (sha256_file(resume_path) != sha256_file(latest)
                    or previous.get("run_id") != run_id or previous.get("stage") != stage
                    or previous.get("epoch_boundary") is not True
                    or previous_identity.get("world_size") != int(os.environ.get("WORLD_SIZE", "1"))
                    or previous_identity.get("config_sha256") != canonical_hash(config)
                    or previous_identity.get("code_sha256") != source_identity(root)["sha256"]):
                raise ValueError("Queued resume requires the latest exact run/config/source/topology checkpoint")
            for split, digest in previous_identity.get("data_sha256", {}).items():
                if sha256_file(input_path(root, config["data"][f"{split}_manifest"])) != digest:
                    raise ValueError("Queued resume data identity changed")
    rank, world = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = select_device(config["runtime"].get("device", "auto"), local_rank if distributed else None)
    if config["runtime"].get("require_cuda", True) and device.type != "cuda" and not component_smoke:
        raise RuntimeError("Formal training is configured to require CUDA; this environment can run explicit CPU component checks")
    if distributed:
        torch.distributed.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    if queued and resume is not None:
        validate_resume_checkpoint(previous, device=device)
        synchronize_checkpoint_recovery(previous, latest, rank=rank, distributed=distributed)
        def replay_queued_resume():
            nonlocal video_queue
            video_queue = VideoQueue(root, run_id, config)
            done_epoch = previous["next_epoch"]
            if video_due(stage, done_epoch, config["stages"][stage]["epochs"], config,
                         component=component_smoke, completed=previous.get("epoch_boundary") is True):
                if not previous.get("top_checkpoints"):
                    raise ValueError("Resume lacks the original best checkpoint identity for video replay")
                best_name = previous["top_checkpoints"][0]["file"]
                if not re.fullmatch(r"epoch_\d{3}_step_\d{8}\.pt", best_name):
                    raise ValueError("Invalid saved best checkpoint alias")
                best_alias = output_path(root, Path("models/action4") / run_id / best_name)
                if not best_alias.exists():
                    latest_name = f"epoch_{done_epoch:03d}_step_{previous['global_step']:08d}.pt"
                    if best_name == latest_name and previous["top_checkpoints"][0]["epoch"] == done_epoch:
                        link_checkpoint_atomic(latest, best_alias)
                    # An older missing best is not reconstructed from newer weights.
                video_queue.ensure_event(stage, done_epoch, best_alias,
                                         component=component_smoke or allow_diagnostic)
            video_queue.require_resume(stage, done_epoch)
        synchronized_rank0_action(replay_queued_resume, rank=rank, distributed=distributed,
                                  label="Queued resume preparation")
    seed = config["runtime"].get("seed", 20261018)
    seed_all(seed, deterministic=config["runtime"].get("deterministic", True))
    torch.set_num_threads(config["runtime"].get("cpu_threads", 2))
    stage_config = config["stages"][stage]
    if type(stage_config.get("weights_only_roi_migration", False)) is not bool:
        raise ValueError("weights_only_roi_migration must be boolean")
    if type(stage_config.get("roi_jitter", False)) is not bool:
        raise ValueError("roi_jitter must be boolean")
    if stage == "D0" and stage_config.get("roi_jitter", False):
        raise ValueError("D0 fixed ROIs cannot use jitter")
    if stage == "A0" and stage_config.get("weights_only_roi_migration", False) and init_weights is None and resume is None:
        raise ValueError("Explicit A0 warm start requires init-weights or exact resume")
    for field in ("epochs", "global_batch", "micro_batch"):
        if type(stage_config.get(field)) is not int or stage_config[field] < 1:
            raise ValueError(f"{field} must be a positive integer")
    for field in ("lr_backbone", "lr_new", "clip_grad_norm"):
        value = stage_config.get(field, 1. if field == "clip_grad_norm" else None)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{field} must be finite and positive")
    video_config = config.get("video", {})
    if type(video_config.get("every_epochs", 5)) is not int or video_config.get("every_epochs", 5) < 1:
        raise ValueError("video.every_epochs must be a positive integer")
    if stage in ("B0", "C1") and not component_smoke:
        if not video_config.get("frame_manifest"):
            raise ValueError("Formal B0/C1 requires the complete development-video frame manifest")
        video_manifest = json.loads(input_path(root, video_config["frame_manifest"]).read_text())
        if not video_manifest.get("complete") or [f["frame_id"] for f in video_manifest["frames"]] != list(range(video_manifest["source"]["frame_count"])):
            raise ValueError("Prepare a complete ordered video cache before long B0/C1 training")
    roi_only = stage in ("D0", "A0")
    paths = {name: input_path(root, config["data"][f"{name}_manifest"]) for name in ("train", "eval", "test")}
    validation_workers = config["runtime"].get("validation_workers", 1)
    base = Action4Dataset(root, paths["train"], split="train", allow_diagnostic=allow_diagnostic,
                          validation_workers=validation_workers, decoded_cache_size=32 if stage == "D0" else 0)
    eval_data = Action4Dataset(root, paths["eval"], split="eval", allow_diagnostic=allow_diagnostic,
                               validation_workers=validation_workers)
    test_records = DataAdapter(root).read_samples(paths["test"], check_images=False)
    if any(record.get("split") != "test" for record in test_records):
        raise ValueError("Test manifest must declare the test split")
    s1 = None
    if stage == "C1":
        paths["s1"] = input_path(root, config["data"]["s1_manifest"])
        s1 = Action4Dataset(root, paths["s1"], split="train", allow_diagnostic=allow_diagnostic)
    split_protocol_binding = _validate_split_isolation(base, eval_data, test_records, s1, config=config, root=root)
    gate_binding = None
    if stage == "D0" or not component_smoke:
        gate_binding = build_gate_binding(root, config, paths["train"], base.records,
                                          component=component_smoke or allow_diagnostic)
    if stage != "D0" and not component_smoke:
        if d0_gate is None:
            raise ValueError("A0/B0/C1 require a saved successful D0 gate")
        gate = json.loads(input_path(root, d0_gate).read_text())
        reuse_ref = config.get("d0_schedule_reuse_ref")
        if reuse_ref is None:
            validate_d0_gate(gate, gate_binding)
        else:
            from .action4_gate import validate_reused_d0_gate
            reuse_path = input_path(root, reuse_ref["path"])
            if sha256_file(reuse_path) != reuse_ref["sha256"]:
                raise ValueError("D0 reuse receipt bytes changed")
            validate_reused_d0_gate(gate, gate_binding, json.loads(reuse_path.read_text()))
    view = _TrainingView(base, s1, roi_only=roi_only, d0_limit=stage_config.get("roi_count", 32) if stage == "D0" else None)
    planned_indices = epoch_indices(view.base_count, epoch=0, seed=seed, s1_count=view.s1_count,
                                    replay_base=stage_config.get("replay_s0", 6000), replay_new=stage_config.get("replay_s1", 2000))
    global_batch, micro_batch = stage_config["global_batch"], stage_config.get("micro_batch", 1)
    updates_per_epoch = math.ceil(len(planned_indices) / global_batch)
    planned_total = updates_per_epoch * stage_config["epochs"]
    if stage == "D0" and not component_smoke and not allow_diagnostic and planned_total > 500:
        raise ValueError("Formal D0 may not exceed the fixed 500-update budget")
    if max_updates is not None and (max_updates < 1 or max_updates > planned_total):
        raise ValueError("Component update limit must be inside the planned budget")
    # A shortened component execution retains the full stage LR schedule; it
    # never pretends that one warmup update is a converged training experiment.
    warmup = stage_config.get("warmup_updates", 500)
    if warmup >= planned_total:
        raise ValueError("Warmup must be shorter than planned training; use an explicit component configuration")
    model = Action4Net(weights=None, **config["network"])
    initialization = None
    resume_state = load_checkpoint(input_path(root, resume)) if resume is not None else None
    if resume_state is not None:
        latest = output_path(root, Path("models/action4") / run_id / "last.pt")
        if not latest.is_file() or sha256_file(input_path(root, resume)) != sha256_file(latest):
            raise ValueError("Resume only the latest committed last.pt; older checkpoints cannot rewrite a later run")
    if resume_state is not None:
        check_checkpoint_binding(resume_state, split_protocol_binding)
        validate_resume_checkpoint(resume_state, device=device)
        prepare_checkpoint_topology(model, resume_state, expected_policy=stage_config.get("shared_feature_policy", "joint"))
        model.load_state_dict(resume_state["model"], strict=True)
        initialization = resume_state["initialization"]
    elif init_weights is not None:
        if stage not in ("B0", "C1") and not (stage == "A0" and stage_config.get("weights_only_roi_migration") is True):
            raise ValueError("D0/A0 start from the declared ImageNet backbone, not debug/task weights")
        path = input_path(root, init_weights)
        init = load_checkpoint(path)
        initialization_contract = validate_weights_initialization(stage, config, init)
        if not component_smoke and init.get("claim_scope") != "accepted_data_model_training":
            raise ValueError("Component checkpoints cannot initialize a claimed formal training stage")
        check_checkpoint_binding(init, split_protocol_binding)
        prepare_checkpoint_topology(model, init)
        model.load_state_dict(init["model"], strict=True)
        initialization = {**initialization_contract, "checkpoint": str(path), "sha256": sha256_file(path)}
        if stage == "B0" and stage_config.get("roi_init_weights"):
            roi_path = input_path(root, stage_config["roi_init_weights"])
            roi_parent = load_checkpoint(roi_path)
            if roi_parent.get("stage") != "A0" or roi_parent.get("claim_scope") != "accepted_data_model_training":
                raise ValueError("Explicit formal A0 ROI parent required")
            if roi_parent["config"]["network"] != config["network"] or roi_parent["config"]["preprocess"] != config["preprocess"]:
                raise ValueError("ROI parent contract mismatch")
            check_checkpoint_binding(roi_parent, split_protocol_binding)
            model.roi_branch.load_state_dict({k:v for k,v in roi_parent["model"].items() if k.split('.')[0] in ('backbone','decoder','corner_head')},strict=True)
            initialization["roi_parent"] = {"path": str(roi_path), "sha256": sha256_file(roi_path)}

    else:
        if stage in ("B0", "C1"):
            raise ValueError("B0 needs A0 weights; C1 needs B0/C1 weights, unless resuming the same run")
        pretrained = config.get("initialization", {}).get("backbone_weights")
        if pretrained is not None:
            path = input_path(root, pretrained)
            initialization = {"mode": "imagenet_backbone", **model.load_backbone_weights(path, sha256=config["initialization"].get("sha256"))}
        elif component_smoke:
            initialization = {"mode": "random_component_only", "reason": "explicit component execution without ImageNet artifact"}
        else:
            raise ValueError("Declare the local verified ImageNet backbone weight path")
    model = model.to(device)
    trainability_policy = configure_stage_trainability(model, stage, stage_config)
    initial_frozen_digest = frozen_state_digest(model, trainability_policy)
    geometry_checks = d0_geometry_checks(model, base, view.units, config, device) if stage == "D0" else None
    # Register the entire backbone with DDP from the start. During A0 warmup,
    # gradients are explicitly discarded before AdamW (so no weight decay or
    # moments update), avoiding an unsafe requires_grad change after wrapping.
    optimizer = build_optimizer(model, stage_config)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: learning_rate_factor(step, planned_total, warmup))
    amp = bool(config["runtime"].get("amp", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=initial_scale)
    identity = {"trainability_policy": trainability_policy, "initial_frozen_digest": initial_frozen_digest, "split_protocol_binding": split_protocol_binding, "config_sha256": canonical_hash(config), "data_sha256": {k: sha256_file(p) for k, p in paths.items()},
                "code_sha256": source_identity(root)["sha256"], "stage": stage, "world_size": world,
                "global_batch": global_batch, "micro_batch": micro_batch, "planned_updates": planned_total,
                "allow_diagnostic": allow_diagnostic, "component_smoke": component_smoke,
                "d0_binding_sha256": canonical_hash(gate_binding) if gate_binding is not None else None}
    next_epoch = global_step = 0
    top = []
    mandatory_snapshots = {}
    if resume_state is not None:
        if resume_state["identity"] != identity or resume_state["run_id"] != run_id or not resume_state.get("epoch_boundary"):
            raise ValueError("Exact resume requires identical run/data/config/code/topology and a complete epoch boundary")
        # PyTorch requires the scheduler to be initialized and restored before
        # optimizer state so it cannot overwrite the saved learning rates.
        validate_resume_checkpoint(resume_state, optimizer=optimizer, scheduler=scheduler, device=device)
        scheduler.load_state_dict(resume_state["scheduler"])
        optimizer.load_state_dict(resume_state["optimizer"])
        scaler.load_state_dict(resume_state["scaler"])
        next_epoch, global_step = resume_state["next_epoch"], resume_state["global_step"]
        top = resume_state.get("top_checkpoints", [])
        synchronize_checkpoint_recovery(resume_state, latest, rank=rank, distributed=distributed)
        mandatory_snapshots = resume_state["mandatory_snapshots"]
        if next_epoch >= stage_config["epochs"] or max_updates is not None and global_step >= max_updates:
            raise ValueError("No updates remain in this run/budget; use a new continued-training run when appropriate")
    if rank == 0:
        model_dir, output_dir = run_directories(root, run_id, resume=resume is not None)
        if resume is None:
            write_json_new(output_dir / "run_manifest.json", {"status": "running", "identity": identity, "config": config,
                            "source": source_identity(root), "environment": environment_identity(torch), "initialization": initialization,
                            "claim_scope": "component_diagnostic" if component_smoke or allow_diagnostic or stage == "D0" else "accepted_data_model_training"})
        else:
            write_json_new(output_dir / f"resume_epoch_{next_epoch:03d}_step_{global_step:08d}_{time.time_ns()}.json",
                           {"checkpoint": str(input_path(root, resume)), "identity": identity, "environment": environment_identity(torch)})
    if distributed:
        torch.distributed.barrier()
    model_dir, output_dir = output_path(root, Path("models/action4") / run_id), output_path(root, Path("outputs/action4") / run_id)
    # A resumed attempt retains earlier logs/partial diagnostics and writes a
    # distinct attempt directory; repeating an unfinished epoch cannot mix logs.
    attempt_name = f"resume_epoch_{next_epoch:03d}_step_{global_step:08d}_{time.time_ns()}" if rank == 0 and resume is not None else None
    if distributed:
        value = [attempt_name]
        torch.distributed.broadcast_object_list(value, src=0)
        attempt_name = value[0]
    work_output = output_dir / attempt_name if attempt_name is not None else output_dir
    if rank == 0 and attempt_name is not None:
        work_output.mkdir()
    if queued and video_queue is None and rank == 0:
        video_queue = VideoQueue(root, run_id, config)
    train_model = DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None,
                                         find_unused_parameters=True, broadcast_buffers=False) if distributed else model
    if resume_state is not None:
        restore_rng(resume_state["rng_by_rank"][rank])
    else:
        seed_all(seed + rank, deterministic=config["runtime"].get("deterministic", True))
    roi_rng = np.random.default_rng(seed + rank)
    if resume_state is not None:
        roi_rng.bit_generator.state = resume_state["roi_rng_by_rank"][rank]
    claim = "component_diagnostic" if component_smoke or allow_diagnostic or stage == "D0" else "accepted_data_model_training"
    finite_backbone_grad_observed = False
    backbone_changed = False
    first_backbone = next(model.backbone.parameters())
    starting_backbone = first_backbone.detach().cpu().clone()
    durations = []
    nonfinite_updates = resume_state["resume_integrity"]["nonfinite_updates"] if resume_state is not None else 0
    synchronized_boundaries = []
    last_report = None
    d0_early_pass = False
    paused_for_review = False
    queued_state = None
    eval_interval = stage_config.get("eval_every_epochs", 1) if stage == "A0" else 1
    if type(eval_interval) is not int or eval_interval < 1:
        raise ValueError("A0 eval_every_epochs must be a positive integer")
    review_interval = stage_config.get("review_interval_epochs", 5) if stage == "C1" else None
    if review_interval is not None and (type(review_interval) is not int or review_interval < 1):
        raise ValueError("C1 review_interval_epochs must be a positive integer")

    def checkpoint_state(epoch, boundary=True):
        rng, roi_state = capture_rng(), roi_rng.bit_generator.state
        rng_states, roi_states = [None] * world, [None] * world
        if distributed:
            torch.distributed.all_gather_object(rng_states, rng)
            torch.distributed.all_gather_object(roi_states, roi_state)
        else:
            rng_states, roi_states = [rng], [roi_state]
        return {"schema_version": "action4.checkpoint.v1", "stage": stage, "run_id": run_id,
                "claim_scope": claim, "config": config, "identity": identity, "initialization": initialization,
                "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(), "next_epoch": epoch, "global_step": global_step,
                "epoch_boundary": boundary, "rng_by_rank": rng_states, "roi_rng_by_rank": roi_states,
                "resume_integrity": {"schema_version": INTEGRITY_SCHEMA, "updates_per_epoch": updates_per_epoch,
                                     "nonfinite_updates": nonfinite_updates, "device_type": device.type, "amp_enabled": amp},
                "mandatory_snapshots": mandatory_snapshots,
                "sampler": {"next_epoch": epoch, "seed": seed, "global_batch": global_batch,
                            "micro_batch": micro_batch, "world_size": world}, "top_checkpoints": top}

    if stage == "B0" and resume is None:
        joint = checkpoint_state(0)
        if rank == 0:
            save_checkpoint_atomic(model_dir / "joint_init.pt", joint)
    if stage == "C1" and resume is None:
        if rank == 0:
            initial_roi, initial_full, initial_predictions = _run_epoch_evaluation(model, eval_data, config, device, stage)
            initial_directory = output_dir / "initial_eval"
            initial_directory.mkdir()
            write_json_new(initial_directory / "gt_roi_diagnostic.json", initial_roi)
            write_json_new(initial_directory / "eval_metrics.json", initial_full)
            write_predictions(initial_directory / "predictions_raw.jsonl", initial_predictions)
        if distributed:
            torch.distributed.barrier()
    async_steps = stage_config.get("eval_every_steps") if stage in ("A0", "B0") else None
    if async_steps is not None and (type(async_steps) is not int or async_steps < 1):
        raise ValueError("eval_every_steps must be a positive optimizer-step interval")
    step_evaluator = None
    if async_steps is not None:
        def start_step_evaluator():
            nonlocal step_evaluator
            from eval.action4_step_eval import AsyncStepEvaluator
            step_evaluator = AsyncStepEvaluator(root, work_output, model_dir, config)
        synchronized_rank0_action(start_step_evaluator, rank=rank, distributed=distributed, label="Start asynchronous eval")
    component_stopped = False
    try:
        for epoch in range(next_epoch, stage_config["epochs"]):
            indices = epoch_indices(view.base_count, epoch=epoch, seed=seed, s1_count=view.s1_count,
                                     replay_base=stage_config.get("replay_s0", 6000), replay_new=stage_config.get("replay_s1", 2000))
            sampler = EpochWindowSampler(indices, global_batch, micro_batch, rank, world)
            loader_generator = torch.Generator().manual_seed(seed + epoch * 1000003 + rank)
            loader = DataLoader(view, batch_sampler=sampler, collate_fn=list,
                                num_workers=config["runtime"].get("stage_workers", {}).get(stage, config["runtime"].get("workers", 0)), worker_init_fn=_worker_seed,
                                generator=loader_generator, pin_memory=device.type == "cuda")
            train_model.train()
            optimizer.zero_grad(set_to_none=True)
            accumulation_loss = {"center": 0., "offset": 0., "box": 0., "heatmap": 0.}
            window_start = time.perf_counter()
            for batch_index, items in enumerate(loader):
                micro_index = batch_index % sampler.accumulation
                window_index = batch_index // sampler.accumulation
                real_count = sum(item["training_real"] for item in items)
                images, records = [item["image"] for item in items], [item["record"] for item in items]
                is_last_micro = micro_index == sampler.accumulation - 1
                sync_context = train_model.no_sync() if distributed and not is_last_micro else contextlib.nullcontext()
                with sync_context:
                    with torch.autocast(device_type=device.type, enabled=amp, dtype=torch.float16):
                        if roi_only:
                            rois, metadata = _roi_builder(images, records, config, device, roi_rng,
                                                        selected_instances=[i["roi_instance_index"] for i in items],
                                                        jitter=stage == "A0" and stage_config.get("roi_jitter", False),
                                                        jitter_parameters=stage_config.get("roi_jitter_parameters"))
                            result = train_model(None, roi_images=rois)
                            hm = corner_heatmap_loss(result["corner_logits"], metadata["targets"], metadata["valid"],
                                                     metadata["roi_batch_index"], len(items))
                            losses = {"center": hm * 0, "offset": hm * 0, "box": hm * 0, "heatmap": hm}
                        else:
                            full, affines = prepare_full_images(images, size=tuple(config["preprocess"]["full_size"]), device=device)
                            targets = build_detection_targets(records, affines, full_size=tuple(config["preprocess"]["full_size"]),
                                                               stride=4, device=device)
                            def builder(detached_outputs):
                                predictions = decode_full_boxes(detached_outputs, affines, config) if epoch >= stage_config.get("predicted_roi_after_epoch", 5) else None
                                return _roi_builder(images, records, config, device, roi_rng, predictions=predictions,
                                                    predicted_probability=stage_config.get("predicted_roi_probability", .5),
                                                    jitter_parameters=stage_config.get("roi_jitter_parameters"))
                            result = train_model(full, roi_builder=builder)
                            metadata = result["roi_metadata"]
                            losses = detection_losses(result, targets)
                            losses["heatmap"] = corner_heatmap_loss(result["corner_logits"], metadata["targets"], metadata["valid"],
                                                                    metadata["roi_batch_index"], len(items))
                        total = sum(losses.values())
                        # DDP averages rank gradients. Scale by the number of
                        # real complete images/ROIs and actual global window,
                        # including zero-real tail ranks and short last batches.
                        weight = real_count * world / sampler.window_counts[window_index]
                        scaled_loss = total * weight
                    if not _all_ranks_finite(torch.isfinite(total).item(), device, distributed):
                        raise FloatingPointError("Non-finite loss; all ranks stop before backward")
                    scaler.scale(scaled_loss).backward()
                for name, value in losses.items():
                    accumulation_loss[name] += float(value.detach()) * real_count / sampler.window_counts[window_index]
                if not is_last_micro:
                    continue
                scaler.unscale_(optimizer)
                gradients_finite = all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters())
                finite = _all_ranks_finite(gradients_finite, device, distributed)
                if finite:
                    if any(p.grad is not None and p.grad.isfinite().all() and torch.count_nonzero(p.grad).item() for p in model.backbone.parameters()):
                        finite_backbone_grad_observed = True
                    if global_step < stage_config.get("freeze_backbone_updates", 0):
                        for parameter in model.backbone.parameters():
                            parameter.grad = None
                    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), stage_config.get("clip_grad_norm", 1.))
                    if not _all_ranks_finite(torch.isfinite(gradient_norm).item(), device, distributed):
                        raise FloatingPointError("Non-finite gradient norm after finite-gradient check")
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    global_step += 1
                else:
                    nonfinite_updates += 1
                    if amp:
                        scaler.update(new_scale=max(scaler.get_scale() * .5, 1.))
                    if nonfinite_updates > stage_config.get("max_nonfinite_updates", 0):
                        raise FloatingPointError("Non-finite gradients synchronized across ranks; optimizer/scheduler not advanced")
                optimizer.zero_grad(set_to_none=True)
                duration = time.perf_counter() - window_start
                durations.append(duration)
                logged_losses = torch.tensor([accumulation_loss[k] for k in ("center", "offset", "box", "heatmap")],
                                             dtype=torch.float64, device=device)
                if distributed:
                    torch.distributed.all_reduce(logged_losses, op=torch.distributed.ReduceOp.SUM)
                if rank == 0:
                    with (work_output / "train_updates.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps({"epoch": epoch + 1, "global_step": global_step, "optimizer_updated": finite,
                                                "losses": dict(zip(("center", "offset", "box", "heatmap"), logged_losses.cpu().tolist())), "seconds": duration,
                                                "global_actual_count": sampler.window_counts[window_index],
                                                "lr": {g["group_name"]: g["lr"] for g in optimizer.param_groups},
                                                "predicted_roi_attempted_last_micro": metadata["predicted_roi_attempted"],
                                                "predicted_roi_used_last_micro": metadata["predicted_roi_used"]}) + "\n")
                if synchronized_timing:
                    marker = synchronized_update_boundary(torch, device, distributed=distributed,
                                                          rank=rank, enabled=True)
                    if rank == 0:
                        boundary = {"epoch": epoch + 1, "global_step": global_step,
                                    "optimizer_updated": finite,
                                    "global_actual_count": sampler.window_counts[window_index],
                                    "completed_monotonic_ns": marker}
                        synchronized_boundaries.append(boundary)
                        with (work_output / "synchronized_update_boundaries.jsonl").open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(boundary) + "\n")
                if async_steps is not None and finite and global_step % async_steps == 0:
                    def publish_step_snapshot():
                        snapshot = {"stage": stage, "run_id": run_id, "claim_scope": claim,
                                    "config": config, "identity": identity, "initialization": initialization,
                                    "model": model.state_dict(), "global_step": global_step}
                        step_evaluator.submit(snapshot)
                    synchronized_rank0_action(publish_step_snapshot, rank=rank, distributed=distributed, label="Publish optimizer-step snapshot")
                accumulation_loss = {"center": 0., "offset": 0., "box": 0., "heatmap": 0.}
                window_start = time.perf_counter()
                if max_updates is not None and global_step >= max_updates:
                    component_stopped = True
                    break
            epoch_completed = not component_stopped or (window_index + 1 == len(sampler.window_counts))
            if async_steps is not None:
                # Recovery index is separate from validation-ranked step selection.
                recovery_name = f"epoch_{epoch + 1:03d}_step_{global_step:08d}.pt"
                top = [{"file": recovery_name, "epoch": epoch + 1, "key": [], "scope": "recovery_only_not_eval_ranked"}]
                recovery = checkpoint_state(epoch + 1 if epoch_completed else epoch, boundary=epoch_completed)
                def save_epoch_recovery():
                    recovery_path = model_dir / ("last.pt" if epoch_completed else "component_partial.pt")
                    save_checkpoint_atomic(recovery_path, recovery)
                    if epoch_completed:
                        link_checkpoint_atomic(recovery_path, model_dir / recovery_name)
                synchronized_rank0_action(save_epoch_recovery, rank=rank, distributed=distributed, label="Save recovery checkpoint")
                if component_stopped:
                    break
                continue
            # All ranks skip evaluation together; checkpoints are written at evaluation boundaries.
            # Final and interrupted component epochs always retain evidence.
            if (stage == "A0" and not component_stopped and epoch_completed
                    and (epoch + 1) % eval_interval != 0
                    and epoch + 1 < stage_config["epochs"]):
                continue
            paused_for_review = (stage == "C1" and not component_smoke and epoch_completed
                                 and (epoch + 1) % review_interval == 0 and epoch + 1 < stage_config["epochs"])
            if distributed:
                torch.distributed.barrier()
            if rank == 0:
                diagnostic_data = base if stage == "D0" else eval_data
                if stage == "D0":
                    # Exactly the fixed D0 ROI subset; duplicate source images
                    # are scored once with selected instances filtered below.
                    diagnostic_data = _D0DiagnosticView(base, view.units)
                roi_report, full_report, predictions = _run_epoch_evaluation(model, diagnostic_data, config, device, stage)
                last_report = full_report or roi_report
                epoch_output = work_output / f"epoch_{epoch + 1:03d}_step_{global_step:08d}"
                epoch_output.mkdir()
                write_json_new(epoch_output / "gt_roi_diagnostic.json", roi_report)
                if full_report is not None:
                    write_json_new(epoch_output / "eval_metrics.json", full_report)
                    write_predictions(epoch_output / "predictions_raw.jsonl", predictions)
                    if stage == "C1":
                        initial = json.loads((output_dir / "initial_eval/eval_metrics.json").read_text())
                        baseline, current = initial["thresholds"]["0.05"], full_report["thresholds"]["0.05"]
                        changes = {}
                        for family, values in current["per_scene_family"].items():
                            before = baseline["per_scene_family"][family]["recall"]
                            after = values["recall"]
                            changes[family] = after - before if before is not None and after is not None else None
                        write_json_new(epoch_output / "c1_comparison.json", {
                            "baseline": "initial_eval/eval_metrics.json", "metric": "F1@5%",
                            "f1_before": baseline["f1"], "f1_after": current["f1"],
                            "scene_recall_changes": changes, "status": "numeric_evidence_pending_video_review",
                            "does_not_confirm_real_video_improvement": True})
                    key = list(selection_key(full_report))
                else:
                    p95 = roi_report["epe_p95_heatmap_px"]
                    key = [-p95 if p95 is not None else -1e30]
                # JSON may not contain infinity; undefined EPE ranks last.
                key = [x if math.isfinite(x) else -1e30 for x in key]
                checkpoint_name = f"epoch_{epoch + 1:03d}_step_{global_step:08d}.pt"
                previous_top = list(top)
                top = sorted([*top, {"file": checkpoint_name, "key": key, "epoch": epoch + 1,
                                    "report_dir": str(epoch_output.relative_to(root))}],
                             key=lambda x: tuple(x["key"]), reverse=True)[:3]
                bind_previous_top(top, model_dir, checkpoint_name)
                d0_early_pass = (stage == "D0" and not component_smoke and not allow_diagnostic
                                 and geometry_checks["gate_pass"]
                                 and roi_report["ideal_heatmap_p95_px"] is not None and roi_report["ideal_heatmap_p95_px"] <= 1
                                 and roi_report["epe_p95_heatmap_px"] <= 2
                                 and roi_report["improvement_over_zero_p95"] is not None
                                 and roi_report["improvement_over_zero_p95"] >= .8
                                 and finite_backbone_grad_observed
                                 and not torch.equal(starting_backbone, first_backbone.detach().cpu()))
            if distributed:
                values = [top, d0_early_pass]
                torch.distributed.broadcast_object_list(values, src=0)
                top, d0_early_pass = values
            if stage == "B0" and epoch + 1 == 20 and epoch_completed:
                mandatory_snapshots = {
                    "epoch_020.pt": {"source_file": f"epoch_020_step_{global_step:08d}.pt", "global_step": global_step, "sha256": None},
                    "best_through_epoch020.pt": {"source_file": top[0]["file"], "global_step": int(top[0]["file"].split("_step_")[1][:-3]), "sha256": top[0].get("sha256")}}
            state = checkpoint_state(epoch + 1 if epoch_completed else epoch, boundary=epoch_completed)
            if rank == 0:
                current_path = model_dir / ("last.pt" if epoch_completed else "component_partial.pt")
                save_checkpoint_atomic(current_path, state)
                if checkpoint_name in {entry["file"] for entry in top}:
                    link_checkpoint_atomic(current_path, model_dir / checkpoint_name)
                link_checkpoint_atomic(model_dir / top[0]["file"], model_dir / "best.pt")
                atomic_json(model_dir / "top3.json", top)
                # Explicit bounded retention: every epoch's scores/predictions
                # survive, checkpoint states retain top3 + last + mandatory
                # B0 snapshots. Only files generated by this run's previous
                # top3 index can be retired; user/other-run files are untouched.
                retained = {entry["file"] for entry in top}
                if not epoch_completed and (model_dir / "last.pt").is_file():
                    # A diagnostic partial can rank separately, but must not
                    # retire aliases needed to resume the last committed epoch.
                    committed = load_checkpoint(model_dir / "last.pt")
                    retained.update(entry["file"] for entry in committed["top_checkpoints"])
                for previous in previous_top:
                    if previous["file"] not in retained:
                        old_path = model_dir / previous["file"]
                        if old_path.parent != model_dir or not old_path.name.startswith("epoch_") or old_path.suffix != ".pt":
                            raise ValueError("Unexpected checkpoint retention path")
                        old_path.unlink()
                current_sha = sha256_file(current_path)
                for entry in top:
                    if entry["file"] == checkpoint_name:
                        entry["sha256"] = current_sha
                if stage == "B0" and epoch_completed and epoch + 1 >= 20:
                    recover_epoch20(state, current_path, repair=True)
                video_config = config.get("video", {})
                due = stage in ("B0", "C1") and (epoch + 1) % video_config.get("every_epochs", 5) == 0
                at_end = epoch + 1 == stage_config["epochs"] or component_stopped
                if queued:
                    if video_due(stage, epoch + 1, stage_config["epochs"], config,
                                 component=component_smoke, completed=epoch_completed):
                        video_queue.ensure_event(stage, epoch + 1, model_dir / top[0]["file"],
                                                 component=claim == "component_diagnostic")
                    queued_state = video_queue.state()
                    paused_for_review = epoch_pause_required(
                        stage, epoch + 1, stage_config["epochs"], len(queued_state["outstanding"]),
                        complete_epoch=epoch_completed, component=component_smoke,
                        review_interval=review_interval or 5)
                    if queued_state["failed"] and epoch_completed:
                        paused_for_review = True
                elif (stage in ("B0", "C1") and video_config.get("frame_manifest")
                        and (not component_smoke or video_config.get("run_during_component", False))
                        and (due or at_end or paused_for_review)):
                    from eval.action4_video import run_video
                    sidework_rng = capture_rng()
                    try:
                        run_video(root, config, model_dir / "best.pt", video_config["frame_manifest"], run_id=run_id,
                                  output_dir=work_output / f"video_epoch_{epoch + 1:03d}", allow_component=claim == "component_diagnostic")
                    finally:
                        restore_rng(sidework_rng)
            if distributed:
                # Only short enqueue/state work precedes this broadcast in
                # queued mode. No rank waits for a complete video inference.
                pause = [paused_for_review, queued_state]
                torch.distributed.broadcast_object_list(pause, src=0)
                paused_for_review, queued_state = pause
            if component_stopped or d0_early_pass or paused_for_review:
                break
        if async_steps is not None:
            def drain_step_evaluation():
                if global_step % async_steps:
                    step_evaluator.submit({"stage": stage, "run_id": run_id, "claim_scope": claim,
                                           "config": config, "identity": identity, "initialization": initialization,
                                           "model": model.state_dict(), "global_step": global_step})
                step_evaluator.mark_complete()
            synchronized_rank0_action(drain_step_evaluation, rank=rank, distributed=distributed, label="Close asynchronous producer")
            while True:
                state = [step_evaluator.poll_finished() if rank == 0 else None]
                if distributed:
                    torch.distributed.broadcast_object_list(state, src=0)
                if state[0] is not None:
                    if state[0] != "complete":raise RuntimeError(state[0])
                    break
                time.sleep(5)
            def drain_step_evaluation():
                selected = step_evaluator.finish()
                atomic_json(work_output / "async_eval_completion.json", selected)
            synchronized_rank0_action(drain_step_evaluation, rank=rank, distributed=distributed, label="Drain asynchronous eval")
            if queued:
                queued_state = {"all_jobs_succeeded": True, "outstanding": [], "failed": []}
        final_frozen_digest = frozen_state_digest(model, trainability_policy)
        frozen_unchanged = None if initial_frozen_digest is None else final_frozen_digest == initial_frozen_digest
        if initial_frozen_digest is not None and not _all_ranks_finite(frozen_unchanged, device, distributed):
            raise RuntimeError("Frozen A0 parameters or buffers changed during detector training")
        backbone_changed = not torch.equal(starting_backbone, first_backbone.detach().cpu())
        digest = hashlib.sha256()
        for name, value in sorted(model.state_dict().items()):
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        state_digests = [digest.hexdigest()]
        if distributed:
            state_digests = [None] * world
            torch.distributed.all_gather_object(state_digests, digest.hexdigest())
            if len(set(state_digests)) != 1:
                raise RuntimeError("Distributed ranks ended with different model/buffer states")
        result = {"status": "component_complete" if component_smoke else "training_complete_pending_video_accuracy_review",
                  "stage": stage, "run_id": run_id, "claim_scope": claim, "global_step": global_step,
                  "planned_updates": planned_total, "actual_timed_updates": len(durations),
                  "median_update_seconds": float(np.median(durations)) if durations else None,
                  "timed_scope": "legacy host update duration; includes warmup and may precede CUDA optimizer tail/loss-log synchronization; not a strict throughput window",
                  "finite_nonzero_backbone_gradient": finite_backbone_grad_observed, "backbone_parameter_changed": backbone_changed,
                  "trainability_policy": trainability_policy["policy"], "frozen_features_unchanged": frozen_unchanged,
                  "nonfinite_updates": nonfinite_updates, "models_dir": str(model_dir), "output_dir": str(output_dir),
                  "best_checkpoint": str(model_dir / "best.pt"), "test_executed": False,
                  "final_model_state_sha256": digest.hexdigest(), "world_size": world,
                  "all_rank_model_states_equal": len(set(state_digests)) == 1 if distributed else None}
        if synchronized_timing:
            timing_report = measured_update_window(synchronized_boundaries, stage=stage, global_batch=global_batch) if rank == 0 else None
            if distributed:
                reports = [timing_report]
                torch.distributed.broadcast_object_list(reports, src=0)
                timing_report = reports[0]
            result["synchronized_update_timing"] = timing_report
            if rank == 0:
                write_json_new(work_output / "synchronized_update_window.json", timing_report)
        if queued:
            # Every rank returns/prints the same classification from the
            # already broadcast queue state. Only canonical file IO is rank0.
            result["video_execution"] = "queued"
            result["video_queue"] = queued_state
            if not component_smoke and not paused_for_review and not queued_state["all_jobs_succeeded"]:
                result["status"] = "training_updates_finished_pending_queued_video"
        if paused_for_review:
            result.update({"status": "paused_at_epoch_boundary_for_video_review", "next_epoch": epoch + 1,
                           "resume_checkpoint": str(model_dir / "last.pt"),
                           "review_required": "Compare the saved full video and fixed eval with the B0 start; resume the same schedule only after verified improvement and ordinary-scene review."})
        if rank == 0:
            if stage == "D0":
                report = last_report
                gate_pass = (not component_smoke and not allow_diagnostic and report is not None
                             and geometry_checks["gate_pass"]
                             and report["rois"] == stage_config.get("roi_count", 32)
                             and report["ideal_heatmap_p95_px"] is not None and report["ideal_heatmap_p95_px"] <= 1
                             and report["epe_p95_heatmap_px"] <= 2
                             and report["improvement_over_zero_p95"] is not None and report["improvement_over_zero_p95"] >= .8
                             and finite_backbone_grad_observed and backbone_changed)
                gate = {"gate_pass": bool(gate_pass), "train_manifest_sha256": sha256_file(paths["train"]),
                        "network": config["network"], "preprocess": config["preprocess"],
                        "epe_report": report, "finite_nonzero_backbone_gradient": finite_backbone_grad_observed,
                        "backbone_parameter_changed": backbone_changed,
                        "scope": claim, "geometry_checks": geometry_checks,
                        "note": "ImageNet/geometry conventions and these diagnostics do not establish real-video accuracy."}
                current_binding = build_gate_binding(root, config, paths["train"], base.records,
                                                     component=component_smoke or allow_diagnostic)
                if current_binding != gate_binding:
                    raise ValueError("D0 source/config/data identity changed during the run; cannot issue a gate")
                gate = make_d0_gate(gate_binding, gate, component=component_smoke or allow_diagnostic,
                                    global_step=global_step)
                write_json_new(work_output / f"d0_gate_step_{global_step:08d}.json", gate)
                atomic_json(output_dir / "d0_gate.json", gate)
                result["d0_gate_pass"] = bool(gate_pass)
                if not gate_pass and not component_smoke:
                    result["status"] = "d0_gate_failed"
            write_json_new(work_output / f"completion_step_{global_step:08d}.json", result)
            atomic_json(output_dir / "completion.json", result)
        return result
    except Exception as exc:
        if rank == 0:
            failure = {"status": "failed", "stage": stage, "global_step": global_step,
                       "error_type": type(exc).__name__, "error": str(exc), "partial_epoch_resumable": False}
            filename = work_output / f"failure_step_{global_step:08d}_{time.time_ns()}.json"
            write_json_new(filename, failure)
        raise
    finally:
        if step_evaluator is not None:
            step_evaluator.stop()
        if distributed and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


class _D0DiagnosticView:
    def __init__(self, base, units):
        self.base = base
        selection = {}
        for sample_index, instance_index in units:
            selection.setdefault(sample_index, set()).add(instance_index)
        self.selection = sorted(selection.items())
        self.records = []
        for sample_index, ids in self.selection:
            record = dict(base.records[sample_index])
            record["instances"] = [gt for i, gt in enumerate(record["instances"]) if i in ids]
            self.records.append(record)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        result = dict(self.base[self.selection[index][0]])
        result["record"] = self.records[index]
        return result
