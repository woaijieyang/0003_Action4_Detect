"""Opt-in synchronized update boundaries; no training algorithm or budget changes."""
from __future__ import annotations

import time


def timing_enabled(config):
    value = config.get("runtime", {}).get("synchronized_update_timing", False)
    if type(value) is not bool:
        raise ValueError("runtime.synchronized_update_timing must be a boolean")
    return value


def synchronized_update_boundary(torch, device, *, distributed, rank, enabled, clock=None):
    """All completed work precedes rank0's marker; next work follows its broadcast.

    Caller must close the ordinary training log before this call. The marker
    deliberately precedes the release broadcast: no rank can start the next
    update before the start timestamp. Only rank0's monotonic clock is used.
    """
    if not enabled:
        return None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if distributed:
        torch.distributed.barrier()
        if device.type == "cuda":
            torch.cuda.synchronize(device)  # also finish the NCCL barrier
    value = (clock or time.perf_counter_ns)() if rank == 0 else None
    if distributed:
        marker = [value]
        torch.distributed.broadcast_object_list(marker, src=0, device=device)
        value = marker[0]
    if type(value) is not int or value < 0:
        raise ValueError("Expected a nonnegative rank0 monotonic timestamp")
    return value


def measured_update_window(records, *, stage, global_batch=8):
    """Summarize only a contiguous, same-epoch full-image window in one attempt.

    No interpolation, missing-update repair, warmup median, or cross-resume
    timestamp stitching is allowed. ROI-only D0/A0 never become images/s.
    """
    start_step, end_step = 10, 30
    if type(global_batch) is not int or global_batch < 1:
        raise ValueError("Timing global_batch must be a positive integer")
    report = {
        "global_batch": global_batch,
        "schema_version": "action4.synchronized_update_window.v1",
        "clock": "rank0 time.perf_counter_ns",
        "start_global_step": start_step, "end_global_step": end_step,
        "measured_updates": end_step - start_step,
        "scope": "completed step10 through completed step30; includes loading, accumulation, forward/backward/optimizer, loss synchronization, ordinary training log close, rank barriers and timing-release overhead",
        "excludes": "model initialization, first10 warm updates, post-step30 evaluation/checkpoint, post-boundary timing-sidecar write at step30",
    }
    if stage not in ("B0", "C1"):
        return {**report, "status": "not_full_image_stage"}
    selected = [row for row in records if start_step <= row["global_step"] <= end_step]
    if [row["global_step"] for row in selected] != list(range(start_step, end_step + 1)):
        return {**report, "status": "unavailable_noncontiguous_or_missing_boundaries"}
    if (any(row.get("optimizer_updated") is not True for row in selected)
            or any(type(row.get("global_actual_count")) is not int
                   or row["global_actual_count"] != global_batch for row in selected[1:])):
        return {**report, "status": "invalid_update_or_batch_count"}
    if len({row["epoch"] for row in selected}) != 1:
        return {**report, "status": "not_steady_cross_epoch_window"}
    times = [row["completed_monotonic_ns"] for row in selected]
    if (any(type(t) is not int or t < 0 for t in times)
            or any(b <= a for a, b in zip(times, times[1:]))):
        return {**report, "status": "invalid_monotonic_boundaries"}
    seconds = (times[-1] - times[0]) / 1e9
    images = sum(row["global_actual_count"] for row in selected[1:])
    return {**report, "status": "measured", "epoch": selected[0]["epoch"],
            "seconds": seconds, "images": images, "images_per_second": images / seconds,
            "start_completed_monotonic_ns": times[0], "end_completed_monotonic_ns": times[-1]}
