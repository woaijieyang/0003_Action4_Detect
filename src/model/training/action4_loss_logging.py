"""Detached scalar logging candidate; no trainer wiring or objective changes."""
import torch


def accumulate_loss_logging(logged, losses, *, role, weights, counts,
                            world_size, active_count=0, global_count=None):
    """Match the stage2 Python-float loop with one packed host transfer.

    Contributions retain their original dtype and operation order. Conversion
    to float64 happens only after the contribution arithmetic, matching Python's
    float conversion without changing mixed-dtype values or accumulation order.
    All input loss tensors remain untouched; this helper never runs backward.
    """
    if role not in ("A0", "B0"):
        raise ValueError("Explicit A0 or B0 role required")
    if role == "B0" and (global_count is None or global_count <= 0 or active_count < 0):
        raise ValueError("Invalid optimizer batch mass")
    keys = []
    contributions = []
    device = None
    for key, value in losses.items():
        if not torch.is_tensor(value) or value.ndim != 0 or not value.is_floating_point():
            raise ValueError("Floating scalar losses required")
        if device is not None and value.device != device:
            raise ValueError("Loss scalars must share a device")
        device = value.device
        detached = value.detach()
        if role == "A0":
            contribution = (weights.get(key, 1.) * world_size * detached / counts[key]
                            if counts[key] > 0 else detached * 0)
        else:
            # Preserve objective({key: value.detach()}, ...), including sum's 0.
            contribution = sum([weights.get(key, 1.) * detached
                                * (world_size * active_count / global_count)])
        keys.append(key)
        contributions.append(contribution.to(dtype=torch.float64))
    if not contributions:
        return logged
    values = torch.stack(contributions).cpu().tolist()
    for key, contribution in zip(keys, values):
        logged[key] = logged.get(key, 0.) + contribution
    return logged
