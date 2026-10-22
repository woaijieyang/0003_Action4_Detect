"""Action 4 model APIs; importing this package constructs/downloads nothing."""

from importlib import import_module


_EXPORTS = {
    "Action4Net": "action4_network",
    "decode_detections": "action4_decode",
    "decode_corners": "action4_decode",
    "detection_losses": "action4_losses",
    "corner_heatmap_loss": "action4_losses",
    "joint_loss": "action4_losses",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    globals()[name] = value
    return value
