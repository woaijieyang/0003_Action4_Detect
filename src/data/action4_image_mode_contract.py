"""Reject video-state configuration at image-only training/evaluation boundaries."""
def ensure_image_mode(cfg):
    option=cfg.get("video_tracking")
    if option is not None:
        if not isinstance(option,dict) or type(option.get("enabled",False)) is not bool:
            raise ValueError("Explicit boolean video_tracking.enabled required")
        if option.get("enabled"):raise ValueError("Video tracking must stay disabled for image training/eval")
    if cfg.get("sam_video_enabled") or cfg.get("cotracker_enabled") or cfg.get("temporal_modules"):
        raise ValueError("Temporal video modules must stay disabled for image training/eval")
