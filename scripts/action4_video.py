#!/usr/bin/env python3
"""Thin complete-frame model annotation and verified MP4 entry."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def configure_local_video_tools(root: Path) -> None:
    """Use existing project-local FFmpeg without altering server-global PATH."""
    import os
    directory = root / "runtime" / "ffmpeg" / "bin"
    if not directory.exists():
        return  # Other installations may already provide system tools.
    resolved = directory.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError("Video tools must resolve inside the project")
    for name in ("ffmpeg", "ffprobe"):
        binary = (resolved / name).resolve(strict=True)
        if not binary.is_relative_to(root) or not binary.is_file() or not os.access(binary, os.X_OK):
            raise ValueError("Executable project-local ffmpeg and ffprobe required")
    os.environ["PATH"] = str(resolved) + os.pathsep + os.environ.get("PATH", "")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path.cwd()))
    parser.add_argument("--config", required=True)
    parser.add_argument("--local-config")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--frame-manifest", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device")
    parser.add_argument("--allow-component", action="store_true")
    parser.add_argument("--debug-numbers", action="store_true")
    args = parser.parse_args()
    from model.training.action4_runtime import input_path, load_config
    from eval.action4_video import run_video
    root = Path(args.root).resolve(strict=True)
    configure_local_video_tools(root)
    config = load_config(input_path(root, args.config), input_path(root, args.local_config) if args.local_config else None)
    if args.device:
        config["runtime"]["device"] = args.device
    result = run_video(root, config, args.checkpoint, args.frame_manifest, run_id=args.run_id,
                        allow_component=args.allow_component, debug_numbers=args.debug_numbers)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
