"""Thin Action 4 data entrypoint; run with the project 'data' extra installed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.action4_adapter import DataAdapter


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root; derived outputs must stay inside it")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("probe", "extract", "profile"):
        command = commands.add_parser(name)
        command.add_argument("--source", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if name == "extract":
            group = command.add_mutually_exclusive_group(required=True)
            group.add_argument("--frames", help="Comma-separated zero-based original frame IDs")
            group.add_argument("--all", action="store_true", help="Explicitly extract the entire video as PNG")
        if name == "profile":
            command.add_argument("--samples", type=int, default=16)
    command = commands.add_parser("validate")
    command.add_argument("--manifest", type=Path, required=True)
    command = commands.add_parser("split")
    command.add_argument("--manifest", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--seed", type=int, default=20261018)
    command.add_argument("--ratios", default="0.8,0.1,0.1")
    command = commands.add_parser("encode")
    command.add_argument("--manifest", type=Path, required=True)
    command.add_argument("--frames-dir", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--fps", help="Optional exact rational assertion, e.g. 2997/100")
    command.add_argument("--crf", type=int, default=16)
    command.add_argument("--threads", type=int, default=4, choices=range(1, 9), help="CPU encoder/decoder threads (1-8)")
    command = commands.add_parser("render-pilot")
    command.add_argument("--asset-dir", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--blender", default="blender", help="Existing Blender executable; never installed automatically")
    command.add_argument("--count", type=int, default=32, choices=[32], help="Current measured pilot recipe has exactly 32 samples")
    command.add_argument("--seed", type=int, default=20261018)
    command.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    try:
        adapter = DataAdapter(args.root)
        if args.command == "probe":
            value = adapter.probe(args.source)
            output = adapter.save_json(value, args.output)
            summary = {"output": str(output), "frame_count": value["frame_count"], "fps": value["observed_cfr_fps"]}
        elif args.command == "extract":
            ids = [int(v) for v in args.frames.split(",")] if args.frames is not None else None
            value = adapter.extract(args.source, args.output, frame_ids=ids, all_frames=args.all)
            summary = {"output": str(args.output), "frames": len(value["frames"]), "complete": value["complete"]}
        elif args.command == "profile":
            value = adapter.profile(args.source, args.output, samples=args.samples)
            summary = {"output": str(args.output), "inspection_frames": len(value["frames"]), "complete": value["complete"]}
        elif args.command == "validate":
            value = adapter.read_samples(args.manifest)
            summary = {"valid_samples": len(value)}
        elif args.command == "split":
            value = adapter.read_samples(args.manifest)
            summary = adapter.write_split(value, args.output, seed=args.seed, ratios=tuple(float(v) for v in args.ratios.split(",")))
        elif args.command == "render-pilot":
            summary = adapter.render_pilot(args.asset_dir, args.output, blender=args.blender,
                                          count=args.count, seed=args.seed, threads=args.threads)
        else:
            output = args.output if args.output.is_absolute() else adapter.root / args.output
            output = output.resolve(strict=False)
            # Run results belong to the real outputs/ subtree, never data/ or a symlink alias.
            outputs = adapter.root / "outputs"
            project_outputs = Path(__file__).resolve().parents[1] / "outputs"
            if (outputs.resolve(strict=False) != outputs or not output.is_relative_to(outputs)
                    or project_outputs.resolve(strict=False) != project_outputs
                    or not output.is_relative_to(project_outputs)):
                raise ValueError("encode CLI output must be inside the real project outputs/ directory")
            summary = adapter.encode_video(args.manifest, args.frames_dir, output, fps=args.fps,
                                           crf=args.crf, threads=args.threads)
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"action4_data: {exc}\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
