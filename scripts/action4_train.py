#!/usr/bin/env python3
"""Thin D0/A0/B0/C1 entry; use the project environment or PYTHONPATH=src."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path.cwd()))
    parser.add_argument("--config", required=True)
    parser.add_argument("--local-config")
    parser.add_argument("--stage", required=True, choices=("D0", "A0", "B0", "C1"))
    parser.add_argument("--run-id", required=True)
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--init-weights")
    initialization.add_argument("--resume")
    parser.add_argument("--d0-gate")
    parser.add_argument("--device")
    parser.add_argument("--allow-diagnostic", action="store_true")
    parser.add_argument("--component-smoke", action="store_true")
    parser.add_argument("--max-updates", type=int)
    args = parser.parse_args()
    from model.training.action4_runtime import input_path, load_config
    from model.training.action4_trainer import train
    root = Path(args.root).resolve(strict=True)
    config = load_config(input_path(root, args.config), input_path(root, args.local_config) if args.local_config else None)
    if args.device:
        config["runtime"]["device"] = args.device
    result = train(root, config, stage=args.stage, run_id=args.run_id, init_weights=args.init_weights,
                   resume=args.resume, d0_gate=args.d0_gate, allow_diagnostic=args.allow_diagnostic,
                   component_smoke=args.component_smoke, max_updates=args.max_updates)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result.get("d0_gate_pass") is False and not args.component_smoke else 0


if __name__ == "__main__":
    raise SystemExit(main())
