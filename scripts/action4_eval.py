#!/usr/bin/env python3
"""Thin fixed eval/test and explicit selection-freeze entry."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path.cwd()))
    parser.add_argument("--config", required=True)
    parser.add_argument("--local-config")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="eval", choices=("eval", "test"))
    parser.add_argument("--run-id")
    parser.add_argument("--gt-roi", action="store_true")
    parser.add_argument("--selection")
    parser.add_argument("--freeze-selection")
    parser.add_argument("--tune-grid")
    parser.add_argument("--device")
    parser.add_argument("--allow-diagnostic", action="store_true")
    args = parser.parse_args()
    from model.training.action4_runtime import input_path, load_config, read_config
    from eval.action4_runner import freeze_test_selection, run_evaluation, tune_decode
    root = Path(args.root).resolve(strict=True)
    config = load_config(input_path(root, args.config), input_path(root, args.local_config) if args.local_config else None)
    if args.device:
        config["runtime"]["device"] = args.device
    if args.tune_grid:
        if args.split != "eval" or args.gt_roi or args.freeze_selection or args.selection or not args.run_id:
            parser.error("Threshold tuning requires eval, a new run-id, and no GT-ROI/test/freeze options")
        result = tune_decode(root, config, args.checkpoint, read_config(input_path(root, args.tune_grid)),
                             run_id=args.run_id, allow_diagnostic=args.allow_diagnostic)
    elif args.freeze_selection:
        if args.gt_roi or args.allow_diagnostic:
            parser.error("Freezing a formal test selection is incompatible with diagnostic modes")
        result = freeze_test_selection(root, config, input_path(root, args.checkpoint), Path(args.freeze_selection))
    else:
        if not args.run_id:
            parser.error("--run-id is required for evaluation")
        result = run_evaluation(root, config, args.checkpoint, split=args.split, run_id=args.run_id,
                                gt_roi=args.gt_roi, selection=args.selection, allow_diagnostic=args.allow_diagnostic)
        result = {"output_dir": result["output_dir"], "status": result["report"]["status"],
                  "scope": result["report"]["scope"]}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
