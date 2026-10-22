#!/usr/bin/env python3
"""Thin candidate review / evidence-bounded synthesis-request entry."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path.cwd()))
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--reviewed-events")
    parser.add_argument("--output", required=True)
    parser.add_argument("--total-quota", default=2000, type=int)
    args = parser.parse_args()
    from eval.action4_feedback import run_feedback
    result = run_feedback(args.root, args.run_dir, args.output, reviewed_events=args.reviewed_events,
                          total_quota=args.total_quota)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
