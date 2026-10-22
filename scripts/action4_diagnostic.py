"""Thin entry for the bounded, non-gating GPU component diagnostic."""
import argparse
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default=str(Path.cwd()))
    p.add_argument('--manifest', required=True)
    p.add_argument('--config', default='configs/model/action4_train.json')
    p.add_argument('--run-id', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--roi-count', type=int, default=32)
    p.add_argument('--updates', type=int, default=500)
    p.add_argument('--max-seconds', type=float, default=3600)
    p.add_argument('--max-memory-gib', type=float, default=12)
    p.add_argument('--probe-only', action='store_true')
    args=p.parse_args()
    from model.training.action4_diagnostic import diagnose
    result=diagnose(args)
    return 0 if result['updates']==args.updates and result['stop_reason'] is None else 1


if __name__=='__main__':
    raise SystemExit(main())
