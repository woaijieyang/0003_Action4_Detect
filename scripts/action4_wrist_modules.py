#!/usr/bin/env python3
"""CPU-only build/reopen entry for independent neutral wrist libraries."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from data.action4_wrist_modules import build, reopen

if __name__ == '__main__':
    args = sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('build');p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--report-dir',required=True)
    p=sub.add_parser('reopen');p.add_argument('--asset',required=True);p.add_argument('--contract',required=True);p.add_argument('--report',required=True)
    a=parser.parse_args(args)
    result=build(a.config,a.output,a.report_dir) if a.command=='build' else reopen(a.asset,a.contract,a.report)
    print(json.dumps(result,indent=2))
