#!/usr/bin/env python3
"""Prepare only by default; explicit approved one-shot CPU build is separate."""
import sys,json,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from data.action4_left_mount_phase_modules import prepare,build
if __name__=='__main__':
    args=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='mode',required=True)
    x=sub.add_parser('prepare');x.add_argument('--config',required=True)
    x=sub.add_parser('build');x.add_argument('--config',required=True);x.add_argument('--approval',required=True)
    a=p.parse_args(args);result=prepare(a.config) if a.mode=='prepare' else build(a.config,a.approval)
    print(json.dumps(result,indent=2))
