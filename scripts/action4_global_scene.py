#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from data.action4_global_scene import build,reopen
if __name__=='__main__':
 args=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]
 p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True)
 b=s.add_parser('build');b.add_argument('--config',required=True);b.add_argument('--output',required=True);b.add_argument('--report-dir',required=True)
 v=s.add_parser('reopen');v.add_argument('--asset',required=True);v.add_argument('--contract',required=True);v.add_argument('--report',required=True)
 a=p.parse_args(args);print(json.dumps(build(a.config,a.output,a.report_dir) if a.command=='build' else reopen(a.asset,a.contract,a.report),indent=2))
