"""Blender thin entry point; defaults to one CPU preview."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from data.action4_prompt_bank import build
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/data/action4_prompt_bank.json')
    p.add_argument('--limit',type=int,default=1);p.add_argument('--full',action='store_true')
    p.add_argument('--reference-id',action='append',dest='reference_ids',help='Repeat to select explicit registered IDs; overrides limit')
    args=p.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    if not 1<=args.limit<=48:p.error('limit must be 1..48')
    print(json.dumps(build(args.config,limit=args.limit,preview=not args.full,reference_ids=args.reference_ids)))
