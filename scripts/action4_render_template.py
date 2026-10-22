"""Thin Blender-worker entry for fixed-scene candidate generation."""
import sys,json
from pathlib import Path
ROOT=Path('/data/zy/vln/a-test');sys.path.insert(0,str(ROOT/'src'))
if __name__=='__main__':
 args=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]
 if len(args)!=2 or args[0]!='--config':raise ValueError('Expected --config project-relative worker configuration')
 from data.action4_template_batch import run_blender
 result=run_blender(args[1]);print(json.dumps({k:v for k,v in result.items() if k!='entries'}),flush=True)
