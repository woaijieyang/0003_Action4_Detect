from pathlib import Path
import sys,argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from eval.action4_component_eval_cached import worker
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--suite',required=True);a=p.parse_args();worker(a.suite)
