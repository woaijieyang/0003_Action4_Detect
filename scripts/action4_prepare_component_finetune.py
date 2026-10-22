"""Freeze selected best weights for a separately configured new experiment."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from model.training.action4_best_initialization import prepare_best_initialization
from data.action4_paths import PROJECT_ROOT,contained_path
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--best",required=True)
    parser.add_argument("--target-config",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    cfg=json.loads(contained_path(PROJECT_ROOT,args.target_config,exists=True).read_text())
    print(json.dumps(prepare_best_initialization(args.best,cfg,args.output),indent=2))
