from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from model.training.action4_component_dispatch import replace_eval_workers
if __name__=='__main__':replace_eval_workers()
