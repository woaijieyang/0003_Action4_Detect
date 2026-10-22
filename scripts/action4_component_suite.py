"""Launch the explicitly configured independent component suite."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from model.training.action4_component_dispatch import main
if __name__=='__main__':main()
