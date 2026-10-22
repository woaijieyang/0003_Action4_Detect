from pathlib import Path
import sys
root=Path('/data/zy/vln/a-test')
if Path.cwd().resolve()!=root:raise ValueError('Physical project root')
sys.path.insert(0,str(root/'src'))
from data.action4_modular_render import main
if __name__=='__main__':main()
