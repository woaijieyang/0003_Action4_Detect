from pathlib import Path
import sys
root=Path('/data/zy/vln/a-test')
if Path.cwd().resolve()!=root:raise ValueError('Physical project root required')
sys.path.insert(0,str(root/'src'))
from data.action4_left_brace_build import main
if __name__=='__main__':main()
