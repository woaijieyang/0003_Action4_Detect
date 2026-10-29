"""Reuse the frozen worker under an explicitly versioned cached rendering setup."""
from pathlib import Path
import sys
R=Path('/data/zy/vln/a-test');sys.path.insert(0,str(R/'src'))
if __name__=='__main__':
    from data.action4_phrase2_acceleration import install
    from data.action4_phrase2_capacity import bounded_worker
    install(__file__)
    args=sys.argv[sys.argv.index('--')+1:]
    bounded_worker(args[0],args[1])
