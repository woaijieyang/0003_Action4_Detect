"""Run the unchanged actual-scene probe with an isolated sampler candidate."""
from pathlib import Path
import sys
R=Path('/data/zy/vln/a-test');sys.path.insert(0,str(R/'src'))
from data import action4_coverage1000_sampling as base
from data.action4_coverage1000_view_transport import propose
base.propose=propose
from data.action4_coverage1000_preflight import run
if __name__=='__main__':
    args=sys.argv[sys.argv.index('--')+1:]
    run(args[0],int(args[1]),int(args[2]),128,'outputs/coverage1000_v001/view_transport_probe_v001',
        'outputs/coverage1000_v001/TRANSPORT_PROBE_SLOTS.json','configs/data/action4_coverage1000_feasibility_wide_v001.json')
