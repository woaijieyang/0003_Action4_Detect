"""Fixed-scene additional-batch worker with the validated Cycles cache profile."""
from pathlib import Path
import sys,os,signal,json
R=Path('/data/zy/vln/a-test');sys.path.insert(0,str(R/'src'))

def run(scene,name):
    from data.action4_phrase2_acceleration import install
    install(__file__)
    from data.action4_occlusion500_worker import run as worker
    from data import action4_occlusion500_queue as q
    state={'drain':False};original=q.claim
    class Drained(Exception):pass
    def signal_drain(signum,frame):state['drain']=True
    signal.signal(signal.SIGUSR1,signal_drain)
    def claim(*a,**kw):
        if state['drain']:raise Drained()
        return original(*a,**kw)
    q.claim=claim
    try:worker(scene,name)
    except Drained:
        p=R/'outputs/phrase2_occlusion500_v001/workers'/name/'CONTROLLED_DRAIN_COMPLETE.json'
        p.write_text(json.dumps({'pid':os.getpid(),'no_owned_frame_interrupted':True}))

if __name__=='__main__':
    args=sys.argv[sys.argv.index('--')+1:];run(args[0],args[1])
