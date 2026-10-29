"""Cross-stage lock and complete prior-stage gates around the frozen dispatcher."""
import argparse,fcntl,json,os,subprocess
from pathlib import Path
from data.action4_paths import PROJECT_ROOT as R
from data.action4_coverage1000_controller import run
from data import action4_coverage1000_queue as queue

def main(run_id):
    out=R/'outputs/coverage1000_v001'
    guard=(out/'controller.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfg=json.loads((R/'configs/data/action4_coverage1000_v001.json').read_text())
    promotion=json.loads((out/'PREVIEW_PROMOTION_COMPLETE.json').read_text());tail=json.loads((out/'GEOMETRY_TAIL_APPLIED.json').read_text())
    if promotion['images']!=128 or promotion['protocol_id']!=cfg['protocol_id'] or tail['protocol_id']!=cfg['protocol_id']:raise ValueError('Prior stage admission differs')
    if set(queue.counts())-{'accepted','pending'}:raise ValueError('Other owned generation/checks still active')
    dispatch=json.loads((out/'geometry_tail_v001/DISPATCH.json').read_text())
    pids=[str(r['pid']) for r in dispatch['jobs']];state=subprocess.run(['ps','-p',','.join(pids),'-o','pid=,stat=,args='],text=True,capture_output=True)
    live=[line for line in state.stdout.splitlines() if len(line.split())>2 and 'Z' not in line.split()[1] and ('action4_coverage1000_geometry_tail' in line or 'action4_coverage1000_preview' in line)]
    if live:raise ValueError('Wait for actual CPU jobs to exit; do not overlap the72-core allocations')
    run(run_id,'bulk')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args();main(a.run_id)
