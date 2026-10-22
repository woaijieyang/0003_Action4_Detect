"""Controlled early closure of owned, superseded A0 joint-feature trials."""
import json,os,signal,subprocess,time
from pathlib import Path
import torch
from model.training.action4_runtime import atomic_json


def cleanup_owned(receipt,cfg_path):
    for pid in [receipt['pid']]+receipt['children']:
        state=subprocess.run(['ps','-p',str(pid),'-o','stat=','-o','args='],capture_output=True,text=True)
        if state.returncode!=0 or state.stdout.strip().startswith('Z'):continue
        if '--config '+cfg_path not in state.stdout or '-m model.training.action4_component_train' not in state.stdout:raise ValueError('Recorded PID reused; no signal sent')
        os.kill(pid,signal.SIGTERM)
    for _ in range(30):
        alive=[]
        for pid in [receipt['pid']]+receipt['children']:
            state=subprocess.run(['ps','-p',str(pid),'-o','stat='],capture_output=True,text=True)
            if state.returncode==0 and not state.stdout.strip().startswith('Z'):alive.append(pid)
        if not alive:return
        time.sleep(.5)
    raise RuntimeError('Owned worker shutdown incomplete; do not reuse slot')


def main():
    R=Path('/data/zy/vln/a-test');suite=R/'outputs/action4_independent_trials_v001';torch.set_num_threads(1)
    processes=json.loads((suite/'PROCESSES.json').read_text())['processes'];closed=[]
    for name in ('a0_joint_noedges','a0_joint'):
        out=suite/name
        cfg_path='configs/model/component_trials_v001/'+name+'.json';record=next(p for p in processes if p.get('config')==cfg_path);pid=record['pid']
        if (out/'EARLY_STOP.json').exists():
            receipt=json.loads((out/'EARLY_STOP.json').read_text());cleanup_owned(receipt,cfg_path);closed.append(receipt);continue
        command=subprocess.check_output(['ps','-p',str(pid),'-o','args='],text=True)
        if '--config '+cfg_path not in command or '-m model.training.action4_component_train' not in command:raise ValueError('PID identity differs')
        status=json.loads((out/'STATUS.json').read_text());best=json.loads((out/'BEST.json').read_text());done=[json.loads(p.read_text())['step'] for p in (out/'eval').glob('*.done.json')]
        if status['epoch']<20 or max(done)<3750 or best['step']!=0:raise ValueError('Pruning evidence not satisfied')
        child_text=subprocess.run(['ps','--ppid',str(pid),'-o','pid=','-o','args='],capture_output=True,text=True).stdout
        children=[int(line.strip().split(None,1)[0]) for line in child_text.splitlines() if '--config '+cfg_path in line]
        os.kill(pid,signal.SIGSTOP)
        try:
            checkpoint=torch.load(out/'last.pt',map_location='cpu',weights_only=True);saved=checkpoint['step'];saved_epoch=checkpoint['epoch'];del checkpoint
            status=json.loads((out/'STATUS.json').read_text());best=json.loads((out/'BEST.json').read_text())
            if best['step']!=0 or not 0<=status['step']-saved<=25:raise ValueError('Evidence changed or recovery too old')
            receipt={'status':'CONTROLLED_EARLY_STOP','reason':'No development gain after >20 epochs; preserve original A0 and prioritize frozen additive enhancement plus scene repair','pid':pid,'children':children,'observed_optimizer_step':status['step'],'retained_recovery_step':saved,'retained_epoch_cursor_epoch':saved_epoch,'best_step':best['step'],'best_all8_at05':best['report']['all8_at05'],'completed_eval_through':max(done),'full80_epochs_completed':False,'engineering_failure':False,'uncheckpointed_or_inflight_work_may_be_discarded':True,'snapshots_and_scores_preserved':True}
            atomic_json(out/'EARLY_STOP.json',receipt);atomic_json(out/'FAILURE.json',receipt)
            os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
        except Exception:
            os.kill(pid,signal.SIGCONT);raise
        cleanup_owned(receipt,cfg_path)
        closed.append(receipt)
    atomic_json(suite/'PRUNED_A0_TRIALS.json',{'trials':closed,'B0_three_80_epoch_trials_unchanged':True,'A0_baseline_and_frozen_auxiliary_continue':True})
    print(json.dumps(closed))

if __name__=='__main__':main()
