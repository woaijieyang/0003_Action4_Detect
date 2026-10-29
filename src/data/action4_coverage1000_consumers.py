"""Read-only Action4 train/eval consumer gate. Never stops a process."""
import json,os,shlex,subprocess,time
from data.action4_paths import PROJECT_ROOT as R,contained_path
from data.action4_occlusion500_similarity import write

def check():
    pattern=r'(model[.]training[.]action4|eval[.]action4|/data/zy/vln/a-test/[^ ]*(train|eval))'
    result=subprocess.run(['pgrep','-af',pattern],text=True,capture_output=True)
    if result.returncode not in [0,1]:raise RuntimeError('Cannot establish active consumer state')
    active=[];other=[];ambiguous=[]
    def refers(value):
        if isinstance(value,str):return value=='data/phrase2' or value.startswith('data/phrase2/') or str(R/'data/phrase2') in value
        if isinstance(value,dict):return any(refers(v) for v in value.values())
        if isinstance(value,list):return any(refers(v) for v in value)
        return False
    for line in result.stdout.splitlines():
        pid_text,_,command=line.partition(' ')
        if int(pid_text)==os.getpid():continue
        if 'model.training.action4' not in command and 'eval.action4' not in command and not(command.startswith(str(R)) and ('train' in command or 'eval' in command)):continue
        try:args=shlex.split(command)
        except ValueError:ambiguous.append({'pid':int(pid_text),'reason':'unparseable_Action4_command'});continue
        if any('data/phrase2' in a for a in args):active.append({'pid':int(pid_text),'reason':'explicit_phrase2_input'});continue
        configs=[args[i+1] for i,x in enumerate(args[:-1]) if x=='--config']
        if not configs:ambiguous.append({'pid':int(pid_text),'reason':'Action4_train_eval_without_explicit_config'});continue
        for name in configs:
            try:path=contained_path(R,name,exists=True)
            except (ValueError,FileNotFoundError):ambiguous.append({'pid':int(pid_text),'reason':'config_not_inside_project_or_missing','config':name});continue
            cfg=json.loads(path.read_text());row={'pid':int(pid_text),'config':str(path.relative_to(R))}
            (active if refers(cfg) else other).append(row)
    report={'status':'BLOCKED_ACTIVE_CONSUMERS' if active else 'BLOCKED_AMBIGUOUS_CONSUMERS' if ambiguous else 'PASS_NO_ACTIVE_OLD_MANIFEST_TRAIN_OR_EVAL_CONSUMERS',
        'active_consumers':active,'ambiguous_consumers':ambiguous,'other_dataset_Action4_jobs':other,'processes_stopped':0,'time_unix':time.time()}
    write(R/'outputs/coverage1000_v001/OLD_CONSUMER_GATE.json',report);return report

if __name__=='__main__':print(json.dumps(check(),indent=2))
