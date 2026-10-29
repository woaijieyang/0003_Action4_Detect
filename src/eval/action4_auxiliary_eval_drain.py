"""Auxiliary-worker lifecycle only; original computation/commit functions stay unchanged."""
import dis,json,os,signal,sys,time
from pathlib import Path

def install_drain(worker,commit,receipt):
    receipt=Path(receipt);returns={i.offset for i in dis.get_instructions(commit) if i.opname in ('RETURN_VALUE','RETURN_CONST')}
    def finish(reason):
        with receipt.open('x') as f:
            json.dump(dict(pid=os.getpid(),reason=reason,time=time.time(),exit_code=0),f);f.flush();os.fsync(f.fileno())
        os._exit(0)
    def profile(frame,event,arg):
        if event=='return' and frame.f_code is commit.__code__ and frame.f_lasti in returns:
            request=frame.f_locals.get('request')
            if request is not None and Path(request).with_suffix('.done.json').is_file():finish('successful_original_commit_completed')
        if event=='c_call' and arg is time.sleep and frame.f_code is worker.__code__ and frame.f_locals.get('request') is None:
            finish('original_worker_idle_without_claim')
    def arm(signum,frame):sys.setprofile(profile)
    signal.signal(signal.SIGUSR2,arm)
    return arm
