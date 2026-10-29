"""Host-side evaluation publication fence: no NCCL wait during queue backpressure.

Every training rank calls this at the same optimizer boundary. Rank0 completes
publication before releasing followers through an attempt-bound atomic token.
"""
import json,time,re,traceback
from pathlib import Path
from model.training.action4_runtime import atomic_json

def coordinated_publish(out,*,rank,attempt,step,config_id,publish_fn,timeout=7200.,poll=.1):
    if type(rank) is not int or rank<0 or type(step) is not int or step<0 or re.fullmatch('[0-9a-f]{32}',attempt or '') is None or re.fullmatch('[0-9a-f]{64}',config_id or '') is None:
        raise ValueError('Explicit rank/attempt/optimizer boundary required')
    root=Path(out);folder=root/'publication';folder.mkdir(exist_ok=True)
    key=dict(attempt_id=attempt,step=step,config_id=config_id)
    ready=folder/f'{attempt}_step_{step:08d}.ready.json';failed=folder/f'{attempt}_step_{step:08d}.failed.json'
    if rank==0:
        try:
            publish_fn()
            request=root/'eval'/f'step_{step:08d}.json';row=json.loads(request.read_text())
            if row.get('step')!=step or row.get('config_id')!=config_id:raise ValueError('Published request has wrong optimizer/config identity')
            atomic_json(ready,dict(key,status='PUBLISHED',request=str(request),time_unix=time.time()))
        except BaseException:
            atomic_json(failed,dict(key,status='PUBLICATION_FAILED',traceback=traceback.format_exc()));raise
    start=time.monotonic()
    while True:
        if failed.exists():
            doc=json.loads(failed.read_text())
            if any(doc.get(k)!=v for k,v in key.items()):raise ValueError('Mixed publication failure identity')
            raise RuntimeError('Leader evaluation publication failed: '+doc['traceback'])
        if ready.exists():
            doc=json.loads(ready.read_text())
            if any(doc.get(k)!=v for k,v in key.items()) or doc.get('status')!='PUBLISHED':raise ValueError('Mixed publication release identity')
            return doc
        if list((root/'eval').glob('*.failed.json')):raise RuntimeError('Async evaluation failed while all ranks wait on host')
        if time.monotonic()-start>timeout:raise TimeoutError('Host publication wait exceeded explicit limit')
        time.sleep(poll)
