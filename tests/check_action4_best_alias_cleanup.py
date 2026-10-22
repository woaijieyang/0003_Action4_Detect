"""Repeated best alias refresh must not leave hard-link temporary names."""
import json,tempfile
from pathlib import Path
from data.action4_paths import PROJECT_ROOT as R
from model.training.action4_runtime import link_checkpoint_atomic
from model.training.action4_async_recovery import commit_result
with tempfile.TemporaryDirectory(prefix="best_alias_",dir=R/"runtime") as directory:
    root=Path(directory);source=root/"source.pt";source.write_bytes(b"fixture")
    target=root/"best.pt"
    for _ in range(4):link_checkpoint_atomic(source,target)
    leftovers=list(root.glob("*.tmp"))
    assert not leftovers,f"Redundant alias temp names remain: {len(leftovers)}"
    assert target.stat().st_ino==source.stat().st_ino
    queue=root/"eval";queue.mkdir()
    for step,score in [(1,10),(2,5),(3,4)]:
        snapshot=root/f"snapshot_{step}.pt";snapshot.write_bytes(bytes([step]))
        request=queue/f"step_{step:08d}.json";request.write_text("{}")
        commit_result(root,request,snapshot,{"step":step,"score":score},lambda x:x["score"])
    assert not list(root.glob("*.tmp")),"Async selection left temporary hard links"
    assert target.read_bytes()==bytes([1])
print(json.dumps({"passed":True,"repeated_alias_no_temp_names":True,"nonwinning_eval_keeps_best":True,"scope":"filesystem alias regression"}))
