"""Filesystem crash/replay tests; fixture bytes are not model results."""
import json,os
from pathlib import Path
from model.training import action4_async_recovery as recovery
from data.action4_paths import PROJECT_ROOT,contained_path
root=contained_path(PROJECT_ROOT,os.environ.get("ACTION4_RECOVERY_TEST_ROOT","outputs/action4_gdr_pose_candidate_v002/recovery_fixture"),exists=False)
root.mkdir(exist_ok=False)
for mode in ("before_done","before_alias"):
    out=root/mode;out.mkdir();queue=out/"eval";queue.mkdir()
    request=queue/"step_00000025.json";request.write_text("{}")
    snapshot=out/"snapshot.pt";snapshot.write_bytes(b"explicit recovery fixture, not a model")
    claim=request.with_suffix(".claim");claim.write_text('{"pid":-1}')
    (out/"TRAIN_FAILURE.json").write_text('{"attempt_id":"old"}')
    assert not recovery.is_current_terminal(out,"new")
    recovery.recover_claims(queue,"new");assert not claim.exists()
    result={"step":25,"score":1}
    original_atomic=recovery.atomic_json;original_link=recovery.os.link
    def crash_atomic(path,data):
        if str(path).endswith(".done.json"):raise RuntimeError("injected crash before done")
        return original_atomic(path,data)
    def crash_link(src,dst):
        if "best_alias_" in str(dst):raise RuntimeError("injected crash before alias")
        return original_link(src,dst)
    if mode=="before_done":recovery.atomic_json=crash_atomic
    else:recovery.os.link=crash_link
    try:
        recovery.commit_result(out,request,snapshot,result,lambda x:x["score"])
        raise AssertionError("Expected injected crash")
    except RuntimeError:pass
    finally:
        recovery.atomic_json=original_atomic;recovery.os.link=original_link
    assert snapshot.exists() and not request.with_suffix(".done.json").exists()
    assert Path(json.loads((out/"BEST.json").read_text())["checkpoint"]).exists()
    recovery.commit_result(out,request,snapshot,result,lambda x:x["score"])
    assert request.with_suffix(".done.json").exists() and not snapshot.exists()
    best=json.loads((out/"BEST.json").read_text());assert (out/"best.pt").stat().st_ino==Path(best["checkpoint"]).stat().st_ino
    (out/"TRAIN_COMPLETE.json").write_text('{"attempt_id":"new"}')
    assert recovery.is_current_terminal(out,"new")
print(json.dumps({"passed":True,"cases":["orphan_claim_reclaimed","stale_failure_ignored","current_completion_recognized","crash_before_done_replay","crash_before_alias_replay","immutable_best_pointer"],"scope":"filesystem protocol fixtures, not model evaluation"}))
