"""Thin CPU-only index/replay entry for completed independent SAM clips."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from eval.action4_clip_replay import ClipReplay,freeze_metadata_refs
from eval.action4_roi_sweep import binding,write_new
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for key in ("registry","clip-id","config","tracker-config","producer-exit","metadata-index","output"):
        p.add_argument("--"+key,required=True)
    p.add_argument("--temporal-mode",required=True,choices=["offline_bidirectional_clip","causal_forward"])
    a=p.parse_args()
    refs=freeze_metadata_refs(binding(a.config),a.metadata_index,producer_exit_ref=binding(a.producer_exit))
    replay=ClipReplay(registry_ref=binding(a.registry),clip_id=a.clip_id,
        config_ref=binding(a.config),tracker_config_ref=binding(a.tracker_config),
        producer_exit_ref=binding(a.producer_exit),evidence_refs=refs)
    result=replay.public_result()
    if result["temporal_mode"]!=a.temporal_mode:raise ValueError("Requested temporal role differs from actual producer")
    write_new(a.output,result)
