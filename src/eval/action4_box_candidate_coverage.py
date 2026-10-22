"""Failure attribution on cached200dev pools; oracle uses GT for diagnosis only."""
import json,collections
from data.action4_paths import PROJECT_ROOT as R
from eval.action4_metrics import bbox_iou,maximum_matching
from eval.action4_soft_nms import select
from model.training.action4_runtime import atomic_json

def run():
    pools=json.loads((R/"outputs/action4_box_quality_soft_nms_v001/POOLS.json").read_text())
    refs={r["sample_id"]:r for r in map(json.loads,(R/"data/eval/annotations.jsonl").read_text().splitlines())}
    counts=collections.Counter();failures=[]
    def hits(ps,gt):
        return len(maximum_matching([[i for i,g in enumerate(gt) if bbox_iou(p["bbox_full_xyxy"],g["bbox_full_xyxy"])>=.5] for p in ps],len(gt)))
    for row in pools:
        gt=[g for g in refs[row["sample_id"]]["instances"] if not g.get("ignore",False)]
        if len(gt)!=2:continue
        ps=[dict(p,score=s) for p,s in zip(row["candidates"],row["new_scores"])]
        final=select(ps,"soft_gaussian50")
        counts["dual_images"]+=1;oracle=hits(ps,gt);counts["oracle_both"]+=oracle==2
        if hits([p for p in final if p["score"]>=.3],gt)==2:
            counts["actual_both"]+=1;continue
        if oracle<2:reason="proposal_missing"
        elif hits([p for p in ps if p["score"]>=.3],gt)<2:reason="raw_score_threshold"
        elif hits(final,gt)<2:reason="ranking_or_suppression"
        else:reason="soft_nms_score_threshold"
        counts[reason]+=1
        failures.append({"sample_id":row["sample_id"],"reason":reason,
                         "best_iou_per_gt":[max(bbox_iou(p["bbox_full_xyxy"],g["bbox_full_xyxy"]) for p in ps) for g in gt]})
    atomic_json(R/"outputs/action4_box_refinement_v001/CANDIDATE_COVERAGE.json",
                {"scope":"GT-assisted oracle diagnosis only, not achievable accuracy claim","counts":dict(counts),"failures":failures})
if __name__=="__main__":run()
