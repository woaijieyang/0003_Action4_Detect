"""Exact cached evaluator for frozen-backbone auxiliary trials."""
import torch,numpy as np
from data.action4_direction_targets import prepare_evidence_batch
from model.action4_a0_enhancement import auxiliary_vote_evidence

@torch.inference_mode()
def prepare_cache(model,data,preprocess):
    model.eval();batches=[]
    for start in range(0,len(data),4):
        items=[data[i] for i in range(start,min(start+4,len(data)))]
        x,points,mask,objects=prepare_evidence_batch(items,preprocess,"cuda:0")
        features=model.base.decoder(model.base.backbone(x));logits=model.base.corner_head(features)
        batches.append({"start":start,"rgb":x.cpu(),"features":features.cpu(),"logits":logits.cpu(),
                        "points":points.cpu(),"mask":mask.cpu(),"objects":objects,
                        "sample_ids":[x["record"]["sample_id"] for x in items]})
    return batches

@torch.inference_mode()
def evaluate_cached(model,batches,preprocess):
    model.evidence.eval();rows=[]
    for batch in batches:
        x=batch["rgb"].cuda();points=batch["points"].cuda();mask=batch["mask"].cuda();objects=batch["objects"]
        output={"corner_logits":batch["logits"].cuda(),**model.evidence(batch["features"].cuda(),x)}
        torch.manual_seed(20261022+batch["start"]);torch.cuda.manual_seed_all(20261022+batch["start"])
        evidence=auxiliary_vote_evidence(output)
        for i,obj in enumerate(objects):
            diagonal=(preprocess["roi_size"]/4)/preprocess["roi_expansion"]
            box=np.array(obj["bbox_full_xyxy"]);extent=box[2:]-box[:2];diagonal*=np.linalg.norm(extent)/max(extent)
            base=torch.linalg.vector_norm(evidence["base_corners_hm"][i]-points[i],dim=-1)/diagonal
            aux=torch.linalg.vector_norm(evidence["auxiliary_corners_hm"][i]-points[i],dim=-1)/diagonal
            pred_mask=output["foreground_logits"][i].argmax(0).bool();gt_mask=mask[i].bool();union=(pred_mask|gt_mask).sum()
            rows.append({"sample_id":batch["sample_ids"][i],"instance_id":obj["instance_id"],
                         "base_errors":base.cpu().tolist(),"auxiliary_errors":aux.cpu().tolist(),
                         "visible":obj["corner_visible"],"heavy":obj["occlusion_inframe"]>=.4,
                         "valid_votes":evidence["numerically_valid"][i].cpu().tolist(),
                         "mask_iou":float((pred_mask&gt_mask).sum()/union.clamp_min(1))})
    def aggregate(key):
        return {"all8_at05":sum(max(r[key])<=.05 for r in rows),"heavy_all8_at05":sum(r["heavy"] and max(r[key])<=.05 for r in rows),
                "hidden_mean":float(np.mean([e for r in rows for e,v in zip(r[key],r["visible"]) if not v])),
                "visible_mean":float(np.mean([e for r in rows for e,v in zip(r[key],r["visible"]) if v]))}
    return {"scope":"GT ROI diagnostic; numerical-validity fallback candidate, not production fusion","rois":len(rows),
            "base":aggregate("base_errors"),"auxiliary":aggregate("auxiliary_errors"),
            "recoveries":sum(max(r["base_errors"])>.05 and max(r["auxiliary_errors"])<=.05 for r in rows),
            "regressions":sum(max(r["base_errors"])<=.05 and max(r["auxiliary_errors"])>.05 for r in rows),
            "valid_vote_points":sum(sum(r["valid_votes"]) for r in rows),
            "mean_mask_iou":float(np.mean([r["mask_iou"] for r in rows])),"per_roi":rows,"production_accepted":False}
