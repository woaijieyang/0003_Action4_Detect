"""Cache true RGB-derived pose inputs at frozen B0 proposal boxes, not GT boxes."""
import json,torch,numpy as np
from data.action4_paths import PROJECT_ROOT as R
from data.action4_component_data import ComponentData
from data.action4_pose_rgb_batch import prepare_pose_rgb
from model.action4_release_bundle import load_bundle
from model.action4_pose_maps import surface_to_maps
from model.training.action4_runtime import atomic_json,save_checkpoint_atomic

def run():
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    root=R/"outputs/action4_a0_roi_contract_v001/predicted_roi_maps";root.mkdir(exist_ok=False)
    data=ComponentData("outputs/required/DATA_VALIDATED_ALL_SPLITS.json","eval","B0")
    source=R/"outputs/action4_photometric_validation_v001/development"
    report=json.loads((source/"REPORT.json").read_text())
    if report["identity"]["B0_step"]!=3220:raise ValueError("Frozen detector selection changed")
    predictions={r["sample_id"]:r["instances"] for r in json.loads((source/"PREDICTIONS.json").read_text())}
    matching={r["sample_id"]:r["matches"] for r in report["bbox_only"]["per_image"]}
    model,base,identity=load_bundle("models/candidates/a0_frozen_surface_pose_v001/best.pt","A0","cpu")
    if identity["frozen_surface_step"]!=11300 or identity["backbone_step"]!=6300:raise ValueError("Frozen feature lineage mismatch")
    model.eval().requires_grad_(False);base.eval().requires_grad_(False)
    parts=[];count=0
    for image_index in range(len(data)):
        item=data[image_index];record=item["record"];sid=record["sample_id"]
        gt_indices=[i for i,g in enumerate(record["instances"]) if not g.get("ignore",False)]
        pairs=matching[sid]
        if not pairs:continue
        items=[dict(item,instance=gt_indices[gi]) for pi,gi in pairs]
        boxes=[predictions[sid][pi]["bbox_full_xyxy"] for pi,gi in pairs]
        rois,geometry,metadata=prepare_pose_rgb(items,boxes=boxes)
        with torch.inference_mode():maps=surface_to_maps(model.surface(base.backbone(rois)),geometry["K"])
        for row,(pi,gi) in zip(metadata,pairs):row["B0_prediction_index"]=pi
        path=root/f"eval_{image_index:04d}.pt"
        save_checkpoint_atomic(path,{"predicted_maps":maps,"geometry_targets":geometry,"metadata":metadata,
            "input_contract":"actual RGB at detector boxes; GT only for target association/scoring"})
        count+=len(metadata);parts.append({"path":str(path.relative_to(R)),"bytes":path.stat().st_size,"count":len(metadata)})
        if image_index%25==0:atomic_json(root/"STATUS.json",{"images":image_index+1,"targets":count})
    expected=report["bbox_only"]["overall"]
    if count!=expected["tp"]:raise ValueError("Matched proposal coverage mismatch")
    atomic_json(root/"MAPS_COMPLETE.json",{"splits":{"eval":{"count":count,"parts":parts}},
        "predicted_maps_only":True,"frozen_surface_step":11300,"backbone_step":6300,
        "evaluation_contract":{"mode":"fixed_detector_predicted_ROI","total_GT":expected["tp"]+expected["fn"],
            "fixed_B0_missed_GT":expected["fn"],"fixed_B0_false_positives":expected["fp"],
            "fixed_B0_prediction_count":expected["tp"]+expected["fp"],
            "B0_step":3220,"B0_config_id":report["identity"]["B0_config_id"],
            "scope":"matched proposal pose scores; misses/FP must remain in joint denominators, never relabel GTROI"},
        "no_test_data":True,"data_modified":False})
if __name__=="__main__":run()
