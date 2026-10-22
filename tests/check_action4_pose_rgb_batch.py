"""Actual RGB crop jitter updates K/targets exactly and matches inference ROI sampling."""
import json,os,torch,numpy as np
from data.action4_paths import PROJECT_ROOT as R
from data.action4_component_data import ComponentData
from data.action4_dataset import prepare_rois
from data.action4_pose_rgb_batch import prepare_pose_rgb

torch.set_num_threads(1)
data=ComponentData("outputs/required/DATA_VALIDATED_ALL_SPLITS.json","train","A0")
items=[data[0],data[1]];before=json.dumps([x["record"] for x in items],sort_keys=True)
plain,g0,m0=prepare_pose_rgb(items)
assert all(x["crop_mode"]=="GT" for x in m0)
jitter,g,m=prepare_pose_rgb(items,seeds=[31,47])
repeat,_,_=prepare_pose_rgb(items,seeds=[31,47])
assert all(x["crop_mode"]=="jitter" for x in m)
assert torch.equal(jitter,repeat) and not torch.equal(plain,jitter)
assert before==json.dumps([x["record"] for x in items],sort_keys=True)
assert torch.equal(g["rotation"],g0["rotation"]) and torch.equal(g["translation"],g0["translation"])
assert torch.equal(g["visible"],g0["visible"])
camera=torch.bmm(g["canonical"],g["rotation"].transpose(1,2))+g["translation"][:,None]
hom=torch.bmm(camera,g["K"].transpose(1,2));projected=hom[:,:,:2]/hom[:,:,2:]
error=float((projected-g["corners"]).abs().max());assert error<.001
assert (g["corners"]>-.5).all() and (g["corners"]<111.5).all()
# Supplied predicted boxes must use the exact production prepare_rois function.
boxes=[row["crop_box"] for row in m]
explicit,_,_=prepare_pose_rgb(items,boxes=boxes)
expected,_=prepare_rois([x["image"] for x in items],boxes,[0,1],roi_size=448,expansion=2.25)
assert torch.equal(explicit,expected) and torch.equal(explicit,jitter)
report={"passed":True,"actual_targets":2,"projected_corner_max_error_px":error,"rotation_translation_visibility_unchanged":True,
        "record_data_unchanged":True,"seed_repeat_exact":True,"production_ROI_sampling_exact":True,
        "scope":"data-contract engineering check; no new accuracy claim"}
out=R/os.environ.get("ACTION4_POSE_RGB_CHECK_OUTPUT","outputs/action4_a0_roi_contract_v001");assert out.resolve().is_relative_to(R);out.mkdir(exist_ok=False)
(out/"DATAFLOW_CHECK.json").write_text(json.dumps(report,indent=2));print(json.dumps(report))
