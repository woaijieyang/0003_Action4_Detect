"""Actual accepted train targets, crop reprojection, and surface/hidden-point gradient check."""
import json,torch,numpy as np
from pathlib import Path
from data.action4_component_data import ComponentData
from data.action4_online_surface_targets import prepare_online_surface_batch
from data.action4_surface_data import prepare_surface_batch
from model.action4_sparse_surface_losses import sparse_surface_losses
from model.action4_pose_maps import geometry_losses
from model.action4_release_bundle import load_bundle
from model.training.action4_runtime import atomic_json


def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    out=Path("outputs/action4_online_surface_contract_v001");out.mkdir(exist_ok=False)
    data=ComponentData("outputs/required/DATA_VALIDATED_ALL_SPLITS.json","train","A0")
    chosen={}
    for idx,(image_index,instance_index) in enumerate(data.units):
        g=data.records[image_index]["instances"][instance_index];T=np.array(g["canonical8_geometry"]["object_to_capture_cv_m"]);v=-T[:3,:3].T@T[:3,3];a=int(np.argmax(abs(v)));face=("+" if v[a]>=0 else "-")+"XYZ"[a]
        if g["occlusion_inframe"]>=.4:chosen.setdefault("heavy",idx)
        else:chosen.setdefault("light",idx)
        if face in ("-Y","-Z"):chosen.setdefault(face,idx)
        if len(chosen)==4:break
    indices=list(dict.fromkeys(chosen.values()));items=[data[i] for i in indices]
    original,dense=prepare_surface_batch(items,{"roi_size":448,"roi_expansion":2.25},"train")
    baseline,sparse,_,base_meta=prepare_online_surface_batch(items,"train")
    assert torch.equal(original,baseline)
    mask_changes=int((dense["masks"]!=sparse["masks"]).sum())
    # Different floating affine paths can only be discussed if measured, never hidden.
    assert mask_changes==0
    for i,row in enumerate(sparse["sparse"]):
        xy=row["xy_hm"].round().long();x,y=xy.unbind(-1)
        assert torch.allclose(row["xy_hm"],xy.float(),atol=1e-4,rtol=0)
        assert torch.allclose(row["xyz"],dense["xyz"][i,:,y,x].T,atol=1e-6,rtol=0)
        assert torch.equal(row["regions"],dense["regions"][i,y,x])
    rois,targets,geometry,metadata=prepare_online_surface_batch(items,"train",seeds=[117+i for i in range(len(items))])
    assert all(m["crop_mode"]=="jitter" for m in metadata)
    assert all(m["surface_projection_error_hm_px"]<.005 for m in metadata)
    assert all(m["dropped_surface_points"]==0 for m in metadata)
    assert not torch.equal(rois,baseline)
    targets["masks"]=targets["masks"].cuda();targets["sparse"]=[{k:v.cuda() for k,v in row.items()} for row in targets["sparse"]]
    geometry={k:v.cuda() for k,v in geometry.items()}
    model,base,state=load_bundle("models/candidates/a0_online_rgb_v001/best.pt","A0","cuda:0")
    model.train();base.eval()
    with torch.no_grad():features=base.backbone(rois.cuda())
    output=model(features,geometry["K"],geometry["canonical"])
    losses=sparse_surface_losses(output["surface"],targets);geo=geometry_losses(output,geometry)
    hidden_grad=torch.autograd.grad(geo["hidden_corners"],tuple(output["surface"][k] for k in ["xyz","regions","masks"]),retain_graph=True)
    hidden_norms={k:float(g.norm()) for k,g in zip(["xyz","regions","masks"],hidden_grad)}
    assert all(np.isfinite(x) and x>0 for x in hidden_norms.values())
    loss=sum(losses.values())+sum(geo.values());assert torch.isfinite(loss);loss.backward()
    norms={name:float(torch.stack([p.grad.norm() for p in module.parameters() if p.grad is not None]).norm()) for name,module in [("surface",model.surface),("pose",model.pose)]}
    assert all(np.isfinite(x) and x>0 for x in norms.values())
    assert all(p.grad is None for p in base.parameters())
    report={"scope":"actual target and backward engineering check; no optimizer update or new training","selected_train_indices":indices,
        "objects":[{"sample_id":m["sample_id"],"instance_id":m["instance_id"],"points":m["surface_points"],"projection_error_hm_px":m["surface_projection_error_hm_px"],"dropped":m["dropped_surface_points"]} for m in metadata],
        "baseline_RGB_equal":True,"baseline_mask_differences":mask_changes,"XYZ_and_region_target_equivalence":True,
        "surface_losses":{k:float(v.detach()) for k,v in losses.items()},"parameter_gradient_norms":norms,"hidden_loss_gradient_to_predicted_surface":hidden_norms,
        "backbone_frozen":True,"source_best_step":state["selected_step"],"model_weights_changed":False,"data_files_changed":False}
    atomic_json(out/"REPORT.json",report);print(json.dumps(report))
if __name__=="__main__":main()
