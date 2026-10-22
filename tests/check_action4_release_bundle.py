"""Actual inference parity and fresh-optimizer updates from sealed stage weights."""
import argparse,json,torch,numpy as np
from data.action4_paths import PROJECT_ROOT as R
from model.action4_release_bundle import warm_start,load_bundle
from eval.action4_staged_pipeline import StagedPipeline,load_pipeline
from data.action4_component_data import ComponentData
from model.training.action4_runtime import atomic_json,save_checkpoint_atomic
from model.training.action4_online_b0_batch import image_loss
from model.action4_surface_head import surface_losses
from model.action4_gdr_pose_losses_v002 import pose_losses as pose_losses_v002
from data.action4_gdr_pose_data import geometry

def run(selection,bundles,output):
    torch.set_num_threads(2);torch.manual_seed(17)
    cfg=json.loads((R/selection).read_text());out=R/output;out.mkdir(exist_ok=False)
    old=load_pipeline(cfg);sealed=StagedPipeline.from_bundles(bundles+"/A0/best.pt",bundles+"/B0/best.pt")
    data=ComponentData("outputs/required/DATA_VALIDATED_ALL_SPLITS.json","eval","B0")
    cad=np.array(json.loads((R/"data/DJI_Action4_construct/r3_v001/asset_manifest.json").read_text())["corners_3d_mm"])/1000
    parity=[]
    for i in [0,1]:
        item=data[i];K=np.array(item["record"]["camera"]["K"])
        a=old.predict(item["image"],K,cad);b=sealed.predict(item["image"],K,cad)
        assert json.dumps(a,sort_keys=True)==json.dumps(b,sort_keys=True),"Bundled inference changed"
        parity.append(item["record"]["sample_id"])
    del old,sealed;torch.cuda.empty_cache()
    proofs={}
    for role in ["A0","B0"]:
        model,base,opt,state=warm_start(bundles+"/"+role+"/best.pt",role,learning_rate=1e-6)
        assert len(opt.state)==0
        if role=="B0":
            item=ComponentData("outputs/required/DATA_VALIDATED_ALL_SPLITS.json","train","B0")[0]
            losses=image_loss(model,item,state["config"],"cuda:0")
        else:
            cache=R/state["config"]["cache_output"]
            manifest=json.loads((cache/"CACHE_COMPLETE.json").read_text())
            part=torch.load(R/manifest["splits"]["train"]["parts"][0]["path"],map_location="cpu",weights_only=True)
            x=part["features"][:2].cuda();y={k:v[:2].cuda() for k,v in part["targets"].items()}
            if state["architecture"]=="surface":
                losses=surface_losses(model(x),y)
            else:
                geo={k:v.cuda() for k,v in geometry(part["metadata"][:2],"train").items()}
                losses=pose_losses_v002(model(x,geo["K"],geo["canonical"]),y,geo)
            losses={k:state["config"]["loss_weights"][k]*v for k,v in losses.items()}
        loss=sum(losses.values());assert torch.isfinite(loss)
        parameter=next(model.parameters());before=parameter.detach().clone()
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        assert not torch.equal(before,parameter)
        test_path=out/(role+"_updated_test.pt")
        save_checkpoint_atomic(test_path,{"model":model.state_dict(),"optimizer":opt.state_dict(),"step":1,"scope":"engineering update only; never candidate selection"})
        reloaded=torch.load(test_path,map_location="cpu",weights_only=True)
        assert all(torch.equal(v.cpu(),reloaded["model"][k]) for k,v in model.state_dict().items())
        proofs[role]={"loss":float(loss.detach()),"gradient_norm":float(norm),"parameter_changed":True,"reload_equal":True,"optimizer_started_fresh":True,"source_step":state["selected_step"]}
        test_path.unlink()
        del model,base,opt,state;torch.cuda.empty_cache()
    atomic_json(out/"REPORT.json",{"passed":True,"prediction_parity_sample_ids":parity,"warm_start_updates":proofs,"scope":"packaging/continuation engineering; not model accuracy"})

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--selection",required=True);p.add_argument("--bundles",required=True);p.add_argument("--output",required=True);a=p.parse_args();run(a.selection,a.bundles,a.output)
