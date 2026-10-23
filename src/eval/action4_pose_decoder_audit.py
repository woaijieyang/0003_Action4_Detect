"""One RGB pass: compare fixed network/PnP decoders and preserve exact pose-input batches."""
import argparse,copy,json,os,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from data import DataAdapter
from data.action4_paths import PROJECT_ROOT as R,contained_path
from data.action4_component_data import ComponentData
from data.action4_dataset import virtual_roi
from model.action4_pose_maps import PoseFromMaps,surface_to_maps
from model.action4_pose_refinement import refine_network_pose
from model.action4_surface_pnp import decode_surface_pose
from model.training.action4_runtime import atomic_json,save_checkpoint_atomic
from eval.action4_staged_pipeline import load_pipeline
from eval.action4_pose_reference_metrics import compare_pose_envelopes

METHODS=("network","iterative","surface_pnp")
@torch.inference_mode()
def run(output,mode):
    out=contained_path(R,output,exists=False);out.mkdir(exist_ok=False)
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    profile=json.loads((R/"configs/eval/action4_a0_online_rgb_best_v001.json").read_text())
    pipeline=load_pipeline(profile);captured={}
    def capture(module,args,result):
        captured.update(result=result,K=args[1],canonical=args[2])
    hook=pipeline.a0.register_forward_hook(capture)
    replay=PoseFromMaps().cuda().eval();replay.pose.load_state_dict(pipeline.a0.pose.state_dict(),strict=True)
    cad=np.array(json.loads((R/"data/DJI_Action4_construct/r3_v001/asset_manifest.json").read_text())["corners_3d_mm"])/1000
    if mode=="development":
        data=ComponentData("outputs/required/DATA_VALIDATED_ALL_SPLITS.json","eval","B0")
        baseline=json.loads((R/"outputs/action4_a0_online_rgb_validation_v001/development/REPORT.json").read_text())
        matched={r["sample_id"]:r["matches"] for r in baseline["bbox_only"]["per_image"]}
        expected={r["sample_id"]:r["instances"] for r in json.loads((R/"outputs/action4_a0_online_rgb_validation_v001/development/PREDICTIONS.json").read_text())}
        ids=list(range(len(data)));(out/"maps").mkdir()
    else:
        reference=json.loads((R/"outputs/action4_full_demo_reference_v001/REFERENCE.json").read_text())
        eligible=[r["frame_id"] for r in reference["rows"] if r["reference_candidate"]]
        ids=[eligible[i] for i in np.linspace(0,len(eligible)-1,60,dtype=int)]
        manifest=json.loads((R/"data/action4_real/cache_v001/frames_manifest.json").read_text())
        expected={x["frame_id"]:x["prediction"]["instances"] for x in map(json.loads,(R/"outputs/action4_a0_online_rgb_validation_v001/video/predictions.jsonl").read_text().splitlines())}
        K=np.array(json.loads(contained_path(R,profile["intrinsics_context"],exists=True).read_text())["K_native"])
    atomic_json(out/"SELECTION.json",{"mode":mode,"ids":ids,"methods":METHODS,"identity":pipeline.identity,"weights_fixed":True,"thresholds_fixed":True,"no_test_selection":True,"production_changed":False})
    rows={m:[] for m in METHODS};preds={m:[] for m in METHODS};parts=[];max_replay=0.;max_saved=0.;start=time.monotonic()
    for seq,i in enumerate(ids):
        if mode=="development":
            item=data[i];record=item["record"];rgb=item["image"];K=np.array(record["camera"]["K"]);sid=record["sample_id"]
        else:
            sid=i
            with Image.open(contained_path(R,manifest["frames"][i]["image"],exists=True)) as im:rgb=torch.from_numpy(np.array(im.convert("RGB"))).permute(2,0,1)
        captured.clear();result=pipeline.predict(rgb,K,cad);instances=result["instances"]
        if len(instances)!=len(expected[sid]):raise ValueError("Baseline candidate count changed")
        variants={m:copy.deepcopy(instances) for m in METHODS}
        down,_=DataAdapter.letterbox_affine((448,448),(112,112))
        transforms=[down@DataAdapter.crop_affine(virtual_roi(p["bbox_full_xyxy"],2.25),(448,448)) for p in instances]
        if instances:
            prediction=captured["result"];km=captured["K"];canonical=captured["canonical"]
            maps=surface_to_maps(prediction["surface"],km)
            # A CPU round trip preserves float32 values; retain original per-image batch shape.
            replayed=replay(maps.cpu().cuda(),km.cpu().cuda(),canonical.cpu().cuda())["corners"]
            max_replay=max(max_replay,float((replayed-prediction["corners"]).abs().max()))
            for j,p in enumerate(instances):
                if p["bbox_full_xyxy"]!=expected[sid][j]["bbox_full_xyxy"]:raise ValueError("B0 boxes changed")
                max_saved=max(max_saved,float(np.max(np.abs(np.array(p["corners_px"])-expected[sid][j]["corners_px"]))))
                xyz=prediction["surface"]["xyz"][j].cpu().numpy()
                mask=prediction["surface"]["masks"][j,0].sigmoid().cpu().numpy()
                diag={"iterative":refine_network_pose(xyz,mask,transforms[j]@K,cad,prediction["rotation"][j].cpu().numpy(),prediction["translation"][j].cpu().numpy()),
                      "surface_pnp":decode_surface_pose(xyz,mask,transforms[j]@K,cad)}
                for method,d in diag.items():
                    variants[method][j].update(pose_valid=d["valid"],pose_diagnostic=d,corners_px=DataAdapter.transform_points(d["corners_hm"],np.linalg.inv(transforms[j])).tolist() if d["valid"] else None)
        if mode=="development":
            gt=[g for g in record["instances"] if not g.get("ignore",False)]
            labels=[]
            for pi,gi in matched[sid]:
                g=gt[gi];diagonal=float(np.linalg.norm(np.array(g["bbox_full_xyxy"])[2:]-g["bbox_full_xyxy"][:2]))
                labels.append({"prediction_index":pi,"instance_id":g["instance_id"],"corners_px":g["corners_px"],"diagonal":diagonal,"heavy":g["occlusion_inframe"]>=.4})
                for method in METHODS:
                    p=variants[method][pi]
                    errors=(np.linalg.norm(np.array(p["corners_px"])-g["corners_px"],axis=1)/diagonal).tolist() if p["pose_valid"] else None
                    rows[method].append({"sample_id":sid,"instance_id":g["instance_id"],"heavy":g["occlusion_inframe"]>=.4,"errors":errors,"all8":errors is not None and max(errors)<=.05})
            if instances:
                path=out/"maps"/f"image_{i:04d}.pt"
                save_checkpoint_atomic(path,{"inputs":{"maps":maps.cpu(),"K":km.cpu(),"canonical":canonical.cpu()},
                     "inverse_affines":np.linalg.inv(np.stack(transforms)).tolist(),"labels_for_scoring_only":labels,
                     "sample_id":sid,"expected_native_corners":[p["corners_px"] for p in instances]})
                parts.append({"path":str(path.relative_to(R)),"predictions":len(instances),"matched":len(labels)})
        for method in METHODS:
            preds[method].append({"frame_id":i,"pts":i*400,"sample_id":sid,"prediction":{"instances":variants[method]}})
        if seq%20==0:atomic_json(out/"STATUS.json",{"completed":seq+1,"total":len(ids),"seconds":time.monotonic()-start})
    hook.remove()
    if max_saved>=.001 or max_replay>=.001:raise ValueError(f"Frozen baseline/replay mismatch {max_saved=} {max_replay=}")
    summary={}
    if mode=="development":
        for method in METHODS:
            summary[method]={"all8":sum(x["all8"] for x in rows[method]),"GT":baseline["corner_GT"],"heavy_all8":sum(x["all8"] and x["heavy"] for x in rows[method]),"heavy_GT":baseline["heavy_GT"],"rows":rows[method]}
        atomic_json(out/"EXACT_CACHE.json",{"schema":"action4.image_grouped_pose_cache.v1","identity":pipeline.identity,"parts":parts,"same_GPU_crop_and_network_as_deployment":True,"preserve_image_batch_shape":True,"total_GT":baseline["corner_GT"],"heavy_GT":baseline["heavy_GT"],"prediction_count":sum(p["predictions"] for p in parts),"missed":baseline["box_missed_targets"],"max_replay_heatmap_difference":max_replay,"no_test_data":True})
    else:
        chosen=set(ids);ref=copy.deepcopy(reference)
        for x in ref["rows"]:x["reference_candidate"]=x["reference_candidate"] and x["frame_id"] in chosen
        for method in METHODS:
            sparse={x["frame_id"]:x for x in preds[method]}
            full=[sparse.get(i,{"frame_id":i,"prediction":{"instances":[]}}) for i in range(2850)]
            summary[method]=compare_pose_envelopes(ref,full)
            summary[method]["outside_probe_frames"]=summary[method].pop("unscorable_reference_frames")
            summary[method]["original_reference_unscorable_frames"]=77
            summary[method]["scope"]="prespecified60 real demo-envelope diagnostic, not whole-video accuracy; other2790 not evaluated"
    for method in METHODS:
        atomic_json(out/(method+"_REPORT.json"),summary[method])
        with (out/(method+"_predictions.jsonl")).open("x") as f:
            for row in preds[method]:f.write(json.dumps(row,allow_nan=False)+"\n")
    atomic_json(out/"COMPLETE.json",{"identity":pipeline.identity,"mode":mode,"count":len(ids),"max_native_difference_vs_saved_baseline":max_saved,"max_replay_heatmap_difference":max_replay,"summary":{m:{k:v for k,v in summary[m].items() if k!="rows"} for m in METHODS},"accuracy_accepted":False})
    print(json.dumps(json.loads((out/"COMPLETE.json").read_text())))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",required=True);p.add_argument("--mode",choices=["development","real60"],required=True);a=p.parse_args();run(a.output,a.mode)
