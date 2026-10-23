"""Fixed-weight real surface evidence: internal fit versus external demo envelope."""
import json,copy,torch,numpy as np
from PIL import Image,ImageDraw
from data.action4_paths import PROJECT_ROOT as R,contained_path
from eval.action4_staged_pipeline import load_pipeline
from eval.action4_metrics import bbox_iou,maximum_matching
from model.action4_pose_refinement import refine_network_pose
from model.training.action4_runtime import atomic_json

@torch.inference_mode()
def run():
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    out=R/"outputs/action4_surface_domain_audit_v001";out.mkdir(exist_ok=False)
    cfg=json.loads((R/"configs/eval/action4_a0_online_rgb_best_v001.json").read_text());pipeline=load_pipeline(cfg)
    manifest=json.loads((R/"data/action4_real/cache_v001/frames_manifest.json").read_text())
    reference=json.loads((R/"outputs/action4_full_demo_reference_v001/REFERENCE.json").read_text())
    chosen=json.loads((R/"outputs/action4_pose_decoder_audit_v001/real60/SELECTION.json").read_text())["ids"]
    K=np.array(json.loads(contained_path(R,cfg["intrinsics_context"],exists=True).read_text())["K_native"])
    cad=np.array(json.loads((R/"data/DJI_Action4_construct/r3_v001/asset_manifest.json").read_text())["corners_3d_mm"])/1000
    captured={}
    def rgb_hook(module,args,result):captured["rgb"]=args[0].detach().cpu()
    def pose_hook(module,args,result):captured.update(result=result,K=args[1])
    h1=pipeline.base.backbone.register_forward_hook(rgb_hook);h2=pipeline.a0.register_forward_hook(pose_hook)
    scale=np.array([950/3248,720/2464]*2);rows=[];pictures=[]
    edges=[(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for index,fid in enumerate(chosen):
        frame=manifest["frames"][fid]
        with Image.open(contained_path(R,frame["image"],exists=True)) as im:rgb=torch.from_numpy(np.array(im.convert("RGB"))).permute(2,0,1)
        captured.clear();result=pipeline.predict(rgb,K,cad);ps=result["instances"];refs=reference["rows"][fid]["components"]
        matches=dict(maximum_matching([[j for j,g in enumerate(refs) if bbox_iou(np.array(p["bbox_full_xyxy"])*scale,g["bbox_demo_px"])>=.5] for p in ps],2))
        if not ps:continue
        output=captured["result"]
        for j,p in enumerate(ps):
            xyz=output["surface"]["xyz"][j].cpu().numpy();prob=output["surface"]["masks"][j,0].sigmoid().cpu().numpy()
            km=captured["K"][j].cpu().numpy();rot=output["rotation"][j].cpu().numpy();t=output["translation"][j].cpu().numpy()
            select=(prob>=.5)&np.isfinite(xyz).all(0)&(xyz.min(0)>=-.05)&(xyz.max(0)<=1.05)
            yy,xx=np.where(select);points=xyz[:,yy,xx].T*(cad.max(0)-cad.min(0))+cad.min(0);pixels=np.c_[xx,yy]
            refined=refine_network_pose(xyz,prob,km,cad,rot,t)
            def fit(rotation,translation):
                camera=points@np.asarray(rotation).T+np.asarray(translation);h=camera@km.T
                if not len(h):return None
                error=np.linalg.norm(h[:,:2]/h[:,2:]-pixels,axis=1)
                return {"median":float(np.median(error)),"mean":float(np.mean(error)),"positive_depth_fraction":float(np.mean(camera[:,2]>0))}
            fits={"network":fit(rot,t),"iterative":fit(refined["rotation"],refined["translation_m"]) if refined["valid"] else None}
            boxes={}
            for method,uv in [("network",output["corners"][j].cpu().numpy()),("iterative",np.array(refined["corners_hm"]) if refined["valid"] else None)]:
                if uv is None:boxes[method]=None;continue
                # Native K and crop K share pixel centres; use the exact affine implied by them for diagnostic geometry.
                A=np.asarray(km,dtype=float)@np.linalg.inv(K);hom=np.c_[uv,np.ones(8)]@np.linalg.inv(A).T;native=hom[:,:2]/hom[:,2:]
                box=np.r_[native.min(0),native.max(0)]*scale
                boxes[method]=float(bbox_iou(box,refs[matches[j]]["bbox_demo_px"])) if j in matches else None
            row={"frame_id":fid,"prediction_index":j,"B0_matches_reference":j in matches,"predicted_mask_pixels":int((prob>=.5).sum()),"usable_correspondences":len(points),"internal_reprojection_hm_px":fits,"reference_envelope_IoU":boxes,"refined":refined["refined"]}
            rows.append(row)
            if index%5==0:
                raw=captured["rgb"][j].numpy()*np.array([.229,.224,.225])[:,None,None]+np.array([.485,.456,.406])[:,None,None]
                raw=Image.fromarray(np.uint8(np.clip(raw.transpose(1,2,0),0,1)*255)).resize((224,224))
                mask=Image.fromarray(np.uint8(prob*255)).resize((224,224));overlay=raw.copy();green=Image.new("RGB",raw.size,(0,255,0));overlay=Image.composite(green,overlay,mask.point(lambda x:int(x*.55)))
                colors=np.uint8(np.clip(xyz.transpose(1,2,0),0,1)*255);colors[prob<.5]=0;xyz_image=Image.fromarray(colors).resize((224,224))
                drawpose=raw.copy();draw=ImageDraw.Draw(drawpose)
                for uv,color in [(output["corners"][j].cpu().numpy(),"red"),(np.array(refined["corners_hm"]) if refined["valid"] else None,"cyan")]:
                    if uv is None:continue
                    uv=uv*2+.5
                    for a,b in edges:draw.line([tuple(uv[a]),tuple(uv[b])],fill=color,width=2)
                panel=Image.new("RGB",(896,250));labels=[f"frame{fid} target{j} RGB","predicted visible mask","predicted XYZ (not GT)","red network / cyan iterative"]
                for c,(im,label) in enumerate(zip([raw,overlay,xyz_image,drawpose],labels)):panel.paste(im,(224*c,26));ImageDraw.Draw(panel).text((224*c+2,6),label,fill="white")
                pictures.append(panel)
        if index%20==0:atomic_json(out/"STATUS.json",{"frames":index+1,"total":60})
    h1.remove();h2.remove()
    for page,start in enumerate(range(0,len(pictures),4)):
        sheet=Image.new("RGB",(896,250*len(pictures[start:start+4])))
        for j,im in enumerate(pictures[start:start+4]):sheet.paste(im,(0,j*250))
        sheet.save(out/f"review_{page:02d}.jpg")
    comparable=[r for r in rows if r["B0_matches_reference"] and all(r["internal_reprojection_hm_px"][m] is not None for m in ["network","iterative"]) and all(r["reference_envelope_IoU"][m] is not None for m in ["network","iterative"])]
    fit_better=[r for r in comparable if r["internal_reprojection_hm_px"]["iterative"]["mean"]<r["internal_reprojection_hm_px"]["network"]["mean"]]
    summary={"frames":60,"predictions":len(rows),"matched_reference_comparable":len(comparable),"internal_mean_fit_improved":len(fit_better),"fit_improved_but_external_IoU_worse":sum(r["reference_envelope_IoU"]["iterative"]<r["reference_envelope_IoU"]["network"] for r in fit_better),"network_pass_iterative_fail":sum(r["reference_envelope_IoU"]["network"]>=.5>r["reference_envelope_IoU"]["iterative"] for r in comparable),"scope":"predicted surface self-consistency vs uncertified demo envelopes; no real surface GT"}
    atomic_json(out/"REPORT.json",{"summary":summary,"rows":rows});print(json.dumps(summary))
if __name__=="__main__":run()
