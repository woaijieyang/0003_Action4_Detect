"""Same-prediction comparison of training quality labels, scoring and decoding."""
import json,hashlib,concurrent.futures
from pathlib import Path
import numpy as np
import torch
from data import DataAdapter
from data.action4_dataset import virtual_roi,is_ignored
from data.action4_sam_roi_inputs import sample_native_nearest
from model.action4_corner_supervision import build_corner_targets
from model.action4_sam_feature_adapter import FeatureIdentity
from eval.action4_corner_scoring_inputs import _mask

def run(root,eval_dir,output):
    root=Path(root).resolve();folder=(root/eval_dir).resolve()
    if not folder.is_relative_to(root):raise ValueError("Audit scope")
    out=root/output;out.mkdir(exist_ok=False)
    scoring=json.loads((folder/"scoring/SCORING_INPUTS.json").read_text())
    artifact=json.loads((folder/"PREDICTION_ARTIFACT.json").read_text())
    annotations={r["sample_id"]:r for r in map(json.loads,(root/"data/phrase2/eval/annotations.jsonl").open())}
    observations={(r["image_id"],o["prediction_id"]):o for r in artifact["results"] for o in r["corner_observations"]}
    records=scoring["records"]
    if len(records)!=1300 or len(artifact["results"])!=650:raise ValueError("Full dev domain required")
    down,_=DataAdapter.letterbox_affine((448,448),(224,224))
    def prepare(s):
        r=annotations[s["image_id"]];g=next(g for g in r["instances"] if str(g["instance_id"])==s["gt_instance_id"])
        o=observations[(s["image_id"],s["prediction_id"])]
        A448=DataAdapter.crop_affine(virtual_roi(o["predicted_full_box"],1.5),(448,448))
        A224=down@A448
        native=np.asarray(g["corners_px"],float);T=np.asarray(g["canonical8_geometry"]["object_to_capture_cv_m"],float)
        cad=np.asarray(r["corners_3d_mm"],float)/1000
        depth=(cad@T[:3,:3].T+T[:3,3])[:,2]
        bounds=DataAdapter.transform_points(np.asarray(g["bbox_full_xyxy"]).reshape(2,2),A224)
        support=sample_native_nearest(_mask(g,r),A448)
        return dict(native=native,depth=depth,A=A224,diag=np.linalg.norm(bounds[1]-bounds[0]),
                    support=bool(support.any()),known=g.get("corner_annotation_known",[True]*8),ignore=is_ignored(g),
                    identity=FeatureIdentity(*o["roi_identity"]),o=o,visible=g["corner_visible"],
                    high_occlusion=g["occlusion_inframe"]>=.4,
                    affine_difference=float(np.max(abs(A224-np.asarray(o["A224_from_native"])))))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:prepared=list(pool.map(prepare,records))
    floating=lambda key:torch.tensor(np.array([r[key] for r in prepared]),dtype=torch.float32)
    boolean=lambda values:torch.tensor(values,dtype=torch.bool)
    n=len(prepared)
    targets=build_corner_targets(uv_native=floating("native"),depth=floating("depth"),A224_from_native=floating("A"),
        gt_diagonal224=floating("diag"),gt_known=boolean([r["known"] for r in prepared]),
        ignore=boolean([[r["ignore"]]*8 for r in prepared]),identity_known=torch.ones(n,dtype=torch.bool),
        identity_correct=torch.ones(n,dtype=torch.bool),image_evidence=boolean([r["support"] for r in prepared]),
        active=torch.ones(n,dtype=torch.bool),identities=tuple(r["identity"] for r in prepared))
    uv=torch.tensor(np.array([r["uv224"] for r in records]),dtype=torch.float32)
    # Exact quality-target expression used by the immutable training loss.
    training_error=torch.linalg.vector_norm(uv-targets.uv224,dim=-1)
    train_label=(targets.valid&(training_error<=.01*targets.diagonal224[:,None])).numpy()
    known=np.array([r["label_known"] for r in records],bool);valid=np.array([r["gt_valid"] for r in records],bool)
    error=np.array([r["error224"] for r in records],float);diag=np.array([r["gt_diagonal224"] for r in records],float)[:,None]
    scoring_label=valid&(error<=.01*diag)
    q=np.array([r["q_corner"] for r in records]);pv=np.array([r["p_valid"] for r in records]);multi=np.array([r["multimodal"] for r in records])
    xy=np.array([r["uv224"] for r in records]);selected=q>=.6
    usable=selected&(pv>=.8)&~multi&((xy>=0)&(xy<=223)).all(-1)
    serialized_uv_equal=all(np.array_equal(r["uv224"],p["o"]["observations"]["uv224"]) for r,p in zip(records,prepared))
    diagnostics=[]
    for r,p,keep in zip(records,prepared,usable):
        d=p["o"]["decoder_diagnostics"]
        if d.get("used_corner_ids")!=np.flatnonzero(keep).tolist():
            diagnostics.append(dict(image_id=r["image_id"],prediction_id=r["prediction_id"],expected=np.flatnonzero(keep).tolist(),
                                    actual=d.get("used_corner_ids"),reason=d.get("reason")))
    def stats(mask):
        m=mask&known;c=int(m.sum())
        return dict(points=c,mean_q=float(q[m].mean()) if c else None,
            correct_at_1pct=int((m&scoring_label).sum()),precision_at_1pct=float(scoring_label[m].mean()) if c else None,
            correct_at_2pct=int((m&valid&(error<=.02*diag)).sum()),
            Brier=float(np.mean((q[m]-scoring_label[m])**2)) if c else None)
    bins=[]
    for lo,hi in zip([0,.1,.2,.3,.4,.5,.6,.7,.8,.9],[.1,.2,.3,.4,.5,.6,.7,.8,.9,1.000001]):
        bins.append(dict(lo=lo,hi=min(hi,1),**stats((q>=lo)&(q<hi))))
    visible=np.array([p["visible"] for p in prepared],bool)
    head_vs_pose=dict(raw=json.loads((folder/"RAW_REPORT.json").read_text())["pose_all8_at_002_success"],
                      candidate=json.loads((folder/"CANDIDATE_REPORT.json").read_text())["pose_all8_at_002_success"])
    result=dict(status="FULL650_SAME_OUTPUT_QUALITY_ALIGNMENT_AUDIT",images=650,instances=1300,points=10400,
        training_target_source=build_corner_targets.__code__.co_filename,
        affine_max_abs_difference=max(r["affine_difference"] for r in prepared),
        known_mask_differences=int((targets.known.numpy()!=known).sum()),
        valid_mask_differences=int((targets.valid.numpy()!=valid).sum()),
        train_vs_score_label_differences=int((train_label!=scoring_label).sum()),
        training_float32_vs_scoring_error_max_px=float(np.max(abs(training_error.numpy()[valid]-error[valid]))),
        scoring_uv_equals_exported_decoder_uv=serialized_uv_equal,
        decoder_membership_differences=diagnostics,
        all_known=stats(known),quality_selected=stats(selected),decoder_usable=stats(usable),
        quality_selected_but_decoder_rejected=stats(selected&~usable),
        selected_visible=stats(selected&visible),selected_hidden=stats(selected&~visible),
        bins=bins,pose_all8_at_002=head_vs_pose,
        model_forward_calls=0,training_modified=False,
        caveat="Same frozen prediction outputs and full GT masks; training label builder executed, no training images treated as eval. Does not establish train distribution equivalence or new video accuracy.")
    (out/"REPORT.json").write_text(json.dumps(result,indent=2,allow_nan=False))
    return result
