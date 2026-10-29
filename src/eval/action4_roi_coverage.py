"""GT-only crop diagnostics after frozen inference; never changes ROI decisions."""
import json
import numpy as np
from PIL import Image
from data import DataAdapter
from data.action4_paths import PROJECT_ROOT as R,contained_path
from data.action4_dataset import virtual_roi,is_ignored
from data.action4_online_surface_targets import _sample_binary_mask_image
from eval.action4_pose_protocol import match_geometry,gt_groups
from eval.action4_roi_sweep import read_binding,RATIOS

def diagnose_part(input_lock,part,parts,output):
    lock=read_binding(input_lock)
    from data.action4_sam_native_export import _v2_read
    records=_v2_read(lock["image_eval"]["annotations"],lambda f:[json.loads(x) for x in f if x.strip()])
    boxes=read_binding(lock["image_eval"]["B0_candidates"]);byid={x["sample_id"]:x["instances"] for x in boxes}
    yy,xx=np.mgrid[:448,:448];grid448=np.c_[xx.ravel(),yy.ravel()]
    downs={n:DataAdapter.letterbox_affine((448,448),(n,n))[0] for n in (112,224,448)}
    rows=[]
    for record in records[part::parts]:
        sid=record["sample_id"];gs=[g for g in record["instances"] if not is_ignored(g)]
        bs=[x for x in byid[sid] if x["score"]>=lock["image_eval"]["score_threshold"]]
        pairs=match_geometry([np.array(x["bbox_full_xyxy"]) for x in bs],[np.array(g["bbox_full_xyxy"]) for g in gs],
            [x["grid_index"] for x in bs],[g["instance_id"] for g in gs])
        bygt={g:b for b,g in pairs}
        for gi,g in enumerate(gs):
            groups=gt_groups(record,g,gs);full=np.asarray(g["bbox_full_xyxy"])
            edge=bool((full[:2]<0).any() or full[2]>record["width"] or full[3]>record["height"])
            groups+=["image_edge_truncated" if edge else "inside_image"]
            if gi not in bygt:
                for e in RATIOS:rows.append(dict(sample_id=sid,instance_id=g["instance_id"],roi_expansion=e,groups=groups,status="B0_missing",visible_retention=None,surface_empty=None))
                continue
            box=bs[bygt[gi]]["bbox_full_xyxy"]
            path=contained_path(R,f"data/phrase2/eval/geometry/{sid}_i{g['instance_id']}.npz",exists=True)
            with np.load(path,allow_pickle=False) as surface:
                if not np.allclose(surface["K"],record["camera"]["K"],atol=1e-8,rtol=0) or not np.allclose(surface["object_to_capture"],g["canonical8_geometry"]["object_to_capture_cv_m"],atol=1e-8,rtol=0):raise ValueError("Surface identity differs")
                native=np.asarray(surface["native_pixels"])
            with Image.open(contained_path(R,g["mask_visible"],exists=True)) as mask:
                if mask.size!=(record["width"],record["height"]):raise ValueError("GT mask geometry changed")
                ys,xs=np.nonzero(np.asarray(mask)>0);total=len(xs)
                for e in RATIOS:
                    bounds=virtual_roi(box,e);A448=DataAdapter.crop_affine(bounds,(448,448));A112=downs[112]@A448
                    uv=DataAdapter.transform_points(native,A112);keep=np.isfinite(uv).all(1)&(uv>=0).all(1)&(uv<=111).all(1)
                    retained=int(((xs>=bounds[0])&(xs<bounds[2])&(ys>=bounds[1])&(ys<bounds[3])).sum())
                    pixels=DataAdapter.transform_points(grid448,np.linalg.inv(A448))
                    visible=int(_sample_binary_mask_image(mask,pixels,expected_wh=mask.size).sum())
                    outside={}
                    for n,A in downs.items():
                        corners=DataAdapter.transform_points(g["corners_px"],A@A448)
                        outside[str(n)]=int((~((corners>=0).all(1)&(corners<=n-1).all(1))).sum())
                    rows.append(dict(sample_id=sid,instance_id=g["instance_id"],roi_expansion=e,groups=groups,status="B0_matched",
                        B0_grid=bs[bygt[gi]]["grid_index"],roi_bounds=bounds,visible_native_pixels=total,
                        visible_retention=retained/total if total else None,retention_definition="native_GT_pixel_centres_inside_continuous_ROI",
                        visible_input_samples448=visible,target_occupancy448=visible/(448*448),
                        corner_outside_counts=outside,surface_points=int(keep.sum()),surface_empty=not bool(keep.any()),
                        no_target_evidence448=visible==0,GT_used_for_inference=False))
    out=contained_path(R,output,exists=False)
    with out.open("x") as f:json.dump({"part":part,"parts":parts,"rows":rows,"scope":"label diagnostics only; no inference changes"},f)
    return len(rows)
