"""Bridge existing same-ROI training metadata to loss-only corner224 targets."""
import math
import torch
from data import DataAdapter
from model.action4_corner_supervision import build_corner_targets,LOSS_WEIGHTS

SCHEMA="action4.training_corner_gt224.v1"


def _gt_id(value):
    if type(value) not in (int,str) or not str(value).strip():
        raise ValueError("Explicit scalar GT-unit identity required")
    return str(value)


def _verify_target_identity(item,meta,row):
    from model.action4_sam_feature_adapter import _identities
    identity=_identities((item["identity"],),1)[0]
    image=row.get("image_id")
    if (not isinstance(image,str) or not image or image!=meta.get("sample_id")
            or image!=identity.image_id):
        raise ValueError("Corner GT packet/image/input identity mismatch")
    gt=_gt_id(row.get("gt_instance_id"))
    if gt!=_gt_id(meta.get("instance_id")):
        raise ValueError("Corner GT packet and ROI unit identity mismatch")
    owner=item.get("sam_input")
    if (not isinstance(owner,dict) or not {"training_gt_instance_id","prediction_id","association"}<=set(owner)
            or gt!=_gt_id(owner.get("training_gt_instance_id"))):
        raise ValueError("Corner GT target lacks matching SAM supervision owner")
    association=owner.get("association")
    if (not isinstance(association,dict) or not {"image_id","gt_instance_id","status","prediction_id"}<=set(association)
            or association.get("image_id")!=image
            or _gt_id(association.get("gt_instance_id"))!=gt):
        raise ValueError("Corner GT target/association source identity mismatch")
    status=association.get("status")
    if status not in ("matched","missing","unknown","mixed","ambiguous"):
        raise ValueError("Explicit corner supervision association status required")
    if status=="matched":
        prediction=association.get("prediction_id")
        if (not isinstance(prediction,str) or not prediction or
                owner.get("prediction_id")!=prediction or identity.instance_id!=prediction):
            raise ValueError("Corner input prediction differs from fixed GT association")
    elif (association.get("prediction_id") is not None or owner.get("prediction_id") is not None
          or identity.instance_id!="training_unassociated:"+gt):
        raise ValueError("Unassociated corner unit must retain its explicit GT-owner identity")


def corner_targets_from_items(items,*,device="cpu",dtype=torch.float32):
    if not items:raise ValueError("Nonempty corner supervision batch required")
    rows=[];identities=[];known=[];ignore=[];identity_known=[];correct=[];evidence=[];active=[]
    conversion,_=DataAdapter.letterbox_affine((112,112),(224,224))
    for item in items:
        meta=item["metadata"];row=meta.get("corner_gt224")
        if not isinstance(row,dict) or row.get("schema")!=SCHEMA or row.get("uv_space")!="native_integer_centers":
            raise ValueError("Explicit native corner GT /224 transform packet required;112 UV cannot substitute")
        _verify_target_identity(item,meta,row)
        for name in ("target_identity_valid","gt_geometry_valid"):
            if type(meta.get(name)) is not bool:raise ValueError("Explicit training eligibility required: "+name)
        ik=meta.get("target_identity_known",True)
        if type(ik) is not bool or type(item["active"]) is not bool:
            raise ValueError("Explicit identity-known/active booleans required")
        support=meta.get("target_visible_input_support_count")
        if isinstance(support,bool) or not isinstance(support,(int,float)) or not math.isfinite(support) or support<0:
            raise ValueError("Actual same-ROI image support count required")
        A=torch.as_tensor(row["A224_from_native"],dtype=torch.float64)
        expected=torch.as_tensor(conversion,dtype=torch.float64)@torch.as_tensor(meta["affine_hm"],dtype=torch.float64)
        if A.shape!=(3,3) or not torch.allclose(A,expected,rtol=0,atol=1e-7):
            raise ValueError("Corner224 transform differs from actual ROI112 transform")
        if any(type(x) is not bool for x in row["gt_known"]+row["ignore"]) or len(row["gt_known"])!=8 or len(row["ignore"])!=8:
            raise ValueError("Explicit fixed8 annotation-known/ignore flags required")
        rows.append(row);identities.append(item["identity"])
        known.append([flag and meta["gt_geometry_valid"] for flag in row["gt_known"]])
        ignore.append(row["ignore"]);identity_known.append(ik);correct.append(meta["target_identity_valid"])
        evidence.append(support>0);active.append(item["active"])
    floating=lambda key:torch.as_tensor([row[key] for row in rows],device=device,dtype=dtype)
    boolean=lambda value:torch.tensor(value,device=device,dtype=torch.bool)
    return build_corner_targets(uv_native=floating("native_uv"),depth=floating("physical_depth_m"),
        A224_from_native=floating("A224_from_native"),gt_diagonal224=floating("full_box_diagonal224"),
        gt_known=boolean(known),ignore=boolean(ignore),identity_known=boolean(identity_known),
        identity_correct=boolean(correct),image_evidence=boolean(evidence),active=boolean(active),identities=tuple(identities))


def corner_target_counts(targets):
    valid=int(targets.valid.sum());known=int(targets.known.sum())
    return {key:(known if key in ("corner_valid_bce","corner_quality_bce") else valid) for key in LOSS_WEIGHTS}
