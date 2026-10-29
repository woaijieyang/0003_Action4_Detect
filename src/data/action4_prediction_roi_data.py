"""Verified frozen-B0 training units, with no GT crop or jitter fallback."""
import copy
from .action4_sam_native_export import _v2_checked,_v2_read,canonical
from .action4_prediction_roi_manifest import associate,POLICY
PROTOCOL="action4.fixed_roi_prediction_adaptation.v1"
SAM_NATIVE_PROTOCOL="action4.sam_prediction_roi_enhancement.native_v1"
CORNER_NATIVE_PROTOCOL="action4.corner_prediction_roi_enhancement.native_v1"

FINETUNE_SCHEMA="action4.corner_roi115_finetune.v1"

def roi_finetune(cfg):
    value=cfg.get("roi_finetune")
    if value is None:return False
    if value!={"schema":FINETUNE_SCHEMA,"parent_roi_expansion":1.5,"roi_expansion":1.15,"epochs":75}:
        raise ValueError("Explicit registered ROI115 corner finetune contract required")
    if cfg.get("training_protocol")!=CORNER_NATIVE_PROTOCOL:
        raise ValueError("ROI115 finetune requires complete native corner protocol")
    return True

def enabled(cfg):return cfg.get("training_protocol")==PROTOCOL

def validate_config(cfg,split):
    if cfg.get("training_protocol") in (SAM_NATIVE_PROTOCOL,CORNER_NATIVE_PROTOCOL):
        corner=cfg["training_protocol"]==CORNER_NATIVE_PROTOCOL
        expansion=1.15 if roi_finetune(cfg) else 1.5
        if (cfg.get("role")!="A0" or split!="train" or cfg.get("mode") not in ("integration","full_training")
            or cfg.get("roi_source")!="frozen_b0_prediction" or cfg.get("roi_expansion")!=expansion
            or cfg.get("roi_size")!=448 or cfg.get("preprocess",{}).get("roi_expansion")!=expansion
            or cfg.get("train_backbone") is not False or cfg.get("sam_enabled") is not True
            or cfg.get("a0_variant")!=("a0_corner_image" if corner else "a0_sam_image")
            or cfg.get("corner_enabled",False) is not corner):
            raise ValueError("Explicit registered native SAM/corner prediction-ROI1.5 contract required")
        forbidden=("photometric","roi_jitter","sam_rgb_variants","sam_roi_routing","feature_cache","continuation_segment","sam_allow_synthetic_fixture")
        if any(cfg.get(k) not in (None,False) for k in forbidden):
            raise ValueError("Registered native SAM/corner protocol cannot mix unregistered variants, GT jitter, ROI routes, legacy caches or baseline extension")
        cache=cfg.get("sam_cache");b0=cfg.get("evaluation_counterpart_sha256")
        if not isinstance(b0,str) or len(b0)!=64 or any(c not in "0123456789abcdef" for c in b0):
            raise ValueError("Explicit frozen B0 SHA256 required")
        if (not isinstance(cfg.get("prediction_roi_manifest"),dict)
            or not isinstance(cfg.get("sam_training_associations"),dict)
            or not isinstance(cache,dict) or set(cache)!={"manifest","expected_sources"}
            or not isinstance(cache["expected_sources"],dict)
            or cache["expected_sources"].get("upstream_mode")!="b0_completed_native_receipt_v2"
            or cache["expected_sources"].get("b0_model_sha256")!=cfg.get("evaluation_counterpart_sha256")):
            raise ValueError("Bound same-B0 native SAM cache and prediction ownership required")
        if corner:
            if cfg.get("b0_scoring_policy") not in ("center","product","learned"):
                raise ValueError("Explicit corner frozen B0 scoring policy required")
            evaluation=cfg.get("sam_eval_cache");counterpart=cfg.get("corner_counterpart")
            if (not isinstance(evaluation,dict) or set(evaluation)!={"manifest","expected_sources"}
                or not isinstance(evaluation["expected_sources"],dict)
                or evaluation["expected_sources"].get("upstream_mode")!="b0_completed_native_receipt_v2"
                or evaluation["expected_sources"].get("b0_model_sha256")!=b0
                or not isinstance(counterpart,dict)
                or counterpart.get("train_sam_cache")!=cache["manifest"]
                or counterpart.get("eval_sam_cache")!=evaluation["manifest"]
                or counterpart.get("B0")!={"path":cfg.get("evaluation_counterpart_bundle"),"sha256":b0}):
                raise ValueError("Corner native train/eval SAM and B0 counterpart bindings differ")
            ref=cfg.get("frozen_eval_b0_predictions")
            if (not isinstance(ref,dict) or set(ref)!={"path","sha256"}
                or not isinstance(ref["path"],str) or not ref["path"]
                or not isinstance(ref["sha256"],str) or len(ref["sha256"])!=64
                or any(c not in "0123456789abcdef" for c in ref["sha256"])):
                raise ValueError("Corner native eval requires explicit frozen B0 predictions")
        return True
    if not enabled(cfg):
        if cfg.get("roi_source")=="frozen_b0_prediction":
            raise ValueError("Declared prediction ROI has no registered training protocol")
        return False
    if (cfg.get("role")!="A0" or split!="train" or cfg.get("roi_source")!="frozen_b0_prediction"
            or cfg.get("roi_expansion")!=1.5 or cfg.get("roi_size")!=448 or cfg.get("train_backbone") is not False):
        raise ValueError("Explicit fixed1.5 prediction-ROI A0 protocol required")
    if any(cfg.get(k) not in (None,False) for k in ("photometric","roi_jitter","sam_enabled","corner_enabled","sam_rgb_variants","sam_roi_routing","sam_mask_augmentation","feature_cache")):
        raise ValueError("Prediction adaptation cannot mix GT jitter, image augmentation, new branches or old feature caches")
    if cfg.get("a0_variant") not in (None,"a0_baseline"):raise ValueError("Preserve baseline A0 architecture")
    if not isinstance(cfg.get("prediction_roi_manifest"),dict):raise ValueError("Frozen prediction manifest required")
    return True

class FrozenPredictionDataset:
    def __init__(self,base,cfg):
        validate_config(cfg,base.split)
        spec=cfg["prediction_roi_manifest"];summary=_v2_checked(spec["summary"])
        rows=_v2_read(spec["rows"],lambda f:[__import__("json").loads(x) for x in f if x.strip()])
        if summary["policy"]!=POLICY or summary["B0_model"]["sha256"]!=cfg["evaluation_counterpart_sha256"]:
            raise ValueError("Frozen detector/policy differs")
        if not all(summary["source_implementation_checks"].values()):raise ValueError("Unresolved train/eval B0 implementation difference")
        producer=_v2_checked(summary["B0_complete"])
        if (producer["identity"]["RGB_variant"]!="native_original" or producer["identity"]["scoring_policy"]!=cfg["b0_scoring_policy"]
                or producer["identity"]["publication_threshold"]!=.3):raise ValueError("Frozen B0 deployment strategy differs")
        annotations=_v2_read(summary["annotations"],lambda f:[__import__("json").loads(x) for x in f if x.strip()])
        if annotations!=base.records:raise ValueError("Training annotations changed")
        predictions=_v2_checked(summary["B0_predictions"]);byid={x["sample_id"]:x["instances"] for x in predictions}
        if len(byid)!=len(base.records) or set(byid)!={x["sample_id"] for x in base.records}:raise ValueError("Prediction image domain differs")
        expected=[]
        for image_index,record in enumerate(base.records):
            local,_=associate(record,byid[record["sample_id"]])
            for row in local:row.update(image_index=image_index,image_path=record["image"],RGB_sha256=record["rgb_sha256"])
            expected.extend(local)
        if rows!=expected:raise ValueError("Prediction ownership or unmodified crop differs")
        self.base=base;self.records=base.records;self.role=base.role;self.split=base.split
        self.selected=[copy.deepcopy(row) for row in rows if row["status"]=="matched_unique"]
        lookup={unit:i for i,unit in enumerate(base.units)}
        self.base_indices=[lookup[(r["image_index"],r["GT_instance_index"])] for r in self.selected]
        self.units=[base.units[i] for i in self.base_indices]
        self.manifest_identity=canonical(spec)
        if not self.units:raise ValueError("No uniquely associated prediction training units")
    def __len__(self):return len(self.units)
    def __getitem__(self,index):
        item=dict(self.base[self.base_indices[index]])
        item["frozen_prediction_roi"]=copy.deepcopy(self.selected[index])
        item["prediction_manifest_identity"]=self.manifest_identity
        return item


def verify_prediction_binding_alias(producer,consumer):
    """Retain both source paths; permit aliases only with identical verified bytes."""
    from .action4_sam_native_export import _v2_sha
    for ref in (producer,consumer):
        if not isinstance(ref,dict) or set(ref)!={"path","sha256"} or not isinstance(ref["path"],str):raise ValueError("Explicit prediction binding required")
        sha=ref["sha256"]
        if not isinstance(sha,str) or len(sha)!=64 or any(c not in "0123456789abcdef" for c in sha):raise ValueError("Prediction SHA malformed")
    if producer["sha256"]!=consumer["sha256"]:raise ValueError("Prediction producer/consumer contents differ")
    for ref in (producer,consumer):
        if _v2_sha(ref["path"])!=ref["sha256"]:raise ValueError("Prediction alias source changed")
    return {"producer":copy.deepcopy(producer),"consumer":copy.deepcopy(consumer),"exact_content_sha256":producer["sha256"]}
