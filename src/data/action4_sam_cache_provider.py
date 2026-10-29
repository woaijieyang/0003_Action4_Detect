"""Thin, explicit index over existing native-SAM-v2 caches; never generates predictions.

No model imports. Training factories are injected by orchestration. Prediction
inputs and loss-only GT are separated; eval calls only frame_predictions.
"""
from collections import OrderedDict
import copy
import hashlib
import json
import os
import numpy as np
from .action4_paths import PROJECT_ROOT as ROOT, contained_path
from .action4_sam_roi_inputs import (ImageFrame,NativeMaskIdentity,native_key,array_digest,
    _sources,read_prediction_cache,verify_image_frame,verify_native_prediction,
    prepare_sam_roi_inputs,prepare_quality_targets,RoiRequest,identity_key)


def bound_json(binding):
    if not isinstance(binding,dict) or set(binding)!={"path","sha256"}:
        raise ValueError("Explicit cache path/SHA binding required")
    path=contained_path(ROOT,binding["path"],exists=True)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=binding["sha256"]:
        raise ValueError("SAM cache dependency SHA mismatch")
    return json.loads(path.read_text())


def native_rgb(value):
    """Existing datasets use CPU uint8 CHW; native SAM uses uint8 HWC."""
    import torch
    if isinstance(value,torch.Tensor):
        if value.device.type!="cpu" or value.dtype!=torch.uint8 or value.ndim!=3 or value.shape[0]!=3:
            raise ValueError("Dataset RGB must be CPU uint8 CHW")
        return value.permute(1,2,0).contiguous().numpy()
    if not isinstance(value,np.ndarray) or value.dtype!=np.uint8 or value.ndim!=3 or value.shape[2]!=3:
        raise ValueError("Native RGB must be uint8 HWC")
    return value



def resolve_training_roi(item,unit,frame,association):
    """Use actual crop identity; only legacy2.25 may omit RoiSpec."""
    import torch
    from .action4_roi_contract import RoiSpec
    record=item["record"];meta=unit["metadata"];obj=record["instances"][item["instance"]]
    if meta.get("sample_id",record["sample_id"])!=record["sample_id"] or str(meta.get("instance_id",obj["instance_id"]))!=str(obj["instance_id"]):
        raise ValueError("Training ROI belongs to another image/instance")
    source=meta.get("roi_source")
    prediction=source=="frozen_b0_prediction" or "frozen_prediction_roi" in item or "frozen_prediction_roi" in meta
    if source not in (None,"frozen_b0_prediction"):raise ValueError("Unknown training ROI source")
    stored=meta.get("roi_spec")
    if stored is None:
        if prediction or meta.get("roi_expansion",2.25)!=2.25:
            raise ValueError("Prediction/nonlegacy ROI requires actual RoiSpec identity")
        spec=RoiSpec.from_full_box(meta["crop_box"],2.25,448)
        scope="legacy2.25_reconstructed_and_affine_checked"
    else:
        if not isinstance(stored,dict):raise ValueError("Actual training RoiSpec record required")
        spec=RoiSpec.from_full_box(stored["full_body_bbox"],stored["expansion"],stored["output_size"])
        if stored!=spec.record() or spec.output_size!=448 or not np.array_equal(meta["crop_box"],spec.full_body_bbox):
            raise ValueError("Stored training ROI bounds/box/output identity differs")
        if meta.get("roi_expansion")!=spec.expansion:raise ValueError("Training ROI expansion metadata differs")
        if source is None and spec.expansion!=2.25:raise ValueError("Nonlegacy ROI requires explicit prediction source")
        scope="actual_training_RoiSpec"
    if prediction:
        row=item.get("frozen_prediction_roi");manifest=item.get("prediction_manifest_identity")
        if (source!="frozen_b0_prediction" or meta.get("crop_mode")!="explicit_prediction"
            or not isinstance(row,dict) or row!=meta.get("frozen_prediction_roi")
            or not isinstance(manifest,str) or len(manifest)!=64 or any(c not in "0123456789abcdef" for c in manifest)
            or manifest!=meta.get("prediction_manifest_identity")
            or row.get("roi_source")!="frozen_b0_prediction" or row.get("status")!="matched_unique"
            or row.get("sample_id")!=record["sample_id"] or row.get("GT_instance_index")!=item["instance"]
            or str(row.get("GT_instance_id"))!=str(obj["instance_id"])
            or row.get("RGB_sha256")!=frame.source_sha256
            or not np.array_equal(row.get("bbox_for_a0"),spec.full_body_bbox)
            or association.get("status")!="matched" or row.get("prediction_id")!=association.get("prediction_id")):
            raise ValueError("Prediction ROI source/manifest/association identity differs")
    A448=spec.affine();A112=spec.affine(112);K112=spec.camera(record["camera"]["K"],112)
    def same(actual,expected,label):
        a=np.asarray(actual,float)
        if a.shape!=(3,3) or not np.isfinite(a).all() or not np.allclose(a,expected,atol=1e-7,rtol=0):
            raise ValueError("Training "+label+" differs from actual RoiSpec")
    same(meta["affine_hm"],A112,"A112")
    if "affine_448" in meta:same(meta["affine_448"],A448,"A448")
    if "K_hm" in meta:same(meta["K_hm"],K112,"K112 metadata")
    actual_K=unit["geometry"]["K"]
    if not torch.equal(torch.as_tensor(K112,dtype=actual_K.dtype,device=actual_K.device),actual_K):
        raise ValueError("Training K112 tensor differs from actual RoiSpec")
    return spec,scope

class SamCacheProvider:
    def __init__(self,spec,*,split,allow_synthetic=False,training_associations=None,mask_augmentation=None,roi_routing=None,validation_workers=1):
        from .action4_sam_native_export import _validation_worker_count
        _validation_worker_count(validation_workers)
        if split not in ("train","eval"):raise ValueError("SAM provider supports train/development only")
        if not isinstance(spec,dict) or set(spec)!={"manifest","expected_sources"}:
            raise ValueError("Enhanced variant requires explicit sam_cache manifest and expected_sources")
        _sources(spec["expected_sources"])
        index=bound_json(spec["manifest"])
        if (index.get("schema")!="action4.sam_native_v2_index.v1" or index.get("split")!=split
                or index.get("prompt_origin")!="frozen_prediction_only"
                or index.get("sources")!=spec["expected_sources"]):
            raise ValueError("SAM prediction index split/source/prompt contract mismatch")
        if validation_workers>1 and index["sources"].get("upstream_mode")!="b0_completed_native_receipt_v2":
            raise ValueError("Parallel validation supports completed-original caches only")
        if index["sources"].get("upstream_mode")=="b0_only_native_receipt_v1":
            if split!="eval" or index.get("scope")!="frozen_B0_parent_step0_development_only":
                raise ValueError("B0-only native export is development-only, not stage3 formal training")
            from .action4_sam_native_export import verify_export_receipt
            verify_export_receipt(index)
        if index["sources"].get("upstream_mode")=="b0_completed_native_receipt_v2":
            from .action4_sam_native_export import verify_completed_export_receipt
            verify_completed_export_receipt(index,workers=validation_workers)
        if index["sources"].get("upstream_mode")=="b0_completed_native_rgb_variant_receipt_v1":
            from .action4_sam_native_export import verify_variant_export_receipt
            verify_variant_export_receipt(index)

            if mask_augmentation is not None and (split!="train" or training_associations is None):
                raise ValueError("Variant mask augmentation requires verified train supervision ownership")
        self.sources=dict(spec["expected_sources"]);self.split=split
        if type(allow_synthetic) is not bool:raise ValueError("Explicit synthetic-fixture permission boolean required")
        self.allow_synthetic=allow_synthetic
        if index.get("origin")!="predicted_sam" and not(self.allow_synthetic and index.get("origin")=="synthetic_fixture"):
            raise ValueError("Formal provider requires predicted SAM; GT/synthetic cache refused")
        self.images={}
        for entry in index["images"]:
            image_id=entry["image_id"]
            if not isinstance(image_id,str) or not image_id or image_id in self.images:raise ValueError("Duplicate/invalid indexed image")
            if len({p["prediction_id"] for p in entry["predictions"]})!=len(entry["predictions"]):
                raise ValueError("Duplicate native prediction ID")
            self.images[image_id]=entry
        if not self.images:raise ValueError("Explicit nonempty cache image coverage required")
        self.frames=OrderedDict();self.file_validations={};self.pid=os.getpid();self.associations=None
        self.mask_augmentation=None;self.mask_augmentation_binding=mask_augmentation
        if mask_augmentation is not None:
            if split!="train":raise ValueError("Mask augmentation is training-only")
            from .action4_sam_native_export import _v2_checked
            from .action4_sam_mask_augmentation import validate_policy
            self.mask_augmentation=validate_policy(_v2_checked(mask_augmentation))
        self.production_associations=False;self.variant_associations=False
        self.variant_cache=self.sources.get("upstream_mode")=="b0_completed_native_rgb_variant_receipt_v1"
        self.association_record_identities=None;self.association_annotations=None
        if training_associations is not None:
            from .action4_sam_native_export import _v2_checked
            assoc=_v2_checked(training_associations)
            production=self.sources.get("upstream_mode")=="b0_completed_native_receipt_v2"
            if production:
                from .action4_sam_training_associations import verify_associations
                self.association_record_identities=verify_associations(assoc,spec,index)
                self.association_annotations=assoc["annotations"]
                self.production_associations=True
            elif self.variant_cache:
                from .action4_sam_training_associations import verify_variant_associations
                self.association_record_identities=verify_variant_associations(assoc,spec,index)
                self.association_annotations=assoc["annotations"]
                self.variant_associations=True
            elif not (allow_synthetic and index.get("origin")=="synthetic_fixture"):
                raise ValueError("Legacy association v1 is explicit synthetic integration only")
            if (split!="train" or assoc.get("schema")!=("action4.sam_training_associations.v2" if production else "action4.sam_training_associations.rgb_variant.v1" if self.variant_associations else "action4.sam_training_associations.v1")
                    or assoc.get("cache_manifest_sha256")!=spec["manifest"]["sha256"]
                    or assoc.get("assignment_policy")!="preregistered_unique_prediction_id_no_mask_selection"):
                raise ValueError("Training association source/split/policy mismatch")
            self.associations={};used=set()
            for row in assoc["rows"]:
                key=(row["image_id"],str(row["gt_instance_id"]))
                if key in self.associations or row["status"] not in ("matched","missing","unknown","mixed","ambiguous"):
                    raise ValueError("Invalid/duplicate training association")
                if not isinstance(row.get("reason"),str) or not row["reason"]:raise ValueError("Association reason required")
                if row["status"]=="matched":
                    pid=row["prediction_id"]
                    if not isinstance(pid,str) or not pid or (key[0],pid) in used:
                        raise ValueError("One predicted instance cannot supervise multiple GT instances")
                    if key[0] not in self.images or pid not in {p["prediction_id"] for p in self.images[key[0]]["predictions"]}:
                        raise ValueError("Association candidate belongs to wrong image or is absent")
                    used.add((key[0],pid))
                elif row.get("prediction_id") is not None:
                    raise ValueError("Unknown/mixed association cannot select a predicted instance")
                self.associations[key]=row

        self.roi_router=None
        if roi_routing is not None:
            if split!="train" or not (self.production_associations or self.variant_associations):
                raise ValueError("ROI routing requires verified production association v2 train provider")
            from .action4_sam_roi_routing import SamRoiRouter
            self.roi_router=SamRoiRouter(roi_routing,assoc,self.images)

    def training_crop_spec(self,item,*,seed,epoch):
        if self.roi_router is None:raise ValueError("Explicit ROI routing policy required")
        self._verify_association_record(item["record"])
        return self.roi_router.choose(item,seed=seed,epoch=epoch)

    def __getstate__(self):
        state=dict(self.__dict__);state["frames"]=OrderedDict();state["file_validations"]={};return state

    def _check_process(self):
        if self.pid!=os.getpid():
            self.frames.clear();self.file_validations.clear();self.pid=os.getpid()

    @staticmethod
    def _stat_key(stat):
        return (stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)

    def _verified_file(self,path,expected_sha):
        """Hash once per process/path/expected SHA/stat; never cache a racing read."""
        self._check_process()
        path=contained_path(ROOT,path,exists=True).resolve(strict=True)
        key=(str(path),expected_sha)
        before=self._stat_key(path.stat())
        if self.file_validations.get(key)==before:return path,before
        # Clear materialized frames on any changed file binding, even if the
        # bytes revalidate, so a replacement cannot retain an old decoded mask.
        self.frames.clear()
        with path.open("rb") as stream:
            if self._stat_key(os.fstat(stream.fileno()))!=before:
                raise ValueError("SAM dependency replaced before hashing")
            actual=hashlib.file_digest(stream,"sha256").hexdigest()
            after_open=self._stat_key(os.fstat(stream.fileno()))
        after_path=self._stat_key(path.stat())
        if before!=after_open or before!=after_path:
            raise ValueError("SAM dependency changed during hashing; verification not registered")
        if actual!=expected_sha:raise ValueError("SAM dependency file SHA mismatch")
        self.file_validations[key]=before
        return path,before

    def _verify_association_record(self,record):
        if not (self.production_associations or self.variant_associations):return
        from .action4_sam_native_export import canonical
        binding=self.association_annotations
        self._verified_file(binding["path"],binding["sha256"])
        if self.association_record_identities.get(record["sample_id"])!=canonical(record):
            raise ValueError("Training record differs from bound production annotations")

    def validate_image_coverage(self,records):
        for record in records:
            self._verify_association_record(record)
            entry=self.images.get(record["sample_id"])
            if entry is None:raise ValueError("Dataset image missing from SAM cache index coverage")
            label_source=entry["original_source"]["path"] if self.variant_cache else entry["image_path"]
            if contained_path(ROOT,label_source,exists=True)!=contained_path(ROOT,record["image"],exists=True):
                raise ValueError("Dataset/cache source image path identity mismatch")

    def frame_predictions(self,image_id,rgb):
        self._check_process()
        if image_id not in self.images:raise ValueError("Image absent from declared SAM cache coverage")
        entry=self.images[image_id]
        rgb=native_rgb(rgb)
        if array_digest(rgb)!=entry["variant_sha256"]:raise ValueError("Actual RGB/augmentation differs from frozen SAM variant")
        source,source_stat=self._verified_file(entry["image_path"],entry["source_sha256"])
        checked=[(source,source_stat)]
        caches=[]
        for binding in entry["predictions"]:
            cache,cache_stat=self._verified_file(binding["path"],binding["sha256"])
            checked.append((cache,cache_stat));caches.append((binding,cache))
        if image_id in self.frames:
            value=self.frames.pop(image_id);self.frames[image_id]=value;return value
        frame=verify_image_frame(ImageFrame(image_id,rgb,entry["pixel_space_id"],entry["source_sha256"],
                                           entry["variant_id"],entry["variant_sha256"]))
        predictions={}
        for binding,cache in caches:
            identity=NativeMaskIdentity(image_id,binding["prediction_id"],frame.pixel_space_id)
            predictions[native_key(identity)]=verify_native_prediction(read_prediction_cache(cache),frame,identity,
                self.sources,allow_synthetic=self.allow_synthetic)
        if any(self._stat_key(path.stat())!=fingerprint for path,fingerprint in checked):
            raise ValueError("SAM dependency changed while decoding; frame cache not registered")
        self.frames[image_id]=(frame,predictions)
        while len(self.frames)>2:self.frames.popitem(last=False)
        return frame,predictions

    def enrich_training(self,item,unit,*,identity_factory,mask_aux_factory):
        if self.split!="train" or self.associations is None:raise ValueError("Explicit training association index required")
        import torch
        from PIL import Image
        record=item["record"];obj=record["instances"][item["instance"]]
        self._verify_association_record(record)
        if (self.production_associations or self.variant_associations) and (record["sample_id"],str(obj["instance_id"])) not in self.associations:
            raise ValueError("Production GT unit missing from complete supervision association")
        frame,predictions=self.frame_predictions(record["sample_id"],item["image"])
        association=copy.deepcopy(self.associations.get((frame.image_id,str(obj["instance_id"])),
            {"status":"unknown","prediction_id":None,"reason":"no_preregistered_training_association"}))
        matched=association["status"]=="matched"
        instance_id=association["prediction_id"] if matched else "training_unassociated:"+str(obj["instance_id"])
        if matched and native_key(NativeMaskIdentity(frame.image_id,instance_id,frame.pixel_space_id)) not in predictions:
            raise ValueError("Declared matched prediction absent from cache index")
        meta=unit["metadata"]
        roi_spec,roi_identity_scope=resolve_training_roi(item,unit,frame,association)
        bounds=roi_spec.continuous_bounds
        roi_id=hashlib.sha256(json.dumps({"sample":frame.image_id,"gt_unit":str(obj["instance_id"]),
            "seed":unit["augmentation_seed"],"affine_hm":meta["affine_hm"]},sort_keys=True).encode()).hexdigest()
        identity=identity_factory(frame.image_id,instance_id,roi_id,"action4.roi448.v1")
        inputs=prepare_sam_roi_inputs([frame],[RoiRequest(identity,0,bounds,record["camera"]["K"],frame.pixel_space_id)],
            predictions if matched else {},expected_sources=self.sources,mask_aux_factory=mask_aux_factory,
            allow_synthetic=self.allow_synthetic,augmentation_policy=self.mask_augmentation,
            augmentation_seeds=[unit["augmentation_seed"]] if self.mask_augmentation is not None else None)
        if self.mask_augmentation is not None:
            inputs.source_dependencies[0]["mask_augmentation"]["policy_binding"]=dict(self.mask_augmentation_binding)
        if not np.allclose(inputs.A448[0],roi_spec.affine(),atol=1e-7,rtol=0):
            raise ValueError("SAM and actual training A448 differ")
        inputs.source_dependencies[0]["training_roi_spec"]=roi_spec.record()
        inputs.source_dependencies[0]["training_roi_identity_scope"]=roi_identity_scope
        if not torch.equal(inputs.rgb[0],unit["roi"]):raise ValueError("SAM and actual training RGB ROI/augmentation differ")
        if not np.allclose(inputs.A112[0],meta["affine_hm"],atol=1e-7,rtol=0):
            raise ValueError("SAM and training ROI affine differ")
        if not torch.equal(torch.as_tensor(inputs.K112[0],dtype=unit["geometry"]["K"].dtype),unit["geometry"]["K"]):
            raise ValueError("SAM and training K differ")
        # Only after prediction input is fixed, read GT visible mask for quality loss.
        path=contained_path(ROOT,obj["mask_visible"],exists=True)
        with Image.open(path) as image:gt=np.asarray(image.convert("L"))>0
        quality=prepare_quality_targets(inputs,{identity_key(identity):{"identity":identity,"native_mask":gt,
            "supervised":bool(unit["active"]),"source_pixel_space_id":frame.pixel_space_id,"native_shape":list(frame.pixels.shape[:2])}})
        reason=inputs.reasons[0] if matched else association["status"]+":"+association["reason"]
        return dict(unit,sam_enabled=True,identity=identity,mask_aux=inputs.mask_aux,gt_identity=identity,
            gt_visible448=quality.gt_visible448[0],sam_quality_supervised=bool(quality.supervised[0]),
            sam_input={"reason":reason,"training_gt_instance_id":str(obj["instance_id"]),
                       "prediction_id":association["prediction_id"],"association":association,
                       "dependencies":inputs.source_dependencies[0]})
