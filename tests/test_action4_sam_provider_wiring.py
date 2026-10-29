"""Real single train/eval RGB and actual model input wiring; synthetic masks only.

Prediction/mask/cache fixtures are explicitly synthetic and are never formal SAM
cache or model accuracy evidence. The dataset wrapper restricts this engineering
test to one actual record; Stage2Units ROI/surface/GT processing is unchanged.
"""
import copy,hashlib,json,os,tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock
import numpy as np
import torch
from PIL import Image
from data.action4_sam_cache_provider import SamCacheProvider,native_rgb
from data.action4_sam_roi_inputs import (ImageFrame,NativeMaskIdentity,NativeSamPrediction,
    SOURCE_KEYS,array_digest,prediction_metadata)
from data.action4_stage2_data import Stage2Units
from data.action4_phrase2_dataset import surface_source
from model.action4_sam_feature_adapter import FeatureIdentity,MaskAux
from model.training.action4_stage2_core import construct,loss_sums_on_batch
from model.training.action4_supervision import local_counts
from eval.action4_sam_pipeline_inputs import sam_eval_provider
from eval.action4_staged_pipeline import StagedPipeline
from eval import action4_staged_eval

ROOT=Path("/data/zy/vln/a-test")
OUT=ROOT/"outputs/action4_sam_provider_preparation_v001"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(path,obj):path.write_text(json.dumps(obj,indent=2)+"\n");return {"path":str(path),"sha256":sha(path)}

class OneActualRecord:
    def __init__(self,record,role="A0"):
        self.records=[record];self.role=role;self.units=[(0,0 if role=="A0" else None)]
    def __len__(self):return 1
    def __getitem__(self,index):
        record=self.records[0]
        with Image.open(ROOT/record["image"]) as image:rgb=torch.from_numpy(np.array(image.convert("RGB"))).permute(2,0,1)
        return {"image":rgb,"record":record,"instance":0 if self.role=="A0" else None}

def fixture(split,*,status="matched"):
    folder=OUT/("fixture_"+split+"_"+status);folder.mkdir(exist_ok=True)
    with (ROOT/f"data/phrase2/{split}/annotations.jsonl").open() as f:record=json.loads(next(f))
    item=OneActualRecord(record)[0];rgb=native_rgb(item["image"])
    sources={key:"a"*64 for key in SOURCE_KEYS};sources["b0_model_sha256"]=sha(ROOT/"models/default/B0.pt")
    frame=ImageFrame(record["sample_id"],rgb,"native.capture",""+sha(ROOT/record["image"]),"original",array_digest(rgb))
    pid="engineering-prediction-only";native_id=NativeMaskIdentity(frame.image_id,pid,frame.pixel_space_id)
    yy,xx=np.indices(rgb.shape[:2]);mask=((xx//40+yy//40)%2==0)  # Explicit non-GT synthetic prediction fixture.
    meta=prediction_metadata(frame,native_id,mask,sources,assigned_instance_id=pid,origin="synthetic_fixture",
        sam_binding={"session_id":"engineering","obj_id":1,"generation":0,"chosen_mask_id":"fixture","match_basis":"explicit_engineering_identity"})
    cache=folder/"SYNTHETIC_NOT_SAM_CACHE.npz";np.savez_compressed(cache,metadata=np.array(json.dumps(meta)),mask=mask)
    entry={"image_id":frame.image_id,"image_path":record["image"],"source_sha256":frame.source_sha256,
        "variant_id":frame.variant_id,"variant_sha256":frame.variant_sha256,"pixel_space_id":frame.pixel_space_id,
        "predictions":[{"prediction_id":pid,"path":str(cache),"sha256":sha(cache)}]}
    index={"schema":"action4.sam_native_v2_index.v1","split":split,"origin":"synthetic_fixture",
           "prompt_origin":"frozen_prediction_only","sources":sources,"images":[entry]}
    binding=write(folder/"index.json",index)
    spec={"manifest":binding,"expected_sources":sources}
    association=write(folder/"association.json",{"schema":"action4.sam_training_associations.v1",
        "cache_manifest_sha256":binding["sha256"],"assignment_policy":"preregistered_unique_prediction_id_no_mask_selection",
        "rows":[{"image_id":frame.image_id,"gt_instance_id":record["instances"][0]["instance_id"],
                 "prediction_id":pid if status=="matched" else None,"status":status,"reason":"explicit_engineering_case"}]})
    cfg=json.loads((ROOT/"outputs/action4_sam_training_contract_preparation_v001/engineering_config.json").read_text())
    cfg.update(dataset_root="data/phrase2",seed=601,train_indices=[0],roi_jitter={"center_fraction":.15,"scale_range":[.65,1.35]},
               surface_source=surface_source(),sam_cache=spec,sam_training_associations=association,sam_allow_synthetic_fixture=True)
    return record,item,spec,cfg,pid

class SamProviderWiringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise RuntimeError("CPU only")
        torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(604)
        OUT.mkdir(exist_ok=True)
        cls.train=fixture("train")
        cls.report={"scope":"REAL_ONE_TRAIN_AND_ONE_EVAL_INPUT_SYNTHETIC_MASK_ENGINEERING_ONLY","formal_cache":False,"formal_training":False}

    def units(self,cfg,record):
        with mock.patch("data.action4_phrase2_dataset.Phrase2Dataset",return_value=OneActualRecord(record)):
            return Stage2Units(cfg,sam_factories=(FeatureIdentity,MaskAux))

    def test_actual_Stage2Units_to_core_and_missing_association(self):
        record,item,spec,cfg,pid=self.train
        dataset=self.units(cfg,record);unit=dataset[(0,0,True,1)]
        self.assertEqual(unit["identity"].instance_id,pid)
        self.assertEqual(unit["sam_input"]["training_gt_instance_id"],str(record["instances"][0]["instance_id"]))
        self.assertEqual(unit["mask_aux"].available.item(),1)
        self.assertEqual(unit["sam_input"]["reason"],"available")
        self.assertEqual(local_counts([unit],sam_enabled=True)["sam_mask_quality"],1)
        model,_=construct(cfg,"cpu")
        consumed=[]
        hook=model.sam_adapter.register_forward_pre_hook(lambda mod,args,kwargs:consumed.append(kwargs["mask_aux"]),with_kwargs=True)
        try:sums=loss_sums_on_batch(model,[unit],cfg,"cpu")
        finally:hook.remove()
        self.assertTrue(torch.equal(consumed[0].visible,unit["mask_aux"].visible))
        self.assertEqual(consumed[0].identities,(unit["identity"],))
        self.assertTrue(all(torch.isfinite(value) for value in sums.values()))
        (sums["sam_mask_quality"]+sums["visible_corners"]).backward()
        self.report["train"]={"sample_id":record["sample_id"],"unit_reason":unit["sam_input"]["reason"],"prediction_id":pid,
                             "mask_pixels448":int(unit["mask_aux"].visible.sum()),"quality_count":1,
                             "loss_sums":{k:float(v.detach()) for k,v in sums.items()},"actual_core_consumed_same_mask":True}
        write(OUT/"TRAIN_CONSUMPTION.json",self.report["train"])
        # Unknown/mixed is not best-GT-IoU selection: no predicted instance, available0.
        for status in ("unknown","mixed","missing"):
            rec,_,_,badcfg,_=fixture("train",status=status)
            missing=self.units(badcfg,rec)[(0,0,False,1)]
            self.assertEqual(missing["mask_aux"].available.item(),0)
            self.assertTrue(missing["sam_input"]["reason"].startswith(status+":"))
            self.assertFalse(missing["sam_quality_supervised"])
            self.assertEqual(local_counts([missing],sam_enabled=True)["sam_mask_quality"],0)
        # Exact actual sampled ROI/K contract, no silent repair.
        altered=dict(unit,roi=unit["roi"]+0.01)
        with self.assertRaisesRegex(ValueError,"ROI/augmentation"):
            dataset.sam_provider.enrich_training(item,altered,identity_factory=FeatureIdentity,mask_aux_factory=MaskAux)

    def test_fail_closed_missing_cache_photometric_and_GT_origin(self):
        record,item,spec,cfg,pid=self.train
        bad=copy.deepcopy(cfg);bad.pop("sam_cache")
        with self.assertRaisesRegex(ValueError,"sam_cache"):self.units(bad,record)
        bad=copy.deepcopy(cfg);bad["photometric"]={"enabled":True}
        with self.assertRaisesRegex(ValueError,"photometric"):self.units(bad,record)
        with self.assertRaisesRegex(ValueError,"GT/synthetic"):
            SamCacheProvider(spec,split="train",training_associations=cfg["sam_training_associations"])
        provider=SamCacheProvider(spec,split="train",allow_synthetic=True)
        rgb=native_rgb(item["image"]).copy();rgb[0,0,0]^=1
        with self.assertRaisesRegex(ValueError,"RGB/augmentation"):provider.frame_predictions(record["sample_id"],rgb)
        with self.assertRaisesRegex(ValueError,"coverage"):provider.frame_predictions("wrong-frame",native_rgb(item["image"]))

    def test_coverage_worker_cache_and_baseline_default_contract(self):
        import pickle
        record,item,spec,cfg,pid=self.train
        provider=SamCacheProvider(spec,split="train",allow_synthetic=True)
        provider.validate_image_coverage([record])
        with self.assertRaisesRegex(ValueError,"coverage"):
            provider.validate_image_coverage([dict(record,sample_id="not-indexed")])
        provider.frame_predictions(record["sample_id"],item["image"])
        restored=pickle.loads(pickle.dumps(provider))
        self.assertFalse(restored.frames)  # Never pickle process-local verified native arrays.
        self.assertEqual(restored.frame_predictions(record["sample_id"],item["image"])[0].image_id,record["sample_id"])
        baseline=copy.deepcopy(cfg)
        for key in ("sam_enabled","a0_variant","sam_cache","sam_training_associations","sam_allow_synthetic_fixture"):baseline.pop(key,None)
        with mock.patch("data.action4_phrase2_dataset.Phrase2Dataset",return_value=OneActualRecord(record)):
            dataset=Stage2Units(baseline)
        self.assertIsNone(dataset.sam_provider)
        unit=dataset[(0,0,True,1)]
        self.assertNotIn("sam_enabled",unit);self.assertNotIn("mask_aux",unit)
        self.assertIsNone(sam_eval_provider(SimpleNamespace(a0_kind="joint_pose"),{},"eval"))

    def test_cropped_mask_quality_is_IoU_not_availability_or_probability(self):
        from model.action4_sam_feature_adapter import mask_quality_loss
        record,item,spec,cfg,pid=self.train
        dataset=self.units(cfg,record);unit=dataset[(0,0,True,1)]
        aux=unit["mask_aux"];gt=unit["gt_visible448"][None]
        q=torch.tensor([[.99]],requires_grad=True)
        output=SimpleNamespace(q_mask=q,identities=aux.identities,visible=aux.visible,
                               available=aux.available,dropout=aux.dropout)
        quality=mask_quality_loss(output,gt,gt_identities=aux.identities)
        intersection=(aux.visible.bool()&gt.bool()).sum()
        union=(aux.visible.bool()|gt.bool()).sum()
        manual=float(intersection/union)
        self.assertAlmostEqual(float(quality.target_iou),manual,places=7)
        self.assertEqual(quality.count.item(),1)
        self.assertNotAlmostEqual(manual,.99,places=2)
        self.assertNotAlmostEqual(manual,1.,places=2)
        # Pure loss counterexample: wrong nonempty mask against empty known GT is eligible target0.
        wrong=SimpleNamespace(q_mask=q,identities=aux.identities,visible=torch.ones_like(aux.visible),
                              available=torch.ones_like(aux.available),dropout=aux.dropout)
        wrong_quality=mask_quality_loss(wrong,torch.zeros_like(gt),gt_identities=aux.identities)
        self.assertEqual(wrong_quality.count.item(),1);self.assertEqual(wrong_quality.target_iou.item(),0)
        missing=SimpleNamespace(q_mask=q,identities=aux.identities,visible=torch.zeros_like(aux.visible),
                                available=torch.zeros_like(aux.available),dropout=aux.dropout)
        self.assertEqual(mask_quality_loss(missing,gt,gt_identities=aux.identities).count.item(),0)
        write(OUT/"QUALITY_IOU_EVIDENCE.json",{"actual_sample":record["sample_id"],"actual_crop_seed":unit["augmentation_seed"],
              "actual_mask_pixels448":int(aux.visible.sum()),"actual_GT_visible_pixels448":int(gt.sum()),
              "available":1,"quality_prediction_fixture":.99,"target_actual_cropped_IoU":manual,
              "wrong_nonempty_mask_empty_GT_count":1,"wrong_nonempty_target":0,"missing_mask_count":0})

    def test_wrong_native_instance_and_GT_origin_rejected(self):
        from data.action4_sam_roi_inputs import cache_key
        record,item,spec,cfg,pid=self.train
        index=json.loads(Path(spec["manifest"]["path"]).read_text())
        entry=index["images"][0]
        source=entry["predictions"][0]["path"]
        with np.load(source,allow_pickle=False) as data:
            metadata=json.loads(data["metadata"].item());mask=data["mask"]
        metadata["native_identity"][1]="wrong-predicted-instance"
        metadata["cache_key"]=cache_key(metadata)
        badcache=OUT/"wrong_native_instance.npz"
        np.savez_compressed(badcache,metadata=np.array(json.dumps(metadata)),mask=mask)
        entry["predictions"][0].update(path=str(badcache),sha256=sha(badcache))
        badspec=dict(spec,manifest=write(OUT/"wrong_instance_index.json",index))
        provider=SamCacheProvider(badspec,split="train",allow_synthetic=True)
        with self.assertRaisesRegex(ValueError,"instance/pixel-space"):
            provider.frame_predictions(record["sample_id"],item["image"])
        index["origin"]="ground_truth"
        badspec["manifest"]=write(OUT/"GT_origin_index.json",index)
        with self.assertRaisesRegex(ValueError,"GT/synthetic"):
            SamCacheProvider(badspec,split="train",allow_synthetic=True)

    def test_actual_staged_eval_calls_strict_bundle_pipeline_with_typed_cache(self):
        record,item,spec,cfg,pid=fixture("eval")
        bundle=ROOT/"outputs/action4_sam_training_contract_preparation_v001/engineering_a0_sam_step3_sam_schema_v1.pt"
        pipeline=StagedPipeline.from_bundles(str(bundle),str(ROOT/"models/default/B0.pt"),device="cpu")
        h,w=native_rgb(item["image"]).shape[:2]
        detection={"score":.8,"prediction_id":pid,"grid_index":0,"bbox_full_xyxy":[w*.25,h*.25,w*.55,h*.55]}
        pipeline._detect=lambda rgb:[detection]  # Explicit predicted-box fixture, not GT-derived.
        profile={"a0_bundle":str(bundle),"b0_bundle":"models/default/B0.pt","sam_cache":spec,"engineering_fixture":True}
        selection=OUT/"EVAL_ENGINEERING_PROFILE.json";write(selection,profile)
        receipt=OUT/"ONE_IMAGE_ENGINEERING_RECEIPT.json";write(receipt,{"engineering_only":True})
        data=OneActualRecord(record,role="B0");data._validation_receipt=receipt
        seen=[];original=pipeline.predict
        def capture(*args,**kwargs):
            self.assertIn("sam_context",kwargs)
            seen.append(kwargs["sam_context"])
            with torch.inference_mode():return original(*args,**kwargs)
        pipeline.predict=capture
        destination=Path(tempfile.mkdtemp(prefix="one_eval_attempt_",dir=OUT))/"staged_output"
        with mock.patch.object(action4_staged_eval,"load_pipeline",return_value=pipeline):
            action4_staged_eval.run(str(selection),str(destination),validation_receipt=str(receipt),validated_data=data)
        predictions=json.loads((destination/"PREDICTIONS.json").read_text())
        trace=predictions[0]["instances"][0]["sam_input"]
        self.assertTrue(trace["available"]);self.assertEqual(trace["reason"],"available")
        self.assertEqual(seen[0].frame.image_id,record["sample_id"])
        self.assertIn("same_RGB_ROI",trace["consumption_path"])
        provider=sam_eval_provider(pipeline,profile,"eval")
        # Prediction context creation reads no instances/GT fields at all.
        context=provider.context({"image":item["image"],"record":{"sample_id":record["sample_id"]}})
        self.assertEqual(context.frame.image_id,record["sample_id"])
        pipeline._detect=lambda rgb:[dict(detection,prediction_id="unassociated-prediction")]
        with torch.inference_mode():missing=original(context.frame.pixels,np.asarray(record["camera"]["K"]),np.ones((8,3)),sam_context=context)
        self.assertFalse(missing["instances"][0]["sam_input"]["available"])
        self.assertEqual(missing["instances"][0]["sam_input"]["reason"],"missing_prediction_cache")
        with self.assertRaisesRegex(ValueError,"dependency"):sam_eval_provider(pipeline,{},"eval")
        wrong=copy.deepcopy(spec);wrong["expected_sources"]["b0_model_sha256"]="b"*64
        with self.assertRaises(ValueError):sam_eval_provider(pipeline,dict(profile,sam_cache=wrong),"eval")
        self.report["eval"]={"sample_id":record["sample_id"],"loaded_bundle":str(bundle),"bundle_sha256":sha(bundle),
                            "actual_staged_eval_pipeline_context":True,"output":str(destination),"trace":trace,"missing_prediction_reason":"missing_prediction_cache",
                            "detector_source":"explicit non-GT synthetic predicted-box fixture; no B0 accuracy claim"}

    @classmethod
    def tearDownClass(cls):
        cls.report.update(cuda_initialized=torch.cuda.is_initialized(),torch_threads=torch.get_num_threads())
        import time
        write(OUT/("VERIFICATION_"+str(time.time_ns())+".json"),cls.report)

if __name__=="__main__":unittest.main(verbosity=2)
