"""CPU engineering tests of predicted ROI448 augmentation; no accuracy claims."""
import copy,json,hashlib,uuid,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image
from data.action4_paths import PROJECT_ROOT as R
from data.action4_sam_mask_augmentation import POLICY,decision,apply
from data.action4_sam_roi_inputs import (ImageFrame,NativeMaskIdentity,NativeSamPrediction,RoiRequest,
    SOURCE_KEYS,array_digest,prediction_metadata,native_key,prepare_sam_roi_inputs,prepare_quality_targets,identity_key,read_prediction_cache)
from model.action4_sam_feature_adapter import FeatureIdentity,MaskAux,mask_quality_loss
from model.training.action4_supervision import local_counts

class MaskAugmentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def setUp(self):
        self.out=R/"outputs/action4_sam_mask_augmentation_preparation_v001"/uuid.uuid4().hex;self.out.mkdir(parents=True)
        self.rgb=np.arange(64*64*3,dtype=np.uint8).reshape(64,64,3)
        self.image=self.out/"rgb.png";Image.fromarray(self.rgb).save(self.image)
        self.mask=np.zeros((64,64),bool);self.mask[20:44,20:44]=True
        self.sources={k:"a"*64 for k in SOURCE_KEYS}
        self.frame=ImageFrame("img",self.rgb,"native",self.sha(self.image),"native_original",array_digest(self.rgb))
        self.ident=FeatureIdentity("img","inst-A","roi-A","roi448")
        self.K=np.array([[100.,0,32],[0,100.,32],[0,0,1.]])
        self.req=RoiRequest(self.ident,0,(-22.,-22.,86.,86.),self.K,"native")
        self.nid=NativeMaskIdentity("img","inst-A","native")
        meta=prediction_metadata(self.frame,self.nid,self.mask,self.sources,assigned_instance_id="inst-A",
            candidate_overlap=.8,origin="synthetic_fixture",sam_binding={"session_id":"fixture","obj_id":1,
                "generation":0,"chosen_mask_id":"fixture","match_basis":"fixture"})
        self.cache=self.out/"native.npz";np.savez_compressed(self.cache,metadata=np.array(json.dumps(meta)),mask=self.mask)
        self.pred={native_key(self.nid):read_prediction_cache(self.cache)}
        self.seed_by_op={}
        for seed in range(2000):
            op=decision(seed,"img","inst-A",POLICY)["operation"]
            self.seed_by_op.setdefault(op,seed)
            if len(self.seed_by_op)==6:break
    def sha(self,path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    def make(self,seed=None,requests=None,pred=None):
        requests=[self.req] if requests is None else requests
        return prepare_sam_roi_inputs([self.frame],requests,self.pred if pred is None else pred,
            expected_sources=self.sources,mask_aux_factory=MaskAux,allow_synthetic=True,
            augmentation_policy=POLICY if seed is not None else None,
            augmentation_seeds=[seed]*len(requests) if seed is not None else None)
    def test_seeded_prevalence_and_reorder(self):
        decisions=[decision(i,"img","inst-A",POLICY) for i in range(20000)]
        drop=sum(d["dropout"] for d in decisions);kept=len(decisions)-drop
        perturb=sum(d["perturb"] for d in decisions)
        self.assertLess(abs(drop/len(decisions)-.2),.012)
        self.assertLess(abs(perturb/kept-.25),.015)
        self.assertEqual(decisions[123],decision(123,"img","inst-A",POLICY))
        req2=RoiRequest(FeatureIdentity("img","inst-A","roi-B","roi448"),0,(0.,0.,64.,64.),self.K,"native")
        seed=self.seed_by_op["translation"]
        a=self.make(seed,[self.req,req2]);b=self.make(seed,[req2,self.req])
        torch.testing.assert_close(a.mask_aux.visible,b.mask_aux.visible.flip(0),rtol=0,atol=0)
        self.prevalence={"N":20000,"dropout":drop,"retained":kept,"perturbed":perturb}
    def test_all_operations_bounds_empty_and_padding(self):
        support=np.zeros((448,448),bool);support[20:428,20:428]=True
        mask=np.zeros_like(support);mask[18:50,18:50]=True
        original=mask.copy()
        for op,seed in self.seed_by_op.items():
            d=decision(seed,"img","inst-A",POLICY);m,log=apply(mask,support,d)
            self.assertFalse(m[~support].any());np.testing.assert_array_equal(mask,original)
            if op in ("erosion","dilation"):self.assertIn(d["parameters"]["radius"],(1,2,3))
            if op=="translation":
                self.assertLessEqual(abs(d["parameters"]["dx"]),3);self.assertLessEqual(abs(d["parameters"]["dy"]),3)
            if op=="holes":
                self.assertLessEqual(int((mask&support).sum()-m.sum()),87)
                self.assertLessEqual(len(log["actual_hole_centres_xy"]),3)
            if op=="dropout":self.assertFalse(m.any())
        empty,log=apply(np.zeros_like(mask),support,decision(self.seed_by_op["dilation"],"img","inst-A",POLICY))
        self.assertFalse(empty.any());self.assertFalse(log["applied"])
        tiny=np.zeros_like(mask);tiny[200,200]=True
        eroded,_=apply(tiny,support,decision(self.seed_by_op["erosion"],"img","inst-A",POLICY))
        self.assertFalse(eroded.any())
    def test_roi_rgb_transforms_cache_unchanged_and_distance_recomputed(self):
        from scipy.ndimage import distance_transform_edt
        before=self.sha(self.cache);original=self.make()
        for op,seed in self.seed_by_op.items():
            aug=self.make(seed)
            torch.testing.assert_close(original.rgb,aug.rgb,rtol=0,atol=0)
            for name in ("A448","A112","A224","K112","K224"):
                np.testing.assert_array_equal(getattr(original,name),getattr(aug,name))
            torch.testing.assert_close(original.mask_aux.candidate_overlap,aug.mask_aux.candidate_overlap,rtol=0,atol=0)
            self.assertEqual(aug.source_dependencies[0]["mask_augmentation"]["operation"],op)
            self.assertFalse(aug.mask_aux.visible[~aug.valid_image_support.bool()].any())
            m=aug.mask_aux.visible[0,0].bool().numpy()
            if m.any():
                d=np.clip((distance_transform_edt(m)-distance_transform_edt(~m))/16,-1,1).astype(np.float32)
                np.testing.assert_array_equal(d,aug.mask_aux.distance[0,0].numpy())
            else:self.assertFalse(aug.mask_aux.distance.any())
        self.assertEqual(before,self.sha(self.cache))
        np.testing.assert_array_equal(self.mask,self.pred[native_key(self.nid)].mask)
        self.assertNotIn("mask_augmentation",original.source_dependencies[0])
        missing=self.make(self.seed_by_op["dilation"],pred={})
        self.assertEqual(missing.reasons,("missing_prediction_cache",))
        self.assertFalse(missing.mask_aux.available.any());self.assertFalse(missing.mask_aux.visible.any())
    def test_empty_after_perturb_is_unavailable(self):
        tiny=np.zeros_like(self.mask);tiny[30,30]=True
        meta=prediction_metadata(self.frame,self.nid,tiny,self.sources,assigned_instance_id="inst-A",origin="synthetic_fixture",
            sam_binding={"session_id":"fixture","obj_id":1,"generation":0,"chosen_mask_id":"tiny","match_basis":"fixture"})
        req=RoiRequest(self.ident,0,(0.,0.,448.,448.),self.K,"native")
        result=self.make(self.seed_by_op["erosion"],[req],{native_key(self.nid):NativeSamPrediction(meta,tiny)})
        self.assertEqual(result.reasons,("empty_mask_after_augmentation",))
        self.assertFalse(result.mask_aux.available.any());self.assertFalse(result.mask_aux.distance.any())

    def test_actual_quality_loss_and_original_a0_counts(self):
        baseline=self.make()
        gt=prepare_quality_targets(baseline,{identity_key(self.ident):{"identity":self.ident,"native_mask":self.mask,
            "supervised":True,"source_pixel_space_id":"native","native_shape":[64,64]}})
        counts=[];targets={}
        for op in ("identity","dilation","dropout"):
            inp=self.make(self.seed_by_op[op]);aux=inp.mask_aux
            output=SimpleNamespace(q_mask=torch.full((1,1),.5,requires_grad=True),visible=aux.visible,
                available=aux.available,dropout=aux.dropout,identities=aux.identities)
            quality=mask_quality_loss(output,gt.gt_visible448,gt_identities=gt.gt_identities,supervised=gt.supervised)
            targets[op]=float(quality.target_iou[0])
            item={"sam_enabled":True,"identity":self.ident,"mask_aux":aux,"gt_identity":self.ident,
                "gt_visible448":gt.gt_visible448[0],"sam_quality_supervised":True,"active":True,
                "metadata":{"target_visible_input_support_count":1,"target_identity_valid":True,"gt_geometry_valid":True},
                "surface_masks":torch.ones(2,112,112),"surface_sparse":{"xy_hm":torch.zeros(1,2)},
                "geometry":{"visible":torch.tensor([True]*4+[False]*4)}}
            counts.append(local_counts([item],sam_enabled=True))
            if op=="dropout":self.assertEqual(int(quality.count),0)
        self.assertEqual(targets["identity"],1.);self.assertLess(targets["dilation"],1.)
        self.assertEqual(counts[2]["sam_mask_quality"],0)
        for k in counts[0]:
            if k!="sam_mask_quality":self.assertEqual(counts[0][k],counts[2][k])
    def test_provider_config_consumption_and_stage2_passthrough(self):
        from data.action4_sam_cache_provider import SamCacheProvider
        from data.action4_stage2_data import Stage2Units
        from data.action4_dataset import prepare_rois
        from data import DataAdapter
        def write(path,value):
            path.write_text(json.dumps(value));return {"path":str(path.relative_to(R)),"sha256":self.sha(path)}
        index={"schema":"action4.sam_native_v2_index.v1","split":"train","origin":"synthetic_fixture","prompt_origin":"frozen_prediction_only",
            "sources":self.sources,"images":[{"image_id":"img","image_path":str(self.image.relative_to(R)),
            "source_sha256":self.frame.source_sha256,"pixel_space_id":"native","variant_id":"native_original",
            "variant_sha256":self.frame.variant_sha256,"predictions":[{"prediction_id":"inst-A","path":str(self.cache.relative_to(R)),"sha256":self.sha(self.cache)}]}]}
        spec={"manifest":write(self.out/"index.json",index),"expected_sources":self.sources}
        assoc=write(self.out/"assoc.json",{"schema":"action4.sam_training_associations.v1","cache_manifest_sha256":spec["manifest"]["sha256"],
            "assignment_policy":"preregistered_unique_prediction_id_no_mask_selection","rows":[{"image_id":"img","gt_instance_id":1,
            "prediction_id":"inst-A","status":"matched","reason":"explicit fixture"}]})
        policy_path=R/"configs/data/action4_sam_mask_augmentation_v001.json"
        binding={"path":str(policy_path.relative_to(R)),"sha256":self.sha(policy_path)}
        provider=SamCacheProvider(spec,split="train",allow_synthetic=True,training_associations=assoc,mask_augmentation=binding)
        gtpath=self.out/"gt.png";Image.fromarray((self.mask*255).astype(np.uint8)).save(gtpath)
        bbox=[8.,8.,56.,56.];record={"sample_id":"img","image":str(self.image.relative_to(R)),"camera":{"K":self.K.tolist()},
            "instances":[{"instance_id":1,"mask_visible":str(gtpath.relative_to(R))}]}
        rois,A=prepare_rois([self.rgb],[bbox],[0],roi_size=448,expansion=2.25,device="cpu")
        down,_=DataAdapter.letterbox_affine((448,448),(112,112));A112=down@A[0]
        unit={"metadata":{"crop_box":bbox,"affine_hm":A112.tolist()},"augmentation_seed":self.seed_by_op["dropout"],
            "roi":rois[0],"geometry":{"K":torch.tensor(A112@self.K,dtype=torch.float32)},"active":True}
        enriched=provider.enrich_training({"record":record,"instance":0,"image":self.rgb},unit,identity_factory=FeatureIdentity,mask_aux_factory=MaskAux)
        self.assertTrue(enriched["mask_aux"].dropout[0]);self.assertFalse(enriched["mask_aux"].available.any())
        self.assertEqual(enriched["sam_input"]["dependencies"]["mask_augmentation"]["policy_binding"],binding)
        self.assertIs(enriched["roi"],unit["roi"])
        class Dataset:
            records=[record];units=[(0,0)]
            def __len__(self):return 1
        cfg={"sam_enabled":True,"role":"A0","a0_variant":"a0_sam_image","mode":"integration","sam_allow_synthetic_fixture":True,
            "sam_cache":spec,"sam_training_associations":assoc,"sam_mask_augmentation":binding,"validation_receipt":"unused"}
        with patch("data.action4_stage2_data.ComponentData",return_value=Dataset()):
            data=Stage2Units(cfg,sam_factories=(FeatureIdentity,MaskAux))
        self.assertEqual(data.sam_provider.mask_augmentation,POLICY)
        self.assertEqual(cfg["a0_variant"],"a0_sam_image")
        eval_index=copy.deepcopy(index);eval_index["split"]="eval"
        eval_spec={"manifest":write(self.out/"eval_index.json",eval_index),"expected_sources":self.sources}
        with self.assertRaisesRegex(ValueError,"training-only"):
            SamCacheProvider(eval_spec,split="eval",allow_synthetic=True,mask_augmentation=binding)

    def test_baseline_and_B0_stray_augmentation_refused_before_work(self):
        from data.action4_stage2_data import Stage2Units
        from model.action4_sam_joint import validate_sam_training_config
        from model.training.action4_stage2_core import construct,validate_run_gate
        for role,enabled in (("A0",False),("B0",False),("B0",True)):
            cfg={"role":role,"sam_enabled":enabled,"mode":"integration",
                 "sam_mask_augmentation":{"path":"unused","sha256":"a"*64}}
            with self.subTest(role=role,enabled=enabled):
                with patch("data.action4_stage2_data.ComponentData",side_effect=AssertionError("dataset must not be constructed")):
                    with self.assertRaisesRegex(ValueError,"baseline/B0"):
                        Stage2Units(cfg)
                with self.assertRaisesRegex(ValueError,"baseline/B0"):validate_sam_training_config(cfg)
                with patch("model.training.action4_stage2_core.JointSurfaceRGB",side_effect=AssertionError("no model construction")),patch("model.training.action4_stage2_core.load_bundle",side_effect=AssertionError("no bundle loading")):
                    with self.assertRaisesRegex(ValueError,"baseline/B0"):construct(cfg,"cpu")
                with self.assertRaisesRegex(ValueError,"baseline/B0"):
                    validate_run_gate(cfg,{"launch_allowed":True,"mode":"integration"})

    def test_augmentation_guard_requires_actual_variant_and_corner_flag(self):
        from model.action4_sam_joint import validate_mask_augmentation_config
        from model.training.action4_stage2_core import construct,validate_run_gate
        base={"role":"A0","mode":"integration","sam_enabled":True,
              "sam_mask_augmentation":{"path":"unused","sha256":"a"*64}}
        wrongs=[dict(base,a0_variant="a0_baseline"),dict(base),
                dict(base,a0_variant="a0_sam_image",corner_enabled=True),
                dict(base,a0_variant="a0_corner_image",corner_enabled=False)]
        for cfg in wrongs:
            with self.subTest(cfg=cfg):
                with self.assertRaises(ValueError):validate_mask_augmentation_config(cfg)
                with patch("model.training.action4_stage2_core.JointSurfaceRGB",side_effect=AssertionError("no model work")):
                    with self.assertRaises(ValueError):construct(cfg,"cpu")
                with self.assertRaises(ValueError):
                    validate_run_gate(cfg,{"launch_allowed":True,"mode":"integration"})
        validate_mask_augmentation_config(dict(base,a0_variant="a0_sam_image"))
        validate_mask_augmentation_config(dict(base,a0_variant="a0_corner_image",corner_enabled=True))

    def test_legacy_explicit_dropout_and_seed_contract(self):
        explicit=torch.ones(1,dtype=torch.bool)
        result=prepare_sam_roi_inputs([self.frame],[self.req],self.pred,expected_sources=self.sources,
            mask_aux_factory=MaskAux,allow_synthetic=True,dropout=explicit)
        self.assertTrue(result.mask_aux.dropout[0]);self.assertEqual(result.reasons,("prediction_dropout",))
        self.assertTrue(explicit[0])
        with self.assertRaises(ValueError):
            prepare_sam_roi_inputs([self.frame],[self.req],self.pred,expected_sources=self.sources,mask_aux_factory=MaskAux,
                allow_synthetic=True,augmentation_policy=POLICY)
        bad=dict(POLICY,dropout_probability=.9)
        with self.assertRaises(ValueError):decision(1,"img","inst-A",bad)

if __name__=="__main__":unittest.main()
