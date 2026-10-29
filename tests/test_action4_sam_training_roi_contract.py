"""Tiny CPU cache/ROI contract fixtures; no weights, SAM inference or training."""
import copy,json,unittest
import numpy as np
import torch
from PIL import Image
from tests import test_action4_sam_provider_file_validation as file_fixtures
sha=file_fixtures.sha
from data.action4_sam_cache_provider import SamCacheProvider
from data.action4_pose_rgb_batch import prepare_pose_rgb
from model.action4_sam_feature_adapter import FeatureIdentity,MaskAux

class TrainingRoiTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        fixture=file_fixtures.FileVerificationTests(methodName="runTest");fixture.setUp()
        self.addCleanup(fixture.tearDown);self.fixture=fixture
        index=fixture.folder/"index.json"
        spec={"manifest":{"path":str(index),"sha256":sha(index)},"expected_sources":fixture.provider.sources}
        association={"schema":"action4.sam_training_associations.v1","cache_manifest_sha256":sha(index),
          "assignment_policy":"preregistered_unique_prediction_id_no_mask_selection",
          "rows":[{"image_id":"fixture","gt_instance_id":1,"prediction_id":"pred","status":"matched","reason":"synthetic ROI fixture"}]}
        ap=fixture.folder/"associations.json";ap.write_text(json.dumps(association))
        self.provider=SamCacheProvider(spec,split="train",allow_synthetic=True,
             training_associations={"path":str(ap),"sha256":sha(ap)})
        gt=fixture.folder/"gt.png";Image.fromarray(np.ones((16,16),np.uint8)*255).save(gt)
        K=np.array([[20.,0,8],[0,20,8],[0,0,1.]])
        cad=np.array([[x,y,z] for z in (-10,10) for y in (-10,10) for x in (-10,10)])
        T=np.eye(4);T[2,3]=.2;camera=cad/1000+T[:3,3];h=camera@K.T
        obj={"instance_id":1,"bbox_full_xyxy":[2.,3.,12.,13.],"corners_px":(h[:,:2]/h[:,2,None]).tolist(),
             "corner_visible":[True]*8,"occlusion_inframe":0.,
             "canonical8_geometry":{"object_to_capture_cv_m":T.tolist()},"mask_visible":str(gt)}
        self.record={"sample_id":"fixture","split":"train","image":str(fixture.image),
            "rgb_sha256":sha(fixture.image),"width":16,"height":16,"camera":{"K":K.tolist()},
            "corners_3d_mm":cad.tolist(),"instances":[obj]}
    def unit(self,expansion=1.5,box=None,legacy=False):
        item={"image":torch.from_numpy(self.fixture.rgb.copy()).permute(2,0,1),
              "record":self.record,"instance":0}
        box=[2.,3.,12.,13.] if box is None else box
        if legacy:
            rois,g,meta=prepare_pose_rgb([item],seeds=[44])
        else:
            rois,g,meta=prepare_pose_rgb([item],boxes=[box],roi_expansion=expansion,roi_source="frozen_b0_prediction")
            row={"sample_id":"fixture","prediction_id":"pred","GT_instance_id":1,"GT_instance_index":0,
                 "RGB_sha256":sha(self.fixture.image),"bbox_for_a0":box,"roi_source":"frozen_b0_prediction","status":"matched_unique"}
            item.update(frozen_prediction_roi=row,prediction_manifest_identity="d"*64)
            meta[0].update(frozen_prediction_roi=copy.deepcopy(row),prediction_manifest_identity="d"*64)
        unit={"roi":rois[0],"metadata":meta[0],"geometry":{k:v[0] for k,v in g.items()},
              "active":True,"augmentation_seed":44}
        return item,unit
    def consume(self,item,unit):
        return self.provider.enrich_training(item,unit,identity_factory=FeatureIdentity,mask_aux_factory=MaskAux)
    def test_actual_prediction_15_and_225_same_RGB_A448_A112_K112(self):
        for expansion in (1.5,2.25):
            item,unit=self.unit(expansion)
            result=self.consume(item,unit)
            self.assertTrue(torch.equal(result["roi"],unit["roi"]))
            dep=result["sam_input"]["dependencies"]
            self.assertEqual(dep["training_roi_spec"],unit["metadata"]["roi_spec"])
            self.assertEqual(dep["training_roi_spec"]["expansion"],expansion)
            self.assertEqual(dep["training_roi_identity_scope"],"actual_training_RoiSpec")
            self.assertEqual(dep["roi_bounds_native"],unit["metadata"]["roi_spec"]["continuous_bounds"])
    def test_outside_image_padding_is_identical(self):
        item,unit=self.unit(1.5,[-4.,-3.,8.,9.])
        result=self.consume(item,unit)
        self.assertTrue(torch.equal(result["roi"],unit["roi"]))
        self.assertLess(float(result["mask_aux"].visible.sum()),448*448)
        self.assertTrue(torch.isfinite(result["roi"]).all())
    def test_legacy_jitter_225_without_new_spec_remains_equivalent(self):
        item,unit=self.unit(legacy=True)
        fresh=self.consume(item,unit);old=copy.deepcopy(unit)
        for key in ("roi_spec","roi_expansion","roi_source"):old["metadata"].pop(key,None)
        legacy=self.consume(item,old)
        self.assertEqual(fresh["identity"],legacy["identity"])
        self.assertTrue(torch.equal(fresh["roi"],legacy["roi"]))
        self.assertTrue(torch.equal(fresh["mask_aux"].visible,legacy["mask_aux"].visible))
        self.assertEqual(legacy["sam_input"]["dependencies"]["training_roi_spec"]["expansion"],2.25)
    def test_prediction_missing_identity_or_wrong_source_rejected(self):
        for key in ("roi_spec","roi_source","frozen_prediction_roi","prediction_manifest_identity"):
            item,unit=self.unit();unit["metadata"].pop(key)
            with self.assertRaises(ValueError):self.consume(item,unit)
        item,unit=self.unit();unit["metadata"]["roi_source"]="GT"
        with self.assertRaises(ValueError):self.consume(item,unit)
        item,unit=self.unit();item["frozen_prediction_roi"]["prediction_id"]="different"
        with self.assertRaises(ValueError):self.consume(item,unit)
    def test_wrong_affine_K_and_RGB_rejected(self):
        for key in ("affine_hm","K_hm"):
            item,unit=self.unit();unit["metadata"][key][0][2]+=1
            with self.assertRaises(ValueError):self.consume(item,unit)
        item,unit=self.unit();unit["metadata"]["affine_448"]=np.eye(3).tolist()
        with self.assertRaises(ValueError):self.consume(item,unit)
        item,unit=self.unit();unit["geometry"]["K"]=unit["geometry"]["K"]+1
        with self.assertRaises(ValueError):self.consume(item,unit)
        item,unit=self.unit();unit["roi"]=unit["roi"]+.01
        with self.assertRaises(ValueError):self.consume(item,unit)
        item,unit=self.unit();item["image"][0,0,0]^=1
        with self.assertRaises(ValueError):self.consume(item,unit)
    def test_existing_prediction_SAM_admission_guard_still_rejects(self):
        from data.action4_prediction_roi_data import validate_config,PROTOCOL
        cfg={"training_protocol":PROTOCOL,"role":"A0","roi_source":"frozen_b0_prediction","roi_expansion":1.5,
             "roi_size":448,"train_backbone":False,"sam_enabled":True,"prediction_roi_manifest":{}}
        with self.assertRaises(ValueError):validate_config(cfg,"train")

if __name__=="__main__":unittest.main()
