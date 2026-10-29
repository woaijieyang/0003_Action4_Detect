import copy
import numpy as np
import unittest
from eval.action4_direct_corner_metrics import score_direct_head,SCHEMA
from data.action4_prediction_roi_data import roi_finetune
from model.training.action4_roi_adaptation import lr_factor

def fixture():
    uv=[[0,0],[10,0],[10,10],[0,10],[2,2],[8,2],[8,8],[2,8]]
    record=dict(sample_id="image",instances=[dict(instance_id=1,bbox_full_xyxy=[0,0,10,10],corners_px=uv)])
    observation=dict(prediction_id="pred",predicted_full_box=[0,0,10,10],A224_from_native=np.eye(3).tolist(),
        observations=dict(corner_ids=list(range(8)),uv224=copy.deepcopy(uv)))
    return [dict(image_id="image",corner_observations=[observation])],[record]

class TestROI115(unittest.TestCase):
    def test_exact_direct_head_and_no_pose_requirement(self):
        p,r=fixture();report=score_direct_head(p,r)
        assert report["score_schema"]==SCHEMA and report["all8_at_002_rate"]==1 and report["PnP_used"] is False

    def test_missing_prediction_and_invalid_head_keep_gt_denominator(self):
        p,r=fixture();p[0]["corner_observations"]=[]
        assert score_direct_head(p,r)["all8_at_002_rate"]==0
        p,r=fixture();p[0]["corner_observations"][0]["observations"]["uv224"][0][0]=float("nan")
        q=score_direct_head(p,r);assert q["gt_count"]==1 and q["invalid_output_count"]==1 and q["all8_at_005_rate"]==0

    def test_permuted_fixed_ids_not_best_permutation_matched(self):
        p,r=fixture();p[0]["corner_observations"][0]["observations"]["uv224"]=list(reversed(p[0]["corner_observations"][0]["observations"]["uv224"]))
        assert score_direct_head(p,r)["all8_at_005_rate"]==0

    def test_affine_mapping_to_native_and_domain_check(self):
        p,r=fixture();o=p[0]["corner_observations"][0]
        o["A224_from_native"]=[[2,0,3],[0,2,5],[0,0,1]]
        o["observations"]["uv224"]=(np.asarray(o["observations"]["uv224"])*2+[3,5]).tolist()
        assert score_direct_head(p,r)["all8_at_002_rate"]==1
        with self.assertRaises(ValueError):score_direct_head([],r)

    def test_explicit_finetune_and_unchanged_legacy(self):
        assert roi_finetune({}) is False
        cfg=dict(training_protocol="action4.corner_prediction_roi_enhancement.native_v1",
            roi_finetune=dict(schema="action4.corner_roi115_finetune.v1",parent_roi_expansion=1.5,roi_expansion=1.15,epochs=75))
        assert roi_finetune(cfg)
        cfg["roi_finetune"]["roi_expansion"]=1.2
        with self.assertRaises(ValueError):roi_finetune(cfg)

    def test_low_lr_warmup_and_cosine_endpoint(self):
        assert lr_factor(0,peak=2e-6,total=16275)==.1
        assert lr_factor(199,peak=2e-6,total=16275)==1
        self.assertAlmostEqual(lr_factor(16274,peak=2e-6,total=16275),.05,places=7)

if __name__=='__main__':unittest.main()
