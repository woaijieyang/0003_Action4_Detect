"""Engineering fixtures for real staged orchestration; no model accuracy claim."""
import unittest
from unittest.mock import Mock
import numpy as np
import torch
from data.action4_dataset import prepare_rois
from data.action4_sam_roi_inputs import ImageFrame,array_digest,SOURCE_KEYS
from eval.action4_sam_pipeline_inputs import SamInferenceContext,prepare_pipeline_sam
from eval.action4_staged_pipeline import StagedPipeline

class CaptureHead(torch.nn.Module):
    def forward(self,features,K,canonical,**kw):
        self.received=kw
        n=len(features);uv=torch.zeros(n,8,2);cam=torch.ones(n,8,3)
        from types import SimpleNamespace
        aux=SimpleNamespace(q_mask=torch.full((n,1),.5),residual=torch.zeros(n,1,1,1))
        return dict(corners=uv,camera_corners=cam),aux

class SamPipelineInputTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.rgb=np.arange(64*80*3,dtype=np.uint8).reshape(64,80,3)
        self.K=np.array([[100.,0,40],[0,100,32],[0,0,1]])
        frame=ImageFrame("fixture",self.rgb,"native.fixture","a"*64,"original",array_digest(self.rgb))
        self.context=SamInferenceContext(frame,{}, {k:"b"*64 for k in SOURCE_KEYS})
        self.dets=[dict(score=.8,prediction_id="camera-left",bbox_full_xyxy=[-5.,2.,24.,25.]),
                   dict(score=.7,prediction_id="camera-right",bbox_full_xyxy=[40.,30.,65.,60.])]
    def test_missing_mask_uses_exact_baseline_rgb_affines_and_k(self):
        x=prepare_pipeline_sam(self.rgb,self.K,self.dets,self.context)
        rgb,A=prepare_rois([self.rgb],[d["bbox_full_xyxy"] for d in self.dets],[0,0],448,2.25)
        self.assertTrue(torch.equal(x.rgb,rgb))
        np.testing.assert_array_equal(x.A448,np.array(A))
        np.testing.assert_allclose(x.K112,np.array([a@self.K for a in x.A112]))
        self.assertEqual(x.mask_aux.available.sum().item(),0)
        self.assertEqual(x.reasons,("missing_prediction_cache",)*2)
        self.assertEqual([i.instance_id for i in x.mask_aux.identities],["camera-left","camera-right"])
    def test_rgb_identity_mismatch_and_duplicate_instance_reject(self):
        rgb=self.rgb.copy();rgb[0,0,0]^=1
        with self.assertRaisesRegex(ValueError,"RGB differs"):
            prepare_pipeline_sam(rgb,self.K,self.dets,self.context)
        dets=[dict(d,prediction_id="same") for d in self.dets]
        # Same instance with two different ROI boxes must not be silently shared.
        with self.assertRaisesRegex(ValueError,"prediction ID"):
            prepare_pipeline_sam(self.rgb,self.K,dets,self.context)
    def test_real_staged_predict_calls_enhanced_head_and_retains_binding(self):
        p=StagedPipeline.__new__(StagedPipeline);p.device="cpu";p.a0_kind="joint_pose_sam";p.pose_method="network"
        p.identity={"fixture":True,"B0_bundle_sha256":"b"*64};p._detect=Mock(return_value=self.dets)
        from types import SimpleNamespace
        p.base=SimpleNamespace(backbone=lambda rgb:torch.zeros(len(rgb),2048,14,14))
        p.a0=CaptureHead()
        result=p.predict(self.rgb,self.K,np.ones((8,3)),sam_context=self.context)
        self.assertEqual(len(result["instances"]),2)
        self.assertEqual([x.instance_id for x in p.a0.received["identities"]],["camera-left","camera-right"])
        self.assertEqual(result["instances"][0]["sam_input"]["available"],False)
        self.assertEqual(result["instances"][0]["corner_ids"],list(range(8)))
        self.assertEqual(result["instances"][0]["source_detection_ids"],["camera-left"])
        with self.assertRaisesRegex(ValueError,"context"):
            p.predict(self.rgb,self.K,np.ones((8,3)))
        p.a0_kind="joint_pose"
        with self.assertRaisesRegex(ValueError,"silently ignore"):
            p.predict(self.rgb,self.K,np.ones((8,3)),sam_context=self.context)
    def test_anonymous_cache_lookup_rejected_and_prefilter_grid_id_stable(self):
        anonymous=[{k:v for k,v in d.items() if k!="prediction_id"} for d in self.dets]
        with self.assertRaisesRegex(ValueError,"stable prediction ID"):
            prepare_pipeline_sam(self.rgb,self.K,anonymous,self.context)
        p=StagedPipeline.__new__(StagedPipeline);p.device="cpu";p.a0_kind="joint_pose_sam";p.pose_method="network"
        p.identity={"B0_bundle_sha256":"b"*64}
        from types import SimpleNamespace
        p.base=SimpleNamespace(backbone=lambda rgb:torch.zeros(len(rgb),2048,14,14));p.a0=CaptureHead()
        low=dict(anonymous[0],score=.1,grid_index=11);high=dict(anonymous[1],grid_index=22)
        p._detect=lambda rgb:[low,high]
        result=p.predict(self.rgb,self.K,np.ones((8,3)),sam_context=self.context)
        self.assertEqual(result["instances"][0]["prediction_id"],"b0:fixture:grid:22")
        p._detect=lambda rgb:[high]
        second=p.predict(self.rgb,self.K,np.ones((8,3)),sam_context=self.context)
        self.assertEqual(second["instances"][0]["prediction_id"],result["instances"][0]["prediction_id"])

if __name__=="__main__":unittest.main()
