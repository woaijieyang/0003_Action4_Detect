"""Sampler fixtures only; production pipeline requires authenticated contexts."""
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from data.action4_dataset import prepare_rois
from eval.action4_video_sam_inputs import prepare_video_pipeline_sam

class SamplerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)
    def prepare(self,missing=False,reverse=False):
        rgb=torch.arange(3*40*60).reshape(3,40,60).remainder(256).to(torch.uint8)
        K=np.array([[70.,0.,30.],[0.,70.,20.],[0.,0.,1.]])
        d=[dict(track_id=t,prediction_id=f"video:{t}:frame:7",bbox_for_a0=b,bbox_b0_raw=None,roi_decision_id=t)
           for t,b in [('a',[-5.,4.,19.,28.]),('b',[30.,8.,52.,30.])]]
        masks={t:np.zeros((40,60),bool) for t in ['a','b']}
        masks['a'][6:20,0:15]=True; masks['b'][10:25,33:49]=True
        if missing: masks={}
        if reverse: d=d[::-1]
        c=SimpleNamespace(frame=SimpleNamespace(pixels=np.moveaxis(rgb.numpy(),0,-1),image_id='fixture'),
            expected_sources={'fixture':True},consume=lambda *a,**k:(masks,{'fixture':True}))
        x,s=prepare_video_pipeline_sam(rgb,K,d,c,video_context=SimpleNamespace(frame_id=7),b0_sha256='a'*64,roi_expansion=1.5)
        return rgb,K,d,x,s
    def test_RGB_padding_affine_and_intrinsics(self):
        rgb,K,d,x,s=self.prepare()
        expected,A,specs=prepare_rois([rgb],[r['bbox_for_a0'] for r in d],[0,0],roi_size=448,expansion=1.5,device='cpu',return_specs=True)
        self.assertTrue(torch.equal(expected,x.rgb));np.testing.assert_array_equal(A,x.A448)
        for i,spec in enumerate(s):
            np.testing.assert_array_equal(x.K112[i],spec.affine(112)@K)
            np.testing.assert_array_equal(x.K224[i],spec.affine(224)@K)
        self.assertTrue(bool((x.valid_image_support[0]==0).any()))
    def test_missing_mask_preserves_RGB_and_explicit_unavailability(self):
        *_,a,_=self.prepare();*_,b,_=self.prepare(missing=True)
        self.assertTrue(torch.equal(a.rgb,b.rgb));self.assertFalse(bool(b.mask_aux.available.any()))
        self.assertFalse(bool(b.mask_aux.visible.any()));self.assertFalse(bool(b.mask_aux.distance.any()))
        self.assertTrue(all(r['use_as_training_negative'] is False for r in b.source_dependencies))
    def test_reordering_preserves_instance_mask_binding(self):
        *_,a,_=self.prepare();*_,b,_=self.prepare(reverse=True)
        self.assertTrue(torch.equal(a.rgb.flip(0),b.rgb));self.assertTrue(torch.equal(a.mask_aux.visible.flip(0),b.mask_aux.visible))
        self.assertEqual([i.instance_id for i in a.mask_aux.identities][::-1],[i.instance_id for i in b.mask_aux.identities])

class BootstrapIdentityTests(unittest.TestCase):
    def test_same_RGB_from_different_B0_is_rejected(self):
        from eval.action4_video_sam_inputs import validate_bootstrap_frame
        record=dict(frame_id=0,pts=0,rgb_sha256='a'*64,prediction={'identity':{'b0_sha256':'b'*64}})
        source=dict(pts=0,sha256='a'*64)
        validate_bootstrap_frame(record,source,0,'b'*64)
        with self.assertRaises(ValueError):validate_bootstrap_frame(record,source,0,'c'*64)
        with self.assertRaises(ValueError):validate_bootstrap_frame(record,dict(source,pts=1),0,'b'*64)
