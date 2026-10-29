import dataclasses
import inspect
import json
import unittest
import numpy as np
import torch
from data.action4_video_roi_context import VideoROIContext,rgb_digest
from data.action4_video_sam_snapshot import VideoMaskBinding,VideoSamSnapshot
from data.action4_sam_roi_inputs import array_digest
from model.action4_sam_feature_adapter import FeatureIdentity,MaskAux
from model.action4_sam_joint import sam_geometry_forward


def fixture():
    rgb=np.arange(3*48*64,dtype=np.uint8).reshape(3,48,64)
    K=np.array([[50.,0,32],[0,50,24],[0,0,1]])
    det=dict(track_id="track00",prediction_id="video:track00:frame:7",
             bbox_full_xyxy=[15.,10.,39.,34.],raw_bbox_full_xyxy=[14.,10.,38.,34.],
             source_detection_ids=["b0:frame7:grid:3"])
    context=VideoROIContext(7,rgb_digest(rgb),"a"*64,"b"*64,tuple(K.flat),json.dumps([det]),"[]")
    mask=np.zeros((48,64),bool);mask[13:28,20:35]=True
    binding=VideoMaskBinding(7,70,(1,30),"video:native:v1","track00",2,5,3,"session7","b"*64,array_digest(mask),"c"*64)
    sources={k:"d"*64 for k in ("b0_source_sha256","sam_model_sha256","sam_source_sha256","sam_prompt_sha256","postprocess_sha256")}
    sources.update(b0_model_sha256="a"*64,upstream_mode="b0_completed_native_receipt_v2")
    return rgb,K,context,mask,binding,sources


class VideoSnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def capture(self,f=None,**kwargs):
        f=fixture() if f is None else f
        rgb,K,c,m,b,s=f
        args=dict(expected_binding=b,prediction_binding=b,mask=m,status="available",reason=None,expected_sources=s)
        args.update(kwargs)
        return VideoSamSnapshot.capture(c,rgb,K,**args)
    def prepare(self,snapshot,f=None,**kwargs):
        rgb,K,c,m,b,s=fixture() if f is None else f
        args=dict(current_binding=b,a0_kind="joint_pose_sam",a0_selection_sha256="c"*64,
                  feature_identity_factory=FeatureIdentity,mask_aux_factory=MaskAux)
        args.update(kwargs)
        return snapshot.prepare(c,rgb,K,**args)
    def test_real_factories_actual_affines_K_and_forward_signature(self):
        f=fixture();snap=self.capture(f);p=self.prepare(snap,f);x=p.inputs
        self.assertIsInstance(x.mask_aux,MaskAux)
        self.assertIsInstance(x.mask_aux.identities[0],FeatureIdentity)
        np.testing.assert_array_equal(x.K112[0],x.A112[0]@f[1])
        from data import DataAdapter
        down,_=DataAdapter.letterbox_affine((448,448),(112,112))
        np.testing.assert_array_equal(x.A112[0],down@x.A448[0])
        self.assertTrue(bool(x.mask_aux.available.item()))
        rgb=x.rgb.clone();K=torch.as_tensor(np.stack(x.K112),dtype=rgb.dtype)
        kwargs=p.forward_kwargs(rgb=rgb,K=K,a0_kind="joint_pose_sam",a0_selection_sha256="c"*64)
        inspect.signature(sam_geometry_forward).bind(None,None,None,rgb,K,torch.zeros(1,8,3),**kwargs)
        self.assertEqual(kwargs["identities"],x.mask_aux.identities)
        self.assertFalse(p.quality_certified)
        self.assertEqual(json.loads(p.quality_evidence_json),{"status":"unknown"})
    def test_native_input_mutation_cannot_change_snapshot_and_outputs_owned(self):
        f=fixture();snap=self.capture(f);identity=snap.snapshot_id
        f[3][:]=False
        self.assertEqual(snap.snapshot_id,identity)
        p=self.prepare(snap,fixture());q=self.prepare(snap,fixture())
        self.assertTrue(torch.equal(p.inputs.mask_aux.visible,q.inputs.mask_aux.visible))
        p.inputs.mask_aux.visible.zero_()
        self.assertTrue(bool(q.inputs.mask_aux.visible.any()))
        K=torch.as_tensor(np.stack(p.inputs.K112),dtype=p.inputs.rgb.dtype)
        with self.assertRaisesRegex(ValueError,"mutated"):
            p.forward_kwargs(rgb=p.inputs.rgb,K=K,a0_kind="joint_pose_sam",a0_selection_sha256="c"*64)
    def test_swapped_mask_and_project_native_ids_rejected(self):
        f=fixture();b=f[4]
        for changed in [dataclasses.replace(b,track_id="track01"),dataclasses.replace(b,native_obj_id=6),
                        dataclasses.replace(b,session_id="other")]:
            with self.assertRaises(ValueError):self.capture(f,prediction_binding=changed)
        swapped=np.roll(f[3],10,axis=1)
        with self.assertRaisesRegex(ValueError,"mask identity"):self.capture(f,mask=swapped)
    def test_stale_replay_generations_pts_pixel_space_and_a0_rejected(self):
        f=fixture();snap=self.capture(f);b=f[4]
        for key,value in [("track_generation",3),("mask_generation",4),("replay_identity","e"*64),
                          ("frame_id",8),("pts",71),("pixel_space_id","other"),("a0_selection_sha256","e"*64)]:
            with self.subTest(key=key):
                with self.assertRaises(ValueError):self.prepare(snap,f,current_binding=dataclasses.replace(b,**{key:value}))
    def test_rgb_K_crop_and_detection_changed_rejected(self):
        f=fixture();snap=self.capture(f)
        changed=f[0].copy();changed[0,0,0]^=1
        for g in [(changed,*f[1:]),(f[0],f[1]+np.eye(3),*f[2:])]:
            with self.assertRaises(ValueError):self.prepare(snap,g)
        det=json.loads(f[2].detections_json);det[0]["bbox_full_xyxy"][0]+=1
        c=dataclasses.replace(f[2],detections_json=json.dumps(det))
        with self.assertRaisesRegex(ValueError,"context changed"):self.prepare(snap,(f[0],f[1],c,*f[3:]))
    def test_absent_missing_error_distinct_and_no_GT_fallback(self):
        f=fixture();b=dataclasses.replace(f[4],mask_sha256=None);g=(*f[:4],b,f[5])
        for status in ["absent","missing_response","inference_error","session_lost","target_visibility_unknown"]:
            snap=self.capture(g,mask=None,status=status,reason="explicit_"+status)
            p=self.prepare(snap,g)
            self.assertEqual(p.upstream_status,status)
            self.assertEqual(p.inputs.mask_aux.available.item(),0)
            self.assertEqual(p.inputs.mask_aux.visible.count_nonzero().item(),0)
            self.assertEqual(p.inputs.mask_aux.distance.count_nonzero().item(),0)
            self.assertFalse(p.quality_certified)
        zero=np.zeros_like(f[3]);zb=dataclasses.replace(f[4],mask_sha256=array_digest(zero));zg=(*f[:4],zb,f[5])
        zp=self.prepare(self.capture(zg,mask=zero,status="missing_response",reason="explicit_zero"),zg)
        self.assertEqual(zp.inputs.mask_aux.available.item(),0)
        with self.assertRaises(ValueError):self.capture(f,status="missing_response",reason="unavailable")
        with self.assertRaises(ValueError):self.capture(g,mask=None,status="available")
        with self.assertRaises(ValueError):self.capture(g,mask=None,status="GT_fallback",reason="bad")
    def test_baseline_never_silently_ignores_aux(self):
        snap=self.capture()
        with self.assertRaisesRegex(ValueError,"Baseline"):self.prepare(snap,a0_kind="joint_pose")
        p=self.prepare(snap);K=torch.as_tensor(np.stack(p.inputs.K112),dtype=p.inputs.rgb.dtype)
        with self.assertRaisesRegex(ValueError,"Baseline"):
            p.forward_kwargs(rgb=p.inputs.rgb,K=K,a0_kind="joint_pose",a0_selection_sha256="c"*64)
    def test_consumer_wrong_roi_or_K_or_model_rejected(self):
        p=self.prepare(self.capture());x=p.inputs;K=torch.as_tensor(np.stack(x.K112),dtype=x.rgb.dtype)
        for rgb,k,sha in [(x.rgb+1,K,"c"*64),(x.rgb,K+1,"c"*64),(x.rgb,K,"e"*64),(x.rgb.double(),K.double(),"c"*64),(x.rgb,K.long(),"c"*64)]:
            with self.assertRaises(ValueError):p.forward_kwargs(rgb=rgb,K=k,a0_kind="joint_pose_sam",a0_selection_sha256=sha)
    def test_explicit_roi_policy_identity_and_actual_consumer_parity(self):
        f=fixture();default=self.capture(f)
        explicit=self.capture(f,roi_expansion=2.25,roi_output_size=448)
        self.assertEqual(default.snapshot_id,explicit.snapshot_id)
        changed=self.capture(f,roi_expansion=3.0)
        self.assertNotEqual(default.snapshot_id,changed.snapshot_id)
        old=self.prepare(default,f);new=self.prepare(changed,f)
        self.assertFalse(np.array_equal(old.inputs.A448,new.inputs.A448))
        self.assertFalse(torch.equal(old.inputs.rgb,new.inputs.rgb))
        old_K=torch.as_tensor(np.stack(old.inputs.K112),dtype=old.inputs.rgb.dtype)
        with self.assertRaises(ValueError):
            new.forward_kwargs(rgb=old.inputs.rgb,K=old_K,a0_kind="joint_pose_sam",a0_selection_sha256="c"*64)
        for expansion in (0,-1,float("nan"),float("inf"),True):
            with self.assertRaises(ValueError):self.capture(f,roi_expansion=expansion)
        for size in (224,448.0,True):
            with self.assertRaises(ValueError):self.capture(f,roi_output_size=size)
        invalid=dataclasses.replace(default,roi_expansion=0)
        with self.assertRaises(ValueError):self.prepare(invalid,f)

    def test_unknown_quality_never_certifies(self):
        snap=self.capture(quality_evidence={"status":"unknown","native_presence":.99})
        p=self.prepare(snap)
        self.assertFalse(p.quality_certified)
        with self.assertRaises(dataclasses.FrozenInstanceError):snap.status="absent"


if __name__=="__main__":unittest.main()
