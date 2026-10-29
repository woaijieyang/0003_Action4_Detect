import copy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from data.action4_sam_roi_inputs import array_digest
from data.action4_video_roi_context import (
    VideoROIContext,VideoSAMSupportReceipt,VideoSAMSupportAdmission,rgb_digest)
from eval.action4_staged_pipeline import StagedPipeline

ROOT=Path("/data/zy/vln/a-test")
OUT=ROOT/"outputs/action4_sam_supported_roi_v001"

def fixture():
    yy,xx=np.indices((64,96));rgb=np.stack([xx*2,yy*3,(xx+yy)%256]).astype(np.uint8)
    K=np.array([[80.,0,48.],[0,80.,32.],[0,0,1.]])
    box=[10.,12.,40.,42.];mask=np.zeros((64,96),bool);mask[15:30,15:35]=True
    track={"track_id":"track_00","status":"sam_supported","sam_current_support":True,
           "raw_detection":None,"candidate_id":None,"bbox_for_roi":box}
    row={"frame":{"global_frame_id":4,"pts":1600},"fused":{"mode":"video_inference","frame_id":4,"tracks":[track]}}
    receipt=VideoSAMSupportReceipt(4,1600,(1,11988),rgb_digest(rgb),"native_integer_centers","track_00",0,
        "session-A",2,3,"b"*64,"c"*64,"d"*64,array_digest(mask),tuple(box))
    admission=VideoSAMSupportAdmission("native_integer_centers",(1,11988),"c"*64,"d"*64,
                                      (receipt,),((receipt,mask),))
    return row,rgb,K,receipt,mask,admission

def context(f=None,admission=True):
    row,rgb,K,receipt,mask,support=fixture() if f is None else f
    return VideoROIContext.from_replay(row,rgb,K,b0_sha256="a"*64,replay_identity="b"*64,
                                      sam_support=support if admission else None)

def observed(row,box=None):
    row=copy.deepcopy(row);track=row["fused"]["tracks"][0];box=box or track["bbox_for_roi"]
    cid="b0:frame4:grid:5";track.update(status="observed",bbox_for_roi=box,candidate_id=cid,
        raw_detection={"candidate_id":cid,"grid_index":5,"score":.8,"bbox_full_xyxy":box})
    return row

class DummyPipeline:
    """No weights/model constructor: actual ROI prep with captured CPU test heads."""
    def __init__(self):
        self.pipeline=StagedPipeline.__new__(StagedPipeline);p=self.pipeline
        p.device="cpu";p.a0_kind="joint_pose";p.pose_method="network";p.identity={"fixture":"CPU_ROI_wiring_only"}
        p._surface_model_bindings={};p.a0=object();self.rois=[];self.K=[]
        def backbone(rois):
            self.rois.append(rois.clone());return torch.zeros(len(rois),1,1,1)
        p.base=SimpleNamespace(backbone=backbone)
    def geometry(self,head,features,K,cad):
        self.K.append(K.clone());n=len(features)
        camera=torch.zeros(n,8,3);camera[:,:,2]=1
        return {"corners":torch.zeros(n,8,2),"camera_corners":camera}
    def run(self,rgb,K,ctx=None):
        rgb=torch.from_numpy(rgb.copy()) if isinstance(rgb,np.ndarray) else rgb
        with patch("eval.action4_surface_snapshot.model_identities",return_value={"B0":{"sha256":"a"*64}}),patch("model.action4_geometry_forward.instancewise_geometry",side_effect=self.geometry):
            if ctx is None:return self.pipeline.predict(rgb,K,np.zeros((8,3)))
            return self.pipeline.predict_video(rgb,K,np.zeros((8,3)),video_context=ctx)

class SupportedROI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def test_explicit_optin_preserves_B0_absence(self):
        f=fixture();ctx=context(f);rows=ctx.consume(f[1],f[2],b0_sha256="a"*64)
        self.assertEqual(len(ctx.select_for_a0(rows)),1)
        p=rows[0];self.assertIsNone(p["score"]);self.assertIsNone(p["grid_index"])
        self.assertEqual(p["source_detection_ids"],[]);self.assertIsNone(p["raw_bbox_full_xyxy"])
        self.assertEqual(p["origin"],"SAM_tracking_geometry");self.assertFalse(p["B0_current_observation"])
        self.assertFalse(p["full_box_accuracy_verified"])
        self.assertEqual(p["sam_support_evidence"]["receipt"]["source_sha256"],"c"*64)
        self.assertFalse(p["sam_support_evidence"]["mask_quality_certified"])
    def test_default_no_receipt_predicted_and_lost_remain_withheld(self):
        f=fixture()
        self.assertEqual(context(f,False).consume(f[1],f[2],b0_sha256="a"*64),[])
        for status in ("predicted","lost"):
            g=list(copy.deepcopy(f));g[0]["fused"]["tracks"][0]["status"]=status
            self.assertEqual(context(g,False).consume(g[1],g[2],b0_sha256="a"*64),[])
            with self.assertRaises(ValueError):context(g)
        with self.assertRaises(ValueError):
            VideoROIContext.from_replay(f[0],f[1],f[2],b0_sha256="a"*64,replay_identity="b"*64,sam_support=True)
    def test_frame_source_ID_generation_and_replay_mismatches(self):
        f=fixture();receipt=f[3];admission=f[5]
        for key,value in [("frame_id",5),("pts",1601),("rgb_identity","e"*64),("pixel_space_id","other"),
                          ("track_id","other"),("native_obj_id",1),("session_id","other"),
                          ("track_generation",3),("mask_generation",4),("replay_identity","e"*64),
                          ("source_sha256","e"*64),("ledger_sha256","e"*64)]:
            with self.subTest(key=key):
                changed=replace(receipt,**{key:value});g=list(f)
                g[5]=replace(admission,predictions=((changed,f[4]),))
                with self.assertRaises(ValueError):context(g)
        # Even internally equal declarations must bind actual current row/RGB/source.
        changed=replace(receipt,frame_id=5);g=list(f);g[5]=replace(admission,expected=(changed,),predictions=((changed,f[4]),))
        with self.assertRaises(ValueError):context(g)
    def test_current_replay_native_domain_cannot_contradict_receipt(self):
        f=fixture()
        for key,value in [("session_id","stale"),("generation",4),("pixel_space_id","wrong"),
                          ("time_base",[1,25]),("native_unavailable",["track_00"])]:
            g=list(copy.deepcopy(f));g[0][key]=value
            with self.assertRaises(ValueError):context(g)
        g=list(copy.deepcopy(f));g[0]["fused"]["tracks"][0]["support_rejection"]="ambiguous_shared_instance_support"
        with self.assertRaises(ValueError):context(g)

    def test_empty_changed_wrongshape_masks_and_bbox_rejected(self):
        f=fixture()
        for mask in [np.zeros_like(f[4]),np.roll(f[4],20,axis=1),np.zeros((4,4),bool),f[4].astype(np.uint8)]:
            g=list(f);g[5]=replace(f[5],predictions=((f[3],mask),))
            with self.assertRaises(ValueError):context(g)
        g=list(copy.deepcopy(f));g[0]["fused"]["tracks"][0]["bbox_for_roi"][0]=float("nan")
        with self.assertRaises(ValueError):context(g)
    def test_duplicate_native_or_project_ID_rejected(self):
        f=fixture();g=list(copy.deepcopy(f));g[0]["fused"]["tracks"].append(copy.deepcopy(g[0]["fused"]["tracks"][0]))
        with self.assertRaises(ValueError):context(g)
        g=list(copy.deepcopy(f));second=copy.deepcopy(g[0]["fused"]["tracks"][0]);second["track_id"]="track_01";second["bbox_for_roi"]=[50.,12.,80.,42.]
        g[0]["fused"]["tracks"].append(second);mask=np.roll(f[4],40,axis=1)
        r2=replace(f[3],track_id="track_01",bbox_full_xyxy=tuple(second["bbox_for_roi"]),mask_sha256=array_digest(mask))
        g[5]=replace(f[5],expected=(f[3],r2),predictions=((f[3],f[4]),(r2,mask)))
        with self.assertRaisesRegex(ValueError,"Duplicate native"):context(g)
        g[5]=replace(f[5],expected=(f[3],f[3]))
        with self.assertRaises(ValueError):context(g)
    def test_unbound_manual_context_and_changed_RGB_K_fail(self):
        f=fixture();ctx=context(f)
        forged=VideoROIContext(ctx.frame_id,ctx.rgb_identity,ctx.b0_sha256,ctx.replay_identity,ctx.K_native,
            ctx.detections_json,ctx.withheld_json,ctx.sam_support_json)
        with self.assertRaises(ValueError):forged.consume(f[1],f[2],b0_sha256="a"*64)
        rgb=f[1].copy();rgb[0,0,0]^=1
        with self.assertRaises(ValueError):ctx.consume(rgb,f[2],b0_sha256="a"*64)
        with self.assertRaises(ValueError):ctx.consume(f[1],f[2]+np.eye(3),b0_sha256="a"*64)
    def test_current_observed_legacy_parity(self):
        f=fixture();row=observed(f[0])
        spec=importlib.util.spec_from_file_location("legacy_roi_context_reference",OUT/"VIDEO_ROI_CONTEXT_BEFORE.py")
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        old=module.VideoROIContext.from_replay(row,f[1],f[2],b0_sha256="a"*64,replay_identity="b"*64)
        new=VideoROIContext.from_replay(row,f[1],f[2],b0_sha256="a"*64,replay_identity="b"*64)
        previous=old.consume(f[1],f[2],b0_sha256="a"*64)
        current=new.consume(f[1],f[2],b0_sha256="a"*64)
        self.assertEqual(len(previous),len(current))
        for before,after in zip(previous,current):
            self.assertEqual(before,{k:after[k] for k in before})
            self.assertEqual(after["bbox_for_a0"],after["bbox_full_xyxy"])
            self.assertEqual(after["bbox_b0_raw"],before["raw_bbox_full_xyxy"])
            self.assertTrue(after["roi_decision_id"])
        self.assertEqual(old.withheld_json,new.withheld_json)
    def test_real_ROI_K_consumed_and_score_None_retained(self):
        f=fixture();probe=DummyPipeline()
        result=probe.run(f[1],f[2],context(f));row=result["instances"][0]
        self.assertIsNone(row["score"]);self.assertIsNone(row["grid_index"]);self.assertEqual(row["source_detection_ids"],[])
        self.assertEqual(row["origin"],"SAM_tracking_geometry");self.assertFalse(row["B0_current_observation"])
        trace=result["video_roi_consumption"];self.assertFalse(trace["A0_mask_features_enabled"])
        self.assertEqual(trace["sam_role"],"SAM_tracking_geometry_to_RGB_ROI")
        actual=trace["instances"][0];np.testing.assert_allclose(np.asarray(actual["A112"])@f[2],actual["K112"],atol=1e-5)
        np.testing.assert_array_equal(probe.K[-1][0].numpy(),np.array(actual["K112"],np.float32))
        before=probe.rois[-1]
        other=VideoROIContext.from_replay(observed(f[0],[40.,12.,70.,42.]),f[1],f[2],b0_sha256="a"*64,replay_identity="b"*64)
        probe.run(f[1],f[2],other)
        self.assertFalse(torch.equal(before,probe.rois[-1]));self.assertFalse(torch.equal(probe.K[0],probe.K[1]))
    def test_image_filter_unchanged_and_temporal_image_input_rejected(self):
        f=fixture();probe=DummyPipeline()
        probe.pipeline._detect=lambda rgb:[{"score":.2,"bbox_full_xyxy":[1,1,10,10]},
                                           {"score":.8,"bbox_full_xyxy":[15,15,35,35]}]
        result=probe.run(f[1],f[2]);self.assertEqual(len(result["detections"]),1);self.assertEqual(result["detections"][0]["score"],.8)
        probe.pipeline._detect=lambda rgb:context(f).consume(f[1],f[2],b0_sha256="a"*64)
        with self.assertRaisesRegex(ValueError,"Temporal"):probe.run(f[1],f[2])
    def test_surface_and_enhanced_video_guard_remains(self):
        f=fixture();probe=DummyPipeline();probe.pipeline.a0_kind="joint_pose_sam"
        with self.assertRaises(ValueError):probe.run(f[1],f[2],context(f))
        probe.pipeline.a0_kind="surface";probe.pipeline.pose_method="pnp"
        with self.assertRaisesRegex(ValueError,"separate admission"):probe.run(f[1],f[2],context(f))
    def test_baseline_cuboid_renderer_accepts_None_score(self):
        from PIL import Image
        from eval.action4_video import draw_cuboids
        f=fixture();row=DummyPipeline().run(f[1],f[2],context(f))["instances"][0]
        image=Image.fromarray(f[1].transpose(1,2,0))
        rendered=draw_cuboids(image,[row]);self.assertEqual(rendered.size,image.size)

if __name__=="__main__":unittest.main()
