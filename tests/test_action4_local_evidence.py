import unittest
from dataclasses import replace
import numpy as np
from model.action4_guidance_contract import *
from model.action4_local_evidence import *

class LocalEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.f=FrameIdentity("image",7,"native",80,64,"a"*64)
        self.b0=SourceIdentity("b0","b"*64,"c"*64)
        self.yolo=SourceIdentity("yolo_text","d"*64,"e"*64,"fixed7","f"*64)
        self.left=self.target("left",(10,10,30,30))
        self.right=self.target("right",(32,10,52,30))
    def target(self,id,bbox):
        return CandidateRecord(self.f,id,id,self.b0,bbox,(("center",.8),))
    def source(self,id,bbox,channel="text_positive",score=.9,mask=None):
        ref="mask:"+id
        p=np.zeros((64,80),bool)
        if mask is None:
            x0,y0,x1,y1=map(int,bbox);p[y0:y1,x0:x1]=True
        else:p=mask
        c=CandidateRecord(self.f,id,id,self.yolo,bbox,(("native",score),),support_kind="mask",source_mask_ref=ref,evidence_channel=channel,evidence_score=score)
        return c,PredictedSupport(self.f,ref,p)
    def test_positive_negative_local_channels_and_six_vector(self):
        a,am=self.source("text",(10,10,30,30))
        b,bm=self.source("visual",(10,10,30,30),"visual_positive",.7)
        c,cm=self.source("negative",(10,10,30,30),"negative",.6)
        x=build_local_evidence([self.left],[a,b,c],[am,bm,cm])
        np.testing.assert_allclose(x.values,[[.9,.7,.6,1,1,1]])
        self.assertIsNone(x.attach([self.left])[0].p_action4) # negative evidence never hard deletes
    def test_nearby_independent_supports_and_reordering(self):
        a,am=self.source("a",(10,10,30,30))
        b,bm=self.source("b",(32,10,52,30),score=.8)
        x=build_local_evidence([self.left,self.right],[a,b],[am,bm])
        y=build_local_evidence([self.right,self.left],[b,a],[bm,am])
        np.testing.assert_array_equal(x.values,y.values[::-1])
        self.assertEqual(x.evidence[0].text_positive.source_candidate_ids,("a",))
        self.assertEqual(x.evidence[1].text_positive.source_candidate_ids,("b",))
        with self.assertRaises(ValueError):x.attach([self.right,self.left])
    def test_one_large_mask_is_ambiguous_for_both_not_cloned(self):
        a,am=self.source("merged",(10,10,52,30))
        x=build_local_evidence([self.left,self.right],[a],[am])
        self.assertFalse(x.values.any())
        self.assertEqual([e.text_positive.match_status for e in x.evidence],["ambiguous","ambiguous"])
    def test_duplicate_mask_aliases_cannot_supply_two_targets(self):
        a,am=self.source("one",(10,10,30,30))
        b,bm=self.source("copy",(10,10,30,30),mask=am.pixels.copy())
        other=replace(self.left,candidate_id="other",source_detection_id="other")
        x=build_local_evidence([self.left,other],[a,b],[am,bm])
        self.assertFalse(x.values.any())
        self.assertTrue(all(e.text_positive.match_status=="ambiguous" for e in x.evidence))
    def test_duplicate_support_single_target_deterministic_dedup(self):
        a,am=self.source("a",(10,10,30,30),score=.8)
        b,bm=self.source("b",(10,10,30,30),score=.9)
        x=build_local_evidence([self.left],[a,b],[am,bm])
        self.assertAlmostEqual(x.evidence[0].text_positive.score,.9)
        self.assertEqual(x.evidence[0].text_positive.source_candidate_ids,("a","b"))
    def test_different_supports_compete_without_bbox_dedup(self):
        a,am=self.source("a",(10,10,30,30))
        p=am.pixels.copy();p[10,10]=False
        b,bm=self.source("b",(10,10,30,30),mask=p)
        x=build_local_evidence([self.left],[a,b],[am,bm])
        self.assertEqual(x.evidence[0].text_positive.reason,"multiple_distinct_supports_compete")
        self.assertFalse(x.values.any())
    def test_global_cloth_score_cannot_transfer(self):
        a,am=self.source("cloth",(55,40,75,60),score=.99)
        x=build_local_evidence([self.left,self.right],[a],[am])
        self.assertFalse(x.values.any())
        self.assertEqual(x.evidence[0].text_positive.reason,"no_unique_eligible_local_support")
    def test_maskless_evidence_does_not_become_rectangular_mask(self):
        a=CandidateRecord(self.f,"a","a",self.yolo,(10,10,30,30),(("score",.9),),evidence_channel="text_positive",evidence_score=.9)
        x=build_local_evidence([self.left],[a],[])
        self.assertFalse(x.values.any())
    def test_single_support_area_and_negative_stricter(self):
        # Actual mask-in-target coverage=.8; boxIoU=2/3 passes positive
        # auxiliary .3 but not negative .7. Pixel fill is no longer scale.
        p=np.zeros((64,80),bool);p[14:26,14:34]=True
        a,am=self.source("a",(14,10,34,30),mask=p)
        b,bm=self.source("b",(14,10,34,30),"negative",mask=p)
        x=build_local_evidence([self.left],[a,b],[am,bm])
        self.assertTrue(x.evidence[0].text_positive.available)
        self.assertFalse(x.evidence[0].negative.available)
        large,sm=self.source("wrist",(0,0,60,50),"negative")
        self.assertFalse(build_local_evidence([self.left],[large],[sm]).values.any())
    def test_central_thin_strip_uses_box_scale_not_visible_fill(self):
        target=self.target("wide",(10,10,50,50))
        p=np.zeros((64,80),bool);p[28:32,10:50]=True
        a,am=self.source("thin",(10,10,50,50),mask=p)
        x=build_local_evidence([target],[a],[am])
        self.assertTrue(x.evidence[0].text_positive.available)
        d=x.pair_diagnostics[0]
        self.assertAlmostEqual(d["visible_fill"],.1)
        self.assertEqual(d["area_ratio"],1)
        self.assertEqual(d["area_ratio_kind"],"candidate_bbox_area_ratio")
        self.assertIsNone(d["mask_iou"])
    def test_two_thin_edges_available_but_one_displaced_edge_is_not(self):
        target=self.target("wide",(10,10,50,50))
        p=np.zeros((64,80),bool);p[10:50,10:12]=True;p[10:50,48:50]=True
        a,am=self.source("edges",(10,10,50,50),mask=p)
        self.assertTrue(build_local_evidence([target],[a],[am]).evidence[0].text_positive.available)
        p[:,48:50]=False
        b,bm=self.source("one_edge",(10,10,50,50),mask=p)
        x=build_local_evidence([target],[b],[bm])
        self.assertFalse(x.evidence[0].text_positive.available)
        self.assertGreater(x.pair_diagnostics[0]["center_ratio"],.25)
    def test_single_negative_uses_box_iou_with_real_support_and_no_hard_delete(self):
        p=np.zeros((64,80),bool);p[19:21,10:30]=True
        a,am=self.source("negative_thin",(10,10,30,30),"negative",mask=p)
        x=build_local_evidence([self.left],[a],[am])
        self.assertTrue(x.evidence[0].negative.available)
        self.assertEqual(x.pair_diagnostics[0]["local_iou_kind"],"candidate_bbox_iou")
        self.assertAlmostEqual(x.pair_diagnostics[0]["visible_fill"],.1)
        self.assertIsNone(x.attach([self.left])[0].p_action4)
        # Same exact bbox but pixels outside the target cannot match by bbox alone.
        p[:]=False;p[45:55,55:65]=True
        b,bm=self.source("off_target",(10,10,30,30),"negative",mask=p)
        self.assertFalse(build_local_evidence([self.left],[b],[bm]).values.any())
    def test_dual_negative_still_requires_actual_mask_iou(self):
        p=np.zeros((64,80),bool);p[10:30,10:30]=True
        q=p.copy();q[10:18,10:30]=False
        t,tm=self.source("target_dual",(10,10,30,30),mask=p)
        s,sm=self.source("neg_dual",(10,10,30,30),"negative",mask=q)
        x=build_local_evidence([t],[s],[tm,sm])
        self.assertFalse(x.evidence[0].negative.available)
        self.assertAlmostEqual(x.pair_diagnostics[0]["mask_iou"],.6)
        self.assertEqual(x.pair_diagnostics[0]["local_iou_kind"],"predicted_mask_iou")
    def test_wide_cloth_enclosing_two_targets_not_two_available(self):
        # Both targets' pixels are covered; source cannot be copied to two identities.
        s,sm=self.source("wide_cloth",(8,8,54,32),"negative")
        x=build_local_evidence([self.left,self.right],[s],[sm])
        self.assertFalse(x.values.any())
        self.assertTrue(all(e.negative.match_status=="ambiguous" for e in x.evidence))

    def test_dual_masks_not_large_bbox_iou(self):
        # Identical bounding boxes with disjoint real supports cannot associate.
        p=np.zeros((64,80),bool);p[10:20,10:30]=True
        q=np.zeros((64,80),bool);q[20:30,10:30]=True
        t,tm=self.source("target",(10,10,30,30),mask=p)
        s,sm=self.source("source",(10,10,30,30),mask=q)
        self.assertFalse(build_local_evidence([t],[s],[tm,sm]).values.any())
    def test_coordinate_identity_and_shape_reject(self):
        s,m=self.source("s",(10,10,30,30))
        with self.assertRaises(ValueError):build_local_evidence([self.left],[replace(s,frame=replace(self.f,pixel_space_id="crop"))],[m])
        with self.assertRaises(ValueError):build_local_evidence([self.left],[s],[replace(m,frame=replace(self.f,image_id="other"))])
        with self.assertRaises(ValueError):PredictedSupport(self.f,"bad",np.zeros((32,40),bool))
        with self.assertRaises(ValueError):build_local_evidence([self.left],[s],[])
    def test_conflicting_cross_role_candidate_identity_reject(self):
        s,m=self.source("left",(10,10,30,30))
        with self.assertRaisesRegex(ValueError,"Conflicting records"):
            build_local_evidence([self.left],[s],[m])

    def test_empty_mask_and_immutable_support(self):
        a,am=self.source("a",(10,10,30,30),mask=np.zeros((64,80),bool))
        self.assertFalse(build_local_evidence([self.left],[a],[am]).values.any())
        with self.assertRaises(ValueError):am.pixels[0,0]=True

if __name__=="__main__":unittest.main()
