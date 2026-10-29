import unittest
from dataclasses import replace
from model.action4_guidance_contract import *

class GuidanceContractsTest(unittest.TestCase):
    def setUp(self):
        self.frame=FrameIdentity("image",7,"native",80,64,"a"*64)
        self.b0=SourceIdentity("b0","b"*64,"c"*64)
        self.c=CandidateRecord(self.frame,"b0:grid:17","17",self.b0,(10,10,30,30),(("center",.8),))

    def test_b0_is_local_box_not_fake_mask(self):
        self.assertIsNone(self.c.source_mask_ref)
        with self.assertRaisesRegex(ValueError,"B0 box"):
            replace(self.c,support_kind="mask",source_mask_ref="rectangle")

    def test_absent_and_ambiguous_are_zero_not_negative(self):
        e=EvidenceMatch(0,False,"ambiguous","mixed")
        self.assertEqual(LocalEvidence(e,e,e).vector,(0,0,0,0,0,0))
        with self.assertRaises(ValueError):EvidenceMatch(.9,False,"missing","missing")
        with self.assertRaises(ValueError):EvidenceMatch(.9,True,"matched","matched")

    def test_provisional_packet_preserves_sources_and_budget(self):
        p=InstanceProposal("inst-local-17",(self.c.candidate_id,),"provisional","unassessed","local_box")
        packet=DetectionPacket(self.frame,(self.b0,),(self.c,),provisional=(p,),budgets=(BudgetRecord("yolo",300,300),))
        self.assertEqual(packet.instances,())
        self.assertTrue(packet.budgets[0].touched_limit)
        self.assertIsNone(packet.candidates[0].p_action4)
        self.assertIsNone(packet.candidates[0].bbox_full)

    def test_packet_crossframe_duplicate_and_unknown_member_reject(self):
        with self.assertRaises(ValueError):DetectionPacket(self.frame,(self.b0,),(self.c,self.c))
        with self.assertRaises(ValueError):DetectionPacket(replace(self.frame,image_id="other"),(self.b0,),(self.c,))
        p=InstanceProposal("p",("unknown",),"provisional","unassessed","local_box")
        with self.assertRaises(ValueError):DetectionPacket(self.frame,(self.b0,),(self.c,),provisional=(p,))

    def test_unknown_geometry_and_no_prediction_explicit(self):
        with self.assertRaises(ValueError):replace(self.c,q_full=.8)
        with self.assertRaises(ValueError):replace(self.c,bbox_candidate=(0,0,0,0))
        x=DetectionPacket(self.frame,(self.b0,),(),status="no_prediction")
        self.assertEqual(x.instances,())
        with self.assertRaises(ValueError):DetectionPacket(self.frame,(self.b0,),(self.c,),status="no_prediction")

    def test_fake_scores_and_placeholder_source_reject(self):
        with self.assertRaises(ValueError):replace(self.c,raw_scores=(("center",True),))
        with self.assertRaises(ValueError):replace(self.c,raw_scores=(("center",float("nan")),))
        with self.assertRaises(ValueError):SourceIdentity("b0","0"*64,"c"*64)

    def test_confirmed_requires_explicit_confirmation_and_separation(self):
        with self.assertRaises(ValueError):
            InstanceProposal("p",("c",),"confirmed","independent","local_box")
        with self.assertRaises(ValueError):
            InstanceProposal("p",("c",),"confirmed","merged","local_box",confirmation_source="head")
        with self.assertRaises(ValueError):SourceIdentity("yolo_text","b"*64,"c"*64)

if __name__=="__main__":unittest.main()
