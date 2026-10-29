import unittest
from unittest.mock import patch
import numpy as np
import torch
from model.action4_guidance_contract import FrameIdentity,SourceIdentity,CandidateRecord
from model.action4_local_evidence import PredictedSupport
from model.action4_b0_fusion_proposals import FusionFrameInput,build_current_proposals

class ProposalAmbiguityTests(unittest.TestCase):
    def fixture(self,two=True):
        frame=FrameIdentity('frame',0,'native',16,16,'a'*64)
        bs=SourceIdentity('b0','b'*64,'c'*64);ys=SourceIdentity('yolo_visual','d'*64,'e'*64,'profile','f'*64)
        m=np.zeros((16,16),bool);m[4:12,4:12]=True
        cs=tuple(CandidateRecord(frame,'yolo:'+str(i),'raw:'+str(i),ys,(4,4,12,12),(('score',.9),),support_kind='mask',source_mask_ref='m'+str(i),evidence_channel='visual_positive',evidence_score=.9) for i in range(2 if two else 1))
        supports=tuple(PredictedSupport(frame,'m'+str(i),m) for i in range(len(cs)))
        inp=FusionFrameInput(frame,bs,np.eye(3),cs,supports)
        dense={'center_logits':torch.zeros(1,1,4,4),'quality_logits':torch.zeros(1,1,4,4)}
        return inp,dense,torch.zeros(1,256,4,4),{'preprocess':{'full_size':[16,16]}}
    def test_identical_YOLO_masks_do_not_get_two_unique_associations(self):
        inp,dense,feature,cfg=self.fixture()
        with patch('model.action4_b0_fusion_proposals.candidate_union',return_value=([],torch.empty(0,4))):
            packet,records,_,trace=build_current_proposals(dense,feature,inp,cfg)
        self.assertEqual(len(records),2)
        self.assertTrue(all(not r.local_evidence.visual_positive.available for r in records))
        self.assertTrue(all(r.local_evidence.visual_positive.match_status=='ambiguous' for r in records))
        self.assertEqual(trace['ambiguous_external_evidence_count'],2)
    def test_ambiguous_B0_support_not_repromoted_on_extra(self):
        inp,dense,feature,cfg=self.fixture(False)
        rows=[{'bbox_full_xyxy':[4,4,12,12],'grid_index':i,'center_score':.8,'old_quality_score':.7,'old_product_score':.56} for i in [1,2]]
        with patch('model.action4_b0_fusion_proposals.candidate_union',return_value=(rows,torch.tensor([[4,4,12,12]]*2))):
            packet,records,_,trace=build_current_proposals(dense,feature,inp,cfg)
        self.assertEqual(len(records),3);self.assertEqual(trace['yolo_only_count'],1)
        self.assertTrue(all(r.local_evidence.visual_positive.match_status=='ambiguous' for r in records))
        self.assertTrue(all(not r.local_evidence.visual_positive.available for r in records))

    def test_trace_uses_final_joint_matches_not_preliminary_matches(self):
        from dataclasses import replace
        inp,dense,feature,cfg=self.fixture()
        y1,y2=inp.external_candidates
        inp=replace(inp,external_candidates=(y1,replace(y2,bbox_candidate=(0,0,16,16))))
        rows=[{'bbox_full_xyxy':[4,4,12,12],'grid_index':1,'center_score':.8,'old_quality_score':.7,'old_product_score':.56}]
        with patch('model.action4_b0_fusion_proposals.candidate_union',return_value=(rows,torch.tensor([[4,4,12,12]]))):
            _,records,_,trace=build_current_proposals(dense,feature,inp,cfg)
        self.assertIn(y1.candidate_id,trace['pre_joint_matched_external_ids'])
        self.assertEqual(trace['matched_external_ids'],[])
        self.assertTrue(all(r.local_evidence.visual_positive.match_status=='ambiguous' for r in records))

if __name__=='__main__':unittest.main(verbosity=2)
