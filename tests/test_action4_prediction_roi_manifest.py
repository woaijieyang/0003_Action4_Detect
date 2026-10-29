import unittest,copy
from data.action4_prediction_roi_manifest import associate
class PredictionOwnership(unittest.TestCase):
 def record(self,boxes):
  return {'sample_id':'x','instances':[{'instance_id':i,'bbox_full_xyxy':b,'ignore':False} for i,b in enumerate(boxes)]}
 def pred(self,grid,box):return {'grid_index':grid,'score':.9,'bbox_full_xyxy':box}
 def test_boxes_unchanged_no_missing_GT_fallback(self):
  r=self.record([[0,0,10,10],[20,0,30,10]]);p=[self.pred(1,[1,0,11,10])];original=copy.deepcopy(p)
  rows,missing=associate(r,p);self.assertEqual(p,original);self.assertEqual(rows[0]['bbox_for_a0'],[1.,0.,11.,10.])
  self.assertEqual(missing,[1]);self.assertEqual(len(rows),1)
 def test_duplicate_not_forced(self):
  rows,_=associate(self.record([[0,0,10,10]]),[self.pred(1,[0,0,10,10]),self.pred(2,[0,0,10,10])])
  self.assertEqual(sorted(x['status'] for x in rows),['duplicate_unselected','matched_unique'])
 def test_ambiguous_not_given_pose(self):
  rows,missing=associate(self.record([[0,0,10,10],[1,0,11,10]]),[self.pred(1,[0,0,11,10])])
  self.assertEqual(rows[0]['status'],'ambiguous_GT_overlap');self.assertFalse(rows[0]['pose_supervision_association']);self.assertEqual(len(missing),2)
 def test_ignored_not_background(self):
  r=self.record([[0,0,10,10]]);r['instances'][0]['ignore']=True
  rows,_=associate(r,[self.pred(1,[0,0,10,10])]);self.assertEqual(rows[0]['status'],'ignore_overlap')

 def test_invalid_GT_rejected_before_matching(self):
  with self.assertRaises(ValueError):associate(self.record([[0,0,float('nan'),10]]),[self.pred(1,[0,0,10,10])])
 def test_boolean_score_rejected(self):
  p=self.pred(1,[0,0,10,10]);p['score']=True
  with self.assertRaises(ValueError):associate(self.record([[0,0,10,10]]),[p])
