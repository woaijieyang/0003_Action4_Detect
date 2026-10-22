import copy,unittest
from eval.action4_video_merge import merge_records
class Tests(unittest.TestCase):
 def setUp(self):
  self.m={'source':{'frame_count':3,'time_base':'1/30'},'frames':[{'frame_id':i,'pts':i,'duration_pts':1,'image':str(i)+'.png','image_sha256':'a'*64} for i in range(3)]};self.r=[{**f,'source_image':f['image'],'source_image_sha256':f['image_sha256'],'time_base':'1/30','checkpoint_sha256':'b'*64,'inference_config_sha256':'c'*64,'instances':[],'claim_scope':'model_raw_predictions_pending_visual_review'} for f in self.m['frames']]
 def run_merge(self,groups):return merge_records(groups,self.m,'b'*64,'c'*64)
 def test_complete_disjoint_reordered_groups(self):self.assertEqual([r['frame_id'] for r in self.run_merge([[self.r[2]],self.r[:2]])],[0,1,2])
 def test_duplicate_rejected(self):
  with self.assertRaises(ValueError):self.run_merge([self.r,self.r[:1]])
 def test_missing_rejected(self):
  with self.assertRaises(ValueError):self.run_merge([self.r[:2]])
 def test_other_checkpoint_rejected(self):
  self.r[1]['checkpoint_sha256']='d'*64
  with self.assertRaises(ValueError):self.run_merge([self.r])
 def test_changed_timeline_rejected(self):
  self.r[1]['pts']=9
  with self.assertRaises(ValueError):self.run_merge([self.r])
 def test_fake_source_rejected(self):
  self.r[0]['source_image_sha256']='d'*64
  with self.assertRaises(ValueError):self.run_merge([self.r])
 def test_nonfinite_corner_rejected(self):
  self.r[0]['instances']=[{'instance_id':0,'corners_px':[[float('nan'),0]]*8,'bbox_full_xyxy':[0,0,1,1],'score':.5}]
  with self.assertRaises(ValueError):self.run_merge([self.r])
if __name__=='__main__':unittest.main()
