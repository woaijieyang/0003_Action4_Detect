import unittest,numpy as np
from model.action4_sam_supported_boxes import SamSupportedBoxes
def seed():
 return [{'track_id':'cam','bbox_native':[95,95,185,165],'initial_appearance128':[1.]+[0.]*127}]
def cand(f,appearance=None):
 return [{'candidate_id':f'b0:frame{f}:grid:1','grid_index':1,'bbox_full_xyxy':[95,95,185,165],'score':.8,'refined_quality_score':.9,'appearance_128':appearance or [1.]+[0.]*127,'appearance_available':True}]
def masks(extra=False):
 m=np.zeros((300,300),bool);m[100:160,100:180]=True
 if extra:m[110:150,180:240]=True
 return {'cam':m}
class ConflictFallback(unittest.TestCase):
 def new(self):
  try:return SamSupportedBoxes(seed(),geometry_mode='sam_b0_fallback_v2')
  except ValueError:self.fail('explicit B0 mask-conflict fallback mode missing')
 def test_good_current_B0_not_vetoed_by_extra_SAM_pixels(self):
  t=self.new();t.update(0,0,cand(0),masks())
  r=t.update(1,1/30,cand(1),masks(True))['tracks'][0]
  self.assertEqual(r['status'],'observed');self.assertTrue(r['B0_mask_conflict_fallback']['accepted'])
  self.assertLess(r['bbox_for_roi'][2],220)
  self.assertEqual(r['support_enclosure_reason'],'current_B0_extent_after_mask_conflict')
 def test_appearance_is_actually_consumed(self):
  t=self.new();t.update(0,0,cand(0),masks())
  r=t.update(1,1/30,cand(1,[0.,1.]+[0.]*126),masks(True))['tracks'][0]
  self.assertNotEqual(r['status'],'observed')
 def test_low_quality_cannot_use_fallback(self):
  t=self.new();t.update(0,0,cand(0),masks())
  c=cand(1);c[0]['refined_quality_score']=.6
  self.assertNotEqual(t.update(1,1/30,c,masks(True))['tracks'][0]['status'],'observed')
 def test_legacy_primary_still_preserves_original_rule(self):
  t=SamSupportedBoxes(seed(),geometry_mode='sam_primary_v1')
  t.update(0,0,cand(0),masks())
  self.assertEqual(t.update(1,1/30,cand(1),masks(True))['tracks'][0]['status'],'sam_supported')

 def test_conflict_does_not_resume_raw_extent_on_one_weak_B0_frame(self):
  t=self.new();t.update(0,0,cand(0),masks());t.update(1,1/30,cand(1),masks(True))
  c=cand(2);c[0]['refined_quality_score']=.6
  r=t.update(2,2/30,c,masks(True))['tracks'][0]
  self.assertLess(r['bbox_for_roi'][2],220)
  self.assertTrue(r['B0_mask_conflict_fallback']['extent_conflicted'])
 def test_three_clean_frames_release_conflict(self):
  t=self.new();t.update(0,0,cand(0),masks());t.update(1,1/30,cand(1),masks(True))
  for f in [2,3]:
   r=t.update(f,f/30,cand(f),masks())['tracks'][0]
   self.assertTrue(r['B0_mask_conflict_fallback']['extent_conflicted'])
  r=t.update(4,4/30,cand(4),masks())['tracks'][0]
  self.assertFalse(r['B0_mask_conflict_fallback']['extent_conflicted'])

if __name__=='__main__':unittest.main()
