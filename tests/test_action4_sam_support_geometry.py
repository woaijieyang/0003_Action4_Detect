import unittest
import numpy as np
from model.action4_sam_supported_boxes import SamSupportedBoxes, geometry_support, mask_geometry

class GeometrySupportTests(unittest.TestCase):
 def test_tiny_remote_component_cannot_expand_support_box(self):
  mask=np.zeros((400,500),bool);mask[200:250,300:350]=True;mask[3,2]=True
  before=mask.copy();clean,receipt=geometry_support(mask)
  np.testing.assert_array_equal(mask,before)
  np.testing.assert_allclose(mask_geometry(clean)[0],[300,200,350,250])
  self.assertEqual(receipt["removed_pixels"],1)
  self.assertTrue(receipt["raw_mask_preserved"])
 def test_legitimate_disconnected_visible_surfaces_are_kept(self):
  mask=np.zeros((100,150),bool);mask[20:50,20:50]=True;mask[20:30,70:80]=True
  clean,r=geometry_support(mask);np.testing.assert_array_equal(mask,clean)
  self.assertEqual(r["retained_components"],2)
 def test_visible_portrait_body_not_cut_by_historical_landscape_size(self):
  t=SamSupportedBoxes([{"track_id":"a","bbox_native":[30,35,90,65]}])
  mask=np.zeros((120,120),bool);mask[40:60,40:80]=True;t.update(0,0,[],{"a":mask})
  turned=np.zeros_like(mask);turned[25:85,45:75]=True
  row=t.update(1,1/30,[],{"a":turned})["tracks"][0];box=row["bbox_for_roi"]
  self.assertLessEqual(box[1],25);self.assertGreaterEqual(box[3],85)
  self.assertTrue(row["support_enclosure_applied"])
  self.assertFalse(row["full_box_accuracy_verified"])
 def test_calibration_uses_in_box_occupancy_not_total_foreground(self):
  t=SamSupportedBoxes([{"track_id":"a","bbox_native":[20,0,145,100]}])
  mask=np.zeros((150,180),bool);mask[0:50,0:100]=True
  t.update(0,0,[],{"a":mask})
  c={"candidate_id":"partial","score":.99,"refined_quality_score":.99,"bbox_full_xyxy":[20,0,145,100]}
  row=t.update(1,1/30,[c],{"a":mask})["tracks"][0]
  self.assertEqual(row["calibration_frames"],[0])
 def test_near_copy_shared_mask_is_ambiguous_not_two_independent_supports(self):
  a=np.zeros((80,80),bool);a[20:30,20:40]=True;b=a.copy();b[20,20]=False
  t=SamSupportedBoxes([{"track_id":"a","bbox_native":[18,18,32,32]},
                       {"track_id":"b","bbox_native":[28,18,42,32]}])
  rows=t.update(0,0,[],{"a":a,"b":b})["tracks"]
  self.assertTrue(all(not x["sam_current_support"] for x in rows))
  self.assertTrue(all(x["support_rejection"]=="ambiguous_shared_instance_support" for x in rows))
 def test_shared_mask_threshold_boundary_and_below(self):
  for columns,ambiguous in [(18,True),(17,False)]:
   a=np.zeros((60,60),bool);a[20:30,20:40]=True;b=a.copy();b[:,20+columns:]=False
   t=SamSupportedBoxes([{"track_id":"a","bbox_native":[18,18,32,32]},
                        {"track_id":"b","bbox_native":[28,18,42,32]}])
   rows=t.update(0,0,[],{"a":a,"b":b})["tracks"]
   self.assertEqual(all(x["support_rejection"]=="ambiguous_shared_instance_support" for x in rows),ambiguous)
 def test_empty_and_invalid_mask(self):
  clean,r=geometry_support(np.zeros((3,4),bool));self.assertEqual(r["removed_pixels"],0)
  with self.assertRaises(ValueError):geometry_support(np.zeros((3,4),np.uint8))
 def test_stray_pixels_cannot_authorize_body_calibration(self):
  t=SamSupportedBoxes([{"track_id":"a","bbox_native":[80,80,180,180]}])
  m=np.zeros((500,500),bool);m[100:150,100:150]=True
  t.update(0,0,[],{"a":m})
  m[450,450]=True
  candidate={"candidate_id":"inflated","score":.99,"refined_quality_score":.99,"bbox_full_xyxy":[55,55,205,205]}
  row=t.update(1,1/30,[candidate],{"a":m})["tracks"][0]
  self.assertEqual(row["calibration_frames"],[0])
  self.assertEqual(row["support_geometry"]["removed_pixels"],1)
if __name__=="__main__":unittest.main()
