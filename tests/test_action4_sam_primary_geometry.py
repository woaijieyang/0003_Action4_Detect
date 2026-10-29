import unittest,json,hashlib
import numpy as np
from model.action4_sam_supported_boxes import SamSupportedBoxes

SEEDS=[{'track_id':'a','bbox_native':[35,35,85,75]}]
def fixture_inputs(count=24):
 for f in range(count):
  mask=np.zeros((160,220),bool);mask[40:70,40+f:80+f]=True
  jitter=(-1 if f%2 else 1)*4
  c={'candidate_id':f'c{f}','score':.95,'refined_quality_score':.9,'bbox_full_xyxy':[35+f+jitter,35,85+f+jitter,75]}
  yield f,f/30,[c],{'a':mask}
def center(row):
 b=np.array(row['bbox_for_roi']);return (b[:2]+b[2:])/2

class PrimaryGeometryTests(unittest.TestCase):
 def test_default_matches_frozen_v007_numerical_golden(self):
  t=SamSupportedBoxes(SEEDS);rows=[t.update(*x) for x in fixture_inputs()]
  digest=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()
  self.assertEqual(digest,'f24d1e74d2853f8233e3d9b8b5e48339e65f80b7420d3cef4bfbfbb17667bd37')
  explicit=SamSupportedBoxes(SEEDS,geometry_mode='legacy')
  self.assertEqual(rows,[explicit.update(*x) for x in fixture_inputs()])
 def test_constant_mask_and_jittering_B0_reduce_output_jitter(self):
  sequences=[]
  for mode in ['legacy','sam_primary_v1']:
   t=SamSupportedBoxes(SEEDS,geometry_mode=mode);centers=[]
   for f in range(40):
    m=np.zeros((160,220),bool);m[40:70,40:80]=True;dx=5*(-1 if f%2 else 1)
    c={'candidate_id':str(f),'score':.8,'refined_quality_score':.7,'bbox_full_xyxy':[35+dx,35,85+dx,75]}
    row=t.update(f,f/30,[c],{'a':m})['tracks'][0];centers.append(center(row))
    self.assertEqual(row['raw_detection'],c)
   sequences.append(np.array(centers))
  self.assertLess(np.mean(np.linalg.norm(np.diff(sequences[1][5:],axis=0),axis=1)),.2*np.mean(np.linalg.norm(np.diff(sequences[0][5:],axis=0),axis=1)))
 def test_moving_SAM_centroid_moves_geometry_without_B0_motion(self):
  t=SamSupportedBoxes(SEEDS,geometry_mode='sam_primary_v1');positions=[]
  for f in range(20):
   m=np.zeros((160,220),bool);m[40:70,40+f:80+f]=True
   c={'candidate_id':str(f),'score':.8,'refined_quality_score':.7,'bbox_full_xyxy':[35,35,85,75]}
   row=t.update(f,f/30,[c],{'a':m})['tracks'][0];positions.append(center(row)[0])
  self.assertGreater(positions[-1]-positions[0],17)
  self.assertEqual(row['geometry_trace']['velocity_source'],'contiguous_raw_SAM_centroids')
  self.assertEqual(row['geometry_trace']['raw_SAM_history_frames'],list(range(15,20)))
  self.assertAlmostEqual(row['geometry_trace']['velocity_native_px_per_second'][0],30.,places=8)
 def test_size_outlier_bounded_and_enclosure_preserves_visible_extent(self):
  t=SamSupportedBoxes(SEEDS,geometry_mode='sam_primary_v1')
  for f in range(6):
   m=np.zeros((160,220),bool);m[40:70,40:80]=True
   c={'candidate_id':str(f),'score':.99,'refined_quality_score':.95,'bbox_full_xyxy':[35,35,85,75]}
   row=t.update(f,f/30,[c],{'a':m})['tracks'][0]
  c=dict(c,candidate_id='outlier',bbox_full_xyxy=[30,30,90,80]);row=t.update(6,.2,[c],{'a':m})['tracks'][0]
  b=np.array(row['bbox_for_roi']);self.assertLess(b[2]-b[0],52)
  turned=np.zeros_like(m);turned[20:90,45:75]=True
  row=t.update(7,7/30,[],{'a':turned})['tracks'][0];b=row['bbox_for_roi']
  self.assertLessEqual(b[1],20);self.assertGreaterEqual(b[3],90)
  self.assertTrue(row['geometry_trace']['visible_enclosure_override']);self.assertFalse(row['full_box_accuracy_verified'])
 def test_offset_ema_and_raw_calibration_evidence_are_explicit(self):
  t=SamSupportedBoxes(SEEDS,geometry_mode='sam_primary_v1',window=10)
  for f in range(8):
   m=np.zeros((160,220),bool);m[40:70,40:80]=True
   c={'candidate_id':str(f),'score':.99,'refined_quality_score':.95,'bbox_full_xyxy':[38,35,88,75]}
   out=t.update(f,f/30,[c],{'a':m});trace=out['tracks'][0]['geometry_trace']
   self.assertTrue(trace['raw_calibration_evidence']['accepted'])
   self.assertEqual(trace['raw_calibration_evidence']['raw_B0_box'],c['bbox_full_xyxy'])
  self.assertEqual(out['geometry_policy'],{'offset_ema_alpha':.1,'calibration_window':5,'raw_SAM_velocity_window':5,'center_alpha':.65,'log_size_alpha':.2})
  self.assertEqual(list(t.states['a']['calibration_frames']),list(range(3,8)))
 def test_missing_duplicate_mask_and_frame_gaps_do_not_certify_SAM(self):
  seeds=SEEDS+[{'track_id':'b','bbox_native':[100,35,150,75]}]
  t=SamSupportedBoxes(seeds,geometry_mode='sam_primary_v1');m=np.zeros((160,220),bool);m[40:70,40:80]=True
  rows=t.update(0,0,[],{'a':m,'b':m.copy()})['tracks'];self.assertTrue(all(not r['sam_current_support'] for r in rows))
  for f in range(1,6):out=t.update(f,f/30,[],{})
  self.assertTrue(all(r['status']=='lost' and r['bbox_for_roi'] is None for r in out['tracks']))
  with self.assertRaises(ValueError):t.update(7,7/30,[],{})
  with self.assertRaises(ValueError):SamSupportedBoxes(SEEDS,geometry_mode='other')
 def test_invalid_mask_breaks_raw_SAM_velocity_chain(self):
  t=SamSupportedBoxes(SEEDS,geometry_mode='sam_primary_v1')
  for f in range(3):
   m=np.zeros((160,220),bool);m[40:70,40+f:80+f]=True;t.update(f,f/30,[],{'a':m})
  t.update(3,.1,[],{})
  row=t.update(4,4/30,[],{'a':m})['tracks'][0]
  self.assertEqual(row['geometry_trace']['raw_SAM_history_frames'],[4])
  self.assertNotEqual(row['geometry_trace']['velocity_source'],'contiguous_raw_SAM_centroids')

if __name__=='__main__':unittest.main()
