import unittest,copy
from tests.test_action4_surface_snapshot import SurfaceSnapshotTests as Base
from data.action4_frozen_detection_context import FrozenDetectionContext
from eval.action4_surface_snapshot import model_identities
class FrozenReplay(Base):
 def context(self,rows=None):
  self.p.b0_scoring_policy='center'
  return FrozenDetectionContext.from_frame(self.frame,self.K,self.p._detect.return_value if rows is None else rows,
   b0_sha256=model_identities(self.p._surface_model_bindings)['B0']['sha256'],scoring_policy='center',source_cache_identity='c'*64)
 def test_exact_225_without_B0_call(self):
  expected=self.p.predict(self.rgb,self.K,self.cad)
  ctx=self.context();self.p._detect.reset_mock()
  actual=self.p.predict(self.rgb,self.K,self.cad,frozen_detections=ctx,roi_expansion=2.25)
  self.p._detect.assert_not_called();self.assertEqual(expected['instances'],actual['instances'])
 def test_empty_cache_stays_empty(self):
  ctx=self.context([]);self.p._detect.reset_mock()
  result=self.p.predict(self.rgb,self.K,self.cad,frozen_detections=ctx)
  self.assertEqual(result['instances'],[]);self.p._detect.assert_not_called()
 def test_tracking_packet_and_changed_RGB_rejected(self):
  rows=copy.deepcopy(self.p._detect.return_value);rows[0]['track_id']='left'
  with self.assertRaises(ValueError):self.context(rows)
  ctx=self.context();rgb=self.rgb.copy();rgb[0,0,0]^=1
  with self.assertRaises(ValueError):self.p.predict(rgb,self.K,self.cad,frozen_detections=ctx)
del Base
