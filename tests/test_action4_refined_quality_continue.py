"""CPU-only reuse/call-ledger failure contracts; no model forward."""
import copy,unittest
from unittest.mock import Mock,patch
import numpy as np
from data.action4_sam_native_export import canonical
from eval.action4_refined_quality_sidecar import quality_rows
from eval.action4_b0_scoring import REFINED_QUALITY_PROTOCOL
from eval.action4_refined_quality_continue import reuse_capture,validate_ledger

class ContinueTests(unittest.TestCase):
 def test_reuse_does_not_invoke_predict_and_retains_quality(self):
  base=[{"grid_index":2,"score":.8,"bbox_full_xyxy":[0.,0.,2.,2.]}]
  q=quality_rows("a",base,[dict(base[0],refined_quality_score=.91,refined_quality_protocol=REFINED_QUALITY_PROTOCOL)])
  consumer=Mock();consumer.consume.return_value={"scope":"pilot32","admitted":False,"qualities":q}
  image={"sample_id":"a","K":np.eye(3).tolist(),"shape_hw":[2,2]}
  with patch("eval.action4_refined_quality_sidecar.predict",side_effect=AssertionError("no inference")):
   got=reuse_capture(consumer,image,np.zeros((2,2,3),np.uint8),base,"a"*64)
  self.assertEqual(got["enriched"][0]["refined_quality_score"],.91)
  q[0]["grid_index"]=3
  with self.assertRaises(ValueError):reuse_capture(consumer,image,np.zeros((2,2,3),np.uint8),base,"a"*64)
 def rows(self):
  ref={"path":"fixture","sha256":"a"*64}
  return [{"sample_id":sid,"method":method,"source_receipt":ref,"candidates":2} for sid,method in
   (("a","pilot_reuse"),("b","new_B0_forward"),("c","new_B0_forward"))]
 def test_partial_failure_never_becomes_full_success(self):
  rows=self.rows()
  self.assertEqual(validate_ledger(["a","b","c"],rows[:2],["a"])["completed_images"],2)
  with self.assertRaises(ValueError):validate_ledger(["a","b","c"],rows[:2],["a"],complete=True)
  self.assertEqual(validate_ledger(["a","b","c"],rows,["a"],complete=True),
   {"completed_images":3,"reused_images":1,"new_forward_images":2})
 def test_duplicate_wrong_order_wrong_method_missing_provenance_rejected(self):
  rows=self.rows()
  bad=[rows[0],rows[0]]
  with self.assertRaises(ValueError):validate_ledger(["a","b","c"],bad,["a"])
  with self.assertRaises(ValueError):validate_ledger(["a","b","c"],rows[::-1],["a"])
  bad=copy.deepcopy(rows);bad[1]["method"]="pilot_reuse"
  with self.assertRaises(ValueError):validate_ledger(["a","b","c"],bad,["a"])
  bad=copy.deepcopy(rows);bad[0].pop("source_receipt")
  with self.assertRaises(ValueError):validate_ledger(["a","b","c"],bad,["a"])
if __name__=="__main__":unittest.main()
