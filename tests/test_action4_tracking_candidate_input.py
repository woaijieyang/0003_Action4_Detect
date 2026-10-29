"""Association pool transport tests; no claim of tracking accuracy."""
import copy,json,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from eval.action4_b0_scoring import predict,predict_candidate_pool,association_pool
from eval.action4_soft_nms import select
R=Path("/data/zy/vln/a-test")
class PoolTests(unittest.TestCase):
 def setUp(self):
  self.rows=[{"grid_index":i,"center_score":s,"old_product_score":s*.8,"old_quality_score":.8,
      "bbox_full_xyxy":[i*30.,0.,i*30.+20.,20.],"center_px":[i*30.+10.,10.]} for i,s in enumerate([.95,.79,.76])]
 def test_third_candidate_kept_before_identity_association(self):
  pool=association_pool([dict(x,score=x["center_score"]) for x in self.rows])
  self.assertEqual(len(pool["single_frame_selected"]),2)
  self.assertEqual(len(pool["association_candidates"]),3)
  self.assertEqual(pool["association_candidates"][2]["grid_index"],2)
  self.assertFalse(pool["identity_confirmed"]);self.assertFalse(pool["tracking_applied"])
  pool["single_frame_selected"][0]["score"]=0
  self.assertEqual(pool["association_candidates"][0]["score"],.95)
 def test_one_forward_and_actual_appearance_owner(self):
  class Appearance(torch.nn.Module):
   def __init__(self):super().__init__();self.appearance=torch.nn.Identity()
  class Refiner(torch.nn.Module):
   def __init__(self):super().__init__();self.appearance=Appearance()
  class Model(torch.nn.Module):
   def __init__(self):super().__init__();self.refiner=Refiner();self.calls=0
   def forward(self,x):self.calls+=1;return {},torch.zeros(1)
  model=Model().eval();features=torch.arange(384,dtype=torch.float32).reshape(3,128)
  boxes=torch.tensor([x["bbox_full_xyxy"] for x in self.rows])
  def proposal(m,*args):
   m.refiner.appearance.appearance(features)
   return copy.deepcopy(self.rows),boxes,(boxes,torch.zeros(3),torch.zeros(3,4))
  with patch("eval.action4_b0_scoring.prepare_full_images",return_value=(torch.zeros(1),[torch.eye(3)])),patch("eval.action4_b0_scoring.proposals",side_effect=proposal):
   baseline=predict(model,None,{"preprocess":{"full_size":[10,10]}},"cpu","center")
   self.assertEqual(baseline,select([dict(x,score=x["center_score"]) for x in self.rows],"soft_gaussian50"))
   model.calls=0;pool=predict_candidate_pool(model,None,{"preprocess":{"full_size":[10,10]}},"cpu","center")
  self.assertEqual(model.calls,1)
  self.assertEqual(len(model.refiner.appearance.appearance._forward_hooks),0)
  for row,vector in zip(pool["association_candidates"],features):
   torch.testing.assert_close(torch.tensor(row["appearance_128"]),torch.nn.functional.normalize(vector,dim=0))
   self.assertEqual(row["refined_quality_score"],.5)
 def test_bad_id_or_training_mode_rejected(self):
  rows=[dict(x,score=x["center_score"]) for x in self.rows];rows[1]["grid_index"]=0
  with self.assertRaises(ValueError):association_pool(rows)
  with self.assertRaises(ValueError):predict_candidate_pool(torch.nn.Linear(1,1),None,{},"cpu","center")
 def test_saved_real_pool_preserves_right_candidate(self):
  data=json.loads((R/"outputs/action4_real_failure_audit_v001/FRAME2400_CANDIDATES.json").read_text())
  result=association_pool(data["raw_candidates"])
  self.assertEqual(result["single_frame_selected"],data["formal_selected_two"])
  self.assertEqual(result["decoded_candidate_count"],30)
  self.assertIn(34730,[r["grid_index"] for r in result["association_candidates"]])
  self.assertNotIn(34730,[r["grid_index"] for r in result["single_frame_selected"]])
if __name__=="__main__":unittest.main(verbosity=2)
