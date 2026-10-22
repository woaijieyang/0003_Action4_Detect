"""Global-rig contract checks; no Blender render or training acceptance."""
import copy,json,unittest
from pathlib import Path
from data.action4_global_rig import validate_config,validate_pose,invariant_contract
ROOT=Path("/data/zy/vln/a-test")
class GlobalRigContractTests(unittest.TestCase):
 def setUp(self):
  self.cfg=json.loads((ROOT/"configs/data/action4_global_rig_init_v001.json").read_text())
 def test_current_config_inputs_are_real_and_unchanged(self):
  validate_config(self.cfg,check_files=True)
 def test_global_pose_accepts_only_angles(self):
  validate_pose({"left_forearm":[0,12,0],"right_hand":[5,0,0]})
  for key in ["camera_scale","bone_lengths","left_brace_A","family","observer_focal"]:
   with self.subTest(key=key),self.assertRaises(ValueError):validate_pose({key:[0,0,0]})
 def test_pose_bounds_nonfinite_bool_and_wrong_shape(self):
  for v in [[26,0,0],[float("nan"),0,0],[True,0,0],[0,0],{"x":0}]:
   with self.subTest(v=v),self.assertRaises(ValueError):validate_pose({"left_hand":v})
 def test_reject_per_scenario_parameters(self):
  for key in ["families","per_frame","camera_scale"]:
   cfg=copy.deepcopy(self.cfg);cfg[key]={}
   with self.subTest(key=key),self.assertRaises(ValueError):validate_config(cfg,False)
 def test_reject_body_scaling_and_bool_alias(self):
  for v in [.78,1.1,True]:
   cfg=copy.deepcopy(self.cfg);cfg["body"]["source_scale"]=v
   with self.subTest(v=v),self.assertRaises(ValueError):validate_config(cfg,False)
 def test_no_camera_scale_or_mount_family_override(self):
  for location,key,value in [("camera","scale",1),("attachment","families",{}),("brace","per_side_override",{})]:
   cfg=copy.deepcopy(self.cfg);cfg[location][key]=value
   with self.subTest(key=key),self.assertRaises(ValueError):validate_config(cfg,False)
 def test_not_fitted_or_generation_accepted(self):
  cfg=copy.deepcopy(self.cfg);cfg["dataset_generation_allowed"]=True
  with self.assertRaises(ValueError):validate_config(cfg,False)
  cfg=copy.deepcopy(self.cfg);cfg["status"]="fitted"
  with self.assertRaises(ValueError):validate_config(cfg,False)
 def test_input_hash_tamper_rejected(self):
  cfg=copy.deepcopy(self.cfg);cfg["camera"]["foundation"]["sha256"]="0"*64
  with self.assertRaisesRegex(ValueError,"identity changed"):validate_config(cfg,True)
 def test_contract_hash_binds_attachment_and_bones(self):
  r=json.loads((ROOT/"outputs/action4_realism_reassessment_v001/global_rig_init/build_v002/report.json").read_text())
  first=invariant_contract(r);self.assertIsNone(first["pose_input"]["calibrated_sampling_ranges"])
  altered=copy.deepcopy(r);altered["brace_and_fixed_attachment"]["left"]["A_brace_camera"][0][3]+=.001
  self.assertNotEqual(first["invariants_sha256"],invariant_contract(altered)["invariants_sha256"])
  altered=copy.deepcopy(r);altered["rest_lengths_m"]["left_forearm"]+=.001
  self.assertNotEqual(first["invariants_sha256"],invariant_contract(altered)["invariants_sha256"])
if __name__=="__main__":unittest.main(verbosity=2)
