"""G1B pure contract tests; no Blender/GPU/model construction."""
import copy,json,unittest
from pathlib import Path
from data import action4_global_environment as env
class EnvironmentContracts(unittest.TestCase):
 def setUp(self):self.cfg=json.loads((env.ROOT/'configs/data/action4_global_environment_init_v001.json').read_text())
 def test_valid_sources_and_observer_K(self):
  env.validate_config(self.cfg);self.assertEqual(env.expected_K(self.cfg),[[2639.,0,1623.5],[0,2639.,1231.5],[0,0,1]])
 def test_no_fitted_or_generation_claim(self):
  for key,val in [('status','fitted'),('dataset_generation_allowed',True)]:
   c=copy.deepcopy(self.cfg);c[key]=val
   with self.assertRaises(ValueError):env.validate_config(c,False)
 def test_no_family_or_nested_override(self):
  for top in (True,False):
   c=copy.deepcopy(self.cfg)
   if top:c['families']=[]
   else:c['initial_priors']['capture']['scene_focal_mm']=20
   with self.assertRaises(ValueError):env.validate_config(c,False)
 def test_pose_allows_extrinsics_only(self):
  good={'location_m':[0,-.3,1.6],'target_m':[0,-.9,.75],'roll_deg':0};env.validate_pose(good)
  for key,val in [('focal_mm',20),('table_size_xy_m',[1,1]),('scale',2),('exposure',1)]:
   with self.assertRaises(ValueError):env.validate_pose({**good,key:val})
 def test_invalid_pose(self):
  for bad in [{'location_m':[0,0,0],'target_m':[0,0,0],'roll_deg':0},{'location_m':[float('nan'),0,1],'target_m':[0,-1,0],'roll_deg':0}]:
   with self.assertRaises(ValueError):env.validate_pose(bad)
 def test_invalid_shared_extents_and_clearance(self):
  for key,val in [('table_size_xy_m',[-1,.8]),('table_top_z_m',-.1),('table_center_xy_m',[0,1]),('wall_center_y_m',-.8),('floor_size_xy_m',[.5,.5])]:
   c=copy.deepcopy(self.cfg);c['initial_priors']['geometry'][key]=val
   with self.assertRaises(ValueError):env.validate_config(c,False)
 def test_global_changes_are_config_not_frame_parameters(self):
  c=copy.deepcopy(self.cfg);c['initial_priors']['capture']['focal_mm']=22;env.validate_config(c,False)
  self.assertNotEqual(env.digest(c),env.digest(self.cfg));self.assertNotEqual(env.expected_K(c),env.expected_K(self.cfg))
 def test_source_mismatch_fails(self):
  c=copy.deepcopy(self.cfg);c['sources']['rig_asset']['sha256']='0'*64
  with self.assertRaises(ValueError):env.validate_config(c)
 def test_bool_and_nonfinite_not_dimensions(self):
  for val in (True,float('nan'),float('inf')):
   c=copy.deepcopy(self.cfg);c['initial_priors']['capture']['focal_mm']=val
   with self.assertRaises(ValueError):env.validate_config(c,False)
 def test_source_appearance_inputs_preserved(self):
  before=copy.deepcopy(self.cfg);env.validate_config(self.cfg);self.assertEqual(before,self.cfg)
 def test_blender_representation_limits_are_not_calibration_ranges(self):
  for key in ('focal_mm','sensor_width_mm'):
   c=copy.deepcopy(self.cfg);c['initial_priors']['capture'][key]=.5
   with self.assertRaises(ValueError):env.validate_config(c,False)
  with self.assertRaises(ValueError):env.validate_pose({'location_m':[1e100,0,1],'target_m':[0,-1,0],'roll_deg':0})
 def test_no_path_escape(self):
  with self.assertRaises(ValueError):env.bounded('../outside')
if __name__=='__main__':unittest.main()
