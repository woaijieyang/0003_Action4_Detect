import unittest,json,sys
from copy import deepcopy
from pathlib import Path
ROOT=Path('/data/zy/vln/a-test');sys.path.insert(0,str(ROOT/'src'))
from data.action4_template_runtime import validate_proposal_binding
class BindingTests(unittest.TestCase):
 def setUp(self):
  self.meta=json.loads((ROOT/'data/action4_scene_templates/v007_candidate/wall/template.json').read_text());self.anchor=json.loads((ROOT/self.meta['baseline_parameters_ref']['path']).read_text())
  with (ROOT/'outputs/action4_diverse_sampling_v007/fixed_scene_v005/wall/parameters.jsonl').open() as f:self.p=json.loads(next(f))
 def test_normal_source_binds(self):self.assertTrue(validate_proposal_binding(self.p,self.meta,self.anchor,'wall'))
 def test_fixed_installation_mutations_rejected(self):
  for side in ('left','right'):
   for key in ('A_brace_from_CAD','S_skin_relative_to_brace','F_body_from_brace'):
    p=deepcopy(self.p);p['sides'][side][key][0][3]+=.001
    with self.assertRaises(ValueError):validate_proposal_binding(p,self.meta,self.anchor,'wall')
 def test_source_identity_and_point_order_rejected(self):
  for kind in ('anchor_ref','anchor_digest','canonical_ref','canonical_order','corner_ids'):
   p=deepcopy(self.p)
   if kind=='anchor_ref':p['provenance']['anchor_ref']['sha256']='0'*64
   elif kind=='anchor_digest':p['provenance']['anchor_parameter_digest']='0'*64
   elif kind=='canonical_ref':p['canonical_source_ref']['sha256']='0'*64
   elif kind=='canonical_order':p['canonical8_m'][0],p['canonical8_m'][1]=p['canonical8_m'][1],p['canonical8_m'][0]
   else:p['corner_ids'][0],p['corner_ids'][1]=p['corner_ids'][1],p['corner_ids'][0]
   with self.assertRaises(ValueError):validate_proposal_binding(p,self.meta,self.anchor,'wall')
 def test_actor_pose_and_presence_consistency(self):
  p=deepcopy(self.p);p['actor_pose_capture_by_side']['left']['core_C_capture'][0][3]+=.01
  with self.assertRaises(ValueError):validate_proposal_binding(p,self.meta,self.anchor,'wall')
  p=deepcopy(self.p);p['sides']['left']['brace_B_capture'][1][3]+=.01
  with self.assertRaises(ValueError):validate_proposal_binding(p,self.meta,self.anchor,'wall')
  p=deepcopy(self.p);p['entry']['target_presence']=[False,True]
  with self.assertRaises(ValueError):validate_proposal_binding(p,self.meta,self.anchor,'wall')
 def test_nonfinite_or_partial_geometry_rejected(self):
  for kind in ('exposure','gain','corner','patch','capture'):
   p=deepcopy(self.p)
   if kind=='exposure':p['environment_controls']['exposure_offset_ev']=float('nan')
   elif kind=='gain':p['environment_controls']['area_light_gain']=float('inf')
   elif kind=='corner':p['derived']['by_side']['left']['canonical8_native_px'][0][0]=float('nan')
   elif kind=='patch':p['occluder_proposals']['left']['vertices_world_m']=p['occluder_proposals']['left']['vertices_world_m'][:1]
   else:p['occluder_proposals']['left']['vertices_capture_m'][0][0]+=1
   with self.assertRaises(ValueError):validate_proposal_binding(p,self.meta,self.anchor,'wall')
 def test_synchronized_wrong_delta_rejected(self):
  p=deepcopy(self.p)
  p['actor_pose_capture_by_side']['left']['whole_actor_delta_capture'][0][3]+=10
  p['sides']['left']['whole_actor_delta_capture'][0][3]+=10
  with self.assertRaises(ValueError):validate_proposal_binding(p,self.meta,self.anchor,'wall')
if __name__=='__main__':unittest.main()
