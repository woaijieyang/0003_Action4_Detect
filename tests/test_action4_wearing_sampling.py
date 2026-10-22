"""CPU-only wearing factory, rigid/presence and two-stage restoration checks."""
import sys,json,unittest,os
from pathlib import Path
from copy import deepcopy
from unittest.mock import patch
import numpy as np
ROOT=Path('/data/zy/vln/a-test');sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'tests'))
from data import action4_modular_scene_sampling as m
from test_action4_modular_scene_sampling import fixture
CONFIG=os.environ.get('ACTION4_WEARING_SAMPLING_CONFIG','outputs/action4_wearing_sampling_prepare_v001/release_v003/sampling_base_v001.json')
class WearingSamplingTests(unittest.TestCase):
 def test_01_explicit_real_factory_and_new_cable(self):
  c=m.prepare_base(CONFIG);self.assertEqual(c['base_kind'],'palmar_wearing');self.assertEqual(c['source_context']['base_kind'],'palmar_wearing');m.verify_sources(c)
  self.assertEqual(c['bilateral_context']['right_context']['mount_contract']['objects']['cable']['name'],'wearing_right_cable_rail_v001')
  for side in ('left','right'):
   self.assertEqual(c['source_context']['sides'][side]['A_brace_from_CAD'],c['wearing_context']['frames'][side]['Anew'])
   self.assertEqual(c['source_context']['sides'][side]['base_B_capture'],c['wearing_context']['frames'][side]['Bnew'])
  bad=deepcopy({k:v for k,v in c.items() if k not in ('wearing_context','bilateral_context')});bad['base_kind']='bilateral'
  with self.assertRaises(ValueError):m.verify_sources(bad)
 def test_02_all_presence_keep_old_cable_hidden_and_move_new(self):
  for presence in [(False,False),(True,False),(False,True),(True,True)]:
   c,p,base,bind=fixture(presence)
   old='superseded_cable';new='replacement_cable'
   for name in (old,new):base['objects'][name]=deepcopy(base['objects']['right_mount']);bind['right']['names'].append(name)
   base['objects'][old]['hide_render']=True;bind['right']['hidden_when_absent'].append(new);bind['right']['always_hidden']=[old]
   e=m.expected_state(base,bind,p,c['source_context']);self.assertTrue(e['objects'][old]['hide_render']);self.assertEqual(e['objects'][new]['hide_render'],not presence[1])
   if presence[1]:self.assertAlmostEqual(e['objects'][new]['world'][0][3],-.002)
   self.assertEqual(e['objects'][new]['basis'],base['objects'][new]['basis']);self.assertEqual(e['objects'][new]['mesh'],base['objects'][new]['mesh'])
   bad=deepcopy(e);bad['objects'][old]['hide_render']=False
   with self.assertRaises(ValueError):m.verify_state(bad,e)
 def test_03_actual_wearing_fixture_local_matrices_A_P_C_retained(self):
  c=m.prepare_base(CONFIG);contract=json.loads((ROOT/'outputs/action4_wearing_actual_v001/runs/cpu_probe_v001/scene_contract.json').read_text());base=contract['expected_state'];bind=json.loads((ROOT/'outputs/action4_modular_layout_actual_v002/runs/cpu_probe_v001/cases/cluster_0/dynamic_contract.json').read_text())['actor_bindings'];bind['right']['names'].append('wearing_right_cable_rail_v001');bind['right']['hidden_when_absent'].append('wearing_right_cable_rail_v001');bind['right']['always_hidden']=['mount_right_cable']
  p={'target_presence':[True,True],'sides':{},'environment_controls':{'area_light_gain':1.,'exposure_offset_ev':0.,'wall_visible':True,'table_visible':True}}
  W=np.asarray(c['source_context']['W_from_capture'])
  for side in ('left','right'):
   B=np.asarray(c['source_context']['sides'][side]['base_B_capture']).copy();B[0,3]+=.003;p['sides'][side]={'brace_B_capture':B.tolist()}
  e=m.expected_state(base,bind,p,c['source_context'])
  for side in ('left','right'):
   delta=W@np.asarray(p['sides'][side]['brace_B_capture'])@np.linalg.inv(W@np.asarray(c['source_context']['sides'][side]['base_B_capture']))
   for name in bind[side]['names']:
    np.testing.assert_allclose(e['objects'][name]['world'],delta@base['objects'][name]['world'],atol=1e-14)
    if name not in bind[side]['roots']:self.assertEqual(e['objects'][name]['basis'],base['objects'][name]['basis'])
    self.assertEqual(e['objects'][name]['mesh'],base['objects'][name]['mesh'])
   np.testing.assert_allclose(e['objects'][side+'_Action4_core']['basis'],c['wearing_context']['frames'][side]['Anew'],atol=0)
  self.assertEqual(e['K'],base['K'])
 def test_04_two_stage_restore_order_and_failure_retention(self):
  for bad in (False,True):
   calls=[];session=object.__new__(m.AppliedScene);session.restored=False
   class Base:
    def restore(self):calls.append('original');return {'status':'original_verified'}
   session.base_session=Base()
   def first():
    calls.append('wearing')
    if bad:raise ValueError('wearing verification failed')
   session._restore_to_base=first
   if bad:
    with self.assertRaisesRegex(RuntimeError,'wearing verification failed'):session.restore()
   else:
    self.assertEqual(session.restore()['status'],'RESTORED_WEARING_THEN_ORIGINAL_BASE')
    with self.assertRaises(ValueError):session.restore()
   self.assertEqual(calls,['wearing','original'])
 def test_05_pending_six_families_still_rejected(self):
  for family in m.PENDING_FAMILY_EFFECTS:
   c,p,_,_=fixture();p['entry']['scene_family']=family;p['scene_family_controls']['family']=family
   with self.assertRaises(m.PendingFamilyError):m.validate_parameters(p,c)
 def test_06_wearing_absence_keeps_white_module_and_ball(self):
  c=m.prepare_base(CONFIG)
  for i,side in enumerate(('left','right'),1):
   mc=c['bilateral_context']['left_mount_contract'] if side=='left' else c['bilateral_context']['right_context']['mount_contract'];hidden=m._hidden_when_absent(mc,i,True)
   for role in ('quick_release','usb_connector','link','cable'):self.assertIn(mc['objects'][role]['name'],hidden)
   for role in ('white_module','ball'):self.assertNotIn(mc['objects'][role]['name'],hidden)
   for part in m.pixels.KNOWN_TARGET_PARTS[i].values():self.assertIn(part,hidden)
   self.assertTrue(set(mc['root_objects'].values()).isdisjoint(hidden))
   legacy=m._hidden_when_absent(mc,i,False)
   self.assertTrue({mc['objects'][r]['name'] for r in ('white_module','ball')}<=legacy)
 def test_07_wrong_factory_SHA_and_legacy_fallback_rejected(self):
  packet=json.loads((ROOT/CONFIG).read_text());packet['base_kind']='bilateral'
  with self.assertRaises(ValueError):m._prepare_wearing_base(ROOT/CONFIG,packet)
  packet=json.loads((ROOT/CONFIG).read_text());packet['wearing_base_config']['sha256']='0'*64
  with self.assertRaises(ValueError):m._prepare_wearing_base(ROOT/CONFIG,packet)
 def test_08_explicit_92quota_audit_and_mismatch_reject(self):
  # In-memory audit fixture only, never a generated/bound production plan.
  from data import action4_modular_parameters as params
  old=json.loads((ROOT/'data/action4_synth/pilot2000_v015_candidate/plan.json').read_text());params.audit_plan(old)
  with self.assertRaises(ValueError):params.audit_plan(old,params.QUOTAS_92)
  candidate=deepcopy(old)
  for split,q in params.QUOTAS_92.items():
   rows=[e for e in candidate['entries'] if e['split']==split]
   for i,e in enumerate(rows):
    count=0 if i<q['0'] else (1 if i<q['0']+q['1'] else 2)
    e['target_count']=count;e['target_presence']={0:[False,False],1:[False,True],2:[True,True]}[count];e['scene_family']='hard_negative' if count==0 else 'wall_rear'
  candidate['instance_counts_by_split']=deepcopy(params.QUOTAS_92);candidate['intended_instance_quotas_by_split']=deepcopy(params.QUOTAS_92)
  self.assertEqual(params.audit_plan(candidate,params.QUOTAS_92)['instance_counts_by_split'],params.QUOTAS_92)
  with self.assertRaises(ValueError):params.audit_plan(candidate,params.QUOTAS)
  bounds=json.loads((ROOT/CONFIG).with_name('diagnostic_legacy_entry_bounds_v002.json').read_text());policy_path='configs/data/action4_pilot2000_presence_quota_v002.json';policy={'path':policy_path,'sha256':m.bilateral.sha(policy_path)}
  bounds['quota_contract']['instance_counts_by_split']=deepcopy(params.QUOTAS_92)
  with self.assertRaises(ValueError):params.quota_counts(bounds)
  bounds['quota_contract']['presence_quota_policy']=policy;bounds['source_refs']['presence_quota_policy']=policy
  self.assertEqual(params.quota_counts(bounds),params.QUOTAS_92)
  with self.assertRaises(ValueError):params.validate_config(bounds) # old entry authority cannot satisfy new quotas
if __name__=='__main__':unittest.main(verbosity=2)
