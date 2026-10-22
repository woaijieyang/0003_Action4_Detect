"""Pure dynamic scene-consumption contracts, no sampler/Blender/GPU invocation."""
from pathlib import Path
from copy import deepcopy
import sys,unittest,numpy as np
R=Path('/data/zy/vln/a-test');sys.dont_write_bytecode=True;sys.path.insert(0,str(R/'src'))
from data import action4_modular_scene_sampling as m

def fixture(presence=(True,True)):
 I=np.eye(4).tolist();K=[[100,0,15.5],[0,100,11.5],[0,0,1]]
 side={'A_brace_from_CAD':I,'S_skin_relative_to_brace':I,'F_body_from_brace':I,'anchor_C_capture_by_frame':{'0':I},'base_C_capture':I,'base_B_capture':I}
 source={'W_from_capture':I,'K_native':K,'sides':{s:deepcopy(side) for s in ['left','right']},'baseline_environment':{'area_energy_w':3.5,'exposure':0.0},'frozen_refs':{}}
 context={'source_context':source,'source_context_sha256':m.env.digest(source)}
 parameters={'schema':'action4.modular_parameters.v1','status':'BOUNDED_AUGMENTATION_CANDIDATE','augmentation':{'counts_as_formal_coverage':False},'estimated':True,'training_accepted':False,'dataset_generation_allowed':False,'W_from_capture':I,'K_native':K,'target_presence':list(presence),'sides':{},'source_anchor_frames_by_side':{},'environment_controls':{'area_light_gain':1.05,'exposure_offset_ev':.1,'wall_visible':True,'table_visible':False},'scene_family_controls':{'family':'wall_rear','required_capabilities':list(m.CAPABILITIES),'pending_reasons':[],'coarse_occluders':[],'effect_claim':'candidate_only_not_verified_family_effect'},'provenance':{'source_context_sha256':context['source_context_sha256']},'entry':{'scene_family':'wall_rear','target_count':sum(presence),'target_presence':list(presence)}}
 for active,s in zip(presence,['left','right']):
  C=np.eye(4);C[0,3]=.001 if s=='left' else -.002
  parameters['sides'][s]={'present':active,**{k:deepcopy(side[k]) for k in ['A_brace_from_CAD','S_skin_relative_to_brace','F_body_from_brace']},'core_C_capture':C.tolist() if active else None,'brace_B_capture':C.tolist() if active else None,'whole_actor_delta_capture':C.tolist() if active else None,'translation_capture_m':C[:3,3].tolist(),'rotation_brace_local_xyz_deg':[0,0,0]}
  parameters['source_anchor_frames_by_side'][s]=0 if active else None
 state={'objects':{},'view':{'exposure':0.0},'K':K,'skin_groups':{'skin':'original_weights'}}
 bindings={}
 for i,s in enumerate(['left','right'],1):
  names=[s+'_root',s+'_core',s+'_decal',s+'_skin',s+'_mount'];bindings[s]={'roots':[names[0]],'names':names,'hidden_when_absent':[names[1],names[2],names[4]]}
  for name in names:state['objects'][name]={'world':I,'basis':I,'scale':[1,1,1],'hide_render':False,'pass_index':i if name.endswith(('core','decal')) else 0,'mesh':{'geometry_uv_sha256':'source','pass_index':i if name.endswith(('core','decal')) else 0},'light':None}
 for name in m.neutral.ENV_NAMES:state['objects'][name]={'world':I,'basis':I,'scale':[1,1,1],'hide_render':False,'pass_index':0,'mesh':None,'light':{'properties':{'energy':3.5}} if name==m.neutral.ENV_NAMES[2] else None}
 return context,parameters,state,bindings

class SamplingConsumerTests(unittest.TestCase):
 def test_prepare_actual_sources_is_CPU_only_and_has_anchor_context(self):
  context=m.prepare_base('configs/data/action4_modular_bilateral_preview_v003.json');s=context['source_context']
  self.assertEqual(set(s['sides']['left']['anchor_C_capture_by_frame']),{'0'});self.assertEqual(set(s['sides']['right']['anchor_C_capture_by_frame']),{'0','540','960'});self.assertNotIn('bpy',sys.modules);m.verify_sources(context)
 def test_presence_sparse_ID_metadata_and_absolute_controls(self):
  for presence in [(False,False),(True,False),(False,True),(True,True)]:
   with self.subTest(presence=presence):
    context,p,base,bind=fixture(presence);m.validate_parameters(p,context);expected=m.expected_state(base,bind,p,context['source_context'])
    for i,side in enumerate(['left','right']):
     for name in bind[side]['names']:
      self.assertEqual(expected['objects'][name]['hide_render'],not presence[i] and name in bind[side]['hidden_when_absent'])
      if not presence[i]:self.assertEqual(expected['objects'][name]['pass_index'],0)
    self.assertEqual(expected['objects'][m.neutral.ENV_NAMES[2]]['light']['properties']['energy'],3.5*1.05)
    self.assertEqual(m.expected_state(base,bind,p,context['source_context']),expected)
 def test_wrong_frames_scale_frozen_contract_and_bounds_rejected(self):
  for change in ['CBA','scale','A','S','F','W','K','anchor','presence','source','translation','rotation','delta']:
   with self.subTest(change=change):
    c,p,_,_=fixture()
    if change=='CBA':p['sides']['left']['core_C_capture'][0][3]+=.01
    elif change=='scale':p['sides']['left']['brace_B_capture'][0][0]=1.1
    elif change in ['A','S','F']:p['sides']['left'][{'A':'A_brace_from_CAD','S':'S_skin_relative_to_brace','F':'F_body_from_brace'}[change]][0][3]=.1
    elif change in ['W','K']:p[{'W':'W_from_capture','K':'K_native'}[change]][0][0]=2
    elif change=='anchor':p['source_anchor_frames_by_side']['left']=540
    elif change=='presence':p['sides']['left']['present']=False
    elif change=='source':p['provenance']['source_context_sha256']='0'*64
    elif change=='translation':p['sides']['left']['translation_capture_m']=[.005,0,0]
    elif change=='rotation':p['sides']['left']['rotation_brace_local_xyz_deg']=[9,0,0]
    else:p['sides']['left']['whole_actor_delta_capture'][0][3]=.002
    with self.assertRaises(ValueError):m.validate_parameters(p,c)
 def test_all_missing_family_effects_failclosed_and_occluders_not_ignored(self):
  for family in m.FAMILIES:
   c,p,_,_=fixture();p['entry']['scene_family']=family;p['scene_family_controls']['family']=family
   if family in ('wall_rear','hard_negative'):
    if family=='hard_negative':
     c,p,_,_=fixture((False,False));p['entry']['scene_family']=family;p['scene_family_controls']['family']=family
    m.validate_parameters(p,c)
   else:
    with self.assertRaises(m.PendingFamilyError):m.validate_parameters(p,c)
  for key,value in [('coarse_occluders',[{'kind':'cloth_patch'}]),('pending_reasons',['missing actual effect']),('required_capabilities',['cloth_patch'])]:
   c,p,_,_=fixture();p['scene_family_controls'][key]=value
   with self.assertRaises(m.PendingFamilyError):m.validate_parameters(p,c)
 def test_global_unknown_reference_bounds_and_baseline_coverage_gate(self):
  c,p,_,_=fixture();p['status']='REFERENCE_BOUNDS_UNKNOWN_PENDING'
  with self.assertRaises(m.PendingFamilyError):m.validate_parameters(p,c)
  c,p,_,_=fixture();p['status']='BASELINE_REGRESSION_ENGINEERING_ONLY';p['augmentation']={'counts_as_formal_coverage':True}
  with self.assertRaises(ValueError):m.validate_parameters(p,c)
  p['augmentation']['counts_as_formal_coverage']=False;m.validate_parameters(p,c)
  c,p,_,_=fixture();p['status']='BOUNDED_REFERENCE_CANDIDATE';p['augmentation'].update(reference_bounds_known=False,reference_bounds_status='pending_evidence',reference_bounds=None,reference_evidence_refs=[])
  with self.assertRaises(m.PendingFamilyError):m.validate_parameters(p,c)
 def test_expected_world_moves_entire_actor_and_keeps_child_local_sources(self):
  c,p,base,bind=fixture();expected=m.expected_state(base,bind,p,c['source_context'])
  for s in ['left','right']:
   for name in bind[s]['names']:
    self.assertAlmostEqual(expected['objects'][name]['world'][0][3],p['sides'][s]['brace_B_capture'][0][3])
    if name not in bind[s]['roots']:self.assertEqual(expected['objects'][name]['basis'],base['objects'][name]['basis'])
    self.assertEqual(expected['objects'][name]['mesh'],base['objects'][name]['mesh'])
  self.assertEqual(expected['K'],base['K']);self.assertEqual(expected['skin_groups'],base['skin_groups'])
 def test_independent_dynamic_contract_rejects_true_nonallowed_changes(self):
  c,p,base,bind=fixture();expected=m.expected_state(base,bind,p,c['source_context']);m.verify_state(expected,expected)
  near=deepcopy(expected);near['objects']['left_root']['world'][0][3]+=1e-7;m.verify_state(near,expected)
  for change in ['matrix','mesh','scale','camera','weights','presence','lamp','exposure']:
   with self.subTest(change=change):
    bad=deepcopy(expected)
    if change=='matrix':bad['objects']['left_root']['world'][0][3]+=.001
    elif change=='mesh':bad['objects']['left_skin']['mesh']['geometry_uv_sha256']='changed'
    elif change=='scale':bad['objects']['left_core']['scale']=[2,1,1]
    elif change=='camera':bad['K'][0][0]*=2
    elif change=='weights':bad['skin_groups']['skin']='changed'
    elif change=='presence':bad['objects']['left_core']['hide_render']=True
    elif change=='lamp':bad['objects'][m.neutral.ENV_NAMES[2]]['light']['properties']['energy']+=.1
    else:bad['view']['exposure']+=.1
    with self.assertRaises(ValueError):m.verify_state(bad,expected)
if __name__=='__main__':unittest.main(verbosity=2)
