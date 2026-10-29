import unittest,numpy as np,torch
from data.action4_sam_roi_inputs import ImageFrame,NativeMaskIdentity,NativeSamPrediction,prediction_metadata,array_digest,SOURCE_KEYS,cache_key
from eval.action4_surface_snapshot import make_snapshot
from model.action4_surface_anchors import seed_material_candidates,SeedPolicy
from model.action4_material_mesh import MaterialMeshBinder
from data import DataAdapter

class SurfaceAnchorTests(unittest.TestCase):
 def fixture(self):
  rgb=np.zeros((112,112,3),np.uint8);frame=ImageFrame('fixture',rgb,'native','a'*64,'v',array_digest(rgb));pid='b0:fixture:grid:1'
  yy,xx=np.mgrid[:112,:112];xyz=np.stack([xx/100,yy/100,np.zeros_like(xx)],0).astype('float32')
  region=np.full((1,65,112,112),-10.,np.float32);region[:,1]=10
  output={'surface':{'xyz':torch.tensor(xyz[None]),'masks':torch.full((1,2,112,112),10.),'regions':torch.tensor(region)},'rotation':torch.eye(3)[None],'translation':torch.tensor([[0.,0.,1.]])}
  cad=np.array([[x,y,z] for x in [0.,1.] for y in [0.,1.] for z in [0.,1.]])
  K=np.diag([100.,100.,1.]);down,_=DataAdapter.letterbox_affine((448,448),(112,112));models={'A0':{'path':'outputs/fixture_A0.pt','sha256':'c'*64},'B0':{'path':'outputs/fixture_B0.pt','sha256':'d'*64}}
  snapshot=make_snapshot(frame=frame,prediction={'grid_index':1,'prediction_id':pid,'bbox_full_xyxy':[0,0,111,111]},output=output,index=0,A112=np.eye(3),A448=np.linalg.inv(down),K_native=K,K112=K,canonical=cad,models=models)
  sources={k:'e'*64 for k in SOURCE_KEYS};sources['b0_model_sha256']='d'*64;mask=np.zeros((112,112),bool);mask[10:50,10:50]=True
  identity=NativeMaskIdentity('fixture',pid,'native');meta=prediction_metadata(frame,identity,mask,sources,assigned_instance_id=pid,origin='synthetic_fixture',sam_binding={'session_id':'s','obj_id':0,'generation':0,'chosen_mask_id':'m','match_basis':'fixture'})
  mesh=MaterialMeshBinder(np.array([[0,0,0],[0,1,0],[1,0,0]],float),np.array([[0,1,2]]),np.array([0]),np.array([[.2,.3,0]]),mesh_identity='fixture_mesh')
  kwargs=dict(expected_models=models,expected_sources=sources,mesh=mesh,track_id='t',track_generation=1,anchor_generation=0,frame_id=3,pts=1200,time_base='1/11988',track_confirmed=True,allow_synthetic=True)
  return snapshot,frame,NativeSamPrediction(meta,mask),kwargs
 def test_grid_budget_units_and_provisional_only(self):
  s,f,p,k=self.fixture();r=seed_material_candidates(s,f,p,**k)
  self.assertEqual(r['status'],'PROVISIONAL_SEEDS_READY');self.assertEqual(len(r['seeds']),96);self.assertFalse(r['geometry_accepted']);self.assertTrue(r['budget']['selection_truncated']);self.assertGreater(r['budget']['unvisited_candidates'],0)
  from collections import Counter
  self.assertLessEqual(max(Counter(x['spatial_grid_id'] for x in r['seeds']).values()),2)
  self.assertGreaterEqual(r['occupied_seed_grids'],4)
  for x in r['seeds']:
   self.assertEqual(x['lifecycle_state'],'provisional');self.assertFalse(x['geometry_accepted']);np.testing.assert_allclose(np.array(x['X_cad_m'])[:2]*100,x['query_uv_native'],atol=1e-5)
 def test_missing_and_wrong_assignment_never_use_A0_mask_as_SAM(self):
  s,f,p,k=self.fixture();r=seed_material_candidates(s,f,None,**k);self.assertEqual(r['seeds'],[]);self.assertEqual(r['reason'],'missing_native_SAM_support')
  meta=dict(p.metadata,assigned_instance_id='different');meta['cache_key']=cache_key(meta)
  r=seed_material_candidates(s,f,NativeSamPrediction(meta,p.mask),**k);self.assertEqual(r['seeds'],[]);self.assertEqual(r['reason'],'wrong_assigned_instance')
 def test_track_and_source_mismatch_reject(self):
  s,f,p,k=self.fixture()
  with self.assertRaises(ValueError):seed_material_candidates(s,f,p,**dict(k,track_confirmed=False))
  wrong=dict(k['expected_sources'],b0_model_sha256='f'*64)
  with self.assertRaises(ValueError):seed_material_candidates(s,f,p,**dict(k,expected_sources=wrong))
  with self.assertRaises(ValueError):seed_material_candidates(s,f,p,**dict(k,allow_synthetic=False))
 def test_no_background_fill_and_geometry_budget_visible(self):
  s,f,p,k=self.fixture();original=k['mesh'];k['mesh']=MaterialMeshBinder(original.vertices,original.faces,original.components,original.regions,mesh_identity='fixture_mesh',excluded_components=[0]);self.assertNotEqual(k['mesh'].binding_identity,original.binding_identity)
  r=seed_material_candidates(s,f,p,policy=SeedPolicy(max_geometry_trials=10),**k)
  self.assertEqual(r['seeds'],[]);self.assertEqual(r['counts']['geometry_trials'],10);self.assertTrue(r['budget']['selection_truncated']);self.assertEqual(r['counts']['display_or_glass_semantic_exclusion'],10)
 def test_mesh_state_immutable_and_discrete_budgets_strict(self):
  _,_,_,k=self.fixture();mesh=k['mesh']
  with self.assertRaises(AttributeError):mesh.excluded=frozenset([0])
  with self.assertRaises(AttributeError):mesh.distance=100
  with self.assertRaises(ValueError):mesh.normals[0]=0
  with self.assertRaises(ValueError):mesh.regions.setflags(write=True)
  copied=mesh.binding_provenance;copied['distance_fraction']=999;self.assertNotEqual(mesh.binding_provenance['distance_fraction'],999)
  for field in ['grid_side','max_per_cell','max_materials','max_geometry_trials']:
   for bad in [True,1.5]:
    with self.assertRaises(ValueError):SeedPolicy(**{field:bad})

 def test_nonengineering_mode_requires_verified_video_timing(self):
  s,f,p,k=self.fixture()
  with self.assertRaisesRegex(ValueError,'timing'):
   seed_material_candidates(s,f,p,**dict(k,allow_synthetic=False))
  r=seed_material_candidates(s,f,p,**k);self.assertFalse(r['time_identity_verified']);self.assertEqual(r['query_role'],'engineering_input_not_material_seed')

if __name__=='__main__':unittest.main(verbosity=2)
