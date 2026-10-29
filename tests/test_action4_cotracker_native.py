import unittest,json,hashlib
from pathlib import Path
from dataclasses import replace
import numpy as np
import torch
from PIL import Image
from model.action4_cotracker_native import TrackWindowRequest,WindowImageReader,prepare_video,track_window_raw
from model.action4_phase2_tracking_contracts import WindowIdentity,CropTransform,AnchorKey
R=Path('/data/zy/vln/a-test');OUT=R/'outputs/action4_cotracker_native_preparation_v001/fixtures'
class FakeCore:
 def __init__(self,invalid=False):self.calls=[];self.invalid=invalid
 def __call__(self,*,video,queries,iters,is_train):
  self.calls.append((video.clone(),queries.clone()));T=video.shape[1];N=queries.shape[1];xy=queries[...,1:].unsqueeze(1).repeat(1,T,1,1)+.25
  v=torch.full((1,T,N),.7);c=torch.full((1,T,N),.8)
  if self.invalid:xy[0,0,0,0]=float('nan');v[0,0,0]=float('nan')
  return xy,v,c,None
class CoTrackerAdapterTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  torch.set_num_threads(2);OUT.mkdir(exist_ok=True);cls.frames=[]
  y,x=np.mgrid[:6,:8];rgb=np.stack([x*20,y*30,np.full_like(x,100)],-1).astype('uint8')
  for i,fid in enumerate([10,20,30]):
   p=OUT/f'f{fid}.png';Image.fromarray(rgb+i).save(p);cls.frames.append({'frame_id':fid,'pts':100+i*400,'image':str(p),'image_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 def request(self,nframes=3,crop=None):
  return TrackWindowRequest(WindowIdentity('t',0,'w',10,31),tuple(self.frames[:nframes]),crop or CropTransform(0,0,8,6),
     (AnchorKey('t','a',0),AnchorKey('t','b',0)),(10,20 if nframes>1 else 10),np.array([[2.,2.],[4.,3.]]),'native','engineering_input_not_material_seed',time_base='1/11988')
 def test_native_frozen_parameters_buffers_and_child_mode_guard(self):
  from model.action4_cotracker_native import _frozen_core_state,_check_frozen_core
  def core():
   c=torch.nn.Sequential(torch.nn.Linear(2,2),torch.nn.Dropout(.5))
   c.register_buffer("state",torch.zeros(1))
   c.eval().requires_grad_(False);c._action4_native_identity={"source":"fixture"}
   c._action4_frozen_state=_frozen_core_state(c)
   return c
  c=core();_check_frozen_core(c,c._action4_native_identity)
  for change in (lambda c:c[1].train(),lambda c:c.state.add_(1),
                 lambda c:c[0].weight.add_(1),lambda c:c[0].weight.requires_grad_(True),
                 lambda c:setattr(c,"state",torch.zeros(1))):
   c=core();change(c)
   with self.assertRaises(ValueError):_check_frozen_core(c,c._action4_native_identity)
 def test_actual_crop_sampling_and_endpoint_mapping(self):
  q=self.request();video,ids,shape=prepare_video(q,WindowImageReader());self.assertEqual(video.shape,(1,3,3,384,512));self.assertEqual(shape,(6,8))
  np.testing.assert_allclose(video[0,0,:,0,0],[0,0,100]);np.testing.assert_allclose(video[0,0,:,-1,-1],[140,150,100])
  padded=replace(q,crop=CropTransform(-2,-1,10,8));v,_,_=prepare_video(padded,WindowImageReader());self.assertTrue(torch.equal(v[0,0,:,0,0],torch.zeros(3)))
 def test_bidirectional_time_support_and_raw_query_preserved(self):
  core=FakeCore();q=self.request();o=track_window_raw(core,q,model_identity='fixture',allow_test_backend=True)
  self.assertEqual(len(core.calls),2);self.assertEqual(o['support_query_count'],36)
  self.assertEqual(core.calls[0][1][0,:2,0].tolist(),[0.,1.]);self.assertEqual(core.calls[1][1][0,:2,0].tolist(),[2.,1.])
  self.assertTrue(torch.equal(core.calls[1][0],core.calls[0][0].flip(1)))
  self.assertFalse(np.array_equal(o['merged']['raw'][0][0,0],q.query_uv_native[0]))
  np.testing.assert_array_equal(o['merged']['query_corrected_xy'][0,0],q.query_uv_native[0]);self.assertTrue(o['merged']['is_query'][0,0]);self.assertFalse(o['geometry_observations_accepted'])
 def test_single_frame_and_padding_filter(self):
  q=self.request(1,crop=CropTransform(-4,-4,16,14));o=track_window_raw(FakeCore(),q,model_identity='fixture',allow_test_backend=True)
  self.assertLess(o['support_query_count'],36);self.assertFalse(o['merged']['nonquery_candidate_mask'].any())
 def test_invalid_core_outputs_remain_errors_not_absent(self):
  o=track_window_raw(FakeCore(True),self.request(),model_identity='fixture',allow_test_backend=True)
  self.assertEqual(o['status'],'RAW_INFERENCE_HAS_INVALID_VALUES');self.assertTrue(np.isnan(o['raw_forward_model'][0]).any());self.assertFalse(o['merged']['visibility_probability_valid'].all())
 def test_identity_query_and_changed_file_rejected(self):
  q=self.request()
  with self.assertRaises(ValueError):track_window_raw(FakeCore(),q,model_identity='pretend_native')
  with self.assertRaises(ValueError):track_window_raw(FakeCore(),replace(q,query_frame_ids=(10,999)),model_identity='fixture',allow_test_backend=True)
  with self.assertRaises(ValueError):track_window_raw(FakeCore(),replace(q,query_uv_native=np.array([[-1.,2.],[3.,3.]])),model_identity='fixture',allow_test_backend=True)
  reader=WindowImageReader();reader.read(q.frames[0]);bad=dict(q.frames[0],image_sha256='f'*64)
  with self.assertRaises(ValueError):reader.read(bad)
 def test_role_PTS_and_timebase_are_snapshot_identity(self):
  q=self.request();a=track_window_raw(FakeCore(),q,model_identity='fixture',allow_test_backend=True)
  b=track_window_raw(FakeCore(),replace(q,query_role='material_candidates'),model_identity='fixture',allow_test_backend=True)
  frames=tuple(dict(row,pts=row['pts']+1) for row in q.frames)
  c=track_window_raw(FakeCore(),replace(q,frames=frames),model_identity='fixture',allow_test_backend=True)
  d=track_window_raw(FakeCore(),replace(q,time_base='1/1000'),model_identity='fixture',allow_test_backend=True)
  self.assertEqual(len({x['snapshot_identity'] for x in (a,b,c,d)}),4)
 def test_invalid_PTS_and_missing_timebase_rejected_even_single_frame(self):
  q=self.request(1)
  for bad in [float('nan'),float('inf'),'0',True,0.5]:
   with self.assertRaises(ValueError):prepare_video(replace(q,frames=(dict(q.frames[0],pts=bad),)),WindowImageReader())
  with self.assertRaises(ValueError):prepare_video(replace(q,time_base=None),WindowImageReader())
  with self.assertRaises(ValueError):prepare_video(replace(q,time_base='0'),WindowImageReader())
  q=self.request()
  for pts in [100,99]:
   frames=(q.frames[0],dict(q.frames[1],pts=pts),q.frames[2])
   with self.assertRaises(ValueError):prepare_video(replace(q,frames=frames),WindowImageReader())

if __name__=='__main__':unittest.main(verbosity=2)
