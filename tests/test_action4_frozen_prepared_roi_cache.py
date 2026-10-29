"""Small constructed CPU fixtures; no real native evidence or performance claims."""
import copy,pickle,unittest
from dataclasses import replace
import numpy as np
import torch,scipy
from data.action4_sam_roi_inputs import ImageFrame,RoiRequest,NativeSamPrediction,NativeMaskIdentity,SOURCE_KEYS,array_digest,prediction_metadata,verify_image_frame,verify_native_prediction,native_key,prepare_sam_roi_inputs
from data.action4_frozen_prepared_roi_cache import FrozenPreparedRoiCache,_charge
from model.action4_sam_feature_adapter import FeatureIdentity,MaskAux

class FrozenRoiCacheTests(unittest.TestCase):
 def setUp(self):
  torch.set_num_threads(1)
  self.sources={k:"a"*64 for k in SOURCE_KEYS};rgb=np.arange(24*32*3,dtype=np.uint8).reshape(24,32,3)
  self.frames=[verify_image_frame(ImageFrame("fixture",rgb,"native-v1","b"*64,"original",array_digest(rgb)))]
  self.requests=[RoiRequest(FeatureIdentity("fixture","one","roi-one","roi448"),0,(-8.,-12.,40.,36.),np.array([[50.,0,16],[0,50.,12],[0,0,1]]),"native-v1")]
  self.mask=np.zeros((24,32),bool);self.mask[4:18,6:23]=True;self.predictions={};self.set_predictions()
  self.sampling=dict(rgb_implementation_sha256="c"*64,mask_implementation_sha256="d"*64,aux_implementation_sha256="e"*64,backend="cpu",rgb_dtype="float32",grid_dtype="float32",affine_dtype="float64",torch_version=torch.__version__,numpy_version=np.__version__,scipy_version=scipy.__version__)
  self.calls=0;self.validations=0
 def set_predictions(self):
  self.predictions={}
  for r in self.requests:
   f=self.frames[r.image_index];ni=NativeMaskIdentity(f.image_id,r.identity.instance_id,f.pixel_space_id)
   # Explicit test-only metadata fixture; this does not claim SAM ran.
   meta=prediction_metadata(f,ni,self.mask,self.sources,assigned_instance_id=ni.instance_id,sam_binding=dict(session_id="unit-fixture",obj_id=ni.instance_id,generation=0,chosen_mask_id="fixture",match_basis="fixture"))
   self.predictions[native_key(ni)]=verify_native_prediction(NativeSamPrediction(meta,self.mask),f,ni,self.sources)
 def validate(self):self.validations+=1;return True
 def prepare(self):
  self.calls+=1
  return prepare_sam_roi_inputs(self.frames,self.requests,self.predictions,expected_sources=self.sources,mask_aux_factory=MaskAux)
 def args(self):return dict(frames=self.frames,requests=self.requests,predictions=self.predictions,expected_sources=self.sources,frozen_b0_predictions_sha256="f"*64,sampling=self.sampling,validate_current=self.validate,prepare=self.prepare)
 def test_exact_equivalence_and_cross_checkpoint_reuse(self):
  cache=FrozenPreparedRoiCache(64*1024**2);direct=self.prepare();a=cache.get_or_prepare(**self.args());b=cache.get_or_prepare(**self.args())
  for k in ["rgb","valid_image_support"]:self.assertTrue(torch.equal(getattr(direct,k),getattr(b,k)))
  for k in ["visible","distance","available","candidate_overlap","dropout"]:self.assertTrue(torch.equal(getattr(direct.mask_aux,k),getattr(b.mask_aux,k)))
  for k in ["A448","A112","A224"]:np.testing.assert_array_equal(getattr(direct,k),getattr(b,k))
  for k in ["K112","K224"]:np.testing.assert_array_equal(getattr(direct,k),getattr(b,k))
  self.assertEqual(direct.source_dependencies,b.source_dependencies);self.assertEqual(direct.reasons,b.reasons)
  self.assertEqual(cache.hits,1);self.assertEqual(self.validations,2);self.assertEqual(self.calls,2)
 def test_clone_isolation(self):
  cache=FrozenPreparedRoiCache();a=cache.get_or_prepare(**self.args());expected=a.rgb.clone();a.rgb.zero_();a.mask_aux.visible.zero_();a.A448[:]=0;a.K112[0][:]=0;a.source_dependencies[0]["source_sha256"]="poison"
  b=cache.get_or_prepare(**self.args());self.assertTrue(torch.equal(b.rgb,expected));self.assertGreater(b.mask_aux.visible.sum(),0);self.assertNotEqual(b.source_dependencies[0]["source_sha256"],"poison");self.assertGreater(abs(b.A448).sum(),0)
 def test_external_current_validation_required_even_on_hit(self):
  cache=FrozenPreparedRoiCache();cache.get_or_prepare(**self.args());args=self.args();args["validate_current"]=lambda:False
  with self.assertRaises(ValueError):cache.get_or_prepare(**args)
  self.assertEqual(cache.hits,0)
 def test_roi_K_and_implementation_change_miss(self):
  cache=FrozenPreparedRoiCache();cache.get_or_prepare(**self.args())
  self.requests=[replace(self.requests[0],bounds_native=(-9.,-12.,39.,36.))];cache.get_or_prepare(**self.args())
  k=self.requests[0].K_native.copy();k[0,2]+=.01;self.requests=[replace(self.requests[0],K_native=k)];cache.get_or_prepare(**self.args())
  self.sampling=dict(self.sampling,mask_implementation_sha256="1"*64);cache.get_or_prepare(**self.args());self.assertEqual(cache.hits,0);self.assertEqual(cache.misses,4)
 def test_image_mask_producer_and_source_changes_miss(self):
  cache=FrozenPreparedRoiCache();cache.get_or_prepare(**self.args());self.mask=self.mask.copy();self.mask[0,0]=True;self.set_predictions();cache.get_or_prepare(**self.args())
  f=self.frames[0];pix=f.pixels.copy();pix[0,0,0]^=1;self.frames=[verify_image_frame(ImageFrame(f.image_id,pix,f.pixel_space_id,"2"*64,f.variant_id,array_digest(pix)))];self.set_predictions();cache.get_or_prepare(**self.args())
  self.sources=dict(self.sources,sam_source_sha256="3"*64);self.set_predictions();cache.get_or_prepare(**self.args());self.assertEqual(cache.misses,4)
 def test_order_and_instances_part_of_key(self):
  self.requests.append(replace(self.requests[0],identity=FeatureIdentity("fixture","two","roi-two","roi448")));self.set_predictions();cache=FrozenPreparedRoiCache();cache.get_or_prepare(**self.args());self.requests.reverse();v=cache.get_or_prepare(**self.args());self.assertEqual(cache.hits,0);self.assertEqual(v.mask_aux.identities[0].instance_id,"two")
 def test_missing_prediction_cannot_hit_available(self):
  cache=FrozenPreparedRoiCache();cache.get_or_prepare(**self.args());self.predictions={};v=cache.get_or_prepare(**self.args());self.assertEqual(cache.hits,0);self.assertEqual(v.mask_aux.available.sum(),0)
 def test_bounded_lru_and_oversize(self):
  charge=_charge(self.prepare())+1024;cache=FrozenPreparedRoiCache(charge*2)
  for i in range(4):
   self.requests=[replace(self.requests[0],bounds_native=(-8.+i,-12.,40.+i,36.))];cache.get_or_prepare(**self.args());self.assertLessEqual(cache.resident_bytes,cache.max_bytes)
  self.assertEqual(len(cache),2);self.assertEqual(cache.evictions,2)
  tiny=FrozenPreparedRoiCache(1);tiny.get_or_prepare(**self.args());self.assertEqual(len(tiny),0);self.assertEqual(tiny.resident_bytes,0);self.assertEqual(tiny.oversize,1)
 def test_lru_hit_refreshes_recency(self):
  charge=_charge(self.prepare())+1024;cache=FrozenPreparedRoiCache(charge*2)
  first=self.requests[0];second=replace(first,bounds_native=(-7.,-12.,41.,36.));third=replace(first,bounds_native=(-6.,-12.,42.,36.))
  for request in [first,second,first,third]:
   self.requests=[request];cache.get_or_prepare(**self.args())
  self.assertEqual(cache.hits,1);self.requests=[first];cache.get_or_prepare(**self.args());self.assertEqual(cache.hits,2)
  self.requests=[second];cache.get_or_prepare(**self.args());self.assertEqual(cache.misses,4)
 def test_precision_and_video_domain_rejected(self):
  args=self.args();args["sampling"]=dict(self.sampling,grid_dtype="float64")
  with self.assertRaises(ValueError):FrozenPreparedRoiCache().get_or_prepare(**args)
  f=self.frames[0];self.frames=[verify_image_frame(ImageFrame(f.image_id,f.pixels,"raw_video_native_integer_centers",f.source_sha256,f.variant_id,f.variant_sha256))]
  with self.assertRaises(ValueError):FrozenPreparedRoiCache().get_or_prepare(**self.args())
 def test_training_temporal_and_unknown_precision_rejected(self):
  for key,value in [("mode","train"),("mode","video"),("roi_expansion",2.25)]:
   args=self.args();args[key]=value
   with self.assertRaises(ValueError):FrozenPreparedRoiCache().get_or_prepare(**args)
  self.sampling=dict(self.sampling,A0_checkpoint="forbidden")
  with self.assertRaises(ValueError):FrozenPreparedRoiCache().get_or_prepare(**self.args())
 def test_GT_and_dropout_payload_rejected(self):
  value=self.prepare();value.mask_aux.dropout[:]=True;args=self.args();args["prepare"]=lambda:value
  with self.assertRaises(ValueError):FrozenPreparedRoiCache().get_or_prepare(**args)
  value=self.prepare();value.source_dependencies[0]["gt_mask"]=[]
  with self.assertRaises(ValueError):FrozenPreparedRoiCache().get_or_prepare(**args)
 def test_unverified_and_process_serialization_rejected(self):
  f=self.frames[0];self.frames=[ImageFrame(f.image_id,f.pixels,f.pixel_space_id,f.source_sha256,f.variant_id,f.variant_sha256)]
  with self.assertRaises(ValueError):FrozenPreparedRoiCache().get_or_prepare(**self.args())
  with self.assertRaises(TypeError):pickle.dumps(FrozenPreparedRoiCache())
