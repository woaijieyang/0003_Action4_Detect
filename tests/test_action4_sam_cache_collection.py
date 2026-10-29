import copy,unittest
from unittest.mock import patch
from types import SimpleNamespace
from data import action4_sam_cache_collection as m
class CollectionContract(unittest.TestCase):
 def fixture(self,sid):
  sources={'upstream_mode':m.VARIANT_MODE}
  spec={'manifest':{'path':sid,'sha256':'a'*64},'expected_sources':sources}
  shard={'sam_cache':spec,'sam_training_associations':{'path':sid+'-assoc','sha256':'b'*64}}
  index={'split':'train','origin':'predicted_sam','prompt_origin':'frozen_prediction_only','sources':sources,'rgb_variant':'light',
    'images':[{'image_id':sid,'image_path':sid+'-light.png','source_sha256':'c'*64,'original_source':{'path':sid+'.png','sha256':'d'*64}}]}
  return shard,index
 def test_duplicate_and_missing_domains(self):
  a,index=self.fixture('a')
  with patch.object(m,'_v2_checked',return_value=index):
   with self.assertRaisesRegex(ValueError,'Duplicate'):m.SamCacheCollection('light',[a,a])
   c=m.SamCacheCollection('light',[a])
  with self.assertRaises(ValueError):c.validate_image_coverage([{'sample_id':'missing'}])
  with self.assertRaises(ValueError):c.validate_image_coverage([{'sample_id':'a','image':'wrong','rgb_sha256':'d'*64}])
 def test_lazy_provenance_owner_and_bounded_providers(self):
  pairs=[self.fixture(str(i)) for i in range(3)];shards=[x[0] for x in pairs];indexes={x[0]['sam_cache']['manifest']['path']:x[1] for x in pairs}
  with patch.object(m,'_v2_checked',side_effect=lambda b:indexes[b['path']]),patch.object(m,'SamCacheProvider',side_effect=lambda *a,**k:SimpleNamespace(variant_associations=True,frames={1:'decoded'})) as provider,patch('data.action4_sam_native_export._v2_sha',side_effect=lambda path:'b'*64 if path.endswith('-assoc') else 'a'*64):
   c=m.SamCacheCollection('light',shards);self.assertEqual(provider.call_count,0)
   shards[0]['sam_cache']['manifest']['path']='tampered'
   for i in range(3):c.provider_for(str(i))
   self.assertEqual(provider.call_count,3);self.assertEqual(len(c.providers),3);self.assertEqual(len(c.hot_providers),2)
   self.assertEqual(c.providers[0].frames,{})
   c.provider_for('0');self.assertEqual(provider.call_count,3)
   self.assertEqual(provider.call_args_list[0].args[0]['manifest']['path'],'0')
 def test_selected_native_proof_failure_propagates(self):
  a,index=self.fixture('a')
  with patch.object(m,'_v2_checked',return_value=index):c=m.SamCacheCollection('light',[a])
  with patch.object(m,'SamCacheProvider',side_effect=ValueError('native proof failed')):
   with self.assertRaisesRegex(ValueError,'native proof failed'):c.provider_for('a')
class ActualCollection(unittest.TestCase):
 def test_actual_light_single_shard_matches_direct_leaf(self):
  import torch
  from tests.test_action4_stage2_variant_selection import actual_config
  from data.action4_stage2_data import Stage2Units
  from model.action4_sam_feature_adapter import FeatureIdentity,MaskAux
  torch.set_num_threads(1)
  cfg=actual_config();other=copy.deepcopy(cfg);leaf=other['sam_rgb_variants']['leaves'][0]
  other['sam_rgb_variants']['leaves'][0]={'name':leaf['name'],'probability':leaf['probability'],'shards':[{'sam_cache':leaf['sam_cache'],'sam_training_associations':leaf['sam_training_associations']}]}
  a=Stage2Units(cfg,sam_factories=(FeatureIdentity,MaskAux))[(0,0,True,1)]
  b=Stage2Units(other,sam_factories=(FeatureIdentity,MaskAux))[(0,0,True,1)]
  for key in ('roi','gt_visible448'):self.assertTrue(torch.equal(a[key],b[key]))
  self.assertTrue(torch.equal(a['mask_aux'].visible,b['mask_aux'].visible))
  self.assertEqual(a['identity'],b['identity'])
