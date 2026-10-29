"""Data-only integration: no model construction, forward, backward or optimizer."""
import copy,importlib,json,unittest
from collections import Counter
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image
from data.action4_paths import PROJECT_ROOT as R
from data.action4_sam_native_export import sha,canonical,_v2_checked
from data.action4_sam_roi_inputs import array_digest
from data.action4_stage2_data import Stage2Units
from model.action4_sam_feature_adapter import FeatureIdentity,MaskAux

OUT=R/'outputs/action4_stage2_variant_selection_v001'

def actual_config():
    from data.action4_phrase2_dataset import surface_source
    spec=json.loads((R/'outputs/action4_sam_variant_export_preparation_v001/verified_light/CACHE_SPEC.json').read_text())
    ap=R/'outputs/action4_sam_variant_association_preparation_v001/ACTUAL_ASSOCIATIONS.json'
    return {'role':'A0','mode':'integration','dataset_root':'data/phrase2','surface_source':surface_source(),
        'seed':761,'roi_jitter':{'center_fraction':.15,'scale_range':[.65,1.35]},'train_indices':[0,1],
        'a0_variant':'a0_sam_image','sam_enabled':True,'photometric':False,
        'sam_rgb_variants':{'schema':'action4.sam_rgb_variant_selection.v1','leaves':[
            {'name':'light','probability':1.,'sam_cache':spec,'sam_training_associations':{'path':str(ap.relative_to(R)),'sha256':sha(ap)}}]}}

def module():return importlib.import_module('data.action4_sam_variant_selection')

class PureSelectionTests(unittest.TestCase):
    def test_stable_reordered_resume_selection(self):
        choose=module().choose_leaf;weights={'original':.5,'light':.3,'blur':.2}
        first={s:choose(weights,seed=17,epoch=4,image_id=s) for s in ['a','b','c']}
        other={s:choose(dict(reversed(list(weights.items()))),seed=17,epoch=4,image_id=s) for s in ['c','a','b']}
        self.assertEqual(first,other)
    def test_positive_normalized_finite_weights(self):
        for weights in ({},{'light':0.,'blur':1.},{'light':-.1,'blur':1.1},{'light':float('nan')},{'light':True},{'light':.8},{'unknown':1.}):
            with self.subTest(weights=weights),self.assertRaises(ValueError):module().choose_leaf(weights,seed=0,epoch=0,image_id='x')
    def test_frequencies_and_epoch_variation(self):
        weights={'original':.5,'light':.3,'blur':.2};choose=module().choose_leaf
        counts=Counter(choose(weights,seed=17,epoch=2,image_id=f'image_{i}')['selected_leaf'] for i in range(10000))
        for name,p in weights.items():self.assertLess(abs(counts[name]/10000-p),.025)
        self.assertGreater(sum(choose(weights,seed=17,epoch=2,image_id=str(i))['selected_leaf']!=choose(weights,seed=17,epoch=3,image_id=str(i))['selected_leaf'] for i in range(100)),20)
    def test_only_seed_epoch_original_image_define_draw(self):
        choose=module().choose_leaf
        for kwargs in ({'seed':-1,'epoch':0,'image_id':'a'},{'seed':0,'epoch':False,'image_id':'a'},{'seed':0,'epoch':0,'image_id':''}):
            with self.assertRaises(ValueError):choose({'light':1.},**kwargs)
        self.assertEqual(choose({'light':1.},seed=1,epoch=0,image_id='x')['selected_leaf'],'light')

class ActualStage2SelectionTests(unittest.TestCase):
    data=None
    @classmethod
    def inputs(cls):
        if cls.data is None:
            torch.set_num_threads(1)
            cls.data=Stage2Units(actual_config(),sam_factories=(FeatureIdentity,MaskAux))
        return cls.data
    def test_actual_two_GT_units_same_variant_exact_ROI(self):
        from data.action4_online_surface_targets import prepare_online_surface_batch
        data=self.inputs();self.assertEqual(len(data),2)
        self.assertIsNone(data.sam_provider)
        unit0=data[(0,0,True,2)];unit1=data[(1,0,True,2)]
        trace=unit0['metadata']['sam_rgb_variant_selection']
        self.assertEqual(trace,unit1['metadata']['sam_rgb_variant_selection'])
        self.assertEqual(trace['selected_leaf'],'light');self.assertFalse(trace['formal_training_admitted'])
        for i,result in enumerate((unit0,unit1)):
            original=data.data[data.indices[i]];identity=canonical(original['record'])
            item,provider,actual_trace=data.sam_variants.prepare_item(original,epoch=0)
            self.assertEqual(canonical(item['record']),identity)
            self.assertNotEqual(array_digest(original['image'].permute(1,2,0).numpy()),array_digest(item['image'].permute(1,2,0).numpy()))
            rgb,_,g,meta=prepare_online_surface_batch([item],'train',seeds=[result['augmentation_seed']],jitter_config=data.cfg['roi_jitter'],surface_source=data.cfg['surface_source'])
            self.assertTrue(torch.equal(result['roi'],rgb[0]));self.assertTrue(torch.equal(result['geometry']['K'],g['K'][0]))
            np.testing.assert_array_equal(result['metadata']['affine_hm'],meta[0]['affine_hm'])
            self.assertEqual(result['sam_input']['dependencies']['rgb_variant_selection'],trace)
            self.assertIn(trace['variant_id'],result['sam_input']['prediction_id'])
            self.assertEqual(result['sam_input']['training_gt_instance_id'],str(i+1))
            self.assertTrue(bool(result['mask_aux'].available.flatten()[0]));self.assertTrue(result['sam_quality_supervised'])
    def test_reordered_access_and_recreated_selector_identical(self):
        data=self.inputs();a=data[(1,3,True,2)];_=data[(0,3,True,2)];b=data[(1,3,True,2)]
        self.assertTrue(torch.equal(a['roi'],b['roi']));self.assertEqual(a['metadata']['sam_rgb_variant_selection'],b['metadata']['sam_rgb_variant_selection'])
        selector=module().SamVariantSelector(actual_config())
        item,_,trace=selector.prepare_item(data.data[1],epoch=3)
        self.assertEqual(trace,a['metadata']['sam_rgb_variant_selection'])
    def test_all_declared_leaves_checked_not_only_selected(self):
        data=self.inputs();selector=data.sam_variants;real=selector.providers['light']
        class MissingLeaf:
            def validate_image_coverage(self,records):raise ValueError('absent leaf image')
        # Isolated dispatch test only; no fabricated production cache is constructed.
        with patch.object(selector,'providers',{'light':real,'blur':MissingLeaf()}),self.assertRaisesRegex(ValueError,'absent leaf image'):
            selector.validate_image_coverage([data.data.records[0]])
        with self.assertRaises(ValueError):selector.validate_image_coverage(data.data.records[:2])
    def test_stale_original_pixels_cannot_replace_variant(self):
        data=self.inputs();original=data.data[0];pixels=original['image'].permute(1,2,0).numpy()
        with patch.object(module(),'_v2_rgb',return_value=pixels),self.assertRaisesRegex(ValueError,'Actual RGB'):
            data.sam_variants.prepare_item(original,epoch=0)
    def test_integration_train_A0_only_and_conflicts(self):
        for changes,split in [({'mode':'full_training'},'train'),({'role':'B0'},'train'),({},'eval'),({},'test'),({'sam_enabled':False},'train'),({'photometric':{'enabled':True}},'train'),({'sam_roi_routing':{}},'train'),({'sam_mask_augmentation':{}},'train'),({'sam_cache':{}},'train'),({'sam_training_associations':{}},'train')]:
            cfg=actual_config();cfg.update(changes)
            with self.subTest(changes=changes,split=split),self.assertRaises(ValueError):Stage2Units(cfg,split,sam_factories=(FeatureIdentity,MaskAux))
    def test_invalid_zero_leaf_probability_and_wrong_leaf_name(self):
        for value in (0.,-.1,.5,float('nan')):
            cfg=actual_config();cfg['sam_rgb_variants']['leaves'][0]['probability']=value
            with self.subTest(value=value),self.assertRaises(ValueError):Stage2Units(cfg,sam_factories=(FeatureIdentity,MaskAux))
        cfg=actual_config();cfg['sam_rgb_variants']['leaves'][0]['name']='blur'
        with self.assertRaises(ValueError):Stage2Units(cfg,sam_factories=(FeatureIdentity,MaskAux))
    def test_absent_selected_image_fails_Stage2_initialization(self):
        cfg=actual_config();cfg['train_indices']=[0,1,2]
        with self.assertRaises(ValueError):Stage2Units(cfg,sam_factories=(FeatureIdentity,MaskAux))
    def test_malformed_leaf_specs_fail_with_ValueError(self):
        for key,value in [('sam_cache',None),('sam_cache',{}),('sam_training_associations',{})]:
            cfg=actual_config();cfg['sam_rgb_variants']['leaves'][0][key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):Stage2Units(cfg,sam_factories=(FeatureIdentity,MaskAux))
    def test_without_option_keeps_original_baseline_data_path(self):
        cfg=actual_config();cfg.pop('sam_rgb_variants');cfg['sam_enabled']=False;cfg.pop('a0_variant')
        data=Stage2Units(cfg);unit=data[(0,0,True,2)]
        self.assertIsNone(data.sam_provider);self.assertNotIn('sam_rgb_variant_selection',unit['metadata'])
        self.assertNotIn('sam_enabled',unit)
if __name__=='__main__':unittest.main(verbosity=2)
