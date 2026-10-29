"""Actual one-image variant input/ownership qualification; no model calls or training."""
import copy,json,unittest,uuid
import numpy as np
import torch
from PIL import Image
from data.action4_paths import PROJECT_ROOT as R
from data import action4_sam_training_associations as assoc
from data.action4_sam_native_export import sha,canonical,_v2_checked
from data.action4_sam_cache_provider import SamCacheProvider

OUT=R/'outputs/action4_sam_variant_association_preparation_v001'
class VariantAssociationTests(unittest.TestCase):
    context=None
    @classmethod
    def inputs(cls):
        if cls.context is not None:return cls.context
        torch.set_num_threads(1)
        base=OUT/('test_'+uuid.uuid4().hex);base.mkdir(parents=True)
        spec=json.loads((R/'outputs/action4_sam_variant_export_preparation_v001/verified_light/CACHE_SPEC.json').read_text())
        index=_v2_checked(spec['manifest']);old=json.loads((R/'outputs/action4_sam_selectedB0_train_cache_v001/TRAIN_ASSOCIATIONS.json').read_text())
        policy=dict(assoc.POLICY,scope='native_rgb_variant_original_box_supervision_ownership_only')
        pp=base/'POLICY.json';pp.write_text(json.dumps(policy));pb={'path':str(pp.relative_to(R)),'sha256':sha(pp)}
        binding=assoc.build_variant_associations(spec,old['annotations'],pb,base/'ASSOCIATIONS.json')
        payload=_v2_checked(binding);records=[json.loads(l) for l in (R/old['annotations']['path']).read_text().splitlines()]
        record=next(r for r in records if r['sample_id']==index['images'][0]['image_id'])
        rgb=torch.from_numpy(np.array(Image.open(R/index['images'][0]['image_path']).convert('RGB'))).permute(2,0,1)
        cls.context=(base,spec,index,binding,payload,record,rgb,records)
        return cls.context
    def provider(self):
        _,spec,_,binding,_,_,_,_=self.inputs()
        return SamCacheProvider(spec,split='train',training_associations=binding)
    def unit(self,instance=0):
        from data.action4_online_surface_targets import prepare_online_surface_batch
        from data.action4_phrase2_dataset import surface_source
        _,_,_,_,_,record,rgb,_=self.inputs()
        item={'image':rgb,'record':record,'instance':instance}
        roi,targets,g,meta=prepare_online_surface_batch([item],'train',seeds=[17+instance],surface_source=surface_source())
        unit={'roi':roi[0],'metadata':meta[0],'geometry':{k:v[0] for k,v in g.items()},'augmentation_seed':17+instance,'active':True}
        return item,unit
    def enrich(self,provider,item,unit):
        from model.action4_sam_feature_adapter import FeatureIdentity,MaskAux
        return provider.enrich_training(item,unit,identity_factory=FeatureIdentity,mask_aux_factory=MaskAux)
    def test_actual_two_unit_contract(self):
        _,spec,index,binding,payload,record,_,_=self.inputs();provider=self.provider()
        self.assertEqual(payload['schema'],'action4.sam_training_associations.rgb_variant.v1')
        self.assertFalse(payload['formal_training_admitted']);self.assertEqual(payload['coverage']['image_count'],1)
        self.assertEqual(payload['coverage']['nonignored_gt_units'],2)
        self.assertFalse(provider.production_associations);self.assertTrue(provider.variant_associations)
        provider.validate_image_coverage([record])
        for i in range(2):
            item,unit=self.unit(i);result=self.enrich(provider,item,unit)
            self.assertTrue(torch.equal(result['roi'],unit['roi']))
            self.assertTrue(torch.equal(result['geometry']['K'],unit['geometry']['K']))
            self.assertEqual(result['sam_input']['training_gt_instance_id'],str(record['instances'][i]['instance_id']))
            self.assertIn(index['images'][0]['variant_id'],result['sam_input']['prediction_id'])
            self.assertTrue(bool(result['mask_aux'].available.flatten()[0]));self.assertTrue(result['sam_quality_supervised'])
    def test_original_rgb_refused(self):
        _,_,_,_,_,record,_,_=self.inputs();item,unit=self.unit()
        item=dict(item,image=torch.from_numpy(np.array(Image.open(R/record['image']).convert('RGB'))).permute(2,0,1))
        with self.assertRaisesRegex(ValueError,'Actual RGB'):self.enrich(self.provider(),item,unit)
    def test_wrong_ROI_and_K_refused(self):
        item,unit=self.unit();provider=self.provider()
        bad=dict(unit,roi=unit['roi']+1)
        with self.assertRaisesRegex(ValueError,'RGB ROI'):self.enrich(provider,item,bad)
        bad=dict(unit,geometry=dict(unit['geometry'],K=unit['geometry']['K']+1))
        with self.assertRaisesRegex(ValueError,'training K'):self.enrich(provider,item,bad)
    def test_missing_image_coverage_refused(self):
        *_,records=self.inputs()
        with self.assertRaises(ValueError):self.provider().validate_image_coverage(records[:2])
    def test_wrong_original_source_refused(self):
        base,spec,_,_,payload,record,_,_=self.inputs();bad=copy.deepcopy(record);bad['rgb_sha256']='0'*64
        path=base/'WRONG_ANNOTATIONS.jsonl';path.write_text(json.dumps(bad)+'\n')
        with self.assertRaisesRegex(ValueError,'source image'):
            assoc.build_variant_associations(spec,{'path':str(path.relative_to(R)),'sha256':sha(path)},payload['policy'],base/'MUST_NOT_EXIST.json')
    def test_wrong_variant_cache_refused(self):
        _,spec,_,_,payload,_,_,_=self.inputs();bad=copy.deepcopy(spec);bad['expected_sources']['sam_prompt_sha256']='0'*64
        with self.assertRaises(ValueError):assoc.verify_variant_associations(payload,bad)
    def test_duplicate_assignment_and_missing_unit_refused(self):
        _,spec,_,_,payload,_,_,_=self.inputs()
        for mode in ('duplicate','missing'):
            bad=copy.deepcopy(payload)
            if mode=='duplicate':bad['rows'][1]['prediction_id']=bad['rows'][0]['prediction_id']
            else:bad['rows'].pop()
            bad['identity']=canonical({k:v for k,v in bad.items() if k!='identity'})
            with self.subTest(mode=mode),self.assertRaises(ValueError):assoc.verify_variant_associations(bad,spec)
    def test_annotation_record_mutation_refused(self):
        _,_,_,_,_,record,_,_=self.inputs();bad=copy.deepcopy(record);bad['camera']['K'][0][0]+=1
        with self.assertRaisesRegex(ValueError,'Training record'):self.provider().validate_image_coverage([bad])
    def test_wrong_affine_refused(self):
        item,unit=self.unit();bad=copy.deepcopy(unit)
        bad['metadata']['affine_hm'][0][2]+=1
        with self.assertRaisesRegex(ValueError,'affine'):self.enrich(self.provider(),item,bad)
    def test_missing_and_ambiguous_keep_fallback_ownership(self):
        _,_,index,_,payload,record,_,_=self.inputs();sid=record['sample_id']
        saved={r['sample_id']:r['instances'] for r in _v2_checked(payload['predictions'])}
        images={r['image_id']:r for r in index['images']};variants={sid:images[sid]['variant_id']}
        statuses={(sid,p['prediction_id']):'available' for p in images[sid]['predictions']}
        missing=copy.deepcopy(record)
        for obj in missing['instances']:obj['bbox_full_xyxy']=[-1000.,-1000.,-900.,-900.]
        rows,_=assoc.derive_rows([missing],saved,images,statuses,variant_ids=variants)
        self.assertTrue(all(r['status']=='missing' and r['prediction_id'] is None for r in rows))
        ambiguous=copy.deepcopy(record);ambiguous['instances'][1]['bbox_full_xyxy']=ambiguous['instances'][0]['bbox_full_xyxy']
        rows,_=assoc.derive_rows([ambiguous],saved,images,statuses,variant_ids=variants)
        self.assertTrue(all(r['status']=='ambiguous' and r['prediction_id'] is None for r in rows))
    def test_native_unavailable_does_not_change_owner(self):
        _,_,index,_,payload,record,_,_=self.inputs();sid=record['sample_id']
        saved={r['sample_id']:r['instances'] for r in _v2_checked(payload['predictions'])};images={r['image_id']:r for r in index['images']}
        statuses={(sid,p['prediction_id']):'ambiguous' for p in images[sid]['predictions']}
        rows,_=assoc.derive_rows([record],saved,images,statuses,variant_ids={sid:images[sid]['variant_id']})
        self.assertEqual([r['prediction_id'] for r in rows],[r['prediction_id'] for r in payload['rows']])
        self.assertTrue(all(r['native_status']=='ambiguous' for r in rows))
    def test_routing_and_augmentation_still_guarded(self):
        _,spec,_,binding,_,_,_,_=self.inputs()
        for key in ('roi_routing','mask_augmentation'):
            with self.subTest(key=key),self.assertRaises(ValueError):SamCacheProvider(spec,split='train',training_associations=binding,**{key:{}})
if __name__=='__main__':unittest.main(verbosity=2)
