import copy,json,unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from data.action4_paths import PROJECT_ROOT as R
from data import action4_sam_native_export as ex
from data.action4_sam_cache_provider import SamCacheProvider
from data.action4_sam_roi_inputs import _sources,native_key,NativeMaskIdentity

class VariantExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec=json.loads((R/'outputs/action4_sam_variant_export_preparation_v001/actual_light_final/CACHE_SPEC.json').read_text())
        cls.index=json.loads((R/cls.spec['manifest']['path']).read_text())
        cls.receipt=ex._v2_checked(cls.index['producer_receipt'])
        cls.lock=ex._v2_checked(cls.receipt['native_lock'])
        cls.progress=ex._v2_checked(cls.receipt['native_completion'])
    def test_actual_native_masks_and_variant_RGB(self):
        provider=SamCacheProvider(self.spec,split='train')
        row=self.index['images'][0];rgb=np.asarray(Image.open(R/row['image_path']).convert('RGB'))
        frame,preds=provider.frame_predictions(row['image_id'],rgb)
        self.assertEqual(frame.variant_id,row['variant_id']);self.assertNotEqual(frame.variant_id,'native_original')
        self.assertEqual(len(preds),2);self.assertFalse(provider.production_associations)
        for mapping in self.receipt['mapping']:
            marker=ex.read(mapping['completed_marker']);result=ex.read(marker['result_path'])
            selected=next(v for v in result['native_candidates'] if v['obj_id']==result['bound_obj_id'])
            st=selected['mask_storage']
            with np.load(R/st['npz'],allow_pickle=False) as raw:mask=raw[st['array_key']][st['row_index']].squeeze()
            p=preds[native_key(NativeMaskIdentity(row['image_id'],mapping['prediction_id'],frame.pixel_space_id))]
            self.assertTrue(np.array_equal(p.mask,mask));self.assertEqual(p.metadata['sam_binding'],mapping['sam_binding'])
    def test_original_RGB_substitution_refused(self):
        provider=SamCacheProvider(self.spec,split='train');row=self.index['images'][0]
        rgb=np.asarray(Image.open(R/row['original_source']['path']).convert('RGB'))
        with self.assertRaisesRegex(ValueError,'Actual RGB'):provider.frame_predictions(row['image_id'],rgb)
    def test_unassociated_training_routes_refused(self):
        # Typed variant ownership is now supported; an unbound association is still invalid.
        with self.assertRaisesRegex(ValueError,'binding required'):
            SamCacheProvider(self.spec,split='train',training_associations={})
        for key in ('mask_augmentation','roi_routing'):
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'requires verified'):
                SamCacheProvider(self.spec,split='train',**{key:{}})
        provider=SamCacheProvider(self.spec,split='train')
        with self.assertRaisesRegex(ValueError,'association'):provider.enrich_training({}, {},identity_factory=None,mask_aux_factory=None)
    def test_no_native_original_alias(self):
        index=copy.deepcopy(self.index);index['sources']['upstream_mode']=ex.COMPLETED_MODE
        with self.assertRaises(ValueError):ex.verify_variant_export_receipt(index)
    def test_domain_completion_required(self):
        progress=dict(self.progress,completed_jobs=1)
        with self.assertRaisesRegex(ValueError,'Incomplete'):ex._variant_domain(self.lock,progress)
    def test_original_schema_retains_original_variant_identity(self):
        self.assertEqual(ex._variant_id({'schema':'action4.sam.prediction_prompt_image.v2','variant_id':'unexpected'}),'native_original')
    def test_no_partial_export(self):
        with self.assertRaisesRegex(ValueError,'entire domain'):ex.export_completed('unused','unused',1)
    def resealed_failure(self,edit):
        index=copy.deepcopy(self.index);receipt=copy.deepcopy(self.receipt);edit(index,receipt)
        receipt['index_payload_sha256']=ex.canonical({k:v for k,v in index.items() if k!='producer_receipt'})
        original=ex._v2_checked
        def checked(binding):return receipt if binding==index['producer_receipt'] else original(binding)
        with patch.object(ex,'_v2_checked',side_effect=checked),self.assertRaises(ValueError):ex.verify_variant_export_receipt(index)
    def test_resealed_original_source_substitution(self):
        def edit(index,receipt):index['images'][0]['source_sha256']=index['images'][0]['original_source']['sha256']
        self.resealed_failure(edit)
    def test_resealed_cache_cross_candidate_substitution(self):
        def edit(index,receipt):
            first,second=index['images'][0]['predictions'];first['path']=second['path'];first['sha256']=second['sha256']
            receipt['mapping'][0]['cache_sha256']=second['sha256']
        self.resealed_failure(edit)
    def test_resealed_training_admission_refused(self):
        self.resealed_failure(lambda index,receipt:receipt.update(formal_training_admitted=True))
    def test_missing_candidate_refused(self):
        self.resealed_failure(lambda index,receipt:index['images'][0]['predictions'].pop())
    def test_result_wrong_variant_and_session_refused(self):
        rows,_=ex._variant_domain(self.lock,self.progress);row=rows[0];c=row['candidates'][0]
        m=self.receipt['mapping'][0];marker=ex.read(m['completed_marker']);result=ex.read(marker['result_path'])
        for key,value in [('variant_id','native_original'),('close_success',False),('variant_pixels_sha256','0'*64)]:
            changed=dict(result);changed[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):ex._verify_variant_result(self.lock,row,c,marker,changed,(R/m['completed_marker']).parent)
    def test_variant_mode_explicit_source_keys(self):
        _sources(self.spec['expected_sources'])
        bad=dict(self.spec['expected_sources']);bad.pop('upstream_mode')
        with self.assertRaises(ValueError):_sources(bad)

if __name__=='__main__':unittest.main(verbosity=2)
