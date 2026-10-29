import copy,json,unittest
import numpy as np
from PIL import Image
from data.action4_paths import PROJECT_ROOT as R
from data.action4_sam_roi_inputs import _sources,SOURCE_KEYS,native_key,NativeMaskIdentity
from data.action4_sam_cache_provider import SamCacheProvider
from data.action4_sam_native_export import verify_export_receipt,canonical

class NativeExportTests(unittest.TestCase):
    def setUp(self):
        self.spec=json.loads((R/'outputs/action4_sam_native_export_v001/first2/CACHE_SPEC.json').read_text())
        self.index=json.loads((R/self.spec['manifest']['path']).read_text())
    def test_actual_native_pixels_and_stable_ids(self):
        provider=SamCacheProvider(self.spec,split='eval')
        proof=verify_export_receipt(self.index)
        for row in self.index['images']:
            rgb=np.asarray(Image.open(R/row['image_path']).convert('RGB'));frame,preds=provider.frame_predictions(row['image_id'],rgb)
            for mapping in [x for x in proof['mapping'] if x['image_id']==row['image_id']]:
                self.assertEqual(mapping['prediction_id'],f"b0:{row['image_id']}:grid:{mapping['grid_index']}")
                marker=json.loads((R/mapping['completed_marker']).read_text());result=json.loads((R/marker['result_path']).read_text())
                c=next(x for x in result['native_candidates'] if x['obj_id']==result['bound_obj_id']);st=c['mask_storage']
                with np.load(R/st['npz']) as raw:mask=raw[st['array_key']][st['row_index']].squeeze().astype(bool)
                pred=preds[native_key(NativeMaskIdentity(row['image_id'],mapping['prediction_id'],frame.pixel_space_id))]
                self.assertTrue(np.array_equal(mask,pred.mask));self.assertEqual(pred.metadata['sam_binding']['generation'],result['generation'])
    def test_changed_index_cannot_relabel_candidate(self):
        index=copy.deepcopy(self.index);index['images'][0]['predictions'][0]['prediction_id']='wrong'
        with self.assertRaises(ValueError):verify_export_receipt(index)
    def test_B0_only_requires_explicit_mode(self):
        sources=dict(self.spec['expected_sources']);sources.pop('upstream_mode')
        with self.assertRaises(ValueError):_sources(sources)
        _sources({k:'a'*64 for k in SOURCE_KEYS})
    def test_development_export_cannot_be_train_cache(self):
        with self.assertRaises(ValueError):SamCacheProvider(self.spec,split='train')
    def test_reordered_index_is_not_silent_new_identity(self):
        index=copy.deepcopy(self.index);index['images'].reverse()
        with self.assertRaises(ValueError):verify_export_receipt(index)

if __name__=='__main__':unittest.main(verbosity=2)
