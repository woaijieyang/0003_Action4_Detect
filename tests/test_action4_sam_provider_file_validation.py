"""Incremental process-local file verification tests; tiny synthetic fixtures only."""
import hashlib,json,os,pickle,tempfile
from pathlib import Path
import unittest
from unittest import mock
import numpy as np
from PIL import Image
from data.action4_sam_cache_provider import SamCacheProvider
from data.action4_sam_roi_inputs import ImageFrame,NativeMaskIdentity,array_digest,prediction_metadata,SOURCE_KEYS

ROOT=Path("/data/zy/vln/a-test")
OUT=ROOT/"outputs/action4_sam_provider_preparation_v001"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

class FileVerificationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix="file_table_fixture_",dir=OUT)
        self.folder=Path(self.tmp.name)
        self.rgb=np.arange(16*16*3,dtype=np.uint8).reshape(16,16,3)
        image=self.folder/"input.png";Image.fromarray(self.rgb).save(image)
        sources={k:"a"*64 for k in SOURCE_KEYS}
        frame=ImageFrame("fixture",self.rgb,"native.fixture",sha(image),"original",array_digest(self.rgb))
        mask=np.ones((16,16),bool)
        metadata=prediction_metadata(frame,NativeMaskIdentity("fixture","pred","native.fixture"),mask,sources,
            assigned_instance_id="pred",origin="synthetic_fixture",
            sam_binding={"session_id":"fixture","obj_id":1,"generation":0,"chosen_mask_id":"m","match_basis":"fixture"})
        cache=self.folder/"mask.npz";np.savez_compressed(cache,metadata=np.array(json.dumps(metadata)),mask=mask)
        index={"schema":"action4.sam_native_v2_index.v1","split":"train","origin":"synthetic_fixture",
               "prompt_origin":"frozen_prediction_only","sources":sources,
               "images":[{"image_id":"fixture","image_path":str(image),"source_sha256":sha(image),"variant_id":"original",
                   "variant_sha256":array_digest(self.rgb),"pixel_space_id":"native.fixture",
                   "predictions":[{"prediction_id":"pred","path":str(cache),"sha256":sha(cache)}]}]}
        path=self.folder/"index.json";path.write_text(json.dumps(index))
        self.provider=SamCacheProvider({"manifest":{"path":str(path),"sha256":sha(path)},"expected_sources":sources},
                                       split="train",allow_synthetic=True)
        self.image=image;self.cache=cache

    def tearDown(self):self.tmp.cleanup()

    def test_first_hash_then_LRU_miss_reuses_validation_but_RGB_is_always_checked(self):
        original=hashlib.file_digest
        with mock.patch("data.action4_sam_cache_provider.hashlib.file_digest",wraps=original) as hashed:
            self.provider.frame_predictions("fixture",self.rgb)
            self.assertEqual(hashed.call_count,2)
            self.provider.frames.clear()  # Simulate random-training LRU eviction.
            self.provider.frame_predictions("fixture",self.rgb)
            self.assertEqual(hashed.call_count,2)
            self.provider.frame_predictions("fixture",self.rgb)
            self.assertEqual(hashed.call_count,2)
            bad=self.rgb.copy();bad[0,0,0]^=1
            with self.assertRaisesRegex(ValueError,"RGB/augmentation"):
                self.provider.frame_predictions("fixture",bad)
            self.assertEqual(hashed.call_count,2)
            with self.assertRaisesRegex(ValueError,"SHA mismatch"):
                self.provider._verified_file(self.cache,"b"*64)
            self.assertEqual(hashed.call_count,3)  # Changed expected SHA cannot reuse old proof.

    def test_changed_replaced_file_and_hash_race_reverify_or_reject(self):
        path=self.folder/"bound.bin";path.write_bytes(b"original")
        original=hashlib.file_digest
        with mock.patch("data.action4_sam_cache_provider.hashlib.file_digest",wraps=original) as hashed:
            expected=sha(path);self.provider._verified_file(path,expected)
            path.write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError,"SHA mismatch"):self.provider._verified_file(path,expected)
            self.assertEqual(hashed.call_count,2)
            expected=sha(path);self.provider._verified_file(path,expected)
            replacement=self.folder/"replacement.bin";replacement.write_bytes(b"modified");os.replace(replacement,path)
            self.provider._verified_file(path,expected)
            self.assertEqual(hashed.call_count,4)  # Same bytes/new inode still rechecked.
        racing=self.folder/"race.bin";racing.write_bytes(b"before")
        expected=sha(racing)
        def mutate(stream,algorithm):
            result=original(stream,algorithm);racing.write_bytes(b"after");return result
        with mock.patch("data.action4_sam_cache_provider.hashlib.file_digest",side_effect=mutate):
            with self.assertRaisesRegex(ValueError,"during hashing"):
                self.provider._verified_file(racing,expected)
        self.assertNotIn((str(racing.resolve()),expected),self.provider.file_validations)

    def test_pickle_and_worker_switch_clear_file_proofs(self):
        self.provider.frame_predictions("fixture",self.rgb)
        self.assertTrue(self.provider.file_validations)
        copied=pickle.loads(pickle.dumps(self.provider))
        self.assertFalse(copied.frames);self.assertFalse(copied.file_validations)
        original=hashlib.file_digest
        with mock.patch("data.action4_sam_cache_provider.hashlib.file_digest",wraps=original) as hashed:
            copied.frame_predictions("fixture",self.rgb)
            self.assertEqual(hashed.call_count,2)
            copied.pid=-1  # Exercise the same branch that detects a forked worker.
            copied.frame_predictions("fixture",self.rgb)
            self.assertEqual(hashed.call_count,4)

if __name__=="__main__":unittest.main(verbosity=2)
