"""Parallel validation contract fixtures; no SAM or model execution."""
import copy,json,threading,unittest
from unittest.mock import patch
from tests import test_action4_sam_payload_proof as proof_tests
from data.action4_sam_cache_provider import SamCacheProvider
import data.action4_sam_native_export as e

class ParallelValidationTests(proof_tests.PayloadProofTests):
    def test_parallel_equals_serial_and_checks_all_payloads(self):
        _,spec,index=self.do_export()
        serial=e.verify_completed_export_receipt(index)
        e._PAYLOAD_PROOFS.clear()
        with patch.object(e,"_check_payload",wraps=e._check_payload) as check:
            self.assertEqual(e.verify_completed_export_receipt(index,workers=8),serial)
            self.assertEqual(check.call_count,3)
        provider=SamCacheProvider(spec,split="train",validation_workers=8)
        self.assertEqual(len(provider.images),1)

    def test_invalid_workers_and_variant_refused(self):
        _,spec,index=self.do_export()
        for workers in (0,9,True,1.5):
            with self.subTest(workers=workers),self.assertRaises(ValueError):
                SamCacheProvider(spec,split="train",validation_workers=workers)
        bad=copy.deepcopy(index);bad["sources"]["upstream_mode"]=e.VARIANT_MODE
        with self.assertRaisesRegex(ValueError,"completed-original"):
            e.verify_completed_export_receipt(bad,workers=2)

    def test_metadata_failure_precedes_parallel_payload(self):
        _,_,index=self.do_export();bad=copy.deepcopy(index);bad["images"]=[]
        with patch.object(e,"_verify_payload_with_proof") as verify:
            with self.assertRaises(ValueError):e.verify_completed_export_receipt(bad,workers=8)
            verify.assert_not_called()

    def test_parallel_prior_completed_payload_mutation_rejected(self):
        _,_,index=self.do_export()
        refs=index["images"][0]["predictions"];first=e.ROOT/refs[0]["path"]
        real=e._verify_payload_with_proof
        finished=threading.Event()
        def check(*args,**kwargs):
            real(*args,**kwargs)
            if args[0]["path"]==refs[0]["path"]:finished.set()
            elif args[0]["path"]==refs[1]["path"]:
                self.assertTrue(finished.wait(5))
                first.write_bytes(first.read_bytes()+b"changed")
        with patch.object(e,"_verify_payload_with_proof",side_effect=check):
            with self.assertRaisesRegex(ValueError,"changed|modified"):
                e.verify_completed_export_receipt(index,workers=3)

    def test_parallel_tamper_and_expectations(self):
        _,_,index=self.do_export()
        ref=index["images"][0]["predictions"][0];path=e.ROOT/ref["path"]
        path.write_bytes(path.read_bytes()+b"tampered")
        with self.assertRaises(ValueError):e.verify_completed_export_receipt(index,workers=8)

    def test_parallel_resealed_pixel_and_index_tamper(self):
        original=e.verify_completed_export_receipt
        def parallel(index,**kwargs):return original(index,workers=8)
        with patch.object(e,"verify_completed_export_receipt",side_effect=parallel):
            self.test_resealed_wrong_pixels_and_index_variant_do_not_reuse_proof()

    def test_parallel_atomic_replacement_on_memo_hit(self):
        original=e.verify_completed_export_receipt
        def parallel(index,**kwargs):return original(index,workers=8)
        with patch.object(e,"verify_completed_export_receipt",side_effect=parallel):
            self.test_atomic_replacement_between_sha_and_memo_reuse_rejected()

    def test_parallel_existing_proof_counterexamples(self):
        original=e.verify_completed_export_receipt
        def parallel(index,**kwargs):return original(index,workers=8)
        with patch.object(e,"verify_completed_export_receipt",side_effect=parallel):
            self.test_changed_native_result_expectation_invalidates_old_proof()

if __name__=="__main__":unittest.main()
