"""Bounded per-process proof reuse: synthetic cache-format fixtures, no model calls."""
import copy,json,os,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from tests import test_action4_sam_completed_cache as fixtures
from data.action4_paths import PROJECT_ROOT as R
from data.action4_sam_cache_provider import SamCacheProvider
from data.action4_sam_roi_inputs import array_digest,cache_key
import data.action4_sam_native_export as exporter


class PayloadProofTests(unittest.TestCase):
    def setUp(self):
        fixtures.CompletedCacheTests.setUp(self)
    def do_export(self):
        return fixtures.CompletedCacheTests.do_export(self)
    def tearDown(self):
        exporter._PAYLOAD_PROOFS.clear();exporter._PAYLOAD_PROOF_PID=None

    def reseal(self,out,spec,index,receipt,name):
        receipt["index_payload_sha256"]=exporter.canonical({k:v for k,v in index.items() if k!="producer_receipt"})
        index["producer_receipt"]=self.jsonfile(self.base/(name+"_receipt.json"),receipt)
        return {"manifest":self.jsonfile(self.base/(name+"_index.json"),index),"expected_sources":spec["expected_sources"]}

    def test_repeat_constructor_skips_payload_decode_but_checks_bindings(self):
        _,spec,index=self.do_export()
        exporter._PAYLOAD_PROOFS.clear()
        original=exporter._v2_payload;check=exporter._check_payload;sha=exporter._v2_sha
        with patch.object(exporter,"_v2_payload",wraps=original) as decode,patch.object(exporter,"_check_payload",wraps=check) as pixels,patch.object(exporter,"_v2_sha",wraps=sha) as hashes:
            SamCacheProvider(spec,split="train")
            self.assertEqual(decode.call_count,3);self.assertEqual(pixels.call_count,3)
            SamCacheProvider(spec,split="train")
            self.assertEqual(decode.call_count,3);self.assertEqual(pixels.call_count,3)
            seen=[str(Path(c.args[0]).resolve()) for c in hashes.call_args_list]
            for p in index["images"][0]["predictions"]:
                self.assertGreaterEqual(seen.count(str((R/p["path"]).resolve())),2)
        self.assertTrue(all(isinstance(k,str) and len(k)==64 and v is None for k,v in exporter._PAYLOAD_PROOFS.items()))

    def test_resealed_wrong_pixels_and_index_variant_do_not_reuse_proof(self):
        out,spec,index=self.do_export()
        for kind in ("pixels","variant"):
            bad=copy.deepcopy(index);receipt=json.loads((out/"PRODUCER_RECEIPT.json").read_text())
            if kind=="variant":bad["images"][0]["variant_sha256"]="b"*64
            else:
                ref=bad["images"][0]["predictions"][0]
                with np.load(R/ref["path"],allow_pickle=False) as z:meta=json.loads(str(z["metadata"]));mask=z["mask"].copy()
                mask[0,0]=not mask[0,0];meta["mask_sha256"]=array_digest(mask);meta["cache_key"]=cache_key(meta)
                path=self.base/"forged_pixels.npz";np.savez(path,metadata=np.array(json.dumps(meta)),mask=mask)
                ref.update(path=str(path.relative_to(R)),sha256=exporter.sha(path))
                next(m for m in receipt["mapping"] if m["prediction_id"]==ref["prediction_id"])["cache_sha256"]=ref["sha256"]
            badspec=self.reseal(out,spec,bad,receipt,kind)
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                SamCacheProvider(badspec,split="train")

    def test_changed_native_result_expectation_invalidates_old_proof(self):
        out,spec,index=self.do_export();receipt=json.loads((out/"PRODUCER_RECEIPT.json").read_text())
        m=next(m for m in receipt["mapping"] if m["status"]=="available")
        marker=R/m["completed_marker"];done=json.loads(marker.read_text());rp=R/done["result_path"]
        result=json.loads(rp.read_text());result["native_candidates"][0]["target_mask_coverage"]=.5
        self.jsonfile(rp,result);done["files_sha256"][done["result_path"]]=exporter.sha(rp);self.jsonfile(marker,done)
        m["result_sha256"]=exporter.sha(rp);m["completed_marker_sha256"]=exporter.sha(marker)
        badspec=self.reseal(out,spec,index,receipt,"native_result")
        with self.assertRaisesRegex(ValueError,"overlap"):SamCacheProvider(badspec,split="train")

    def test_atomic_replacement_between_sha_and_memo_reuse_rejected(self):
        _,spec,index=self.do_export()
        target=R/index["images"][0]["predictions"][0]["path"]
        replacement=self.base/"same_bytes_replacement.npz";replacement.write_bytes(target.read_bytes())
        real=exporter._v2_sha;swapped=[]
        def swap(p):
            value=real(p)
            if Path(p).resolve()==target.resolve() and not swapped:
                replacement.replace(target);swapped.append(True)
            return value
        with patch.object(exporter,"_v2_sha",side_effect=swap):
            with self.assertRaisesRegex(ValueError,"replaced"):SamCacheProvider(spec,split="train")

    def test_fork_does_not_inherit_verified_payload_proofs(self):
        import multiprocessing
        _,spec,_=self.do_export()
        self.assertEqual(exporter._PAYLOAD_PROOF_PID,os.getpid())
        context=multiprocessing.get_context("fork");receive,send=context.Pipe(duplex=False)
        def child():
            before=exporter._PAYLOAD_PROOF_PID
            with patch.object(exporter,"_v2_payload",wraps=exporter._v2_payload) as decode:
                SamCacheProvider(spec,split="train")
                send.send({"before":before,"after":exporter._PAYLOAD_PROOF_PID,"pid":os.getpid(),"decoded":decode.call_count})
            send.close()
        process=context.Process(target=child);process.start();send.close()
        try:
            self.assertTrue(receive.poll(15),"owned fork verifier timed out")
            answer=receive.recv();process.join(5)
            self.assertEqual(process.exitcode,0);self.assertEqual(answer["before"],os.getpid())
            self.assertEqual(answer["after"],answer["pid"]);self.assertEqual(answer["decoded"],3)
        finally:
            receive.close()
            if process.is_alive():process.terminate();process.join()

    def test_proof_cache_is_bounded_and_arrays_are_not_retained(self):
        _,_,index=self.do_export()
        exporter._PAYLOAD_PROOFS.clear()
        with patch.object(exporter,"_PAYLOAD_PROOF_MAX_ENTRIES",2):
            exporter.verify_completed_export_receipt(index)
            self.assertEqual(len(exporter._PAYLOAD_PROOFS),2)
            self.assertTrue(all(isinstance(k,str) and len(k)==64 and v is None for k,v in exporter._PAYLOAD_PROOFS.items()))


if __name__=="__main__":unittest.main(verbosity=2)
