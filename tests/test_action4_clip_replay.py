"""Synthetic CPU native_tracks artifacts; no native model or GPU execution."""
import copy
import hashlib
import json
import shutil
import tempfile
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from eval.action4_clip_replay import ClipReplay, _digest
from eval.action4_sam_b0_sequence import native_tracks
from model.action4_sam_session_contracts import RegionSpec

ROOT = Path("/data/zy/vln/a-test")
BASE = ROOT / "outputs/action4_clip_replay_preparation_v001"


class ClipReplayTests(unittest.TestCase):
    def write(self, path, value):
        path.write_text(json.dumps(value, allow_nan=False))
        return self.ref(path)

    def ref(self, path):
        return {"path": str(path.relative_to(ROOT)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def fixture(self, count=2, missing_mask=False, fail=False, copied=False, forward=False, forward_error=None, discovery=False, decode_prefix=True):
        BASE.mkdir(exist_ok=True)
        folder = Path(tempfile.mkdtemp(dir=BASE))
        def cleanup():
            assert folder.resolve().is_relative_to(BASE.resolve())
            for p in folder.rglob("config.json"):
                p.chmod(0o644)
            shutil.rmtree(folder)
        self.addCleanup(cleanup)
        staged = folder / "frames"; staged.mkdir()
        source = folder / "source"; source.mkdir()
        out = folder / "run"; out.mkdir()
        frames, original = [], []
        for i in range(4):
            p = staged / f"{i:06d}.png"
            Image.fromarray(np.full((40,40,3), i * 20, np.uint8)).save(p)
            ref = self.ref(p)
            frames.append({"local_index": i, "global_frame_id": 780+i, "pts": (780+i)*400,
                           "image": ref["path"], "sha256": ref["sha256"]})
            original.append({"frame_id": 780+i, "pts": (780+i)*400,
                             "image": ref["path"], "image_sha256": ref["sha256"]})
        original_ref = self.write(folder / "original.json", {"frames": original})
        doc = {"region_id":"clip", "frames":frames, "source_manifest":original_ref,
               "staged_frames":str(staged.relative_to(ROOT)), "pixel_space_id":"native_integer_centers", "time_base":[1,11988],
               "width":40, "height":40, "core":[780,784], "session_policy":"independent_clip_cold_start",
               "inherited_session":None, "core_only_scoring":True}
        region_ref = self.write(folder / "REGION.json", doc)
        region = RegionSpec("clip", tuple(range(780,784)), tuple((780+i)*400 for i in range(4)),
                            (1,11988), region_ref["sha256"], "native_integer_centers", 40,40)
        clip = {"clip_id":"clip", "core":[780,784], "context":[780,784],
                "input_frame_ids":list(range(780,784)), "evaluation_frame_ids":list(range(780,784)),
                "expected_core_frames":4, "session_policy":"independent_clip_cold_start",
                "reuse_whole_video_track_state":False,"core_only_scoring":True,
                "core_frame_bindings":[{"frame_id":f["global_frame_id"],"pts":f["pts"],"RGB_sha256":f["sha256"]}
                                       for f in frames]}
        registry = self.write(folder / "registry.json", {"clips":[clip], "source_manifest":original_ref})
        cfg = {"schema":"action4.sam.b0_video_sequence.v1", "output":str(out.relative_to(ROOT)),
               "source_snapshot":str(source.relative_to(ROOT)), "region":region_ref,
               "sequence":{"frame_count":4}, "b0":{"sha256":"b"*64},
               "seed_policy":{"confirmations":3,"score_min":.9,"quality_min":.8,"max_sessions":2},
               "resources":{"gpu_uuid":"GPU-fixture"}, "native":{"text":"camera","weight":{"sha256":"c"*64},"source_commit":"fixture"},
               "binding":{"mask_in_box_min":.8,"unique_margin":.15,"multi_target_support_min":.7}}
        if discovery:
            cfg["discovery_seed_only"]=True
            cfg["discovery_decode_prefix"]=decode_prefix
        if forward:
            cfg["sequence"]["direction"]="forward"
            cfg["temporal_mode"]="causal_forward_B0_assist"
        config = self.write(source / "config.json", cfg)
        (source / "config.json").chmod(0o444)
        manifest = {"schema":"action4.sam.immutable_source.v1","files_sha256":{"config.json":config["sha256"]},
                    "config_file":"config.json"}
        manifest["identity"] = _digest(manifest)
        self.write(source / "SNAPSHOT_MANIFEST.json", manifest)
        boxes = [[4.,4.,15.,15.],[24.,24.,35.,35.]]
        pools = []
        for i, f in enumerate(frames):
            candidates = [{"grid_index":k*10, "bbox_full_xyxy":boxes[k],"score":.99-k*.01,
                           "refined_quality_score":.9,"appearance_available":True,
                           "appearance_128":np.eye(128)[k].tolist()} for k in range(count)] if i<3 else []
            pools.append({"frame":f, "pool":{"schema":"action4.b0_association_pool.v1",
                                            "association_candidates":candidates}})
        poolpath=out/"B0_POOLS.jsonl"
        poolpath.write_text("".join(json.dumps(x)+"\n" for x in pools))
        self.write(out/"B0_COMPLETE.json", {"status":"B0_REGION_EVIDENCE_COMPLETE","b0":cfg["b0"],
            "region":region_ref,"frames":4,"source_identity":manifest["identity"],"pools":self.ref(poolpath)})
        class Fake:
            def __init__(self, config):
                self.index=-1
            def raw(self, i):
                mask=np.zeros((1,40,40),bool)
                k=0 if copied and i==3 else self.index
                x,y,r,b=map(int,boxes[k])
                if not (missing_mask and i==3):
                    mask[:,y+1:b-1,x+1:r-1]=True
                return {"frame_index":i,"outputs":{"out_obj_ids":np.array([0]),
                    "out_probs":np.array([.9],np.float32),"out_boxes_xywh":np.array([[x/40,y/40,(r-x)/40,(b-y)/40]],np.float32),
                    "out_binary_masks":mask}}
            def call(self, req):
                if req["type"]=="start_session":
                    self.index+=1
                    return {"session_id":"sid-"+str(self.index)}
                if req["type"]=="add_prompt":return self.raw(req["frame_index"])
                return {"is_success":True}
            def propagate(self, req):
                if forward:raise AssertionError("Causal producer called bidirectional API")
                yield self.raw(2)
                if fail:raise RuntimeError("synthetic system failure")
                for i in (3,1,0):yield self.raw(i)
            def propagate_forward(self, req):
                if not forward or req["propagation_direction"]!="forward" or req["start_frame_index"]!=2:
                    raise AssertionError("Actual forward request differs")
                if forward_error=="future":
                    yield self.raw(3);return
                if forward_error=="preseed":
                    yield self.raw(1);return
                yield self.raw(2)
                if forward_error=="missing_end":return
                if fail:raise RuntimeError("synthetic system failure")
                raw=self.raw(3)
                if forward_error=="missing_transport":raw["outputs"]=None
                yield raw
            def assert_idle(self):pass
        with patch("eval.action4_sam_b0_sequence.context",return_value=(manifest,doc,region)), \
             patch("eval.action4_sam_b0_sequence.SequenceBackend",Fake):
            if fail or count==0 or forward_error:
                with self.assertRaises((ValueError,RuntimeError)):native_tracks(cfg)
            else:native_tracks(cfg)
        refs={str(p.relative_to(ROOT)):self.ref(p)["sha256"] for p in folder.rglob("*.json")}
        tracker = self.write(folder / "TRACKER.json", {
            "geometry_mode":"sam_b0_fallback_v2", "window":5,
            "fallback_policy":{"minimum_score":.7,"minimum_quality":.8,"minimum_appearance_cosine":.8,
                "minimum_box_occupancy":.5,"maximum_motion":.35,"maximum_other_mask_coverage":.2,
                "fallback_cost_penalty":.15,"extent_recovery_frames":3,
                "extent_recovery_max_side_ratio":1.25,"extent_recovery_max_center_shift":.15}})
        # Explicit synthetic native backend fixture: this short CPU child is
        # ONLY its exit-receipt proxy, not a real native inference producer.
        from eval.action4_clip_replay import record_process_exit
        process=subprocess.Popen([sys.executable,"-B","-c",
            "import sys; sys.exit(" + ("1" if fail or count==0 or forward_error else "0") + ")",
            "synthetic_native_fixture_exit_proxy","--config",config["path"],"--mode","sam"],cwd=ROOT)
        process.wait(timeout=10)
        exit_ref=record_process_exit(process,config,folder/"PRODUCER_EXIT.json")
        args=dict(registry_ref=registry,clip_id="clip",config_ref=config,
                  tracker_config_ref=tracker,producer_exit_ref=exit_ref,evidence_refs=refs)
        return folder,out,args,pools

    def test_two_independent_sessions_same_native_zero_and_missing_B0(self):
        folder,out,args,pools=self.fixture()
        replay=ClipReplay(**args)
        rows=replay.core_rows()
        self.assertEqual(replay.b0_pool_identity,self.ref(out/"B0_POOLS.jsonl")["sha256"])
        self.assertEqual(replay.public_result()["b0_pool_ref"],self.ref(out/"B0_POOLS.jsonl"))
        self.assertEqual([r["frame"]["global_frame_id"] for r in rows],list(range(780,784)))
        self.assertEqual(rows[-1]["B0_raw"],pools[-1]["pool"])
        self.assertEqual({o["native_obj_id"] for o in replay.owners.values()},{0})
        self.assertEqual(len({o["session_id"] for o in replay.owners.values()}),2)
        rgb,ctx=replay.load(783,np.eye(3))
        dets=ctx.select_for_a0(ctx.consume(rgb,np.eye(3),b0_sha256="b"*64))
        self.assertEqual(len(dets),2)
        self.assertIsInstance(rgb,torch.Tensor)
        self.assertEqual(rgb.dtype,torch.uint8)
        from data.action4_dataset import prepare_rois
        rois,affines=prepare_rois([rgb],[d["bbox_for_a0"] for d in dets],[0,0],
                                 roi_size=32,expansion=1.5)
        self.assertEqual(tuple(rois.shape),(2,3,32,32))
        # Call the real pipeline API DIRECTLY, without DummyPipeline.run's
        # numpy-to-tensor conversion wrapper.
        from test_action4_sam_supported_roi import DummyPipeline
        probe=DummyPipeline()
        with patch("eval.action4_surface_snapshot.model_identities",
                   return_value={"B0":{"sha256":"b"*64}}), patch(
                   "model.action4_geometry_forward.instancewise_geometry",side_effect=probe.geometry):
            result=probe.pipeline.predict_video(rgb,np.eye(3),np.zeros((8,3)),
                                                video_context=ctx,roi_expansion=1.5)
        self.assertEqual(tuple(probe.rois[-1].shape),(2,3,448,448))
        trace=result["video_roi_consumption"]["instances"]
        self.assertEqual([r["bbox_used"] for r in trace],[d["bbox_for_a0"] for d in dets])
        for i,r in enumerate(trace):
            np.testing.assert_array_equal(probe.K[-1][i].numpy(),np.asarray(r["K112"],np.float32))
            np.testing.assert_allclose(np.asarray(r["A112"])@np.eye(3),r["K112"],atol=1e-5)
        for d in dets:
            self.assertIsNone(d["bbox_b0_raw"])
            self.assertNotEqual(d["bbox_for_a0"],d["bbox_sam_visible"])
            self.assertIsNone(d["last_trusted_full_frame_id"])
            self.assertFalse(d["full_box_accuracy_verified"])

    def test_seed_zero_one_two_keep_all_core_rows(self):
        for count in (0,1,2):
            _,_,args,_=self.fixture(count=count)
            replay=ClipReplay(**args)
            self.assertEqual(len(replay.core_rows()),4)
            self.assertEqual(len(replay.seeds),count)
            self.assertEqual(replay.status,"seed_failure" if count==0 else "native_complete_not_accuracy")

    def test_missing_mask_is_visibility_unknown_not_system_failure(self):
        _,_,args,_=self.fixture(missing_mask=True)
        replay=ClipReplay(**args)
        self.assertEqual(replay.status,"native_complete_not_accuracy")
        self.assertEqual(set(replay.native_status[(s["track_id"],3)] for s in replay.seeds),
                         {"target_visibility_unknown"})
        rgb,ctx=replay.load(783,np.eye(3))
        self.assertEqual(ctx.consume(rgb,np.eye(3),b0_sha256="b"*64),[])

    def test_system_failure_preserved_and_blocks_A0(self):
        _,_,args,_=self.fixture(fail=True)
        replay=ClipReplay(**args)
        self.assertEqual(replay.status,"system_failure")
        self.assertEqual(len(replay.core_rows()),4)
        self.assertTrue(replay.failure_evidence)
        with self.assertRaisesRegex(ValueError,"diagnostic-only"):replay.load(783,np.eye(3))

    def test_shared_current_mask_rejected(self):
        _,_,args,_=self.fixture(copied=True)
        replay=ClipReplay(**args)
        tracks=replay.core_rows()[-1]["fused"]["tracks"]
        self.assertTrue(all(t["support_rejection"]=="ambiguous_shared_instance_support" for t in tracks))
        rgb,ctx=replay.load(783,np.eye(3))
        self.assertEqual(ctx.consume(rgb,np.eye(3),b0_sha256="b"*64),[])

    def test_unsealed_close_and_tampered_raw_fail_closed(self):
        _,out,args,_=self.fixture()
        broken=copy.deepcopy(args)
        del broken["evidence_refs"][str((out/"track_00/CLOSE.json").relative_to(ROOT))]
        with self.assertRaisesRegex(ValueError,"Unsealed"):ClipReplay(**broken)
        p=out/"track_00/response_0000/raw.npz"
        p.write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError,"SHA"):ClipReplay(**args)

    def test_foreign_snapshot_generation_and_bad_close_rejected(self):
        for mutation in ("generation","close"):
            _,out,args,_=self.fixture()
            if mutation=="close":
                path=out/"track_00/CLOSE.json"
                value=json.loads(path.read_text());value["released"]=False
                args["evidence_refs"][str(path.relative_to(ROOT))]=self.write(path,value)["sha256"]
            else:
                path=out/"track_00/SNAPSHOT.json"
                value=json.loads(path.read_text());value["snapshot"]["generation"]=1
                ref=self.write(path,value)
                complete=out/"NATIVE_COMPLETE.json";done=json.loads(complete.read_text())
                done["tracks"][0]["snapshot_sha256"]=ref["sha256"]
                args["evidence_refs"][str(complete.relative_to(ROOT))]=self.write(complete,done)["sha256"]
            with self.assertRaises(ValueError):ClipReplay(**args)

    def test_registered_mapping_tamper_and_nonterminal_rejected(self):
        _,out,args,_=self.fixture()
        registry=ROOT/args["registry_ref"]["path"]
        doc=json.loads(registry.read_text());doc["clips"][0]["core_frame_bindings"][0]["pts"]+=1
        args["registry_ref"]=self.write(registry,doc)
        with self.assertRaisesRegex(ValueError,"bindings"):ClipReplay(**args)
        _,out,args,_=self.fixture()
        (out/"NATIVE_COMPLETE.json").unlink()
        with self.assertRaisesRegex(ValueError,"Nonterminal"):ClipReplay(**args)


    def test_duplicate_native_session_and_corrupt_local_raw_mapping_rejected(self):
        _,out,args,_=self.fixture()
        path=out/"track_01/SNAPSHOT.json"
        value=json.loads(path.read_text());value["snapshot"]["session_id"]="sid-0"
        ref=self.write(path,value)
        complete=out/"NATIVE_COMPLETE.json";done=json.loads(complete.read_text())
        done["tracks"][1]["snapshot_sha256"]=ref["sha256"]
        args["evidence_refs"][str(complete.relative_to(ROOT))]=self.write(complete,done)["sha256"]
        with self.assertRaisesRegex(ValueError,"Duplicate"):ClipReplay(**args)
        _,out,args,_=self.fixture()
        directory=out/"track_00/response_0000"
        header=directory/"raw.json"
        value=json.loads(header.read_text());value["frame_index"]=782
        ref=self.write(header,value)
        receipt=directory/"RAW_RECEIPT.json";value=json.loads(receipt.read_text())
        value["header_sha256"]=ref["sha256"]
        args["evidence_refs"][str(receipt.relative_to(ROOT))]=self.write(receipt,value)["sha256"]
        with self.assertRaises(ValueError):ClipReplay(**args)



    def test_explicit_tracker_policy_is_bound_and_mismatch_rejected(self):
        folder,_,args,_=self.fixture()
        replay=ClipReplay(**args)
        path=folder/"TRACKER.json";policy=json.loads(path.read_text())
        policy["fallback_policy"]["minimum_score"]=.123
        args["tracker_config_ref"]=self.write(path,policy)
        with self.assertRaisesRegex(ValueError,"fallback policy"):ClipReplay(**args)
        args["tracker_config_ref"]=self.write(path,{"geometry_mode":"legacy","window":5})
        other=ClipReplay(**args)
        self.assertNotEqual(replay.replay_identity,other.replay_identity)
        self.assertEqual(other.core_rows()[0]["tracker_config"]["geometry_mode"],"legacy")


    def test_terminal_metadata_index_is_reusable_and_authenticates_actual_format(self):
        from eval.action4_clip_replay import freeze_metadata_refs
        folder,out,args,_=self.fixture()
        destination=folder/"consumer"/"EVIDENCE.json"
        refs=freeze_metadata_refs(args["config_ref"],str(destination),producer_exit_ref=args["producer_exit_ref"])
        self.assertEqual(refs,freeze_metadata_refs(args["config_ref"],str(destination),producer_exit_ref=args["producer_exit_ref"]))
        replay=ClipReplay(**dict(args,evidence_refs=refs))
        rgb,ctx=replay.load(783,np.eye(3))
        self.assertEqual(json.loads(ctx.source_ledger_json)["native_owners"],replay.owners)
        with self.assertRaises(ValueError):
            freeze_metadata_refs(args["config_ref"],str(out/"BAD_INDEX.json"),producer_exit_ref=args["producer_exit_ref"])
    def test_index_refuses_nonterminal_native_run(self):
        from eval.action4_clip_replay import freeze_metadata_refs
        folder,out,args,_=self.fixture()
        (out/"NATIVE_COMPLETE.json").unlink()
        with self.assertRaises(ValueError):
            freeze_metadata_refs(args["config_ref"],str(folder/"INDEX.json"),producer_exit_ref=args["producer_exit_ref"])


    def test_exit_recorder_refuses_live_process_and_records_real_exit(self):
        from eval.action4_clip_replay import record_process_exit
        folder,_,args,_=self.fixture()
        command=[sys.executable,"-B","-c","import sys; sys.stdin.read(); sys.exit(7)",
                 "--config",args["config_ref"]["path"],"--mode","sam"]
        process=subprocess.Popen(command,stdin=subprocess.PIPE,cwd=ROOT)
        try:
            with self.assertRaisesRegex(ValueError,"still running"):
                record_process_exit(process,args["config_ref"],folder/"LIVE.json")
            self.assertFalse((folder/"LIVE.json").exists())
            process.communicate(input=b"end",timeout=10)
            ref=record_process_exit(process,args["config_ref"],folder/"REAL_EXIT.json")
            value=json.loads((ROOT/ref["path"]).read_text())
            self.assertEqual(value["pid"],process.pid)
            self.assertEqual(value["command"],command)
            self.assertEqual(value["returncode"],7)
            self.assertEqual(value["config_ref"],args["config_ref"])
        finally:
            if process.poll() is None:
                process.kill();process.wait()

    def test_exit_status_mismatch_and_failed_cleanup_refuse_freeze(self):
        from eval.action4_clip_replay import freeze_metadata_refs
        folder,out,args,_=self.fixture(fail=True)
        close=out/"track_00/CLOSE.json";close.unlink()
        with self.assertRaisesRegex(ValueError,"cleanup_unconfirmed"):
            freeze_metadata_refs(args["config_ref"],folder/"NO_INDEX.json",
                                 producer_exit_ref=args["producer_exit_ref"])
        self.assertFalse((folder/"NO_INDEX.json").exists())
        with self.assertRaisesRegex(ValueError,"cleanup_unconfirmed"):ClipReplay(**args)
        folder,_,args,_=self.fixture()
        path=ROOT/args["producer_exit_ref"]["path"]
        value=json.loads(path.read_text());value["returncode"]=1
        args["producer_exit_ref"]=self.write(path,value)
        with self.assertRaisesRegex(ValueError,"nonzero"):ClipReplay(**args)
        folder,_,args,_=self.fixture(fail=True)
        path=ROOT/args["producer_exit_ref"]["path"]
        value=json.loads(path.read_text());value["returncode"]=0
        args["producer_exit_ref"]=self.write(path,value)
        with self.assertRaisesRegex(ValueError,"nonzero"):ClipReplay(**args)

    def test_public_failure_native_status_and_noncausal_preseed_are_complete(self):
        _,_,args,_=self.fixture(fail=True)
        result=ClipReplay(**args).public_result()
        self.assertEqual(len(result["core_rows"]),4)
        failure=result["failure_evidence"][0]
        self.assertEqual(len(failure["sha256"]),64)
        self.assertIn("synthetic system failure",failure["error"])
        self.assertIn("Traceback",failure["traceback"])
        self.assertFalse(result["causal"])
        self.assertEqual(result["temporal_mode"],"offline_bidirectional_clip")
        self.assertTrue(result["not_approved_for_causal_B0_track"])
        _,_,args,_=self.fixture(missing_mask=True)
        result=ClipReplay(**args).public_result()
        rows=result["core_rows"]
        self.assertEqual(rows[0]["seed_frame_by_track"],{"track_00":782,"track_01":782})
        self.assertTrue(rows[0]["preseed_replayed"])
        self.assertFalse(rows[2]["preseed_replayed"])
        self.assertEqual(set(rows[-1]["native_status"].values()),{"target_visibility_unknown"})
        self.assertTrue(all(r["causal"] is False and r["propagation_direction"]=="both" for r in rows))

    def test_consumer_implementation_identity_changes_replay_identity(self):
        from eval.action4_clip_replay import implementation_identity
        _,_,args,_=self.fixture()
        before=ClipReplay(**args)
        altered=implementation_identity()
        self.assertIn("model.action4_sam_supported_boxes",altered["modules"])
        altered["modules"]["eval.action4_clip_replay"]["sha256"]="f"*64
        with patch("eval.action4_clip_replay.implementation_identity",return_value=altered):
            after=ClipReplay(**args)
        self.assertNotEqual(before.replay_identity,after.replay_identity)



    def test_exit_argv_config_mode_and_duplicate_options_rejected(self):
        from eval.action4_clip_replay import record_process_exit
        folder,_,args,_=self.fixture()
        cfg=args["config_ref"]["path"]
        other=folder/"OTHER_CONFIG.json";other.write_text("{}")
        cases=[
            ["--config",str(other),"--mode","sam"],
            ["--mode","sam"],
            ["--config",cfg,"--config",cfg,"--mode","sam"],
            ["--config",cfg],
            ["--config",cfg,"--mode","b0"],
            ["--config",cfg,"--mode","sam","--mode","sam"],
            ["--config",cfg,"--config="+cfg,"--mode","sam"],
            ["--","--config",cfg,"--mode","sam"],
        ]
        for index,options in enumerate(cases):
            process=subprocess.Popen([sys.executable,"-B","-c","pass",
                "synthetic_exit_argv_fixture",*options],cwd=ROOT)
            process.wait(timeout=10)
            target=folder/f"REJECTED_{index}.json"
            with self.assertRaises(ValueError):
                record_process_exit(process,args["config_ref"],target)
            self.assertFalse(target.exists())
        # Consumer validation independently rejects a tampered/resigned receipt.
        path=ROOT/args["producer_exit_ref"]["path"];value=json.loads(path.read_text())
        value["command"][value["command"].index("--config")+1]=str(other)
        args["producer_exit_ref"]=self.write(path,value)
        with self.assertRaisesRegex(ValueError,"argv config"):ClipReplay(**args)



    def test_causal_actual_producer_tail_mapping_and_preseed_withheld(self):
        _,out,args,pools=self.fixture(forward=True)
        replay=ClipReplay(**args);result=replay.public_result();rows=result["core_rows"]
        self.assertTrue(result["causal"])
        self.assertEqual(result["temporal_mode"],"causal_forward")
        self.assertEqual(result["propagation_direction"],"forward")
        self.assertEqual(len(rows),4)
        self.assertEqual(result["core_coverage"]["native_status_counts"],
                         {"initializing_no_native_response":4,"target_present":4})
        self.assertEqual(result["core_coverage"]["registered_track_core_slots"],8)
        self.assertTrue(all(not r["preseed_replayed"] for r in rows))
        self.assertTrue(all(t["status"]=="initializing" and t["bbox_for_roi"] is None
                            for r in rows[:2] for t in r["fused"]["tracks"]))
        for fid in (780,781):
            rgb,ctx=replay.load(fid,np.eye(3))
            self.assertEqual(ctx.consume(rgb,np.eye(3),b0_sha256="b"*64),[])
        self.assertEqual(rows[-1]["B0_raw"],pools[-1]["pool"])
        rgb,ctx=replay.load(783,np.eye(3))
        self.assertEqual(len(ctx.consume(rgb,np.eye(3),b0_sha256="b"*64)),2)
        for tid in ("track_00","track_01"):
            proof=json.loads((out/tid/"RESPONSIBILITY.json").read_text())
            self.assertEqual(proof["mapping"],[
                {"native_local_index":2,"responsibility_local_index":0,"global_frame_id":782,"pts":312800},
                {"native_local_index":3,"responsibility_local_index":1,"global_frame_id":783,"pts":313200}])
            self.assertEqual([x["global_frame_id"] for x in proof["seed_confirmation_frames"]],[780,781,782])
            seedmap=json.loads((out/tid/"SEED_REQUEST_MAPPING.json").read_text())
            self.assertEqual(seedmap["native_request"]["frame_index"],2)
            self.assertEqual(seedmap["collector_request"]["frame_index"],0)
            snap=json.loads((out/tid/"SNAPSHOT.json").read_text())["snapshot"]
            self.assertEqual([x["global_frame_id"] for x in snap["frames"]],[782,783])
            self.assertEqual([x["local_index"] for x in snap["frames"]],[0,1])
            self.assertEqual(len(list((out/tid).glob("response_*"))),2)

    def test_causal_future_preseed_missing_tail_and_transport_fail(self):
        for error in ("future","preseed","missing_end","missing_transport"):
            _,out,args,_=self.fixture(forward=True,forward_error=error)
            self.assertFalse((out/"NATIVE_COMPLETE.json").exists())
            replay=ClipReplay(**args)
            self.assertEqual(replay.status,"system_failure")
            self.assertEqual(len(replay.core_rows()),4)
            self.assertTrue(replay.failure_evidence)
        _,_,args,_=self.fixture(forward=True,missing_mask=True)
        replay=ClipReplay(**args)
        self.assertEqual(set(replay.core_rows()[-1]["native_status"].values()),{"target_visibility_unknown"})
        self.assertEqual(replay.status,"native_complete_not_accuracy")

    def test_causal_consumer_rejects_resigned_native_mapping_and_missing_tail(self):
        _,out,args,_=self.fixture(forward=True)
        directory=out/"track_00/response_0000";path=directory/"raw.json"
        value=json.loads(path.read_text());value["frame_index"]=3
        ref=self.write(path,value)
        path=directory/"RAW_RECEIPT.json";receipt=json.loads(path.read_text())
        receipt["header_sha256"]=ref["sha256"]
        args["evidence_refs"][str(path.relative_to(ROOT))]=self.write(path,receipt)["sha256"]
        with self.assertRaisesRegex(ValueError,"out of order"):ClipReplay(**args)
        _,out,args,_=self.fixture(forward=True)
        directory=out/"track_00/response_0001"
        assert directory.resolve().is_relative_to(BASE.resolve())
        shutil.rmtree(directory)
        with self.assertRaises(ValueError):ClipReplay(**args)



    def test_causal_zero_one_seed_preserves_full_core_denominator(self):
        for count in (0,1):
            _,_,args,_=self.fixture(count=count,forward=True)
            replay=ClipReplay(**args);result=replay.public_result()
            self.assertEqual(result["core_coverage"]["retained_core_frames"],4)
            self.assertEqual(result["core_coverage"]["seed_count"],count)
            self.assertEqual(result["core_coverage"]["registered_track_core_slots"],4*count)
            self.assertEqual(replay.status,"seed_failure" if count==0 else "native_complete_not_accuracy")



    def point_seed_source(self, *, discovery=True, decode_prefix=True):
        folder,out,args,_=self.fixture(discovery=discovery,decode_prefix=decode_prefix)
        prior=json.loads((ROOT/args["config_ref"]["path"]).read_text())
        seeds=json.loads((out/"SEEDS.json").read_text())["seeds"]
        load=self.write(out/"NATIVE_LOAD.json",{"status":"LOADED_ZERO_FINAL_MISSING_UNEXPECTED",
                                              "fixture":"synthetic native loader only"})
        refs={"config":args["config_ref"],"seeds":self.ref(out/"SEEDS.json"),"native_load":load,
              "binding":self.ref(out/"track_00/SEED_BINDING.json"),
              "raw_receipt":self.ref(out/"track_00/RAW_RECEIPT.json"),
              "discovery_complete":self.ref(out/"NATIVE_COMPLETE.json"),
              "discovery_close":self.ref(out/"track_00/CLOSE.json")}
        if (out/"DISCOVERY_RESOURCE.json").exists():
            refs["discovery_resource"]=self.ref(out/"DISCOVERY_RESOURCE.json")
        cfg=copy.deepcopy(prior);cfg["sequence"]["direction"]="forward"
        cfg["temporal_mode"]="causal_forward_B0_assist"
        cfg.pop("discovery_seed_only",None);cfg.pop("discovery_decode_prefix",None)
        cfg["instance_seed_sources"]={"track_00":refs}
        return folder,out,cfg,seeds,refs

    def test_causal_points_accept_only_authenticated_seed_prefix(self):
        from eval.action4_sam_b0_sequence import load_instance_seed_points
        _,_,cfg,seeds,refs=self.point_seed_source()
        points,labels,evidence=load_instance_seed_points(cfg,seeds[0],seeds)
        self.assertGreaterEqual(labels.count(1),2)
        self.assertGreaterEqual(labels.count(0),2)
        self.assertEqual(evidence["discovery_refs"],refs)
        self.assertEqual(len(points),len(labels))

    def test_causal_points_reject_hash_correct_bidirectional_and_nonprefix_sources(self):
        from eval.action4_sam_b0_sequence import load_instance_seed_points
        for discovery,decode in ((False,False),(True,False)):
            _,_,cfg,seeds,_=self.point_seed_source(discovery=discovery,decode_prefix=decode)
            with self.assertRaisesRegex(ValueError,"seed-only prefix"):
                load_instance_seed_points(cfg,seeds[0],seeds)
        _,out,cfg,seeds,refs=self.point_seed_source()
        # Same bytes, frame, weights and SHA; wrong request/response location.
        wrong=out/"track_00/response_0000";wrong.mkdir()
        receipt=json.loads((out/"track_00/RAW_RECEIPT.json").read_text())
        refs["raw_receipt"]=self.write(wrong/"RAW_RECEIPT.json",receipt)
        with self.assertRaisesRegex(ValueError,"fixed seed directory"):
            load_instance_seed_points(cfg,seeds[0],seeds)

    def test_causal_points_reject_future_prefix_missing_proof_and_wrong_seed_request(self):
        from eval.action4_sam_b0_sequence import load_instance_seed_points
        for kind in ("future","missing","request"):
            _,out,cfg,seeds,refs=self.point_seed_source()
            if kind=="future":
                path=out/"DISCOVERY_RESOURCE.json";value=json.loads(path.read_text())
                value["decoded_frames"]=4
                refs["discovery_resource"]=self.write(path,value)
            elif kind=="missing":
                del refs["discovery_complete"]
            else:
                path=out/"track_00/SEED_BINDING.json";value=json.loads(path.read_text())
                value["request"]["frame_index"]=1
                refs["binding"]=self.write(path,value)
            with self.assertRaises(ValueError):
                load_instance_seed_points(cfg,seeds[0],seeds)



    def test_legacy_both_points_keep_existing_nonprefix_source_contract(self):
        from eval.action4_sam_b0_sequence import load_instance_seed_points
        _,_,cfg,seeds,_=self.point_seed_source(discovery=False,decode_prefix=False)
        cfg["sequence"]["direction"]="both";cfg.pop("temporal_mode")
        points,labels,_=load_instance_seed_points(cfg,seeds[0],seeds)
        self.assertEqual(len(points),len(labels))
        self.assertGreaterEqual(labels.count(1),2)



    def test_project_body_creation_scoped_and_preseed_not_available(self):
        _,_,args,_=self.fixture(forward=True)
        replay=ClipReplay(**args)
        _,preseed=replay.load(780,np.eye(3))
        ledger=json.loads(preseed.source_ledger_json)
        self.assertEqual(ledger["project_body_registrations"],{})
        self.assertIsNone(ledger["project_track_generation"])
        rgb,ctx=replay.load(783,np.eye(3))
        ledger=json.loads(ctx.source_ledger_json);events=ledger["project_body_registrations"]
        self.assertEqual(set(events),{"track_00","track_01"})
        self.assertNotEqual(events["track_00"]["scoped_body_id"],events["track_01"]["scoped_body_id"])
        self.assertTrue(all(e["project_generation"]==0 and e["semantic_gt"] is False for e in events.values()))
        self.assertEqual({e["creation_global_frame"] for e in events.values()},{782})
        from data.action4_video_roi_context import validate_project_body_registration
        # Native session/generation may change without changing explicit project generation.
        owner=copy.deepcopy(ledger["native_owners"]["track_00"])
        owner.update(session_id="replacement-native-only",generation=8)
        validated=validate_project_body_registration(events["track_00"],ledger["project_body_scope"],
            track_id="track_00",frame_id=783,pts=313200,time_base=(1,11988),pixel_space_id="native_integer_centers",
            replay_identity=replay.replay_identity,native_owner=owner)
        self.assertEqual(validated["project_generation"],0)
        for event in (events["track_01"],dict(events["track_00"],creation_global_frame=784)):
            with self.assertRaises(ValueError):
                validate_project_body_registration(event,ledger["project_body_scope"],track_id="track_00",
                    frame_id=783,pts=313200,time_base=(1,11988),pixel_space_id="native_integer_centers",
                    replay_identity=replay.replay_identity,native_owner=owner)
        from dataclasses import replace
        changed=copy.deepcopy(ledger)
        changed["project_body_registrations"]["track_00"]["project_generation"]=9
        with self.assertRaises(ValueError):
            replace(ctx,source_ledger_json=json.dumps(changed)).consume(rgb,np.eye(3),b0_sha256="b"*64)

    def test_two_sam_only_real_surface_snapshots_load_and_face_identity(self):
        from test_action4_sam_supported_roi import DummyPipeline
        from data.action4_sam_roi_inputs import ImageFrame,array_digest
        from data.action4_surface_snapshot import save_snapshot,load_snapshot
        from data.action4_roi_contract import RoiSpec
        from model.action4_face_geometry import FaceGeometryContract
        from eval.action4_face_inputs import prepare_face_inputs
        folder,_,args,_=self.fixture(forward=True)
        replay=ClipReplay(**args);K=np.array([[80.,0,20.],[0,80.,20.],[0,0,1.]])
        rgb,ctx=replay.load(783,K)
        pixels=rgb.numpy().transpose(1,2,0).copy()
        frame=ImageFrame("clip:783",pixels,replay.region.pixel_identity,
            replay.doc["frames"][3]["sha256"],"native_original",array_digest(pixels))
        contract=FaceGeometryContract.from_bytes((ROOT/"configs/model/action4_six_face_anchor_contract_v001.json").read_bytes(),
            (ROOT/"data/DJI_Action4_construct/r3_v001/asset_manifest.json").read_bytes())
        models={"A0":{"path":"outputs/action4_clip_replay_preparation_v001/CPU_NOT_MODEL_A0","sha256":"a"*64},
                "B0":{"path":"outputs/action4_clip_replay_preparation_v001/CPU_NOT_MODEL_B0","sha256":"b"*64}}
        probe=DummyPipeline()
        def geometry(head,features,km,cad):
            n=len(features);camera=cad.clone();camera[:,:,2]+=1.
            return {"corners":torch.zeros(n,8,2),"camera_corners":camera,
                "rotation":torch.eye(3).repeat(n,1,1),"translation":torch.tensor([[0.,0.,1.]]).repeat(n,1),
                "surface":{"xyz":torch.zeros(n,3,112,112),"masks":torch.zeros(n,2,112,112),
                           "regions":torch.zeros(n,65,112,112)}}
        with patch("eval.action4_surface_snapshot.model_identities",return_value=models),              patch("model.action4_geometry_forward.instancewise_geometry",side_effect=geometry):
            result=probe.pipeline.predict_video(rgb,K,np.asarray(contract.canonical8_m),video_context=ctx,
                return_surface=True,surface_frame=frame,roi_expansion=1.5)
        self.assertEqual(len(result["surface_snapshots"]),2)
        scoped=[]
        for index,snapshot in enumerate(result["surface_snapshots"]):
            path=folder/f"surface_{index}.npz";save_snapshot(path,snapshot)
            loaded=load_snapshot(path,frame=frame,expected_models=models)
            pred=loaded.metadata["video_context"]["candidate"]
            face=prepare_face_inputs(loaded,frame=frame,expected_models=models,expected_prediction_id=pred["prediction_id"],
                roi_spec=RoiSpec.from_full_box(pred["bbox_for_a0"],1.5),contract=contract,min_depth_m=1e-6,grazing_cos_threshold=.05)
            body=face.metadata["body_identity"];scoped.append(body["scoped_body_id"])
            self.assertTrue(body["persistent_identity_ready"])
            self.assertEqual(body["generation"],0)
            self.assertEqual(body["sequence_identity"],replay.body_scope["sequence_identity"])
            self.assertEqual(pred["temporal_status"],"sam_supported")
            self.assertIsNone(pred["grid_index"])
            self.assertFalse(body["physical_left_right_verified"])
            from data.action4_surface_snapshot import SurfaceSnapshot,validate_snapshot,_canonical
            def altered(change):
                metadata=copy.deepcopy(loaded.metadata)
                change(metadata["video_context"]["source_ledger"])
                video=metadata["video_context"]
                video["identity"]=_canonical({k:v for k,v in video.items() if k!="identity"})
                metadata["receipt"]=_canonical({k:v for k,v in metadata.items() if k!="receipt"})
                return SurfaceSnapshot(metadata,loaded.arrays)
            def face_for(snapshot):
                return prepare_face_inputs(snapshot,frame=frame,expected_models=models,
                    expected_prediction_id=pred["prediction_id"],roi_spec=RoiSpec.from_full_box(pred["bbox_for_a0"],1.5),
                    contract=contract,min_depth_m=1e-6,grazing_cos_threshold=.05)
            def remove_registration(ledger):
                ledger.pop("project_body_scope");ledger.pop("project_body_registrations")
                ledger["project_track_generation"]=0  # Unregistered legacy scalar must not become ready.
            legacy=face_for(altered(remove_registration))
            self.assertIsNone(legacy.metadata["body_identity"]["generation"])
            self.assertFalse(legacy.metadata["persistent_identity_ready"])
            tid=pred["track_id"]
            for kind in ("wrong_track","future","wrong_scope","native_obj"):
                def tamper(ledger):
                    if kind=="native_obj":
                        ledger["native_owners"][tid]["native_obj_id"]=99
                        return
                    event=ledger["project_body_registrations"][tid]
                    if kind=="wrong_track":event["track_id"]="foreign-track"
                    elif kind=="future":event["creation_global_frame"]=784
                    else:event["sequence_identity"]="f"*64
                    event["event_identity"]=_canonical({k:v for k,v in event.items() if k!="event_identity"})
                with self.assertRaises(ValueError):face_for(altered(tamper))
        self.assertEqual(len(set(scoped)),2)


if __name__=="__main__":
    unittest.main()
