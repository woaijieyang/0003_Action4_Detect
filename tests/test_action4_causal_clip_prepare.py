"""Synthetic CPU preparation fixtures, never a production PASS or native run."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np

from eval import action4_causal_clip_prepare as prep
from eval.action4_sam_b0_sequence import native_tracks
from eval.action4_clip_replay import record_process_exit,freeze_metadata_refs
from model.action4_sam_session_contracts import RegionSpec
import test_action4_clip_replay as clip_fixtures
ROOT=clip_fixtures.ROOT


class PrepareTests(unittest.TestCase):
    def setup_fixture(self):
        owner=clip_fixtures.ClipReplayTests(methodName="runTest");self.addCleanup(owner.doCleanups)
        folder,out,args,_=owner.fixture()
        self.owner=owner
        cfg=json.loads((ROOT/args["config_ref"]["path"]).read_text())
        inputs=owner.write(folder/"CLIPS.json",{"status":"CLIP_INPUTS_PREPARED_NOT_TRACKING_RUN",
            "protocol":args["registry_ref"],"clips":[{"clip_id":"clip","region":cfg["region"],
                "B0":owner.ref(out/"B0_COMPLETE.json")}]})
        b0source=owner.ref(folder/"source/SNAPSHOT_MANIFEST.json")
        live=folder/"live"
        names=["src/eval/action4_sam_b0_sequence.py","src/eval/action4_sam_sequence_backend.py",
            "src/eval/action4_clip_replay.py","src/data/action4_paths.py","runtime/surface_deps/fixture_pkg/__init__.py"]
        for name in names:
            path=live/name;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text("# synthetic tested CURRENT_CAUSAL_CODE fixture\n")
        launcher=live/prep.LAUNCHER;launcher.parent.mkdir(parents=True)
        launcher.write_bytes((ROOT/prep.LAUNCHER).read_bytes())
        tested=owner.write(folder/"TESTED.json",{"schema":"action4.causal_clip_tested_project.v1",
            "tests":{"passed":1,"failed":0,"scope":"synthetic CPU fixture only"},
            "files_sha256":{str(p.relative_to(live)):owner.ref(p)["sha256"] for p in live.rglob("*") if p.is_file()}})
        official=folder/"official";(official/"upstream/sam3/sam3").mkdir(parents=True)
        builder=official/"upstream/sam3/sam3/model_builder.py";builder.write_text("# synthetic official fixture\n");builder.chmod(0o444)
        weight=folder/"weight.bin";weight.write_bytes(b"synthetic-not-real-weights");weight.chmod(0o444)
        source_receipt=owner.write(official/"SOURCE.json",{"verified_commit":"fixture",
            "inspected_source_sha256":{"sam3/model_builder.py":owner.ref(builder)["sha256"]}})
        asset=owner.write(official/"ASSET.json",{"status":"ASSET_SHA_AND_WEIGHTS_ONLY_CPU_INSPECTION_PASSED",
            "sha256":owner.ref(weight)["sha256"],"bytes":weight.stat().st_size,"scope":"synthetic fixture"})
        native={"source_root":str((official/"upstream/sam3").relative_to(ROOT)),
            "source_commit":"fixture","source_receipt":source_receipt["path"],"asset_receipt":asset["path"],
            "weight":str(weight.relative_to(ROOT)),"text":"camera",
            "builder_kwargs":{"compile":False,"warm_up":False,"use_fa3":False,"use_rope_real":False,
                "async_loading_frames":False,"max_num_objects":16,"multiplex_count":16}}
        template={"schema":"action4.sam.b0_video_sequence.v1","source_snapshot":str(official.relative_to(ROOT)),
            "native":native,"binding":cfg["binding"]}
        owner.write(official/"config.json",template)
        for p in official.rglob("*"):
            if p.is_file():p.chmod(0o444)
        manifest={"schema":"action4.sam.immutable_source.v1","config_file":"config.json","official_commit":"fixture",
            "official_source_relative":"upstream/sam3","files_sha256":{str(p.relative_to(official)):owner.ref(p)["sha256"]
                for p in official.rglob("*") if p.is_file()}}
        manifest["identity"]=prep._digest(manifest)
        official_ref=owner.write(official/"SNAPSHOT_MANIFEST.json",manifest)
        allocation={"worker_id":"fixture","physical_gpu":7,"gpu_uuid":"GPU-fixture","cpu_affinity":[31],
                    "allocator_gib":32,"fresh_free_mib":40960}
        policy={"forbidden_gpu_ids":[0,1,2,3],"authorized_evaluation_gpu":[7],"cpu_policy":{"allowed_logical_cpu_ids":[31],"max_logical_cpus":72},"zeroshot_allowed_resources":{
            "sam_worker_allocations":[allocation],"sam_max_workers":1,"sam_per_gpu_worker_limits":{"7":1},
            "sam_allowed_allocator_gib":[8,32],"sam_per_gpu_allocator_budget_gib":32}}
        authority=owner.write(folder/"AUTHORITY.json",policy)
        kwargs=dict(clip_inputs_ref=inputs,clip_id="clip",b0_source_manifest_ref=b0source,
            official_snapshot_ref=official_ref,project_code_ref=tested,authority_ref=authority,
            resources=dict(allocation,torch_threads=1),seed_policy=cfg["seed_policy"],output=folder/"discovery")
        return folder,live,kwargs

    def prepare(self):
        folder,live,kwargs=self.setup_fixture()
        with patch.object(prep,"CURRENT_PROJECT_ROOT",live):
            receipt=prep.prepare_discovery(**kwargs)
        return folder,live,kwargs,receipt

    def run_synthetic_discovery(self,receipt):
        cfg=json.loads((ROOT/receipt["config_ref"]["path"]).read_text())
        source=prep.validate_source_snapshot(cfg);doc=prep.ReceiptVerifier().bound_json(cfg["region"])
        region=RegionSpec(doc["region_id"],tuple(f["global_frame_id"] for f in doc["frames"]),
            tuple(f["pts"] for f in doc["frames"]),tuple(doc["time_base"]),cfg["region"]["sha256"],
            doc["pixel_space_id"],doc["width"],doc["height"])
        out=ROOT/cfg["output"];owner=self.owner
        class Synthetic:
            def __init__(self,c):
                self.index=-1
                owner.write(out/"NATIVE_LOAD.json",{"status":"LOADED_ZERO_FINAL_MISSING_UNEXPECTED",
                                                  "scope":"synthetic native backend fixture"})
            def call(self,request):
                if request["type"]=="start_session":
                    self.index+=1;return {"session_id":"synthetic-prefix-"+str(self.index)}
                if request["type"]=="close_session":return {"is_success":True}
                x=4 if self.index==0 else 24
                m=np.zeros((1,40,40),bool);m[:,x+1:x+10,x+1:x+10]=True
                return {"frame_index":request["frame_index"],"outputs":{"out_obj_ids":np.array([0]),
                    "out_probs":np.array([.9],np.float32),"out_boxes_xywh":np.array([[x/40,x/40,11/40,11/40]],np.float32),
                    "out_binary_masks":m}}
            def propagate(self,request):raise AssertionError("Discovery cannot propagate")
            def assert_idle(self):pass
        with patch("eval.action4_sam_b0_sequence.context",return_value=(source,doc,region)), \
             patch("eval.action4_sam_b0_sequence.SequenceBackend",Synthetic):
            native_tracks(cfg)
        process=subprocess.Popen([sys.executable,"-B","-c","pass","synthetic_exit_proxy",
            "--config",receipt["config_ref"]["path"],"--mode","sam"],cwd=ROOT)
        process.wait(timeout=10)
        exit_ref=record_process_exit(process,receipt["config_ref"],out.parent/"EXIT.json")
        refs=freeze_metadata_refs(receipt["config_ref"],out.parent/"INDEX.json",producer_exit_ref=exit_ref)
        return dict(discovery_config_ref=receipt["config_ref"],producer_exit_ref=exit_ref,evidence_refs=refs,
            authority_ref=cfg["authority_ref"],resources=cfg["resources"],output=out.parent.parent/"forward")

    def test_discovery_copies_current_tested_source_no_hardlinks_and_verifies_launcher(self):
        folder,live,kwargs,receipt=self.prepare()
        cfg=json.loads((ROOT/receipt["config_ref"]["path"]).read_text());source=ROOT/cfg["source_snapshot"]
        self.assertEqual(cfg["b0_evidence"],json.loads((ROOT/kwargs["clip_inputs_ref"]["path"]).read_text())["clips"][0]["B0"])
        self.assertTrue(cfg["discovery_seed_only"]);self.assertTrue(cfg["discovery_decode_prefix"])
        self.assertEqual(cfg["sequence"]["direction"],"both")
        path="src/eval/action4_sam_b0_sequence.py"
        self.assertNotEqual((source/path).stat().st_ino,(live/path).stat().st_ino)
        self.assertEqual((source/path).read_bytes(),(live/path).read_bytes())
        self.assertFalse((source/path).stat().st_mode&0o222)
        spec=importlib.util.spec_from_file_location("synthetic_frozen_launcher",source/prep.LAUNCHER)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        verified,_=module.verify(receipt["config_ref"]["path"])
        self.assertEqual(verified,cfg)
        self.assertEqual(receipt["status"],"PREPARED_NOT_EXECUTED")
        with patch.object(prep,"CURRENT_PROJECT_ROOT",live):
            with self.assertRaises(FileExistsError):prep.prepare_discovery(**kwargs)

    def test_reject_stale_test_inventory_and_unregistered_32GiB_without_outputs(self):
        folder,live,kwargs=self.setup_fixture()
        (live/"src/eval/action4_sam_b0_sequence.py").write_text("# mutated after test receipt")
        with patch.object(prep,"CURRENT_PROJECT_ROOT",live):
            with self.assertRaisesRegex(ValueError,"SHA"):prep.prepare_discovery(**kwargs)
        self.assertFalse(Path(kwargs["output"]).exists())
        folder,live,kwargs=self.setup_fixture()
        path=ROOT/kwargs["authority_ref"]["path"];policy=json.loads(path.read_text())
        policy["zeroshot_allowed_resources"]["sam_allowed_allocator_gib"]=[8]
        kwargs["authority_ref"]=self.owner.write(path,policy)
        with patch.object(prep,"CURRENT_PROJECT_ROOT",live):
            with self.assertRaises(ValueError):prep.prepare_discovery(**kwargs)
        self.assertFalse(Path(kwargs["output"]).exists())

    def test_forward_inherits_discovery_not_mutated_worktree(self):
        folder,live,_,receipt=self.prepare()
        kwargs=self.run_synthetic_discovery(receipt)
        path="src/eval/action4_sam_b0_sequence.py"
        (live/path).write_text("# post-discovery mutation must not propagate")
        with patch.object(prep,"CURRENT_PROJECT_ROOT",folder/"does_not_exist"):
            result=prep.prepare_forward(**kwargs)
        cfg=json.loads((ROOT/result["config_ref"]["path"]).read_text())
        old=json.loads((ROOT/receipt["config_ref"]["path"]).read_text())
        self.assertEqual((ROOT/cfg["source_snapshot"]/path).read_bytes(),(ROOT/old["source_snapshot"]/path).read_bytes())
        self.assertEqual(cfg["sequence"]["direction"],"forward")
        self.assertEqual(cfg["temporal_mode"],"causal_forward_B0_assist")
        self.assertEqual(cfg["seed_mode"],"instance_points")
        self.assertEqual(cfg["b0_evidence"],old["b0_evidence"])
        self.assertEqual(cfg["region"],old["region"])
        self.assertTrue(all({"discovery_resource","discovery_complete","discovery_close"}<=set(r)
                            for r in cfg["instance_seed_sources"].values()))
        self.assertFalse(result["native_inference_started"])

    def test_forward_refuses_unconfirmed_cleanup(self):
        _,_,_,receipt=self.prepare();kwargs=self.run_synthetic_discovery(receipt)
        cfg=json.loads((ROOT/receipt["config_ref"]["path"]).read_text())
        (ROOT/cfg["output"]/"track_00/CLOSE.json").unlink()
        with self.assertRaisesRegex(ValueError,"cleanup_unconfirmed"):prep.prepare_forward(**kwargs)
        self.assertFalse(Path(kwargs["output"]).exists())



    def test_forward_refuses_zero_seed_and_wrong_actual_binding(self):
        for kind in ("zero","binding"):
            _,_,_,receipt=self.prepare();kwargs=self.run_synthetic_discovery(receipt)
            cfg=json.loads((ROOT/receipt["config_ref"]["path"]).read_text());out=ROOT/cfg["output"]
            if kind=="zero":
                path=out/"SEEDS.json";value=json.loads(path.read_text());value["seeds"]=[]
            else:
                path=out/"track_00/SEED_BINDING.json";value=json.loads(path.read_text())
                value["request"]["frame_index"]=1
            kwargs["evidence_refs"][str(path.relative_to(ROOT))]=self.owner.write(path,value)["sha256"]
            with self.assertRaises(ValueError):prep.prepare_forward(**kwargs)
            self.assertFalse(Path(kwargs["output"]).exists())

    def test_flat_resource_proposal_is_not_execution_authority(self):
        folder,live,kwargs=self.setup_fixture()
        kwargs["authority_ref"]=self.owner.write(folder/"PROPOSAL.json",{
            "schema":"action4.causal_clip_resources_proposal.v1","sam_allowed_allocator_gib":[8,32]})
        with patch.object(prep,"CURRENT_PROJECT_ROOT",live):
            with self.assertRaisesRegex(ValueError,"proposal"):prep.prepare_discovery(**kwargs)
        self.assertFalse(Path(kwargs["output"]).exists())


    def test_registration_cannot_override_GPU_revocation_or_CPU_budget(self):
        for kind in ("forbidden","not_allowed","cpu_budget"):
            folder,live,kwargs=self.setup_fixture()
            path=ROOT/kwargs["authority_ref"]["path"];policy=json.loads(path.read_text())
            if kind=="forbidden":policy["forbidden_gpu_ids"].append(7)
            elif kind=="not_allowed":policy["authorized_evaluation_gpu"]=[4,5,6]
            else:policy["cpu_policy"]["max_logical_cpus"]=0
            kwargs["authority_ref"]=self.owner.write(path,policy)
            with patch.object(prep,"CURRENT_PROJECT_ROOT",live):
                with self.assertRaises(ValueError):prep.prepare_discovery(**kwargs)
            self.assertFalse(Path(kwargs["output"]).exists())

if __name__=="__main__":
    unittest.main()
