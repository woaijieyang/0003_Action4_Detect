"""Small CPU engineering predictors only; never invoke a model or CUDA."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from data.action4_rgb_variants import DEFAULT_POLICY
from data.action4_rgb_variant_manifest import materialize
import eval.action4_rgb_variant_b0 as producer

R=producer.ROOT
OUT=R/"outputs/action4_rgb_variant_b0_preparation_v001"

class ProducerTests(unittest.TestCase):
    def setUp(self):
        OUT.mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=OUT);self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);rows=[]
        for i,width in enumerate((5,7,9)):
            p=self.root/("image"+str(i)+".png")
            Image.fromarray(np.full((4,width,3),80+i*20,np.uint8)).save(p)
            rows.append({"sample_id":"image"+str(i),"image":producer.binding(p)["path"],
                "rgb_sha256":producer.binding(p)["sha256"],"height":4,"width":width,
                "split":"train","status":"training_accepted","instances":[{"GT":"must never enter worker"}]})
        ann=self.root/"annotations.jsonl";ann.write_text("\n".join(json.dumps(r) for r in rows))
        policy=self.root/"policy.json";policy.write_text(json.dumps(DEFAULT_POLICY))
        self.manifest=materialize(producer.binding(ann),producer.binding(policy),
            [r["sample_id"] for r in rows],self.root/"rgb",seed=5)
    @staticmethod
    def predict(rgb):
        return [{"grid_index":i+10,"score":.2+i*.1,"bbox_full_xyxy":[1.,2.,3.,4.],
                 "center_px":[1.5,2.5],"center_score":.88,"old_quality_score":.91,
                 "old_product_score":.8008,"engineering_extra":{"preserve":[1,2,3]}}
                for i in range(rgb.shape[1]-5)]
    def run_fixture(self,name="leaf",variant="light",predictor=None,resume=False):
        return producer.engineering_fixture(self.manifest,variant,self.root/name,
            self.predict if predictor is None else predictor,resume=resume)
    def test_complete_zero_two_four_and_raw_fields(self):
        complete=json.loads(producer.read(self.run_fixture()))
        self.assertTrue(complete["engineering_fixture"])
        self.assertEqual(complete["status"],"ENGINEERING_FIXTURE_COMPLETE_NOT_PRODUCTION")
        self.assertEqual(complete["images"],3);self.assertEqual(complete["saved_candidates"],6)
        rows=json.loads(producer.read(complete["predictions"]))
        self.assertEqual([len(r["instances"]) for r in rows],[0,2,4])
        self.assertEqual([r["sample_id"] for r in rows],["image0","image1","image2"])
        for row in rows:
            raw=self.predict(producer.decode(row))
            self.assertEqual(row["instances"],raw)
            for i,c in enumerate(row["candidates"]):
                self.assertEqual(c["raw_saved_prediction"],raw[i])
                self.assertIn(row["variant_id"],c["prediction_id"])
                self.assertEqual(c["raw_saved_ordinal"],i)
        self.assertFalse(rows[1]["candidates"][0]["published"])
        self.assertFalse(complete["formal_training_admitted"])
    def test_same_grid_distinct_variants_and_swap_rejected(self):
        light=producer.leaf_inputs(self.manifest,"light")
        blur=producer.leaf_inputs(self.manifest,"blur")
        a=producer.pack_row(light["rows"][1],self.predict(producer.decode(light["rows"][1])))
        b=producer.pack_row(blur["rows"][1],self.predict(producer.decode(blur["rows"][1])))
        self.assertNotEqual(a["candidates"][0]["prediction_id"],b["candidates"][0]["prediction_id"])
        bad=copy.deepcopy(light);bad["rows"][1]=blur["rows"][1]
        with self.assertRaises(ValueError):producer.validate_inputs(bad)
        with self.assertRaises(ValueError):producer.leaf_inputs(self.manifest,"original")
    def test_interruption_resume_exact_prefix(self):
        calls=[]
        def interrupted(rgb):
            calls.append(1)
            if len(calls)==2:raise RuntimeError("fixture interruption")
            return self.predict(rgb)
        with self.assertRaisesRegex(RuntimeError,"interruption"):self.run_fixture(predictor=interrupted)
        run=self.root/"leaf"/"run"
        self.assertFalse((run/"COMPLETE.json").exists())
        self.assertEqual(len(list((run/"rows").glob("*.row.json"))),1)
        resumed=[]
        def continue_predict(rgb):resumed.append(1);return self.predict(rgb)
        self.run_fixture(predictor=continue_predict,resume=True)
        self.assertEqual(len(resumed),2)
        self.assertTrue(list(run.glob("FAILED_*.json")))
        with self.assertRaises(FileExistsError):self.run_fixture(resume=True)
    def test_input_and_resume_tamper_rejected(self):
        inputs=producer.leaf_inputs(self.manifest,"light")
        target=R/inputs["rows"][0]["actual_input"]["path"];target.write_bytes(b"changed")
        with self.assertRaises(ValueError):producer.validate_inputs(inputs)
    def test_resume_row_hash_and_missing_marker_rejected(self):
        def fail(rgb):
            if rgb.shape[1]==7:raise RuntimeError("stop")
            return self.predict(rgb)
        with self.assertRaises(RuntimeError):self.run_fixture(predictor=fail)
        row=self.root/"leaf"/"run"/"rows"/"000000.row.json"
        row.chmod(0o644);row.write_text("{}");row.chmod(0o444)
        with self.assertRaises(ValueError):self.run_fixture(resume=True)
        with self.assertRaises(RuntimeError):self.run_fixture(name="other",predictor=fail)
        done=self.root/"other"/"run"/"rows"/"000000.done.json";done.unlink()
        with self.assertRaises(ValueError):self.run_fixture(name="other",resume=True)
    def test_domain_missing_duplicate_and_resume_variant(self):
        inputs=producer.leaf_inputs(self.manifest,"light")
        for transform in (lambda v:v["rows"].pop(),lambda v:v["image_ids"].append("image0")):
            bad=copy.deepcopy(inputs);transform(bad)
            with self.assertRaises(ValueError):producer.validate_inputs(bad)
        def fail(rgb):raise RuntimeError("stop")
        with self.assertRaises(RuntimeError):self.run_fixture(predictor=fail)
        with self.assertRaises(ValueError):self.run_fixture(variant="blur",resume=True)
    def test_native_raw_validation(self):
        entry=producer.leaf_inputs(self.manifest,"light")["rows"][1]
        for raw in ([{"grid_index":True,"score":.5}],[{"grid_index":1,"score":float("nan")}],
                    [{"grid_index":1,"score":.5},{"grid_index":1,"score":.6}]):
            with self.assertRaises(ValueError):producer.pack_row(entry,raw)
    def test_freeze_metadata_only_and_production_guard(self):
        resource={"device":"cpu","cpu_affinity":[36],"threads":1,"project_total_reserved_cpu":1}
        with patch.object(producer,"sealed_source",return_value=({}, {}, {"evaluation_scoring_policy":"center"})):
            prepared=producer.prepare(self.manifest,"light",{"path":"explicit_fixture_no_model","sha256":"a"*64},
                self.root/"prepared",resource)
        receipt=json.loads(producer.read(prepared));cfg=json.loads(producer.read(receipt["config"]))
        self.assertFalse(receipt["inference_started"])
        self.assertEqual(producer.read(receipt["worker"]),Path(producer.__file__).read_bytes())
        for b in (receipt["worker"],receipt["config"],cfg["input"]):
            self.assertFalse(producer.path(b["path"]).stat().st_mode & 0o222)
        self.assertNotIn("must never enter worker",producer.read(cfg["input"]).decode())
        with self.assertRaisesRegex(ValueError,"frozen bound"):
            producer.production_worker(receipt["config"])
    def test_stdlib_preimport_source_bootstrap(self):
        code=self.root/"frozen";code.mkdir()
        source=code/"verifier.py";source.write_text("raise RuntimeError('must never execute')")
        config=code/"resolved_config.json"
        specpath=self.root/"RUN_SPEC.json"
        config.write_text(json.dumps({"run_spec":str(specpath.relative_to(R)),"evaluation_scoring_policy":"center"}))
        spec={"project_root":str(R),"code_root":str(code.relative_to(R)),
              "config":str(config.relative_to(R)),"mutable_worktree_not_used_by_children":True,
              "files":{"verifier.py":producer.binding(source)["sha256"],
                       "resolved_config.json":producer.binding(config)["sha256"]}}
        specpath.write_text(json.dumps(spec));bundle=self.root/"B0.fixture";bundle.write_bytes(b"no model")
        sealpath=self.root/"SEAL.json"
        seal={"role":"B0","strict_load":True,"epochs":80,"completed_optimizer_steps":13040,
              "completion":{"evaluations":653},"run_spec":str(specpath.relative_to(R)),
              "run_spec_sha256":producer.binding(specpath)["sha256"],
              "bundle":str(bundle.relative_to(R)),"sha256":producer.binding(bundle)["sha256"]}
        sealpath.write_text(json.dumps(seal))
        for f in (source,config,specpath,bundle,sealpath):f.chmod(0o444)
        self.assertEqual(producer.sealed_source(producer.binding(sealpath))[2]["evaluation_scoring_policy"],"center")
        source.chmod(0o644);source.write_text("# changed readonly verifier would self certify");source.chmod(0o444)
        with self.assertRaisesRegex(ValueError,"SHA/stat mismatch"):
            producer.sealed_source(producer.binding(sealpath))
        source.chmod(0o644)
        with self.assertRaisesRegex(ValueError,"writable"):
            producer.verify_source_before_import(producer.binding(specpath),spec)
        source.write_text("raise RuntimeError('must never execute')");source.chmod(0o444)
        bad=copy.deepcopy(spec);bad["files"]["../B0.fixture"]=producer.binding(bundle)["sha256"]
        with self.assertRaisesRegex(ValueError,"not relative"):
            producer.verify_source_before_import(producer.binding(specpath),bad)
        bad=copy.deepcopy(spec);bad["project_root"]="wrong"
        with self.assertRaisesRegex(ValueError,"project/root"):
            producer.verify_source_before_import(producer.binding(specpath),bad)
        config.chmod(0o644);config.write_text(json.dumps({"run_spec":"wrong"}));config.chmod(0o444)
        bad=copy.deepcopy(spec);bad["files"]["resolved_config.json"]=producer.binding(config)["sha256"]
        with self.assertRaisesRegex(ValueError,"cfg.run_spec"):
            producer.verify_source_before_import(producer.binding(specpath),bad)
    def test_closed_import_inventory(self):
        code=self.root/"inventory";code.mkdir()
        config=code/"resolved_config.json";specpath=self.root/"inventory_spec.json"
        config.write_text(json.dumps({"run_spec":str(specpath.relative_to(R))}));config.chmod(0o444)
        spec={"project_root":str(R),"code_root":str(code.relative_to(R)),
              "config":str(config.relative_to(R)),"mutable_worktree_not_used_by_children":True,
              "files":{"resolved_config.json":producer.binding(config)["sha256"]}}
        specpath.write_text(json.dumps(spec));specpath.chmod(0o444)
        sb=producer.binding(specpath)
        for relative in ("src/model/__init__.py", "src/model/extra.pyc", "src/model/native.cpython-311-x86_64-linux-gnu.so",
                         "runtime/surface_deps/native.cp311-win_amd64.pyd", "runtime/surface_deps/libnative.so.1"):
            candidate=code/relative;candidate.parent.mkdir(parents=True,exist_ok=True)
            candidate.write_bytes(b"owned fixture, never imported");candidate.chmod(0o444)
            with self.assertRaisesRegex(ValueError,"Unlisted sealed importable"):
                producer.verify_source_before_import(sb,spec)
            bound=copy.deepcopy(spec);bound["files"][relative]=producer.binding(candidate)["sha256"]
            producer.verify_source_before_import(sb,bound)
            candidate.unlink()
        target=code/"src"/"target.py";target.write_bytes(b"# bound target");target.chmod(0o444)
        spec["files"]["src/target.py"]=producer.binding(target)["sha256"]
        alias=code/"src"/"alias.py";alias.symlink_to(target)
        with self.assertRaisesRegex(ValueError,"symlink"):
            producer.verify_source_before_import(sb,spec)
        alias.unlink()
        directory_alias=code/"runtime"/"surface_deps"/"alias_dir";directory_alias.symlink_to(target.parent,target_is_directory=True)
        with self.assertRaisesRegex(ValueError,"symlink"):
            producer.verify_source_before_import(sb,spec)
        directory_alias.unlink()
        producer.verify_source_before_import(sb,spec)
    def test_foreign_run_prefix_rejected_before_factory(self):
        def foreign(rgb):
            if rgb.shape[1]==7:raise RuntimeError("stop")
            return [{"grid_index":99,"score":.91}]
        def current(rgb):
            if rgb.shape[1]==7:raise RuntimeError("stop")
            return []
        with self.assertRaises(RuntimeError):self.run_fixture(name="foreign",predictor=foreign)
        with self.assertRaises(RuntimeError):self.run_fixture(name="current",predictor=current)
        foreign_rows=self.root/"foreign"/"run"/"rows"
        current_rows=self.root/"current"/"run"/"rows"
        row=current_rows/"000000.row.json";done=current_rows/"000000.done.json"
        row.chmod(0o644);row.write_bytes((foreign_rows/row.name).read_bytes());row.chmod(0o444)
        marker_value=json.loads((foreign_rows/done.name).read_text());marker_value["row"]=producer.binding(row)
        done.chmod(0o644);done.write_text(json.dumps(marker_value));done.chmod(0o444)
        from unittest.mock import Mock
        predictor=Mock(side_effect=AssertionError("factory must not be consumed"))
        with self.assertRaisesRegex(ValueError,"Resume row marker identity"):
            self.run_fixture(name="current",predictor=predictor,resume=True)
        predictor.assert_not_called()
        self.assertFalse((self.root/"current"/"run"/"COMPLETE.json").exists())
    def test_resource_reject_before_GPU_query(self):
        spec={"device":"cpu","cpu_affinity":sorted(os.sched_getaffinity(0)),
              "threads":1,"project_total_reserved_cpu":72}
        with patch.object(producer.subprocess,"check_output",side_effect=AssertionError("no GPU query")):
            producer.resources(spec)
            with self.assertRaises(ValueError):producer.resources(dict(spec,project_total_reserved_cpu=73))
            with self.assertRaises(ValueError):producer.resources(dict(spec,device="cuda:0",physical_gpu=0,gpu_uuid="GPU-fixture",allocator_mib=4096))

if __name__=="__main__":unittest.main()
