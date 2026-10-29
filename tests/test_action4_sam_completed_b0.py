"""Completed frozen B0 v2 native-cache CPU contracts. No native inference."""
import copy,json,unittest,uuid
from unittest.mock import patch
import numpy as np
from PIL import Image
from eval.action4_sam_image_eval import *
from test_action4_sam_image_eval import FakeBackend

class CompletedB0NativeTests(unittest.TestCase):
    def setUp(self):
        self.base=PROJECT_ROOT/"outputs/action4_sam_completed_b0_preparation_v001"/uuid.uuid4().hex
        self.base.mkdir(parents=True)
        self.config=copy.deepcopy(read_json("configs/eval/action4_sam_full_dev_r096_v005.json"))
        self.config.update(schema=SCHEMA_V2,upstream_role=COMPLETED_ROLE,split="train",rgb_variant="native_original",
            expected_images=2,expected_saved_candidates=3,expected_prompts=2,output=str((self.base/"run").relative_to(PROJECT_ROOT)))
        self.config["resources"]["cpu_affinity"]=[88,89]
        self.snapshot_patch=patch("eval.action4_sam_image_eval.validate_source_snapshot",return_value={"identity":"c"*64})
        self.authority_patch=patch("eval.action4_sam_image_eval.validate_live_authority",return_value={"cpu_fixture":True})
        self.snapshot_patch.start();self.authority_patch.start()
        self.addCleanup(self.snapshot_patch.stop);self.addCleanup(self.authority_patch.stop)
        def bound(name,value,readonly=False,raw=False):
            p=self.base/name
            p.write_bytes(value if raw else json.dumps(value).encode())
            if readonly:p.chmod(0o444)
            return {"path":str(p.relative_to(PROJECT_ROOT)),"sha256":file_sha(p)}
        self.bound=bound
        bundle=bound("B0.pt",b"engineering_fixture_not_weights",True,True)
        run=bound("RUN_SPEC.json",{"schema":"action4.immutable_run.v1","code_root":"fixture/source",
                                  "mutable_worktree_not_used_by_children":True},True)
        seal=bound("SEALED_ROLE.json",{"schema":"action4.sealed_independent_role.v2","role":"B0",
            "strict_load":True,"bundle":bundle["path"],"sha256":bundle["sha256"],"selected_step":9500,"config_id":"fixture",
            "run_spec":run["path"],"run_spec_sha256":run["sha256"],"actual_code_root":"fixture/source",
            "completion":{"status":"COMPLETION_METADATA_READY","role":"B0","best_step":9500,"evaluations":653,"config_id":"fixture"}},True)
        raw0={"score":.8,"grid_index":7,"bbox_full_xyxy":[1,1,4,5]}
        raw1={"score":.2,"grid_index":9,"bbox_full_xyxy":[6,1,9,5]}
        raw2={"score":.8,"grid_index":8,"bbox_full_xyxy":[1,1,4,5]}
        predictions=bound("PREDICTIONS.json",[{"sample_id":"fixture0","instances":[raw0,raw1]},
                                           {"sample_id":"fixture1","instances":[raw2]}])
        source={"sealed_role":seal,"run_spec":run,"b0_bundle":bundle,"selected_step":9500,
                "config_id":"fixture","predictions":predictions}
        self.config["source_contract"]=source;self.rows=[]
        for i,raws in enumerate(([raw0,raw1],[raw2])):
            image=self.base/f"image{i}.png";Image.fromarray(np.zeros((8,10,3),np.uint8)).save(image)
            cs=[]
            for j,raw in enumerate(raws):
                b=raw["bbox_full_xyxy"];eligible=raw["score"]>=.3
                cs.append({"candidate_id":f'b0:fixture{i}:grid:{raw["grid_index"]}',
                    "identity_binding":{"sample_id":f"fixture{i}","grid_index":raw["grid_index"],
                        "raw_saved_ordinal":j,"prediction_file_sha256":predictions["sha256"]},
                    "source":"frozen_completed_B0_saved_prediction","raw_saved_prediction":raw,
                    "score":raw["score"],"published":eligible,"eligible_for_SAM_diagnostic_prompt":eligible,
                    "bbox_full_native_xyxy":b,"bbox_sam_native_xyxy":b,
                    "bbox_sam_normalized_xywh":[b[0]/10,b[1]/8,(b[2]-b[0])/10,(b[3]-b[1])/8]})
            self.rows.append({"schema":"action4.sam.prediction_prompt_image.v2","split":"train","rgb_variant":"native_original",
                "image_id":f"fixture{i}","image_path":str(image.relative_to(PROJECT_ROOT)),"image_sha256":file_sha(image),
                "native_shape_hw":[8,10],"pixel_space_id":"fixture.native","pixel_convention":"integer_centers",
                "source_receipt":copy.deepcopy(source),"raw_saved_candidate_count":len(raws),
                "SAM_prompt_count":sum(c["published"] for c in cs),"candidates":cs,"GT_used_for_prompt_selection":False})
        self.write_manifest()
    def write_manifest(self):
        p=self.base/"prompts.jsonl";p.write_text("".join(json.dumps(r)+"\n" for r in self.rows))
        self.config.update(manifest=str(p.relative_to(PROJECT_ROOT)),manifest_sha256=file_sha(p))
    def test_prepare_run_resume_generic_domain(self):
        validate_config(self.config)
        rows,jobs,lock,first=prepare_run(self.config)
        self.assertEqual(lock["schema"],SCHEMA_V2);self.assertEqual(len(jobs),2)
        backend=FakeBackend();progress=run_jobs(self.config,backend,first_batch=True)
        self.assertEqual(progress["completed_jobs"],2);self.assertEqual(progress["upstream_unprompted_candidates"],1)
        count=len(backend.calls);run_jobs(self.config,backend);self.assertEqual(len(backend.calls),count)
        self.assertFalse(backend.active)
        for p in (self.base/"run/jobs").glob("*/attempt_*/RESULT.json"):
            result=read_json(p);self.assertEqual(result["split"],"train")
            self.assertEqual(result["schema"],SCHEMA_V2);self.assertEqual(result["rgb_variant"],"native_original")
            self.assertEqual(result["prediction_id"],result["candidate_id"])
            self.assertTrue(result["close_success"])
    def test_eval_mode_and_old_v1_source_boundary(self):
        self.config["split"]="eval"
        for row in self.rows:row["split"]="eval"
        self.write_manifest();self.assertEqual(len(load_manifest(self.config)[1]),2)
        old=copy.deepcopy(self.config);old["schema"]=SCHEMA;old["upstream_role"]="frozen_B0_parent_step0_post_soft_nms"
        for k in ("split","rgb_variant","source_contract"):old.pop(k)
        old["resources"]["cpu_affinity"]=[47,48]
        validate_config(old)
        with self.assertRaisesRegex(ValueError,"image identity"):load_manifest(old)
        old["resources"]["cpu_affinity"]=[88,89]
        with self.assertRaisesRegex(ValueError,"CPU47-48"):validate_config(old)
    def test_no_split_augmentation_or_wrong_source(self):
        for key,value in (("split","test"),("rgb_variant","brightness_aug")):
            bad=copy.deepcopy(self.config);bad[key]=value
            with self.assertRaises(ValueError):validate_config(bad)
        self.rows[0]["rgb_variant"]="brightness_aug";self.write_manifest()
        with self.assertRaisesRegex(ValueError,"native RGB"):load_manifest(self.config)
    def test_changed_prediction_mapping_or_gt_refused(self):
        original=copy.deepcopy(self.rows)
        for change in ("grid","score","ordinal","gt"):
            self.rows=copy.deepcopy(original);c=self.rows[0]["candidates"][0]
            if change=="grid":c["identity_binding"]["grid_index"]=123
            if change=="score":c["score"]=.9
            if change=="ordinal":c["identity_binding"]["raw_saved_ordinal"]=1
            if change=="gt":c["GT_instance_id"]="target"
            self.write_manifest()
            with self.assertRaises(ValueError):load_manifest(self.config)
    def test_release_bundle_and_completed_evidence_refused(self):
        source=self.config["source_contract"]
        p=contained(source["b0_bundle"]["path"]);p.chmod(0o644);p.write_bytes(b"replaced")
        with self.assertRaisesRegex(ValueError,"bundle bytes"):load_manifest(self.config)
    def test_ambiguity_failure_close_and_resume_preserved(self):
        b=FakeBackend("ambiguous");p=run_jobs(self.config,b)
        self.assertEqual(p["statuses"],{"ambiguous_identity":2});self.assertEqual(p["available_masks"],0);self.assertFalse(b.active)
        self.config["output"]=str((self.base/"error_run").relative_to(PROJECT_ROOT))
        b=FakeBackend("none")
        with self.assertRaises(NativeJobFailure):run_jobs(self.config,b)
        self.assertFalse(b.active)
        with self.assertRaises(NativeJobFailure):run_jobs(self.config,FakeBackend())
        p=run_jobs(self.config,FakeBackend(),retry_failed=True);self.assertEqual(p["completed_jobs"],2)
    def test_rehashed_incomplete_seal_refused(self):
        source=self.config["source_contract"];path=contained(source["sealed_role"]["path"])
        seal=read_json(path);seal["completion"]["evaluations"]=652
        path.chmod(0o644);path.write_text(json.dumps(seal));path.chmod(0o444)
        source["sealed_role"]["sha256"]=file_sha(path)
        for row in self.rows:row["source_receipt"]=copy.deepcopy(source)
        self.write_manifest()
        with self.assertRaisesRegex(ValueError,"Full completed"):load_manifest(self.config)
    def test_arbitrary_published_candidate_suppression_refused(self):
        self.rows[0]["candidates"][0]["eligible_for_SAM_diagnostic_prompt"]=False
        self.rows[0]["SAM_prompt_count"]=0;self.config["expected_prompts"]=1;self.write_manifest()
        with self.assertRaisesRegex(ValueError,"eligibility"):load_manifest(self.config)
    def test_zero_prompt_image_retained(self):
        # Preserve one image whose only saved candidate is below publication threshold.
        original=read_json(self.config["source_contract"]["predictions"]["path"])
        original[1]["instances"][0]["score"]=.2
        binding=self.bound("PREDICTIONS_ZERO.json",original)
        self.config["source_contract"]["predictions"]=binding
        for row in self.rows:
            row["source_receipt"]=copy.deepcopy(self.config["source_contract"])
            for candidate in row["candidates"]:
                candidate["identity_binding"]["prediction_file_sha256"]=binding["sha256"]
        candidate=self.rows[1]["candidates"][0]
        candidate["raw_saved_prediction"]["score"]=candidate["score"]=.2
        candidate["published"]=candidate["eligible_for_SAM_diagnostic_prompt"]=False
        self.rows[1]["SAM_prompt_count"]=0;self.config["expected_prompts"]=1;self.write_manifest()
        result=run_jobs(self.config,FakeBackend())
        self.assertEqual(result["completed_jobs"],1);self.assertEqual(result["image_domain_count"],2)
        self.assertEqual(result["upstream_unprompted_candidates"],2)

    def test_gpu03_never_allowed_even_with_fixture_authority(self):
        for gpu in range(4):
            bad=copy.deepcopy(self.config);bad["resources"]["physical_gpu"]=gpu
            with self.assertRaisesRegex(ValueError,"GPU4-7"):validate_config(bad)

if __name__=="__main__":unittest.main()
