import json,unittest
from pathlib import Path
import numpy as np
from PIL import Image
from data.action4_paths import PROJECT_ROOT
from data.action4_synthesis import _run_output_dir, _asset_identities, target_indices

class BlenderExtensionTests(unittest.TestCase):
    def test_runtime_evidence_routes_to_outputs(self):
        self.assertEqual(_run_output_dir(PROJECT_ROOT / "data/action4_synth/example/chunk"),
                         PROJECT_ROOT / "outputs/action4_synthesis_runs/action4_synth/example/chunk")
    def test_runtime_route_rejects_nondata(self):
        with self.assertRaises(ValueError):
            _run_output_dir(PROJECT_ROOT / "outputs/not_a_dataset")
    def test_real_single_render_masks_cover_full_body(self):
        for side,expected,presence in [("left",1,[True,False]),("right",2,[False,True])]:
            with self.subTest(side=side):
                root=PROJECT_ROOT/"data/action4_synth/server_r4_single_check_v001"/side
                self.assertTrue(root.exists(),"server integration fixture is required")
                record=json.loads((root/"manifest.jsonl").read_text())
                self.assertEqual(record["target_presence"],presence)
                self.assertEqual([x["instance_id"] for x in record["instances"]],[expected])
                self.assertEqual(list(target_indices(record)),[expected if x else 0 for x in presence])
                inst=record["instances"][0];box=inst["bbox_full_xyxy"]
                mask=np.array(Image.open(PROJECT_ROOT/inst["mask_visible"]))>127
                y,x=np.nonzero(mask);self.assertGreater(len(x),0)
                self.assertGreaterEqual(x.min(),box[0]-1);self.assertLessEqual(x.max(),box[2]+1)
                self.assertGreaterEqual(y.min(),box[1]-1);self.assertLessEqual(y.max(),box[3]+1)
                p=root/"masks"/f'{record["sample_id"]}_i{3-expected}_visible_0001.png'
                self.assertFalse(np.array(Image.open(p)).any())
    def test_material_provenance_frozen(self):
        path=PROJECT_ROOT/"configs/data/action4_server_native_material_after_v003.json"
        recipe=json.loads(path.read_text());identities=_asset_identities(recipe)
        self.assertEqual(len([p for p in identities if "server_textures_v00" in p]),9)
        manifest=json.loads((PROJECT_ROOT/recipe["material_evidence"]["denim_manifest"]).read_text())
        self.assertEqual(manifest["files"]["diff"]["color_space"],"sRGB")
        self.assertEqual(manifest["files"]["nor_gl"]["color_space"],"Non-Color")
        self.assertEqual(manifest["files"]["nor_gl"]["normal_convention"],"OpenGL +Y")
        self.assertTrue(np.allclose(manifest["physical_tile_m"],[.1,.1],atol=1e-7))

    def test_native_material_revision_preserves_target_geometry(self):
        root=PROJECT_ROOT
        before=[json.loads(x) for x in (root/"data/action4_synth/server_native_material_before_v001/chunk_000/manifest.jsonl").read_text().splitlines()]
        for i,a in enumerate(before):
            b=json.loads((root/f"data/action4_synth/server_native_material_after_v003/chunk_{i:03d}/manifest.jsonl").read_text())
            self.assertEqual(a["corners_3d_mm"],b["corners_3d_mm"])
            self.assertEqual(len(a["instances"]),len(b["instances"]))
            for x,y in zip(a["instances"],b["instances"]):
                self.assertEqual(x["corners_px"],y["corners_px"])
                self.assertEqual(x["object_to_world_m"],y["object_to_world_m"])
                self.assertEqual(x["mask_visible_sha256"],y["mask_visible_sha256"])

    def test_new_dataset_shards_do_not_contain_runtime_records(self):
        for i in range(2):
            data=PROJECT_ROOT/f"data/action4_synth/server_native_material_after_v003/chunk_{i:03d}"
            run=_run_output_dir(data)
            for name in ("launch.json","blender.log","render_completion.json","geometry_validation.json","synthesis_source_snapshot.py"):
                self.assertFalse((data/name).exists())
                self.assertTrue((run/name).is_file())

    def test_helper_identity_enabled_paths_and_disabled_baseline(self):
        import copy
        recipe=json.loads((PROJECT_ROOT/"configs/data/action4_server_native_material_after_v003.json").read_text())
        helper="src/data/action4_blender_foundation.py"
        foundation=recipe["foundation_blend"]
        for mode in ("baseline","blend","material","wrist"):
            with self.subTest(mode=mode):
                candidate=copy.deepcopy(recipe)
                candidate["asset_backend"]="blend_r4" if mode=="blend" else "procedural_r3"
                if mode!="material":candidate.pop("material_evidence",None)
                candidate["wrist_skin_union"]=mode=="wrist"
                identities=_asset_identities(candidate)
                self.assertEqual(helper in identities,mode!="baseline")
                self.assertEqual(foundation in identities,mode=="blend")

    def test_old_plan_rejects_helper_drift_without_mutating_source(self):
        import copy
        from unittest.mock import patch
        from data import action4_synthesis as module
        recipe=json.loads((PROJECT_ROOT/"configs/data/action4_server_native_material_after_v003.json").read_text())
        old_hash=module._hash
        for mode in ("blend","material","wrist"):
            with self.subTest(mode=mode):
                candidate=copy.deepcopy(recipe)
                candidate["asset_backend"]="blend_r4" if mode=="blend" else "procedural_r3"
                if mode!="material":candidate.pop("material_evidence",None)
                candidate["wrist_skin_union"]=mode=="wrist"
                identities=module._asset_identities(candidate)
                plan={"recipe":candidate,"asset_identities":identities,
                      "synthesis_code_sha256":old_hash(module.__file__),
                      "render_base_sha256":old_hash(PROJECT_ROOT/"src/data/action4_render.py")}
                def changed_hash(path):
                    if Path(path).name=="action4_blender_foundation.py":return "0"*64
                    return old_hash(path)
                with patch.object(module,"_hash",side_effect=changed_hash),patch.object(module,"_read",side_effect=lambda path: plan if Path(path).name=="plan.json" else json.loads(Path(path).read_text())):
                    self.assertNotEqual(module._asset_identities(candidate),identities)
                    with self.assertRaisesRegex(ValueError,"assets changed"):
                        module.render_shard("data/action4_synth/server_native_material_after_v003/plan.json",
                           "data/action4_synth/BQ01_mock_never_created","runtime/blender-3.6.23-linux-x64/blender")

    def test_consolidated_family_conditions_are_observed(self):
        records=[]
        for path in (PROJECT_ROOT/"data/action4_synth/server_contact_review_v008_preview").glob("chunk_*/manifest.jsonl"):
            records.extend(json.loads(line) for line in path.read_text().splitlines())
        self.assertEqual(sorted(x["index"] for x in records),list(range(8)))
        by_name={x["scene_family"]:x for x in records}
        self.assertEqual(len(by_name),8)
        self.assertEqual(by_name["hard_negative"]["instances"],[])
        self.assertGreater(max(x["occlusion_inframe"] for x in by_name["near_double"]["instances"]),.5)
        self.assertGreater(max(x["occlusion_inframe"] for x in by_name["self_forearm_occlusion"]["instances"]),.5)
        self.assertGreater(max(x["truncation"] for x in by_name["wall_transition_truncation"]["instances"]),.2)
        for name in ("table_fastening","table_turn","cloth_lift"):
            contact=by_name[name]["scene_parameters"]["garment_lift"]
            self.assertEqual(len(contact["minimum_hand_cloth_distance_m"]),2)
            self.assertLess(max(contact["minimum_hand_cloth_distance_m"]),.002)
            self.assertIn("evaluated",contact["distance_basis"])
        for record in records:
            self.assertEqual(record["geometry_collision_check"]["intersections"],[])

    def test_consolidated_plan_keeps_snapshot_identity_and_rejects_source_drift(self):
        import hashlib
        import tempfile
        from unittest import mock
        from data import action4_synthesis as synthesis
        plan_path=PROJECT_ROOT/"data/action4_synth/server_contact_review_v008_preview/plan.json"
        plan=json.loads(plan_path.read_text())
        source_snapshot=PROJECT_ROOT/"outputs/action4_synthesis_runs/action4_synth/server_contact_review_v008_preview/chunk_000_003/synthesis_source_snapshot.py"
        helper_snapshot=PROJECT_ROOT/"outputs/action4_blender_review_contact_v008/source_snapshot/action4_blender_foundation.py"
        self.assertEqual(plan["synthesis_code_sha256"],hashlib.sha256(source_snapshot.read_bytes()).hexdigest())
        self.assertEqual(plan["asset_identities"]["src/data/action4_blender_foundation.py"],hashlib.sha256(helper_snapshot.read_bytes()).hexdigest())
        self.assertTrue(plan["recipe"]["connected_garment"])
        self.assertFalse(any(f.get("thumb_opposition") for f in plan["recipe"]["families"]))
        actual_hash=synthesis._hash
        def changed_source(path):
            return "0"*64 if Path(path).resolve()==Path(synthesis.__file__).resolve() else actual_hash(path)
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT/"outputs") as temp:
            with mock.patch.object(synthesis,"_hash",side_effect=changed_source), mock.patch.object(synthesis.subprocess,"run",side_effect=AssertionError("stale plan must not launch")):
                with self.assertRaisesRegex(ValueError,"Renderer changed"):
                    synthesis.render_shard(plan_path,Path(temp)/"not_created",PROJECT_ROOT/"runtime/blender-3.6.23-linux-x64/blender",limit=1)
            self.assertFalse((Path(temp)/"not_created").exists())

if __name__=="__main__":unittest.main()
