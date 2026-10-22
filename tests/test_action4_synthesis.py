"""Host-side synthesis contracts; actual Blender renders are separate evidence."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from data.action4_synthesis import ROOT, assemble, create_plan, create_s1_plan, validate_request_recipe


class SynthesisContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="synthesis_contract_", dir=ROOT / "data")
        self.directory = Path(self.temporary.name)
        self.recipe = ROOT / "configs/data/action4_synthesis_r3_v004.json"
        self.controls = {
            "scene_family": "self_forearm_occlusion", "view": {"wrist_delta_xyz_deg": [2., -3., 1.]},
            "scale": 1.05, "occluder": "scene_default", "occluder_location": "linked_geometry",
            "allowed_perturbations": {"rotation_jitter_deg": 2., "position_jitter_m": [.002, .001, .001],
                                       "light_intensity_gain": 1.03, "exposure_delta_ev": -.04},
            "reference_frame_ids": [1980],
        }

    def tearDown(self):
        self.temporary.cleanup()

    def request(self, rid, event, quota):
        return {"request_id": rid, "source_event_id": event, "quota": quota,
                "status": "reviewed_request_pending_render", "split": "train",
                "source_frame_start": 1970, "source_frame_end": 1990,
                "recipe": copy.deepcopy(self.controls),
                "ground_truth": "recompute_from_renderer_geometry",
                "prediction_coordinates_used_as_training_labels": False}

    def test_s0_exact_quotas_group_separation_and_immutability(self):
        output = self.directory / "plan.json"
        plan = create_plan(self.recipe, output, 10000, 20261019)
        self.assertEqual(plan["split_counts"], {"train": 8000, "eval": 1000, "test": 1000})
        self.assertEqual(plan["instance_counts_by_split"], {"train":{"0":400,"1":1200,"2":6400}, "eval":{"0":50,"1":150,"2":800}, "test":{"0":50,"1":150,"2":800}})
        self.assertEqual(plan["family_counts"]["hard_negative"], 500)
        self.assertFalse(plan["preview_coverage_exception"])
        groups = {}
        for entry in plan["entries"]:
            groups.setdefault(entry["group_id"], set()).add(entry["split"])
        self.assertTrue(all(len(splits) == 1 for splits in groups.values()))
        self.assertEqual(len({e["sample_id"] for e in plan["entries"]}), 10000)
        self.assertEqual(len({e["scene_family"] for e in plan["entries"][:8]}), 8)
        self.assertTrue(plan["asset_identities"])
        with self.assertRaises(FileExistsError):
            create_plan(self.recipe, output, 10000, 20261019)

    def test_s1_consumes_exact_event_quotas_and_controls_train_only(self):
        requests = [self.request("r01", "event01", 3), self.request("r02", "event02", 5)]
        source = self.directory / "requests.jsonl"
        source.write_text("".join(json.dumps(r) + "\n" for r in requests))
        plan = create_s1_plan(self.recipe, source, self.directory / "s1.json", max_count=8)
        self.assertEqual(plan["request_counts"], {"r01": 3, "r02": 5})
        self.assertEqual(plan["split_counts"], {"train": 8, "eval": 0, "test": 0})
        self.assertTrue(all(e["request_recipe"] == self.controls for e in plan["entries"]))
        self.assertTrue(all(e["split"] == "train" for e in plan["entries"]))
        with self.assertRaisesRegex(ValueError, "budget"):
            create_s1_plan(self.recipe, source, self.directory / "too_many.json", max_count=7)

    def test_rejects_unimplemented_scene_controls_and_excessive_perturbation(self):
        families = {"self_forearm_occlusion"}
        self.assertEqual(validate_request_recipe(self.controls, families), self.controls)
        for key, value in (("occluder", "random_black_rectangle"), ("scale", 4), ("unused_override", 1)):
            controls = copy.deepcopy(self.controls)
            controls[key] = value
            with self.assertRaises(ValueError):
                validate_request_recipe(controls, families)
        controls = copy.deepcopy(self.controls)
        controls["allowed_perturbations"]["position_jitter_m"][0] = .20
        with self.assertRaisesRegex(ValueError, "Position jitter"):
            validate_request_recipe(controls, families)

    def test_rejects_pseudo_labels_unreviewed_test_request_and_outside_event(self):
        for mutation in (
            {"prediction_coordinates_used_as_training_labels": True},
            {"status": "unreviewed"}, {"split": "test"},
            {"source_frame_start": 1981},
        ):
            request = self.request("r", "event", 1)
            request.update(mutation)
            source = self.directory / ("request_" + str(len(list(self.directory.iterdir()))) + ".jsonl")
            source.write_text(json.dumps(request) + "\n")
            with self.assertRaises(ValueError):
                create_s1_plan(self.recipe, source, self.directory / "invalid.json")

    def test_single_target_sides_are_balanced_without_changing_quotas(self):
        from data.action4_synthesis import target_indices
        a=create_plan(self.recipe,self.directory / "sides_a.json",10000,20261019)
        b=create_plan(self.recipe,self.directory / "sides_b.json",10000,20261019)
        self.assertEqual(a["entries"],b["entries"])
        for split in ("train","eval","test"):
            singles=[e for e in a["entries"] if e["split"]==split and e["target_count"]==1]
            self.assertEqual(sum(e["target_presence"][0] for e in singles),len(singles)//2)
            self.assertEqual(sum(e["target_presence"][1] for e in singles),len(singles)//2)
        for e in a["entries"]:
            self.assertEqual(sum(e["target_presence"]),e["target_count"])
            self.assertEqual(sum(bool(v) for v in target_indices(e)),e["target_count"])
        self.assertEqual({tuple(e["target_presence"]) for e in a["entries"][:128]},
                         {(False,False),(True,False),(False,True),(True,True)})

    def test_s1_right_only_presence_and_invalid_values(self):
        from data.action4_synthesis import target_indices,validate_target_presence
        from eval.action4_feedback import validate_synthesis_recipe
        request=self.request("right_only","event_right",3)
        request["recipe"]["target_presence"]=[False,True]
        validate_synthesis_recipe(request["recipe"])
        source=self.directory / "right_requests.jsonl"
        source.write_text(json.dumps(request)+"\n")
        result=create_s1_plan(self.recipe,source,self.directory / "right_plan.json",max_count=3)
        self.assertTrue(all(e["target_count"]==1 and target_indices(e)==[0,2] for e in result["entries"]))
        for bad in ([0,1],[True],None,[False,False]):
            with self.assertRaises(ValueError):
                validate_target_presence(bad,"table_turn")
        with self.assertRaises(ValueError):
            validate_target_presence([True,False],"hard_negative")

    def test_legacy_target_entry_keeps_side_identity(self):
        from data.action4_synthesis import target_indices
        self.assertEqual(target_indices({"target_count":1}),[1,0])
        self.assertEqual(target_indices({"target_count":2}),[1,2])
        self.assertEqual(target_indices({"target_count":0}),[0,0])
        with self.assertRaises(ValueError):
            target_indices({"target_count":2,"target_presence":[False,True]})

    def test_acceptance_rejects_flag_only_preview_promotion_and_mixed_dimensions(self):
        # Assembly boundary fixture only: the earlier geometry prerequisite is
        # isolated, and no image or accepted dataset is manufactured by this test.
        cases = [([812, 616], [812, 616], "Accepted synthesis requires native resolution"),
                 ([3248, 2464], [812, 616], "Every accepted sample must match native resolution"),
                 ([3248, 2464], [3248, 2464], "every declared visual/geometry criterion")]
        for index, (recipe_wh, record_wh, error) in enumerate(cases):
            fixture = self.directory / f"acceptance_fixture_{index}"
            fixture.mkdir()
            plan_path = fixture / "plan.json"
            plan_path.write_text(json.dumps({"entries": [{"sample_id": "fixture_only"}],
                                            "recipe": {"resolution_wh": recipe_wh,
                                                       "quality_gate": {"native_resolution_pending": False}}}))
            identity = hashlib.sha256(plan_path.read_bytes()).hexdigest()
            (fixture / "launch.json").write_text(json.dumps({"plan_sha256": identity}))
            (fixture / "manifest.jsonl").write_text(json.dumps({"sample_id": "fixture_only", "width": record_wh[0], "height": record_wh[1]}) + "\n")
            acceptance = fixture / "acceptance.json"
            acceptance.write_text(json.dumps({"decision": "accepted", "plan_sha256": identity}))
            output = fixture / "assembled"
            with patch("data.action4_synthesis.validate_shard", return_value={}), self.assertRaisesRegex(ValueError, error):
                assemble(plan_path, [fixture], output, acceptance)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
