import copy
import unittest
import numpy as np
from eval.action4_phase2_sparse_tracking_metrics import score_sparse, compare_common_domain, VisibleBoxScale


class SparseMetricsTests(unittest.TestCase):
    def base(self, n=4):
        return dict(gt_uv=np.zeros((n, 2)), prediction_uv=np.zeros((n, 2)),
                    annotation_valid=np.ones(n, bool), gt_visible=np.ones(n, bool), is_query=np.zeros(n, bool),
                    prediction_available=np.ones(n, bool), accepted=np.ones(n, bool), predicted_visible=np.ones(n, bool),
                    visibility_response_available=np.ones(n, bool),
                    image_width=512, image_height=256)

    def test_missing_is_failure_not_removed_denominator(self):
        k = self.base(); k["prediction_uv"][3] = [100, 0]
        before = score_sparse(**k)
        k["prediction_available"][3] = False; k["accepted"][3] = False
        after = score_sparse(**k)
        self.assertEqual(before["raw"]["pck"]["2"], .75)
        self.assertEqual(after["raw"]["pck"]["2"], .75)
        self.assertEqual(after["raw"]["coverage"], .75)
        self.assertEqual(after["tap_sparse"]["thresholds"]["1"]["fp"], 1)

    def test_unknown_query_never_success_or_occlusion(self):
        k = self.base(); k["annotation_valid"][1] = False; k["is_query"][2] = True
        k["gt_uv"][1:] = np.nan; k["gt_uv"][2] = 0; k["gt_visible"][3] = False
        result = score_sparse(**k)
        self.assertEqual(result["visible_count"], 1); self.assertEqual(result["known_invisible_count"], 1)
        self.assertEqual(result["tap_sparse"]["thresholds"]["1"]["tp"], 1)
        self.assertEqual(result["tap_sparse"]["thresholds"]["1"]["fp"], 1)
        self.assertEqual(result["accepted"]["false_visible_rate"], 1.)

    def test_native_thresholds_strict_and_tap_full_image(self):
        k = self.base(3); k["prediction_uv"][:, 0] = [2, 5, 10]
        result = score_sparse(**k)
        self.assertEqual(result["raw"]["pck"], {"2": 0., "5": 1/3, "10": 2/3})
        self.assertEqual(result["tap_sparse"]["thresholds"]["1"]["pck"], 0.)
        self.assertEqual(result["tap_sparse"]["thresholds"]["2"]["pck"], 1/3)

    def test_sparse_tap_hand_counted_denominators(self):
        k = self.base(6)
        k["gt_visible"] = np.array([True, True, True, False, True, False])
        k["annotation_valid"][-1] = False; k["is_query"][4] = True
        k["prediction_uv"][1] = [20, 0]
        k["prediction_uv"][2] = np.nan; k["accepted"][2] = False
        r = score_sparse(**k)["tap_sparse"]
        # Three visible GT rows: one correct, one wrong, one missing. Also one
        # known-invisible false-visible. Query and unknown do not participate.
        self.assertEqual(r["thresholds"]["1"]["tp"], 1)
        self.assertEqual(r["thresholds"]["1"]["fp"], 3)
        self.assertEqual(r["thresholds"]["1"]["jaccard"], 1/6)
        self.assertEqual(r["thresholds"]["1"]["pck"], 1/3)
        self.assertEqual(r["occlusion_accuracy"], .75)

    def test_reject_all_does_not_improve_raw(self):
        k = self.base(); k["prediction_uv"][0] = [80, 0]; k["accepted"][:] = False
        r = score_sparse(**k)
        self.assertEqual(r["accepted"]["coverage"], 0)
        self.assertIsNone(r["accepted"]["epe_px"]["median"])
        self.assertEqual(r["raw"]["pck"]["2"], .75)

    def test_normalized_requires_independent_visible_annotation(self):
        k = self.base()
        self.assertEqual(score_sparse(**k)["normalized"]["status"], "not_measured")
        with self.assertRaises(ValueError): VisibleBoxScale([10] * 4, "seed-envelope", "six_point_envelope")
        k["visible_box_scale"] = VisibleBoxScale([100, np.nan, 100, 100], "human-box-v1")
        self.assertEqual(score_sparse(**k)["normalized"]["scale_coverage"], .75)

    def test_all_invisible_and_empty_domains(self):
        k = self.base(); k["gt_visible"][:] = False; k["accepted"][:] = False; k["predicted_visible"][:] = False
        r = score_sparse(**k)
        self.assertIsNone(r["raw"]["pck"]["2"]); self.assertEqual(r["tap_sparse"]["occlusion_accuracy"], 1)
        k["annotation_valid"][:] = False
        self.assertEqual(score_sparse(**k)["status"], "insufficient_evidence")

    def test_nonfinite_predictions_and_invalid_shapes(self):
        k = self.base(); k["prediction_uv"][0] = np.nan
        with self.assertRaises(ValueError): score_sparse(**k)
        k["accepted"][0] = False
        self.assertEqual(score_sparse(**k)["raw"]["missing_count"], 1)
        k["annotation_valid"] = np.ones(4)
        with self.assertRaises(ValueError): score_sparse(**k)

    def test_missing_response_is_not_legitimate_invisible_response(self):
        k = self.base(3)
        k["gt_visible"][:] = False; k["predicted_visible"][:] = False; k["accepted"][:] = False
        k["visibility_response_available"][0] = False
        r = score_sparse(**k)["tap_sparse"]
        self.assertEqual(r["occlusion_accuracy"], 1.)
        self.assertEqual(r["visibility_response_coverage"], 2/3)
        self.assertEqual(r["missing_visibility_response_count"], 1)
        k["visibility_response_available"][:] = False
        self.assertIsNone(score_sparse(**k)["tap_sparse"]["occlusion_accuracy"])
        k["gt_visible"][:] = True; k["prediction_available"][:] = False
        self.assertEqual(score_sparse(**k)["raw"]["pck"]["2"], 0.)
        k["accepted"][0] = True
        with self.assertRaises(ValueError): score_sparse(**k)

    def common_fixture(self):
        ids = [("t", str(i), 0, 100 + i) for i in range(4)]
        run = dict(annotation_ids=ids, registered_query_ids=[], prediction_uv=np.zeros((4, 2)),
                   prediction_available=np.ones(4, bool), accepted=np.ones(4, bool), predicted_visible=np.ones(4, bool), visibility_response_available=np.ones(4, bool))
        return dict(annotation_ids=ids, annotation_valid=np.ones(4, bool), gt_visible=np.ones(4, bool),
                    gt_uv=np.zeros((4, 2)), groups=["easy", "easy", "hard", "hard"],
                    runs={c: copy.deepcopy(run) for c in ("C0", "C1", "C2")}, image_width=512, image_height=256,
                    required_groups=["easy", "hard"], common_annotation_manifest_id="frozen-human-manifest-v1")

    def test_requires_every_expected_run_and_visible_required_groups(self):
        k = self.common_fixture(); del k["runs"]["C2"]
        with self.assertRaises(ValueError): compare_common_domain(**k)
        k["expected_run_names"] = ("C0", "C1")
        with self.assertRaises(ValueError): compare_common_domain(**k)
        k = self.common_fixture(); k["gt_visible"][2:] = False
        r = compare_common_domain(**k)
        self.assertEqual(r["status"], "insufficient_evidence")
        self.assertEqual(r["common_visible_count"], 2)
        self.assertEqual(r["common_visible_group_counts"]["hard"], 0)
        self.assertEqual(r["runs"]["C0"]["system_seed_failure_counts"]["status"], "not_measured")

    def test_failed_run_stays_in_common_comparison(self):
        k = self.common_fixture()
        failed = k["runs"]["C2"]
        for field in ("prediction_available", "accepted", "predicted_visible", "visibility_response_available"):
            failed[field][:] = False
        failed["prediction_uv"][:] = np.nan
        failed["system_seed_failure_counts"] = {"inference_failure": 1}
        r = compare_common_domain(**k)
        self.assertEqual(r["common_count"], 4)
        self.assertEqual(r["runs"]["C2"]["common"]["raw"]["pck"]["2"], 0.)
        self.assertEqual(r["runs"]["C2"]["common"]["tap_sparse"]["visibility_response_coverage"], 0.)
        self.assertEqual(r["per_run_common_missing"]["C2"], 4)

    def test_manifest_identity_includes_requests_outside_sparse_rows(self):
        k = self.common_fixture(); before = compare_common_domain(**k)
        k["runs"]["C2"]["registered_query_ids"].append(("t", "a", 5, 999))
        k["runs"]["C2"]["system_seed_failure_counts"] = {"seed_invalid": 2}
        after = compare_common_domain(**k)
        self.assertEqual(before["common_mask"], after["common_mask"])
        self.assertNotEqual(before["query_union_manifest_id"], after["query_union_manifest_id"])
        self.assertEqual(after["common_annotation_manifest_id"], "frozen-human-manifest-v1")
        self.assertEqual(after["runs"]["C2"]["system_seed_failure_counts"]["counts"], {"seed_invalid": 2})
        k["runs"]["C3"] = copy.deepcopy(k["runs"]["C2"])
        with self.assertRaises(ValueError): compare_common_domain(**k)
        k["expected_run_names"] = ("C0", "C1", "C2", "C3")
        self.assertEqual(compare_common_domain(**k)["expected_run_names"], ["C0", "C1", "C2", "C3"])

    def test_common_missing_and_failure_kept(self):
        k = self.common_fixture(); k["runs"]["C1"]["prediction_available"][3] = False; k["runs"]["C1"]["accepted"][3] = False
        r = compare_common_domain(**k)
        self.assertEqual(r["common_count"], 4)
        self.assertEqual(r["runs"]["C1"]["common"]["raw"]["pck"]["2"], .75)
        self.assertEqual(r["per_run_common_missing"]["C1"], 1)
        self.assertEqual(r["group_counts"]["hard"], 2)

    def test_query_union_rescores_every_config_and_identity(self):
        k = self.common_fixture(); before = compare_common_domain(**k)
        k["runs"]["C2"]["registered_query_ids"] = [k["annotation_ids"][3]]
        after = compare_common_domain(**k)
        self.assertNotEqual(before["common_mask_identity"], after["common_mask_identity"])
        for r in after["runs"].values(): self.assertEqual(r["common"]["visible_count"], 3)
        self.assertEqual(after["runs"]["C0"]["own"]["visible_count"], 4)
        self.assertEqual(after["query_excluded_fraction"], .25)

    def test_misalignment_and_required_empty(self):
        k = self.common_fixture(); k["runs"]["C1"]["annotation_ids"] = list(reversed(k["annotation_ids"]))
        with self.assertRaises(ValueError): compare_common_domain(**k)
        k = self.common_fixture(); k["runs"]["C2"]["registered_query_ids"] = k["annotation_ids"][2:]
        result = compare_common_domain(**k)
        self.assertEqual(result["status"], "insufficient_evidence"); self.assertEqual(result["empty_required_groups"], ["hard"])
        k["runs"]["C2"]["registered_query_ids"] = k["annotation_ids"]
        self.assertEqual(compare_common_domain(**k)["common_count"], 0)


if __name__ == "__main__": unittest.main()
