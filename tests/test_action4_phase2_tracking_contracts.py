import unittest
import numpy as np
from model.action4_phase2_tracking_contracts import (
    AnchorKey, Binding, AnchorRegistry, partition_geometry, assert_consumers_exclude_heldout,
    CropTransform, support_queries, query_local_times, reverse_query_times,
    merge_bidirectional, GateScale, assert_shared_gate_scales, WindowIdentity, owner_boundaries, overlap_diagnostic, cycle_diagnostic)


class TrackingContractsTests(unittest.TestCase):
    def fixture(self, n=14):
        keys = [AnchorKey("t", str(i), 0) for i in range(n)]
        registry = AnchorRegistry([Binding(k, (i, 0, 0), "A0_estimated", 10, "provisional", i >= 12) for i, k in enumerate(keys)])
        return keys, registry

    def test_heldout_changes_cannot_change_solver_packet_or_decision(self):
        ids, registry = self.fixture()
        uv = np.arange(28.).reshape(14, 2)
        kwargs = dict(registry=registry, ids=ids, observed=np.ones(14, bool), is_query=np.zeros(14, bool),
                      weights=np.ones(14), grid_ids=[i % 4 for i in range(14)])
        fit, scoring = partition_geometry(predicted_uv=uv, human_uv=uv, **kwargs)
        for replacement in (1e9, np.nan, -999.):
            changed = uv.copy(); changed[12:] = replacement
            other, score2 = partition_geometry(predicted_uv=changed, human_uv=changed, **kwargs)
            for field in ("xyz_m", "predicted_uv", "weights"):
                np.testing.assert_array_equal(getattr(fit, field), getattr(other, field))
            self.assertEqual((fit.ids, fit.count, fit.grid_count, fit.eligible), (other.ids, other.count, other.grid_count, other.eligible))
        self.assertTrue(fit.eligible)
        self.assertEqual(len(scoring["ids"]), 2)
        with self.assertRaises(ValueError):
            fit.predicted_uv.flags.writeable = True

    def test_recompute_count_and_coverage_after_exclusion(self):
        ids, registry = self.fixture()
        fit, _ = partition_geometry(registry, ids, np.zeros((14, 2)), np.ones(14, bool),
                                    np.zeros(14, bool), np.ones(14), [0] * 12 + [1, 2])
        self.assertEqual(fit.grid_count, 1)
        self.assertFalse(fit.eligible)

    def test_mixed_tracks_cannot_combine_into_twelve_fit_points(self):
        keys = [AnchorKey("cameraA" if i < 6 else "cameraB", str(i), i % 2) for i in range(12)]
        registry = AnchorRegistry([Binding(k, (0, 0, 0), "A0", 0, "active") for k in keys])
        with self.assertRaisesRegex(ValueError, "multiple track"):
            partition_geometry(registry, keys, np.zeros((12, 2)), np.ones(12, bool), np.zeros(12, bool), np.ones(12), [i % 4 for i in range(12)])
        keys = [AnchorKey("one_track", str(i), i % 2) for i in range(12)]
        registry = AnchorRegistry([Binding(k, (0, 0, 0), "A0", 0, "active") for k in keys])
        fit, _ = partition_geometry(registry, keys, np.zeros((12, 2)), np.ones(12, bool), np.zeros(12, bool), np.ones(12), [i % 4 for i in range(12)])
        self.assertTrue(fit.eligible)  # Anchor generations need not match one another.
        ids, registry = self.fixture()
        query = np.zeros(14, bool); query[0] = True
        fit, _ = partition_geometry(registry, ids, np.zeros((14, 2)), np.ones(14, bool), query,
                                    np.ones(14), [i % 4 for i in range(14)])
        self.assertEqual(fit.count, 11)
        self.assertFalse(fit.eligible)

    def test_identity_immutable_and_consumer_assertions(self):
        ids, registry = self.fixture()
        with self.assertRaises(AttributeError): registry._bindings = {}
        with self.assertRaises(ValueError): AnchorRegistry([registry[ids[0]], registry[ids[0]]])
        with self.assertRaises(KeyError): registry[AnchorKey("t", "0", 1)]
        for name in ("solver_ids", "acceptance_ids", "fallback_ids"):
            kwargs = dict(solver_ids=[], acceptance_ids=[], fallback_ids=[]); kwargs[name] = [ids[-1]]
            with self.assertRaises(AssertionError): assert_consumers_exclude_heldout(registry, **kwargs)

    def test_same_material_multiple_generations_cannot_fill_fit_gate(self):
        keys = [AnchorKey("t", str(i), 0) for i in range(11)] + [AnchorKey("t", "0", 1)]
        registry = AnchorRegistry([Binding(k, (0, 0, 0), "A0", 0, "active") for k in keys])
        with self.assertRaisesRegex(ValueError, "multiple generations"):
            partition_geometry(registry, keys, np.zeros((12, 2)), np.ones(12, bool),
                               np.zeros(12, bool), np.ones(12), [i % 4 for i in range(12)])

    def test_empty_packet_and_invalid_fit(self):
        fit, score = partition_geometry(AnchorRegistry([]), [], np.empty((0, 2)), np.zeros(0, bool), np.zeros(0, bool), np.zeros(0), [])
        self.assertEqual(fit.xyz_m.shape, (0, 3)); self.assertFalse(fit.eligible)
        ids, registry = self.fixture()
        with self.assertRaises(ValueError):
            partition_geometry(registry, ids, np.full((14, 2), np.nan), np.ones(14, bool), np.zeros(14, bool), np.ones(14), [0] * 14)

    def test_crop_roundtrip_endpoints_and_changed_crop(self):
        rng = np.random.default_rng(19)
        for crop in (CropTransform(-21, -9, 900, 500), CropTransform(100, 40, 2, 2), CropTransform(8, 30, 512, 384)):
            points = rng.normal(size=(12, 2)) * 100
            np.testing.assert_allclose(crop.to_native(crop.to_model(points)), points, atol=1e-10)
            np.testing.assert_allclose(crop.to_model([[crop.x0, crop.y0], [crop.x0 + crop.width - 1, crop.y0 + crop.height - 1]]), [[0, 0], [511, 383]])
            float32_roundtrip = crop.to_native(crop.to_model(points).astype(np.float32))
            self.assertLess(np.max(np.abs(float32_roundtrip - points)), 1e-3)
        with self.assertRaises(ValueError): CropTransform(0, 0, 1, 2)

    def test_support_grid_padding_and_no_xyz(self):
        q = support_queries(CropTransform(0, 0, 512, 384), 512, 384)
        self.assertEqual(q.shape, (36, 3)); self.assertTrue((q[:, 0] == 0).all())
        np.testing.assert_allclose(q[:, 1].min(), 8); np.testing.assert_allclose(q[:, 1].max(), 504)
        padded = support_queries(CropTransform(-256, 0, 512, 384), 512, 384)
        self.assertEqual(len(padded), 18)

    def test_local_time_not_global_frame_id(self):
        times = query_local_times([100, 102, 106], [106, 100])
        np.testing.assert_array_equal(times, [2, 0])
        np.testing.assert_array_equal(reverse_query_times(times, 3), [0, 2])
        with self.assertRaises(ValueError): reverse_query_times(np.array([100]), 3)
        with self.assertRaises(ValueError): query_local_times([106, 100, 102], [100])
        np.testing.assert_array_equal(query_local_times(np.array([100, 102, 106], dtype=np.int64), np.array([102], dtype=np.int64)), [1])

    def test_bidirectional_three_outputs_query_raw_preservation(self):
        shape = (3, 2)
        f = (np.ones((*shape, 2)), np.full(shape, .2), np.full(shape, .3))
        b = (np.full((*shape, 2), 9.), np.arange(6.).reshape(shape) / 10, np.full(shape, .7))
        result = merge_bidirectional(f, b, np.array([0, 2]), np.array([[4., 5.], [6., 7.]]))
        self.assertEqual(result["raw"][0][0, 1, 0], 9)
        self.assertEqual(result["raw"][1][0, 1], .5)
        self.assertEqual(result["raw"][2][0, 1], .7)
        self.assertEqual(result["raw"][0][2, 1, 0], 1)
        np.testing.assert_array_equal(result["query_corrected_xy"][2, 1], [6, 7])
        self.assertEqual(result["raw"][2][2, 1], .3)
        self.assertEqual(f[0][2, 1, 0], 1)
        with self.assertRaises(ValueError): merge_bidirectional((f[0], f[1].astype(bool), f[2]), b, np.array([0, 2]), np.zeros((2, 2)))

    def test_one_frame_no_temporal_observation(self):
        output = (np.zeros((1, 1, 2)), np.ones((1, 1), float), np.ones((1, 1), float))
        result = merge_bidirectional(output, output, np.array([0]), np.zeros((1, 2)))
        self.assertFalse(result["nonquery_candidate_mask"].any())
        self.assertNotIn("temporal_observed", result)

    def test_invalid_probabilities_preserved_as_diagnostics_not_observations(self):
        output = (np.zeros((2, 1, 2)), np.array([[.9], [np.nan]]), np.array([[.7], [1.2]]))
        result = merge_bidirectional(output, output, np.array([0]), np.zeros((1, 2)))
        self.assertTrue(result["nonquery_candidate_mask"][1, 0])
        self.assertFalse(result["visibility_probability_valid"][1, 0])
        self.assertFalse(result["confidence_probability_valid"][1, 0])
        self.assertTrue(np.isnan(result["raw"][1][1, 0]))
        self.assertEqual(result["raw"][2][1, 0], 1.2)

    def test_owners_mixed_tail_and_conflicts(self):
        self.assertEqual(owner_boundaries([(0, 120), (100, 160), (120, 240)]), [(0, 110), (110, 140), (140, 240)])
        self.assertEqual(owner_boundaries([(0, 120), (80, 121)]), [(0, 100), (100, 121)])
        with self.assertRaises(ValueError): owner_boundaries([(0, 120), (120, 160)])
        s = GateScale(400, "full", 0, True, "upstream-fixed")
        a = np.zeros((3, 2)); v = np.ones(3, bool); q = np.zeros(3, bool)
        identity = dict(frame_ids_a=[5, 6, 7], frame_ids_b=[5, 6, 7], anchor_key_a=AnchorKey("t", "a", 0), anchor_key_b=AnchorKey("t", "a", 0),
                        window_a=WindowIdentity("t", 0, "w0", 0, 10), window_b=WindowIdentity("t", 0, "w1", 5, 15))
        self.assertEqual(overlap_diagnostic(a, a + [2, 0], v, v, q, q, [s] * 3, **identity)["status"], "consistent")
        self.assertEqual(overlap_diagnostic(a, a + [5, 0], v, v, q, q, [s] * 3, **identity)["status"], "conflict")
        q[0] = True
        self.assertEqual(overlap_diagnostic(a, a, v, v, q, q, [s] * 3, **identity)["status"], "unknown")
        identity["anchor_key_b"] = AnchorKey("other", "a", 0)
        with self.assertRaises(ValueError): overlap_diagnostic(a, a, v, v, q, q, [s] * 3, **identity)
        identity["anchor_key_b"] = identity["anchor_key_a"]
        identity["frame_ids_b"] = [5, 7, 6]
        with self.assertRaises(ValueError): overlap_diagnostic(a, a, v, v, q, q, [s] * 3, **identity)
        identity["frame_ids_b"] = identity["frame_ids_a"]
        identity["window_b"] = WindowIdentity("t", 1, "w1", 5, 15)
        with self.assertRaises(ValueError): overlap_diagnostic(a, a, v, v, q, q, [s] * 3, **identity)

    def test_cycle_unavailable_and_frozen_scale_provenance(self):
        s = GateScale(400, "history", 17, True, "upstream-fixed")
        self.assertIsNone(cycle_diagnostic([0, 0], None, s)["error_px"])
        self.assertTrue(cycle_diagnostic([0, 0], [2, 0], s)["passed"])
        self.assertFalse(cycle_diagnostic([0, 0], [2.1, 0], s)["passed"])
        with self.assertRaises(ValueError): GateScale(400, "gt_bbox", 17, True, "gt")

    def test_shared_gate_scales_reject_config_drift(self):
        a = {("t", 17): GateScale(400, "history", 10, True, "frozen")}
        b = {("t", 17): GateScale(401, "history", 10, True, "frozen")}
        self.assertTrue(assert_shared_gate_scales({"C0": a, "C1": a, "C2": a}))
        with self.assertRaises(ValueError): assert_shared_gate_scales({"C0": a, "C1": b})


if __name__ == "__main__": unittest.main()
