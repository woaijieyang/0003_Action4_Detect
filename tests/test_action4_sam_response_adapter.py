"""Official-schema-shaped CPU fixtures. No SAM inference or mask accuracy evidence."""
import unittest
from dataclasses import replace
import numpy as np
from model.action4_sam_session_contracts import RegionSpec, SeedBinding, SessionLedger, AbsenceDecision
from model.action4_sam_response_adapter import (
    ResponseEnvelope, ResponseSchemaError, ResponseIdentityError, MissingOutput,
    MaskRegistrationReceipt, validate_raw_response, parse_target_response, ingest_response)


class SamResponseAdapterTests(unittest.TestCase):
    def setup_session(self, *, bound=True):
        region = RegionSpec("fixture-region", (100, 101, 102), (0, 400, 800), (1, 11988), "frames-fixed", "native-rgb", 4, 3)
        seed = SeedBinding("camera", region.identity, "candidate", 1, 101, (.1, .1, .2, .2), ("frozen-b0",), "frozen_b0")
        ledger = SessionLedger(); record = ledger.plan(region, seed, worker_id="cpu-fixture", device_identity="no-gpu", dependency_identity="fixed-upstream")
        ledger.begin_create(record.record_id); ledger.created(record.record_id, "sid")
        ledger.seed_request(record.record_id)
        if bound: ledger.bind_seed(record.record_id, [2])
        return ledger, ledger.get(record.record_id), region

    def envelope(self, record, index=None, response_id="response"):
        return ResponseEnvelope(record.session_id, record.region_identity, record.generation, record.attempt,
                                record.active_operation, response_id, index)

    def raw(self, index=0, ids=(9, 2)):
        n = len(ids); masks = np.zeros((n, 3, 4), dtype=bool)
        for i in range(n): masks[i, i % 3, i % 4] = True
        return {"frame_index": np.int64(index), "outputs": {
            "out_obj_ids": np.array(ids, dtype=np.int64), "out_probs": np.full(n, .8, np.float32),
            "out_boxes_xywh": np.tile(np.array([.1, .1, .3, .3], np.float32), (n, 1)),
            "out_binary_masks": masks, "frame_stats": None}}

    def registry(self):
        records = {}
        def register(request):
            records[request.mask_identity] = {"mask": request.mask.copy(), "request": request}
            return MaskRegistrationReceipt(request.mask_identity, request.pixel_sha256,
                                           request.binding_identity, "in-memory-fixture:" + request.mask_identity)
        return records, register

    def test_numpy_ids_convert_only_after_integer_schema_validation_then_bind_seed(self):
        ledger, record, region = self.setup_session(bound=False)
        output = validate_raw_response(self.raw(index=1), record=record, region=region, envelope=self.envelope(record, 1))
        self.assertTrue(all(type(i) is int for i in output.object_ids))
        # Fixture selects the requested synthetic ID; real selection still needs
        # independent B0/YOLO support association outside this parser.
        bound = ledger.bind_seed(record.record_id, [output.object_ids[1]])
        self.assertEqual(bound.obj_id, 2)
        for dtype in (np.float32, np.float64, bool):
            raw = self.raw(index=1); raw["outputs"]["out_obj_ids"] = raw["outputs"]["out_obj_ids"].astype(dtype)
            with self.assertRaises(ResponseSchemaError): validate_raw_response(raw, record=record, region=region, envelope=self.envelope(record, 1))

    def test_none_is_nonframe_missing_and_cannot_acknowledge(self):
        ledger, record, region = self.setup_session()
        collector = ledger.begin_propagation(record.record_id)
        records, register = self.registry()
        parsed = ingest_response(collector, {"frame_index": 0, "outputs": None},
                                 envelope=self.envelope(collector.record), register_mask=register)
        self.assertIsNone(parsed.frame_response); self.assertFalse(parsed.acknowledged)
        self.assertEqual(parsed.status, "missing_response"); self.assertFalse(records)
        for index in (1, 2): ingest_response(collector, self.raw(index), envelope=self.envelope(collector.record), register_mask=register)
        snapshot = ledger.finish_propagation(record.record_id, collector)
        self.assertEqual(snapshot.frames[0].status, "missing_response")
        self.assertFalse(snapshot.complete)

    def test_point_none_is_incomplete_not_correction_success(self):
        ledger, record, region = self.setup_session()
        # Official-schema-shaped request record fixture, no SAM invocation.
        pending = replace(record, generation=1, state="ASSOCIATED", active_operation="correction:e0")
        records, register = self.registry()
        result = parse_target_response({"frame_index": np.int64(1), "outputs": None}, record=pending,
                                       region=region, envelope=self.envelope(pending, 1), register_mask=register)
        self.assertFalse(result.acknowledged); self.assertIsNone(result.frame_response)
        self.assertFalse(records)

    def test_successful_empty_or_target_missing_defaults_to_visibility_unknown(self):
        ledger, record, region = self.setup_session(); collector = ledger.begin_propagation(record.record_id)
        records, register = self.registry()
        for ids in ((), (9,)):
            parsed = parse_target_response(self.raw(ids=ids), record=collector.record, region=region,
                                           envelope=self.envelope(collector.record), register_mask=register)
            self.assertEqual(parsed.frame_response.status, "target_visibility_unknown"); self.assertTrue(parsed.acknowledged)
            self.assertFalse(parsed.available); self.assertFalse(parsed.use_as_training_negative)
            self.assertFalse(parsed.frame_response.available); self.assertFalse(parsed.frame_response.use_as_training_negative)
            self.assertFalse(parsed.public_output_contains_bound_id)
        self.assertFalse(records)
        raw = self.raw(ids=()); del raw["outputs"]["out_probs"]
        with self.assertRaises(ResponseSchemaError): parse_target_response(raw, record=collector.record, region=region,
                                                                          envelope=self.envelope(collector.record), register_mask=register)

    def test_tracked_internal_object_with_public_zero_is_unknown_not_absent(self):
        ledger, record, region = self.setup_session(); collector = ledger.begin_propagation(record.record_id)
        records, register = self.registry(); raw = self.raw(ids=())
        raw["outputs"]["frame_stats"] = {"num_obj_tracked": 1, "num_obj_dropped": 0}
        parsed = ingest_response(collector, raw, envelope=self.envelope(collector.record), register_mask=register)
        self.assertEqual(parsed.status, "target_visibility_unknown"); self.assertTrue(parsed.acknowledged)
        self.assertFalse(parsed.available); self.assertFalse(parsed.use_as_training_negative)
        self.assertFalse(records)

    def test_public_bound_id_with_empty_mask_is_unavailable_unknown(self):
        ledger, record, region = self.setup_session(); collector = ledger.begin_propagation(record.record_id)
        records, register = self.registry(); raw = self.raw()
        raw["outputs"]["out_binary_masks"][1] = False
        parsed = ingest_response(collector, raw, envelope=self.envelope(collector.record), register_mask=register)
        self.assertEqual(parsed.status, "target_visibility_unknown")
        self.assertEqual(parsed.reason, "native_bound_id_present_but_empty_mask")
        self.assertTrue(parsed.acknowledged); self.assertTrue(parsed.public_output_contains_bound_id)
        self.assertIs(parsed.public_mask_nonempty, False)
        self.assertFalse(parsed.available); self.assertFalse(parsed.use_as_training_negative)
        self.assertIsNone(parsed.frame_response.mask_identity); self.assertFalse(records)
        snapshot = ledger.finish_propagation(record.record_id, collector)
        self.assertEqual(snapshot.mask_available_count, 0)
        self.assertFalse(snapshot.frames[0].available)

    def test_explicit_absence_decision_must_match_version_frame_and_object(self):
        ledger, record, region = self.setup_session(); collector = ledger.begin_propagation(record.record_id)
        record = collector.record; records, register = self.registry()
        decision = AbsenceDecision("fixture-decision", "independent-fixture-source", "fixture-v1", record.track_id,
            record.region_identity, record.session_id, record.generation, record.attempt, 0, region.frame_ids[0], record.obj_id)
        parsed = parse_target_response(self.raw(ids=()), record=record, region=region, envelope=self.envelope(record),
                                       register_mask=register, absence_decision=decision)
        self.assertEqual(parsed.status, "target_absent"); self.assertFalse(parsed.available); self.assertFalse(parsed.use_as_training_negative)
        for changed in (replace(decision, obj_id=9), replace(decision, local_index=1), replace(decision, global_frame_id=999), replace(decision, source_identity="different", session_id="other")):
            with self.assertRaises(ResponseIdentityError): parse_target_response(self.raw(ids=()), record=record, region=region,
                envelope=self.envelope(record), register_mask=register, absence_decision=changed)
        with self.assertRaises(ValueError): replace(decision, schema_version="unversioned")
        with self.assertRaises(ResponseIdentityError): parse_target_response(self.raw(), record=record, region=region,
            envelope=self.envelope(record), register_mask=register, absence_decision=decision)

    def test_mask_order_matches_object_id_and_identity_binds_actual_pixels(self):
        ledger, record, region = self.setup_session(); collector = ledger.begin_propagation(record.record_id)
        records, register = self.registry(); raw = self.raw()
        parsed = ingest_response(collector, raw, envelope=self.envelope(collector.record), register_mask=register)
        selected = records[parsed.registration.mask_identity]
        np.testing.assert_array_equal(selected["mask"], raw["outputs"]["out_binary_masks"][1])
        self.assertFalse(np.array_equal(selected["mask"], raw["outputs"]["out_binary_masks"][0]))
        with self.assertRaises(ValueError): selected["request"].mask.flags.writeable = True
        # Reordering raw object rows together preserves the exact target binding.
        swapped = self.raw()
        for name in ("out_obj_ids", "out_probs", "out_boxes_xywh", "out_binary_masks"):
            swapped["outputs"][name] = swapped["outputs"][name][::-1].copy()
        same = parse_target_response(swapped, record=collector.record, region=region,
                                     envelope=self.envelope(collector.record, response_id="second-arrival"), register_mask=register)
        self.assertEqual(parsed.registration.mask_identity, same.registration.mask_identity)
        changed = self.raw(); changed["outputs"]["out_binary_masks"][1, 2, 3] = True
        different = parse_target_response(changed, record=collector.record, region=region,
                                          envelope=self.envelope(collector.record), register_mask=register)
        self.assertNotEqual(parsed.registration.mask_identity, different.registration.mask_identity)

    def test_registration_mismatch_poisons_collector(self):
        ledger, record, region = self.setup_session(); collector = ledger.begin_propagation(record.record_id)
        def false_receipt(req): return MaskRegistrationReceipt("invented", req.pixel_sha256, req.binding_identity, "not-registered")
        with self.assertRaises(ResponseSchemaError): ingest_response(collector, self.raw(), envelope=self.envelope(collector.record), register_mask=false_receipt)
        with self.assertRaises(ValueError): collector.seal()

    def test_bad_shapes_lengths_nonfinite_and_duplicates_reject(self):
        ledger, record, region = self.setup_session(); collector = ledger.begin_propagation(record.record_id)
        cases = [("out_obj_ids", np.array([2, 2], np.int64)),
                 ("out_obj_ids", np.array([-1, 2], np.int64)),
                 ("out_probs", np.array([np.nan, .5], np.float32)),
                 ("out_probs", np.array([.5], np.float32)),
                 ("out_boxes_xywh", np.full((2, 4), np.nan, np.float32)),
                 ("out_boxes_xywh", np.full((2, 4), .9, np.float32)),
                 ("out_binary_masks", np.zeros((2, 4, 3), bool)),
                 ("out_binary_masks", np.full((2, 3, 4), np.nan)),
                 ("out_binary_masks", np.zeros((1, 3, 4), bool))]
        for key, value in cases:
            raw = self.raw(); raw["outputs"][key] = value
            with self.assertRaises(ResponseSchemaError): validate_raw_response(raw, record=collector.record, region=region, envelope=self.envelope(collector.record))

    def test_frame_session_region_and_operation_identity_reject(self):
        ledger, record, region = self.setup_session(bound=False)
        raw = self.raw(1); env = self.envelope(record, 1)
        for changed in (replace(env, session_id="other"), replace(env, region_identity="other"),
                        replace(env, generation=1), replace(env, attempt=1), replace(env, expected_local_index=0)):
            with self.assertRaises(ResponseIdentityError): validate_raw_response(raw, record=record, region=region, envelope=changed)
        for index in (True, 1.0, -1):
            with self.assertRaises(ResponseSchemaError): validate_raw_response(dict(raw, frame_index=index), record=record, region=region, envelope=env)
        with self.assertRaises(ResponseIdentityError): validate_raw_response(dict(raw, frame_index=3), record=record, region=region, envelope=env)
        with self.assertRaises(ResponseIdentityError): validate_raw_response(dict(raw, session_id="other"), record=record, region=region, envelope=env)
        with self.assertRaises(ResponseSchemaError): validate_raw_response(dict(raw, generation=False), record=record, region=region, envelope=env)

    def test_schema_bridge_failure_cannot_be_ignored_then_publish(self):
        ledger, record, region = self.setup_session(); collector = ledger.begin_propagation(record.record_id)
        records, register = self.registry(); bad = self.raw(); bad["outputs"]["out_obj_ids"] = np.array([9., 2.])
        with self.assertRaises(ResponseSchemaError): ingest_response(collector, bad, envelope=self.envelope(collector.record), register_mask=register)
        for i in range(3): ingest_response(collector, self.raw(i), envelope=self.envelope(collector.record), register_mask=register)
        with self.assertRaises(ValueError): ledger.finish_propagation(record.record_id, collector)
        ledger.fail(record.record_id, status="inference_error", failure_kind="schema")
        self.assertFalse(ledger.get(record.record_id).published)

    def test_complete_bridge_to_ledger_ready_with_unavailable_unknown_slot(self):
        ledger, record, region = self.setup_session(); collector = ledger.begin_propagation(record.record_id)
        records, register = self.registry()
        for index in (2, 0, 1):
            parsed = ingest_response(collector, self.raw(index, ids=() if index == 1 else (9, 2)),
                                     envelope=self.envelope(collector.record), register_mask=register)
            self.assertTrue(parsed.acknowledged)
        snapshot = ledger.finish_propagation(record.record_id, collector)
        ledger.publish(record.record_id, identity_confirmed=True)
        self.assertEqual([f.global_frame_id for f in snapshot.frames], [100, 101, 102])
        self.assertEqual(snapshot.frames[1].status, "target_visibility_unknown")
        self.assertTrue(snapshot.responses_complete)
        self.assertEqual(snapshot.mask_available_count, 2); self.assertAlmostEqual(snapshot.mask_available_coverage, 2/3)
        self.assertEqual(len(records), 2)
        self.assertEqual(ledger.current_publication(record.region_identity, record.track_id).identity, snapshot.identity)


if __name__ == "__main__": unittest.main()
