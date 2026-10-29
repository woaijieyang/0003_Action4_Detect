import unittest
from dataclasses import replace, FrozenInstanceError
from model.action4_sam_session_contracts import (
    RegionSpec, SeedBinding, PromptEvent, lock_replay_plan, FrameResponse, AbsenceDecision,
    RoundCollector, SessionLedger, BudgetExceeded, replace_generation_snapshot)


class SamSessionTests(unittest.TestCase):
    def region(self, name="dev", count=40):
        return RegionSpec(name, tuple(range(820, 820 + count)), tuple(i * 400 for i in range(count)),
                          (1, 11988), "frames-sha-" + name, "native-rgb-identity", 100, 100)

    def seed(self, region, track="camera-a"):
        return SeedBinding(track, region.identity, "candidate-" + track, 1, region.frame_ids[1],
                           (.1, .1, .2, .2), ("b0-frozen-evidence-" + track,), "frozen_b0")

    def planned(self, ledger, region=None, track="camera-a", worker="w0"):
        region = self.region() if region is None else region
        return ledger.plan(region, self.seed(region, track), worker_id=worker,
                           device_identity="cpu-contract-fixture", dependency_identity="frozen-code-weight-pixel-identity")

    def seeded(self, ledger, record=None, obj=4, sid="session-a"):
        record = self.planned(ledger) if record is None else record
        ledger.begin_create(record.record_id); ledger.created(record.record_id, sid)
        ledger.seed_request(record.record_id)
        return ledger.bind_seed(record.record_id, [obj])

    def event(self, region, seed, index=5, event_id="e0", segment="bad0"):
        native = ((15., 15.), (20., 20.), (80., 80.), (90., 90.))
        return PromptEvent(event_id, seed.track_id, region.identity, seed.fingerprint, index,
                           region.frame_ids[index], native, tuple((x / 100, y / 100) for x, y in native),
                           (1, 1, 0, 0), ("b0-independent-support",), "merged", segment, "frozen_b0")

    def respond(self, collector, index, status="target_present", mask=None, **kwargs):
        r = collector.record
        if status == "target_absent" and "absence_decision" not in kwargs:
            kwargs["absence_decision"] = AbsenceDecision("fixture-absence-" + str(index), "synthetic-explicit-decision-source",
                "fixture-rule-v1", r.track_id, r.region_identity, r.session_id, r.generation, r.attempt,
                index, collector.region.frame(index)[0], r.obj_id)
        if status == "target_visibility_unknown": kwargs.setdefault("reason", "native_confirmation_or_visibility_unknown")
        return FrameResponse(r.region_identity, r.session_id, r.generation, index, status,
                             r.obj_id if status in {"target_present", "target_absent", "target_visibility_unknown"} else None,
                             mask or f"mask-g{r.generation}-o{r.obj_id}-f{index}" if status == "target_present" else None, **kwargs)

    def complete(self, ledger, record, *, absent=(), omit=(), unknown=()):
        collector = ledger.begin_propagation(record.record_id)
        # Both-direction streams can arrive in reverse time order.
        for i in reversed(range(len(collector.region.frame_ids))):
            if i not in omit:
                collector.add(self.respond(collector, i, "target_absent" if i in absent else "target_visibility_unknown" if i in unknown else "target_present"))
        return ledger.finish_propagation(record.record_id, collector)

    def gen1(self, ledger):
        region = self.region(); seed = self.seed(region)
        old = self.seeded(ledger, self.planned(ledger, region))
        snapshot = self.complete(ledger, old)
        ledger.publish(old.record_id, identity_confirmed=True); ledger.close(old.record_id, released=True)
        plan = lock_replay_plan(region, [seed], [self.event(region, seed)])
        new = ledger.replacement(old.record_id, plan)
        new = self.seeded(ledger, new, obj=77, sid="session-new")
        return old, snapshot, new, plan

    def test_region_and_seed_local_global_mapping(self):
        region = self.region()
        self.assertEqual(region.frame(0), (820, 0))
        seed = self.seed(region); seed.validate_region(region)
        with self.assertRaises(ValueError): replace(seed, global_frame_id=1).validate_region(region)
        with self.assertRaises(ValueError): seed.validate_region(self.region("locked"))
        with self.assertRaises(ValueError): replace(region, pts=(0,) * 40)

    def test_two_cameras_use_separate_sessions_and_binding(self):
        ledger = SessionLedger()
        a = self.seeded(ledger, self.planned(ledger, track="a"), obj=7, sid="sid-a")
        b = self.seeded(ledger, self.planned(ledger, track="b"), obj=7, sid="sid-b")
        self.assertEqual(ledger.active_count, 2)
        self.assertNotEqual(a.session_id, b.session_id)
        # SAM object numbers may coincide across different sessions.
        self.assertEqual(a.obj_id, b.obj_id)
        sa, sb = self.complete(ledger, a), self.complete(ledger, b)
        self.assertNotEqual(sa.identity, sb.identity)
        self.assertEqual((sa.track_id, sb.track_id), ("a", "b"))

    def test_worker_global_budget_queue_and_release_acknowledgement(self):
        ledger = SessionLedger()
        records = [self.planned(ledger, track=str(i), worker="w0" if i < 3 else "w1") for i in range(5)]
        for index in (0, 1, 3, 4): ledger.begin_create(records[index].record_id)
        with self.assertRaises(BudgetExceeded): ledger.begin_create(records[2].record_id)
        self.assertEqual(ledger.get(records[2].record_id).state, "PLANNED")
        ledger.fail(records[0].record_id, status="session_lost", failure_kind="worker")
        ledger.close(records[0].record_id, released=False)
        self.assertEqual(ledger.active_count, 4)
        ledger.close(records[0].record_id, released=True)
        self.assertEqual(ledger.active_count, 3)
        ledger.begin_create(records[2].record_id)
        self.assertEqual(ledger.active_count, 4)
        self.assertTrue(any(event[0] == "queued_budget" for event in ledger.audit_log))

    def test_regional_track_budget_persists_after_close(self):
        ledger = SessionLedger()
        for i in range(8):
            record = self.planned(ledger, track=str(i)); ledger.close(record.record_id, released=True)
        with self.assertRaises(BudgetExceeded): self.planned(ledger, track="ninth")
        self.assertEqual(ledger.active_count, 0)

    def test_serial_operations(self):
        ledger = SessionLedger(); record = self.seeded(ledger)
        ledger.begin_propagation(record.record_id)
        with self.assertRaises(ValueError): ledger.begin_propagation(record.record_id)
        with self.assertRaises(ValueError): ledger.seed_request(record.record_id)
        with self.assertRaises(ValueError): ledger.close(record.record_id, released=True)
        ledger.fail(record.record_id, status="session_lost", failure_kind="worker")
        ledger.close(record.record_id, released=True)
        self.assertEqual(ledger.active_count, 0)

    def test_missing_response_is_not_absent_or_prompt_fallback(self):
        ledger = SessionLedger(); record = self.seeded(ledger)
        collector = ledger.begin_propagation(record.record_id)
        collector.add(self.respond(collector, 0), source="prompt")
        collector.add(self.respond(collector, 1, "target_absent"))
        snapshot = ledger.finish_propagation(record.record_id, collector)
        self.assertEqual(snapshot.frames[0].status, "missing_response")
        self.assertIsNone(snapshot.frames[0].mask_identity)
        self.assertEqual(snapshot.frames[1].status, "target_absent")
        self.assertEqual(len(snapshot.prompt_diagnostics), 1)
        self.assertFalse(snapshot.complete)
        self.assertEqual(ledger.get(record.record_id).state, "FAILED")
        with self.assertRaises(ValueError): ledger.publish(record.record_id, identity_confirmed=True)

    def test_prompt_seed_output_is_diagnostic_propagation_wins(self):
        ledger = SessionLedger(); record = self.seeded(ledger)
        collector = ledger.begin_propagation(record.record_id)
        collector.add(self.respond(collector, 1, mask="instant-mask"), source="prompt")
        for i in range(40): collector.add(self.respond(collector, i, mask="propagated-mask"))
        snapshot = ledger.finish_propagation(record.record_id, collector)
        self.assertEqual(snapshot.frames[1].mask_identity, "propagated-mask")
        self.assertEqual(snapshot.prompt_diagnostics[0].mask_identity, "instant-mask")
        self.assertTrue(snapshot.complete)

    def test_explicit_absent_whole_region_is_complete_not_missing(self):
        ledger = SessionLedger(); record = self.seeded(ledger)
        snapshot = self.complete(ledger, record, absent=range(40))
        self.assertTrue(snapshot.complete)
        self.assertTrue(all(f.status == "target_absent" and f.mask_identity is None for f in snapshot.frames))
        ledger.publish(record.record_id, identity_confirmed=True)
        self.assertEqual(ledger.get(record.record_id).state, "READY")

    def test_unknown_responses_complete_without_mask_or_negative_label_or_retry(self):
        ledger = SessionLedger(); record = self.seeded(ledger)
        snapshot = self.complete(ledger, record, unknown=range(40))
        self.assertTrue(snapshot.responses_complete); self.assertTrue(snapshot.complete)
        self.assertEqual(snapshot.mask_available_count, 0); self.assertEqual(snapshot.mask_available_coverage, 0.)
        self.assertTrue(all(not f.available and not f.use_as_training_negative and f.mask_identity is None for f in snapshot.frames))
        current = ledger.get(record.record_id)
        self.assertEqual(current.state, "ASSOCIATED"); self.assertIsNone(current.failure_kind); self.assertIsNone(current.termination)
        ledger.publish(record.record_id, identity_confirmed=True); ledger.close(record.record_id, released=True)
        with self.assertRaises(ValueError): ledger.retry(record.record_id)

    def test_absent_requires_versioned_exact_frame_object_decision(self):
        ledger = SessionLedger(); record = self.seeded(ledger)
        collector = ledger.begin_propagation(record.record_id)
        with self.assertRaises(ValueError): FrameResponse(record.region_identity, record.session_id, 0, 0, "target_absent", record.obj_id)
        valid = self.respond(collector, 0, "target_absent")
        self.assertFalse(valid.available); self.assertFalse(valid.use_as_training_negative)
        bad = replace(valid, absence_decision=replace(valid.absence_decision, global_frame_id=999))
        with self.assertRaises(ValueError): collector.add(bad)
        with self.assertRaises(ValueError): collector.seal()

    def test_duplicate_identical_is_idempotent_conflict_poisoned(self):
        ledger = SessionLedger(); record = self.seeded(ledger)
        collector = ledger.begin_propagation(record.record_id)
        response = self.respond(collector, 0)
        collector.add(response); collector.add(response)
        with self.assertRaises(ValueError): collector.add(replace(response, mask_identity="different"))
        with self.assertRaises(ValueError): collector.seal()

    def test_cross_region_session_object_generation_rejected(self):
        ledger = SessionLedger(); record = self.seeded(ledger)
        collector = ledger.begin_propagation(record.record_id)
        response = self.respond(collector, 0)
        for fields in ({"region_identity": "other"}, {"session_id": "other"}, {"generation": 1}, {"obj_id": 999}, {"local_index": 40}):
            with self.assertRaises(ValueError): collector.add(replace(response, **fields))

    def test_caught_bad_response_poisons_round_even_after_all_valid_frames(self):
        for fault in ("region", "session", "generation", "object", "frame", "type", "source"):
            ledger = SessionLedger(); record = self.seeded(ledger)
            collector = ledger.begin_propagation(record.record_id)
            response = self.respond(collector, 0)
            changes = {"region": {"region_identity": "other"}, "session": {"session_id": "other"},
                       "generation": {"generation": 1}, "object": {"obj_id": 999}, "frame": {"local_index": 40}}
            bad = replace(response, **changes[fault]) if fault in changes else {} if fault == "type" else response
            with self.assertRaises((ValueError, TypeError)):
                collector.add(bad, source="invalid" if fault == "source" else "propagation")
            for i in range(40): collector.add(self.respond(collector, i))
            with self.assertRaises(ValueError): ledger.finish_propagation(record.record_id, collector)
            self.assertNotEqual(ledger.get(record.record_id).state, "READY")

    def test_gen1_rebinds_obj_and_replays_fixed_points(self):
        ledger = SessionLedger(); old, previous, new, plan = self.gen1(ledger)
        self.assertEqual(new.obj_id, 77); self.assertNotEqual(new.obj_id, old.obj_id)
        self.complete(ledger, new)
        request = ledger.correction_request(new.record_id, plan.events[0].event_id)
        self.assertEqual(request["obj_id"], 77)
        self.assertEqual(request["session_id"], "session-new")
        self.assertNotIn("text", request); self.assertNotIn("bounding_boxes", request)
        ledger.correction_applied(new.record_id, plan.events[0].event_id)
        revised = self.complete(ledger, new, absent=(0, 39))
        ledger.publish(new.record_id, identity_confirmed=True)
        result = ledger.publish_replacement(old.record_id, new.record_id)
        self.assertIs(result, revised)
        self.assertEqual(result.frames[0].status, "target_absent"); self.assertIsNone(result.frames[0].mask_identity)
        self.assertEqual(result.obj_id, 77)
        self.assertEqual([f.global_frame_id for f in result.frames], list(range(820, 860)))
        self.assertEqual(ledger.snapshot(new.record_id).identity, revised.identity)
        self.assertEqual(ledger.current_publication(new.region_identity, new.track_id).identity, revised.identity)

    def test_ready_failure_revokes_publication_and_preserves_history(self):
        ledger = SessionLedger(); record = self.seeded(ledger)
        snapshot = self.complete(ledger, record); ledger.publish(record.record_id, identity_confirmed=True)
        self.assertEqual(ledger.current_publication(record.region_identity, record.track_id).identity, snapshot.identity)
        failed = ledger.fail(record.record_id, status="ambiguous_identity", failure_kind="identity")
        self.assertFalse(failed.published)
        self.assertEqual(ledger.snapshot(record.record_id).identity, snapshot.identity)
        with self.assertRaises(KeyError): ledger.current_publication(record.region_identity, record.track_id)

    def test_closed_published_snapshot_can_be_invalidated_without_reopening_resources(self):
        ledger = SessionLedger(); record = self.seeded(ledger)
        snapshot = self.complete(ledger, record); ledger.publish(record.record_id, identity_confirmed=True)
        ledger.close(record.record_id, released=True)
        failed = ledger.fail(record.record_id, status="ambiguous_identity", failure_kind="identity")
        self.assertFalse(failed.published); self.assertEqual(ledger.active_count, 0)
        self.assertEqual(ledger.snapshot(record.record_id).identity, snapshot.identity)
        with self.assertRaises(KeyError): ledger.current_publication(record.region_identity, record.track_id)

    def test_complete_unconfirmed_gen1_cannot_formally_replace_gen0(self):
        ledger = SessionLedger(); old, previous, new, plan = self.gen1(ledger)
        self.complete(ledger, new)
        ledger.correction_request(new.record_id, "e0"); ledger.correction_applied(new.record_id, "e0")
        revised = self.complete(ledger, new)
        self.assertTrue(revised.complete)
        with self.assertRaises(ValueError): ledger.publish_replacement(old.record_id, new.record_id)
        self.assertEqual(ledger.current_publication(old.region_identity, old.track_id).identity, previous.identity)
        ledger.publish(new.record_id, identity_confirmed=True)
        ledger.publish_replacement(old.record_id, new.record_id)
        ledger.fail(new.record_id, status="ambiguous_identity", failure_kind="identity")
        self.assertFalse(ledger.get(new.record_id).published)
        with self.assertRaises(KeyError): ledger.current_publication(old.region_identity, old.track_id)
        self.assertEqual(ledger.snapshot(old.record_id).identity, previous.identity)
        self.assertEqual(ledger.snapshot(new.record_id).identity, revised.identity)

    def test_gen1_missing_responses_cannot_merge_old_masks(self):
        ledger = SessionLedger(); old, previous, new, plan = self.gen1(ledger)
        initial = self.complete(ledger, new)
        with self.assertRaises(ValueError): replace_generation_snapshot(previous, initial)
        ledger.correction_request(new.record_id, "e0"); ledger.correction_applied(new.record_id, "e0")
        revised = self.complete(ledger, new, omit=(7,))
        self.assertEqual(revised.frames[7].status, "missing_response")
        self.assertIsNone(revised.frames[7].mask_identity)
        self.assertIsNotNone(previous.frames[7].mask_identity)
        with self.assertRaises(ValueError): replace_generation_snapshot(previous, revised)
        self.assertEqual(ledger.get(new.record_id).state, "FAILED")

    def test_gen1_unknown_slots_replace_and_erase_old_masks(self):
        ledger = SessionLedger(); old, previous, new, plan = self.gen1(ledger)
        self.complete(ledger, new)
        ledger.correction_request(new.record_id, "e0"); ledger.correction_applied(new.record_id, "e0")
        revised = self.complete(ledger, new, unknown=range(40))
        self.assertTrue(revised.responses_complete); self.assertEqual(revised.mask_available_count, 0)
        ledger.publish(new.record_id, identity_confirmed=True)
        published = ledger.publish_replacement(old.record_id, new.record_id)
        self.assertEqual(published.identity, revised.identity)
        self.assertTrue(all(f.status == "target_visibility_unknown" and f.mask_identity is None and not f.available for f in published.frames))
        self.assertTrue(all(f.mask_identity is not None for f in previous.frames))
        self.assertEqual(ledger.snapshot(old.record_id).identity, previous.identity)

    def test_restore_sealed_unknown_baseline_is_not_backend_memory_resume(self):
        source = SessionLedger(); record = self.seeded(source)
        snapshot = self.complete(source, record, unknown=range(40))
        restored_ledger = SessionLedger(); region = self.region(); seed = self.seed(region)
        restored = restored_ledger.restore_closed_published_baseline(snapshot, region, seed,
            expected_snapshot_identity=snapshot.identity, source_receipt_identity="verified-raw-fixture-receipt",
            worker_id="new-controller", device_identity="cpu-fixture")
        self.assertEqual(restored.state, "CLOSED"); self.assertEqual(restored_ledger.active_count, 0)
        self.assertEqual(restored_ledger.current_publication(region.identity, seed.track_id).identity, snapshot.identity)
        self.assertTrue(any(e[0] == "restored_closed_baseline" for e in restored_ledger.audit_log))
        self.assertFalse(any(e[0] == "creating" for e in restored_ledger.audit_log))
        with self.assertRaises(ValueError): SessionLedger().restore_closed_published_baseline(
            replace(snapshot, mask_available_count=1), region, seed,
            expected_snapshot_identity=replace(snapshot, mask_available_count=1).identity,
            source_receipt_identity="fixture", worker_id="w", device_identity="cpu")
        bad_ledger = SessionLedger(); bad = replace(snapshot, prompt_log_identity="wrong")
        with self.assertRaises(ValueError): bad_ledger.restore_closed_published_baseline(bad, region, seed,
            expected_snapshot_identity=bad.identity, source_receipt_identity="fixture", worker_id="w", device_identity="cpu")
        self.assertEqual(bad_ledger.audit_log, ())

    def test_gen1_cannot_publish_before_locked_complete_replay(self):
        ledger = SessionLedger(); old, previous, new, plan = self.gen1(ledger)
        with self.assertRaises(ValueError): ledger.correction_request(new.record_id, "e0")
        self.complete(ledger, new)
        with self.assertRaises(ValueError): ledger.publish(new.record_id, identity_confirmed=True)
        with self.assertRaises(ValueError): ledger.correction_request(new.record_id, "not-in-plan")
        with self.assertRaises(ValueError): ledger.begin_propagation(new.record_id)

    def test_retry_is_bounded_full_replay_and_new_obj_binding(self):
        ledger = SessionLedger(); record = self.seeded(ledger)
        ledger.fail(record.record_id, status="inference_error", failure_kind="transient")
        with self.assertRaises(ValueError): ledger.retry(record.record_id)
        ledger.close(record.record_id, released=True)
        retry = ledger.retry(record.record_id, worker_id="replacement-worker")
        self.assertEqual(retry.attempt, 1); self.assertIsNone(retry.obj_id)
        self.assertEqual(retry.prompt_log_identity, record.prompt_log_identity)
        retry = self.seeded(ledger, retry, obj=90, sid="retry-session")
        ledger.fail(retry.record_id, status="session_lost", failure_kind="worker")
        ledger.close(retry.record_id, released=True)
        with self.assertRaises(ValueError): ledger.retry(retry.record_id)
        self.assertEqual(ledger.get(record.record_id).failure_kind, "transient")

    def test_oom_never_becomes_absent_and_reduced_budget_is_enforced(self):
        ledger = SessionLedger()
        a = self.seeded(ledger, self.planned(ledger, track="a"), sid="a")
        b = self.seeded(ledger, self.planned(ledger, track="b"), sid="b")
        collector = ledger.begin_propagation(b.record_id)
        collector.add(self.respond(collector, 0, "inference_error", reason="CUDA OOM", failure_kind="oom"))
        snapshot = ledger.finish_propagation(b.record_id, collector)
        self.assertEqual(snapshot.frames[0].status, "inference_error")
        self.assertEqual(ledger.get(b.record_id).failure_kind, "oom")
        ledger.close(b.record_id, released=True)
        with self.assertRaises(ValueError): ledger.retry(b.record_id, oom_concurrency_limit=2)
        retry = ledger.retry(b.record_id, oom_concurrency_limit=1)
        with self.assertRaises(BudgetExceeded): ledger.begin_create(retry.record_id)
        ledger.close(a.record_id, released=True)
        ledger.begin_create(retry.record_id)
        another = self.planned(ledger, track="c", worker="w1")
        with self.assertRaises(BudgetExceeded): ledger.begin_create(another.record_id)

    def test_schema_identity_not_retryable_and_single_oom_cannot_reduce(self):
        for status, kind in (("ambiguous_identity", "identity"), ("inference_error", "schema"), ("inference_error", "oom")):
            ledger = SessionLedger(); record = self.seeded(ledger)
            ledger.fail(record.record_id, status=status, failure_kind=kind); ledger.close(record.record_id, released=True)
            with self.assertRaises(ValueError): ledger.retry(record.record_id, **({"oom_concurrency_limit": 1} if kind == "oom" else {}))

    def test_missing_earlier_frame_cannot_downgrade_later_oom_or_schema(self):
        for kind in ("oom", "schema"):
            ledger = SessionLedger(); record = self.seeded(ledger)
            collector = ledger.begin_propagation(record.record_id)
            collector.add(self.respond(collector, 5, "inference_error", reason=kind, failure_kind=kind))
            snapshot = ledger.finish_propagation(record.record_id, collector)
            self.assertEqual(snapshot.frames[0].status, "missing_response")
            self.assertEqual(ledger.get(record.record_id).termination, "inference_error")
            self.assertEqual(ledger.get(record.record_id).failure_kind, kind)

    def test_no_unique_seed_binding_not_formal_target(self):
        for matches in ([], [2, 3]):
            ledger = SessionLedger(); record = self.planned(ledger)
            ledger.begin_create(record.record_id); ledger.created(record.record_id, "sid"); ledger.seed_request(record.record_id)
            failed = ledger.bind_seed(record.record_id, matches)
            self.assertEqual(failed.termination, "ambiguous_identity"); self.assertIsNone(failed.obj_id)
            with self.assertRaises(ValueError): ledger.begin_propagation(record.record_id)

    def test_replay_log_budget_spacing_order_and_coordinates(self):
        region = self.region(); seed = self.seed(region)
        a = self.event(region, seed, 5); b = self.event(region, seed, 20, "e1", "bad1")
        plan = lock_replay_plan(region, [seed], [a, b])
        self.assertEqual(plan.events, (a, b))
        with self.assertRaises(FrozenInstanceError): plan.events = ()
        for events in ([b, a], [a, replace(b, error_segment_id=a.error_segment_id)],
                       [a, self.event(region, seed, 10, "e1", "bad1")],
                       [replace(a, points_xy01=((0., 0.),) * 4)],
                       [replace(a, region_identity="locked-region")]):
            with self.assertRaises(ValueError): lock_replay_plan(region, [seed], events)
        with self.assertRaises(ValueError): replace(a, labels=(1, 1, 1, 1))
        with self.assertRaises(ValueError): replace(a, evidence_source="GT")
        with self.assertRaises(ValueError): replace(a, evidence_source="A0_pose_projection")

    def test_replacement_requires_ready_gen0_and_same_region(self):
        ledger = SessionLedger(); record = self.seeded(ledger)
        self.complete(ledger, record); ledger.close(record.record_id, released=True)
        region = self.region(); plan = lock_replay_plan(region, [self.seed(region)], [self.event(region, self.seed(region))])
        with self.assertRaises(ValueError): ledger.replacement(record.record_id, plan)

    def test_snapshot_identity_binds_generation_dependencies_and_log(self):
        ledger = SessionLedger(); record = self.seeded(ledger)
        snapshot = self.complete(ledger, record)
        for fields in ({"generation": 1}, {"dependency_identity": "different"}, {"prompt_log_identity": "different"}, {"obj_id": 88}):
            self.assertNotEqual(snapshot.identity, replace(snapshot, **fields).identity)
        with self.assertRaises(FrozenInstanceError): snapshot.frames = ()


if __name__ == "__main__": unittest.main()
