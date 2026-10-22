"""Dispatcher control-plane checks; these do not claim GPU/model quality."""
import copy
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

from model.training import action4_dispatch as dispatch

ROOT = Path(__file__).resolve().parents[1]
UUID = dispatch.ALLOWED_GPU_UUIDS[1]


class DispatcherTests(unittest.TestCase):
    def setUp(self):
        cases = ROOT / "outputs/action4_u06_validation_v001/test_cases"
        cases.mkdir(parents=True, exist_ok=True)
        self.base = Path(tempfile.mkdtemp(prefix="case_", dir=cases))
        self.queue = dispatch.Dispatcher(ROOT, self.base / "queue")
        self.config = self.base / "fixture_config.json"
        self.config.write_text('{"fixture": true}\n')
        self.spec = dict(job_id="cpu_test", kind="cpu", cwd=".",
            argv=[".venv/bin/python", "scripts/action4_dispatch.py", "--help"],
            output_dir=str(self.base / "attempt"), allowed_gpu_uuids=[], reservation_mib=0,
            cpu_threads=2, code={"scripts/action4_dispatch.py": dispatch.sha256(ROOT / "scripts/action4_dispatch.py")},
            configs={str(self.config.relative_to(ROOT)): dispatch.sha256(self.config)}, inputs={})

    def tearDown(self):
        # Keep test evidence. No directory deletion or process termination.
        pass

    def test_admission_preserves_headroom_without_double_counting(self):
        self.assertTrue(dispatch.admission(20000, 2000, 4000, 18000-dispatch.HEADROOM_MIB))
        self.assertFalse(dispatch.admission(20000, 2000, 4000, 18001-dispatch.HEADROOM_MIB))
        self.assertFalse(dispatch.admission(dispatch.HEADROOM_MIB, 0, 0, 1))

    def test_gpu3_component_allowed_under_expanded_gpu2_to7_authority(self):
        # User explicitly expanded training/work authorization to physical GPUs2-7.
        spec=dict(self.spec,kind="component",allowed_gpu_uuids=[dispatch.QUALITY_GPU_UUID],
                  reservation_mib=1000,profile_key="authorization_fixture")
        checked=dispatch.Dispatcher(ROOT)._validate(spec,profile=True)
        self.assertEqual(checked["allowed_gpu_uuids"],[dispatch.QUALITY_GPU_UUID])
    def test_quality_entry_exclusive(self):
        spec=dict(self.spec,argv=[".venv/bin/python","scripts/action4_quality.py","--config",str(self.config),"--out",str(self.base/"result")])
        with self.assertRaisesRegex(ValueError,"exclusive trusted quality"):
            self.queue.enqueue(spec)
    def test_reject_foreign_gpu(self):
        spec = dict(self.spec, allowed_gpu_uuids=["GPU-foreign"], reservation_mib=1000)
        with self.assertRaisesRegex(ValueError, "authorized"):
            self.queue.enqueue(spec)

    def test_reject_shell_and_unhashed_entry(self):
        for argv in (["bash", "-c", "echo bad"], [".venv/bin/python", "-c", "print(1)"],
                     [".venv/bin/python", "scripts/action4_dispatch.py", "--root", "/outside"]):
            with self.subTest(argv=argv), self.assertRaises((ValueError, FileNotFoundError)):
                self.queue.enqueue(dict(self.spec, argv=argv))
        spec = dict(self.spec, code={})
        with self.assertRaisesRegex(ValueError, "source SHA"):
            self.queue.enqueue(spec)

    def test_paths_resolve_symlinks_and_parent_escape(self):
        link = self.base / "outside"
        link.symlink_to("/", target_is_directory=True)
        for path in (str(link / "tmp/new"), "../outside", "/tmp/outside"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.queue.enqueue(dict(self.spec, output_dir=path))

    def test_enqueue_checks_stale_file_identity(self):
        self.config.write_text("changed")
        with self.assertRaisesRegex(ValueError, "Stale"):
            self.queue.enqueue(self.spec)

    def test_stale_queued_input_fails_before_launch(self):
        self.queue.enqueue(self.spec)
        self.config.write_text("changed")
        with mock.patch.object(dispatch.subprocess, "Popen") as launch:
            state = self.queue.run(max_seconds=.1)
        self.assertEqual(state["jobs"]["cpu_test"]["status"], "failed")
        launch.assert_not_called()

    def test_duplicate_output_and_ids_rejected(self):
        self.queue.enqueue(self.spec)
        with self.assertRaisesRegex(ValueError, "job_id"):
            self.queue.enqueue(self.spec)
        with self.assertRaisesRegex(ValueError, "Output"):
            self.queue.enqueue(dict(self.spec, job_id="other"))
        with self.assertRaisesRegex(ValueError, "Output"):
            self.queue.enqueue(dict(self.spec, job_id="nested", output_dir=self.spec["output_dir"] + "/nested"))

    def test_single_scheduler_lock_prevents_second_owner(self):
        with self.queue.lock("scheduler"):
            with self.assertRaisesRegex(RuntimeError, "Another scheduler"):
                self.queue.run(max_seconds=.01)

    def test_measured_profile_required_and_bound_to_config(self):
        spec = dict(self.spec, allowed_gpu_uuids=[UUID], reservation_mib=1000, profile_key="fixed-shape-v1", profile_id="measured")
        with self.assertRaises(FileNotFoundError):
            self.queue.enqueue(spec)
        record = dict(status="conclusive", profile_key="fixed-shape-v1", recommended_reservation_mib=1250,
                      code=spec["code"], configs=spec["configs"])
        dispatch.atomic(self.queue.directory / "profiles/measured.json", record)
        with self.assertRaisesRegex(ValueError, "below"):
            self.queue.enqueue(spec)
        spec["reservation_mib"] = 1250
        record["configs"] = {}
        dispatch.atomic(self.queue.directory / "profiles/measured.json", record)
        with self.assertRaisesRegex(ValueError, "changed"):
            self.queue.enqueue(spec)

    def test_short_profile_inconclusive(self):
        spec = dict(self.spec, profile_key="small", profile=True)
        job = dict(spec=spec, status="succeeded", launched_at=0., finished_at=1., peak_mib=100, memory_samples=5)
        self.queue._finish_profile(job)
        result = dispatch.read(self.queue.directory / "profiles/cpu_test.json")
        self.assertEqual(result["status"], "inconclusive")
        self.assertEqual(result["recommended_reservation_mib"], 125)

    def test_live_worker_adopted_missing_identity_failed_no_restart(self):
        self.queue.enqueue(self.spec)
        state = self.queue.state()
        job = state["jobs"]["cpu_test"]
        identity = {"pid": 123, "start": "fixture"}
        job.update(status="running", launched_at=time.time()-100, worker=identity)
        with mock.patch.object(dispatch, "is_live", side_effect=lambda p: p == identity):
            self.queue._refresh(state)
        self.assertEqual(job["status"], "running")
        with mock.patch.object(dispatch, "is_live", return_value=False):
            self.queue._refresh(state)
        self.assertEqual(job["status"], "failed")
        self.assertIsNone(job["exitcode"])
        self.assertIn("never auto-relaunch", job["failure"])

    def test_pid_start_mismatch_not_live(self):
        identity = {"pid": 12, "start": "old"}
        with mock.patch.object(dispatch, "process_identity", return_value={"pid": 12, "start": "new"}):
            self.assertFalse(dispatch.is_live(identity))

    def test_process_query_error_does_not_claim_process_stopped(self):
        with mock.patch.object(dispatch.subprocess, "run", return_value=mock.Mock(returncode=2, stdout="")):
            with self.assertRaisesRegex(RuntimeError, "inconclusive"):
                dispatch.process_identity(123)

    def test_live_process_observation_timeout_preserves_running(self):
        self.queue.enqueue(self.spec)
        state = self.queue.state()
        state["jobs"]["cpu_test"].update(status="running", launched_at=time.time()-100,
                                          worker={"pid": 123, "start": "fixture"})
        self.queue.save(state)
        with mock.patch.object(dispatch, "is_live", return_value=True):
            observed = self.queue.run(poll_seconds=.01, max_seconds=.01)
        self.assertEqual(observed["jobs"]["cpu_test"]["status"], "running")

    def test_exit_token_mismatch_rejected(self):
        self.queue.enqueue(self.spec)
        state = self.queue.state()
        state["jobs"]["cpu_test"].update(status="running", launched_at=time.time())
        dispatch.atomic(Path(self.spec["output_dir"]) / "dispatch_exit.json", dict(token="foreign", exitcode=0, finished_at=time.time()))
        with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
            self.queue._refresh(state)

    def test_gpu_cpu_and_disk_admission(self):
        spec = dict(self.spec, allowed_gpu_uuids=[UUID], reservation_mib=1000, profile=False)
        # User-authorized aggregate budget is72; each individual job remains<=32.
        state = {"jobs": {f"existing_{i}": dict(spec=dict(spec, cpu_threads=n),
                 status="running", device=UUID, current_mib=500) for i,n in enumerate((31,31,9))}}
        self.assertIs(self.queue._choose_device(spec, state), False)  #71+2>72
        state["jobs"]["existing_2"]["spec"]["cpu_threads"] = 8
        with mock.patch.object(dispatch, "gpu_identity", return_value={"free_mib":12000}), \
             mock.patch.object(dispatch.shutil, "disk_usage", return_value=mock.Mock(free=10**12)):
            self.assertEqual(self.queue._choose_device(spec,state),UUID)  #70+2==72
        state["jobs"] = {}
        with mock.patch.object(dispatch.shutil, "disk_usage", return_value=mock.Mock(free=dispatch.DISK_FLOOR - 1)):
            self.assertIs(self.queue._choose_device(spec, state), False)
        with mock.patch.object(dispatch, "gpu_identity", return_value={"free_mib": 12000}), \
             mock.patch.object(dispatch.shutil, "disk_usage", return_value=mock.Mock(free=10**12)):
            self.assertEqual(self.queue._choose_device(spec, state), UUID)

    def test_duplicate_root_output_and_abbreviation_are_rejected(self):
        for args in (["--root", str(ROOT), "--root", str(self.base)],
                     ["--output", str(self.base / "r"), "--output=" + str(self.base / "elsewhere")],
                     ["--roo", str(self.base)], ["--device", "cuda:1"]):
            spec = dict(self.spec, argv=self.spec["argv"][:2] + args)
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.queue.enqueue(spec)

    def test_output_flags_and_result_collisions(self):
        result = self.base / "result"
        spec = dict(self.spec, argv=self.spec["argv"][:2] + ["--output", str(result)], result_dirs=[str(result)])
        self.queue.enqueue(spec)
        other = dict(spec, job_id="other", output_dir=str(self.base / "other_logs"))
        with self.assertRaisesRegex(ValueError, "overlaps"):
            self.queue.enqueue(other)
        spec = dict(self.spec, argv=self.spec["argv"][:2] + ["--output", str(self.base / "unregistered")])
        with self.assertRaisesRegex(ValueError, "registered"):
            self.queue.enqueue(spec)

    def test_explicit_gpu_cannot_disagree_with_mask(self):
        spec = dict(self.spec, argv=self.spec["argv"][:2] + ["--gpu-uuid", dispatch.ALLOWED_GPU_UUIDS[0]],
                    allowed_gpu_uuids=[UUID], reservation_mib=1000, profile_key="fixture")
        with self.assertRaisesRegex(ValueError, "single registered"):
            self.queue.enqueue(spec, profile=True)

    def test_render_threads_cannot_exceed_reservation(self):
        spec = dict(self.spec, argv=self.spec["argv"][:2] + ["--threads", "33"])
        with self.assertRaisesRegex(ValueError, "CPU reservation"):
            self.queue.enqueue(spec)

    def test_gpu_jobs_require_one_canonical_queue(self):
        spec = dict(self.spec, allowed_gpu_uuids=[UUID], reservation_mib=1000, profile_key="fixture")
        with self.assertRaisesRegex(ValueError, "canonical"):
            self.queue.enqueue(spec, profile=True)

    def test_imported_source_is_automatically_bound(self):
        job = self.queue.enqueue(self.spec)
        self.assertIn("src/model/training/action4_dispatch.py", job["spec"]["code"])
        self.assertIn("src/data/action4_devices.py", job["spec"]["code"])

    def test_video_requires_immutable_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            self.queue.enqueue(dict(self.spec, kind="video"))

    def test_quality_foreign_root_rejected_before_any_mkdir_or_read(self):
        for foreign in ("/unread-review-sentinel",str(ROOT.parent),str(ROOT.parent/"unread-sibling")):
            with self.subTest(root=foreign), \
                 mock.patch.object(Path,"resolve",lambda path,strict=False:path), \
                 mock.patch.object(Path,"mkdir") as mkdir, \
                 mock.patch.object(Path,"is_dir",return_value=True) as is_dir:
                with self.assertRaisesRegex(ValueError,"authorized"):
                    dispatch.Dispatcher(foreign)
                mkdir.assert_not_called()
                is_dir.assert_not_called()

    def test_quality_symlink_root_escape_is_rejected_without_external_io(self):
        link = ROOT/"outputs"/"mocked_external_link"
        real_resolve = Path.resolve
        with mock.patch.object(Path,"resolve",autospec=True,side_effect=lambda p,strict=False: Path("/unread-symlink-target") if p==link else real_resolve(p,strict=strict)), \
             mock.patch.object(Path,"mkdir") as mkdir:
            with self.assertRaisesRegex(ValueError,"authorized"):
                dispatch.Dispatcher(link)
        mkdir.assert_not_called()

    def test_quality_nested_fixture_cannot_create_another_gpu_ledger(self):
        nested = self.base/"nested_cpu_fixture"
        (nested/"scripts").mkdir(parents=True)
        script = nested/"scripts/action4_dispatch.py"
        script.write_text("# inert fixture, never run\n")
        config = nested/"config.json"
        config.write_text("{}")
        queue = dispatch.Dispatcher(nested)
        spec = dict(self.spec,argv=[".venv/bin/python","scripts/action4_dispatch.py","--help"],
                    output_dir="outputs/never_run",code={"scripts/action4_dispatch.py":dispatch.sha256(script)},
                    configs={"config.json":dispatch.sha256(config)},allowed_gpu_uuids=[UUID],
                    reservation_mib=1000,profile_key="nested-fixture")
        with self.assertRaisesRegex(ValueError,"canonical|actual project"):
            queue._validate(spec,profile=True)

    def test_quality_real_training_config_cannot_underreserve_four_ranks(self):
        spec = self.group_spec(4)
        config = ROOT/"configs/model/action4_train.json"
        spec["argv"][3] = str(config)
        spec["configs"] = {str(config.relative_to(ROOT)):dispatch.sha256(config)}
        spec["cpu_threads"] = 4
        original = self.queue.directory
        self.queue.directory = ROOT/"outputs/action4_dispatch"
        try:
            with self.assertRaisesRegex(ValueError,"CPU.*24|24.*CPU"):
                self.queue._validate(spec,profile=True)
        finally:
            self.queue.directory = original

    def test_quality_low_resolution_profile_cannot_accept_native_plan(self):
        recipe = {"resolution_wh":[812,616],"asset_backend":"procedural_r3", "render":{"device":"CPU","samples":8}}
        asset = {str(self.config.relative_to(ROOT)):dispatch.sha256(self.config)}
        low, native = self.base/"low_plan.json", self.base/"native_plan.json"
        dispatch.atomic(low,{"recipe":recipe,"asset_identities":asset,"seed":1,"entries":[]})
        native_recipe = copy.deepcopy(recipe);native_recipe["resolution_wh"]=[3248,2464]
        dispatch.atomic(native,{"recipe":native_recipe,"asset_identities":asset,"seed":2,"entries":[]})
        def workload(plan):
            spec = self.render_spec()
            spec.update(allowed_gpu_uuids=[UUID],reservation_mib=1000,profile_key="same-caller-label")
            spec["argv"][4] = str(plan)
            spec["inputs"] = {str(plan.relative_to(ROOT)):dispatch.sha256(plan)}
            return spec
        original = self.queue.directory
        self.queue.directory = ROOT/"outputs/action4_dispatch"
        try:
            profiled = self.queue._validate(workload(low),profile=True)
            record = dict(status="conclusive",profile_key=profiled["profile_key"],gpu_count=1,
                          recommended_reservation_mib=1000,code=profiled["code"],configs=profiled["configs"],
                          effective_workload_sha256=profiled.get("effective_workload_sha256"))
            path = self.base/"profile_fixture.json";dispatch.atomic(path,record)
            candidate = workload(native);candidate["profile_id"]="isolated_fixture_never_enqueued"
            expected = self.queue.directory/"profiles/isolated_fixture_never_enqueued.json"
            real_bounded = dispatch.bounded
            with mock.patch.object(dispatch,"bounded",side_effect=lambda r,v,**kw: path if Path(v)==expected else real_bounded(r,v,**kw)):
                with self.assertRaisesRegex(ValueError,"effective workload"):
                    self.queue._validate(candidate)
        finally:
            self.queue.directory = original

    def test_quality_actual_merged_training_cpu_budget_and_local_override(self):
        spec = self.group_spec(4)
        config = ROOT/"configs/model/action4_train.json"
        spec["argv"][3] = str(config)
        spec["configs"] = {str(config.relative_to(ROOT)):dispatch.sha256(config)}
        spec["cpu_threads"] = 24
        original = self.queue.directory
        self.queue.directory = ROOT/"outputs/action4_dispatch"
        try:
            validated = self.queue._validate(spec,profile=True)
            self.assertEqual(validated["effective_workload"]["cpu_contract"]["minimum"],24)
            local = self.base/"local_cpu.json"
            dispatch.atomic(local,{"runtime":{"cpu_threads":1,"workers":1}})
            spec["argv"] += ["--local-config",str(local)]
            spec["argv"][5] = "A0"
            spec["configs"][str(local.relative_to(ROOT))] = dispatch.sha256(local)
            spec["cpu_threads"] = 8
            validated = self.queue._validate(spec,profile=True)
            budget = validated["effective_workload"]["cpu_contract"]
            self.assertEqual((budget["compute_per_rank"],budget["loader_workers_per_rank"],budget["minimum"]),(1,1,8))
            spec["cpu_threads"] = 7
            with self.assertRaisesRegex(ValueError,"requirement 8"):
                self.queue._validate(spec,profile=True)
        finally:
            self.queue.directory = original

    def test_quality_diagnostic_fixed_two_threads_need_real_budget(self):
        run_id = self.base.name + "_diagnostic"
        spec = dict(self.spec,kind="component",cpu_threads=1,
            argv=[".venv/bin/python","scripts/action4_diagnostic.py","--config",str(self.config),
                  "--manifest",str(self.config),"--run-id",run_id],
            code={"scripts/action4_diagnostic.py":dispatch.sha256(ROOT/"scripts/action4_diagnostic.py")},
            inputs={str(self.config.relative_to(ROOT)):dispatch.sha256(self.config)},
            result_dirs=[str(ROOT/"outputs/action4"/run_id),str(ROOT/"models/action4"/run_id)])
        with self.assertRaisesRegex(ValueError,"requirement 2"):
            self.queue._validate(spec)
        spec["cpu_threads"] = 2
        self.assertEqual(self.queue._validate(spec)["effective_workload"]["cpu_contract"]["minimum"],2)

    def test_quality_diagnostic_cpu_contract_tracks_actual_entry(self):
        import ast
        source=ast.parse((ROOT/"src/model/training/action4_diagnostic.py").read_text())
        calls=[node for node in ast.walk(source) if isinstance(node,ast.Call)
               and isinstance(node.func,ast.Attribute) and node.func.attr=="set_num_threads"]
        self.assertEqual([ast.literal_eval(call.args[0]) for call in calls],[2])

    def test_quality_video_budget_counts_actual_codec_defaults(self):
        run_id = self.base.name + "_video"
        budget = self.queue._encoding_cpu_budget()
        self.assertEqual((budget["decode_threads"],budget["encode_threads"],budget["filter_threads"]),(4,4,1))
        spec = dict(self.spec,kind="video",cpu_threads=budget["minimum"]-1,
            argv=[".venv/bin/python","scripts/action4_video.py","--config",str(self.config),
                  "--checkpoint",str(self.config),"--frame-manifest",str(self.config),"--run-id",run_id],
            code={"scripts/action4_video.py":dispatch.sha256(ROOT/"scripts/action4_video.py")},
            inputs={str(self.config.relative_to(ROOT)):dispatch.sha256(self.config)},
            result_dirs=[str(ROOT/"outputs/action4"/run_id)])
        with self.assertRaisesRegex(ValueError,"requirement 9"):
            self.queue._validate(spec)
        spec["cpu_threads"] = 9
        self.assertEqual(self.queue._validate(spec)["effective_workload"]["cpu_contract"],budget)

    def test_quality_same_recipe_seed_and_shard_changes_share_effective_profile(self):
        spec = self.render_spec()
        original = Path(spec["argv"][4]); changed = self.base/"other_seed_plan.json"
        plan = dispatch.read(original)
        plan.update(seed=999,entries=[{"sample_id":"different-shard-input"}],group_count=8)
        dispatch.atomic(changed,plan)
        first = self.queue._validate(spec)
        other = copy.deepcopy(spec)
        other["argv"][4] = str(changed)
        other["argv"] += ["--shard-index","1","--num-shards","4","--entry-start","3","--entry-count","8"]
        other["inputs"] = {str(changed.relative_to(ROOT)):dispatch.sha256(changed)}
        second = self.queue._validate(other)
        self.assertEqual(first["effective_workload_sha256"],second["effective_workload_sha256"])

    def test_quality_render_backend_settings_and_asset_changes_invalidate_profile(self):
        spec = self.render_spec()
        base_plan = dispatch.read(Path(spec["argv"][4]))
        baseline = self.queue._validate(spec)["effective_workload_sha256"]
        alternate_asset = self.base/"other_asset.json";alternate_asset.write_text('{"different_geometry": true}')
        variants = []
        sample = copy.deepcopy(base_plan);sample["recipe"]["render"]["samples"] = 64;variants.append(sample)
        backend = copy.deepcopy(base_plan);backend["recipe"]["asset_backend"] = "blend_r4";variants.append(backend)
        asset = copy.deepcopy(base_plan);asset["asset_identities"] = {str(alternate_asset.relative_to(ROOT)):dispatch.sha256(alternate_asset)};variants.append(asset)
        for i,plan in enumerate(variants):
            path=self.base/f"variant_{i}.json";dispatch.atomic(path,plan)
            changed=copy.deepcopy(spec);changed["argv"][4]=str(path)
            changed["inputs"]={str(path.relative_to(ROOT)):dispatch.sha256(path)}
            with self.subTest(variant=i):
                self.assertNotEqual(self.queue._validate(changed)["effective_workload_sha256"],baseline)

    def render_spec(self):
        from data.action4_synthesis import _run_output_dir
        data_output = ROOT / "data" / (self.base.name + "_never_rendered")
        derived = _run_output_dir(data_output)
        plan = self.base / "control_only_render_plan.json"
        if not plan.exists():
            dispatch.atomic(plan,{"recipe":{"resolution_wh":[812,616],"asset_backend":"procedural_r3","render":{"device":"CPU","samples":8}},
                                  "asset_identities":{str(self.config.relative_to(ROOT)):dispatch.sha256(self.config)},"seed":1,"entries":[]})
        return dict(self.spec,kind="render",cpu_threads=1,
            argv=[".venv/bin/python","scripts/action4_synthesis.py","render-shard",
                  "--plan",str(plan),"--output",str(data_output),
                  "--blender","runtime/blender-3.6.23-linux-x64/blender","--threads","1"],
            code={"scripts/action4_synthesis.py":dispatch.sha256(ROOT / "scripts/action4_synthesis.py")},
            inputs={str(plan.relative_to(ROOT)):dispatch.sha256(plan)},
            result_dirs=[str(data_output),str(derived)])

    def test_render_omitted_threads_cannot_underreserve_default(self):
        for registered in (1,2):
            spec = self.render_spec()
            spec.update(cpu_threads=registered,argv=spec["argv"][:-2])
            with self.subTest(cpu_threads=registered), self.assertRaisesRegex(ValueError,"explicit --threads"):
                self.queue.enqueue(spec)
        self.assertFalse(self.queue.state()["jobs"])

    def test_render_explicit_threads_exact_resource_boundary(self):
        spec = self.render_spec()
        spec["argv"][-1] = "2"
        with self.assertRaisesRegex(ValueError,"CPU reservation"):
            self.queue.enqueue(spec)
        spec["cpu_threads"] = 2
        job = self.queue.enqueue(spec)
        self.assertEqual(job["spec"]["cpu_threads"],2)
        self.assertEqual(job["status"],"queued")
        # This fixture is never run: no render/data directory gets created.
        self.assertFalse(any(Path(p).exists() for p in spec["result_dirs"]))

    def test_render_must_register_authoritative_derived_directory(self):
        from data import action4_synthesis
        spec = self.render_spec()
        spec["result_dirs"] = spec["result_dirs"][:1]
        with mock.patch.object(action4_synthesis,"_run_output_dir",wraps=action4_synthesis._run_output_dir) as mapping:
            with self.assertRaisesRegex(ValueError,"Derived render runtime output"):
                self.queue.enqueue(spec)
        mapping.assert_called_once_with(Path(spec["result_dirs"][0]))
        self.assertFalse(self.queue.state()["jobs"])

    def test_render_and_other_actual_enqueue_cannot_share_derived_runtime(self):
        render = self.render_spec()
        self.queue.enqueue(render)
        other = dict(self.spec,job_id="other",output_dir=str(self.base/"other_logs"),
                     result_dirs=[render["result_dirs"][1]])
        with self.assertRaisesRegex(ValueError,"overlaps"):
            self.queue.enqueue(other)
        self.assertEqual(list(self.queue.state()["jobs"]),[render["job_id"]])
        self.assertFalse(Path(render["result_dirs"][1]).exists())

    def test_render_actual_enqueue_rejects_runtime_owned_by_earlier_job(self):
        render = self.render_spec()
        other = dict(self.spec,job_id="other",output_dir=str(self.base/"other_logs"),
                     result_dirs=[render["result_dirs"][1]])
        self.queue.enqueue(other)
        with self.assertRaisesRegex(ValueError,"overlaps"):
            self.queue.enqueue(render)
        self.assertEqual(list(self.queue.state()["jobs"]),["other"])

    def test_render_existing_derived_directory_rejected(self):
        spec = self.render_spec()
        derived = Path(spec["result_dirs"][1])
        real_exists = Path.exists
        with mock.patch.object(Path,"exists",autospec=True,side_effect=lambda p: p==derived or real_exists(p)):
            with self.assertRaisesRegex(ValueError,"already exists"):
                self.queue.enqueue(spec)
        self.assertFalse(self.queue.state()["jobs"])

    def group_spec(self, count=4):
        devices = list(dispatch.ALLOWED_GPU_UUIDS[:count])
        run_id = self.base.name + "_train"
        return dict(self.spec, kind="train", training_stage="B0", gpu_count=count, allowed_gpu_uuids=devices,
            reservation_mib=10000, reservation_mib_by_gpu={d: 10000 for d in devices},
            cpu_threads=2*count, profile=False, profile_key="fixed-training-fixture", profile_id="group_profile",
            argv=[".venv/bin/python", "scripts/action4_train.py", "--config", str(self.config),
                  "--stage", "B0", "--run-id", run_id, "--device", "cuda"],
            code={"scripts/action4_train.py": dispatch.sha256(ROOT / "scripts/action4_train.py")},
            result_dirs=[str(ROOT / "outputs/action4" / run_id), str(ROOT / "models/action4" / run_id)])

    def test_group_one_insufficient_gpu_never_partially_launches(self):
        spec = self.group_spec()
        self.queue.save({"schema": "fixture", "jobs": {"group": {"spec": spec, "status": "queued", "token": "fixture"}}})
        state = self.queue.state()
        before = copy.deepcopy(state)
        def device_info(device):
            return {"free_mib": dispatch.HEADROOM_MIB + spec["reservation_mib_by_gpu"][device] - 1 if device == spec["allowed_gpu_uuids"][2] else 30000}
        with mock.patch.object(dispatch, "gpu_identity", side_effect=device_info), \
             mock.patch.object(dispatch.shutil, "disk_usage", return_value=mock.Mock(free=10**12)), \
             mock.patch.object(self.queue, "_launch") as launch:
            self.assertIs(self.queue._choose_device(spec, state), False)
            observed = self.queue.run(poll_seconds=.01, max_seconds=.01)
        launch.assert_not_called()
        self.assertEqual(state, before)
        self.assertEqual(observed["jobs"]["group"]["status"], "queued")
        self.assertNotIn("devices", observed["jobs"]["group"])

    def test_all_four_devices_must_pass_together(self):
        spec = self.group_spec()
        devices = spec["allowed_gpu_uuids"]
        with mock.patch.object(dispatch.shutil, "disk_usage", return_value=mock.Mock(free=10**12)):
            with mock.patch.object(dispatch, "gpu_identity", return_value={"free_mib":30000}):
                self.assertEqual(self.queue._choose_device(spec,{"jobs":{}}), devices)
            for denied in devices:
                with mock.patch.object(dispatch, "gpu_identity", side_effect=lambda d: {"free_mib":0 if d==denied else 30000}):
                    self.assertIs(self.queue._choose_device(spec,{"jobs":{}}), False)

    def test_missing_one_gpu_peak_makes_group_profile_inconclusive(self):
        spec = self.group_spec()
        spec["profile"] = True
        devices = spec["allowed_gpu_uuids"]
        job = dict(spec=spec,status="succeeded",devices=devices,launched_at=0,finished_at=10,
                   peak_mib_by_gpu={d:1000 for d in devices[:-1]},
                   samples_by_gpu={d:3 for d in devices},memory_samples=3)
        self.queue._finish_profile(job)
        record = dispatch.read(self.queue.directory / "profiles" / (spec["job_id"]+".json"))
        self.assertEqual(record["status"],"inconclusive")

    def test_duplicate_worker_identity_cannot_spawn_second_group(self):
        spec = self.group_spec(2)
        job = dict(spec=spec,status="launching",devices=spec["allowed_gpu_uuids"],token="fixture",
                   worker={"pid":100,"start":"original"})
        self.queue.save({"jobs":{"group":job}})
        with mock.patch.object(dispatch,"process_identity",return_value={"pid":200,"start":"duplicate"}), \
             mock.patch.object(dispatch.subprocess,"Popen") as spawn:
            with self.assertRaisesRegex(RuntimeError,"Another worker identity"):
                self.queue.worker("group","fixture")
        spawn.assert_not_called()

    def test_group_launch_books_all_devices_before_single_spawn(self):
        spec = self.group_spec()
        job = dict(spec=spec, status="queued", token="fixture", peak_mib=0, memory_samples=0)
        state = {"schema": "fixture", "jobs": {spec["job_id"]: job}}
        Path(spec["output_dir"]).mkdir()
        spawn_states = []
        def spawn(*args, **kwargs):
            spawn_states.append(self.queue.state())
            self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], ",".join(spec["allowed_gpu_uuids"]))
            self.assertNotIn("RANK", kwargs["env"])
            self.assertNotIn("TORCHELASTIC_RUN_ID", kwargs["env"])
            return mock.Mock(pid=123)
        with mock.patch.object(self.queue, "_validate", return_value=spec), \
             mock.patch.object(dispatch, "gpu_identity", return_value={}), \
             mock.patch.object(dispatch, "worker_environment", return_value={"RANK":"3", "TORCHELASTIC_RUN_ID":"foreign"}), \
             mock.patch.object(dispatch, "process_identity", return_value={"pid":123,"start":"fixture"}), \
             mock.patch.object(dispatch.subprocess, "Popen", side_effect=spawn) as popen:
            self.queue._launch(state, job, spec["allowed_gpu_uuids"])
        self.assertEqual(popen.call_count, 1)
        recorded = spawn_states[0]["jobs"][spec["job_id"]]
        self.assertEqual(recorded["devices"], spec["allowed_gpu_uuids"])
        self.assertEqual(recorded["status"], "launching")
        self.assertEqual(recorded["spec"]["reservation_mib_by_gpu"], spec["reservation_mib_by_gpu"])

    def test_group_reservation_blocks_single_gpu_overbooking(self):
        group = self.group_spec()
        devices = group["allowed_gpu_uuids"]
        active = dict(spec=group, status="running", devices=devices,
                      current_mib_by_gpu={d:8000 for d in devices})
        state = {"jobs": {"training": active}}
        single = dict(self.spec, allowed_gpu_uuids=[devices[0]], gpu_count=1,
                      reservation_mib=20000-group["reservation_mib_by_gpu"][devices[0]]-dispatch.HEADROOM_MIB+1, profile=False)
        with mock.patch.object(dispatch, "gpu_identity", return_value={"free_mib":12000}), \
             mock.patch.object(dispatch.shutil, "disk_usage", return_value=mock.Mock(free=10**12)):
            self.assertIs(self.queue._choose_device(single, state), False)
            single["reservation_mib"] = 20000-group["reservation_mib_by_gpu"][devices[0]]-dispatch.HEADROOM_MIB
            self.assertEqual(self.queue._choose_device(single, state), devices[0])

    def test_training_profile_gpu_count_and_order_must_match(self):
        spec = self.group_spec(2)
        record = dict(status="conclusive", profile_key=spec["profile_key"], training_stage="B0", gpu_count=4,
                      devices=spec["allowed_gpu_uuids"], recommended_reservation_mib=10000,
                      recommended_reservation_mib_by_gpu=spec["reservation_mib_by_gpu"])
        dispatch.atomic(self.queue.directory / "profiles/group_profile.json", record)
        with self.assertRaisesRegex(ValueError, "gpu_count/topology"):
            self.queue._validate(spec)
        record["gpu_count"] = 2
        record["devices"] = list(reversed(spec["allowed_gpu_uuids"]))
        dispatch.atomic(self.queue.directory / "profiles/group_profile.json", record)
        with self.assertRaisesRegex(ValueError, "ordered GPU topology"):
            self.queue._validate(spec)

    def test_group_observation_timeout_keeps_entire_live_reservation(self):
        spec = self.group_spec()
        job = dict(spec=spec, status="running", devices=spec["allowed_gpu_uuids"],
                   launched_at=time.time()-100, worker={"pid":123,"start":"fixture"})
        self.queue.save({"schema":"fixture", "jobs":{"group":job}})
        with mock.patch.object(dispatch, "is_live", return_value=True), \
             mock.patch.object(self.queue, "_telemetry"):
            state = self.queue.run(poll_seconds=.01, max_seconds=.01)
        self.assertEqual(state["jobs"]["group"]["status"], "running")
        self.assertEqual(state["jobs"]["group"]["devices"], spec["allowed_gpu_uuids"])
        self.assertEqual(state["jobs"]["group"]["spec"]["reservation_mib_by_gpu"], spec["reservation_mib_by_gpu"])

    def test_ddp_telemetry_includes_rank_processes_and_per_gpu_peaks(self):
        spec = self.group_spec(2)
        spec["profile"] = True
        a,b = spec["allowed_gpu_uuids"]
        worker = {"pid":101,"start":"fixture"}
        job = dict(spec=spec,status="running",devices=[a,b],worker=worker,
                   peak_mib=0,memory_samples=0,launched_at=time.time()-10)
        memory = {a:{102:1500,999:9999},b:{103:2000,999:9999}}
        with mock.patch.object(dispatch, "descendants", side_effect=lambda identity: {101,102,103} if identity else set()), \
             mock.patch.object(dispatch, "process_identity", side_effect=lambda pid: {"pid":pid,"start":"fixture"}), \
             mock.patch.object(dispatch, "gpu_process_memory", side_effect=lambda device:memory[device]):
            for _ in range(3):
                self.queue._telemetry({"jobs":{"group":job}})
        self.assertEqual(job["current_mib_by_gpu"], {a:1500,b:2000})
        self.assertEqual({p["pid"] for p in job["tracked_processes"]}, {101,102,103})
        job.update(status="succeeded",finished_at=time.time())
        self.queue._finish_profile(job)
        record = dispatch.read(self.queue.directory / "profiles" / (spec["job_id"]+".json"))
        self.assertEqual(record["status"],"conclusive")
        self.assertEqual(record["gpu_count"],2)
        self.assertEqual(record["devices"],[a,b])
        self.assertEqual(record["recommended_reservation_mib_by_gpu"],{a:1875,b:2500})

    def test_surviving_rank_prevents_receipt_from_releasing_group(self):
        spec = self.group_spec(2)
        out = Path(spec["output_dir"])
        out.mkdir()
        rank = {"pid":103,"start":"fixture"}
        job = dict(spec=spec,status="running",devices=spec["allowed_gpu_uuids"], token="fixture",
                   launched_at=time.time()-10,tracked_processes=[rank])
        dispatch.atomic(out / "dispatch_exit.json", dict(token="fixture",exitcode=1,finished_at=time.time()))
        with mock.patch.object(dispatch,"is_live",side_effect=lambda p:p==rank):
            self.queue._refresh({"jobs":{"group":job}})
        self.assertEqual(job["status"],"running")
        self.assertEqual(job["devices"],spec["allowed_gpu_uuids"])

    def test_fixed_torchrun_wrapper_cannot_accept_launcher_override(self):
        spec = self.group_spec(4)
        argv = dispatch.workload_argv(ROOT,spec,"unique-attempt-token")
        self.assertEqual(argv[1:3],["-m","torch.distributed.run"])
        self.assertIn("--nproc_per_node=4",argv)
        self.assertIn("--max_restarts=0",argv)
        self.assertIn("--rdzv_endpoint=127.0.0.1:0",argv)
        self.assertIn("--log_dir="+str(Path(spec["output_dir"])/"torchrun"),argv)
        bad = dict(spec,argv=spec["argv"]+["--nproc_per_node=8"])
        with self.assertRaisesRegex(ValueError,"Unknown"):
            self.queue._validate(bad,profile=True)
        bad = dict(spec,kind="render")
        with self.assertRaisesRegex(ValueError,"Training jobs"):
            self.queue._validate(bad,profile=True)

    def test_fixed_wrapper_parses_with_installed_torchrun_without_launch(self):
        from torch.distributed.run import get_args_parser
        spec = self.group_spec(4)
        argv = dispatch.workload_argv(ROOT,spec,"parser-only-token")
        parsed = get_args_parser().parse_args(argv[3:])
        self.assertEqual(parsed.nproc_per_node,"4")
        self.assertEqual(parsed.max_restarts,0)
        self.assertEqual(parsed.rdzv_endpoint,"127.0.0.1:0")
        self.assertEqual(parsed.training_script,"scripts/action4_train.py")
        self.assertEqual(parsed.log_dir,str(Path(spec["output_dir"])/"torchrun"))

    def test_training_stage_cannot_reuse_another_stage_profile(self):
        spec = self.group_spec(2)
        record = dict(status="conclusive",profile_key=spec["profile_key"],training_stage="A0")
        dispatch.atomic(self.queue.directory / "profiles/group_profile.json",record)
        with self.assertRaisesRegex(ValueError,"Training stage differs"):
            self.queue._validate(spec)

    def test_training_resume_cannot_change_topology(self):
        spec = self.group_spec(2)
        previous = dict(spec=spec,status="succeeded",devices=spec["allowed_gpu_uuids"])
        self.queue.save({"jobs":{"previous":previous}})
        resumed = dict(spec,resume_from_job="previous",allowed_gpu_uuids=list(reversed(spec["allowed_gpu_uuids"])))
        with self.assertRaisesRegex(ValueError,"topology"):
            self.queue._validate_resume(resumed,{"run-id":"fixture","resume":"unused"})

    def test_real_cpu_success_exit_and_failed_exit(self):
        # Real processes exercise worker receipts, locks, stdout and exit status.
        # Only --help/invalid-flag parsing runs: no GPU or model work.
        self.queue.enqueue(self.spec)
        bad = copy.deepcopy(self.spec)
        bad.update(job_id="cpu_bad", output_dir=str(self.base / "bad"))
        bad["argv"][-1] = "not-a-dispatch-command"
        self.queue.enqueue(bad)
        state = self.queue.run(poll_seconds=.05, max_seconds=15)
        good, failed = state["jobs"]["cpu_test"], state["jobs"]["cpu_bad"]
        self.assertEqual(good["status"], "succeeded", good)
        self.assertEqual(good["exitcode"], 0)
        self.assertEqual(failed["status"], "failed", failed)
        self.assertEqual(failed["exitcode"], 2)
        self.assertTrue((Path(self.spec["output_dir"]) / "dispatch_worker.json").exists())


if __name__ == "__main__":
    unittest.main()
