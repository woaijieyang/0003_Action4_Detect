import copy
import hashlib
import json
import unittest
from model.training.action4_loader_continuation import validate_loader_continuation, canonical_config


def fixtures():
    files={"src/model/training/train.py":"a"*64,"src/data/loader.py":"b"*64,
           "inputs/A0_parent.pt":"c"*64,"inputs/B0_counterpart.pt":"d"*64,
           "inputs/parent_registry.json":"e"*64,"resolved_config.json":"f"*64}
    consumed={k:v for k,v in files.items() if k.startswith("src/")}
    code_id=hashlib.sha256(json.dumps(consumed,sort_keys=True).encode()).hexdigest()
    def cfg(name,workers):
        root="runtime/action4_runs/"+name+"/source"
        return dict(role="A0",mode="full_training",batch=48,micro=2,world_size=1,epochs=80,
                    eval_every=10,expected_units=10400,expected_eval_images=650,require_immutable_source=True,
                    seed=20261024,loader_workers=workers,threads=2,eval_loader_workers=2,
                    training_gpus=[7],eval_gpus=[4,6,7,7],project_total_reserved_cpu=69,
                    reserved_cpu_budget=workers+19,
                    process_cpu_affinity={"train":list(range(48,48+workers+2)),
                        "eval_0":[44,45,46,47],"eval_1":[40,41,42,43],
                        "eval_2":[28,29,30,31],"eval_3":[32,33,34,35]},
                    output="outputs/"+name,run_spec="runtime/action4_runs/"+name+"/RUN_SPEC.json",
                    parent_bundle=root+"/inputs/A0_parent.pt",parent_registry=root+"/inputs/parent_registry.json",
                    evaluation_counterpart_bundle=root+"/inputs/B0_counterpart.pt",
                    parent_bundle_sha256="c"*64,evaluation_counterpart_sha256="d"*64,
                    source_identity={root+"/"+k:v for k,v in consumed.items()},
                    project_code_sha256=code_id,tested_project_code_sha256=code_id,
                    integration_or_admission_receipt="outputs/"+name+"/ADMISSION.json",
                    loss_weights={"xyz":1.0},lr=0.00005,weight_decay=0.0001,freeze_bn=True,
                    dataset_root="data/phrase2",surface_source={"manifest_sha256":{"train":"1"*64}},
                    roi_jitter={"center_fraction":0.15},parent_best_step=14000,
                    validation_receipt_sha256="2"*64)
    old,new=cfg("old",4),cfg("new",12)
    state=dict(resumable=True,role="A0",config_id=canonical_config(old),world_size=1,
               step=7847,epoch=36,cursor=1680,data_update_attempt=7847,no_supervision_update_count=0,
               training_model={"weight":[1.0]},optimizer={"state":{0:{"step":7847}},"param_groups":[{"lr":1e-5}]},
               scheduler={"last_epoch":7847},rng_by_rank=[{"torch_cpu":[1,2]}])
    return old,new,state,files,copy.deepcopy(files)


class LoaderContinuation(unittest.TestCase):
    def test_valid_is_nonlaunchable_and_inputs_unchanged(self):
        args=fixtures();before=copy.deepcopy(args)
        result=validate_loader_continuation(*args)
        self.assertEqual(args,before)
        self.assertFalse(result.launch_allowed)
        self.assertEqual(result.actual_consumed_units,376080)
        self.assertTrue(all(s=="UNSATISFIED" for _,s in result.required_runtime_switch_gates))
        self.assertEqual(args[2]["config_id"],canonical_config(args[0]))
        self.assertNotEqual(args[2]["config_id"],result.successor_config_id)
    def test_semantic_fields_fail_closed(self):
        changes={"batch":96,"micro":4,"world_size":2,"epochs":81,"eval_every":20,
                 "seed":0,"lr":1e-3,"weight_decay":0,"freeze_bn":False,"parent_best_step":0,
                 "loss_weights":{"xyz":True},"surface_source":{},"roi_jitter":{},
                 "dataset_root":"data/other","parent_bundle_sha256":"9"*64,
                 "evaluation_counterpart_sha256":"9"*64,"validation_receipt_sha256":"9"*64}
        for key,value in changes.items():
            with self.subTest(key=key):
                args=list(fixtures());args[1][key]=value
                with self.assertRaises(ValueError):validate_loader_continuation(*args)
    def test_source_body_and_inventory_rejected_even_with_fresh_config_hashes(self):
        for name in ["src/data/loader.py","inputs/A0_parent.pt","inputs/parent_registry.json"]:
            with self.subTest(name=name):
                args=list(fixtures());args[4][name]="9"*64
                if name.startswith("src/"):
                    args[1]["source_identity"]["runtime/action4_runs/new/source/"+name]="9"*64
                    consumed={k:v for k,v in args[4].items() if k.startswith("src/")}
                    fresh=hashlib.sha256(json.dumps(consumed,sort_keys=True).encode()).hexdigest()
                    args[1]["project_code_sha256"]=fresh;args[1]["tested_project_code_sha256"]=fresh
                with self.assertRaises(ValueError):validate_loader_continuation(*args)
        args=list(fixtures());args[4]["runtime/surface_deps/injected.py"]="9"*64
        with self.assertRaises(ValueError):validate_loader_continuation(*args)
    def test_actual_consumed_source_binding(self):
        args=list(fixtures());args[1]["source_identity"]["runtime/action4_runs/new/source/src/data/loader.py"]="8"*64
        with self.assertRaises(ValueError):validate_loader_continuation(*args)
    def test_cursor_attempt_and_tail_mass(self):
        for changes in [{"cursor":1681},{"cursor":10400},{"epoch":80},{"step":7848},
                        {"data_update_attempt":7848},{"no_supervision_update_count":1},{"step":True},
                        {"scheduler":{"last_epoch":0}}]:
            with self.subTest(changes=changes):
                args=list(fixtures());args[2].update(changes)
                with self.assertRaises(ValueError):validate_loader_continuation(*args)
        args=list(fixtures());args[2].update(epoch=1,cursor=0,step=217,data_update_attempt=217,
                                            scheduler={"last_epoch":217})
        self.assertEqual(validate_loader_continuation(*args).actual_consumed_units,10400)
        args[2].update(epoch=0,cursor=10368,step=216,data_update_attempt=216,
                       scheduler={"last_epoch":216})
        self.assertEqual(validate_loader_continuation(*args).actual_consumed_units,10368)
    def test_skipped_update_accounting(self):
        args=list(fixtures());args[2].update(step=7846,no_supervision_update_count=1,scheduler={"last_epoch":7846})
        self.assertEqual(validate_loader_continuation(*args).data_update_attempt,7847)
    def test_state_identity_and_payload_presence(self):
        for key,value in [("config_id","0"*64),("resumable",False),("role","B0"),("world_size",2),
                          ("training_model",{}),("optimizer",{}),("rng_by_rank",[]),("attempt_id","")]:
            with self.subTest(key=key):
                args=list(fixtures());args[2][key]=value
                with self.assertRaises(ValueError):validate_loader_continuation(*args)
    def test_cpu_gpu_constraints(self):
        for mutation in [
            lambda c:c.update(training_gpus=[0]),
            lambda c:c.update(eval_gpus=[4,6,7,3]),
            lambda c:c.update(project_total_reserved_cpu=73),
            lambda c:c.update(reserved_cpu_budget=23),
            lambda c:c["process_cpu_affinity"].update(train=[48,49]),
            lambda c:c["process_cpu_affinity"].update(train=list(range(10,24))),
            lambda c:c["process_cpu_affinity"].update(train=[48]*14),
            lambda c:c["process_cpu_affinity"].update(train=list(range(40,54))),
            lambda c:c.update(loader_workers=8)]:
            args=list(fixtures());mutation(args[1])
            with self.assertRaises(ValueError):validate_loader_continuation(*args)
    def test_path_and_unknown_field_rejected(self):
        for key,value in [("output","../outside"),("run_spec","/tmp/RUN_SPEC.json"),
                          ("parent_bundle","runtime/action4_runs/new/source/inputs/../other.pt"),
                          ("integration_or_admission_receipt","outputs/unrelated/ADMISSION.json")]:
            args=list(fixtures());args[1][key]=value
            with self.assertRaises(ValueError):validate_loader_continuation(*args)
        args=list(fixtures());args[1]["unknown_override"]=True
        with self.assertRaises(ValueError):validate_loader_continuation(*args)
    def test_metadata_manifest_may_differ_only_for_resolved_config(self):
        args=list(fixtures());args[4]["resolved_config.json"]="0"*64
        self.assertFalse(validate_loader_continuation(*args).launch_allowed)
        args[4]["../escape"]="0"*64
        with self.assertRaises(ValueError):validate_loader_continuation(*args)


if __name__=="__main__":
    unittest.main()
