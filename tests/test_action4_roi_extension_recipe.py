"""CPU recipe checks against the actually sealed parent; no model execution."""
import copy,hashlib,json,math,unittest
from pathlib import Path
from model.training.action4_roi_adaptation import validate_recipe,extension_segment,validate_extension_parent
from eval.action4_roi_pair_selection import validate_pair
from test_action4_role_seal_metadata import seal

ROOT=Path("/data/zy/vln/a-test")
PARENT=ROOT/"models/candidates/action4_improvement/roi15_prediction50_completed_20261026/SEALED_ROLE.json"


def extension_config():
    parent=json.loads(PARENT.read_text())
    spec=json.loads((ROOT/parent["run_spec"]).read_text())
    cfg=json.loads((ROOT/spec["config"]).read_text())
    units=cfg["expected_units"]
    cfg.update(epochs=10,formal_epochs=10,formal_training_units=units,batch=64,micro=2,world_size=4,eval_every=10,
        schedule_total_updates=10*math.ceil(units/64),parent_bundle=parent["bundle"],
        parent_bundle_sha256=parent["sha256"],parent_best_step=parent["selected_step"],
        continuation_segment={"schema":"action4.roi15_best_extension.v1","epochs":10,"batch":64,"eval_every":10,
          "parent_seal":{"path":str(PARENT.relative_to(ROOT)),"sha256":hashlib.sha256(PARENT.read_bytes()).hexdigest()}})
    return cfg,parent


class ExtensionRecipeTests(unittest.TestCase):
    def test_actual_parent_dynamic_identity_and_full_budget(self):
        cfg,parent=extension_config()
        validate_recipe(cfg)
        self.assertEqual(validate_extension_parent(cfg)["sha256"],parent["sha256"])
        self.assertEqual(cfg["schedule_total_updates"],1630)
        self.assertEqual(len(set([0,1630]+list(range(10,1631,10)))),164)
        self.assertTrue(seal.validate_formal_recipe(cfg))

    def test_integration_subset_keeps_formal_budget(self):
        cfg,_=extension_config()
        cfg.update(mode="integration",epochs=1,expected_units=1312,train_indices=list(range(1312)))
        validate_recipe(cfg)
        self.assertEqual(math.ceil(cfg["expected_units"]/cfg["batch"]),21)
        self.assertEqual(cfg["schedule_total_updates"],1630)
        for bad in (None,list(range(1311)),[0]*1312,list(range(1300))+list(range(10400,10412))):
            changed=copy.deepcopy(cfg)
            if bad is None:changed.pop("train_indices")
            else:changed["train_indices"]=bad
            with self.assertRaises(ValueError):validate_recipe(changed)

    def test_no_arbitrary_short_recipe_or_parent_override(self):
        cfg,_=extension_config()
        for key,value in (("epochs",9),("world_size",2),("batch",48),("eval_every",25),
                          ("schedule_total_updates",1600),("formal_training_units",1312),
                          ("parent_best_step",0),("parent_bundle_sha256","a"*64),
                          ("evaluation_counterpart_sha256","b"*64),("roi_source","GT"),
                          ("dataset_root","wrong"),("lr",2e-5)):
            changed=copy.deepcopy(cfg);changed[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_recipe(changed)
        changed=copy.deepcopy(cfg);changed.pop("continuation_segment")
        with self.assertRaises(ValueError):validate_recipe(changed)
        changed=copy.deepcopy(cfg);changed["continuation_segment"]["epochs"]=11
        with self.assertRaises(ValueError):extension_segment(changed)
        changed=copy.deepcopy(cfg);changed["train_indices"]=list(range(10400))
        with self.assertRaises(ValueError):validate_recipe(changed)

    def test_pair_extension_requires_full_seal_coverage_and_contract(self):
        cfg,_=extension_config()
        b={"role":"B0","strict_load":True,"sha256":cfg["evaluation_counterpart_sha256"]}
        contract={"roi_expansion":1.5,"B0_sha256":b["sha256"],
                  "frozen_eval_b0_predictions":cfg["frozen_eval_b0_predictions"],
                  "continuation_segment":cfg["continuation_segment"]}
        a={"role":"A0","strict_load":True,"epochs":10,"score_schema":cfg["score_schema"],
           "continuation_segment":cfg["continuation_segment"],"completed_optimizer_steps":1630,
           "frozen_backbone_verified":True,"completion":{"evaluations":164,"inference_contract":contract}}
        validate_pair(a,cfg,b,{"role":"B0"})
        for key,value in (("epochs",9),("strict_load",False),("completed_optimizer_steps",1620),
                          ("frozen_backbone_verified",False)):
            wrong=copy.deepcopy(a);wrong[key]=value
            with self.assertRaises(ValueError):validate_pair(wrong,cfg,b,{"role":"B0"})
        wrong=copy.deepcopy(a);wrong["completion"]["evaluations"]=163
        with self.assertRaises(ValueError):validate_pair(wrong,cfg,b,{"role":"B0"})

    def test_old_fifty_recipe_remains_default(self):
        cfg,_=extension_config()
        source=json.loads((ROOT/"runtime/action4_runs/p2_a0_roi15_prediction50_v001/source/resolved_config.json").read_text())
        validate_recipe(source)
        self.assertFalse(extension_segment(source))
        self.assertTrue(seal.validate_formal_recipe(source))



    def test_segment_bound_admission_and_formal_scheduler_budget(self):
        from model.training.action4_roi_adaptation import validate_admission,make_scheduler
        import torch
        cfg,_=extension_config()
        cfg.update(mode="integration",epochs=1,expected_units=1312,train_indices=list(range(1312)))
        gate={"launch_allowed":True,"mode":"integration","training_protocol":cfg["training_protocol"],
              "continuation_segment":copy.deepcopy(cfg["continuation_segment"])}
        validate_admission(cfg,gate)
        wrong=copy.deepcopy(gate);wrong.pop("continuation_segment")
        with self.assertRaises(ValueError):validate_admission(cfg,wrong)
        optimizer=torch.optim.AdamW([torch.nn.Parameter(torch.tensor(1.))],lr=cfg["lr"])
        scheduler=make_scheduler(optimizer,cfg,21)
        self.assertAlmostEqual(scheduler.lr_lambdas[0](1630),cfg["eta_min"]/cfg["lr"])
        formal,_=extension_config()
        with self.assertRaises(ValueError):make_scheduler(optimizer,formal,1629)


if __name__=="__main__":
    unittest.main()
