"""Evaluation dispatcher CPU contracts; no real model, GPU, or queue execution."""
import copy
import tempfile
import unittest
from pathlib import Path
from model.training import action4_dispatch as dispatcher

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/action4_eval_dispatch_regression_fixtures"


class EvalDispatchTests(unittest.TestCase):
    def setUp(self):
        BASE.mkdir(parents=True, exist_ok=True)
        self.case=Path(tempfile.mkdtemp(prefix="fixture_",dir=BASE))
        self.queue=dispatcher.Dispatcher(ROOT,self.case/"queue")
        self.config=self.case/"fixture_config.json"
        self.config.write_text('{"scope":"CPU dispatch fixture, not model config"}')
        self.checkpoint=self.case/"fixture_checkpoint.txt"
        self.checkpoint.write_text("NOT A MODEL: immutable input binding fixture only")
        self.selection=self.case/"fixture_selection.json"
        self.selection.write_text('{"scope":"CPU input identity fixture only"}')
        self.grid=self.case/"fixture_grid.json"
        self.grid.write_text('{"score_threshold":[0.3]}')
        self.run="eval_dispatch_fixture_"+self.case.name
        self.result="outputs/action4/"+self.run
        self.base={"job_id":self.run,"kind":"cpu","cwd":str(ROOT),
            "argv":[".venv/bin/python","scripts/action4_eval.py","--config",str(self.config.relative_to(ROOT)),
                    "--checkpoint",str(self.checkpoint.relative_to(ROOT)),"--run-id",self.run,"--device","cpu"],
            "output_dir":str((self.case/"attempt").relative_to(ROOT)),"result_dirs":[self.result],
            "allowed_gpu_uuids":[],"reservation_mib":0,"cpu_threads":2,
            "code":{"scripts/action4_eval.py":dispatcher.sha256(ROOT/"scripts/action4_eval.py")},
            "configs":{str(self.config.relative_to(ROOT)):dispatcher.sha256(self.config)},
            "inputs":{str(self.checkpoint.relative_to(ROOT)):dispatcher.sha256(self.checkpoint)}}

    def val(self,s=None):
        return self.queue._validate(self.base if s is None else s,profile=False)

    def setflag(self,s,key,value=None):
        s["argv"].append("--"+key)
        if value is not None:s["argv"].append(value)
        return s

    def freeze(self):
        s=copy.deepcopy(self.base)
        i=s["argv"].index("--run-id");del s["argv"][i:i+2]
        return self.setflag(s,"freeze-selection",self.result+"/selection.json")

    def selection_spec(self):
        s=self.setflag(copy.deepcopy(self.base),"split","test")
        s=self.setflag(s,"selection",str(self.selection.relative_to(ROOT)))
        s["inputs"][str(self.selection.relative_to(ROOT))]=dispatcher.sha256(self.selection)
        return s

    def tune(self):
        s=self.setflag(copy.deepcopy(self.base),"tune-grid",str(self.grid.relative_to(ROOT)))
        s["configs"][str(self.grid.relative_to(ROOT))]=dispatcher.sha256(self.grid)
        return s

    def test_plain_eval_and_gt_roi(self):
        self.val()
        self.val(self.setflag(copy.deepcopy(self.base),"split","eval"))
        self.val(self.setflag(copy.deepcopy(self.base),"gt-roi"))
        self.val(self.setflag(copy.deepcopy(self.base),"allow-diagnostic"))

    def test_test_with_hashed_selection(self):
        self.val(self.selection_spec())

    def test_tune_with_hashed_grid(self):
        self.val(self.tune())
        self.val(self.setflag(self.tune(),"allow-diagnostic"))

    def test_cpu_freeze_fresh_file(self):
        r=self.val(self.freeze())
        self.assertEqual(r["gpu_count"],0)
        self.assertFalse((ROOT/self.result).exists())

    def test_eval_default_output_must_be_registered(self):
        s=copy.deepcopy(self.base);s["result_dirs"]=[]
        with self.assertRaisesRegex(ValueError,"default output"):self.val(s)

    def test_missing_run_id(self):
        s=copy.deepcopy(self.base);i=s["argv"].index("--run-id");del s["argv"][i:i+2]
        for candidate in [s,self.setflag(copy.deepcopy(s),"gt-roi")]:
            with self.assertRaisesRegex(ValueError,"run-id"):self.val(candidate)

    def test_missing_selection_or_diagnostic_test(self):
        s=self.setflag(copy.deepcopy(self.base),"split","test")
        with self.assertRaisesRegex(ValueError,"Formal test"):self.val(s)
        for flag in ["gt-roi","allow-diagnostic"]:
            with self.subTest(flag=flag),self.assertRaisesRegex(ValueError,"Formal test"):
                self.val(self.setflag(self.selection_spec(),flag))

    def test_eval_cannot_use_selection(self):
        s=self.selection_spec();s["argv"][s["argv"].index("--split")+1]="eval"
        with self.assertRaisesRegex(ValueError,"only to formal test"):self.val(s)

    def test_tune_cross_modes(self):
        for flag,value in [("gt-roi",None),("selection",str(self.selection.relative_to(ROOT))),
                           ("freeze-selection",self.result+"/x.json"),("split","test")]:
            with self.subTest(flag=flag),self.assertRaisesRegex(ValueError,"Threshold tuning"):
                self.val(self.setflag(self.tune(),flag,value))

    def test_freeze_cross_modes(self):
        for flag,value in [("gt-roi",None),("allow-diagnostic",None),
                           ("selection",str(self.selection.relative_to(ROOT))),("run-id","ignored"),("split","test")]:
            with self.subTest(flag=flag),self.assertRaisesRegex(ValueError,"standalone mode"):
                self.val(self.setflag(self.freeze(),flag,value))

    def test_freeze_requires_cpu(self):
        s=self.freeze();s["argv"][s["argv"].index("--device")+1]="cuda"
        with self.assertRaisesRegex(ValueError,"CPU-only"):self.val(s)
        s=self.freeze();s["kind"]="inference"
        with self.assertRaisesRegex(ValueError,"CPU-only"):self.val(s)
        s=self.freeze();s["allowed_gpu_uuids"]=[dispatcher.ALLOWED_GPU_UUIDS[0]]
        with self.assertRaisesRegex(ValueError,"CPU-only"):self.val(s)

    def test_freeze_cannot_escape_registered_output_or_be_directory(self):
        for value in ["outputs/other_unregistered/selection.json","models/action4/selection.json",self.result]:
            s=self.freeze();s["argv"][-1]=value
            with self.subTest(value=value),self.assertRaises(ValueError):self.val(s)

    def test_freeze_existing_output_rejected(self):
        s=self.freeze();s["result_dirs"]=[str(self.case.relative_to(ROOT))];s["argv"][-1]=str(self.selection.relative_to(ROOT));s["output_dir"]=str((BASE/(self.case.name+"_separate_attempt")).relative_to(ROOT))
        with self.assertRaisesRegex(ValueError,"already exists"):self.val(s)

    def test_unhashed_or_stale_inputs(self):
        s=self.selection_spec();del s["inputs"][str(self.selection.relative_to(ROOT))]
        with self.assertRaisesRegex(ValueError,"immutable hash"):self.val(s)
        s=self.tune();del s["configs"][str(self.grid.relative_to(ROOT))]
        with self.assertRaisesRegex(ValueError,"immutable hash"):self.val(s)
        s=self.selection_spec();self.selection.write_text("changed")
        with self.assertRaisesRegex(ValueError,"Stale"):self.val(s)

    def test_tune_grid_must_be_config_binding(self):
        s=self.tune();key=str(self.grid.relative_to(ROOT));s["inputs"][key]=s["configs"].pop(key)
        with self.assertRaisesRegex(ValueError,"immutable hash"):self.val(s)

    def test_paths_reject_traversal_and_outside(self):
        for option,value in [("selection","../outside.json"),("tune-grid","/outside/grid.json"),("freeze-selection","../selection.json")]:
            s=copy.deepcopy(self.base);self.setflag(s,option,value)
            with self.subTest(option=option),self.assertRaises(ValueError):self.val(s)

    def test_specific_allowlist_no_spillover(self):
        s=copy.deepcopy(self.base);s["argv"][1]="scripts/action4_dispatch.py"
        s["code"]={"scripts/action4_dispatch.py":dispatcher.sha256(ROOT/"scripts/action4_dispatch.py")}
        self.setflag(s,"split","eval")
        with self.assertRaisesRegex(ValueError,"Unknown"):self.val(s)
        for flag in ["stage","component-smoke","frame-manifest"]:
            with self.subTest(flag=flag),self.assertRaisesRegex(ValueError,"Unknown"):
                self.val(self.setflag(copy.deepcopy(self.base),flag,"B0" if flag=="stage" else None))

    def test_argument_arity_duplicates_abbreviations(self):
        for extra in [["--gt-roi=true"],["--gt-roi","true"],["--split"],["--spl","eval"],["--split","eval","orphan"],["--split=eval","orphan"],["--split","eval","--split","test"]]:
            s=copy.deepcopy(self.base);s["argv"]+=extra
            with self.subTest(extra=extra),self.assertRaises(ValueError):self.val(s)

    def test_equal_style_values(self):
        s=self.selection_spec()
        i=s["argv"].index("--split");s["argv"][i:i+2]=["--split=test"]
        i=s["argv"].index("--selection");s["argv"][i:i+2]=["--selection="+s["argv"][i+1]]
        self.val(s)

    def test_preexisting_entry_modes_still_validate(self):
        s=copy.deepcopy(self.base)
        s["argv"]=[".venv/bin/python","scripts/action4_dispatch.py","--help"]
        s["code"]={"scripts/action4_dispatch.py":dispatcher.sha256(ROOT/"scripts/action4_dispatch.py")}
        self.val(s)

    def test_effective_profiles_separate_eval_modes(self):
        full=self.val()["effective_workload_sha256"]
        gt=self.val(self.setflag(copy.deepcopy(self.base),"gt-roi"))["effective_workload_sha256"]
        test=self.val(self.selection_spec())["effective_workload_sha256"]
        tune=self.val(self.tune())["effective_workload_sha256"]
        freeze=self.val(self.freeze())["effective_workload_sha256"]
        self.assertEqual(len({full,gt,test,tune,freeze}),5)
        explicit_eval=self.val(self.setflag(copy.deepcopy(self.base),"split","eval"))["effective_workload_sha256"]
        self.assertEqual(full,explicit_eval)



if __name__ == "__main__":
    unittest.main()
