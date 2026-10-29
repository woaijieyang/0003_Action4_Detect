import copy
import unittest
from model.training.action4_eval_schedule import SCHEDULE, SAM_PROTOCOL, eval_due, expected_eval_steps, validate_schedule

class EvalScheduleTests(unittest.TestCase):
    def cfg(self):
        return dict(training_protocol=SAM_PROTOCOL,role="A0",eval_every=25,
                    eval_schedule=copy.deepcopy(SCHEDULE),expected_units=49,batch=48,epochs=32)
    def ledger(self,skip=()):
        rows=[];step=zero=0
        for i in range(64):
            active=i not in skip
            step+=int(active);zero+=int(not active)
            rows.append(dict(epoch=i//2+1,cursor=48 if i%2==0 else 49,
                data_update_attempt=i+1,did_update=active,step=step,no_supervision_update_count=zero))
        return rows,dict(step=step,epochs=32,no_supervision_update_count=zero)
    def test_boundary_global_alignment(self):
        c=self.cfg()
        self.assertFalse(eval_due(c,6510,30))
        self.assertTrue(eval_due(c,6510,31))
        self.assertTrue(eval_due(c,6500,30))
        self.assertFalse(eval_due(c,6501,31))
    def test_actual_ledger_switch(self):
        c=self.cfg();rows,t=self.ledger()
        self.assertEqual(expected_eval_steps(c,t,rows),{0,25,50,64})
        # With60updates before31,61..64 has no global multiple10.
    def test_skips_do_not_create_eval(self):
        c=self.cfg();rows,t=self.ledger((0,1))
        self.assertEqual(expected_eval_steps(c,t,rows),{0,25,50,60,62})
        self.assertFalse(rows[0]["did_update"])
    def test_tampered_missing_duplicate_ledger_rejected(self):
        c=self.cfg();rows,t=self.ledger()
        for bad in [rows[:-1],rows+[rows[-1]],rows[::-1]]:
            with self.assertRaises(ValueError):expected_eval_steps(c,t,bad)
        bad=copy.deepcopy(rows);bad[60]["epoch"]=30
        with self.assertRaises(ValueError):expected_eval_steps(c,t,bad)
    def test_unscheduled_contract_unchanged(self):
        self.assertEqual(expected_eval_steps(dict(eval_every=25),dict(step=28)),{0,25,28})
    def test_unknown_contract_rejected(self):
        c=self.cfg();c["eval_schedule"]["after_every"]=5
        with self.assertRaises(ValueError):validate_schedule(c)
        c=self.cfg();c["training_protocol"]="baseline"
        with self.assertRaises(ValueError):validate_schedule(c)
    def test_resume_at_epoch_boundary(self):
        c=self.cfg()
        # Completed30epochs checkpoint: last data update still belongs to30.
        self.assertFalse(eval_due(c,6510,30))
        # First successful update at31 uses10 even if prior attempt skipped.
        self.assertTrue(eval_due(c,6520,31))
    def test_zero_only_final_is_not_successful_training(self):
        c=self.cfg();rows,t=self.ledger(tuple(range(64)))
        self.assertEqual(expected_eval_steps(c,t,rows),{0})
        # Trainer separately refuses completed runs with optimizer step0.

    def test_resume_uses_checkpoint_not_abandoned_log_tail(self):
        from model.training.action4_eval_schedule import resume_ledger
        c=self.cfg();rows,t=self.ledger()
        state=dict(step=50,data_update_attempt=50,no_supervision_update_count=0,
                   eval_schedule_ledger=copy.deepcopy(rows[:50]))
        restored=resume_ledger(c,state)
        self.assertEqual(len(restored),50)
        self.assertEqual(expected_eval_steps(c,t,restored+rows[50:]),{0,25,50,64})
        state["eval_schedule_ledger"].append(rows[50])
        with self.assertRaises(ValueError):resume_ledger(c,state)
    def test_final_ledger_identity(self):
        import tempfile,json,hashlib
        from pathlib import Path
        from model.training.action4_eval_schedule import read_schedule_ledger
        c=self.cfg();rows,t=self.ledger()
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"EVAL_SCHEDULE_LEDGER.json";raw=json.dumps(rows).encode();p.write_bytes(raw)
            t["eval_schedule_ledger_sha256"]=hashlib.sha256(raw).hexdigest()
            self.assertEqual(read_schedule_ledger(c,Path(tmp),t),rows)
            p.write_bytes(raw+b" ")
            with self.assertRaises(ValueError):read_schedule_ledger(c,Path(tmp),t)
