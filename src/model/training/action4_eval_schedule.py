"""Epoch-aware eval cadence; counts only actual optimizer updates."""
import math

SAM_PROTOCOL = "action4.sam_prediction_roi_enhancement.native_v1"
SCHEDULE = {"schema": "action4.epoch_eval_schedule.v1",
            "switch_after_epochs": 30, "before_every": 25, "after_every": 10,
            "step_alignment": "global_optimizer_step"}


def validate_schedule(cfg):
    schedule = cfg.get("eval_schedule")
    if schedule is None:
        return None
    if cfg.get("training_protocol") != SAM_PROTOCOL or cfg.get("role") != "A0":
        raise ValueError("Epoch eval schedule is registered only for independent SAM A0")
    if schedule != SCHEDULE or cfg.get("eval_every") != 25:
        raise ValueError("Unregistered epoch eval schedule")
    return schedule


def eval_due(cfg, step, epoch_one_based):
    schedule = validate_schedule(cfg)
    if type(step) is not int or step < 0 or type(epoch_one_based) is not int or epoch_one_based < 1:
        raise ValueError("Valid optimizer step and one-based data epoch required")
    if step == 0:
        return True
    interval = cfg["eval_every"] if schedule is None else (
        schedule["before_every"] if epoch_one_based <= schedule["switch_after_epochs"]
        else schedule["after_every"])
    return step % interval == 0


def expected_eval_steps(cfg, terminal, rows=None):
    """Scheduled runs require the complete actual traversal log, including skipped updates."""
    if validate_schedule(cfg) is None:
        step = terminal["step"]
        return {0, step} | set(range(cfg["eval_every"], step + 1, cfg["eval_every"]))
    if rows is None:
        raise ValueError("Epoch schedule coverage requires actual update ledger")
    units, batch, epochs = cfg["expected_units"], cfg["batch"], cfg["epochs"]
    per_epoch = math.ceil(units / batch)
    if len(rows) != epochs * per_epoch:
        raise ValueError("Incomplete scheduled update ledger")
    expected, step, skipped = {0}, 0, 0
    for i, row in enumerate(rows):
        epoch = i // per_epoch + 1
        cursor = min((i % per_epoch + 1) * batch, units)
        if (row.get("epoch") != epoch or row.get("cursor") != cursor
                or row.get("data_update_attempt") != i + 1
                or type(row.get("did_update")) is not bool):
            raise ValueError("Update ledger traversal/epoch mismatch")
        if row["did_update"]:
            step += 1
            if eval_due(cfg, step, epoch):
                expected.add(step)
        else:
            skipped += 1
        if row.get("step") != step or row.get("no_supervision_update_count") != skipped:
            raise ValueError("Update ledger optimizer/skip count mismatch")
    if (step != terminal["step"] or skipped != terminal.get("no_supervision_update_count", 0)
            or terminal.get("epochs") != epochs):
        raise ValueError("Scheduled terminal differs from actual ledger")
    expected.add(step)
    return expected


def read_schedule_ledger(cfg, output, terminal):
    """Use checkpoint-restored effective traversal, never abandoned attempt log tails."""
    if validate_schedule(cfg) is None:
        return None
    import hashlib
    import json
    raw = (output / "EVAL_SCHEDULE_LEDGER.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != terminal.get("eval_schedule_ledger_sha256"):
        raise ValueError("Final scheduled traversal identity differs")
    return json.loads(raw)


def resume_ledger(cfg, state):
    if validate_schedule(cfg) is None:
        return []
    rows = state.get("eval_schedule_ledger")
    if not isinstance(rows, list) or len(rows) != state["data_update_attempt"]:
        raise ValueError("Same-run recovery requires checkpoint-bound eval traversal")
    if rows and (rows[-1]["step"] != state["step"]
                 or rows[-1]["no_supervision_update_count"] != state["no_supervision_update_count"]):
        raise ValueError("Recovery eval ledger step differs")
    return rows
