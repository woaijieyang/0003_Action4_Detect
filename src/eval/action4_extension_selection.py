"""Create cross-run selection through existing full sealed-role verification."""
import json
from data.action4_extension_selection import (
    SCHEMA,POLICY,_path,_read,_resolve,validate_global_selection,comparison_key)

def freeze_global_selection(incumbent50_ref,predecessor10_ref,output):
    """No partial-run fallback: the10epoch seal must exist and verify in full."""
    target=_path(output)
    if target.exists():raise FileExistsError("Preserve existing global-selection receipt")
    # Reuse production verifier, including independent immutable strict reload,
    # full eval coverage/rank and frozen-backbone comparison.
    from eval.action4_roi_pair_selection import _verify_seal
    refs=(incumbent50_ref,predecessor10_ref)
    expected_seals=[_read(ref) for ref in refs]  # Missing predecessor fails before expensive verification.
    for ref,expected in zip(refs,expected_seals):
        verified=_verify_seal(ref["path"])
        if verified["seal"]!=expected or verified.get("seal_sha256")!=ref["sha256"]:
            raise ValueError("Full sealed-role verification/source identity differs")
    a,b,winner=_resolve(incumbent50_ref,predecessor10_ref)
    result={"schema":SCHEMA,"status":"COMPLETED_PREDECESSOR_GLOBAL_BEST_SELECTED",
        "incumbent_seal":incumbent50_ref,"predecessor_seal":predecessor10_ref,
        "incumbent_best":a["best_ref"],"predecessor_best":b["best_ref"],
        "winner":winner,"winner_seal":predecessor10_ref if winner=="predecessor" else incumbent50_ref,
        "policy":POLICY,"tie_policy":"retain_incumbent","local_optimizer_step_used_for_cross_run_rank":False,
        "verification":"existing_sealed_role_full_immutable_verifier",
        "formal_training_units":a["config"]["expected_units"],"defaults_modified":False}
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open("x") as stream:json.dump(result,stream,indent=2,allow_nan=False)
    target.chmod(0o444)
    return result
