"""Prepare new-method training from a selected best checkpoint, not last recovery.

Selection is frozen into a new artifact. Optimizer/scheduler/RNG are deliberately
not transferred across methods. Existing training jobs are never modified.
"""
from pathlib import Path
import copy
import torch
from data.action4_paths import PROJECT_ROOT,contained_path
from model.action4_component_experiments import Candidate
from model.training.action4_runtime import save_checkpoint_atomic

NEW_MODULE_PREFIXES=("quality_head.","token_projection.","corner_identity",
                     "visibility_head.","corner_relation.","correction_head.")

def prepare_best_initialization(best_path, target_config, output_path):
    source=contained_path(PROJECT_ROOT,Path(best_path),exists=True)
    output=contained_path(PROJECT_ROOT,Path(output_path),exists=False)
    if output.exists():
        raise FileExistsError(output)
    state=torch.load(source,map_location="cpu",weights_only=True)
    if state.get("schema_version")!="action4.component_eval.v1":
        raise ValueError("Use selected evaluation weights, not last.pt recovery state")
    if state["role"]!=target_config["role"]:
        raise ValueError("Cross-role initialization needs a separate explicit migration")
    if state["config"]["network"]!=target_config["network"]:
        raise ValueError("Architecture changed; explicit weight migration required")
    target=Candidate(target_config["role"],target_config["network"],target_config["variant"])
    missing,unexpected=target.load_state_dict(state["model"],strict=False)
    if unexpected or any(not name.startswith(NEW_MODULE_PREFIXES) for name in missing):
        raise ValueError({"unexpected":unexpected,"unapproved_missing":missing})
    artifact={"schema_version":"action4.component_initialization.v1",
              "role":state["role"],"model":state["model"],
              "network":state["config"]["network"],
              "source_variant":state["variant"],"source_step":state["step"],
              "source_config_id":state["config_id"],"source_best_path":str(source.relative_to(PROJECT_ROOT)),
              "target_variant":target_config["variant"],"new_parameters":missing,
              "resumable":False,"purpose":"best-weight initialization; fresh optimizer/schedule/RNG"}
    save_checkpoint_atomic(output,artifact)
    return {key:value for key,value in artifact.items() if key!="model"}
