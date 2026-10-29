"""Independent actual single-image corner eval after strict integration bundle load."""
import json,torch,numpy as np
from PIL import Image
from pathlib import Path
from data.action4_paths import PROJECT_ROOT as R,contained_path
from eval.action4_staged_pipeline import StagedPipeline

def run():
    out=R/"outputs/action4_stage4_real_input_updates_v002"
    cfg=json.loads((out/"CONFIG.json").read_text());torch.set_num_threads(3);torch.set_num_interop_threads(1)
    pipe=StagedPipeline.from_corner_bundles(out/"ENGINEERING_CORNER_STEP2.pt",cfg["evaluation_counterpart_bundle"],cfg["sam_eval_cache"],device="cpu")
    with (R/"data/phrase2/eval/annotations.jsonl").open() as stream:record=json.loads(next(stream))
    with Image.open(contained_path(R,record["image"],exists=True)) as image:rgb=np.asarray(image.convert("RGB"))
    K=np.asarray(record["camera"]["K"]);cad=np.asarray(record["corners_3d_mm"])/1000
    counts={"B0":0,"A0_backbone":0}
    pipe.b0.register_forward_hook(lambda *args:counts.__setitem__("B0",counts["B0"]+1))
    pipe.base.backbone.register_forward_hook(lambda *args:counts.__setitem__("A0_backbone",counts["A0_backbone"]+1))
    context=pipe.corner_context(record["sample_id"],rgb)
    result=pipe.predict_corner(rgb,K,cad,sam_context=context,diagnostic_candidate=True)
    assert counts=={"B0":0,"A0_backbone":1} and pipe.roi_expansion==1.5
    assert result["quality_ready"] is False and result["deployment_admitted"] is False
    assert len(result["raw_instances"])==len(result["corner_observations"])==2
    (out/"ACTUAL_EVAL_IMAGE_v002.json").write_text(json.dumps(result,allow_nan=False))
    report=dict(status="REAL_SINGLE_IMAGE_CORNER_EVAL_PASS_NOT_FULL_EVAL_OR_ACCURACY",sample_id=record["sample_id"],
        model_calls=counts,ROI=1.5,raw_instances=len(result["raw_instances"]),observations=len(result["corner_observations"]),
        quality_ready=False,deployment_admitted=False,GT_pose_not_input=True,frozen_B0_source_used=True)
    (out/"ACTUAL_EVAL_CHECK_v002.json").write_text(json.dumps(report,indent=2));print(json.dumps(report))
