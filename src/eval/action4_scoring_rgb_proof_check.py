"""Full650 offline equivalence of prior-pixel-proof scoring; no model calls."""
import json,os,sys,time,hashlib,importlib.util
from pathlib import Path
def main():
    R=Path("/data/zy/vln/a-test");assert Path.cwd().resolve()==R
    out=R/"outputs/action4_scoring_rgb_proof_v001";S=out/"source"
    os.environ["ACTION4_PROJECT_ROOT"]=str(R);sys.path[:0]=[str(S/"runtime/surface_deps"),str(S/"src")]
    import torch,numpy as np
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    from data.action4_sam_native_export import _v2_checked,_v2_read,_v2_rgb
    from eval.action4_corner_quality_bridge import CornerQualityBridge
    from eval.action4_corner_scoring_inputs import build_scoring_inputs,_deps
    from unittest.mock import patch
    artifact=R/"outputs/action4_corner_capacity32_v001/worker_00/PREDICTION_ARTIFACT.json"
    def ref(path):
        with path.open("rb") as f:sha=hashlib.file_digest(f,"sha256").hexdigest()
        return {"path":str(path.relative_to(R)),"sha256":sha}
    binding=ref(artifact);payload=_v2_checked(binding);identity=payload["effective_identity"]
    ann=ref(R/"data/phrase2/eval/annotations.jsonl")
    rows=_v2_read(ann,lambda stream:[json.loads(line) for line in stream if line.strip()])
    _,saved=_deps(identity,False)
    bridge=CornerQualityBridge(identity["refined_quality_bridge"],b0_sha256=identity["models"]["B0"]["sha256"],
        b0_config_id=identity["B0_config_id"],b0_step=identity["B0_step"],
        base_predictions=identity["input_contract"]["frozen_predictions"],quality_gate=identity["full_box_quality_receipt"])
    start=time.monotonic()
    for row in rows:
        rgb=_v2_rgb(row["image"],row["rgb_sha256"])
        bridge.consume(sample_id=row["sample_id"],rgb=rgb,K=np.asarray(row["camera"]["K"]),
                       base_instances=saved[row["sample_id"]],b0_sha256=identity["models"]["B0"]["sha256"])
    prime=time.monotonic()-start
    producer=_v2_checked(payload["source_manifest"])
    legacy_path=R/producer["code_root"]/"src/eval/action4_corner_scoring_inputs.py"
    spec=importlib.util.spec_from_file_location("eval._legacy_scoring_rgb_replay",legacy_path)
    legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
    begin=time.monotonic()
    baseline=legacy.build_scoring_inputs(binding,ann,payload["source_manifest"],out/"legacy_scoring",diagnostic=False)
    baseline_seconds=time.monotonic()-begin
    calls=[]
    def counted(*args,**kwargs):
        calls.append(str(args[0]));return _v2_rgb(*args,**kwargs)
    begin=time.monotonic()
    with patch("data.action4_sam_native_export._v2_rgb",side_effect=counted):
        candidate=build_scoring_inputs(binding,ann,payload["source_manifest"],out/"proof_scoring",diagnostic=False)
    fast_seconds=time.monotonic()-begin
    if calls:raise ValueError("Expected prior RGB proofs for every image; decode fallback occurred")
    if candidate!=baseline:raise ValueError("Full scoring/domain/quality targets changed")
    result={"status":"FULL650_SCORING_PARITY_PASS","images":650,"GT":1300,"model_forward_calls":0,
        "complete_result_exact":True,"RGB_redecode_calls_in_candidate_scoring":len(calls),
        "prime_sidecar_consumption_seconds":prime,"legacy_scoring_seconds":baseline_seconds,
        "proof_scoring_seconds":fast_seconds,"same_process_dependency_caches_warm":True,
        "priming_is_not_model_inference":True,"not_full_service_capacity_admission":True,
        "input":binding,"annotations":ann,"producer_source":payload["source_manifest"]}
    (out/"COMPLETE.json").write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=="__main__":main()
