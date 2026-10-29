from eval.action4_roi_video_job import validate_reuse_identity

def test_reuse_requires_exact_inputs_and_inference_code():
    cfg={"input_lock":{"sha256":"a"},"corner_trial_binding":{"sha256":"b"},"ratios":[1.5]}
    source={"files":{"src/model/head.py":"x","src/eval/action4_roi_video_job.py":"new"}}
    old={"files":{"src/model/head.py":"x","src/eval/action4_roi_video_job.py":"old"}}
    validate_reuse_identity(cfg,dict(cfg),source,old)
    for bc,bs in [({**cfg,"ratios":[2.25]},old),(cfg,{"files":{"src/model/head.py":"wrong"}}),({**cfg,"corner_trial_binding":{"sha256":"wrong"}},old)]:
        try:validate_reuse_identity(cfg,bc,source,bs)
        except ValueError:pass
        else:raise AssertionError("Incompatible reuse accepted")
