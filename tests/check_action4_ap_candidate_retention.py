"""Regression: operating score filtering must not discard AP candidates."""
import json,tempfile
from pathlib import Path
import eval.action4_staged_eval as module
from data.action4_paths import PROJECT_ROOT as R

def test_protocol():
    root=Path(tempfile.mkdtemp(prefix="ap_protocol_",dir=R/"runtime"))
    selection=root/"selection.json";selection.write_text("{}")
    corners=[[10,10],[30,10],[30,30],[10,30],[12,12],[28,12],[28,28],[12,28]]
    records=[{"sample_id":sid,"instances":[{"instance_id":1,"bbox_full_xyxy":[10,10,30,30],"corners_px":corners,"occlusion_inframe":0}],
        "camera":{"K":[[1,0,0],[0,1,0],[0,0,1]]},"width":100,"scene_family":"fixture"} for sid in ["low_positive","operating_positive"]]
    class FakePipeline:
        identity={"scope":"synthetic protocol fixture only"}
        def predict(self,image,K,canonical,*,return_box_candidates=False):
            assert return_box_candidates
            good={"bbox_full_xyxy":[10,10,30,30],"score":.2 if image==0 else .8}
            if image==0:return {"instances":[],"box_candidates":[good]}
            low={"bbox_full_xyxy":[60,60,80,80],"score":.1}
            return {"instances":[dict(good,pose_valid=True,corners_px=corners)],"box_candidates":[low,good]}
    original_data,original_factory=module.ComponentData,module.load_pipeline
    try:
        module.ComponentData=lambda *args:[{"image":i,"record":r} for i,r in enumerate(records)]
        module.load_pipeline=lambda cfg:FakePipeline()
        output=root/"result"
        module.run(str(selection.relative_to(R)),str(output.relative_to(R)))
        report=json.loads((output/"REPORT.json").read_text())
        assert report["bbox_only"]["mAP50"]==1.0
        assert report["bbox_only"]["overall"]["tp"]==1 and report["bbox_only"]["overall"]["fn"]==1
        assert report["corner_all8_success"]==1 and report["corner_GT"]==2
        assert report["bbox_only"]["AP_candidate_policy"]["threshold_applied_before_AP"] is False
        print(json.dumps({"passed":True,"AP_keeps_low_score_true_positive":True,"operating_index_mapping_correct":True,"scope":"synthetic protocol regression, not accuracy"}))
    finally:
        module.ComponentData, module.load_pipeline=original_data,original_factory
if __name__=="__main__":test_protocol()
