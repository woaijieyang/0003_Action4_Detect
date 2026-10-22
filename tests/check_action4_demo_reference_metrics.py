import json
from eval.action4_demo_reference_metrics import compare
boxes=[[10,10,30,30],[60,60,80,80]]
ref={"frames":3,"certified_GT":False,"training_labels_allowed":False,"rows":[{"frame_id":i,"reference_candidate":i!=1,"components":[{"bbox_demo_px":b} for b in boxes]} for i in range(3)]}
pred=[{"frame_id":i,"pts":400*i,"prediction":{"instances":[{"bbox_full_xyxy":b,"score":.9} for b in (boxes[::-1] if i!=2 else boxes[:1])]}} for i in range(3)]
result=compare(ref,pred,source_wh=(950,720))
assert result["summary"]["0.5"]=={"both":1,"frames":2,"rate":.5}
assert result["unscorable_reference_frames"]==1
pred[2]["frame_id"]=4
try:compare(ref,pred,source_wh=(950,720))
except ValueError:pass
else:raise AssertionError("Discontinuous frame accepted")
print(json.dumps({"passed":True,"order_independent_match":True,"reference_exclusions_explicit":True,"frame_discontinuity_rejected":True}))
