"""Constructed fixture proves comparator retains coordinate regressions."""
from pathlib import Path
import json
from eval.action4_geometry_equivalence import compare

def main():
    root=Path.cwd();out=root/'outputs/action4_geometry_speed_v001/equivalence_fixture';a=out/'reference';b=out/'candidate';a.mkdir(parents=True,exist_ok=False);b.mkdir()
    variants=['raw','rigid','visibility_residual'];rows=[{'sample_id':f'constructed_{i}','instances':[{'instance_id':0,'object_id':1,'corner_ids':list(range(8)),'bbox_full_xyxy':[0,0,10,10],'score':.8,'corners_px':[[j,j] for j in range(8)]}]} for i in range(200)]
    metrics={'tp':200,'fp':0,'fn':0,'valid_gt':200,'predictions':200,'f1':1,'precision':1,'recall':1};report={'scores':{v:{'thresholds':{'0.05':metrics}} for v in variants}}
    for path in [a,b]:
        (path/'REPORT.json').write_text(json.dumps(report))
        for variant in variants:(path/(variant+'_predictions.jsonl')).write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
    compare(a,b);assert json.loads((b/'EQUIVALENCE.json').read_text())['status']=='PAIRED_EQUIVALENCE_PASS'
    rows[0]['instances'][0]['corners_px'][0][0]=.02
    (b/'visibility_residual_predictions.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
    compare(a,b);assert json.loads((b/'EQUIVALENCE.json').read_text())['status']=='PAIRED_EQUIVALENCE_DIFFERENCES'
    result={'status':'PASS','scope':'constructed comparator test, not model accuracy','identical_outputs_pass':True,'same_scores_but_changed_corner_rejected':True}
    (out/'TEST.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
