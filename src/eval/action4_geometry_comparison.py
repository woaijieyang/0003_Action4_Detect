"""Complete200 RGB-only paired comparison with a fixed compatible feature parent."""
from pathlib import Path
import json,torch,numpy as np
from data.action4_paths import PROJECT_ROOT as R
from data.action4_component_data import ComponentData
from eval.action4_geometry_pipeline import GeometryPipeline
from eval.action4_metrics import score_dataset
from model.training.action4_runtime import atomic_json


def run(output,solver="reference"):
    if solver=="analytic":
        from eval.action4_geometry_pipeline_analytic import GeometryPipeline as Pipeline
    elif solver=="reference":Pipeline=GeometryPipeline
    else:raise ValueError("Unknown solver")
    out=R/output;torch.set_num_threads(2)
    pipeline=Pipeline(out/'B0.pt',out/'A0.pt',out/'residual.pt','cuda:0')
    data=ComponentData('outputs/action4_independent_trials_v001/DATA_VALIDATED.json','eval','B0');predictions={key:[] for key in ('raw','rigid','visibility_residual')}
    for index in range(len(data)):
        item=data[index];record=item['record'];variants=pipeline.predict(item['image'],np.asarray(record['camera']['K']),np.asarray(record['corners_3d_mm'])/1000)
        for name,instances in variants.items():predictions[name].append({'sample_id':record['sample_id'],'instances':instances})
        atomic_json(out/'STATUS.json',{'completed_images':index+1,'total':len(data)})
    scores={}
    for name,rows in predictions.items():
        with (out/(name+'_predictions.jsonl')).open('x') as f:
            for row in rows:f.write(json.dumps(row)+'\n')
        scores[name]=score_dataset(rows,data.records)
    atomic_json(out/'REPORT.json',{'scope':'full200 synthetic development RGB-only paired candidate comparison','reference_boxes_enter_inference':False,'GT_visibility_or_pose_enter_inference':False,'known_synthetic_K_and_fixed_CAD_used':True,'fit_missing_fallback':pipeline.fit_missing,'nonfinite_fallback':pipeline.nonfinite_fallback,'scores':scores,'production_accepted':False})
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--solver',choices=['reference','analytic'],default='reference');a=p.parse_args();run(a.output,a.solver)
