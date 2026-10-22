"""Paired solver timings on existing real predictions, same ROI/K/starts/budget."""
import json,time
import numpy as np
from data.action4_paths import PROJECT_ROOT as R
from data import DataAdapter
from data.action4_dataset import virtual_roi
from model.action4_rigid_projection import fit_rigid_cuboid as reference
from model.action4_rigid_projection_analytic import fit_rigid_cuboid as candidate
from model.training.action4_runtime import atomic_json


def run():
    out=R/'outputs/action4_geometry_speed_v001'
    real=json.loads((R/'outputs/action4_current_geometry_real_review_v001/REPORT.json').read_text());K=np.array(real['K'])
    record=json.loads((R/'data/eval/annotations.jsonl').read_text().splitlines()[0]);xyz=np.array(record['corners_3d_mm'])/1000;rows=[]
    for frame in real['frames']:
        for index,pred in enumerate(frame['variants']['raw']):
            a=DataAdapter.crop_affine(virtual_roi(pred['bbox_full_xyxy'],2.25),(448,448));uv=DataAdapter.transform_points(pred['corners_px'],a);kr=a@K
            times={};answers={}
            for name,fn in [('reference',reference),('analytic',candidate)]:
                start=time.perf_counter();answers[name]=fn(uv,xyz,kr,pred['corner_peak_scores']);times[name]=time.perf_counter()-start
            old,new=answers['reference'],answers['analytic'];delta=None
            if old['fitted_corners_px'] is not None and new['fitted_corners_px'] is not None:delta=float(np.max(np.abs(np.array(old['fitted_corners_px'])-new['fitted_corners_px'])))
            rows.append({'frame_id':frame['frame_id'],'instance_index':index,'seconds':times,'max_ROI_corner_delta_px':delta,'reference_reason':old.get('reason'),'analytic_reason':new.get('reason'),'reference_rms':old.get('weighted_rms_px'),'analytic_rms':new.get('weighted_rms_px')})
            atomic_json(out/'STATUS.json',{'completed':len(rows),'total':sum(len(f['variants']['raw']) for f in real['frames'])})
    total={n:sum(x['seconds'][n] for x in rows) for n in ['reference','analytic']}
    atomic_json(out/'REPORT.json',{'scope':'12real detected proposals solver comparison, not accuracy','times':total,'speed_ratio_reference_over_analytic':total['reference']/total['analytic'],'max_corner_delta':max(x['max_ROI_corner_delta_px'] or 0 for x in rows),'rows':rows,'production_switched':False})
if __name__=='__main__':run()
